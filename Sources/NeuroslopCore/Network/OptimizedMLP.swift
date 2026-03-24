@preconcurrency import Accelerate
import Foundation

/// Pre-allocated, fused MLP for maximum throughput on small models.
///
/// Optimizations vs generic MLP:
/// 1. All intermediate buffers pre-allocated at init (zero malloc during train)
/// 2. Fused matmul+bias+activation (one pass over output memory)
/// 3. In-place weight updates (vDSP_vsma: w -= lr * grad, no temp arrays)
/// 4. Direct Accelerate calls, no protocol dispatch
/// 5. ReLU backward fused with elementwise multiply (no separate derivative matrix)
public final class OptimizedMLP {
    // Network structure
    private let layerSizes: [Int]
    private let activations: [ActivationKind]
    private let numLayers: Int

    // Weights and biases (mutable, updated during training)
    private var weights: [UnsafeMutableBufferPointer<Float>]  // [inputSize x outputSize] per layer
    private var biases: [UnsafeMutableBufferPointer<Float>]   // [outputSize] per layer

    // Pre-allocated forward buffers
    private var preActivations: [UnsafeMutableBufferPointer<Float>]  // z = input*W + b
    private var postActivations: [UnsafeMutableBufferPointer<Float>] // a = activation(z)

    // Pre-allocated backward buffers
    private var dz: [UnsafeMutableBufferPointer<Float>]              // dL/dz per layer
    private var weightGrads: [UnsafeMutableBufferPointer<Float>]     // dL/dW per layer
    private var biasGrads: [UnsafeMutableBufferPointer<Float>]       // dL/db per layer
    private var inputGrads: [UnsafeMutableBufferPointer<Float>]      // dL/d(input) per layer

    // Raw memory storage (to free later)
    private var allAllocations: [UnsafeMutableRawPointer]

    private let batchSize: Int

    public init(layerSizes: [Int], activations: [ActivationKind], batchSize: Int) {
        precondition(layerSizes.count >= 2)
        precondition(activations.count == layerSizes.count - 1)

        self.layerSizes = layerSizes
        self.activations = activations
        self.numLayers = activations.count
        self.batchSize = batchSize

        var allocs: [UnsafeMutableRawPointer] = []

        func alloc(_ count: Int) -> UnsafeMutableBufferPointer<Float> {
            let ptr = UnsafeMutablePointer<Float>.allocate(capacity: count)
            ptr.initialize(repeating: 0, count: count)
            allocs.append(UnsafeMutableRawPointer(ptr))
            return UnsafeMutableBufferPointer(start: ptr, count: count)
        }

        // Allocate weights + biases with He/Xavier init
        var w: [UnsafeMutableBufferPointer<Float>] = []
        var b: [UnsafeMutableBufferPointer<Float>] = []
        for i in 0..<numLayers {
            let fanIn = layerSizes[i]
            let fanOut = layerSizes[i + 1]
            let wBuf = alloc(fanIn * fanOut)
            let bBuf = alloc(fanOut)

            // Initialize weights
            let stddev: Float
            switch activations[i] {
            case .relu:
                stddev = sqrtf(2.0 / Float(fanIn))
            default:
                stddev = sqrtf(2.0 / Float(fanIn + fanOut))
            }
            for j in 0..<(fanIn * fanOut) {
                let u1 = Float.random(in: 0.0001...1.0)
                let u2 = Float.random(in: 0.0...1.0)
                wBuf[j] = sqrtf(-2.0 * logf(u1)) * cosf(2.0 * .pi * u2) * stddev
            }
            w.append(wBuf)
            b.append(bBuf)
        }
        self.weights = w
        self.biases = b

        // Allocate forward buffers
        var preAct: [UnsafeMutableBufferPointer<Float>] = []
        var postAct: [UnsafeMutableBufferPointer<Float>] = []
        for i in 0..<numLayers {
            preAct.append(alloc(batchSize * layerSizes[i + 1]))
            postAct.append(alloc(batchSize * layerSizes[i + 1]))
        }
        self.preActivations = preAct
        self.postActivations = postAct

        // Allocate backward buffers
        var dzBufs: [UnsafeMutableBufferPointer<Float>] = []
        var wgBufs: [UnsafeMutableBufferPointer<Float>] = []
        var bgBufs: [UnsafeMutableBufferPointer<Float>] = []
        var igBufs: [UnsafeMutableBufferPointer<Float>] = []
        for i in 0..<numLayers {
            dzBufs.append(alloc(batchSize * layerSizes[i + 1]))
            wgBufs.append(alloc(layerSizes[i] * layerSizes[i + 1]))
            bgBufs.append(alloc(layerSizes[i + 1]))
            igBufs.append(alloc(batchSize * layerSizes[i]))
        }
        self.dz = dzBufs
        self.weightGrads = wgBufs
        self.biasGrads = bgBufs
        self.inputGrads = igBufs

        self.allAllocations = allocs
    }

    deinit {
        for ptr in allAllocations {
            ptr.deallocate()
        }
    }

    // MARK: - Forward

    /// Forward pass. Input: batchSize x layerSizes[0]. Writes into pre-allocated buffers.
    /// Returns pointer to final output (softmax/activation output of last layer).
    public func forward(_ input: UnsafePointer<Float>, actualBatchSize: Int) -> UnsafeMutableBufferPointer<Float> {
        var currentInput = input
        let bs = actualBatchSize

        for i in 0..<numLayers {
            let inSize = layerSizes[i]
            let outSize = layerSizes[i + 1]
            let z = preActivations[i].baseAddress!
            let a = postActivations[i].baseAddress!
            let w = weights[i].baseAddress!
            let b = biases[i].baseAddress!

            // z = input * W  (bs x inSize) * (inSize x outSize) = (bs x outSize)
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Int32(bs), Int32(outSize), Int32(inSize),
                1.0,
                currentInput, Int32(inSize),
                w, Int32(outSize),
                0.0,
                z, Int32(outSize)
            )

            // Fused: z += bias, then a = activation(z)
            switch activations[i] {
            case .relu:
                for r in 0..<bs {
                    let offset = r * outSize
                    for j in 0..<outSize {
                        let val = z[offset + j] + b[j]
                        z[offset + j] = val
                        a[offset + j] = val > 0 ? val : 0
                    }
                }
            case .sigmoid:
                for r in 0..<bs {
                    let offset = r * outSize
                    for j in 0..<outSize {
                        let val = z[offset + j] + b[j]
                        z[offset + j] = val
                        a[offset + j] = 1.0 / (1.0 + expf(-val))
                    }
                }
            case .tanh:
                for r in 0..<bs {
                    let offset = r * outSize
                    for j in 0..<outSize {
                        let val = z[offset + j] + b[j]
                        z[offset + j] = val
                        a[offset + j] = tanhf(val)
                    }
                }
            case .softmax:
                for r in 0..<bs {
                    let offset = r * outSize
                    // Add bias
                    for j in 0..<outSize {
                        z[offset + j] += b[j]
                    }
                    // Softmax with numerical stability
                    var maxVal: Float = z[offset]
                    for j in 1..<outSize {
                        if z[offset + j] > maxVal { maxVal = z[offset + j] }
                    }
                    var sum: Float = 0
                    for j in 0..<outSize {
                        let e = expf(z[offset + j] - maxVal)
                        a[offset + j] = e
                        sum += e
                    }
                    let invSum = 1.0 / sum
                    for j in 0..<outSize {
                        a[offset + j] *= invSum
                    }
                }
            }

            currentInput = UnsafePointer(a)
        }

        return postActivations[numLayers - 1]
    }

    // MARK: - Backward

    private func backwardClean(_ lossGrad: UnsafePointer<Float>, input: UnsafePointer<Float>, actualBatchSize bs: Int) {
        // currentGrad starts as dL/d(output) and gets propagated backward
        var currentGrad = lossGrad

        for i in stride(from: numLayers - 1, through: 0, by: -1) {
            let inSize = layerSizes[i]
            let outSize = layerSizes[i + 1]
            let dzBuf = dz[i].baseAddress!
            let wgBuf = weightGrads[i].baseAddress!
            let bgBuf = biasGrads[i].baseAddress!
            let igBuf = inputGrads[i].baseAddress!
            let w = weights[i].baseAddress!
            let a = postActivations[i].baseAddress!
            let count = bs * outSize

            // Step 1: dz = currentGrad * activation'(a)
            if activations[i] == .softmax {
                // Combined softmax+CE: dz = currentGrad (passthrough)
                memcpy(dzBuf, currentGrad, count * MemoryLayout<Float>.stride)
            } else if activations[i] == .relu {
                // Fused: dz = currentGrad * (a > 0 ? 1 : 0)
                for j in 0..<count {
                    dzBuf[j] = a[j] > 0 ? currentGrad[j] : 0
                }
            } else if activations[i] == .sigmoid {
                // dz = currentGrad * a * (1 - a)
                for j in 0..<count {
                    dzBuf[j] = currentGrad[j] * a[j] * (1.0 - a[j])
                }
            } else { // tanh
                // dz = currentGrad * (1 - a^2)
                for j in 0..<count {
                    dzBuf[j] = currentGrad[j] * (1.0 - a[j] * a[j])
                }
            }

            // Step 2: dW = layerInput^T * dz
            let layerInput: UnsafePointer<Float>
            if i == 0 {
                layerInput = input
            } else {
                layerInput = UnsafePointer(postActivations[i - 1].baseAddress!)
            }

            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                Int32(inSize), Int32(outSize), Int32(bs),
                1.0,
                layerInput, Int32(inSize),
                dzBuf, Int32(outSize),
                0.0,
                wgBuf, Int32(outSize)
            )

            // Step 3: db = sum of dz across rows
            // Initialize to zero
            memset(bgBuf, 0, outSize * MemoryLayout<Float>.stride)
            for r in 0..<bs {
                let offset = r * outSize
                for j in 0..<outSize {
                    bgBuf[j] += dzBuf[offset + j]
                }
            }

            // Step 4: dInput = dz * W^T (only needed if not first layer)
            if i > 0 {
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(bs), Int32(inSize), Int32(outSize),
                    1.0,
                    dzBuf, Int32(outSize),
                    w, Int32(outSize),
                    0.0,
                    igBuf, Int32(inSize)
                )
                currentGrad = UnsafePointer(igBuf)
            }
        }
    }

    // MARK: - Weight Update

    /// In-place SGD: w -= lr * grad, b -= lr * grad. No temporary allocations.
    public func updateWeights(learningRate lr: Float) {
        var negLr = -lr
        for i in 0..<numLayers {
            let wCount = layerSizes[i] * layerSizes[i + 1]
            let bCount = layerSizes[i + 1]
            // w += (-lr) * wGrad  →  vDSP_vsma: C = A * scalar + B
            vDSP_vsma(weightGrads[i].baseAddress!, 1,
                      &negLr,
                      weights[i].baseAddress!, 1,
                      weights[i].baseAddress!, 1,
                      vDSP_Length(wCount))
            vDSP_vsma(biasGrads[i].baseAddress!, 1,
                      &negLr,
                      biases[i].baseAddress!, 1,
                      biases[i].baseAddress!, 1,
                      vDSP_Length(bCount))
        }
    }

    // MARK: - High-level API

    /// Full train step: forward + CE loss gradient + backward + update.
    /// Returns loss value.
    public func trainStep(
        batchX: UnsafePointer<Float>,
        batchY: UnsafePointer<Float>,  // one-hot
        actualBatchSize: Int,
        learningRate: Float,
        lossGradBuffer: UnsafeMutablePointer<Float>  // pre-allocated, batchSize x outputSize
    ) -> Float {
        let bs = actualBatchSize
        let outSize = layerSizes[numLayers]

        // Forward
        let output = forward(batchX, actualBatchSize: bs)

        // CE loss + gradient: dL/dz = (softmax - target) / batchSize
        let batchSizeF = Float(bs)
        var loss: Float = 0
        for i in 0..<(bs * outSize) {
            let p = max(output[i], 1e-7)
            loss -= batchY[i] * logf(p)
            lossGradBuffer[i] = (output[i] - batchY[i]) / batchSizeF
        }
        loss /= batchSizeF

        // Backward
        backwardClean(UnsafePointer(lossGradBuffer), input: batchX, actualBatchSize: bs)

        // Update
        updateWeights(learningRate: learningRate)

        return loss
    }

    // MARK: - Inference

    /// Forward pass returning a Matrix (for evaluation/accuracy computation).
    /// Processes in chunks of batchSize to stay within pre-allocated buffers.
    public func predict(_ input: Matrix) -> Matrix {
        let outSize = layerSizes[numLayers]
        var resultData = [Float](repeating: 0, count: input.rows * outSize)
        var offset = 0
        input.data.withUnsafeBufferPointer { inBuf in
            while offset < input.rows {
                let bs = min(batchSize, input.rows - offset)
                let output = forward(inBuf.baseAddress!.advanced(by: offset * input.cols), actualBatchSize: bs)
                resultData.withUnsafeMutableBufferPointer { dst in
                    memcpy(dst.baseAddress!.advanced(by: offset * outSize),
                           output.baseAddress!, bs * outSize * MemoryLayout<Float>.stride)
                }
                offset += bs
            }
        }
        return Matrix(rows: input.rows, cols: outSize, data: resultData)
    }

    /// Export weights as Matrix arrays (for verification against generic MLP).
    public func exportWeights() -> [(weights: Matrix, biases: Matrix)] {
        var result: [(Matrix, Matrix)] = []
        for i in 0..<numLayers {
            let inSize = layerSizes[i]
            let outSize = layerSizes[i + 1]
            let w = Matrix(rows: inSize, cols: outSize,
                           data: Array(weights[i]))
            let b = Matrix(rows: 1, cols: outSize,
                           data: Array(biases[i]))
            result.append((w, b))
        }
        return result
    }
}
