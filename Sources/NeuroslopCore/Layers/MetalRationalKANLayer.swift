import Metal

/// Rational KAN layer accelerated with Metal GPU compute shaders.
/// Forward: parallel P(x)/Q(x) per edge (one thread per edge) + reduce sum.
/// Backward: parallel gradient computation per edge + reduce.
///
/// Same math as RationalKANLayer but runs on GPU.
public struct MetalRationalKANLayer: @unchecked Sendable {
    public let inputSize: Int
    public let outputSize: Int
    public let numDegree: Int  // p
    public let denDegree: Int  // q
    public let batchSize: Int

    let ctx: MetalContext

    // Parameters as MTLBuffers (live on GPU, shared memory = zero-copy)
    var numCoeffsBuf: MTLBuffer  // [outSize, inSize * (p+1)]
    var denCoeffsBuf: MTLBuffer  // [outSize, inSize * q]

    // Pre-allocated work buffers
    private let phiBuf: MTLBuffer       // [batchSize, outSize, inSize]
    private let pValsBuf: MTLBuffer     // [batchSize, outSize, inSize]
    private let qValsBuf: MTLBuffer     // [batchSize, outSize, inSize]
    private let outputBuf: MTLBuffer    // [batchSize, outSize]

    // Backward buffers
    private let numGradBuf: MTLBuffer   // [batchSize, outSize, inSize * (p+1)]
    private let denGradBuf: MTLBuffer   // [batchSize, outSize, inSize * q]
    private let inputGradEdgeBuf: MTLBuffer  // [batchSize, outSize, inSize]

    // Cached input for backward
    private var inputBuf: MTLBuffer?

    public var totalParameters: Int {
        outputSize * inputSize * (numDegree + 1) + outputSize * inputSize * denDegree
    }

    public init(inputSize: Int, outputSize: Int, numDegree: Int = 3, denDegree: Int = 2, batchSize: Int = 64) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.numDegree = numDegree
        self.denDegree = denDegree
        self.batchSize = batchSize
        self.ctx = MetalContext()

        let edgeCount = batchSize * outputSize * inputSize
        let numCoeffCount = outputSize * inputSize * (numDegree + 1)
        let denCoeffCount = outputSize * inputSize * denDegree

        // Init parameters
        var numData = [Float](repeating: 0, count: numCoeffCount)
        for i in 0..<numData.count {
            let k = i % (numDegree + 1)
            numData[i] = k == 0 ? Float.random(in: -0.1...0.1) : Float.random(in: -0.01...0.01)
        }
        var denData = (0..<denCoeffCount).map { _ in Float.random(in: -0.01...0.01) as Float }

        self.numCoeffsBuf = ctx.makeBuffer(numData)
        self.denCoeffsBuf = ctx.makeBuffer(denData)

        // Work buffers
        self.phiBuf = ctx.makeBuffer(count: edgeCount)
        self.pValsBuf = ctx.makeBuffer(count: edgeCount)
        self.qValsBuf = ctx.makeBuffer(count: edgeCount)
        self.outputBuf = ctx.makeBuffer(count: batchSize * outputSize)

        // Backward buffers
        self.numGradBuf = ctx.makeBuffer(count: batchSize * numCoeffCount)
        self.denGradBuf = ctx.makeBuffer(count: batchSize * denCoeffCount)
        self.inputGradEdgeBuf = ctx.makeBuffer(count: edgeCount)
    }

    private func encode(
        _ cmdBuf: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        buffers: [(MTLBuffer, Int)],
        uints: [(UInt32, Int)],
        threads: Int
    ) {
        guard let encoder = cmdBuf.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        for (buf, idx) in buffers {
            encoder.setBuffer(buf, offset: 0, index: idx)
        }
        for (val, idx) in uints {
            var v = val
            encoder.setBytes(&v, length: 4, index: idx)
        }
        let tw = min(pipeline.maxTotalThreadsPerThreadgroup, threads)
        encoder.dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tw, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Forward for a single chunk that fits in pre-allocated buffers.
    private mutating func forwardChunk(_ input: Matrix) -> Matrix {
        let bs = input.rows
        precondition(bs <= batchSize)
        let edgeCount = bs * outputSize * inputSize

        let inBuf = ctx.makeBuffer(input.data)
        self.inputBuf = inBuf

        // Single command buffer: forward + reduce
        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer() else {
            return Matrix(rows: bs, cols: outputSize)
        }

        // Kernel 1: compute P(x)/Q(x) per edge
        encode(cmdBuf, pipeline: ctx.rkanForward,
               buffers: [(inBuf, 0), (numCoeffsBuf, 1), (denCoeffsBuf, 2),
                         (phiBuf, 3), (pValsBuf, 4), (qValsBuf, 5)],
               uints: [(UInt32(bs), 6), (UInt32(inputSize), 7), (UInt32(outputSize), 8),
                       (UInt32(numDegree), 9), (UInt32(denDegree), 10)],
               threads: edgeCount)

        // Kernel 2: reduce across input dimension
        encode(cmdBuf, pipeline: ctx.rkanReduce,
               buffers: [(phiBuf, 0), (outputBuf, 1)],
               uints: [(UInt32(inputSize), 2), (UInt32(outputSize), 3)],
               threads: bs * outputSize)

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        return Matrix(rows: bs, cols: outputSize,
                      data: ctx.readBuffer(outputBuf, count: bs * outputSize))
    }
}

extension MetalRationalKANLayer: Layer {
    public mutating func forward(_ input: Matrix, backend: ComputeBackend) -> Matrix {
        // Chunk large inputs to fit pre-allocated buffers
        if input.rows <= batchSize {
            return forwardChunk(input)
        }
        var resultData = [Float]()
        resultData.reserveCapacity(input.rows * outputSize)
        var offset = 0
        while offset < input.rows {
            let bs = min(batchSize, input.rows - offset)
            let chunk = input.slice(rowStart: offset, rowCount: bs)
            let out = forwardChunk(chunk)
            resultData.append(contentsOf: out.data)
            offset += bs
        }
        return Matrix(rows: input.rows, cols: outputSize, data: resultData)
    }

    public func layerBackward(_ outputGradient: Matrix, backend: ComputeBackend) -> LayerBackwardResult {
        guard let inBuf = inputBuf else {
            fatalError("MetalRationalKANLayer: backward before forward")
        }
        let bs = outputGradient.rows
        let edgeCount = bs * outputSize * inputSize

        let outGradBuf = ctx.makeBuffer(outputGradient.data)

        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer() else {
            return LayerBackwardResult(inputGradient: Matrix(rows: bs, cols: inputSize), storage: AnySendable(0))
        }

        // Backward kernel: compute per-edge gradients
        encode(cmdBuf, pipeline: ctx.rkanBackward,
               buffers: [(inBuf, 0), (numCoeffsBuf, 1), (denCoeffsBuf, 2),
                         (pValsBuf, 3), (qValsBuf, 4), (outGradBuf, 5),
                         (numGradBuf, 6), (denGradBuf, 7), (inputGradEdgeBuf, 8)],
               uints: [(UInt32(bs), 9), (UInt32(inputSize), 10), (UInt32(outputSize), 11),
                       (UInt32(numDegree), 12), (UInt32(denDegree), 13)],
               threads: edgeCount)

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Reduce gradients on CPU (sum across batch for coefficient grads, across outSize for input grads)
        let numGradAll = ctx.readBuffer(numGradBuf, count: bs * outputSize * inputSize * (numDegree + 1))
        let denGradAll = ctx.readBuffer(denGradBuf, count: bs * outputSize * inputSize * denDegree)
        let inputGradEdge = ctx.readBuffer(inputGradEdgeBuf, count: edgeCount)

        // Sum numGrad across batch: [outSize, inSize*(p+1)]
        let numCoeffCount = outputSize * inputSize * (numDegree + 1)
        let denCoeffCount = outputSize * inputSize * denDegree
        var numGrad = [Float](repeating: 0, count: numCoeffCount)
        var denGrad = [Float](repeating: 0, count: denCoeffCount)

        for b in 0..<bs {
            for k in 0..<numCoeffCount {
                numGrad[k] += numGradAll[b * numCoeffCount + k]
            }
            for k in 0..<denCoeffCount {
                denGrad[k] += denGradAll[b * denCoeffCount + k]
            }
        }

        // Sum input grad across outSize: inputGrad[b, i] = Σ_j inputGradEdge[b, j, i]
        var inputGrad = Matrix(rows: bs, cols: inputSize)
        for b in 0..<bs {
            for j in 0..<outputSize {
                for i in 0..<inputSize {
                    inputGrad.data[b * inputSize + i] += inputGradEdge[b * outputSize * inputSize + j * inputSize + i]
                }
            }
        }

        let grads = MetalRationalKANGradients(
            numGrad: Matrix(rows: outputSize, cols: inputSize * (numDegree + 1), data: numGrad),
            denGrad: Matrix(rows: outputSize, cols: inputSize * denDegree, data: denGrad)
        )
        return LayerBackwardResult(inputGradient: inputGrad, storage: AnySendable(grads))
    }

    public mutating func updateParameters(gradients: LayerBackwardResult, learningRate: Float, backend: ComputeBackend) {
        guard let grads = gradients.storage.get(MetalRationalKANGradients.self) else {
            fatalError("MetalRationalKANLayer: wrong gradient type")
        }

        // Update on CPU then re-upload (simpler than GPU kernel for this)
        let numCoeffCount = outputSize * inputSize * (numDegree + 1)
        let denCoeffCount = outputSize * inputSize * denDegree

        var numData = ctx.readBuffer(numCoeffsBuf, count: numCoeffCount)
        var denData = ctx.readBuffer(denCoeffsBuf, count: denCoeffCount)

        for i in 0..<numCoeffCount {
            numData[i] -= learningRate * grads.numGrad.data[i]
        }
        for i in 0..<denCoeffCount {
            denData[i] -= learningRate * grads.denGrad.data[i]
        }

        numCoeffsBuf = ctx.makeBuffer(numData)
        denCoeffsBuf = ctx.makeBuffer(denData)
    }
}

struct MetalRationalKANGradients: Sendable {
    let numGrad: Matrix
    let denGrad: Matrix
}
