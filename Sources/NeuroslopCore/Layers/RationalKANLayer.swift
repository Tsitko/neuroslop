import Foundation

/// KAN layer using rational functions P(x)/Q(x) on each edge.
///
/// Each edge (i→j):
///   φ_{j,i}(x) = P(x) / Q(x)
///   P(x) = Σ(k=0..p) a[j,i,k] · x^k
///   Q(x) = 1 + Σ(k=1..q) b[j,i,k]² · x^(2k)    [Q > 0 always]
///
/// Optimized: vectorized evaluation of P and Q using power matrices,
/// then sum across input dim via matmul-like operation.
public struct RationalKANLayer: Sendable {
    public let inputSize: Int
    public let outputSize: Int
    public let numDegree: Int  // p
    public let denDegree: Int  // q

    // Parameters stored as matrices for easier gradient computation
    // numCoeffs: [outSize, inSize * (p+1)] — numerator coefficients
    // denCoeffs: [outSize, inSize * q] — denominator coefficients (squared for stability)
    public var numCoeffs: Matrix
    public var denCoeffs: Matrix

    // Cached
    private var lastInput: Matrix?
    private var lastPhiValues: Matrix?  // [batch, outSize * inSize] — per-edge φ values
    private var lastP: [Float]?
    private var lastQ: [Float]?

    public var totalParameters: Int { numCoeffs.count + denCoeffs.count }

    public init(inputSize: Int, outputSize: Int, numDegree: Int = 3, denDegree: Int = 2) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.numDegree = numDegree
        self.denDegree = denDegree

        // Init: small random, a[0] slightly larger
        var numData = [Float](repeating: 0, count: outputSize * inputSize * (numDegree + 1))
        for idx in 0..<numData.count {
            let k = idx % (numDegree + 1)
            numData[idx] = k == 0 ? Float.random(in: -0.1...0.1) : Float.random(in: -0.01...0.01)
        }
        self.numCoeffs = Matrix(rows: outputSize, cols: inputSize * (numDegree + 1), data: numData)
        self.denCoeffs = Matrix(rows: outputSize, cols: inputSize * denDegree,
                                data: (0..<outputSize * inputSize * denDegree).map { _ in Float.random(in: -0.01...0.01) })
    }
}

extension RationalKANLayer: Layer {
    public mutating func forward(_ input: Matrix, backend: ComputeBackend) -> Matrix {
        lastInput = input
        let bs = input.rows
        let p = numDegree
        let q = denDegree

        // Evaluate P and Q for all (batch, output, input) combinations
        let edgeCount = outputSize * inputSize
        var pVals = [Float](repeating: 0, count: bs * edgeCount)
        var qVals = [Float](repeating: 0, count: bs * edgeCount)

        // Build power-of-x matrix: xPow[b, i*(p+1) + k] = x[b,i]^k
        // Use for P evaluation via matmul-like operation
        for b in 0..<bs {
            for i in 0..<inputSize {
                let x = input[b, i]
                for j in 0..<outputSize {
                    let eIdx = b * edgeCount + j * inputSize + i
                    let aBase = j * inputSize * (p + 1) + i * (p + 1)
                    let bBase = j * inputSize * q + i * q

                    // P(x) via Horner
                    var pv: Float = numCoeffs.data[aBase + p]
                    for k in stride(from: p - 1, through: 0, by: -1) {
                        pv = pv * x + numCoeffs.data[aBase + k]
                    }
                    pVals[eIdx] = pv

                    // Q(x) = 1 + Σ b²·x^(2k)
                    var qv: Float = 1.0
                    var x2k: Float = x * x
                    for k in 0..<q {
                        let bk = denCoeffs.data[bBase + k]
                        qv += bk * bk * x2k
                        x2k *= x * x
                    }
                    qVals[eIdx] = qv
                }
            }
        }

        lastP = pVals
        lastQ = qVals

        // output[b,j] = Σ_i P/Q
        // Build phi matrix [batch, outSize * inSize], then sum groups of inSize
        var result = Matrix(rows: bs, cols: outputSize)
        for b in 0..<bs {
            for j in 0..<outputSize {
                var sum: Float = 0
                for i in 0..<inputSize {
                    let idx = b * edgeCount + j * inputSize + i
                    sum += pVals[idx] / qVals[idx]
                }
                result[b, j] = sum
            }
        }
        return result
    }

    public func layerBackward(_ outputGradient: Matrix, backend: ComputeBackend) -> LayerBackwardResult {
        guard let input = lastInput, let pVals = lastP, let qVals = lastQ else {
            fatalError("RationalKANLayer: backward called before forward")
        }
        let bs = input.rows
        let p = numDegree
        let q = denDegree
        let edgeCount = outputSize * inputSize

        var numGrad = Matrix(rows: numCoeffs.rows, cols: numCoeffs.cols)
        var denGrad = Matrix(rows: denCoeffs.rows, cols: denCoeffs.cols)
        var inputGrad = Matrix(rows: bs, cols: inputSize)

        for b in 0..<bs {
            for j in 0..<outputSize {
                let delta = outputGradient[b, j]
                for i in 0..<inputSize {
                    let x = input[b, i]
                    let eIdx = b * edgeCount + j * inputSize + i
                    let pv = pVals[eIdx]
                    let qv = qVals[eIdx]
                    let qInv = 1.0 / qv
                    let qInv2 = qInv * qInv
                    let aBase = j * inputSize * (p + 1) + i * (p + 1)
                    let bBase = j * inputSize * q + i * q

                    // ∂(P/Q)/∂a_k = x^k / Q
                    var xk: Float = 1.0
                    for k in 0...p {
                        numGrad.data[aBase + k] += delta * xk * qInv
                        xk *= x
                    }

                    // ∂(P/Q)/∂b_k = -P · 2b_k · x^(2(k+1)) / Q²
                    var x2k: Float = x * x
                    for k in 0..<q {
                        let bk = denCoeffs.data[bBase + k]
                        denGrad.data[bBase + k] += delta * (-pv * 2.0 * bk * x2k * qInv2)
                        x2k *= x * x
                    }

                    // ∂(P/Q)/∂x = (P'Q - PQ') / Q²
                    var dPdx: Float = 0
                    xk = 1.0
                    for k in 1...p {
                        dPdx += Float(k) * numCoeffs.data[aBase + k] * xk
                        xk *= x
                    }
                    var dQdx: Float = 0
                    var x2km1: Float = x
                    for k in 0..<q {
                        let bk = denCoeffs.data[bBase + k]
                        dQdx += Float(2 * (k + 1)) * bk * bk * x2km1
                        x2km1 *= x * x
                    }
                    inputGrad[b, i] += delta * (dPdx * qv - pv * dQdx) * qInv2
                }
            }
        }

        let grads = RationalKANGradients(numGrad: numGrad, denGrad: denGrad)
        return LayerBackwardResult(inputGradient: inputGrad, storage: AnySendable(grads))
    }

    public mutating func updateParameters(gradients: LayerBackwardResult, learningRate: Float, backend: ComputeBackend) {
        guard let grads = gradients.storage.get(RationalKANGradients.self) else {
            fatalError("RationalKANLayer: wrong gradient type")
        }
        numCoeffs = backend.subtract(numCoeffs, backend.scalarMultiply(grads.numGrad, learningRate))
        denCoeffs = backend.subtract(denCoeffs, backend.scalarMultiply(grads.denGrad, learningRate))
    }
}

struct RationalKANGradients: Sendable {
    let numGrad: Matrix
    let denGrad: Matrix
}
