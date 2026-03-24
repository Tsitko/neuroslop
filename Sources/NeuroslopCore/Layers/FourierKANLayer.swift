import Foundation

/// KAN layer using Fourier basis functions on each edge.
///
/// Each edge (i→j) has a learnable function:
///   φ_{j,i}(x) = Σ(k=0..K) [c_cos[j,i,k]·cos(k·x) + c_sin[j,i,k]·sin(k·x)]
///
/// Optimized: forward is a single matmul of basis matrix × coefficient matrix.
///   basis[batch, i*(2K+1) + k] = cos/sin(k * input[batch, i])
///   coeffs[i*(2K+1) + k, outSize] = interleaved cos/sin coefficients
///   output = basis @ coeffs
public struct FourierKANLayer: Sendable {
    public let inputSize: Int
    public let outputSize: Int
    public let numFreqs: Int  // K
    public let basisSize: Int // 2K+1 per input, total = inSize * (2K+1)

    // Coefficient matrix: [inSize * (2K+1), outSize]
    // Layout: for input i, frequencies k: [cos_0, sin_1, cos_1, sin_2, cos_2, ..., sin_K, cos_K]
    public var coeffs: Matrix

    // Cached for backward
    private var lastInput: Matrix?
    private var lastBasis: Matrix?  // [batch, inSize * (2K+1)]

    public var totalParameters: Int { coeffs.count }

    public init(inputSize: Int, outputSize: Int, numFreqs: Int = 5) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.numFreqs = numFreqs
        self.basisSize = inputSize * (2 * numFreqs + 1)

        // Small random init
        self.coeffs = Matrix(rows: basisSize, cols: outputSize, randomIn: -0.01...0.01)
    }

    /// Build basis matrix: [batch, inSize * (2K+1)]
    /// For each input i, the basis functions are: [cos(0·x), sin(1·x), cos(1·x), sin(2·x), cos(2·x), ...]
    private func buildBasis(_ input: Matrix) -> Matrix {
        let bs = input.rows
        let K = numFreqs
        let bpf = 2 * K + 1  // basis per feature
        var basis = Matrix(rows: bs, cols: basisSize)

        for b in 0..<bs {
            for i in 0..<inputSize {
                let x = input[b, i]
                let offset = i * bpf
                basis.data[b * basisSize + offset] = 1.0  // cos(0·x) = 1 (DC component)
                for k in 1...K {
                    let kx = Float(k) * x
                    basis.data[b * basisSize + offset + 2 * k - 1] = sinf(kx)
                    basis.data[b * basisSize + offset + 2 * k] = cosf(kx)
                }
            }
        }
        return basis
    }

    /// Build basis derivative matrix for input gradients.
    /// d(cos(k·x))/dx = -k·sin(k·x), d(sin(k·x))/dx = k·cos(k·x)
    private func buildBasisDerivative(_ input: Matrix) -> Matrix {
        let bs = input.rows
        let K = numFreqs
        let bpf = 2 * K + 1
        var dbasis = Matrix(rows: bs, cols: basisSize)

        for b in 0..<bs {
            for i in 0..<inputSize {
                let x = input[b, i]
                let offset = i * bpf
                dbasis.data[b * basisSize + offset] = 0  // d(1)/dx = 0
                for k in 1...K {
                    let kf = Float(k)
                    let kx = kf * x
                    dbasis.data[b * basisSize + offset + 2 * k - 1] = kf * cosf(kx)   // d(sin)/dx
                    dbasis.data[b * basisSize + offset + 2 * k] = -kf * sinf(kx)       // d(cos)/dx
                }
            }
        }
        return dbasis
    }
}

extension FourierKANLayer: Layer {
    public mutating func forward(_ input: Matrix, backend: ComputeBackend) -> Matrix {
        lastInput = input
        let basis = buildBasis(input)
        lastBasis = basis
        // output = basis @ coeffs : [batch, basisSize] × [basisSize, outSize] = [batch, outSize]
        return backend.matmul(basis, coeffs)
    }

    public func layerBackward(_ outputGradient: Matrix, backend: ComputeBackend) -> LayerBackwardResult {
        guard let input = lastInput, let basis = lastBasis else {
            fatalError("FourierKANLayer: backward called before forward")
        }

        // dCoeffs = basis^T @ dOutput : [basisSize, batch] × [batch, outSize] = [basisSize, outSize]
        let coeffGrad = backend.transposedMatmul(basis, outputGradient)

        // dBasis = dOutput @ coeffs^T : [batch, outSize] × [outSize, basisSize] = [batch, basisSize]
        let dBasis = backend.matmulTransposedB(outputGradient, coeffs)

        // Input gradient: dInput[b,i] = Σ_k dBasis[b, i*bpf+k] * d(basis_k)/dx
        let basisDeriv = buildBasisDerivative(input)
        let dBasisScaled = backend.elementwiseMultiply(dBasis, basisDeriv)

        // Sum over basis functions for each input feature
        let bs = input.rows
        let bpf = 2 * numFreqs + 1
        var inputGrad = Matrix(rows: bs, cols: inputSize)
        for b in 0..<bs {
            for i in 0..<inputSize {
                var sum: Float = 0
                let offset = b * basisSize + i * bpf
                for k in 0..<bpf {
                    sum += dBasisScaled.data[offset + k]
                }
                inputGrad[b, i] = sum
            }
        }

        return LayerBackwardResult(
            inputGradient: inputGrad,
            storage: AnySendable(coeffGrad)
        )
    }

    public mutating func updateParameters(gradients: LayerBackwardResult, learningRate: Float, backend: ComputeBackend) {
        guard let coeffGrad = gradients.storage.get(Matrix.self) else {
            fatalError("FourierKANLayer: wrong gradient type")
        }
        coeffs = backend.subtract(coeffs, backend.scalarMultiply(coeffGrad, learningRate))
    }
}
