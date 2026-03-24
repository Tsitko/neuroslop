import Foundation

/// KAN layer using B-spline basis functions on each edge.
///
/// Each edge (i→j):
///   φ_{j,i}(x) = w_base * SiLU(x) + Σ(k=0..G+deg-1) c[j,i,k] * B_{k,deg}(x)
///
/// B-spline basis B_{k,deg}(x) computed via Cox-de Boor recursion.
/// Knots: uniform grid on [-1, 1] with (deg) extra knots on each side.
///
/// Optimized: pre-compute basis matrix [batch, inSize * numBasis], then matmul with coefficients.
public struct BSplineKANLayer: Sendable {
    public let inputSize: Int
    public let outputSize: Int
    public let gridSize: Int      // G: number of grid intervals
    public let splineDegree: Int  // k: B-spline degree (typically 3)
    public let numBasis: Int      // G + k: number of basis functions per input

    // Knot vector: G + 2k + 1 knots (uniform, clamped)
    public let knots: [Float]

    // Coefficients: [inSize * numBasis, outSize] — same layout as Fourier for matmul
    public var coeffs: Matrix

    // Base weight + SiLU: [inSize, outSize]
    public var baseWeights: Matrix

    // Cached
    private var lastInput: Matrix?
    private var lastBasis: Matrix?    // [batch, inSize * numBasis]
    private var lastSiLU: Matrix?     // [batch, inSize] — SiLU(input) for base path

    public var totalParameters: Int { coeffs.count + baseWeights.count }

    public init(inputSize: Int, outputSize: Int, gridSize: Int = 5, splineDegree: Int = 3) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.gridSize = gridSize
        self.splineDegree = splineDegree
        self.numBasis = gridSize + splineDegree

        // Uniform knot vector on [-1, 1] with clamped ends
        let numKnots = gridSize + 2 * splineDegree + 1
        var knotVec = [Float](repeating: 0, count: numKnots)
        let gridStep = 2.0 / Float(gridSize) // span [-1, 1]
        for i in 0..<numKnots {
            let t = -1.0 + Float(i - splineDegree) * gridStep
            knotVec[i] = min(max(t, -1.0), 1.0) // clamp
        }
        // Extend knots slightly beyond [-1,1] for edge basis
        for i in 0..<splineDegree {
            knotVec[i] = -1.0 - Float(splineDegree - i) * gridStep
        }
        for i in 0..<splineDegree {
            knotVec[numKnots - splineDegree + i] = 1.0 + Float(i + 1) * gridStep
        }
        self.knots = knotVec

        // Small random init for spline coefficients
        self.coeffs = Matrix(rows: inputSize * numBasis, cols: outputSize, randomIn: -0.01...0.01)

        // Base weights — small
        self.baseWeights = Matrix(rows: inputSize, cols: outputSize, randomIn: -0.01...0.01)
    }

    /// Evaluate all B-spline basis functions for a single x value.
    /// Returns array of `numBasis` values.
    private func evaluateBasis(x: Float) -> [Float] {
        let n = numBasis
        let k = splineDegree
        let t = knots

        // Cox-de Boor: start with degree 0
        var basis = [Float](repeating: 0, count: n + k)
        for i in 0..<(n + k) {
            if i < t.count - 1 {
                basis[i] = (x >= t[i] && x < t[i + 1]) ? 1.0 : 0.0
            }
        }
        // Handle right boundary
        if x >= t[t.count - k - 1] {
            basis[n - 1] = 1.0
        }

        // Build up degrees
        for d in 1...k {
            var newBasis = [Float](repeating: 0, count: n + k - d)
            for i in 0..<(n + k - d) {
                let denom1 = t[i + d] - t[i]
                let denom2 = t[i + d + 1] - t[i + 1]
                var val: Float = 0
                if abs(denom1) > 1e-10 {
                    val += (x - t[i]) / denom1 * basis[i]
                }
                if abs(denom2) > 1e-10 {
                    val += (t[i + d + 1] - x) / denom2 * basis[i + 1]
                }
                newBasis[i] = val
            }
            basis = newBasis
        }

        return Array(basis.prefix(n))
    }

    /// SiLU activation: x * sigmoid(x)
    private func silu(_ x: Float) -> Float {
        x / (1.0 + expf(-x))
    }

    private func siluDerivative(_ x: Float) -> Float {
        let sig = 1.0 / (1.0 + expf(-x))
        return sig * (1.0 + x * (1.0 - sig))
    }

    /// Build basis matrix [batch, inSize * numBasis] + SiLU matrix [batch, inSize]
    private func buildBasisAndSiLU(_ input: Matrix) -> (Matrix, Matrix) {
        let bs = input.rows
        let totalBasis = inputSize * numBasis
        var basisData = [Float](repeating: 0, count: bs * totalBasis)
        var siluData = [Float](repeating: 0, count: bs * inputSize)

        for b in 0..<bs {
            for i in 0..<inputSize {
                let x = input[b, i]
                // Clamp to knot range for stability
                let xClamped = min(max(x, knots[0]), knots[knots.count - 1] - 1e-6)
                let basisVals = evaluateBasis(x: xClamped)
                for k in 0..<numBasis {
                    basisData[b * totalBasis + i * numBasis + k] = basisVals[k]
                }
                siluData[b * inputSize + i] = silu(x)
            }
        }

        return (
            Matrix(rows: bs, cols: totalBasis, data: basisData),
            Matrix(rows: bs, cols: inputSize, data: siluData)
        )
    }
}

extension BSplineKANLayer: Layer {
    public mutating func forward(_ input: Matrix, backend: ComputeBackend) -> Matrix {
        lastInput = input

        let (basis, siluVals) = buildBasisAndSiLU(input)
        lastBasis = basis
        lastSiLU = siluVals

        // Spline path: basis @ coeffs → [batch, outSize]
        let splineOut = backend.matmul(basis, coeffs)

        // Base path: siluVals @ baseWeights → [batch, outSize]
        let baseOut = backend.matmul(siluVals, baseWeights)

        // output = baseOut + splineOut
        return backend.add(baseOut, splineOut)
    }

    public func layerBackward(_ outputGradient: Matrix, backend: ComputeBackend) -> LayerBackwardResult {
        guard let input = lastInput, let basis = lastBasis, let siluVals = lastSiLU else {
            fatalError("BSplineKANLayer: backward called before forward")
        }

        // Spline gradient: dCoeffs = basis^T @ dOutput
        let coeffGrad = backend.transposedMatmul(basis, outputGradient)

        // Base gradient: dBaseWeights = siluVals^T @ dOutput
        let baseGrad = backend.transposedMatmul(siluVals, outputGradient)

        // Input gradient: two paths
        // Spline: dBasis = dOutput @ coeffs^T, then dInput from basis derivatives
        let dBasis = backend.matmulTransposedB(outputGradient, coeffs)

        // Base: dSiLU = dOutput @ baseWeights^T, then dInput = dSiLU * SiLU'(x)
        let dSiLU = backend.matmulTransposedB(outputGradient, baseWeights)

        let bs = input.rows
        var inputGrad = Matrix(rows: bs, cols: inputSize)

        for b in 0..<bs {
            for i in 0..<inputSize {
                let x = input[b, i]

                // Base path derivative
                var grad = dSiLU[b, i] * siluDerivative(x)

                // Spline path: approximate derivative by finite difference of basis
                // (exact B-spline derivative is complex, this is simpler and works)
                let eps: Float = 1e-4
                let basisPlus = evaluateBasis(x: min(x + eps, knots[knots.count - 1] - 1e-6))
                let basisMinus = evaluateBasis(x: max(x - eps, knots[0]))
                let invEps2 = 1.0 / (2.0 * eps)
                for k in 0..<numBasis {
                    let dBdx = (basisPlus[k] - basisMinus[k]) * invEps2
                    grad += dBasis[b, i * numBasis + k] * dBdx
                }

                inputGrad[b, i] = grad
            }
        }

        let grads = BSplineKANGradients(coeffGrad: coeffGrad, baseGrad: baseGrad)
        return LayerBackwardResult(inputGradient: inputGrad, storage: AnySendable(grads))
    }

    public mutating func updateParameters(gradients: LayerBackwardResult, learningRate: Float, backend: ComputeBackend) {
        guard let grads = gradients.storage.get(BSplineKANGradients.self) else { return }
        coeffs = backend.subtract(coeffs, backend.scalarMultiply(grads.coeffGrad, learningRate))
        baseWeights = backend.subtract(baseWeights, backend.scalarMultiply(grads.baseGrad, learningRate))
    }
}

// MARK: - Adam support

extension BSplineKANLayer: AdamLayer {
    public var adamStateCount: Int { 2 }

    public func createAdamStates() -> [AdamState] {
        [AdamState(count: coeffs.count), AdamState(count: baseWeights.count)]
    }

    public mutating func updateWithAdam(gradients: LayerBackwardResult, states: inout [AdamState],
                                         lr: Float, beta1: Float, beta2: Float, eps: Float) {
        guard let grads = gradients.storage.get(BSplineKANGradients.self) else { return }
        states[0].update(params: &coeffs.data, grads: grads.coeffGrad.data, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
        states[1].update(params: &baseWeights.data, grads: grads.baseGrad.data, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
    }
}

struct BSplineKANGradients: Sendable {
    let coeffGrad: Matrix
    let baseGrad: Matrix
}
