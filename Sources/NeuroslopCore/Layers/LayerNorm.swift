import Foundation

/// Layer Normalization — normalizes each sample independently.
/// Used between Dense and KAN layers to keep inputs in a bounded range.
/// Normalizes to mean=0, std=1, then applies learnable scale and shift.
///
/// LayerNorm has no "inputSize vs outputSize" — it preserves dimensions.
public struct LayerNormLayer: Sendable {
    public let size: Int
    public var gamma: [Float]  // scale (learnable)
    public var beta: [Float]   // shift (learnable)
    private let eps: Float

    private var lastInput: Matrix?
    private var lastNormalized: Matrix?
    private var lastStd: [Float]?

    public init(size: Int, eps: Float = 1e-5) {
        self.size = size
        self.eps = eps
        self.gamma = [Float](repeating: 1.0, count: size)
        self.beta = [Float](repeating: 0.0, count: size)
    }
}

extension LayerNormLayer: Layer {
    public var inputSize: Int { size }
    public var outputSize: Int { size }

    public mutating func forward(_ input: Matrix, backend: ComputeBackend) -> Matrix {
        lastInput = input
        let bs = input.rows
        var result = Matrix(rows: bs, cols: size)
        var stds = [Float](repeating: 0, count: bs)
        var normalized = Matrix(rows: bs, cols: size)

        for b in 0..<bs {
            // Compute mean
            var mean: Float = 0
            for j in 0..<size { mean += input[b, j] }
            mean /= Float(size)

            // Compute variance
            var variance: Float = 0
            for j in 0..<size {
                let diff = input[b, j] - mean
                variance += diff * diff
            }
            variance /= Float(size)
            let std = sqrtf(variance + eps)
            stds[b] = std

            // Normalize and apply gamma/beta
            for j in 0..<size {
                let norm = (input[b, j] - mean) / std
                normalized[b, j] = norm
                result[b, j] = gamma[j] * norm + beta[j]
            }
        }

        lastNormalized = normalized
        lastStd = stds
        return result
    }

    public func layerBackward(_ outputGradient: Matrix, backend: ComputeBackend) -> LayerBackwardResult {
        guard let normalized = lastNormalized, let stds = lastStd else {
            fatalError("LayerNormLayer: backward called before forward")
        }
        let bs = outputGradient.rows

        var gammaGrad = [Float](repeating: 0, count: size)
        var betaGrad = [Float](repeating: 0, count: size)
        var inputGrad = Matrix(rows: bs, cols: size)

        for b in 0..<bs {
            let invStd = 1.0 / stds[b]
            let invN = 1.0 / Float(size)

            // Gamma and beta gradients
            for j in 0..<size {
                gammaGrad[j] += outputGradient[b, j] * normalized[b, j]
                betaGrad[j] += outputGradient[b, j]
            }

            // Input gradient (simplified LayerNorm backward)
            // dL/dx = gamma/std * (dL/dy - mean(dL/dy) - normalized * mean(dL/dy * normalized))
            var meanDy: Float = 0
            var meanDyNorm: Float = 0
            for j in 0..<size {
                let dy = outputGradient[b, j] * gamma[j]
                meanDy += dy
                meanDyNorm += dy * normalized[b, j]
            }
            meanDy *= invN
            meanDyNorm *= invN

            for j in 0..<size {
                let dy = outputGradient[b, j] * gamma[j]
                inputGrad[b, j] = invStd * (dy - meanDy - normalized[b, j] * meanDyNorm)
            }
        }

        let grads = LayerNormGradients(gammaGrad: gammaGrad, betaGrad: betaGrad)
        return LayerBackwardResult(inputGradient: inputGrad, storage: AnySendable(grads))
    }

    public mutating func updateParameters(gradients: LayerBackwardResult, learningRate: Float, backend: ComputeBackend) {
        guard let grads = gradients.storage.get(LayerNormGradients.self) else {
            fatalError("LayerNormLayer: wrong gradient type")
        }
        for j in 0..<size {
            gamma[j] -= learningRate * grads.gammaGrad[j]
            beta[j] -= learningRate * grads.betaGrad[j]
        }
    }
}

struct LayerNormGradients: Sendable {
    let gammaGrad: [Float]
    let betaGrad: [Float]
}
