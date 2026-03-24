import Foundation

/// Batch Normalization for convolutional layers.
/// Normalizes per-channel across batch and spatial dimensions.
///
/// Input:  Matrix(batch, C * H * W) — channel-first layout
/// Output: Matrix(batch, C * H * W) — normalized, same shape
///
/// During training: uses batch statistics (mean, variance).
/// Learnable parameters: gamma (scale) and beta (shift) per channel.
/// Also tracks running mean/variance for inference (not implemented yet).
public struct BatchNormLayer: Sendable {
    public let channels: Int
    public let height: Int
    public let width: Int
    private let eps: Float
    private let spatialSize: Int

    // Learnable parameters (per channel)
    public var gamma: [Float]  // scale
    public var beta: [Float]   // shift

    // Cached for backward
    private var lastInput: Matrix?
    private var lastNormalized: [Float]?  // (batch * C * H * W)
    private var lastMean: [Float]?        // (C)
    private var lastInvStd: [Float]?      // (C)

    public init(channels: Int, height: Int, width: Int, eps: Float = 1e-5) {
        self.channels = channels
        self.height = height
        self.width = width
        self.eps = eps
        self.spatialSize = height * width
        self.gamma = [Float](repeating: 1.0, count: channels)
        self.beta = [Float](repeating: 0.0, count: channels)
    }
}

extension BatchNormLayer: Layer {
    public var inputSize: Int { channels * height * width }
    public var outputSize: Int { channels * height * width }

    public mutating func forward(_ input: Matrix, backend: ComputeBackend) -> Matrix {
        lastInput = input
        let bs = input.rows
        let chSize = spatialSize
        let totalPerChannel = bs * chSize  // samples contributing to each channel's stats

        // Compute per-channel mean
        var mean = [Float](repeating: 0, count: channels)
        for b in 0..<bs {
            for c in 0..<channels {
                let base = b * inputSize + c * chSize
                for s in 0..<chSize {
                    mean[c] += input.data[base + s]
                }
            }
        }
        let invN = 1.0 / Float(totalPerChannel)
        for c in 0..<channels { mean[c] *= invN }

        // Compute per-channel variance
        var variance = [Float](repeating: 0, count: channels)
        for b in 0..<bs {
            for c in 0..<channels {
                let base = b * inputSize + c * chSize
                for s in 0..<chSize {
                    let diff = input.data[base + s] - mean[c]
                    variance[c] += diff * diff
                }
            }
        }
        for c in 0..<channels { variance[c] *= invN }

        // Normalize and apply gamma/beta
        var invStd = [Float](repeating: 0, count: channels)
        for c in 0..<channels { invStd[c] = 1.0 / sqrtf(variance[c] + eps) }

        var normalized = [Float](repeating: 0, count: bs * inputSize)
        var result = Matrix(rows: bs, cols: inputSize)

        for b in 0..<bs {
            for c in 0..<channels {
                let base = b * inputSize + c * chSize
                for s in 0..<chSize {
                    let idx = base + s
                    let norm = (input.data[idx] - mean[c]) * invStd[c]
                    normalized[idx] = norm
                    result.data[idx] = gamma[c] * norm + beta[c]
                }
            }
        }

        lastNormalized = normalized
        lastMean = mean
        lastInvStd = invStd
        return result
    }

    public func layerBackward(_ outputGradient: Matrix, backend: ComputeBackend) -> LayerBackwardResult {
        guard let input = lastInput, let normalized = lastNormalized,
              let invStd = lastInvStd else {
            fatalError("BatchNormLayer: backward before forward")
        }
        let bs = outputGradient.rows
        let chSize = spatialSize
        let N = Float(bs * chSize)

        var gammaGrad = [Float](repeating: 0, count: channels)
        var betaGrad = [Float](repeating: 0, count: channels)

        // Compute gamma/beta gradients
        for b in 0..<bs {
            for c in 0..<channels {
                let base = b * inputSize + c * chSize
                for s in 0..<chSize {
                    let idx = base + s
                    gammaGrad[c] += outputGradient.data[idx] * normalized[idx]
                    betaGrad[c] += outputGradient.data[idx]
                }
            }
        }

        // Input gradient
        // dL/dx = (1/N) * gamma * invStd * (N*dL/dy - sum(dL/dy) - normalized * sum(dL/dy * normalized))
        var inputGrad = Matrix(rows: bs, cols: inputSize)

        for c in 0..<channels {
            // Pre-compute sums for this channel
            var sumDy: Float = 0
            var sumDyNorm: Float = 0
            for b in 0..<bs {
                let base = b * inputSize + c * chSize
                for s in 0..<chSize {
                    let idx = base + s
                    let dy = outputGradient.data[idx]
                    sumDy += dy
                    sumDyNorm += dy * normalized[idx]
                }
            }

            let scale = gamma[c] * invStd[c] / N
            for b in 0..<bs {
                let base = b * inputSize + c * chSize
                for s in 0..<chSize {
                    let idx = base + s
                    inputGrad.data[idx] = scale * (N * outputGradient.data[idx] - sumDy - normalized[idx] * sumDyNorm)
                }
            }
        }

        let grads = BatchNormGradients(gammaGrad: gammaGrad, betaGrad: betaGrad)
        return LayerBackwardResult(inputGradient: inputGrad, storage: AnySendable(grads))
    }

    public mutating func updateParameters(gradients: LayerBackwardResult, learningRate: Float, backend: ComputeBackend) {
        guard let grads = gradients.storage.get(BatchNormGradients.self) else { return }
        for c in 0..<channels {
            gamma[c] -= learningRate * grads.gammaGrad[c]
            beta[c] -= learningRate * grads.betaGrad[c]
        }
    }
}

// MARK: - Adam

extension BatchNormLayer: AdamLayer {
    public var adamStateCount: Int { 2 }
    public func createAdamStates() -> [AdamState] {
        [AdamState(count: channels), AdamState(count: channels)]
    }
    public mutating func updateWithAdam(gradients: LayerBackwardResult, states: inout [AdamState],
                                         lr: Float, beta1: Float, beta2: Float, eps: Float) {
        guard let grads = gradients.storage.get(BatchNormGradients.self) else { return }
        states[0].update(params: &gamma, grads: grads.gammaGrad, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
        states[1].update(params: &self.beta, grads: grads.betaGrad, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
    }
}

struct BatchNormGradients: Sendable {
    let gammaGrad: [Float]
    let betaGrad: [Float]
}
