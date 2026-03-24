import Foundation

/// Dropout layer — randomly zeroes elements during training.
/// At test time passes through unchanged.
///
/// During training: each element has probability `rate` of being set to zero.
/// Remaining elements are scaled by 1/(1-rate) to maintain expected sum (inverted dropout).
public struct DropoutLayer: Sendable {
    public let size: Int
    public let rate: Float  // probability of dropping (typically 0.5)
    public var training: Bool

    // Cached mask for backward
    private var lastMask: [Float]?

    public init(size: Int, rate: Float = 0.5) {
        self.size = size
        self.rate = rate
        self.training = true
    }
}

extension DropoutLayer: Layer {
    public var inputSize: Int { size }
    public var outputSize: Int { size }

    public mutating func forward(_ input: Matrix, backend: ComputeBackend) -> Matrix {
        if !training {
            lastMask = nil
            return input
        }

        let scale = 1.0 / (1.0 - rate)
        var mask = [Float](repeating: 0, count: input.count)
        var result = Matrix(rows: input.rows, cols: input.cols)

        for i in 0..<input.count {
            if Float.random(in: 0..<1) >= rate {
                mask[i] = scale
                result.data[i] = input.data[i] * scale
            }
            // else: mask[i] = 0, result.data[i] = 0 (already initialized)
        }

        lastMask = mask
        return result
    }

    public func layerBackward(_ outputGradient: Matrix, backend: ComputeBackend) -> LayerBackwardResult {
        guard let mask = lastMask else {
            // Not training — passthrough
            return LayerBackwardResult(inputGradient: outputGradient, storage: AnySendable(0))
        }

        // Gradient flows only through non-dropped elements
        var inputGrad = Matrix(rows: outputGradient.rows, cols: outputGradient.cols)
        for i in 0..<outputGradient.count {
            inputGrad.data[i] = outputGradient.data[i] * mask[i]
        }

        return LayerBackwardResult(inputGradient: inputGrad, storage: AnySendable(0))
    }

    public mutating func updateParameters(gradients: LayerBackwardResult, learningRate: Float, backend: ComputeBackend) {}
}
