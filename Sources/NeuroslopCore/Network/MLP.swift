/// Multi-layer perceptron — a stack of dense layers.
public struct MLP: Sendable {
    public var layers: [DenseLayer]
    public let backend: ComputeBackend

    /// Create an MLP with the given layer sizes.
    /// Example: [784, 128, 64, 10] creates 3 layers.
    /// activations[i] is the activation for layer i.
    public init(layerSizes: [Int], activations: [ActivationKind], backend: ComputeBackend) {
        precondition(layerSizes.count >= 2, "Need at least input and output sizes")
        precondition(activations.count == layerSizes.count - 1, "Need one activation per layer")

        self.backend = backend
        self.layers = []
        for i in 0..<activations.count {
            layers.append(DenseLayer(
                inputSize: layerSizes[i],
                outputSize: layerSizes[i + 1],
                activation: activations[i]
            ))
        }
    }

    /// Forward pass through all layers.
    /// Input: (batchSize x inputSize), Output: (batchSize x outputSize)
    public mutating func forward(_ input: Matrix) -> Matrix {
        var x = input
        for i in 0..<layers.count {
            x = layers[i].forward(x, backend: backend)
        }
        return x
    }

    /// Backward pass: propagate loss gradient through all layers (reverse order).
    /// Returns gradients for each layer's weights and biases.
    public func backward(_ lossGradient: Matrix) -> [BackwardResult] {
        var gradient = lossGradient
        var results: [BackwardResult] = []
        for i in stride(from: layers.count - 1, through: 0, by: -1) {
            let result = layers[i].backward(gradient, backend: backend)
            results.append(result)
            gradient = result.inputGradient
        }
        results.reverse() // align with layer indices
        return results
    }

    /// SGD weight update.
    public mutating func updateWeights(gradients: [BackwardResult], learningRate: Float) {
        for i in 0..<layers.count {
            layers[i].weights = backend.subtract(
                layers[i].weights,
                backend.scalarMultiply(gradients[i].weightGradient, learningRate)
            )
            layers[i].biases = backend.subtract(
                layers[i].biases,
                backend.scalarMultiply(gradients[i].biasGradient, learningRate)
            )
        }
    }
}
