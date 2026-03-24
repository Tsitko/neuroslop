/// Network with heterogeneous layer types (Dense, KAN, Fourier, Rational).
/// Uses the Layer protocol for uniform forward/backward/update.
public struct HybridNetwork {
    public var layers: [any Layer]
    public let backend: ComputeBackend

    public init(layers: [any Layer], backend: ComputeBackend) {
        self.layers = layers
        self.backend = backend
    }

    /// Forward pass through all layers.
    public mutating func forward(_ input: Matrix) -> Matrix {
        var x = input
        for i in 0..<layers.count {
            x = layers[i].forward(x, backend: backend)
        }
        return x
    }

    /// Backward pass: propagate loss gradient through all layers.
    public func backward(_ lossGradient: Matrix) -> [LayerBackwardResult] {
        var gradient = lossGradient
        var results: [LayerBackwardResult] = []
        for i in stride(from: layers.count - 1, through: 0, by: -1) {
            let result = layers[i].layerBackward(gradient, backend: backend)
            results.append(result)
            gradient = result.inputGradient
        }
        results.reverse()
        return results
    }

    /// Update all layer parameters.
    public mutating func updateParameters(gradients: [LayerBackwardResult], learningRate: Float) {
        for i in 0..<layers.count {
            layers[i].updateParameters(gradients: gradients[i], learningRate: learningRate, backend: backend)
        }
    }

    /// Update using Adam optimizer.
    public mutating func updateWithAdam(gradients: [LayerBackwardResult], states: inout [[AdamState]],
                                         lr: Float, beta1: Float = 0.9, beta2: Float = 0.999, eps: Float = 1e-8) {
        for i in 0..<layers.count {
            if var adamLayer = layers[i] as? any AdamLayer {
                adamLayer.updateWithAdam(gradients: gradients[i], states: &states[i],
                                          lr: lr, beta1: beta1, beta2: beta2, eps: eps)
                layers[i] = adamLayer
            } else {
                layers[i].updateParameters(gradients: gradients[i], learningRate: lr, backend: backend)
            }
        }
    }

    /// Create Adam states for all layers.
    public func createAdamStates() -> [[AdamState]] {
        layers.map { layer in
            if let adamLayer = layer as? any AdamLayer {
                return adamLayer.createAdamStates()
            }
            return []
        }
    }

    /// Total parameter count across all layers.
    public var parameterCount: Int {
        layers.reduce(0) { total, layer in
            total + layer.inputSize * layer.outputSize
        }
    }
}
