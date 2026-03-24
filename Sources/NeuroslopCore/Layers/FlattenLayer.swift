/// Identity layer that documents the transition from conv (C*H*W) to dense.
/// Data is already flat in Matrix(batch, C*H*W), so forward/backward are passthrough.
public struct FlattenLayer: Sendable {
    public let size: Int
    public var inputSize: Int { size }
    public var outputSize: Int { size }

    public init(size: Int) {
        self.size = size
    }
}

extension FlattenLayer: Layer {
    public mutating func forward(_ input: Matrix, backend: ComputeBackend) -> Matrix { input }

    public func layerBackward(_ outputGradient: Matrix, backend: ComputeBackend) -> LayerBackwardResult {
        LayerBackwardResult(inputGradient: outputGradient, storage: AnySendable(0))
    }

    public mutating func updateParameters(gradients: LayerBackwardResult, learningRate: Float, backend: ComputeBackend) {}
}
