/// Protocol for neural network layers.
/// Supports heterogeneous layer stacks for hybrid architectures
/// (Dense, KAN spline, Fourier, Rational).
///
/// Each layer type has its own learnable parameters and gradient storage.
/// The `LayerBackwardResult.storage` carries type-specific gradient data
/// that only the producing layer type knows how to interpret.
public protocol Layer {
    var inputSize: Int { get }
    var outputSize: Int { get }

    /// Forward pass. Input: (batchSize x inputSize) -> (batchSize x outputSize).
    mutating func forward(_ input: Matrix, backend: ComputeBackend) -> Matrix

    /// Backward pass. Returns gradient w.r.t. input + internal gradient storage.
    func layerBackward(_ outputGradient: Matrix, backend: ComputeBackend) -> LayerBackwardResult

    /// Apply parameter update from stored gradients.
    mutating func updateParameters(gradients: LayerBackwardResult, learningRate: Float, backend: ComputeBackend)
}

/// Result of a layer's backward pass.
/// `inputGradient` propagates through the chain rule.
/// `storage` holds layer-specific parameter gradients (e.g., weight/bias for Dense,
/// control points for KAN splines, frequencies for Fourier, coefficients for Rational).
public struct LayerBackwardResult: Sendable {
    public let inputGradient: Matrix

    // Type-erased storage for parameter gradients.
    // Each layer type downcasts in its own updateParameters().
    public let storage: AnySendable

    public init(inputGradient: Matrix, storage: AnySendable) {
        self.inputGradient = inputGradient
        self.storage = storage
    }
}

/// Type-erased Sendable wrapper (since `Any` is not Sendable).
public struct AnySendable: Sendable {
    private let _value: any Sendable
    public init(_ value: any Sendable) { self._value = value }
    public func get<T: Sendable>(_ type: T.Type) -> T? { _value as? T }
}
