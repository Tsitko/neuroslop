/// Fully-connected (dense) layer.
/// Forward:  output = activation(input * weights + bias)
/// Backward: computes gradients for input, weights, and bias.
public struct DenseLayer: Sendable {
    public var weights: Matrix    // (inputSize x outputSize)
    public var biases: Matrix     // (1 x outputSize)
    public let activation: ActivationKind

    // Cached during forward pass for backprop
    private(set) var lastInput: Matrix?       // (batchSize x inputSize)
    private(set) var lastPreActivation: Matrix? // z = input * weights + bias
    private(set) var lastOutput: Matrix?      // a = activation(z)

    public let inputSize: Int
    public let outputSize: Int

    public init(inputSize: Int, outputSize: Int, activation: ActivationKind) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = activation

        // He init for ReLU, Xavier for others
        switch activation {
        case .relu:
            self.weights = Matrix.heInit(rows: inputSize, cols: outputSize, fanIn: inputSize)
        default:
            self.weights = Matrix.xavierInit(rows: inputSize, cols: outputSize, fanIn: inputSize, fanOut: outputSize)
        }
        self.biases = Matrix(rows: 1, cols: outputSize)
    }

    /// Forward pass. Input shape: (batchSize x inputSize).
    /// Returns: (batchSize x outputSize).
    public mutating func forward(_ input: Matrix, backend: ComputeBackend) -> Matrix {
        lastInput = input
        let z = backend.addBias(backend.matmul(input, weights), biases)
        lastPreActivation = z
        let a = backend.activate(z, activation)
        lastOutput = a
        return a
    }

    /// Backward pass. Receives gradient of loss w.r.t. this layer's output.
    /// Returns gradient of loss w.r.t. this layer's input.
    /// Also computes weight and bias gradients (stored in returned struct).
    public func backward(_ outputGradient: Matrix, backend: ComputeBackend) -> BackwardResult {
        guard let input = lastInput, let output = lastOutput else {
            fatalError("backward() called before forward()")
        }

        // For softmax + cross-entropy, the gradient is already dL/dz (combined).
        // For other activations: dL/dz = dL/da * activation'(z)
        let dz: Matrix
        if activation == .softmax {
            dz = outputGradient
        } else {
            let activDeriv = backend.activationDerivative(output, activation)
            dz = backend.elementwiseMultiply(outputGradient, activDeriv)
        }

        // dL/dW = input^T * dz
        let weightGradient = backend.transposedMatmul(input, dz)

        // dL/db = sum of dz across batch (rows)
        let biasGradient = backend.sumRows(dz)

        // dL/d(input) = dz * W^T
        let inputGradient = backend.matmulTransposedB(dz, weights)

        return BackwardResult(
            inputGradient: inputGradient,
            weightGradient: weightGradient,
            biasGradient: biasGradient
        )
    }
}

public struct BackwardResult: Sendable {
    public let inputGradient: Matrix
    public let weightGradient: Matrix
    public let biasGradient: Matrix
}

// MARK: - Layer protocol conformance

extension DenseLayer: Layer {
    public func layerBackward(_ outputGradient: Matrix, backend: ComputeBackend) -> LayerBackwardResult {
        let result: BackwardResult = backward(outputGradient, backend: backend)
        return LayerBackwardResult(
            inputGradient: result.inputGradient,
            storage: AnySendable(result)
        )
    }

    public mutating func updateParameters(gradients: LayerBackwardResult, learningRate: Float, backend: ComputeBackend) {
        guard let result = gradients.storage.get(BackwardResult.self) else {
            fatalError("DenseLayer.updateParameters: wrong gradient type")
        }
        weights = backend.subtract(weights, backend.scalarMultiply(result.weightGradient, learningRate))
        biases = backend.subtract(biases, backend.scalarMultiply(result.biasGradient, learningRate))
    }
}
