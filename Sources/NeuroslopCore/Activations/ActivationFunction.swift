import Foundation

/// Activation function type — used as a key for backend dispatch.
public enum ActivationKind: Sendable {
    case relu
    case sigmoid
    case tanh
    case softmax
}

// MARK: - CPU implementations (element-wise, except softmax which is row-wise)

extension ActivationKind {

    /// Apply activation to matrix (forward pass).
    public func apply(_ m: Matrix) -> Matrix {
        switch self {
        case .relu:
            return m.map { max(0, $0) }
        case .sigmoid:
            return m.map { 1.0 / (1.0 + expf(-$0)) }
        case .tanh:
            return m.map { Foundation.tanhf($0) }
        case .softmax:
            return softmaxForward(m)
        }
    }

    /// Compute activation derivative given the activation OUTPUT (not input).
    /// For backprop: given `a = activation(z)` and upstream gradient `dL/da`,
    /// we need `dL/dz = dL/da * activation'(z)`.
    /// Most activations can compute the derivative from the output `a` directly.
    public func derivative(activationOutput a: Matrix) -> Matrix {
        switch self {
        case .relu:
            return a.map { $0 > 0 ? 1.0 : 0.0 }
        case .sigmoid:
            // sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)) = a * (1 - a)
            return Matrix(rows: a.rows, cols: a.cols,
                          data: a.data.map { $0 * (1.0 - $0) })
        case .tanh:
            // tanh'(z) = 1 - tanh(z)^2 = 1 - a^2
            return Matrix(rows: a.rows, cols: a.cols,
                          data: a.data.map { 1.0 - $0 * $0 })
        case .softmax:
            // Softmax derivative is handled differently in the loss layer
            // (combined with cross-entropy for numerical stability).
            // Returning ones here as a passthrough — the real derivative
            // is computed in CrossEntropy.derivative().
            return Matrix(rows: a.rows, cols: a.cols, fill: 1.0)
        }
    }

    private func softmaxForward(_ m: Matrix) -> Matrix {
        var result = Matrix(rows: m.rows, cols: m.cols)
        for i in 0..<m.rows {
            // Numerical stability: subtract max per row
            var maxVal: Float = -Float.infinity
            for j in 0..<m.cols {
                maxVal = max(maxVal, m[i, j])
            }
            var sum: Float = 0
            for j in 0..<m.cols {
                let e = expf(m[i, j] - maxVal)
                result[i, j] = e
                sum += e
            }
            for j in 0..<m.cols {
                result[i, j] /= sum
            }
        }
        return result
    }
}
