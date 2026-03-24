import Foundation

public enum LossKind: Sendable {
    case mse
    case crossEntropy
}

extension LossKind {

    /// Compute loss value (scalar, averaged over batch).
    public func loss(predicted: Matrix, target: Matrix) -> Float {
        precondition(predicted.rows == target.rows && predicted.cols == target.cols)
        switch self {
        case .mse:
            var sum: Float = 0
            for i in 0..<predicted.data.count {
                let diff = predicted.data[i] - target.data[i]
                sum += diff * diff
            }
            return sum / Float(predicted.rows)

        case .crossEntropy:
            // -sum(target * log(predicted)) / batchSize
            var sum: Float = 0
            for i in 0..<predicted.data.count {
                let p = max(predicted.data[i], 1e-7) // clamp for log stability
                sum -= target.data[i] * logf(p)
            }
            return sum / Float(predicted.rows)
        }
    }

    /// Compute gradient of loss w.r.t. predicted values.
    /// Returns dL/d(predicted), same shape as predicted.
    public func gradient(predicted: Matrix, target: Matrix) -> Matrix {
        precondition(predicted.rows == target.rows && predicted.cols == target.cols)
        let batchSize = Float(predicted.rows)

        switch self {
        case .mse:
            // dL/dp = 2*(p - t) / batchSize
            var result = Matrix(rows: predicted.rows, cols: predicted.cols)
            for i in 0..<predicted.data.count {
                result.data[i] = 2.0 * (predicted.data[i] - target.data[i]) / batchSize
            }
            return result

        case .crossEntropy:
            // Combined softmax + cross-entropy gradient: dL/dz = (p - t) / batchSize
            // This is the simplified form when softmax is the last activation.
            var result = Matrix(rows: predicted.rows, cols: predicted.cols)
            for i in 0..<predicted.data.count {
                result.data[i] = (predicted.data[i] - target.data[i]) / batchSize
            }
            return result
        }
    }
}
