import Foundation

// MARK: - Naive CPU matrix operations
// These are intentionally simple (triple-nested loops) to serve as baseline.
// Optimized versions live in AccelerateBackend and MetalBackend.

extension Matrix {

    /// C = A * B  (naive triple loop, O(n^3))
    public func matmul(_ other: Matrix) -> Matrix {
        precondition(cols == other.rows, "Shape mismatch: (\(rows)x\(cols)) * (\(other.rows)x\(other.cols))")
        var result = Matrix(rows: rows, cols: other.cols)
        for i in 0..<rows {
            for k in 0..<cols {
                let a = data[i * cols + k]
                for j in 0..<other.cols {
                    result.data[i * other.cols + j] += a * other.data[k * other.cols + j]
                }
            }
        }
        return result
    }

    /// C = A * B^T  (avoids explicit transpose, better cache access on B)
    public func matmulTransposedB(_ other: Matrix) -> Matrix {
        precondition(cols == other.cols, "Shape mismatch for A*B^T: (\(rows)x\(cols)) * (\(other.rows)x\(other.cols))^T")
        var result = Matrix(rows: rows, cols: other.rows)
        for i in 0..<rows {
            for j in 0..<other.rows {
                var sum: Float = 0
                for k in 0..<cols {
                    sum += data[i * cols + k] * other.data[j * other.cols + k]
                }
                result.data[i * other.rows + j] = sum
            }
        }
        return result
    }

    /// C = A^T * B
    public func transposedMatmul(_ other: Matrix) -> Matrix {
        precondition(rows == other.rows, "Shape mismatch for A^T*B: (\(rows)x\(cols))^T * (\(other.rows)x\(other.cols))")
        var result = Matrix(rows: cols, cols: other.cols)
        for i in 0..<cols {
            for j in 0..<other.cols {
                var sum: Float = 0
                for k in 0..<rows {
                    sum += data[k * cols + i] * other.data[k * other.cols + j]
                }
                result.data[i * other.cols + j] = sum
            }
        }
        return result
    }

    /// Transpose
    public func transposed() -> Matrix {
        var result = Matrix(rows: cols, cols: rows)
        for i in 0..<rows {
            for j in 0..<cols {
                result.data[j * rows + i] = data[i * cols + j]
            }
        }
        return result
    }

    // MARK: - Element-wise operations

    public static func + (lhs: Matrix, rhs: Matrix) -> Matrix {
        precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols, "Shape mismatch for +")
        var result = Matrix(rows: lhs.rows, cols: lhs.cols)
        for i in 0..<lhs.data.count {
            result.data[i] = lhs.data[i] + rhs.data[i]
        }
        return result
    }

    public static func - (lhs: Matrix, rhs: Matrix) -> Matrix {
        precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols, "Shape mismatch for -")
        var result = Matrix(rows: lhs.rows, cols: lhs.cols)
        for i in 0..<lhs.data.count {
            result.data[i] = lhs.data[i] - rhs.data[i]
        }
        return result
    }

    /// Element-wise multiplication (Hadamard product)
    public static func * (lhs: Matrix, rhs: Matrix) -> Matrix {
        precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols, "Shape mismatch for *")
        var result = Matrix(rows: lhs.rows, cols: lhs.cols)
        for i in 0..<lhs.data.count {
            result.data[i] = lhs.data[i] * rhs.data[i]
        }
        return result
    }

    /// Scalar multiplication
    public static func * (lhs: Matrix, rhs: Float) -> Matrix {
        var result = Matrix(rows: lhs.rows, cols: lhs.cols)
        for i in 0..<lhs.data.count {
            result.data[i] = lhs.data[i] * rhs
        }
        return result
    }

    public static func * (lhs: Float, rhs: Matrix) -> Matrix {
        rhs * lhs
    }

    /// Add bias vector (1 x cols) to each row of the matrix.
    /// This is the broadcast add: each row gets the same bias added.
    public func addingBias(_ bias: Matrix) -> Matrix {
        precondition(bias.rows == 1 && bias.cols == cols,
                     "Bias shape (\(bias.rows)x\(bias.cols)) must be (1x\(cols))")
        var result = Matrix(rows: rows, cols: cols)
        for i in 0..<rows {
            for j in 0..<cols {
                result.data[i * cols + j] = data[i * cols + j] + bias.data[j]
            }
        }
        return result
    }

    /// Sum columns: returns a (1 x cols) matrix where each element is the sum of that column.
    /// Used in backprop to compute bias gradients from batch.
    public func sumRows() -> Matrix {
        var result = Matrix(rows: 1, cols: cols)
        for i in 0..<rows {
            for j in 0..<cols {
                result.data[j] += data[i * cols + j]
            }
        }
        return result
    }

    /// Contiguous row slice — single memcpy.
    public func slice(rowStart: Int, rowCount: Int) -> Matrix {
        let start = rowStart * cols
        let end = start + rowCount * cols
        return Matrix(rows: rowCount, cols: cols, data: Array(data[start..<end]))
    }

    /// Apply a function element-wise
    public func map(_ transform: (Float) -> Float) -> Matrix {
        Matrix(rows: rows, cols: cols, data: data.map(transform))
    }
}
