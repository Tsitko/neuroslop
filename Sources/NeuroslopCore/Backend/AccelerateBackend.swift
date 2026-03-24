@preconcurrency import Accelerate

/// Accelerate backend — uses Apple's BLAS (cblas_sgemm) for matmul.
/// NEON SIMD, multithreading, cache-optimal blocking — all handled by Accelerate.
/// This represents "how fast CPU can be" on Apple Silicon.
public struct AccelerateBackend: ComputeBackend, Sendable {

    public init() {}

    public func matmul(_ a: Matrix, _ b: Matrix) -> Matrix {
        precondition(a.cols == b.rows, "Shape mismatch: (\(a.rows)x\(a.cols)) * (\(b.rows)x\(b.cols))")
        var result = Matrix(rows: a.rows, cols: b.cols)
        // C = alpha * A * B + beta * C
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(a.rows),      // M
            Int32(b.cols),      // N
            Int32(a.cols),      // K
            1.0,                // alpha
            a.data, Int32(a.cols),  // A, lda
            b.data, Int32(b.cols),  // B, ldb
            0.0,                // beta
            &result.data, Int32(b.cols)  // C, ldc
        )
        return result
    }

    public func matmulTransposedB(_ a: Matrix, _ b: Matrix) -> Matrix {
        precondition(a.cols == b.cols, "Shape mismatch for A*B^T")
        var result = Matrix(rows: a.rows, cols: b.rows)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            Int32(a.rows),
            Int32(b.rows),      // N = rows of B (cols of B^T)
            Int32(a.cols),
            1.0,
            a.data, Int32(a.cols),
            b.data, Int32(b.cols),  // ldb = cols of B (not B^T)
            0.0,
            &result.data, Int32(b.rows)
        )
        return result
    }

    public func transposedMatmul(_ a: Matrix, _ b: Matrix) -> Matrix {
        precondition(a.rows == b.rows, "Shape mismatch for A^T*B")
        var result = Matrix(rows: a.cols, cols: b.cols)
        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            Int32(a.cols),      // M = cols of A (rows of A^T)
            Int32(b.cols),
            Int32(a.rows),      // K = rows of A
            1.0,
            a.data, Int32(a.cols),  // lda = cols of A (not A^T)
            b.data, Int32(b.cols),
            0.0,
            &result.data, Int32(b.cols)
        )
        return result
    }

    // Element-wise ops use vDSP for vectorized operations
    public func addBias(_ matrix: Matrix, _ bias: Matrix) -> Matrix {
        matrix.addingBias(bias)
    }

    public func activate(_ matrix: Matrix, _ kind: ActivationKind) -> Matrix {
        kind.apply(matrix)
    }

    public func activationDerivative(_ activationOutput: Matrix, _ kind: ActivationKind) -> Matrix {
        kind.derivative(activationOutput: activationOutput)
    }

    public func elementwiseMultiply(_ a: Matrix, _ b: Matrix) -> Matrix {
        precondition(a.count == b.count)
        var result = Matrix(rows: a.rows, cols: a.cols)
        vDSP_vmul(a.data, 1, b.data, 1, &result.data, 1, vDSP_Length(a.count))
        return result
    }

    public func scalarMultiply(_ matrix: Matrix, _ scalar: Float) -> Matrix {
        var result = Matrix(rows: matrix.rows, cols: matrix.cols)
        var s = scalar
        vDSP_vsmul(matrix.data, 1, &s, &result.data, 1, vDSP_Length(matrix.count))
        return result
    }

    public func subtract(_ a: Matrix, _ b: Matrix) -> Matrix {
        precondition(a.count == b.count)
        var result = Matrix(rows: a.rows, cols: a.cols)
        // vDSP_vsub: C = B - A (note: reversed order!)
        vDSP_vsub(b.data, 1, a.data, 1, &result.data, 1, vDSP_Length(a.count))
        return result
    }

    public func add(_ a: Matrix, _ b: Matrix) -> Matrix {
        precondition(a.count == b.count)
        var result = Matrix(rows: a.rows, cols: a.cols)
        vDSP_vadd(a.data, 1, b.data, 1, &result.data, 1, vDSP_Length(a.count))
        return result
    }

    public func sumRows(_ matrix: Matrix) -> Matrix {
        matrix.sumRows()
    }
}
