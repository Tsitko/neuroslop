/// Protocol abstracting compute operations.
/// Implementations: CPUBackend (naive), AccelerateBackend, MetalBackend.
public protocol ComputeBackend: Sendable {
    /// C = A * B
    func matmul(_ a: Matrix, _ b: Matrix) -> Matrix
    /// C = A * B^T
    func matmulTransposedB(_ a: Matrix, _ b: Matrix) -> Matrix
    /// C = A^T * B
    func transposedMatmul(_ a: Matrix, _ b: Matrix) -> Matrix
    /// Add bias (1 x cols) to each row
    func addBias(_ matrix: Matrix, _ bias: Matrix) -> Matrix
    /// Apply activation function
    func activate(_ matrix: Matrix, _ kind: ActivationKind) -> Matrix
    /// Compute activation derivative from activation output
    func activationDerivative(_ activationOutput: Matrix, _ kind: ActivationKind) -> Matrix
    /// Element-wise multiply
    func elementwiseMultiply(_ a: Matrix, _ b: Matrix) -> Matrix
    /// Scalar multiply
    func scalarMultiply(_ matrix: Matrix, _ scalar: Float) -> Matrix
    /// A - B
    func subtract(_ a: Matrix, _ b: Matrix) -> Matrix
    /// A + B
    func add(_ a: Matrix, _ b: Matrix) -> Matrix
    /// Sum across rows → (1 x cols)
    func sumRows(_ matrix: Matrix) -> Matrix
}
