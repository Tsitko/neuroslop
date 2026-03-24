/// Naive CPU backend — triple-nested loops, no SIMD, no Accelerate.
/// This is the performance baseline. Every optimization gets compared to this.
public struct CPUBackend: ComputeBackend, Sendable {

    public init() {}

    public func matmul(_ a: Matrix, _ b: Matrix) -> Matrix { a.matmul(b) }
    public func matmulTransposedB(_ a: Matrix, _ b: Matrix) -> Matrix { a.matmulTransposedB(b) }
    public func transposedMatmul(_ a: Matrix, _ b: Matrix) -> Matrix { a.transposedMatmul(b) }
    public func addBias(_ matrix: Matrix, _ bias: Matrix) -> Matrix { matrix.addingBias(bias) }
    public func activate(_ matrix: Matrix, _ kind: ActivationKind) -> Matrix { kind.apply(matrix) }
    public func activationDerivative(_ activationOutput: Matrix, _ kind: ActivationKind) -> Matrix { kind.derivative(activationOutput: activationOutput) }
    public func elementwiseMultiply(_ a: Matrix, _ b: Matrix) -> Matrix { a * b }
    public func scalarMultiply(_ matrix: Matrix, _ scalar: Float) -> Matrix { matrix * scalar }
    public func subtract(_ a: Matrix, _ b: Matrix) -> Matrix { a - b }
    public func add(_ a: Matrix, _ b: Matrix) -> Matrix { a + b }
    public func sumRows(_ matrix: Matrix) -> Matrix { matrix.sumRows() }
}
