import Foundation

/// Row-major dense matrix of Float32 values.
/// Memory layout is compatible with Metal buffers and Accelerate (cblas_sgemm).
public struct Matrix: Sendable {
    public let rows: Int
    public let cols: Int
    public var data: [Float]

    public init(rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        self.data = [Float](repeating: 0, count: rows * cols)
    }

    public init(rows: Int, cols: Int, data: [Float]) {
        precondition(data.count == rows * cols, "Data size mismatch: expected \(rows * cols), got \(data.count)")
        self.rows = rows
        self.cols = cols
        self.data = data
    }

    /// Create a matrix filled with a constant value.
    public init(rows: Int, cols: Int, fill: Float) {
        self.rows = rows
        self.cols = cols
        self.data = [Float](repeating: fill, count: rows * cols)
    }

    /// Create a matrix with random uniform values in [min, max).
    public init(rows: Int, cols: Int, randomIn range: ClosedRange<Float>) {
        self.rows = rows
        self.cols = cols
        self.data = (0..<rows * cols).map { _ in Float.random(in: range) }
    }

    /// He initialization — good default for ReLU layers.
    /// stddev = sqrt(2 / fanIn)
    public static func heInit(rows: Int, cols: Int, fanIn: Int) -> Matrix {
        let stddev = sqrtf(2.0 / Float(fanIn))
        var matrix = Matrix(rows: rows, cols: cols)
        for i in 0..<matrix.data.count {
            // Box-Muller transform for normal distribution
            let u1 = Float.random(in: 0.0001...1.0)
            let u2 = Float.random(in: 0.0...1.0)
            let normal = sqrtf(-2.0 * logf(u1)) * cosf(2.0 * .pi * u2)
            matrix.data[i] = normal * stddev
        }
        return matrix
    }

    /// Xavier/Glorot initialization — good for Sigmoid/Tanh.
    /// stddev = sqrt(2 / (fanIn + fanOut))
    public static func xavierInit(rows: Int, cols: Int, fanIn: Int, fanOut: Int) -> Matrix {
        let stddev = sqrtf(2.0 / Float(fanIn + fanOut))
        var matrix = Matrix(rows: rows, cols: cols)
        for i in 0..<matrix.data.count {
            let u1 = Float.random(in: 0.0001...1.0)
            let u2 = Float.random(in: 0.0...1.0)
            let normal = sqrtf(-2.0 * logf(u1)) * cosf(2.0 * .pi * u2)
            matrix.data[i] = normal * stddev
        }
        return matrix
    }

    // MARK: - Subscript

    public subscript(row: Int, col: Int) -> Float {
        get { data[row * cols + col] }
        set { data[row * cols + col] = newValue }
    }

    // MARK: - Shape

    public var shape: (Int, Int) { (rows, cols) }
    public var count: Int { rows * cols }
}

// MARK: - Debug

extension Matrix: CustomStringConvertible {
    public var description: String {
        var s = "Matrix(\(rows)x\(cols)):\n"
        for r in 0..<min(rows, 8) {
            let rowValues = (0..<min(cols, 8)).map { c in
                String(format: "%8.4f", self[r, c])
            }
            s += "  [\(rowValues.joined(separator: ", "))]"
            if cols > 8 { s += " ..." }
            s += "\n"
        }
        if rows > 8 { s += "  ...\n" }
        return s
    }
}
