@testable import NeuroslopCore
import Foundation

func check(_ condition: Bool, _ message: String = "", file: String = #file, line: Int = #line) {
    if !condition {
        print("  FAIL [\(file):\(line)] \(message)")
        assertionsFailed += 1
    }
}

func assertEqual<T: Equatable>(_ a: T, _ b: T, _ message: String = "", file: String = #file, line: Int = #line) {
    if a != b {
        print("  FAIL [\(file):\(line)] Expected \(b), got \(a). \(message)")
        assertionsFailed += 1
    }
}

func assertApprox(_ a: Float, _ b: Float, accuracy: Float = 1e-5, _ message: String = "", file: String = #file, line: Int = #line) {
    if abs(a - b) > accuracy {
        print("  FAIL [\(file):\(line)] Expected ≈\(b), got \(a). \(message)")
        assertionsFailed += 1
    }
}

nonisolated(unsafe) var assertionsFailed = 0
nonisolated(unsafe) var testsPassed = 0
nonisolated(unsafe) var testsFailed = 0

func runTest(_ name: String, _ body: () -> Void) {
    assertionsFailed = 0
    body()
    if assertionsFailed == 0 {
        print("  ✓ \(name)")
        testsPassed += 1
    } else {
        print("  ✗ \(name) (\(assertionsFailed) assertions failed)")
        testsFailed += 1
    }
}

func runMatrixTests() {
    print("Matrix Tests:")

    runTest("Creation and subscript") {
        var m = Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        assertEqual(m[0, 0], Float(1))
        assertEqual(m[0, 2], Float(3))
        assertEqual(m[1, 1], Float(5))
        m[1, 2] = 99
        assertEqual(m[1, 2], Float(99))
    }

    runTest("Matmul") {
        let a = Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let b = Matrix(rows: 2, cols: 2, data: [5, 6, 7, 8])
        let c = a.matmul(b)
        assertEqual(c[0, 0], Float(19))
        assertEqual(c[0, 1], Float(22))
        assertEqual(c[1, 0], Float(43))
        assertEqual(c[1, 1], Float(50))
    }

    runTest("Matmul non-square") {
        let a = Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        let b = Matrix(rows: 3, cols: 1, data: [1, 2, 3])
        let c = a.matmul(b)
        assertEqual(c.rows, 2)
        assertEqual(c.cols, 1)
        assertEqual(c[0, 0], Float(14))
        assertEqual(c[1, 0], Float(32))
    }

    runTest("Transpose") {
        let a = Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        let t = a.transposed()
        assertEqual(t.rows, 3)
        assertEqual(t.cols, 2)
        assertEqual(t[0, 0], Float(1))
        assertEqual(t[1, 0], Float(2))
        assertEqual(t[2, 1], Float(6))
    }

    runTest("MatmulTransposedB") {
        let a = Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        let b = Matrix(rows: 2, cols: 3, data: [7, 8, 9, 10, 11, 12])
        let expected = a.matmul(b.transposed())
        let result = a.matmulTransposedB(b)
        assertEqual(result.rows, expected.rows)
        assertEqual(result.cols, expected.cols)
        for i in 0..<result.data.count {
            assertApprox(result.data[i], expected.data[i])
        }
    }

    runTest("TransposedMatmul") {
        let a = Matrix(rows: 3, cols: 2, data: [1, 2, 3, 4, 5, 6])
        let b = Matrix(rows: 3, cols: 2, data: [7, 8, 9, 10, 11, 12])
        let expected = a.transposed().matmul(b)
        let result = a.transposedMatmul(b)
        assertEqual(result.rows, expected.rows)
        assertEqual(result.cols, expected.cols)
        for i in 0..<result.data.count {
            assertApprox(result.data[i], expected.data[i])
        }
    }

    runTest("Element-wise ops") {
        let a = Matrix(rows: 1, cols: 3, data: [1, 2, 3])
        let b = Matrix(rows: 1, cols: 3, data: [4, 5, 6])
        assertEqual((a + b).data, [5, 7, 9])
        assertEqual((b - a).data, [3, 3, 3])
        assertEqual((a * b).data, [4, 10, 18])
        assertEqual((a * 2.0).data, [2, 4, 6])
    }

    runTest("Add bias") {
        let m = Matrix(rows: 3, cols: 2, data: [1, 2, 3, 4, 5, 6])
        let bias = Matrix(rows: 1, cols: 2, data: [10, 20])
        let result = m.addingBias(bias)
        assertEqual(result[0, 0], Float(11))
        assertEqual(result[0, 1], Float(22))
        assertEqual(result[2, 0], Float(15))
        assertEqual(result[2, 1], Float(26))
    }

    runTest("Sum rows") {
        let m = Matrix(rows: 3, cols: 2, data: [1, 2, 3, 4, 5, 6])
        let s = m.sumRows()
        assertEqual(s.rows, 1)
        assertEqual(s.cols, 2)
        assertEqual(s[0, 0], Float(9))
        assertEqual(s[0, 1], Float(12))
    }
}
