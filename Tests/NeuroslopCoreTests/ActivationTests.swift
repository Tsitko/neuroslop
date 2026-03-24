@testable import NeuroslopCore
import Foundation

func runActivationTests() {
    print("\nActivation Tests:")

    runTest("ReLU") {
        let m = Matrix(rows: 1, cols: 4, data: [-2, -0.5, 0, 3])
        let result = ActivationKind.relu.apply(m)
        assertEqual(result.data, [0, 0, 0, 3])
    }

    runTest("ReLU derivative") {
        let output = Matrix(rows: 1, cols: 4, data: [0, 0, 0, 3])
        let d = ActivationKind.relu.derivative(activationOutput: output)
        assertEqual(d.data, [0, 0, 0, 1])
    }

    runTest("Sigmoid range") {
        let m = Matrix(rows: 1, cols: 3, data: [-10, 0, 10])
        let result = ActivationKind.sigmoid.apply(m)
        check(result[0, 0] < 0.001, "sigmoid(-10) should be ≈ 0")
        assertApprox(result[0, 1], 0.5, accuracy: 0.001)
        check(result[0, 2] > 0.999, "sigmoid(10) should be ≈ 1")
    }

    runTest("Softmax sums to 1") {
        let m = Matrix(rows: 2, cols: 3, data: [1, 2, 3, 1, 1, 1])
        let result = ActivationKind.softmax.apply(m)
        for r in 0..<result.rows {
            var sum: Float = 0
            for c in 0..<result.cols {
                sum += result[r, c]
                check(result[r, c] >= 0, "softmax output should be non-negative")
            }
            assertApprox(sum, 1.0)
        }
    }

    runTest("Softmax numerical stability") {
        let m = Matrix(rows: 1, cols: 3, data: [1000, 1001, 1002])
        let result = ActivationKind.softmax.apply(m)
        var sum: Float = 0
        for c in 0..<result.cols {
            check(result[0, c].isFinite, "softmax should be finite")
            sum += result[0, c]
        }
        assertApprox(sum, 1.0)
    }
}
