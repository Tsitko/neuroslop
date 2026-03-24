@testable import NeuroslopCore
import Foundation

/// Load a matrix saved by export_trace.py (little-endian: i32 rows, i32 cols, float32[])
func loadTraceMatrix(_ path: String) -> Matrix {
    let url = URL(fileURLWithPath: path)
    let data = try! Data(contentsOf: url)
    let rows = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: Int32.self) }
    let cols = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: Int32.self) }
    let count = Int(rows) * Int(cols)
    let floats = data.withUnsafeBytes { ptr -> [Float] in
        let start = ptr.baseAddress!.advanced(by: 8).assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: start, count: count))
    }
    return Matrix(rows: Int(rows), cols: Int(cols), data: floats)
}

func compareMatrices(_ name: String, _ ours: Matrix, _ theirs: Matrix, tolerance: Float = 1e-5) {
    guard ours.rows == theirs.rows && ours.cols == theirs.cols else {
        print("  ✗ \(name): shape mismatch (\(ours.rows)x\(ours.cols)) vs (\(theirs.rows)x\(theirs.cols))")
        testsFailed += 1
        return
    }
    var maxDiff: Float = 0
    var maxDiffIdx = 0
    for i in 0..<ours.data.count {
        let diff = abs(ours.data[i] - theirs.data[i])
        if diff > maxDiff {
            maxDiff = diff
            maxDiffIdx = i
        }
    }
    if maxDiff <= tolerance {
        print("  ✓ \(name): max diff = \(String(format: "%.2e", maxDiff)) (\(ours.rows)x\(ours.cols))")
        testsPassed += 1
    } else {
        let row = maxDiffIdx / ours.cols
        let col = maxDiffIdx % ours.cols
        print("  ✗ \(name): max diff = \(String(format: "%.2e", maxDiff)) at [\(row),\(col)]" +
              " (ours=\(ours.data[maxDiffIdx]), theirs=\(theirs.data[maxDiffIdx]))")
        testsFailed += 1
    }
}

func runTraceCompareTests() {
    let dir = "Reference/trace"
    let backend = AccelerateBackend()

    print("\nTrace Comparison (vs PyTorch):")

    // Load PyTorch initial weights
    let w0 = loadTraceMatrix("\(dir)/w0_init.bin")
    let b0 = loadTraceMatrix("\(dir)/b0_init.bin")
    let w1 = loadTraceMatrix("\(dir)/w1_init.bin")
    let b1 = loadTraceMatrix("\(dir)/b1_init.bin")
    let w2 = loadTraceMatrix("\(dir)/w2_init.bin")
    let b2 = loadTraceMatrix("\(dir)/b2_init.bin")

    // Load input
    let batchX = loadTraceMatrix("\(dir)/batch_x.bin")
    let batchY = loadTraceMatrix("\(dir)/batch_y_onehot.bin")

    // Build network with PyTorch's weights
    var layer0 = DenseLayer(inputSize: 784, outputSize: 128, activation: .relu)
    layer0.weights = w0
    layer0.biases = b0

    var layer1 = DenseLayer(inputSize: 128, outputSize: 64, activation: .relu)
    layer1.weights = w1
    layer1.biases = b1

    var layer2 = DenseLayer(inputSize: 64, outputSize: 10, activation: .softmax)
    layer2.weights = w2
    layer2.biases = b2

    // Forward pass — step by step
    let a0_ours = layer0.forward(batchX, backend: backend)
    let a0_theirs = loadTraceMatrix("\(dir)/a0.bin")

    // Check pre-activation too
    let z0_theirs = loadTraceMatrix("\(dir)/z0.bin")
    if let z0 = layer0.lastPreActivation {
        compareMatrices("z0 (pre-activation layer 0)", z0, z0_theirs)
    }
    compareMatrices("a0 (output layer 0)", a0_ours, a0_theirs)

    let a1_ours = layer1.forward(a0_ours, backend: backend)
    let a1_theirs = loadTraceMatrix("\(dir)/a1.bin")
    compareMatrices("a1 (output layer 1)", a1_ours, a1_theirs)

    let a2_ours = layer2.forward(a1_ours, backend: backend)
    let softmax_theirs = loadTraceMatrix("\(dir)/softmax_out.bin")
    compareMatrices("softmax output", a2_ours, softmax_theirs)

    // Check logits (pre-softmax)
    let z2_theirs = loadTraceMatrix("\(dir)/z2_logits.bin")
    if let z2 = layer2.lastPreActivation {
        compareMatrices("z2 (logits, pre-softmax)", z2, z2_theirs)
    }

    // Loss
    let ourLoss = LossKind.crossEntropy.loss(predicted: a2_ours, target: batchY)
    print("  Loss: ours = \(String(format: "%.6f", ourLoss)), PyTorch = 2.304600")
    if abs(ourLoss - 2.304600) < 0.001 {
        print("  ✓ Loss matches")
        testsPassed += 1
    } else {
        print("  ✗ Loss mismatch!")
        testsFailed += 1
    }

    // Backward pass
    let lossGrad = LossKind.crossEntropy.gradient(predicted: a2_ours, target: batchY)

    // Compare dL/dz2 (combined softmax+CE gradient)
    let dz2_theirs = loadTraceMatrix("\(dir)/dz2.bin")
    compareMatrices("dL/dz2 (softmax+CE gradient)", lossGrad, dz2_theirs)

    // Backprop through layers
    let result2 = layer2.backward(lossGrad, backend: backend)
    let result1 = layer1.backward(result2.inputGradient, backend: backend)
    let result0 = layer0.backward(result1.inputGradient, backend: backend)

    // Compare weight gradients
    let w2_grad_theirs = loadTraceMatrix("\(dir)/w2_grad.bin")
    let b2_grad_theirs = loadTraceMatrix("\(dir)/b2_grad.bin")
    compareMatrices("dW2 (weight grad layer 2)", result2.weightGradient, w2_grad_theirs)
    compareMatrices("db2 (bias grad layer 2)", result2.biasGradient, b2_grad_theirs)

    let w1_grad_theirs = loadTraceMatrix("\(dir)/w1_grad.bin")
    let b1_grad_theirs = loadTraceMatrix("\(dir)/b1_grad.bin")
    compareMatrices("dW1 (weight grad layer 1)", result1.weightGradient, w1_grad_theirs)
    compareMatrices("db1 (bias grad layer 1)", result1.biasGradient, b1_grad_theirs)

    let w0_grad_theirs = loadTraceMatrix("\(dir)/w0_grad.bin")
    let b0_grad_theirs = loadTraceMatrix("\(dir)/b0_grad.bin")
    compareMatrices("dW0 (weight grad layer 0)", result0.weightGradient, w0_grad_theirs)
    compareMatrices("db0 (bias grad layer 0)", result0.biasGradient, b0_grad_theirs)

    // SGD step and compare weights after
    let lr: Float = 0.01
    layer0.weights = backend.subtract(layer0.weights, backend.scalarMultiply(result0.weightGradient, lr))
    layer0.biases = backend.subtract(layer0.biases, backend.scalarMultiply(result0.biasGradient, lr))
    layer1.weights = backend.subtract(layer1.weights, backend.scalarMultiply(result1.weightGradient, lr))
    layer1.biases = backend.subtract(layer1.biases, backend.scalarMultiply(result1.biasGradient, lr))
    layer2.weights = backend.subtract(layer2.weights, backend.scalarMultiply(result2.weightGradient, lr))
    layer2.biases = backend.subtract(layer2.biases, backend.scalarMultiply(result2.biasGradient, lr))

    let w0_after = loadTraceMatrix("\(dir)/w0_after_step.bin")
    let w1_after = loadTraceMatrix("\(dir)/w1_after_step.bin")
    let w2_after = loadTraceMatrix("\(dir)/w2_after_step.bin")
    compareMatrices("w0 after SGD step", layer0.weights, w0_after)
    compareMatrices("w1 after SGD step", layer1.weights, w1_after)
    compareMatrices("w2 after SGD step", layer2.weights, w2_after)
}
