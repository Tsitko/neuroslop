import NeuroslopCore
import Foundation
#if canImport(Darwin)
import Darwin
#endif

setbuf(stdout, nil)

let monitor = MemoryMonitor()

func ms(_ elapsed: Duration) -> Double {
    Double(elapsed.components.seconds) * 1000.0 + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000
}

print("=== Profiling Accelerate Backend: 1 epoch MNIST ===\n")

let loader = MNISTLoader()
let (train, test) = try loader.load()
monitor.log("Data loaded")

let backend = AccelerateBackend()
var network = MLP(
    layerSizes: [784, 128, 64, 10],
    activations: [.relu, .relu, .softmax],
    backend: backend
)

let batchSize = 64
let numSamples = train.images.rows
let inputCols = train.images.cols
let outputCols = train.labels.cols
let numBatches = (numSamples + batchSize - 1) / batchSize
print("Samples: \(numSamples), Batch size: \(batchSize), Batches: \(numBatches)\n")

let clock = ContinuousClock()

// Pre-shuffle
var shuffledX = Matrix(rows: numSamples, cols: inputCols)
var shuffledY = Matrix(rows: numSamples, cols: outputCols)
var indices = Array(0..<numSamples)
indices.shuffle()

let tShuffle = clock.measure {
    shuffledX.data.withUnsafeMutableBufferPointer { dst in
        train.images.data.withUnsafeBufferPointer { src in
            for (newRow, oldRow) in indices.enumerated() {
                dst.baseAddress!.advanced(by: newRow * inputCols)
                    .update(from: src.baseAddress!.advanced(by: oldRow * inputCols), count: inputCols)
            }
        }
    }
    shuffledY.data.withUnsafeMutableBufferPointer { dst in
        train.labels.data.withUnsafeBufferPointer { src in
            for (newRow, oldRow) in indices.enumerated() {
                dst.baseAddress!.advanced(by: newRow * outputCols)
                    .update(from: src.baseAddress!.advanced(by: oldRow * outputCols), count: outputCols)
            }
        }
    }
}
print("Shuffle+copy: \(String(format: "%.2f", ms(tShuffle)))ms")

// Profile each operation
var tSlice = 0.0
var tForward = 0.0
var tLossGrad = 0.0
var tBackward = 0.0
var tUpdate = 0.0

var batchStart = 0
while batchStart < numSamples {
    let batchEnd = min(batchStart + batchSize, numSamples)
    let batchRows = batchEnd - batchStart

    let t0 = clock.measure {
        _ = shuffledX.slice(rowStart: batchStart, rowCount: batchRows)
        _ = shuffledY.slice(rowStart: batchStart, rowCount: batchRows)
    }
    tSlice += ms(t0)

    let batchX = shuffledX.slice(rowStart: batchStart, rowCount: batchRows)
    let batchY = shuffledY.slice(rowStart: batchStart, rowCount: batchRows)

    let t1 = clock.measure { _ = network.forward(batchX) }
    tForward += ms(t1)
    let predicted = network.forward(batchX)

    let t2 = clock.measure { _ = LossKind.crossEntropy.gradient(predicted: predicted, target: batchY) }
    tLossGrad += ms(t2)
    let lossGrad = LossKind.crossEntropy.gradient(predicted: predicted, target: batchY)

    let t3 = clock.measure { _ = network.backward(lossGrad) }
    tBackward += ms(t3)
    let gradients = network.backward(lossGrad)

    let t4 = clock.measure { network.updateWeights(gradients: gradients, learningRate: 0.01) }
    tUpdate += ms(t4)

    batchStart = batchEnd
}

let total = tSlice + tForward + tLossGrad + tBackward + tUpdate
print("\nPer-epoch breakdown (\(numBatches) batches):")
print("  slice:          \(String(format: "%8.1f", tSlice))ms  (\(String(format: "%.1f", tSlice / total * 100))%)")
print("  forward:        \(String(format: "%8.1f", tForward))ms  (\(String(format: "%.1f", tForward / total * 100))%)")
print("  loss gradient:  \(String(format: "%8.1f", tLossGrad))ms  (\(String(format: "%.1f", tLossGrad / total * 100))%)")
print("  backward:       \(String(format: "%8.1f", tBackward))ms  (\(String(format: "%.1f", tBackward / total * 100))%)")
print("  update weights: \(String(format: "%8.1f", tUpdate))ms  (\(String(format: "%.1f", tUpdate / total * 100))%)")
print("  TOTAL:          \(String(format: "%8.1f", total))ms")
print("  Target:            374ms (PyTorch CPU)")

// Eval
var evalNet = network
let tEval = clock.measure { _ = evalNet.forward(test.images) }
print("\n  eval forward (10k): \(String(format: "%.1f", ms(tEval)))ms")

monitor.log("Profile done")

// Profile individual matmul sizes used in forward pass
print("\n=== Matmul sizes used in MLP (Accelerate) ===")
let sizes: [(String, Int, Int, Int)] = [
    ("input*w0", batchSize, 784, 128),    // 64x784 * 784x128
    ("input*w1", batchSize, 128, 64),     // 64x128 * 128x64
    ("input*w2", batchSize, 64, 10),      // 64x64 * 64x10
    ("a0^T*dz0", 784, batchSize, 128),    // 784x64 * 64x128 (transposed matmul in backward)
    ("a1^T*dz1", 128, batchSize, 64),
    ("dz1*w1^T", batchSize, 64, 128),     // 64x64 * 64x128 (matmulTransposedB)
    ("dz0*w0^T", batchSize, 128, 784),
]
for (name, m, k, n) in sizes {
    let a = Matrix(rows: m, cols: k, randomIn: -1...1)
    let b = Matrix(rows: k, cols: n, randomIn: -1...1)
    _ = backend.matmul(a, b) // warmup
    var times: [Double] = []
    for _ in 0..<100 {
        let t = clock.measure { _ = backend.matmul(a, b) }
        times.append(ms(t))
    }
    times.sort()
    print("  \(name) (\(m)x\(k) * \(k)x\(n)): median \(String(format: "%.3f", times[50]))ms")
}
