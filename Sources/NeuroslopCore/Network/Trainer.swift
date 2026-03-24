import Foundation
#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

public struct TrainingConfig: Sendable {
    public let learningRate: Float
    public let epochs: Int
    public let batchSize: Int
    public let loss: LossKind
    public let shuffleEachEpoch: Bool

    public init(learningRate: Float = 0.01, epochs: Int = 10, batchSize: Int = 32,
                loss: LossKind = .crossEntropy, shuffleEachEpoch: Bool = true) {
        self.learningRate = learningRate
        self.epochs = epochs
        self.batchSize = batchSize
        self.loss = loss
        self.shuffleEachEpoch = shuffleEachEpoch
    }
}

public struct EpochResult: Sendable {
    public let epoch: Int
    public let loss: Float
    public let accuracy: Float
    public let durationMs: Double
}

public struct Trainer: Sendable {
    public let config: TrainingConfig

    public init(config: TrainingConfig) {
        self.config = config
    }

    /// Train the network. Returns per-epoch results.
    public func train(
        network: inout MLP,
        trainX: Matrix,   // (numSamples x inputSize)
        trainY: Matrix,   // (numSamples x outputSize), one-hot for classification
        validX: Matrix? = nil,
        validY: Matrix? = nil
    ) -> [EpochResult] {
        let numSamples = trainX.rows
        let inputCols = trainX.cols
        let outputCols = trainY.cols
        var results: [EpochResult] = []

        // Pre-allocate shuffled copies to avoid per-batch extractRows
        var shuffledX = Matrix(rows: numSamples, cols: inputCols)
        var shuffledY = Matrix(rows: numSamples, cols: outputCols)

        for epoch in 0..<config.epochs {
            let clock = ContinuousClock()
            let elapsed = clock.measure {
                // Shuffle: permute indices and copy entire dataset once with memcpy per row
                var indices = Array(0..<numSamples)
                if config.shuffleEachEpoch {
                    indices.shuffle()
                }

                // Copy rows in shuffled order — one memcpy per row
                shuffledX.data.withUnsafeMutableBufferPointer { dst in
                    trainX.data.withUnsafeBufferPointer { src in
                        for (newRow, oldRow) in indices.enumerated() {
                            let dstStart = newRow * inputCols
                            let srcStart = oldRow * inputCols
                            dst.baseAddress!.advanced(by: dstStart)
                                .update(from: src.baseAddress!.advanced(by: srcStart), count: inputCols)
                        }
                    }
                }
                shuffledY.data.withUnsafeMutableBufferPointer { dst in
                    trainY.data.withUnsafeBufferPointer { src in
                        for (newRow, oldRow) in indices.enumerated() {
                            let dstStart = newRow * outputCols
                            let srcStart = oldRow * outputCols
                            dst.baseAddress!.advanced(by: dstStart)
                                .update(from: src.baseAddress!.advanced(by: srcStart), count: outputCols)
                        }
                    }
                }

                // Mini-batch training — contiguous slices, no per-batch copying
                var batchStart = 0
                while batchStart < numSamples {
                    autoreleasepool {
                        let batchEnd = min(batchStart + config.batchSize, numSamples)
                        let batchRows = batchEnd - batchStart

                        let batchX = shuffledX.slice(rowStart: batchStart, rowCount: batchRows)
                        let batchY = shuffledY.slice(rowStart: batchStart, rowCount: batchRows)

                        // Forward
                        let predicted = network.forward(batchX)

                        // Backward
                        let lossGrad = config.loss.gradient(predicted: predicted, target: batchY)
                        let gradients = network.backward(lossGrad)

                        // Update
                        network.updateWeights(gradients: gradients, learningRate: config.learningRate)

                        batchStart = batchEnd
                    }
                }
            }

            // Evaluate on validation set (or train set if no validation)
            let evalX = validX ?? trainX
            let evalY = validY ?? trainY
            var evalNet = network
            let predictions = evalNet.forward(evalX)
            let loss = config.loss.loss(predicted: predictions, target: evalY)
            let accuracy = computeAccuracy(predicted: predictions, target: evalY)

            let ms = Double(elapsed.components.seconds) * 1000.0 + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000
            let result = EpochResult(epoch: epoch, loss: loss, accuracy: accuracy, durationMs: ms)
            results.append(result)

            let mem = MemoryMonitor()
            let rss = MemoryMonitor.formatBytes(mem.currentRSS)
            print("Epoch \(epoch + 1)/\(config.epochs) — loss: \(String(format: "%.4f", loss)), accuracy: \(String(format: "%.2f%%", accuracy * 100)), time: \(String(format: "%.1f", ms))ms, mem: \(rss)")
            fflush(stdout)
        }
        return results
    }

    private func computeAccuracy(predicted: Matrix, target: Matrix) -> Float {
        var correct = 0
        for i in 0..<predicted.rows {
            var predMax = 0, targetMax = 0
            for j in 1..<predicted.cols {
                if predicted[i, j] > predicted[i, predMax] { predMax = j }
                if target[i, j] > target[i, targetMax] { targetMax = j }
            }
            if predMax == targetMax { correct += 1 }
        }
        return Float(correct) / Float(predicted.rows)
    }
}
