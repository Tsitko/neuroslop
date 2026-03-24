import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Optimized trainer that works with OptimizedMLP.
/// No allocations during training loop — everything pre-allocated.
public struct OptimizedTrainer {
    public let config: TrainingConfig

    public init(config: TrainingConfig) {
        self.config = config
    }

    public func train(
        network: OptimizedMLP,
        trainX: Matrix,
        trainY: Matrix,
        validX: Matrix? = nil,
        validY: Matrix? = nil
    ) -> [EpochResult] {
        let numSamples = trainX.rows
        let inputCols = trainX.cols
        let outputCols = trainY.cols
        var results: [EpochResult] = []

        // Pre-allocate shuffled data
        var shuffledX = [Float](repeating: 0, count: numSamples * inputCols)
        var shuffledY = [Float](repeating: 0, count: numSamples * outputCols)

        // Pre-allocate loss gradient buffer
        let lossGradBuf = UnsafeMutablePointer<Float>.allocate(capacity: config.batchSize * outputCols)
        defer { lossGradBuf.deallocate() }

        for epoch in 0..<config.epochs {
            let clock = ContinuousClock()
            let elapsed = clock.measure {
                // Shuffle
                var indices = Array(0..<numSamples)
                if config.shuffleEachEpoch {
                    indices.shuffle()
                }

                trainX.data.withUnsafeBufferPointer { src in
                    shuffledX.withUnsafeMutableBufferPointer { dst in
                        for (newRow, oldRow) in indices.enumerated() {
                            dst.baseAddress!.advanced(by: newRow * inputCols)
                                .update(from: src.baseAddress!.advanced(by: oldRow * inputCols), count: inputCols)
                        }
                    }
                }
                trainY.data.withUnsafeBufferPointer { src in
                    shuffledY.withUnsafeMutableBufferPointer { dst in
                        for (newRow, oldRow) in indices.enumerated() {
                            dst.baseAddress!.advanced(by: newRow * outputCols)
                                .update(from: src.baseAddress!.advanced(by: oldRow * outputCols), count: outputCols)
                        }
                    }
                }

                // Mini-batch training
                shuffledX.withUnsafeBufferPointer { xBuf in
                    shuffledY.withUnsafeBufferPointer { yBuf in
                        var batchStart = 0
                        while batchStart < numSamples {
                            let batchEnd = min(batchStart + config.batchSize, numSamples)
                            let bs = batchEnd - batchStart

                            let xPtr = xBuf.baseAddress!.advanced(by: batchStart * inputCols)
                            let yPtr = yBuf.baseAddress!.advanced(by: batchStart * outputCols)

                            _ = network.trainStep(
                                batchX: xPtr,
                                batchY: yPtr,
                                actualBatchSize: bs,
                                learningRate: config.learningRate,
                                lossGradBuffer: lossGradBuf
                            )

                            batchStart = batchEnd
                        }
                    }
                }
            }

            // Evaluate
            let evalX = validX ?? trainX
            let evalY = validY ?? trainY
            let predictions = network.predict(evalX)
            let loss = config.loss.loss(predicted: predictions, target: evalY)
            let accuracy = computeAccuracy(predicted: predictions, target: evalY)

            let ms = Double(elapsed.components.seconds) * 1000.0 + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000
            results.append(EpochResult(epoch: epoch, loss: loss, accuracy: accuracy, durationMs: ms))

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
