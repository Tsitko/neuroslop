import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Trainer for HybridNetwork (heterogeneous layer types).
public enum OptimizerKind: Sendable {
    case sgd
    case adam(beta1: Float = 0.9, beta2: Float = 0.999, eps: Float = 1e-8)
}

public struct HybridTrainer: Sendable {
    public let config: TrainingConfig
    public let optimizer: OptimizerKind
    public let lrDecay: Bool  // cosine annealing

    public init(config: TrainingConfig, optimizer: OptimizerKind = .sgd, lrDecay: Bool = false) {
        self.config = config
        self.optimizer = optimizer
        self.lrDecay = lrDecay
    }

    public func train(
        network: inout HybridNetwork,
        trainX: Matrix,
        trainY: Matrix,
        validX: Matrix? = nil,
        validY: Matrix? = nil,
        adamStates: inout [[AdamState]]?
    ) -> [EpochResult] {
        let numSamples = trainX.rows
        let inputCols = trainX.cols
        let outputCols = trainY.cols
        var results: [EpochResult] = []

        var shuffledX = Matrix(rows: numSamples, cols: inputCols)
        var shuffledY = Matrix(rows: numSamples, cols: outputCols)

        // Init Adam states if needed
        if case .adam = optimizer, adamStates == nil {
            adamStates = network.createAdamStates()
        }

        for epoch in 0..<config.epochs {
            // Cosine annealing LR decay
            let currentLR: Float
            if lrDecay {
                let progress = Float(epoch) / Float(max(config.epochs - 1, 1))
                currentLR = config.learningRate * 0.5 * (1.0 + cosf(Float.pi * progress))
            } else {
                currentLR = config.learningRate
            }

            // Set dropout to training mode
            setDropoutMode(&network, training: true)

            let clock = ContinuousClock()
            let elapsed = clock.measure {
                var indices = Array(0..<numSamples)
                if config.shuffleEachEpoch { indices.shuffle() }

                shuffledX.data.withUnsafeMutableBufferPointer { dst in
                    trainX.data.withUnsafeBufferPointer { src in
                        for (newRow, oldRow) in indices.enumerated() {
                            dst.baseAddress!.advanced(by: newRow * inputCols)
                                .update(from: src.baseAddress!.advanced(by: oldRow * inputCols), count: inputCols)
                        }
                    }
                }
                shuffledY.data.withUnsafeMutableBufferPointer { dst in
                    trainY.data.withUnsafeBufferPointer { src in
                        for (newRow, oldRow) in indices.enumerated() {
                            dst.baseAddress!.advanced(by: newRow * outputCols)
                                .update(from: src.baseAddress!.advanced(by: oldRow * outputCols), count: outputCols)
                        }
                    }
                }

                var batchStart = 0
                while batchStart < numSamples {
                    autoreleasepool {
                        let batchEnd = min(batchStart + config.batchSize, numSamples)
                        let batchRows = batchEnd - batchStart

                        let batchX = shuffledX.slice(rowStart: batchStart, rowCount: batchRows)
                        let batchY = shuffledY.slice(rowStart: batchStart, rowCount: batchRows)

                        let predicted = network.forward(batchX)
                        let lossGrad = config.loss.gradient(predicted: predicted, target: batchY)
                        let gradients = network.backward(lossGrad)

                        switch optimizer {
                        case .sgd:
                            network.updateParameters(gradients: gradients, learningRate: currentLR)
                        case .adam(let beta1, let beta2, let eps):
                            network.updateWithAdam(gradients: gradients, states: &adamStates!,
                                                    lr: currentLR, beta1: beta1, beta2: beta2, eps: eps)
                        }

                        batchStart = batchEnd
                    }
                }
            }

            // Set dropout to eval mode for validation
            setDropoutMode(&network, training: false)

            let evalX = validX ?? trainX
            let evalY = validY ?? trainY
            var evalNet = network
            let predictions = evalNet.forward(evalX)
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

    private func setDropoutMode(_ network: inout HybridNetwork, training: Bool) {
        for i in 0..<network.layers.count {
            if var dropout = network.layers[i] as? DropoutLayer {
                dropout.training = training
                network.layers[i] = dropout
            }
        }
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
