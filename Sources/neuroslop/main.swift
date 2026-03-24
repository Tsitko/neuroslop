import NeuroslopCore
import Foundation
#if canImport(Darwin)
import Darwin
#endif

setbuf(stdout, nil)

let args = CommandLine.arguments
let backendName = args.count > 1 ? args[1] : "optimized"
let trainSubsetSize = args.count > 2 ? Int(args[2]) : nil
let testSubsetSize = args.count > 3 ? Int(args[3]) : nil
let numEpochs = args.count > 4 ? Int(args[4]) ?? 10 : 10

let monitor = MemoryMonitor()
monitor.log("Start")

// Load MNIST
print("Loading MNIST...")
let loader = MNISTLoader()
let (train, test) = try loader.load()
monitor.log("MNIST loaded")

// Subset if requested
let trainX: Matrix
let trainY: Matrix
let testX: Matrix
let testY: Matrix

if let ts = trainSubsetSize {
    trainX = Matrix(rows: ts, cols: train.images.cols,
                    data: Array(train.images.data[0..<ts * train.images.cols]))
    trainY = Matrix(rows: ts, cols: train.labels.cols,
                    data: Array(train.labels.data[0..<ts * train.labels.cols]))
} else {
    trainX = train.images
    trainY = train.labels
}

if let ts = testSubsetSize {
    testX = Matrix(rows: ts, cols: test.images.cols,
                   data: Array(test.images.data[0..<ts * test.images.cols]))
    testY = Matrix(rows: ts, cols: test.labels.cols,
                   data: Array(test.labels.data[0..<ts * test.labels.cols]))
} else {
    testX = test.images
    testY = test.labels
}

print("Train: \(trainX.rows) samples, Test: \(testX.rows) samples, Epochs: \(numEpochs)")
monitor.log("Data ready")

let config = TrainingConfig(
    learningRate: 0.01,
    epochs: numEpochs,
    batchSize: 64,
    loss: .crossEntropy,
    shuffleEachEpoch: true
)

let results: [EpochResult]

switch backendName {
case "optimized", "opt":
    print("Backend: Optimized (pre-allocated buffers + fused ops)")
    let network = OptimizedMLP(
        layerSizes: [784, 128, 64, 10],
        activations: [.relu, .relu, .softmax],
        batchSize: config.batchSize
    )
    monitor.log("Network created")
    print("\nTraining 784->128->64->10...")
    let trainer = OptimizedTrainer(config: config)
    results = trainer.train(
        network: network,
        trainX: trainX, trainY: trainY,
        validX: testX, validY: testY
    )

case "accelerate", "accel":
    print("Backend: Accelerate (cblas_sgemm + vDSP)")
    let backend = AccelerateBackend()
    var network = MLP(
        layerSizes: [784, 128, 64, 10],
        activations: [.relu, .relu, .softmax],
        backend: backend
    )
    monitor.log("Network created")
    print("\nTraining 784->128->64->10...")
    let trainer = Trainer(config: config)
    results = trainer.train(
        network: &network,
        trainX: trainX, trainY: trainY,
        validX: testX, validY: testY
    )

case "naive", "cpu":
    print("Backend: CPU Naive")
    let backend = CPUBackend()
    var network = MLP(
        layerSizes: [784, 128, 64, 10],
        activations: [.relu, .relu, .softmax],
        backend: backend
    )
    monitor.log("Network created")
    print("\nTraining 784->128->64->10...")
    let trainer = Trainer(config: config)
    results = trainer.train(
        network: &network,
        trainX: trainX, trainY: trainY,
        validX: testX, validY: testY
    )

case "metal", "gpu":
    print("Backend: Metal GPU")
    let backend = MetalBackend()
    var network = MLP(
        layerSizes: [784, 128, 64, 10],
        activations: [.relu, .relu, .softmax],
        backend: backend
    )
    monitor.log("Network created")
    print("\nTraining 784->128->64->10...")
    let trainer = Trainer(config: config)
    results = trainer.train(
        network: &network,
        trainX: trainX, trainY: trainY,
        validX: testX, validY: testY
    )

case "cnn":
    print("=== CNN on Fashion-MNIST ===\n")
    let cnnBackend = AccelerateBackend()

    print("Loading Fashion-MNIST...")
    let fmnistLoader = MNISTLoader.fashionMNIST()
    let (fTrain, fTest) = try fmnistLoader.load()
    monitor.log("Fashion-MNIST loaded")

    let fTrainX = trainSubsetSize != nil
        ? Matrix(rows: trainSubsetSize!, cols: 784, data: Array(fTrain.images.data[0..<trainSubsetSize!*784]))
        : fTrain.images
    let fTrainY = trainSubsetSize != nil
        ? Matrix(rows: trainSubsetSize!, cols: 10, data: Array(fTrain.labels.data[0..<trainSubsetSize!*10]))
        : fTrain.labels
    let fTestX = testSubsetSize != nil
        ? Matrix(rows: testSubsetSize!, cols: 784, data: Array(fTest.images.data[0..<testSubsetSize!*784]))
        : fTest.images
    let fTestY = testSubsetSize != nil
        ? Matrix(rows: testSubsetSize!, cols: 10, data: Array(fTest.labels.data[0..<testSubsetSize!*10]))
        : fTest.labels

    print("Train: \(fTrainX.rows), Test: \(fTestX.rows), Epochs: \(numEpochs)")

    // CNN: Conv(1,32,3x3)->ReLU->MaxPool(2)->Conv(32,64,3x3)->ReLU->MaxPool(2)->Flatten->Dense->Softmax
    var cnnNet = HybridNetwork(layers: [
        Conv2DLayer(inChannels: 1, inHeight: 28, inWidth: 28, outChannels: 32, kernelSize: 3, padding: 1, activation: .relu),
        MaxPool2DLayer(channels: 32, inHeight: 28, inWidth: 28, poolSize: 2),
        Conv2DLayer(inChannels: 32, inHeight: 14, inWidth: 14, outChannels: 64, kernelSize: 3, padding: 1, activation: .relu),
        MaxPool2DLayer(channels: 64, inHeight: 14, inWidth: 14, poolSize: 2),
        FlattenLayer(size: 64 * 7 * 7),
        DenseLayer(inputSize: 64 * 7 * 7, outputSize: 128, activation: .relu),
        DenseLayer(inputSize: 128, outputSize: 10, activation: .softmax),
    ] as [any Layer], backend: cnnBackend)
    monitor.log("CNN created")

    let cnnConfig = TrainingConfig(learningRate: 0.001, epochs: numEpochs, batchSize: 64,
                                    loss: .crossEntropy, shuffleEachEpoch: true)
    let cnnTrainer = HybridTrainer(config: cnnConfig, optimizer: .adam())
    var cnnStates: [[AdamState]]? = nil
    print("\nTraining CNN...")
    let cnnResults = cnnTrainer.train(
        network: &cnnNet, trainX: fTrainX, trainY: fTrainY,
        validX: fTestX, validY: fTestY, adamStates: &cnnStates
    )
    if let last = cnnResults.last {
        print("\nFinal: accuracy: \(String(format: "%.2f%%", last.accuracy * 100))")
    }
    results = []

case "cifar", "cifar10":
    print("=== CNN Discovery on CIFAR-10 ===\n")
    let cifarBackend = AccelerateBackend()

    print("Loading CIFAR-10...")
    let cifarLoader = CIFAR10Loader()
    let (cTrain, cTest) = try cifarLoader.load()
    monitor.log("CIFAR-10 loaded")

    let cTrainX = trainSubsetSize != nil
        ? Matrix(rows: trainSubsetSize!, cols: 3072, data: Array(cTrain.images.data[0..<trainSubsetSize!*3072]))
        : cTrain.images
    let cTrainY = trainSubsetSize != nil
        ? Matrix(rows: trainSubsetSize!, cols: 10, data: Array(cTrain.labels.data[0..<trainSubsetSize!*10]))
        : cTrain.labels
    let cTestX = testSubsetSize != nil
        ? Matrix(rows: testSubsetSize!, cols: 3072, data: Array(cTest.images.data[0..<testSubsetSize!*3072]))
        : cTest.images
    let cTestY = testSubsetSize != nil
        ? Matrix(rows: testSubsetSize!, cols: 10, data: Array(cTest.labels.data[0..<testSubsetSize!*10]))
        : cTest.labels

    print("Train: \(cTrainX.rows), Test: \(cTestX.rows), Epochs: \(numEpochs)")

    struct CifarExperiment {
        let name: String
        let layers: [any Layer]
    }

    let cifarExperiments: [CifarExperiment] = [
        CifarExperiment(name: "CNN + BN + Dropout + LR decay", layers: [
            Conv2DLayer(inChannels: 3, inHeight: 32, inWidth: 32, outChannels: 32, kernelSize: 3, padding: 1, activation: .relu),
            BatchNormLayer(channels: 32, height: 32, width: 32),
            MaxPool2DLayer(channels: 32, inHeight: 32, inWidth: 32, poolSize: 2),
            Conv2DLayer(inChannels: 32, inHeight: 16, inWidth: 16, outChannels: 64, kernelSize: 3, padding: 1, activation: .relu),
            BatchNormLayer(channels: 64, height: 16, width: 16),
            MaxPool2DLayer(channels: 64, inHeight: 16, inWidth: 16, poolSize: 2),
            FlattenLayer(size: 64 * 8 * 8),
            DenseLayer(inputSize: 64 * 8 * 8, outputSize: 128, activation: .relu),
            DropoutLayer(size: 128, rate: 0.3),
            DenseLayer(inputSize: 128, outputSize: 10, activation: .softmax),
        ]),
        CifarExperiment(name: "CNN + BN + Dropout + Fourier KAN", layers: [
            Conv2DLayer(inChannels: 3, inHeight: 32, inWidth: 32, outChannels: 32, kernelSize: 3, padding: 1, activation: .relu),
            BatchNormLayer(channels: 32, height: 32, width: 32),
            MaxPool2DLayer(channels: 32, inHeight: 32, inWidth: 32, poolSize: 2),
            Conv2DLayer(inChannels: 32, inHeight: 16, inWidth: 16, outChannels: 64, kernelSize: 3, padding: 1, activation: .relu),
            BatchNormLayer(channels: 64, height: 16, width: 16),
            MaxPool2DLayer(channels: 64, inHeight: 16, inWidth: 16, poolSize: 2),
            FlattenLayer(size: 64 * 8 * 8),
            FourierKANLayer(inputSize: 64 * 8 * 8, outputSize: 128, numFreqs: 5),
            LayerNormLayer(size: 128),
            DropoutLayer(size: 128, rate: 0.3),
            DenseLayer(inputSize: 128, outputSize: 10, activation: .softmax),
        ]),
    ]

    let cifarConfig = TrainingConfig(learningRate: 0.001, epochs: numEpochs, batchSize: 64,
                                      loss: .crossEntropy, shuffleEachEpoch: true)
    let cifarTrainer = HybridTrainer(config: cifarConfig, optimizer: .adam(), lrDecay: true)

    var cifarSummaries: [(String, Float, Float, Double)] = []
    for exp in cifarExperiments {
        print("\n--- \(exp.name) ---")
        var network = HybridNetwork(layers: exp.layers, backend: cifarBackend)
        var adamSt: [[AdamState]]? = nil
        let expResults = cifarTrainer.train(
            network: &network, trainX: cTrainX, trainY: cTrainY,
            validX: cTestX, validY: cTestY, adamStates: &adamSt
        )
        let last = expResults.last!
        cifarSummaries.append((exp.name, last.accuracy, last.loss, last.durationMs))
    }

    print("\n\n=== CIFAR-10 Results ===\n")
    for (name, acc, loss, ms) in cifarSummaries {
        print("  \(name)")
        print("    accuracy: \(String(format: "%.2f%%", acc * 100)), loss: \(String(format: "%.4f", loss)), time: \(String(format: "%.1f", ms))ms/epoch\n")
    }
    results = []

case "discover", "discovery":
    print("=== Discovery Sandbox: comparing architectures on MNIST ===\n")
    let backend = AccelerateBackend()
    let kanConfig = TrainingConfig(learningRate: 0.001, epochs: numEpochs, batchSize: 64,
                                   loss: .crossEntropy, shuffleEachEpoch: true)
    let hybridTrainer = HybridTrainer(config: kanConfig)

    struct Experiment {
        let name: String
        let layers: [any Layer]
    }

    // Architecture: each model = MLP backbone + one KAN layer type + LayerNorm at transitions
    // Phase 1: train 3 models for few epochs, pick best by loss
    // Phase 2: continue training the winner

    let experiments: [Experiment] = [
        Experiment(name: "MLP 784→128→64→10", layers: [
            DenseLayer(inputSize: 784, outputSize: 128, activation: .relu),
            DenseLayer(inputSize: 128, outputSize: 64, activation: .relu),
            DenseLayer(inputSize: 64, outputSize: 10, activation: .softmax),
        ]),
        Experiment(name: "Fourier KAN→LN→Dense→Softmax", layers: [
            FourierKANLayer(inputSize: 784, outputSize: 64, numFreqs: 5),
            LayerNormLayer(size: 64),
            DenseLayer(inputSize: 64, outputSize: 32, activation: .relu),
            DenseLayer(inputSize: 32, outputSize: 10, activation: .softmax),
        ]),
        Experiment(name: "Rational KAN (Metal)→LN→Dense→Softmax", layers: [
            MetalRationalKANLayer(inputSize: 784, outputSize: 64, numDegree: 3, denDegree: 2, batchSize: 64),
            LayerNormLayer(size: 64),
            DenseLayer(inputSize: 64, outputSize: 32, activation: .relu),
            DenseLayer(inputSize: 32, outputSize: 10, activation: .softmax),
        ]),
    ]

    // Phase 1: discovery (few epochs)
    let discoveryEpochs = min(numEpochs, 3)
    let discoveryConfig = TrainingConfig(learningRate: 0.001, epochs: discoveryEpochs, batchSize: 64,
                                          loss: .crossEntropy, shuffleEachEpoch: true)
    let discoveryTrainer = HybridTrainer(config: discoveryConfig, optimizer: .adam())

    var summaries: [(String, Float, Float, Double)] = []
    var bestIdx = 0
    var bestLoss: Float = Float.infinity
    var trainedNetworks: [HybridNetwork] = []
    var trainedAdamStates: [[[AdamState]]?] = []

    print("Phase 1: Discovery (\(discoveryEpochs) epochs each, Adam optimizer)\n")

    for (idx, exp) in experiments.enumerated() {
        print("\n--- \(exp.name) ---")
        var network = HybridNetwork(layers: exp.layers, backend: backend)
        var states: [[AdamState]]? = nil
        let expResults = discoveryTrainer.train(
            network: &network,
            trainX: trainX, trainY: trainY,
            validX: testX, validY: testY,
            adamStates: &states
        )
        let last = expResults.last!
        summaries.append((exp.name, last.accuracy, last.loss, last.durationMs))
        trainedNetworks.append(network)
        trainedAdamStates.append(states)
        if last.loss < bestLoss {
            bestLoss = last.loss
            bestIdx = idx
        }
    }

    print("\n\n=== Phase 1 Results ===\n")
    for (i, (name, acc, loss, ms)) in summaries.enumerated() {
        let marker = i == bestIdx ? " ← BEST" : ""
        print("  \(name)")
        print("    accuracy: \(String(format: "%.2f%%", acc * 100)), loss: \(String(format: "%.4f", loss)), time: \(String(format: "%.1f", ms))ms/epoch\(marker)\n")
    }

    // Phase 2: continue training the winner with same Adam states
    let remainingEpochs = numEpochs - discoveryEpochs
    if remainingEpochs > 0 {
        print("=== Phase 2: Training winner '\(experiments[bestIdx].name)' for \(remainingEpochs) more epochs (Adam) ===\n")
        let phase2Config = TrainingConfig(learningRate: 0.001, epochs: remainingEpochs, batchSize: 64,
                                           loss: .crossEntropy, shuffleEachEpoch: true)
        let phase2Trainer = HybridTrainer(config: phase2Config, optimizer: .adam())
        var winner = trainedNetworks[bestIdx]
        var winnerStates = trainedAdamStates[bestIdx]
        let phase2Results = phase2Trainer.train(
            network: &winner,
            trainX: trainX, trainY: trainY,
            validX: testX, validY: testY,
            adamStates: &winnerStates
        )
        if let last = phase2Results.last {
            print("\nFinal: accuracy: \(String(format: "%.2f%%", last.accuracy * 100)), loss: \(String(format: "%.4f", last.loss))")
        }
    }
    results = []

default:
    fatalError("Unknown backend: \(backendName). Use: optimized, accelerate, metal, naive, discover")
}

if !results.isEmpty {
    monitor.log("Training complete")
    let finalAccuracy = results.last!.accuracy
    print("\nFinal test accuracy: \(String(format: "%.2f%%", finalAccuracy * 100))")
}
