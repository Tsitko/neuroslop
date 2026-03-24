@testable import NeuroslopCore

func runMLPTests() {
    print("\nMLP Tests:")

    runTest("Forward pass shapes") {
        let backend = CPUBackend()
        var network = MLP(
            layerSizes: [4, 8, 3],
            activations: [.relu, .softmax],
            backend: backend
        )
        let input = Matrix(rows: 2, cols: 4, data: [Float](repeating: 1, count: 8))
        let output = network.forward(input)
        assertEqual(output.rows, 2)
        assertEqual(output.cols, 3)
    }

    runTest("XOR convergence") {
        let backend = CPUBackend()

        let x = Matrix(rows: 4, cols: 2, data: [
            0, 0,
            0, 1,
            1, 0,
            1, 1
        ])
        let y = Matrix(rows: 4, cols: 2, data: [
            1, 0,
            0, 1,
            0, 1,
            1, 0,
        ])

        var network = MLP(
            layerSizes: [2, 8, 2],
            activations: [.relu, .softmax],
            backend: backend
        )

        let config = TrainingConfig(
            learningRate: 0.1,
            epochs: 500,
            batchSize: 4,
            loss: .crossEntropy,
            shuffleEachEpoch: false
        )
        let trainer = Trainer(config: config)
        let results = trainer.train(network: &network, trainX: x, trainY: y)

        let lastAccuracy = results.last!.accuracy
        check(lastAccuracy >= 0.99,
            "XOR should converge to 100%, got \(lastAccuracy * 100)%")
    }
}
