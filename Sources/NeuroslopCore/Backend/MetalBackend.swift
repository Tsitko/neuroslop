import Metal

/// Metal GPU backend implementing ComputeBackend.
/// Each operation creates a command buffer, encodes, commits, and waits.
/// For small models this is slower than CPU due to dispatch overhead.
/// For large models the GPU parallelism wins.
public struct MetalBackend: ComputeBackend, @unchecked Sendable {
    private let ctx: MetalContext

    public init() {
        self.ctx = MetalContext()
        print("Metal device: \(ctx.device.name)")
    }

    // MARK: - Helpers

    private func dispatch1D(pipeline: MTLComputePipelineState, buffers: [MTLBuffer], count: Int) {
        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        for (i, buf) in buffers.enumerated() {
            encoder.setBuffer(buf, offset: 0, index: i)
        }
        let threadWidth = min(pipeline.maxTotalThreadsPerThreadgroup, count)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadWidth, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    private func dispatchWithConstants(pipeline: MTLComputePipelineState, buffers: [MTLBuffer],
                                        constants: [UInt32], gridSize: MTLSize, groupSize: MTLSize) {
        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        for (i, buf) in buffers.enumerated() {
            encoder.setBuffer(buf, offset: 0, index: i)
        }
        for (i, val) in constants.enumerated() {
            var v = val
            encoder.setBytes(&v, length: MemoryLayout<UInt32>.stride, index: buffers.count + i)
        }
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    // MARK: - ComputeBackend

    public func matmul(_ a: Matrix, _ b: Matrix) -> Matrix {
        precondition(a.cols == b.rows)
        let M = UInt32(a.rows), N = UInt32(b.cols), K = UInt32(a.cols)
        let bufA = ctx.makeBuffer(a.data)
        let bufB = ctx.makeBuffer(b.data)
        let bufC = ctx.makeBuffer(count: a.rows * b.cols)

        // Use tiled for larger matrices, naive for small
        let useTiled = a.rows >= 16 && b.cols >= 16 && a.cols >= 16
        let pipeline = useTiled ? ctx.matmulTiled : ctx.matmulNaive
        let tileSize = useTiled ? 16 : 1
        let groupSize = useTiled
            ? MTLSize(width: tileSize, height: tileSize, depth: 1)
            : MTLSize(width: min(Int(N), 16), height: min(Int(M), 16), depth: 1)

        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return Matrix(rows: a.rows, cols: b.cols)
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufC, offset: 0, index: 2)
        var m = M, n = N, k = K
        encoder.setBytes(&m, length: 4, index: 3)
        encoder.setBytes(&n, length: 4, index: 4)
        encoder.setBytes(&k, length: 4, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: Int(N), height: Int(M), depth: 1),
            threadsPerThreadgroup: groupSize
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        return Matrix(rows: a.rows, cols: b.cols, data: ctx.readBuffer(bufC, count: a.rows * b.cols))
    }

    public func matmulTransposedB(_ a: Matrix, _ b: Matrix) -> Matrix {
        // A * B^T: transpose B first, then matmul
        // For simplicity, transpose on CPU (small overhead) and use GPU matmul
        let bT = b.transposed()
        return matmul(a, bT)
    }

    public func transposedMatmul(_ a: Matrix, _ b: Matrix) -> Matrix {
        // A^T * B
        let aT = a.transposed()
        return matmul(aT, b)
    }

    public func addBias(_ matrix: Matrix, _ bias: Matrix) -> Matrix {
        precondition(bias.rows == 1 && bias.cols == matrix.cols)
        let bufM = ctx.makeBuffer(matrix.data)
        let bufB = ctx.makeBuffer(bias.data)
        let bufOut = ctx.makeBuffer(count: matrix.count)
        var cols = UInt32(matrix.cols)

        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return matrix
        }
        encoder.setComputePipelineState(ctx.addBias)
        encoder.setBuffer(bufM, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufOut, offset: 0, index: 2)
        encoder.setBytes(&cols, length: 4, index: 3)
        let tw = min(ctx.addBias.maxTotalThreadsPerThreadgroup, matrix.count)
        encoder.dispatchThreads(
            MTLSize(width: matrix.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tw, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        return Matrix(rows: matrix.rows, cols: matrix.cols, data: ctx.readBuffer(bufOut, count: matrix.count))
    }

    public func activate(_ matrix: Matrix, _ kind: ActivationKind) -> Matrix {
        let bufIn = ctx.makeBuffer(matrix.data)
        let bufOut = ctx.makeBuffer(count: matrix.count)

        switch kind {
        case .relu:
            dispatch1D(pipeline: ctx.reluForward, buffers: [bufIn, bufOut], count: matrix.count)
        case .sigmoid:
            dispatch1D(pipeline: ctx.sigmoidForward, buffers: [bufIn, bufOut], count: matrix.count)
        case .tanh:
            dispatch1D(pipeline: ctx.tanhForward, buffers: [bufIn, bufOut], count: matrix.count)
        case .softmax:
            var cols = UInt32(matrix.cols)
            guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
                  let encoder = cmdBuf.makeComputeCommandEncoder() else { return matrix }
            encoder.setComputePipelineState(ctx.softmaxForward)
            encoder.setBuffer(bufIn, offset: 0, index: 0)
            encoder.setBuffer(bufOut, offset: 0, index: 1)
            encoder.setBytes(&cols, length: 4, index: 2)
            encoder.dispatchThreads(
                MTLSize(width: matrix.cols, height: matrix.rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(matrix.cols, 64), height: 1, depth: 1)
            )
            encoder.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }

        return Matrix(rows: matrix.rows, cols: matrix.cols, data: ctx.readBuffer(bufOut, count: matrix.count))
    }

    public func activationDerivative(_ activationOutput: Matrix, _ kind: ActivationKind) -> Matrix {
        let bufIn = ctx.makeBuffer(activationOutput.data)
        let bufOut = ctx.makeBuffer(count: activationOutput.count)

        switch kind {
        case .relu:
            dispatch1D(pipeline: ctx.reluDerivative, buffers: [bufIn, bufOut], count: activationOutput.count)
        case .sigmoid:
            dispatch1D(pipeline: ctx.sigmoidDerivative, buffers: [bufIn, bufOut], count: activationOutput.count)
        case .tanh:
            dispatch1D(pipeline: ctx.tanhDerivative, buffers: [bufIn, bufOut], count: activationOutput.count)
        case .softmax:
            // Softmax derivative handled by combined softmax+CE gradient
            return Matrix(rows: activationOutput.rows, cols: activationOutput.cols, fill: 1.0)
        }

        return Matrix(rows: activationOutput.rows, cols: activationOutput.cols,
                      data: ctx.readBuffer(bufOut, count: activationOutput.count))
    }

    public func elementwiseMultiply(_ a: Matrix, _ b: Matrix) -> Matrix {
        precondition(a.count == b.count)
        let bufA = ctx.makeBuffer(a.data)
        let bufB = ctx.makeBuffer(b.data)
        let bufC = ctx.makeBuffer(count: a.count)
        dispatch1D(pipeline: ctx.elementwiseMultiply, buffers: [bufA, bufB, bufC], count: a.count)
        return Matrix(rows: a.rows, cols: a.cols, data: ctx.readBuffer(bufC, count: a.count))
    }

    public func scalarMultiply(_ matrix: Matrix, _ scalar: Float) -> Matrix {
        let bufA = ctx.makeBuffer(matrix.data)
        let bufC = ctx.makeBuffer(count: matrix.count)
        var s = scalar

        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return matrix }
        encoder.setComputePipelineState(ctx.scalarMultiply)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBytes(&s, length: MemoryLayout<Float>.stride, index: 1)
        encoder.setBuffer(bufC, offset: 0, index: 2)
        let tw = min(ctx.scalarMultiply.maxTotalThreadsPerThreadgroup, matrix.count)
        encoder.dispatchThreads(
            MTLSize(width: matrix.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tw, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        return Matrix(rows: matrix.rows, cols: matrix.cols, data: ctx.readBuffer(bufC, count: matrix.count))
    }

    public func subtract(_ a: Matrix, _ b: Matrix) -> Matrix {
        precondition(a.count == b.count)
        let bufA = ctx.makeBuffer(a.data)
        let bufB = ctx.makeBuffer(b.data)
        let bufC = ctx.makeBuffer(count: a.count)
        dispatch1D(pipeline: ctx.subtractArrays, buffers: [bufA, bufB, bufC], count: a.count)
        return Matrix(rows: a.rows, cols: a.cols, data: ctx.readBuffer(bufC, count: a.count))
    }

    public func add(_ a: Matrix, _ b: Matrix) -> Matrix {
        precondition(a.count == b.count)
        let bufA = ctx.makeBuffer(a.data)
        let bufB = ctx.makeBuffer(b.data)
        let bufC = ctx.makeBuffer(count: a.count)
        dispatch1D(pipeline: ctx.addArrays, buffers: [bufA, bufB, bufC], count: a.count)
        return Matrix(rows: a.rows, cols: a.cols, data: ctx.readBuffer(bufC, count: a.count))
    }

    public func sumRows(_ matrix: Matrix) -> Matrix {
        let bufM = ctx.makeBuffer(matrix.data)
        let bufOut = ctx.makeBuffer(count: matrix.cols)
        var rows = UInt32(matrix.rows)
        var cols = UInt32(matrix.cols)

        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return matrix.sumRows()
        }
        encoder.setComputePipelineState(ctx.sumRows)
        encoder.setBuffer(bufM, offset: 0, index: 0)
        encoder.setBuffer(bufOut, offset: 0, index: 1)
        encoder.setBytes(&rows, length: 4, index: 2)
        encoder.setBytes(&cols, length: 4, index: 3)
        let tw = min(ctx.sumRows.maxTotalThreadsPerThreadgroup, matrix.cols)
        encoder.dispatchThreads(
            MTLSize(width: matrix.cols, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tw, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        return Matrix(rows: 1, cols: matrix.cols, data: ctx.readBuffer(bufOut, count: matrix.cols))
    }
}
