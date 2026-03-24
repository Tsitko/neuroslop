import Metal

/// Owns all Metal infrastructure: device, queue, compiled pipeline states.
/// Created once, reused for all operations.
final class MetalContext: @unchecked Sendable {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary

    // Pipeline states (one per kernel, cached at init)
    let matmulNaive: MTLComputePipelineState
    let matmulTiled: MTLComputePipelineState
    let reluForward: MTLComputePipelineState
    let reluDerivative: MTLComputePipelineState
    let sigmoidForward: MTLComputePipelineState
    let sigmoidDerivative: MTLComputePipelineState
    let tanhForward: MTLComputePipelineState
    let tanhDerivative: MTLComputePipelineState
    let softmaxForward: MTLComputePipelineState
    let elementwiseMultiply: MTLComputePipelineState
    let scalarMultiply: MTLComputePipelineState
    let subtractArrays: MTLComputePipelineState
    let addArrays: MTLComputePipelineState
    let addBias: MTLComputePipelineState
    let sumRows: MTLComputePipelineState
    let rkanForward: MTLComputePipelineState
    let rkanReduce: MTLComputePipelineState
    let rkanBackward: MTLComputePipelineState

    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create Metal command queue")
        }
        self.commandQueue = queue

        // Compile shaders from source at runtime
        do {
            self.library = try device.makeLibrary(source: MetalShaderSource.source, options: nil)
        } catch {
            fatalError("Failed to compile Metal shaders: \(error)")
        }

        // Create pipeline states for all kernels
        let lib = self.library
        let dev = device
        func mp(_ name: String) -> MTLComputePipelineState {
            guard let fn = lib.makeFunction(name: name) else {
                fatalError("Metal function '\(name)' not found")
            }
            do { return try dev.makeComputePipelineState(function: fn) }
            catch { fatalError("Failed to create pipeline for '\(name)': \(error)") }
        }

        self.matmulNaive = mp("matmul_naive")
        self.matmulTiled = mp("matmul_tiled")
        self.reluForward = mp("relu_forward")
        self.reluDerivative = mp("relu_derivative")
        self.sigmoidForward = mp("sigmoid_forward")
        self.sigmoidDerivative = mp("sigmoid_derivative")
        self.tanhForward = mp("tanh_forward")
        self.tanhDerivative = mp("tanh_derivative")
        self.softmaxForward = mp("softmax_forward")
        self.elementwiseMultiply = mp("elementwise_multiply")
        self.scalarMultiply = mp("scalar_multiply")
        self.subtractArrays = mp("subtract_arrays")
        self.addArrays = mp("add_arrays")
        self.addBias = mp("add_bias")
        self.sumRows = mp("sum_rows")
        self.rkanForward = mp("rational_kan_forward")
        self.rkanReduce = mp("rational_kan_reduce")
        self.rkanBackward = mp("rational_kan_backward")
    }

    /// Create a shared MTLBuffer from Float array data.
    func makeBuffer(_ data: [Float]) -> MTLBuffer {
        data.withUnsafeBytes { rawBuf in
            device.makeBuffer(bytes: rawBuf.baseAddress!, length: rawBuf.count, options: .storageModeShared)!
        }
    }

    /// Create an empty shared MTLBuffer of given Float count.
    func makeBuffer(count: Int) -> MTLBuffer {
        device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared)!
    }

    /// Read Float array from MTLBuffer.
    func readBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
