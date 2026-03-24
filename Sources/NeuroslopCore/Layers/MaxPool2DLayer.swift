import Foundation

/// 2D Max Pooling layer. No learnable parameters.
///
/// Input:  Matrix(batch, C * H_in * W_in)
/// Output: Matrix(batch, C * H_out * W_out)
///
/// Takes max value in each poolSize × poolSize window.
/// Caches max indices for backward pass (gradient routing).
public struct MaxPool2DLayer: Sendable {
    public let channels: Int
    public let inHeight: Int
    public let inWidth: Int
    public let poolSize: Int
    public let poolStride: Int

    public let outHeight: Int
    public let outWidth: Int

    // Cached: flat index into input for each output position
    private var maxIndices: [Int]?

    public init(channels: Int, inHeight: Int, inWidth: Int, poolSize: Int = 2, poolStride: Int? = nil) {
        self.channels = channels
        self.inHeight = inHeight
        self.inWidth = inWidth
        self.poolSize = poolSize
        self.poolStride = poolStride ?? poolSize

        self.outHeight = (inHeight - poolSize) / self.poolStride + 1
        self.outWidth = (inWidth - poolSize) / self.poolStride + 1
    }
}

extension MaxPool2DLayer: Layer {
    public var inputSize: Int { channels * inHeight * inWidth }
    public var outputSize: Int { channels * outHeight * outWidth }

    public mutating func forward(_ input: Matrix, backend: ComputeBackend) -> Matrix {
        let bs = input.rows
        let outSpatial = outHeight * outWidth
        let inSpatial = inHeight * inWidth
        var output = Matrix(rows: bs, cols: outputSize)
        var indices = [Int](repeating: 0, count: bs * outputSize)

        for b in 0..<bs {
            for c in 0..<channels {
                for oh in 0..<outHeight {
                    for ow in 0..<outWidth {
                        var maxVal: Float = -.infinity
                        var maxIdx = 0
                        let inBaseC = b * inputSize + c * inSpatial

                        for ph in 0..<poolSize {
                            for pw in 0..<poolSize {
                                let ih = oh * poolStride + ph
                                let iw = ow * poolStride + pw
                                if ih < inHeight && iw < inWidth {
                                    let idx = inBaseC + ih * inWidth + iw
                                    let val = input.data[idx]
                                    if val > maxVal {
                                        maxVal = val
                                        maxIdx = idx
                                    }
                                }
                            }
                        }

                        let outIdx = b * outputSize + c * outSpatial + oh * outWidth + ow
                        output.data[outIdx] = maxVal
                        indices[outIdx] = maxIdx
                    }
                }
            }
        }

        maxIndices = indices
        return output
    }

    public func layerBackward(_ outputGradient: Matrix, backend: ComputeBackend) -> LayerBackwardResult {
        guard let indices = maxIndices else {
            fatalError("MaxPool2DLayer: backward called before forward")
        }
        let bs = outputGradient.rows
        var inputGrad = Matrix(rows: bs, cols: inputSize)

        for i in 0..<(bs * outputSize) {
            inputGrad.data[indices[i]] += outputGradient.data[i]
        }

        return LayerBackwardResult(inputGradient: inputGrad, storage: AnySendable(0))
    }

    public mutating func updateParameters(gradients: LayerBackwardResult, learningRate: Float, backend: ComputeBackend) {}
}
