import Foundation

/// 2D Convolution layer using im2col + matmul for speed.
///
/// Input:  Matrix(batch, C_in * H_in * W_in)
/// Output: Matrix(batch, C_out * H_out * W_out)
///
/// im2col extracts patches as matrix columns, then matmul with filter weights
/// reduces convolution to cblas_sgemm. This is the standard approach used by
/// most ML frameworks.
public struct Conv2DLayer: Sendable {
    // Shape info
    public let inChannels: Int
    public let inHeight: Int
    public let inWidth: Int
    public let outChannels: Int
    public let kernelSize: Int
    public let stride: Int
    public let padding: Int
    public let activation: ActivationKind

    // Derived
    public let outHeight: Int
    public let outWidth: Int
    private let patchSize: Int  // C_in * kH * kW

    // Parameters
    public var filters: Matrix   // (C_out, C_in * kH * kW)
    public var biases: Matrix    // (1, C_out)

    // Cached for backward
    private var lastIm2col: Matrix?
    private var lastPreActivation: Matrix?
    private var lastOutput: Matrix?

    public init(inChannels: Int, inHeight: Int, inWidth: Int,
                outChannels: Int, kernelSize: Int = 3, stride: Int = 1,
                padding: Int = 1, activation: ActivationKind = .relu) {
        self.inChannels = inChannels
        self.inHeight = inHeight
        self.inWidth = inWidth
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.activation = activation

        self.outHeight = (inHeight + 2 * padding - kernelSize) / stride + 1
        self.outWidth = (inWidth + 2 * padding - kernelSize) / stride + 1
        self.patchSize = inChannels * kernelSize * kernelSize

        // He init
        self.filters = Matrix.heInit(rows: outChannels, cols: patchSize, fanIn: patchSize)
        self.biases = Matrix(rows: 1, cols: outChannels)
    }

    // MARK: - im2col / col2im

    /// Extract patches into column matrix.
    /// Input: Matrix(batch, C_in*H_in*W_in)
    /// Output: Matrix(batch*outH*outW, C_in*kH*kW)
    private func im2col(_ input: Matrix) -> Matrix {
        let bs = input.rows
        let spatialOut = outHeight * outWidth
        let totalRows = bs * spatialOut
        var result = Matrix(rows: totalRows, cols: patchSize)
        let inSpatial = inHeight * inWidth
        let kk = kernelSize * kernelSize

        result.data.withUnsafeMutableBufferPointer { dst in
            input.data.withUnsafeBufferPointer { src in
                for b in 0..<bs {
                    let bInputBase = b * inChannels * inSpatial
                    let bOutBase = b * spatialOut
                    for oh in 0..<outHeight {
                        for ow in 0..<outWidth {
                            let rowOff = (bOutBase + oh * outWidth + ow) * patchSize
                            for c in 0..<inChannels {
                                let cBase = bInputBase + c * inSpatial
                                let colBase = c * kk
                                for kh in 0..<kernelSize {
                                    let ih = oh * stride + kh - padding
                                    if ih < 0 || ih >= inHeight {
                                        // Zero row — already initialized
                                        continue
                                    }
                                    let ihOff = ih * inWidth
                                    let colOff = colBase + kh * kernelSize
                                    let iw0 = ow * stride - padding

                                    // Copy kernelSize elements from input row
                                    // Handle boundary: clip to valid range
                                    let kwStart = max(0, -iw0)
                                    let kwEnd = min(kernelSize, inWidth - iw0)
                                    if kwStart < kwEnd {
                                        let srcOff = cBase + ihOff + iw0 + kwStart
                                        let dstOff = rowOff + colOff + kwStart
                                        let count = kwEnd - kwStart
                                        dst.baseAddress!.advanced(by: dstOff)
                                            .update(from: src.baseAddress!.advanced(by: srcOff), count: count)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return result
    }

    /// Scatter im2col gradients back to input shape.
    /// dIm2col: Matrix(batch*outH*outW, C_in*kH*kW)
    /// Returns: Matrix(batch, C_in*H_in*W_in)
    private func col2im(_ dIm2col: Matrix, batchSize bs: Int) -> Matrix {
        let spatialOut = outHeight * outWidth
        let inputSize = inChannels * inHeight * inWidth
        var result = Matrix(rows: bs, cols: inputSize)

        for b in 0..<bs {
            for oh in 0..<outHeight {
                for ow in 0..<outWidth {
                    let rowIdx = b * spatialOut + oh * outWidth + ow
                    for c in 0..<inChannels {
                        for kh in 0..<kernelSize {
                            for kw in 0..<kernelSize {
                                let ih = oh * stride + kh - padding
                                let iw = ow * stride + kw - padding
                                let colIdx = c * kernelSize * kernelSize + kh * kernelSize + kw

                                if ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth {
                                    let inputIdx = b * inputSize + c * (inHeight * inWidth) + ih * inWidth + iw
                                    result.data[inputIdx] += dIm2col.data[rowIdx * patchSize + colIdx]
                                }
                            }
                        }
                    }
                }
            }
        }
        return result
    }
}

extension Conv2DLayer: Layer {
    public var inputSize: Int { inChannels * inHeight * inWidth }
    public var outputSize: Int { outChannels * outHeight * outWidth }

    public mutating func forward(_ input: Matrix, backend: ComputeBackend) -> Matrix {
        let bs = input.rows
        let spatialOut = outHeight * outWidth

        // Step 1: im2col
        let im2colMat = im2col(input)
        lastIm2col = im2colMat

        // Step 2: matmul — im2col @ filters^T = (batch*outH*outW, C_out)
        let conv = backend.matmulTransposedB(im2colMat, filters)

        // Step 3: add bias (broadcast across spatial positions)
        var biased = conv
        for r in 0..<conv.rows {
            for c in 0..<outChannels {
                biased.data[r * outChannels + c] += biases.data[c]
            }
        }

        // Step 4: activate
        let activated = backend.activate(biased, activation)

        // Step 5: reshape to (batch, C_out * outH * outW)
        // Currently shape is (batch*outH*outW, C_out). Need to rearrange to (batch, C_out*outH*outW)
        // where layout is C_out-major: for each sample, all spatial positions of channel 0, then channel 1, etc.
        var output = Matrix(rows: bs, cols: outputSize)
        for b in 0..<bs {
            for oh in 0..<outHeight {
                for ow in 0..<outWidth {
                    let srcRow = b * spatialOut + oh * outWidth + ow
                    for c in 0..<outChannels {
                        let dstIdx = b * outputSize + c * spatialOut + oh * outWidth + ow
                        output.data[dstIdx] = activated.data[srcRow * outChannels + c]
                    }
                }
            }
        }

        lastPreActivation = biased
        lastOutput = activated
        return output
    }

    public func layerBackward(_ outputGradient: Matrix, backend: ComputeBackend) -> LayerBackwardResult {
        guard let im2colMat = lastIm2col, let preAct = lastPreActivation, let output = lastOutput else {
            fatalError("Conv2DLayer: backward called before forward")
        }
        let bs = outputGradient.rows
        let spatialOut = outHeight * outWidth

        // Step 1: un-reshape output gradient from (batch, C_out*outH*outW) to (batch*outH*outW, C_out)
        var dOutReshaped = Matrix(rows: bs * spatialOut, cols: outChannels)
        for b in 0..<bs {
            for oh in 0..<outHeight {
                for ow in 0..<outWidth {
                    let dstRow = b * spatialOut + oh * outWidth + ow
                    for c in 0..<outChannels {
                        let srcIdx = b * outputSize + c * spatialOut + oh * outWidth + ow
                        dOutReshaped.data[dstRow * outChannels + c] = outputGradient.data[srcIdx]
                    }
                }
            }
        }

        // Step 2: activation derivative
        let dz: Matrix
        if activation == .softmax {
            dz = dOutReshaped
        } else {
            let activDeriv = backend.activationDerivative(output, activation)
            dz = backend.elementwiseMultiply(dOutReshaped, activDeriv)
        }

        // Step 3: filter gradient — dFilters = dz^T @ im2col = (C_out, batch*outH*outW) @ (batch*outH*outW, patchSize)
        let filterGrad = backend.transposedMatmul(dz, im2colMat)

        // Step 4: bias gradient — sum dz across all spatial positions
        let biasGrad = backend.sumRows(dz)

        // Step 5: input gradient — dIm2col = dz @ filters = (batch*outH*outW, C_out) @ (C_out, patchSize)
        let dIm2col = backend.matmul(dz, filters)
        let inputGrad = col2im(dIm2col, batchSize: bs)

        let grads = Conv2DGradients(filterGradient: filterGrad, biasGradient: biasGrad)
        return LayerBackwardResult(inputGradient: inputGrad, storage: AnySendable(grads))
    }

    public mutating func updateParameters(gradients: LayerBackwardResult, learningRate: Float, backend: ComputeBackend) {
        guard let grads = gradients.storage.get(Conv2DGradients.self) else { return }
        filters = backend.subtract(filters, backend.scalarMultiply(grads.filterGradient, learningRate))
        biases = backend.subtract(biases, backend.scalarMultiply(grads.biasGradient, learningRate))
    }
}

// MARK: - Adam

extension Conv2DLayer: AdamLayer {
    public var adamStateCount: Int { 2 }
    public func createAdamStates() -> [AdamState] {
        [AdamState(count: filters.count), AdamState(count: biases.count)]
    }
    public mutating func updateWithAdam(gradients: LayerBackwardResult, states: inout [AdamState],
                                         lr: Float, beta1: Float, beta2: Float, eps: Float) {
        guard let grads = gradients.storage.get(Conv2DGradients.self) else { return }
        states[0].update(params: &filters.data, grads: grads.filterGradient.data, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
        states[1].update(params: &biases.data, grads: grads.biasGradient.data, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
    }
}

struct Conv2DGradients: Sendable {
    let filterGradient: Matrix
    let biasGradient: Matrix
}
