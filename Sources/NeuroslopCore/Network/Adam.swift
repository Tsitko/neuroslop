import Foundation

/// Adam optimizer state for a single parameter group.
/// Stores first moment (m) and second moment (v) per parameter.
public struct AdamState {
    var m: [Float]  // first moment (momentum)
    var v: [Float]  // second moment (adaptive lr)
    var t: Int      // timestep (for bias correction)

    init(count: Int) {
        self.m = [Float](repeating: 0, count: count)
        self.v = [Float](repeating: 0, count: count)
        self.t = 0
    }

    /// Apply Adam update in-place to parameters.
    /// params -= lr * m̂ / (√v̂ + ε)
    mutating func update(params: inout [Float], grads: [Float],
                         lr: Float, beta1: Float, beta2: Float, eps: Float) {
        t += 1
        let b1corr = 1.0 - powf(beta1, Float(t))
        let b2corr = 1.0 - powf(beta2, Float(t))

        for i in 0..<params.count {
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i] * grads[i]
            let mHat = m[i] / b1corr
            let vHat = v[i] / b2corr
            params[i] -= lr * mHat / (sqrtf(vHat) + eps)
        }
    }
}

/// Adam-aware parameter update for Layer types.
/// Each layer type gets its own AdamState(s) — one per parameter group.
public protocol AdamLayer: Layer {
    /// Number of parameter groups (e.g., Dense has 2: weights + biases)
    var adamStateCount: Int { get }

    /// Initialize Adam states for all parameter groups.
    func createAdamStates() -> [AdamState]

    /// Update parameters using Adam with stored gradients and states.
    mutating func updateWithAdam(gradients: LayerBackwardResult, states: inout [AdamState],
                                  lr: Float, beta1: Float, beta2: Float, eps: Float)
}

// MARK: - DenseLayer Adam

extension DenseLayer: AdamLayer {
    public var adamStateCount: Int { 2 }

    public func createAdamStates() -> [AdamState] {
        [AdamState(count: weights.count), AdamState(count: biases.count)]
    }

    public mutating func updateWithAdam(gradients: LayerBackwardResult, states: inout [AdamState],
                                         lr: Float, beta1: Float, beta2: Float, eps: Float) {
        guard let result = gradients.storage.get(BackwardResult.self) else { return }
        states[0].update(params: &weights.data, grads: result.weightGradient.data, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
        states[1].update(params: &biases.data, grads: result.biasGradient.data, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
    }
}

// MARK: - FourierKANLayer Adam

extension FourierKANLayer: AdamLayer {
    public var adamStateCount: Int { 1 }

    public func createAdamStates() -> [AdamState] {
        [AdamState(count: coeffs.count)]
    }

    public mutating func updateWithAdam(gradients: LayerBackwardResult, states: inout [AdamState],
                                         lr: Float, beta1: Float, beta2: Float, eps: Float) {
        guard let coeffGrad = gradients.storage.get(Matrix.self) else { return }
        states[0].update(params: &coeffs.data, grads: coeffGrad.data, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
    }
}

// MARK: - RationalKANLayer Adam

extension RationalKANLayer: AdamLayer {
    public var adamStateCount: Int { 2 }

    public func createAdamStates() -> [AdamState] {
        [AdamState(count: numCoeffs.count), AdamState(count: denCoeffs.count)]
    }

    public mutating func updateWithAdam(gradients: LayerBackwardResult, states: inout [AdamState],
                                         lr: Float, beta1: Float, beta2: Float, eps: Float) {
        guard let grads = gradients.storage.get(RationalKANGradients.self) else { return }
        states[0].update(params: &numCoeffs.data, grads: grads.numGrad.data, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
        states[1].update(params: &denCoeffs.data, grads: grads.denGrad.data, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
    }
}

// MARK: - LayerNormLayer Adam

extension LayerNormLayer: AdamLayer {
    public var adamStateCount: Int { 2 }

    public func createAdamStates() -> [AdamState] {
        [AdamState(count: gamma.count), AdamState(count: beta.count)]
    }

    public mutating func updateWithAdam(gradients: LayerBackwardResult, states: inout [AdamState],
                                         lr: Float, beta1: Float, beta2: Float, eps: Float) {
        guard let grads = gradients.storage.get(LayerNormGradients.self) else { return }
        states[0].update(params: &gamma, grads: grads.gammaGrad, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
        states[1].update(params: &beta, grads: grads.betaGrad, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
    }
}

// MARK: - MetalRationalKANLayer Adam

extension MetalRationalKANLayer: AdamLayer {
    public var adamStateCount: Int { 2 }

    public func createAdamStates() -> [AdamState] {
        let numCount = outputSize * inputSize * (numDegree + 1)
        let denCount = outputSize * inputSize * denDegree
        return [AdamState(count: numCount), AdamState(count: denCount)]
    }

    public mutating func updateWithAdam(gradients: LayerBackwardResult, states: inout [AdamState],
                                         lr: Float, beta1: Float, beta2: Float, eps: Float) {
        guard let grads = gradients.storage.get(MetalRationalKANGradients.self) else { return }
        // Read current params from GPU, update on CPU, re-upload
        let numCount = outputSize * inputSize * (numDegree + 1)
        let denCount = outputSize * inputSize * denDegree
        var numData = ctx.readBuffer(numCoeffsBuf, count: numCount)
        var denData = ctx.readBuffer(denCoeffsBuf, count: denCount)

        states[0].update(params: &numData, grads: grads.numGrad.data, lr: lr, beta1: beta1, beta2: beta2, eps: eps)
        states[1].update(params: &denData, grads: grads.denGrad.data, lr: lr, beta1: beta1, beta2: beta2, eps: eps)

        numCoeffsBuf = ctx.makeBuffer(numData)
        denCoeffsBuf = ctx.makeBuffer(denData)
    }
}
