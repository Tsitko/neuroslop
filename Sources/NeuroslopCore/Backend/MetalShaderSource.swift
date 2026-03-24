/// All Metal Shading Language kernels embedded as a Swift string.
/// Compiled at runtime via device.makeLibrary(source:).
enum MetalShaderSource {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // ============================================================
    // MARK: - Matrix Multiplication
    // ============================================================

    /// Naive matmul: one thread per output element.
    /// C[M x N] = A[M x K] * B[K x N]
    kernel void matmul_naive(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C       [[buffer(2)]],
        constant uint& M      [[buffer(3)]],
        constant uint& N      [[buffer(4)]],
        constant uint& K      [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint row = gid.y;
        uint col = gid.x;
        if (row >= M || col >= N) return;

        float sum = 0.0;
        for (uint i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }

    /// Tiled matmul with threadgroup (shared) memory.
    /// Each threadgroup loads TILE_SIZE x TILE_SIZE tiles of A and B
    /// into fast threadgroup memory, reducing global memory accesses
    /// by a factor of TILE_SIZE.
    constant uint TILE_SIZE = 16;

    kernel void matmul_tiled(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C       [[buffer(2)]],
        constant uint& M      [[buffer(3)]],
        constant uint& N      [[buffer(4)]],
        constant uint& K      [[buffer(5)]],
        uint2 gid  [[thread_position_in_grid]],
        uint2 lid  [[thread_position_in_threadgroup]])
    {
        threadgroup float tileA[TILE_SIZE][TILE_SIZE];
        threadgroup float tileB[TILE_SIZE][TILE_SIZE];

        uint row = gid.y;
        uint col = gid.x;

        float sum = 0.0;
        uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

        for (uint t = 0; t < numTiles; t++) {
            // Load tile of A
            uint aCol = t * TILE_SIZE + lid.x;
            if (row < M && aCol < K) {
                tileA[lid.y][lid.x] = A[row * K + aCol];
            } else {
                tileA[lid.y][lid.x] = 0.0;
            }

            // Load tile of B
            uint bRow = t * TILE_SIZE + lid.y;
            if (bRow < K && col < N) {
                tileB[lid.y][lid.x] = B[bRow * N + col];
            } else {
                tileB[lid.y][lid.x] = 0.0;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate
            for (uint i = 0; i < TILE_SIZE; i++) {
                sum += tileA[lid.y][i] * tileB[i][lid.x];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (row < M && col < N) {
            C[row * N + col] = sum;
        }
    }

    // ============================================================
    // MARK: - Activation Functions (Forward)
    // ============================================================

    kernel void relu_forward(
        device const float* input  [[buffer(0)]],
        device float* output       [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
    {
        output[gid] = max(0.0f, input[gid]);
    }

    kernel void sigmoid_forward(
        device const float* input  [[buffer(0)]],
        device float* output       [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
    {
        output[gid] = 1.0f / (1.0f + exp(-input[gid]));
    }

    kernel void tanh_forward(
        device const float* input  [[buffer(0)]],
        device float* output       [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
    {
        output[gid] = tanh(input[gid]);
    }

    /// Per-row softmax with numerical stability (subtract max).
    /// One threadgroup per row. Each thread handles one column.
    kernel void softmax_forward(
        device const float* input  [[buffer(0)]],
        device float* output       [[buffer(1)]],
        constant uint& cols        [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint row = gid.y;
        uint col = gid.x;
        if (col >= cols) return;

        uint offset = row * cols;

        // Find max (single thread per row scans — fine for small cols like 10)
        float maxVal = input[offset];
        for (uint j = 1; j < cols; j++) {
            maxVal = max(maxVal, input[offset + j]);
        }

        // Compute exp and sum
        float expVal = exp(input[offset + col] - maxVal);

        float sum = 0.0;
        for (uint j = 0; j < cols; j++) {
            sum += exp(input[offset + j] - maxVal);
        }

        output[offset + col] = expVal / sum;
    }

    // ============================================================
    // MARK: - Activation Derivatives
    // ============================================================

    /// ReLU derivative from activation output: out = (a > 0) ? 1 : 0
    kernel void relu_derivative(
        device const float* activation_output [[buffer(0)]],
        device float* output                  [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
    {
        output[gid] = activation_output[gid] > 0.0f ? 1.0f : 0.0f;
    }

    /// Sigmoid derivative from output: out = a * (1 - a)
    kernel void sigmoid_derivative(
        device const float* activation_output [[buffer(0)]],
        device float* output                  [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
    {
        float a = activation_output[gid];
        output[gid] = a * (1.0f - a);
    }

    /// Tanh derivative from output: out = 1 - a^2
    kernel void tanh_derivative(
        device const float* activation_output [[buffer(0)]],
        device float* output                  [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
    {
        float a = activation_output[gid];
        output[gid] = 1.0f - a * a;
    }

    // ============================================================
    // MARK: - Element-wise Operations
    // ============================================================

    kernel void elementwise_multiply(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C       [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = A[gid] * B[gid];
    }

    kernel void scalar_multiply(
        device const float* A    [[buffer(0)]],
        constant float& scalar   [[buffer(1)]],
        device float* C          [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = A[gid] * scalar;
    }

    kernel void subtract_arrays(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C       [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = A[gid] - B[gid];
    }

    kernel void add_arrays(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C       [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = A[gid] + B[gid];
    }

    // ============================================================
    // MARK: - Bias & Reduction
    // ============================================================

    /// Add bias[col] to each row: out[row*cols + col] = matrix[row*cols + col] + bias[col]
    kernel void add_bias(
        device const float* matrix [[buffer(0)]],
        device const float* bias   [[buffer(1)]],
        device float* output       [[buffer(2)]],
        constant uint& cols        [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        output[gid] = matrix[gid] + bias[gid % cols];
    }

    /// Sum across rows: output[col] = sum over all rows of matrix[row*cols + col]
    /// One thread per column, loops over rows.
    kernel void sum_rows(
        device const float* matrix [[buffer(0)]],
        device float* output       [[buffer(1)]],
        constant uint& rows        [[buffer(2)]],
        constant uint& cols        [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        if (gid >= cols) return;
        float sum = 0.0;
        for (uint r = 0; r < rows; r++) {
            sum += matrix[r * cols + gid];
        }
        output[gid] = sum;
    }

    // ============================================================
    // MARK: - Rational KAN Layer
    // ============================================================

    /// Forward: compute P(x)/Q(x) for each edge, one thread per (batch, outIdx, inIdx).
    /// Output: phi[batch * outSize * inSize] — per-edge values before reduction.
    /// P(x) = Σ(k=0..p) a[j*inSize*(p+1) + i*(p+1) + k] * x^k   (Horner)
    /// Q(x) = 1 + Σ(k=0..q-1) b[j*inSize*q + i*q + k]^2 * x^(2(k+1))
    kernel void rational_kan_forward(
        device const float* input      [[buffer(0)]],  // [batch, inSize]
        device const float* numCoeffs  [[buffer(1)]],  // [outSize, inSize*(p+1)]
        device const float* denCoeffs  [[buffer(2)]],  // [outSize, inSize*q]
        device float* phi              [[buffer(3)]],  // [batch, outSize, inSize]
        device float* pVals            [[buffer(4)]],  // [batch, outSize, inSize] — cached P
        device float* qVals            [[buffer(5)]],  // [batch, outSize, inSize] — cached Q
        constant uint& batchSize       [[buffer(6)]],
        constant uint& inSize          [[buffer(7)]],
        constant uint& outSize         [[buffer(8)]],
        constant uint& p               [[buffer(9)]],  // numerator degree
        constant uint& q               [[buffer(10)]],  // denominator degree
        uint gid [[thread_position_in_grid]])
    {
        uint totalEdges = batchSize * outSize * inSize;
        if (gid >= totalEdges) return;

        uint b = gid / (outSize * inSize);
        uint ji = gid % (outSize * inSize);
        uint j = ji / inSize;
        uint i = ji % inSize;

        float x = input[b * inSize + i];
        uint aBase = j * inSize * (p + 1) + i * (p + 1);
        uint bBase = j * inSize * q + i * q;

        // P(x) via Horner
        float pv = numCoeffs[aBase + p];
        for (int k = (int)p - 1; k >= 0; k--) {
            pv = pv * x + numCoeffs[aBase + k];
        }

        // Q(x) = 1 + Σ b^2 * x^(2(k+1))
        float qv = 1.0;
        float x2k = x * x;
        for (uint k = 0; k < q; k++) {
            float bk = denCoeffs[bBase + k];
            qv += bk * bk * x2k;
            x2k *= x * x;
        }

        pVals[gid] = pv;
        qVals[gid] = qv;
        phi[gid] = pv / qv;
    }

    /// Reduce: sum phi across input dimension.
    /// output[batch, j] = Σ(i=0..inSize-1) phi[batch, j, i]
    kernel void rational_kan_reduce(
        device const float* phi   [[buffer(0)]],  // [batch, outSize, inSize]
        device float* output      [[buffer(1)]],  // [batch, outSize]
        constant uint& inSize     [[buffer(2)]],
        constant uint& outSize    [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        // gid = b * outSize + j
        uint b = gid / outSize;
        uint j = gid % outSize;

        float sum = 0.0;
        uint base = b * outSize * inSize + j * inSize;
        for (uint i = 0; i < inSize; i++) {
            sum += phi[base + i];
        }
        output[gid] = sum;
    }

    /// Backward: compute gradients for numCoeffs, denCoeffs, and input.
    /// One thread per (batch, outIdx, inIdx) — same as forward.
    kernel void rational_kan_backward(
        device const float* input        [[buffer(0)]],   // [batch, inSize]
        device const float* numCoeffs    [[buffer(1)]],   // [outSize, inSize*(p+1)]
        device const float* denCoeffs    [[buffer(2)]],   // [outSize, inSize*q]
        device const float* pVals        [[buffer(3)]],   // cached P values
        device const float* qVals        [[buffer(4)]],   // cached Q values
        device const float* outGrad      [[buffer(5)]],   // [batch, outSize]
        device float* numGradOut         [[buffer(6)]],   // [batch, outSize, inSize*(p+1)]
        device float* denGradOut         [[buffer(7)]],   // [batch, outSize, inSize*q]
        device float* inputGradOut       [[buffer(8)]],   // [batch, outSize, inSize] — per-edge input grad
        constant uint& batchSize         [[buffer(9)]],
        constant uint& inSize            [[buffer(10)]],
        constant uint& outSize           [[buffer(11)]],
        constant uint& p                 [[buffer(12)]],
        constant uint& q                 [[buffer(13)]],
        uint gid [[thread_position_in_grid]])
    {
        uint totalEdges = batchSize * outSize * inSize;
        if (gid >= totalEdges) return;

        uint b = gid / (outSize * inSize);
        uint ji = gid % (outSize * inSize);
        uint j = ji / inSize;
        uint i = ji % inSize;

        float x = input[b * inSize + i];
        float pv = pVals[gid];
        float qv = qVals[gid];
        float qInv = 1.0 / qv;
        float qInv2 = qInv * qInv;
        float delta = outGrad[b * outSize + j];

        uint aBase = j * inSize * (p + 1) + i * (p + 1);
        uint bBase = j * inSize * q + i * q;

        // Numerator gradient: d(P/Q)/da_k = x^k / Q
        uint numGradBase = gid * (p + 1);  // per-edge offset in expanded gradient
        // Wait, we need a different layout. Let me use the same layout as coefficients
        // but with batch dimension. Actually for reduction across batch we need:
        // numGradOut[b * outSize * inSize * (p+1) + j * inSize * (p+1) + i * (p+1) + k]
        uint ngBase = b * outSize * inSize * (p + 1) + j * inSize * (p + 1) + i * (p + 1);
        float xk = 1.0;
        for (uint k = 0; k <= p; k++) {
            numGradOut[ngBase + k] = delta * xk * qInv;
            xk *= x;
        }

        // Denominator gradient: d(P/Q)/db_k = -P * 2*b_k * x^(2(k+1)) / Q^2
        uint dgBase = b * outSize * inSize * q + j * inSize * q + i * q;
        float x2k = x * x;
        for (uint k = 0; k < q; k++) {
            float bk = denCoeffs[bBase + k];
            denGradOut[dgBase + k] = delta * (-pv * 2.0 * bk * x2k * qInv2);
            x2k *= x * x;
        }

        // Input gradient: d(P/Q)/dx = (P'*Q - P*Q') / Q^2
        float dPdx = 0.0;
        xk = 1.0;
        for (uint k = 1; k <= p; k++) {
            dPdx += float(k) * numCoeffs[aBase + k] * xk;
            xk *= x;
        }
        float dQdx = 0.0;
        float x2km1 = x;
        for (uint k = 0; k < q; k++) {
            float bk = denCoeffs[bBase + k];
            dQdx += float(2 * (k + 1)) * bk * bk * x2km1;
            x2km1 *= x * x;
        }
        inputGradOut[gid] = delta * (dPdx * qv - pv * dQdx) * qInv2;
    }
    """
}
