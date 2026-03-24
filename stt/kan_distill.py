#!/usr/bin/env python3
"""
Knowledge distillation for KAN adapter.

Idea: encoder features from clean/easy samples = "good" features.
      encoder features from hard/noisy samples = "bad" features.
      Train KAN adapter to transform "bad" → closer to "good".

Approach:
  1. Run encoder on all test samples, collect features + WER
  2. Group: "clean" (WER=0) and "hard" (WER>0)
  3. For each hard sample, find closest clean sample (by text similarity)
  4. Train adapter: minimize MSE(adapter(hard_features), clean_features)
     at matching timesteps

Simpler first approach:
  - Train adapter to minimize reconstruction loss on CLEAN samples (autoencoder-like)
  - The adapter learns the "manifold" of good features
  - Then apply to hard samples — it should project noisy features onto clean manifold
"""
import os, csv, struct, sys
import numpy as np
import onnxruntime as ort

def read_wav_float(path):
    with open(path, 'rb') as f:
        f.read(12)
        sr = 16000
        audio = None
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4: break
            chunk_size = struct.unpack('<I', f.read(4))[0]
            if chunk_id == b'fmt ':
                fmt = f.read(chunk_size)
                sr = struct.unpack('<I', fmt[4:8])[0]
            elif chunk_id == b'data':
                audio = np.frombuffer(f.read(chunk_size), dtype=np.float32)
            else:
                f.read(chunk_size)
    return audio, sr

def compute_fbank(audio, sr=16000, n_mels=80):
    frame_length = int(sr * 0.025)
    frame_shift = int(sr * 0.010)
    n_fft = 512
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    num_frames = max(1, (len(emphasized) - frame_length) // frame_shift + 1)
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * frame_shift
        end = min(start + frame_length, len(emphasized))
        frames[i, :end-start] = emphasized[start:end]
    frames *= np.hamming(frame_length)
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    low_freq, high_freq = 20, sr // 2
    mel_low = 2595 * np.log10(1 + low_freq / 700)
    mel_high = 2595 * np.log10(1 + high_freq / 700)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        for j in range(bins[i], bins[i+1]):
            filterbank[i, j] = (j - bins[i]) / max(1, bins[i+1] - bins[i])
        for j in range(bins[i+1], bins[i+2]):
            filterbank[i, j] = (bins[i+2] - j) / max(1, bins[i+2] - bins[i+1])
    mel_spec = np.maximum(np.dot(spectrum, filterbank.T), 1e-10)
    return np.log(mel_spec).astype(np.float32)

def extract_encoder_features(enc_session, fbank):
    """Run encoder, return features [T_enc, 512]."""
    x = fbank[np.newaxis, :, :]
    x_lens = np.array([fbank.shape[0]], dtype=np.int64)
    enc_out, enc_lens = enc_session.run(None, {'x': x, 'x_lens': x_lens})
    T = int(enc_lens[0])
    return enc_out[0, :T, :]  # [T_enc, 512]

class FourierKANAdapter:
    """Fourier KAN: residual adapter 512→512 with K frequencies per dimension."""
    def __init__(self, dim=512, K=3, lr=0.001):
        self.dim = dim
        self.K = K
        self.lr = lr
        self.basis_size = 2 * K + 1
        # Coefficients: [dim, 2K+1]
        self.coeffs = np.zeros((dim, self.basis_size), dtype=np.float32)
        # Adam state
        self.m = np.zeros_like(self.coeffs)
        self.v = np.zeros_like(self.coeffs)
        self.t = 0

    def forward(self, x):
        """x: [T, 512] → [T, 512]. Residual: output = x + correction."""
        T, D = x.shape
        correction = np.zeros_like(x)
        self.last_basis = np.zeros((T, D, self.basis_size), dtype=np.float32)

        for d in range(D):
            col = x[:, d]
            basis = np.zeros((T, self.basis_size), dtype=np.float32)
            basis[:, 0] = 1.0
            for k in range(1, self.K + 1):
                basis[:, 2*k-1] = np.sin(k * col)
                basis[:, 2*k] = np.cos(k * col)
            self.last_basis[:, d, :] = basis
            correction[:, d] = basis @ self.coeffs[d]

        self.last_x = x
        return x + correction

    def backward_mse(self, target):
        """Compute gradient of MSE(adapter(x), target) w.r.t. coeffs."""
        output = self.last_x + self._compute_correction()
        diff = output - target  # [T, D]
        T = diff.shape[0]

        # dL/d_coeffs[d, k] = (2/T) * sum_t diff[t,d] * basis[t,d,k]
        grad = np.zeros_like(self.coeffs)
        for d in range(self.dim):
            grad[d] = (2.0 / T) * (self.last_basis[:, d, :].T @ diff[:, d])

        return grad

    def _compute_correction(self):
        T = self.last_x.shape[0]
        correction = np.zeros_like(self.last_x)
        for d in range(self.dim):
            correction[:, d] = self.last_basis[:, d, :] @ self.coeffs[d]
        return correction

    def adam_step(self, grad, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        self.m = beta1 * self.m + (1 - beta1) * grad
        self.v = beta2 * self.v + (1 - beta2) * grad ** 2
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        self.coeffs -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

class DenseAdapter:
    """Simple Dense residual adapter for comparison."""
    def __init__(self, dim=512, lr=0.001):
        self.dim = dim
        self.lr = lr
        self.W = np.zeros((dim, dim), dtype=np.float32)  # starts as identity (zero residual)
        self.b = np.zeros(dim, dtype=np.float32)
        self.m_w = np.zeros_like(self.W)
        self.v_w = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.t = 0

    def forward(self, x):
        self.last_x = x
        return x + x @ self.W + self.b  # residual

    def backward_mse(self, target):
        output = self.last_x + self.last_x @ self.W + self.b
        diff = output - target
        T = diff.shape[0]
        grad_W = (2.0 / T) * (self.last_x.T @ diff)
        grad_b = (2.0 / T) * diff.sum(axis=0)
        return grad_W, grad_b

    def adam_step(self, grad_W, grad_b, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        self.m_w = beta1 * self.m_w + (1 - beta1) * grad_W
        self.v_w = beta2 * self.v_w + (1 - beta2) * grad_W ** 2
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * grad_b ** 2
        m_hat_w = self.m_w / (1 - beta1 ** self.t)
        v_hat_w = self.v_w / (1 - beta2 ** self.t)
        m_hat_b = self.m_b / (1 - beta1 ** self.t)
        v_hat_b = self.v_b / (1 - beta2 ** self.t)
        self.W -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
        self.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

def main():
    print("=== KAN Adapter: Knowledge Distillation ===\n")

    enc = ort.InferenceSession("stt/vosk-onnx/encoder.int8.onnx", providers=['CPUExecutionProvider'])

    # Load all test samples
    samples = []
    with open("stt/test/farfield/manifest.tsv", 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            samples.append(row)

    # Extract features for first N samples
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(f"Extracting encoder features for {N} samples...")

    features_list = []
    for i, s in enumerate(samples[:N]):
        wav_path = os.path.join("stt/test/farfield", s['path'])
        if not os.path.exists(wav_path):
            continue
        audio, sr = read_wav_float(wav_path)
        if audio is None:
            continue
        fbank = compute_fbank(audio, sr)
        feats = extract_encoder_features(enc, fbank)
        features_list.append(feats)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N} extracted")

    print(f"Extracted {len(features_list)} feature sets")
    print(f"Feature shapes: {[f.shape for f in features_list[:3]]}")

    # Stack all timesteps into one big matrix for training
    all_features = np.concatenate(features_list, axis=0)  # [total_T, 512]
    print(f"Total feature frames: {all_features.shape[0]}, dim: {all_features.shape[1]}")

    # Train adapter as autoencoder on clean features
    # Goal: adapter learns to reconstruct features → learns the feature manifold
    # Then when applied to noisy/hard features, it projects them onto the learned manifold

    print(f"\n=== Training Fourier KAN Adapter (K=3) ===")
    kan = FourierKANAdapter(dim=512, K=3, lr=0.0001)

    print(f"=== Training Dense Adapter ===")
    dense = DenseAdapter(dim=512, lr=0.0001)

    # Train both on autoencoder task: minimize MSE(adapter(x), x)
    # This seems trivial (identity), but with regularization or bottleneck it learns structure
    # Actually for residual adapter starting at zero, we train to STAY at zero on clean data
    # Then fine-tune on hard samples

    # Better approach: train on clean data to minimize MSE(adapter(x), x)
    # Then the adapter learns to NOT change clean features
    # For hard features, the adapter's deviation from identity = learned correction

    # Even better: pick pairs of (noisy_frame, clean_frame) from similar contexts
    # But we don't have paired data...

    # Simplest viable experiment:
    # 1. Compute mean feature vector from all clean data
    # 2. Train adapter to push features toward mean (denoise)
    # 3. This won't improve things, it'll make everything average...

    # Actually the right approach for unpaired distillation:
    # Train adapter to minimize variance of features while preserving mean
    # = learn a smoother feature representation

    # Or: train on reconstruction with a BOTTLENECK
    # adapter(x) passes through lower dim then back up

    # Let's try the simplest thing that could work:
    # Train KAN to minimize MSE(adapter(x), x) with L2 regularization on coefficients
    # This keeps adapter near identity on training data
    # Then see if features become "cleaner" on hard samples

    n_epochs = 20
    batch_size = 256
    n_frames = all_features.shape[0]

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_frames)
        total_loss_kan = 0
        total_loss_dense = 0
        n_batches = 0

        for start in range(0, n_frames - batch_size, batch_size):
            idx = perm[start:start+batch_size]
            batch = all_features[idx]

            # KAN adapter
            kan_out = kan.forward(batch)
            kan_loss = np.mean((kan_out - batch) ** 2)
            grad = kan.backward_mse(batch)
            # Add L2 regularization to encourage small corrections
            grad += 0.01 * kan.coeffs
            kan.adam_step(grad)
            total_loss_kan += kan_loss

            # Dense adapter
            dense_out = dense.forward(batch)
            dense_loss = np.mean((dense_out - batch) ** 2)
            grad_W, grad_b = dense.backward_mse(batch)
            grad_W += 0.01 * dense.W
            grad_b += 0.01 * dense.b
            dense.adam_step(grad_W, grad_b)
            total_loss_dense += dense_loss

            n_batches += 1

        avg_kan = total_loss_kan / n_batches
        avg_dense = total_loss_dense / n_batches
        kan_norm = np.linalg.norm(kan.coeffs)
        dense_norm = np.linalg.norm(dense.W)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: KAN loss={avg_kan:.6f} (|c|={kan_norm:.4f}), "
                  f"Dense loss={avg_dense:.6f} (|W|={dense_norm:.4f})")

    # Now test: decode with adapters vs baseline
    print(f"\n=== Testing adapters on hard samples ===\n")

    dec = ort.InferenceSession("stt/vosk-onnx/decoder.onnx", providers=['CPUExecutionProvider'])
    join = ort.InferenceSession("stt/vosk-onnx/joiner.onnx", providers=['CPUExecutionProvider'])

    # Load tokens
    id2token = {}
    with open("stt/vosk-onnx/tokens.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                id2token[int(parts[1])] = parts[0]

    def greedy_decode(enc_out):
        T = enc_out.shape[0]
        blank_id = 0
        context_size = 2
        decoder_input = np.array([[blank_id] * context_size], dtype=np.int64)
        decoder_out = dec.run(None, {'y': decoder_input})[0]
        tokens = []
        for t in range(T):
            enc_frame = enc_out[t:t+1, :]
            logit = join.run(None, {'encoder_out': enc_frame, 'decoder_out': decoder_out})[0]
            token_id = int(np.argmax(logit[0]))
            if token_id != blank_id:
                tokens.append(token_id)
                ctx = ([blank_id] * context_size + tokens)[-context_size:]
                decoder_input = np.array([ctx], dtype=np.int64)
                decoder_out = dec.run(None, {'y': decoder_input})[0]
        text = ''.join(id2token.get(t, '') for t in tokens).replace('▁', ' ').strip()
        return text

    def edit_distance(ref, hyp):
        n, m = len(ref), len(hyp)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1): dp[i][0] = i
        for j in range(m+1): dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                dp[i][j] = dp[i-1][j-1] if ref[i-1] == hyp[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[n][m]

    # Test on hard samples
    hard = []
    with open("stt/features/hard_samples.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            hard.append(row)

    n_test = min(50, len(hard))
    base_err, kan_err, dense_err = 0, 0, 0
    total_words = 0

    for i, s in enumerate(hard[:n_test]):
        wav_path = os.path.join("stt/test/farfield", s['path'])
        if not os.path.exists(wav_path): continue
        audio, sr = read_wav_float(wav_path)
        if audio is None: continue

        fbank = compute_fbank(audio, sr)
        feats = extract_encoder_features(enc, fbank)

        gt = s['transcription'].strip().lower()
        ref = gt.split()
        total_words += len(ref)

        text_base = greedy_decode(feats)
        text_kan = greedy_decode(kan.forward(feats))
        text_dense = greedy_decode(dense.forward(feats))

        e_base = edit_distance(ref, text_base.lower().split())
        e_kan = edit_distance(ref, text_kan.lower().split())
        e_dense = edit_distance(ref, text_dense.lower().split())

        base_err += e_base
        kan_err += e_kan
        dense_err += e_dense

        if e_base > 0 and (e_kan != e_base or e_dense != e_base):
            print(f"[{i}] GT:    '{gt}'")
            print(f"     Base:  '{text_base}' (err={e_base})")
            print(f"     KAN:   '{text_kan}' (err={e_kan})")
            print(f"     Dense: '{text_dense}' (err={e_dense})")
            improved = "KAN" if e_kan < e_base else ("Dense" if e_dense < e_base else "none")
            print(f"     → Improved by: {improved}")
            print()

    print(f"\n=== Final WER on {n_test} hard samples ===")
    print(f"Baseline: {base_err/total_words*100:.1f}% ({base_err}/{total_words})")
    print(f"KAN:      {kan_err/total_words*100:.1f}% ({kan_err}/{total_words})")
    print(f"Dense:    {dense_err/total_words*100:.1f}% ({dense_err}/{total_words})")

if __name__ == "__main__":
    main()
