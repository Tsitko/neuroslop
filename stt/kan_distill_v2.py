#!/usr/bin/env python3
"""
KAN adapter v2: CTC-based training.

Instead of unsupervised distillation, train adapter + linear head with CTC loss.
This gives a real gradient signal — "these features should predict these tokens."

Pipeline:
  Frozen encoder → features[T,512] → KAN adapter[512→512] → Linear[512→vocab] → CTC loss

Compare: no adapter (just linear head) vs Dense adapter vs Fourier KAN adapter
"""
import os, csv, struct, sys, math
import numpy as np
import onnxruntime as ort

def read_wav_float(path):
    with open(path, 'rb') as f:
        f.read(12)
        sr = 16000; audio = None
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4: break
            chunk_size = struct.unpack('<I', f.read(4))[0]
            if chunk_id == b'fmt ':
                fmt = f.read(chunk_size); sr = struct.unpack('<I', fmt[4:8])[0]
            elif chunk_id == b'data':
                audio = np.frombuffer(f.read(chunk_size), dtype=np.float32)
            else: f.read(chunk_size)
    return audio, sr

def compute_fbank(audio, sr=16000, n_mels=80):
    frame_length = int(sr * 0.025); frame_shift = int(sr * 0.010); n_fft = 512
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    num_frames = max(1, (len(emphasized) - frame_length) // frame_shift + 1)
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        s = i * frame_shift; e = min(s + frame_length, len(emphasized))
        frames[i, :e-s] = emphasized[s:e]
    frames *= np.hamming(frame_length)
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    mel_low = 2595 * np.log10(1 + 20 / 700); mel_high = 2595 * np.log10(1 + sr/2 / 700)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        for j in range(bins[i], bins[i+1]): fb[i,j] = (j-bins[i]) / max(1, bins[i+1]-bins[i])
        for j in range(bins[i+1], bins[i+2]): fb[i,j] = (bins[i+2]-j) / max(1, bins[i+2]-bins[i+1])
    return np.log(np.maximum(np.dot(spectrum, fb.T), 1e-10)).astype(np.float32)

def load_tokens(path):
    t2id = {}; id2t = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                t2id[parts[0]] = int(parts[1])
                id2t[int(parts[1])] = parts[0]
    return t2id, id2t

def text_to_token_ids(text, token2id):
    """Convert text to token IDs using greedy character/BPE matching."""
    text = text.lower().strip()
    # Simple: try to match sentencepiece-style tokens
    tokens = []
    # Add word boundary marker
    text = '▁' + text.replace(' ', '▁')
    i = 0
    while i < len(text):
        # Try longest match
        best_len = 0
        best_id = None
        for length in range(min(10, len(text) - i), 0, -1):
            substr = text[i:i+length]
            if substr in token2id:
                best_len = length
                best_id = token2id[substr]
                break
        if best_id is not None:
            tokens.append(best_id)
            i += best_len
        else:
            i += 1  # skip unknown char
    return tokens

def ctc_loss_and_grad(logits, targets, blank=0):
    """
    Simple CTC forward-backward.
    logits: [T, V] (log probabilities)
    targets: [S] (token IDs, no blanks)
    Returns: loss (scalar), grad [T, V]
    """
    T, V = logits.shape
    S = len(targets)

    # Expand targets with blanks: [b, t1, b, t2, b, ...]
    L = 2 * S + 1
    labels = [blank] * L
    for i, t in enumerate(targets):
        labels[2*i+1] = t

    # Log-space forward
    LOG_0 = -1e30
    alpha = np.full((T, L), LOG_0)
    alpha[0, 0] = logits[0, blank]
    if L > 1:
        alpha[0, 1] = logits[0, labels[1]]

    for t in range(1, T):
        for s in range(L):
            a = alpha[t-1, s]
            if s > 0:
                a = np.logaddexp(a, alpha[t-1, s-1])
            if s > 1 and labels[s] != blank and labels[s] != labels[s-2]:
                a = np.logaddexp(a, alpha[t-1, s-2])
            alpha[t, s] = a + logits[t, labels[s]]

    # Loss = -log P(targets | logits)
    loss = -np.logaddexp(alpha[T-1, L-1], alpha[T-1, L-2]) if L > 1 else -alpha[T-1, 0]

    # Backward
    beta = np.full((T, L), LOG_0)
    beta[T-1, L-1] = 0.0
    if L > 1:
        beta[T-1, L-2] = 0.0

    for t in range(T-2, -1, -1):
        for s in range(L):
            b = beta[t+1, s] + logits[t+1, labels[s]]
            if s < L-1:
                b = np.logaddexp(b, beta[t+1, s+1] + logits[t+1, labels[s+1]])
            if s < L-2 and labels[s] != blank and labels[s] != labels[s+2]:
                b = np.logaddexp(b, beta[t+1, s+2] + logits[t+1, labels[s+2]])
            beta[t, s] = b

    # Gradient
    log_prob = np.logaddexp(alpha[T-1, L-1], alpha[T-1, L-2]) if L > 1 else alpha[T-1, 0]
    grad = np.exp(logits)  # softmax probabilities

    for t in range(T):
        for s in range(L):
            lab = labels[s]
            gamma = alpha[t, s] + beta[t, s]
            if gamma > LOG_0 + 10:
                grad[t, lab] -= np.exp(gamma - log_prob)

    return loss, grad

class FourierKANAdapterCTC:
    """Fourier KAN adapter + linear CTC head."""
    def __init__(self, dim=512, vocab_size=500, K=3, lr=0.001):
        self.dim = dim
        self.K = K
        self.lr = lr
        self.bs = 2*K+1
        # KAN coefficients [dim, 2K+1]
        self.coeffs = np.random.randn(dim, self.bs).astype(np.float32) * 0.001
        # Linear head [dim, vocab_size]
        self.W_head = np.random.randn(dim, vocab_size).astype(np.float32) * 0.01
        self.b_head = np.zeros(vocab_size, dtype=np.float32)
        # Adam states
        self.m_c = np.zeros_like(self.coeffs); self.v_c = np.zeros_like(self.coeffs)
        self.m_w = np.zeros_like(self.W_head); self.v_w = np.zeros_like(self.W_head)
        self.m_b = np.zeros_like(self.b_head); self.v_b = np.zeros_like(self.b_head)
        self.t = 0

    def forward(self, x):
        """x: [T, 512] → logits [T, vocab]"""
        T, D = x.shape
        # KAN correction
        self.last_basis = np.zeros((T, D, self.bs), dtype=np.float32)
        correction = np.zeros_like(x)
        for d in range(D):
            col = x[:, d]
            basis = np.zeros((T, self.bs), dtype=np.float32)
            basis[:, 0] = 1.0
            for k in range(1, self.K+1):
                basis[:, 2*k-1] = np.sin(k * col)
                basis[:, 2*k] = np.cos(k * col)
            self.last_basis[:, d, :] = basis
            correction[:, d] = basis @ self.coeffs[d]

        self.last_x = x
        self.adapted = x + correction
        # Linear head → log softmax
        logits = self.adapted @ self.W_head + self.b_head
        # Log softmax
        logits -= logits.max(axis=1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True) + 1e-10)
        self.last_log_probs = log_probs
        return log_probs

    def backward(self, ctc_grad):
        """ctc_grad: [T, vocab] gradient from CTC loss."""
        T = ctc_grad.shape[0]
        # Through linear head
        d_adapted = ctc_grad @ self.W_head.T  # [T, 512]
        d_W = self.adapted.T @ ctc_grad       # [512, vocab]
        d_b = ctc_grad.sum(axis=0)

        # Through KAN
        d_coeffs = np.zeros_like(self.coeffs)
        for d in range(self.dim):
            d_coeffs[d] = self.last_basis[:, d, :].T @ d_adapted[:, d]

        return d_coeffs, d_W, d_b

    def adam_step(self, d_coeffs, d_W, d_b, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        bc1 = 1 - beta1**self.t; bc2 = 1 - beta2**self.t
        for p, g, m, v in [(self.coeffs, d_coeffs, self.m_c, self.v_c),
                            (self.W_head, d_W, self.m_w, self.v_w),
                            (self.b_head, d_b, self.m_b, self.v_b)]:
            m[:] = beta1*m + (1-beta1)*g
            v[:] = beta2*v + (1-beta2)*g**2
            p -= self.lr * (m/bc1) / (np.sqrt(v/bc2) + eps)

def main():
    print("=== KAN Adapter v2: CTC Training ===\n")

    enc = ort.InferenceSession("stt/vosk-onnx/encoder.int8.onnx", providers=['CPUExecutionProvider'])
    token2id, id2token = load_tokens("stt/vosk-onnx/tokens.txt")
    vocab_size = len(id2token)
    print(f"Vocab: {vocab_size} tokens")

    # Load training data — use train farfield
    train_manifest = "stt/test/farfield/manifest.tsv"  # using test as train for now
    samples = []
    with open(train_manifest) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            samples.append(row)

    N_train = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    N_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # Prepare data: extract features + tokenize transcriptions
    print(f"Preparing {N_train} samples...")
    data = []
    for i, s in enumerate(samples[:N_train]):
        wav_path = os.path.join("stt/test/farfield", s['path'])
        if not os.path.exists(wav_path): continue
        audio, sr = read_wav_float(wav_path)
        if audio is None: continue
        fbank = compute_fbank(audio, sr)
        x = fbank[np.newaxis, :, :]
        x_lens = np.array([fbank.shape[0]], dtype=np.int64)
        enc_out, enc_lens = enc.run(None, {'x': x, 'x_lens': x_lens})
        feats = enc_out[0, :int(enc_lens[0]), :]

        text = s.get('transcription', '').strip()
        token_ids = text_to_token_ids(text, token2id)
        if len(token_ids) == 0: continue
        if feats.shape[0] <= len(token_ids): continue  # CTC needs T > S

        data.append({'feats': feats, 'tokens': token_ids, 'text': text})

    print(f"Prepared {len(data)} valid samples")

    # Train KAN adapter with CTC
    adapter = FourierKANAdapterCTC(dim=512, vocab_size=vocab_size, K=3, lr=0.0005)

    print(f"\nTraining for {N_epochs} epochs...\n")
    for epoch in range(N_epochs):
        np.random.shuffle(data)
        total_loss = 0
        n_samples = 0

        for sample in data:
            feats = sample['feats']
            tokens = sample['tokens']

            log_probs = adapter.forward(feats)

            try:
                loss, grad = ctc_loss_and_grad(log_probs, tokens)
            except:
                continue

            if not np.isfinite(loss):
                continue

            d_coeffs, d_W, d_b = adapter.backward(grad)

            # Gradient clipping
            for g in [d_coeffs, d_W, d_b]:
                norm = np.linalg.norm(g)
                if norm > 5.0:
                    g *= 5.0 / norm

            adapter.adam_step(d_coeffs, d_W, d_b)
            total_loss += loss
            n_samples += 1

        avg_loss = total_loss / max(n_samples, 1)
        print(f"  Epoch {epoch+1}/{N_epochs}: CTC loss = {avg_loss:.3f} ({n_samples} samples)")

    # Test: greedy CTC decode vs original transducer decode
    print(f"\n=== Testing CTC adapter decode ===\n")
    hard = []
    with open("stt/features/hard_samples.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            hard.append(row)

    dec = ort.InferenceSession("stt/vosk-onnx/decoder.onnx", providers=['CPUExecutionProvider'])
    join = ort.InferenceSession("stt/vosk-onnx/joiner.onnx", providers=['CPUExecutionProvider'])

    def transducer_decode(enc_out):
        T = enc_out.shape[0]; blank = 0; ctx = 2
        di = np.array([[blank]*ctx], dtype=np.int64)
        do = dec.run(None, {'y': di})[0]
        toks = []
        for t in range(T):
            logit = join.run(None, {'encoder_out': enc_out[t:t+1,:], 'decoder_out': do})[0]
            tid = int(np.argmax(logit[0]))
            if tid != blank:
                toks.append(tid)
                c = ([blank]*ctx + toks)[-ctx:]
                di = np.array([c], dtype=np.int64)
                do = dec.run(None, {'y': di})[0]
        return ''.join(id2token.get(t,'') for t in toks).replace('▁',' ').strip()

    def ctc_decode(adapter, feats):
        log_probs = adapter.forward(feats)
        # Greedy CTC: take argmax, remove blanks and repeats
        pred = np.argmax(log_probs, axis=1)
        toks = []
        prev = -1
        for p in pred:
            if p != 0 and p != prev:  # 0 = blank
                toks.append(int(p))
            prev = p
        return ''.join(id2token.get(t,'') for t in toks).replace('▁',' ').strip()

    def edit_distance(ref, hyp):
        n,m = len(ref),len(hyp)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1): dp[i][0]=i
        for j in range(m+1): dp[0][j]=j
        for i in range(1,n+1):
            for j in range(1,m+1):
                dp[i][j] = dp[i-1][j-1] if ref[i-1]==hyp[j-1] else 1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
        return dp[n][m]

    n_test = 30
    trans_err = ctc_err = 0; total_w = 0
    for i, s in enumerate(hard[:n_test]):
        wav_path = os.path.join("stt/test/farfield", s['path'])
        if not os.path.exists(wav_path): continue
        audio, sr = read_wav_float(wav_path)
        if audio is None: continue
        fbank = compute_fbank(audio, sr)
        x = fbank[np.newaxis,:,:]; xl = np.array([fbank.shape[0]], dtype=np.int64)
        eo, el = enc.run(None, {'x': x, 'x_lens': xl})
        feats = eo[0,:int(el[0]),:]

        gt = s['transcription'].strip().lower()
        ref = gt.split(); total_w += len(ref)

        t_text = transducer_decode(feats).lower()
        c_text = ctc_decode(adapter, feats).lower()

        te = edit_distance(ref, t_text.split())
        ce = edit_distance(ref, c_text.split())
        trans_err += te; ctc_err += ce

        if te > 0 or ce > 0:
            print(f"[{i}] GT:         '{gt}'")
            print(f"     Transducer: '{t_text}' (err={te})")
            print(f"     CTC+KAN:    '{c_text}' (err={ce})")
            print()

    print(f"\n=== WER on {n_test} hard samples ===")
    print(f"Transducer (original): {trans_err/total_w*100:.1f}%")
    print(f"CTC + KAN adapter:     {ctc_err/total_w*100:.1f}%")

if __name__ == "__main__":
    main()
