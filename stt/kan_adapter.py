#!/usr/bin/env python3
"""
KAN adapter for Vosk STT: insert trainable adapter between frozen encoder and decoder.

Architecture:
  Audio → Fbank(80) → [Frozen Encoder] → [T, 512] → [KAN Adapter 512→512] → [Frozen Joiner+Decoder] → text

Compares: no adapter (baseline) vs Dense adapter vs Fourier KAN adapter
"""
import os, csv, struct, math
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
    """Compute log mel filterbank features using numpy (matching Kaldi defaults)."""
    # Frame params (Kaldi defaults)
    frame_length_ms = 25
    frame_shift_ms = 10
    frame_length = int(sr * frame_length_ms / 1000)  # 400
    frame_shift = int(sr * frame_shift_ms / 1000)    # 160
    n_fft = 512

    # Pre-emphasis
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Framing
    num_frames = max(1, (len(emphasized) - frame_length) // frame_shift + 1)
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * frame_shift
        end = min(start + frame_length, len(emphasized))
        frames[i, :end-start] = emphasized[start:end]

    # Hamming window
    window = np.hamming(frame_length)
    frames *= window

    # FFT
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2

    # Mel filterbank
    low_freq = 20
    high_freq = sr // 2
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

    mel_spec = np.dot(spectrum, filterbank.T)
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)

    return log_mel.astype(np.float32)

def greedy_decode(enc_session, dec_session, join_session, features, blank_id=0):
    """Greedy transducer decoding with encoder features."""
    # Run encoder
    x = features[np.newaxis, :, :]  # [1, T, 80]
    x_lens = np.array([features.shape[0]], dtype=np.int64)
    enc_out, enc_lens = enc_session.run(None, {'x': x, 'x_lens': x_lens})

    T = int(enc_lens[0])
    context_size = 2  # decoder context

    # Init decoder state
    decoder_input = np.array([[blank_id] * context_size], dtype=np.int64)
    decoder_out = dec_session.run(None, {'y': decoder_input})[0]  # [1, 512]

    tokens = []
    for t in range(T):
        enc_frame = enc_out[0, t:t+1, :]  # [1, 512]
        logit = join_session.run(None, {
            'encoder_out': enc_frame,
            'decoder_out': decoder_out
        })[0]  # [1, 500]

        token_id = int(np.argmax(logit[0]))
        if token_id != blank_id:
            tokens.append(token_id)
            decoder_input = np.array([[tokens[-min(context_size, len(tokens)):].pop() if len(tokens) < context_size else tokens[-context_size+j] for j in range(context_size)]], dtype=np.int64)
            # Simpler: just use last context_size tokens
            ctx = ([blank_id] * context_size + tokens)[-context_size:]
            decoder_input = np.array([ctx], dtype=np.int64)
            decoder_out = dec_session.run(None, {'y': decoder_input})[0]

    return tokens

def greedy_decode_with_adapter(enc_session, dec_session, join_session, features, adapter_fn, blank_id=0):
    """Same as greedy_decode but applies adapter to encoder output."""
    x = features[np.newaxis, :, :]
    x_lens = np.array([features.shape[0]], dtype=np.int64)
    enc_out, enc_lens = enc_session.run(None, {'x': x, 'x_lens': x_lens})

    # Apply adapter to encoder output
    T = int(enc_lens[0])
    adapted = adapter_fn(enc_out[0, :T, :])  # [T, 512] → [T, 512]
    enc_out_adapted = np.zeros_like(enc_out)
    enc_out_adapted[0, :T, :] = adapted

    context_size = 2
    decoder_input = np.array([[blank_id] * context_size], dtype=np.int64)
    decoder_out = dec_session.run(None, {'y': decoder_input})[0]

    tokens = []
    for t in range(T):
        enc_frame = enc_out_adapted[0, t:t+1, :]
        logit = join_session.run(None, {
            'encoder_out': enc_frame,
            'decoder_out': decoder_out
        })[0]

        token_id = int(np.argmax(logit[0]))
        if token_id != blank_id:
            tokens.append(token_id)
            ctx = ([blank_id] * context_size + tokens)[-context_size:]
            decoder_input = np.array([ctx], dtype=np.int64)
            decoder_out = dec_session.run(None, {'y': decoder_input})[0]

    return tokens

def load_tokens(path):
    """Load token vocabulary."""
    id2token = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                token = parts[0]
                idx = int(parts[1])
                id2token[idx] = token
    return id2token

def tokens_to_text(token_ids, id2token):
    """Convert token IDs to text."""
    pieces = []
    for tid in token_ids:
        t = id2token.get(tid, '')
        pieces.append(t)
    # Join BPE pieces (▁ = word boundary in sentencepiece)
    text = ''.join(pieces).replace('▁', ' ').strip()
    return text

def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]

class FourierKANAdapter:
    """Simple Fourier KAN adapter: 512→512 with K frequencies."""
    def __init__(self, dim=512, K=3, scale=0.001):
        self.dim = dim
        self.K = K
        # coeffs: [dim, 2K+1] — cos/sin for each feature
        self.coeffs = np.random.randn(dim, 2*K+1).astype(np.float32) * scale

    def forward(self, x):
        """x: [T, 512] → [T, 512]. Residual connection: output = x + adapter(x)"""
        T, D = x.shape
        # Build basis for each feature: cos(k*x), sin(k*x)
        out = np.zeros_like(x)
        for d in range(D):
            col = x[:, d]  # [T]
            basis = np.zeros((T, 2*self.K+1), dtype=np.float32)
            basis[:, 0] = 1.0  # DC
            for k in range(1, self.K+1):
                basis[:, 2*k-1] = np.sin(k * col)
                basis[:, 2*k] = np.cos(k * col)
            out[:, d] = basis @ self.coeffs[d]  # [T]
        return x + out  # residual

def main():
    print("=== KAN Adapter for Vosk STT ===\n")

    # Load models
    enc = ort.InferenceSession("stt/vosk-onnx/encoder.int8.onnx", providers=['CPUExecutionProvider'])
    dec = ort.InferenceSession("stt/vosk-onnx/decoder.onnx", providers=['CPUExecutionProvider'])
    join = ort.InferenceSession("stt/vosk-onnx/joiner.onnx", providers=['CPUExecutionProvider'])
    id2token = load_tokens("stt/vosk-onnx/tokens.txt")
    print(f"Vocabulary: {len(id2token)} tokens")

    # Load hard samples
    hard_samples = []
    with open("stt/features/hard_samples.tsv", 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            hard_samples.append(row)

    # Test baseline on first 20 hard samples
    n_test = 20
    print(f"\nTesting on {n_test} hard samples...\n")

    baseline_errors = 0
    baseline_words = 0
    adapter_errors = 0
    adapter_words = 0

    adapter = FourierKANAdapter(dim=512, K=3, scale=0.0)  # scale=0 means identity (no change)

    for i, s in enumerate(hard_samples[:n_test]):
        wav_path = os.path.join("stt/test/farfield", s['path'])
        if not os.path.exists(wav_path):
            continue

        audio, sr = read_wav_float(wav_path)
        if audio is None:
            continue

        # Compute features
        fbank = compute_fbank(audio, sr)

        # Baseline decode
        tokens_base = greedy_decode(enc, dec, join, fbank)
        text_base = tokens_to_text(tokens_base, id2token)

        # Adapter decode (currently identity)
        tokens_adapt = greedy_decode_with_adapter(enc, dec, join, fbank,
                                                   adapter.forward)
        text_adapt = tokens_to_text(tokens_adapt, id2token)

        gt = s['transcription'].strip().lower()
        base_ed = edit_distance(gt.split(), text_base.lower().split())
        adapt_ed = edit_distance(gt.split(), text_adapt.lower().split())
        n_words = len(gt.split())

        baseline_errors += base_ed
        baseline_words += n_words
        adapter_errors += adapt_ed
        adapter_words += n_words

        if base_ed > 0 or i < 5:
            print(f"[{i}] GT:       '{gt}'")
            print(f"     Baseline: '{text_base}' (errors={base_ed})")
            print(f"     Adapter:  '{text_adapt}' (errors={adapt_ed})")
            print()

    base_wer = baseline_errors / max(baseline_words, 1) * 100
    adapt_wer = adapter_errors / max(adapter_words, 1) * 100
    print(f"\n=== Results ({n_test} hard samples) ===")
    print(f"Baseline WER: {base_wer:.1f}%")
    print(f"Adapter WER:  {adapt_wer:.1f}% (currently identity — should match baseline)")
    print(f"\nNext step: train adapter on hard samples to minimize WER")

if __name__ == "__main__":
    main()
