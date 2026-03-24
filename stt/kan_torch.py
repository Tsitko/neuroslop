#!/usr/bin/env python3
"""
KAN adapter with PyTorch: full backprop through decoder.

1. Convert ONNX decoder+joiner to PyTorch (small, simple models)
2. Pre-compute encoder features (frozen, via ONNX)
3. Insert Fourier KAN adapter (PyTorch, trainable)
4. Train end-to-end with RNN-T loss via greedy decoding + cross-entropy proxy

Since full RNN-T loss is complex, we use a proxy:
  - Run greedy decode with current adapter
  - At each timestep, compute cross-entropy loss for correct next token
  - Backprop through adapter
  This is teacher-forcing through the transducer.
"""
import os, csv, struct, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort

def read_wav_float(path):
    with open(path, 'rb') as f:
        f.read(12); sr = 16000; audio = None
        while True:
            cid = f.read(4)
            if len(cid) < 4: break
            cs = struct.unpack('<I', f.read(4))[0]
            if cid == b'fmt ': fmt = f.read(cs); sr = struct.unpack('<I', fmt[4:8])[0]
            elif cid == b'data': audio = np.frombuffer(f.read(cs), dtype=np.float32)
            else: f.read(cs)
    return audio, sr

def compute_fbank(audio, sr=16000, n_mels=80):
    fl = int(sr*0.025); fs = int(sr*0.010); nf = 512
    em = np.append(audio[0], audio[1:]-0.97*audio[:-1])
    nfr = max(1, (len(em)-fl)//fs+1)
    frames = np.zeros((nfr, fl))
    for i in range(nfr):
        s=i*fs; e=min(s+fl,len(em)); frames[i,:e-s]=em[s:e]
    frames *= np.hamming(fl)
    spec = np.abs(np.fft.rfft(frames, n=nf))**2
    ml = 2595*np.log10(1+20/700); mh = 2595*np.log10(1+sr/2/700)
    mp = np.linspace(ml,mh,n_mels+2); hp = 700*(10**(mp/2595)-1)
    bins = np.floor((nf+1)*hp/sr).astype(int)
    fb = np.zeros((n_mels, nf//2+1))
    for i in range(n_mels):
        for j in range(bins[i],bins[i+1]): fb[i,j]=(j-bins[i])/max(1,bins[i+1]-bins[i])
        for j in range(bins[i+1],bins[i+2]): fb[i,j]=(bins[i+2]-j)/max(1,bins[i+2]-bins[i+1])
    return np.log(np.maximum(np.dot(spec,fb.T),1e-10)).astype(np.float32)

class FourierKANAdapter(nn.Module):
    """Fourier KAN adapter: residual 512→512 with K frequencies per dim."""
    def __init__(self, dim=512, K=3):
        super().__init__()
        self.dim = dim
        self.K = K
        self.coeffs = nn.Parameter(torch.zeros(dim, 2*K+1) * 0.001)

    def forward(self, x):
        # x: [T, 512]
        T, D = x.shape
        basis_list = [torch.ones(T, 1, device=x.device)]
        for k in range(1, self.K+1):
            basis_list.append(torch.sin(k * x))  # [T, D] but we need per-dim
            basis_list.append(torch.cos(k * x))

        # Actually per-dimension basis:
        correction = torch.zeros_like(x)
        for d in range(D):
            col = x[:, d]  # [T]
            basis = torch.ones(T, 2*self.K+1, device=x.device)
            for k in range(1, self.K+1):
                basis[:, 2*k-1] = torch.sin(k * col)
                basis[:, 2*k] = torch.cos(k * col)
            correction[:, d] = basis @ self.coeffs[d]

        return x + correction

class DenseAdapter(nn.Module):
    """Simple Dense residual adapter for comparison."""
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return x + self.linear(x)

def load_tokens(path):
    t2id = {}; id2t = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2: t2id[p[0]]=int(p[1]); id2t[int(p[1])]=p[0]
    return t2id, id2t

def text_to_token_ids(text, token2id):
    text = '▁' + text.lower().strip().replace(' ', '▁')
    tokens = []; i = 0
    while i < len(text):
        best_len = 0; best_id = None
        for length in range(min(10, len(text)-i), 0, -1):
            sub = text[i:i+length]
            if sub in token2id: best_len=length; best_id=token2id[sub]; break
        if best_id is not None: tokens.append(best_id); i+=best_len
        else: i+=1
    return tokens

def edit_distance(ref, hyp):
    n,m=len(ref),len(hyp)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j]=dp[i-1][j-1] if ref[i-1]==hyp[j-1] else 1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
    return dp[n][m]

def main():
    print("=== KAN Adapter: PyTorch with backprop ===\n")

    # Load frozen encoder (ONNX)
    enc = ort.InferenceSession("stt/vosk-onnx/encoder.int8.onnx", providers=['CPUExecutionProvider'])

    # Build decoder + joiner in PyTorch from ONNX weights
    print("Building decoder + joiner from ONNX weights...")
    import onnx
    from onnx import numpy_helper

    def get_onnx_weights(path):
        model = onnx.load(path)
        weights = {}
        for init in model.graph.initializer:
            weights[init.name] = numpy_helper.to_array(init)
        return weights

    # Decoder: Embedding(500,512) → Conv1D(512,512,k=2) → ReLU → Linear(512,512)
    class TransducerDecoder(nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.embedding = nn.Embedding(500, 512)
            # Grouped conv: weight [512, 4, 2] = groups=128, each group: 4 in → 4 out
            self.conv = nn.Conv1d(512, 512, kernel_size=2, padding=0, groups=128)
            self.proj = nn.Linear(512, 512)
            # Load weights
            self.embedding.weight.data = torch.tensor(weights['decoder.embedding.weight'])
            self.conv.weight.data = torch.tensor(weights['decoder.conv.weight'])
            if 'decoder.conv.bias' in weights:
                self.conv.bias.data = torch.tensor(weights['decoder.conv.bias'])
            else:
                self.conv.bias = None
            self.proj.weight.data = torch.tensor(weights['decoder_proj.weight'])
            self.proj.bias.data = torch.tensor(weights['decoder_proj.bias'])

        def forward(self, y):
            # y: [N, 2] (context token IDs)
            y = y.clamp(min=0)
            mask = (y >= 0).unsqueeze(-1).float()
            emb = self.embedding(y) * mask  # [N, 2, 512]
            x = emb.transpose(1, 2)  # [N, 512, 2]
            x = self.conv(x)  # [N, 512, 1]
            x = F.relu(x)
            x = x.squeeze(2)  # [N, 512]
            x = self.proj(x)  # [N, 512]
            return x

    # Joiner: Add → Tanh → Linear(512, 500)
    class TransducerJoiner(nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.linear = nn.Linear(512, 500)
            self.linear.weight.data = torch.tensor(weights['output_linear.weight'])
            self.linear.bias.data = torch.tensor(weights['output_linear.bias'])

        def forward(self, encoder_out, decoder_out):
            x = encoder_out + decoder_out
            x = torch.tanh(x)
            return self.linear(x)

    dec_weights = get_onnx_weights("stt/vosk-onnx/decoder.onnx")
    join_weights = get_onnx_weights("stt/vosk-onnx/joiner.onnx")

    decoder_pt = TransducerDecoder(dec_weights).eval()
    joiner_pt = TransducerJoiner(join_weights).eval()

    for p in decoder_pt.parameters(): p.requires_grad = False
    for p in joiner_pt.parameters(): p.requires_grad = False

    print(f"Decoder params: {sum(p.numel() for p in decoder_pt.parameters()):,}")
    print(f"Joiner params: {sum(p.numel() for p in joiner_pt.parameters()):,}")

    token2id, id2token = load_tokens("stt/vosk-onnx/tokens.txt")
    vocab_size = len(id2token)
    blank_id = 0
    context_size = 2

    # Load data
    samples = []
    with open("stt/test/farfield/manifest.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader: samples.append(row)

    N_train = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    N_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"\nPreparing {N_train} samples...")
    data = []
    for i, s in enumerate(samples[:N_train]):
        wav_path = os.path.join("stt/test/farfield", s['path'])
        if not os.path.exists(wav_path): continue
        audio, sr = read_wav_float(wav_path)
        if audio is None: continue
        fbank = compute_fbank(audio, sr)
        x = fbank[np.newaxis,:,:]; xl = np.array([fbank.shape[0]], dtype=np.int64)
        eo, el = enc.run(None, {'x': x, 'x_lens': xl})
        feats = eo[0, :int(el[0]), :]

        text = s.get('transcription','').strip()
        token_ids = text_to_token_ids(text, token2id)
        if len(token_ids) == 0 or feats.shape[0] <= len(token_ids): continue

        data.append({
            'feats': torch.tensor(feats, dtype=torch.float32),
            'tokens': token_ids,
            'text': text,
        })
    print(f"Prepared {len(data)} samples")

    # Train adapters
    def train_adapter(adapter, name):
        optimizer = torch.optim.Adam(adapter.parameters(), lr=0.0005)

        for epoch in range(N_epochs):
            np.random.shuffle(data)
            total_loss = 0; n = 0

            for sample in data:
                feats = sample['feats']  # [T, 512]
                target_tokens = sample['tokens']

                # Forward through adapter
                adapted = adapter(feats)  # [T, 512]

                # Teacher-forced transducer: at each timestep, predict next token
                T = adapted.shape[0]
                ctx = [blank_id] * context_size
                loss = torch.tensor(0.0)
                target_idx = 0

                for t in range(min(T, len(target_tokens) * 3)):  # limit steps
                    if target_idx >= len(target_tokens):
                        break

                    enc_frame = adapted[t:t+1, :]  # [1, 512]
                    dec_input = torch.tensor([ctx], dtype=torch.long)

                    with torch.no_grad():
                        dec_out = decoder_pt(dec_input)  # [1, 512]

                    # Joiner (needs grad for adapter)
                    logit = joiner_pt(enc_frame, dec_out)  # [1, vocab]
                    log_prob = F.log_softmax(logit, dim=-1)

                    # Target: either emit next token or blank
                    # Simple: cross-entropy to next target token
                    target = target_tokens[target_idx]
                    loss = loss - log_prob[0, target]

                    # Advance: if model would emit this token, move to next
                    pred = logit.argmax(dim=-1).item()
                    if pred == target or pred != blank_id:
                        target_idx += 1
                        ctx = ([blank_id]*context_size + target_tokens[:target_idx])[-context_size:]

                if target_idx > 0:
                    loss = loss / target_idx
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    n += 1

            avg = total_loss / max(n, 1)
            if (epoch+1) % 2 == 0 or epoch == 0:
                print(f"  [{name}] Epoch {epoch+1}/{N_epochs}: loss={avg:.3f}")

        return adapter

    # Train Fourier KAN adapter
    print(f"\n--- Training Fourier KAN adapter (K=3) ---")
    kan_adapter = train_adapter(FourierKANAdapter(512, K=3), "KAN")

    print(f"\n--- Training Dense adapter ---")
    dense_adapter = train_adapter(DenseAdapter(512), "Dense")

    # Test: decode with adapters using ORIGINAL transducer decoder
    print(f"\n=== Testing on hard samples ===\n")
    hard = []
    with open("stt/features/hard_samples.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader: hard.append(row)

    def decode_with_adapter(feats_np, adapter):
        with torch.no_grad():
            feats_t = torch.tensor(feats_np, dtype=torch.float32)
            adapted = adapter(feats_t).numpy() if adapter else feats_np

        T = adapted.shape[0]
        ctx = [blank_id] * context_size
        tokens = []
        dec_in = np.array([ctx], dtype=np.int64)

        # Use ONNX decoder for inference (exact same as baseline)
        dec_onnx = ort.InferenceSession("stt/vosk-onnx/decoder.onnx", providers=['CPUExecutionProvider'])
        join_onnx = ort.InferenceSession("stt/vosk-onnx/joiner.onnx", providers=['CPUExecutionProvider'])

        dec_out = dec_onnx.run(None, {'y': dec_in})[0]
        for t in range(T):
            logit = join_onnx.run(None, {
                'encoder_out': adapted[t:t+1,:].astype(np.float32),
                'decoder_out': dec_out
            })[0]
            tid = int(np.argmax(logit[0]))
            if tid != blank_id:
                tokens.append(tid)
                ctx = ([blank_id]*context_size + tokens)[-context_size:]
                dec_in = np.array([ctx], dtype=np.int64)
                dec_out = dec_onnx.run(None, {'y': dec_in})[0]

        return ''.join(id2token.get(t,'') for t in tokens).replace('▁',' ').strip()

    n_test = 30
    base_err = kan_err = dense_err = 0
    total_w = 0

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

        t_base = decode_with_adapter(feats, None).lower()
        t_kan = decode_with_adapter(feats, kan_adapter).lower()
        t_dense = decode_with_adapter(feats, dense_adapter).lower()

        eb = edit_distance(ref, t_base.split())
        ek = edit_distance(ref, t_kan.split())
        ed = edit_distance(ref, t_dense.split())
        base_err += eb; kan_err += ek; dense_err += ed

        if eb > 0:
            improved = []
            if ek < eb: improved.append(f"KAN(-{eb-ek})")
            if ed < eb: improved.append(f"Dense(-{eb-ed})")
            if ek > eb: improved.append(f"KAN(+{ek-eb})")
            if ed > eb: improved.append(f"Dense(+{ed-eb})")
            imp = ', '.join(improved) if improved else 'no change'

            print(f"[{i}] GT:    '{gt}'")
            print(f"     Base:  '{t_base}' (err={eb})")
            if ek != eb: print(f"     KAN:   '{t_kan}' (err={ek})")
            if ed != eb: print(f"     Dense: '{t_dense}' (err={ed})")
            print(f"     → {imp}")
            print()

    print(f"\n=== WER on {n_test} hard samples ===")
    print(f"Baseline:      {base_err/total_w*100:.1f}% ({base_err}/{total_w})")
    print(f"Fourier KAN:   {kan_err/total_w*100:.1f}% ({kan_err}/{total_w})")
    print(f"Dense adapter: {dense_err/total_w*100:.1f}% ({dense_err}/{total_w})")

if __name__ == "__main__":
    main()
