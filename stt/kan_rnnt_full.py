#!/usr/bin/env python3
"""
Full KAN adapter training: train on crowd data, test on farfield.

Compares Fourier KAN adapter with K=0 (baseline), K=16, K=32, K=64.
Uses RNN-T loss for proper alignment-aware training.

Usage: python3 stt/kan_rnnt_full.py [n_train] [n_epochs]
  Default: n_train=5000, n_epochs=15
"""
import os, csv, struct, sys, json, time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # line-buffered
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
from torchaudio.transforms import RNNTLoss

# ---- Audio utils ----

def read_wav(path):
    """Read WAV file (supports both IEEE Float and PCM16)."""
    with open(path, 'rb') as f:
        f.read(12)  # RIFF + size + WAVE
        sr = 16000; audio = None; fmt_tag = 1; bps = 16
        while True:
            cid = f.read(4)
            if len(cid) < 4: break
            cs = struct.unpack('<I', f.read(4))[0]
            if cid == b'fmt ':
                fmt = f.read(cs)
                fmt_tag = struct.unpack('<H', fmt[0:2])[0]
                sr = struct.unpack('<I', fmt[4:8])[0]
                bps = struct.unpack('<H', fmt[14:16])[0]
            elif cid == b'data':
                raw = f.read(cs)
                if fmt_tag == 3:  # IEEE float
                    audio = np.frombuffer(raw, dtype=np.float32)
                else:  # PCM16
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                f.read(cs)
    return audio, sr

def compute_fbank(audio, sr=16000, n_mels=80):
    fl=int(sr*0.025); fs=int(sr*0.010); nf=512
    em=np.append(audio[0], audio[1:]-0.97*audio[:-1])
    nfr=max(1,(len(em)-fl)//fs+1)
    frames=np.zeros((nfr,fl))
    for i in range(nfr):
        s=i*fs; e=min(s+fl,len(em)); frames[i,:e-s]=em[s:e]
    frames*=np.hamming(fl)
    spec=np.abs(np.fft.rfft(frames,n=nf))**2
    ml=2595*np.log10(1+20/700); mh=2595*np.log10(1+sr/2/700)
    mp=np.linspace(ml,mh,n_mels+2); hp=700*(10**(mp/2595)-1)
    bins=np.floor((nf+1)*hp/sr).astype(int)
    fb=np.zeros((n_mels,nf//2+1))
    for i in range(n_mels):
        for j in range(bins[i],bins[i+1]): fb[i,j]=(j-bins[i])/max(1,bins[i+1]-bins[i])
        for j in range(bins[i+1],bins[i+2]): fb[i,j]=(bins[i+2]-j)/max(1,bins[i+2]-bins[i+1])
    return np.log(np.maximum(np.dot(spec,fb.T),1e-10)).astype(np.float32)

# ---- Token utils ----

def load_tokens(path):
    t2id={}; id2t={}
    with open(path) as f:
        for line in f:
            p=line.strip().split()
            if len(p)>=2: t2id[p[0]]=int(p[1]); id2t[int(p[1])]=p[0]
    return t2id, id2t

def text_to_token_ids(text, token2id):
    text='▁'+text.lower().strip().replace(' ','▁')
    tokens=[]; i=0
    while i<len(text):
        best_len=0; best_id=None
        for length in range(min(10,len(text)-i),0,-1):
            sub=text[i:i+length]
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

# ---- Models ----

class FourierKANAdapter(nn.Module):
    def __init__(self, dim=512, K=3):
        super().__init__()
        self.dim = dim; self.K = K
        if K == 0:
            # Identity — no parameters
            self.coeffs = None
        else:
            self.coeffs = nn.Parameter(torch.randn(dim, 2*K+1) * 0.001)

    def forward(self, x):
        if self.K == 0:
            return x  # identity baseline
        T, D = x.shape
        correction = torch.zeros_like(x)
        for d in range(D):
            col = x[:, d]
            basis = torch.ones(T, 2*self.K+1, device=x.device)
            for k in range(1, self.K+1):
                basis[:, 2*k-1] = torch.sin(k * col)
                basis[:, 2*k] = torch.cos(k * col)
            correction[:, d] = basis @ self.coeffs[d]
        return x + correction

    @property
    def num_params(self):
        return self.dim * (2*self.K+1) if self.K > 0 else 0

class TransducerDecoder(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.embedding = nn.Embedding(500, 512)
        self.conv = nn.Conv1d(512, 512, kernel_size=2, padding=0, groups=128)
        self.proj = nn.Linear(512, 512)
        self.embedding.weight.data = torch.tensor(weights['decoder.embedding.weight'])
        self.conv.weight.data = torch.tensor(weights['decoder.conv.weight'])
        self.proj.weight.data = torch.tensor(weights['decoder_proj.weight'])
        self.proj.bias.data = torch.tensor(weights['decoder_proj.bias'])
    def forward(self, y):
        y=y.clamp(min=0); mask=(y>=0).unsqueeze(-1).float()
        emb=self.embedding(y)*mask; x=emb.transpose(1,2)
        x=self.conv(x); x=F.relu(x).squeeze(2)
        return self.proj(x)

class TransducerJoiner(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.linear = nn.Linear(512, 500)
        self.linear.weight.data = torch.tensor(weights['output_linear.weight'])
        self.linear.bias.data = torch.tensor(weights['output_linear.bias'])
    def forward(self, enc, dec):
        return self.linear(torch.tanh(enc + dec))

def compute_rnnt_logits(encoder_out, decoder, joiner, targets, blank_id=0):
    T=encoder_out.shape[0]; U=len(targets); ctx=2
    dec_inputs=[]
    for u in range(U+1):
        if u==0: c=[blank_id]*ctx
        elif u==1: c=[blank_id, targets[0]]
        else: c=[targets[u-2], targets[u-1]]
        dec_inputs.append(c)
    dec_out = decoder(torch.tensor(dec_inputs, dtype=torch.long))
    enc_exp = encoder_out.unsqueeze(1)
    dec_exp = dec_out.unsqueeze(0)
    logits = joiner.linear(torch.tanh(enc_exp + dec_exp))
    return logits.unsqueeze(0)

# ---- Main ----

def main():
    N_train = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    N_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    print(f"=== Fourier KAN Adapter: Full Training ===")
    print(f"Train samples: {N_train}, Epochs: {N_epochs}")
    print(f"Comparing K=0 (baseline), K=16, K=32, K=64\n")

    enc = ort.InferenceSession("stt/vosk-onnx/encoder.int8.onnx", providers=['CPUExecutionProvider'])

    import onnx
    from onnx import numpy_helper
    def get_w(p):
        m=onnx.load(p); return {i.name:numpy_helper.to_array(i) for i in m.graph.initializer}

    dec_weights = get_w("stt/vosk-onnx/decoder.onnx")
    join_weights = get_w("stt/vosk-onnx/joiner.onnx")
    token2id, id2token = load_tokens("stt/vosk-onnx/tokens.txt")
    rnnt_loss_fn = RNNTLoss(blank=0)

    # ---- Load train data (crowd shard 9) ----
    print("Loading train data from manifest...")
    train_data = []
    with open("stt/train/manifest.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if '/9/' not in d['audio_filepath']: continue
            path = os.path.join("stt/train", d['audio_filepath'])
            if not os.path.exists(path): continue
            train_data.append({'path': path, 'text': d['text'], 'duration': d.get('duration', 0)})
            if len(train_data) >= N_train: break

    print(f"Found {len(train_data)} train samples. Extracting features...")

    # Cache features to disk
    cache_path = f"stt/features/train_feats_{N_train}.npz"
    if os.path.exists(cache_path):
        print(f"  Loading cached features from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        train_feats = list(cached['data'])
        print(f"  Loaded {len(train_feats)} cached samples")
    else:
        train_feats = []
        for i, s in enumerate(train_data):
            audio, sr = read_wav(s['path'])
            if audio is None: continue
            fbank = compute_fbank(audio, sr)
            if fbank.shape[0] < 50: continue
            try:
                x = fbank[np.newaxis,:,:]; xl = np.array([fbank.shape[0]], dtype=np.int64)
                eo, el = enc.run(None, {'x': x, 'x_lens': xl})
                feats = eo[0,:int(el[0]),:]
            except:
                continue
            tokens = text_to_token_ids(s['text'], token2id)
            if len(tokens) == 0 or feats.shape[0] <= len(tokens): continue
            train_feats.append({'feats': feats, 'tokens': tokens, 'text': s['text']})
            if (i+1) % 200 == 0:
                print(f"  {i+1}/{len(train_data)} extracted ({len(train_feats)} valid)")

        # Save cache
        np.savez(cache_path, data=np.array(train_feats, dtype=object))
        print(f"  Cached to {cache_path}")

    # Convert to torch tensors
    for s in train_feats:
        if isinstance(s['feats'], np.ndarray):
            s['feats'] = torch.tensor(s['feats'], dtype=torch.float32)

    print(f"Train: {len(train_feats)} valid samples\n")

    # ---- Load test data (farfield) ----
    print("Loading test data...")
    test_data = []
    with open("stt/features/hard_samples.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader: test_data.append(row)

    test_feats = []
    for s in test_data[:50]:
        wav_path = os.path.join("stt/test/farfield", s['path'])
        if not os.path.exists(wav_path): continue
        audio, sr = read_wav(wav_path)
        if audio is None: continue
        fbank = compute_fbank(audio, sr)
        if fbank.shape[0] < 50: continue
        try:
            x = fbank[np.newaxis,:,:]; xl = np.array([fbank.shape[0]], dtype=np.int64)
            eo, el = enc.run(None, {'x': x, 'x_lens': xl})
            feats = eo[0,:int(el[0]),:]
        except:
            continue
        test_feats.append({'feats': feats, 'text': s['transcription'].strip().lower()})

    print(f"Test: {len(test_feats)} samples\n")

    # ---- Decode function ----
    dec_onnx = ort.InferenceSession("stt/vosk-onnx/decoder.onnx", providers=['CPUExecutionProvider'])
    join_onnx = ort.InferenceSession("stt/vosk-onnx/joiner.onnx", providers=['CPUExecutionProvider'])

    def decode_onnx(feats_np):
        T=feats_np.shape[0]; blank=0; ctx=2
        di=np.array([[blank]*ctx],dtype=np.int64)
        do=dec_onnx.run(None,{'y':di})[0]; toks=[]
        for t in range(T):
            logit=join_onnx.run(None,{'encoder_out':feats_np[t:t+1,:].astype(np.float32),'decoder_out':do})[0]
            tid=int(np.argmax(logit[0]))
            if tid!=blank:
                toks.append(tid)
                ctx_ids=([blank]*ctx+toks)[-ctx:]
                di=np.array([ctx_ids],dtype=np.int64)
                do=dec_onnx.run(None,{'y':di})[0]
        return ''.join(id2token.get(t,'') for t in toks).replace('▁',' ').strip()

    def eval_wer(adapter, label=""):
        errs=0; total=0
        for s in test_feats:
            with torch.no_grad():
                adapted = adapter(torch.tensor(s['feats'], dtype=torch.float32)).numpy()
            pred = decode_onnx(adapted).lower()
            ref = s['text'].split(); hyp = pred.split()
            errs += edit_distance(ref, hyp); total += len(ref)
        wer = errs/max(total,1)*100
        return wer, errs, total

    # ---- Baseline WER ----
    identity = FourierKANAdapter(512, K=0)
    base_wer, _, _ = eval_wer(identity, "baseline")
    print(f"Baseline WER (no adapter): {base_wer:.1f}%\n")

    # ---- Train each K ----
    results = [("K=0 (baseline)", base_wer, 0, 0)]

    for K in [3, 8, 16]:
        print(f"{'='*60}")
        print(f"Training K={K} ({512*(2*K+1)} params)...")
        print(f"{'='*60}")

        decoder = TransducerDecoder(dec_weights).eval()
        joiner = TransducerJoiner(join_weights).eval()
        for p in decoder.parameters(): p.requires_grad = False
        for p in joiner.parameters(): p.requires_grad = False

        adapter = FourierKANAdapter(512, K=K)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=0.0003)

        t0 = time.time()
        for epoch in range(N_epochs):
            indices = np.random.permutation(len(train_feats))
            total_loss = 0; n = 0

            for idx in indices:
                sample = train_feats[idx]
                feats = sample['feats']
                tokens = sample['tokens']

                adapted = adapter(feats)
                logits = compute_rnnt_logits(adapted, decoder, joiner, tokens)
                log_probs = F.log_softmax(logits, dim=-1)

                tgt = torch.tensor([tokens], dtype=torch.int32)
                tgt_len = torch.tensor([len(tokens)], dtype=torch.int32)
                src_len = torch.tensor([adapted.shape[0]], dtype=torch.int32)

                try:
                    loss = rnnt_loss_fn(log_probs, tgt, src_len, tgt_len)
                except:
                    continue
                if not torch.isfinite(loss): continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item(); n += 1

            avg_loss = total_loss / max(n, 1)
            elapsed = time.time() - t0

            if (epoch+1) % 3 == 0 or epoch == 0 or epoch == N_epochs-1:
                wer, errs, total = eval_wer(adapter, f"K={K}")
                print(f"  Epoch {epoch+1}/{N_epochs}: loss={avg_loss:.3f}, "
                      f"WER={wer:.1f}% ({errs}/{total}), time={elapsed:.0f}s")
            else:
                print(f"  Epoch {epoch+1}/{N_epochs}: loss={avg_loss:.3f}, time={elapsed:.0f}s")

        final_wer, errs, total = eval_wer(adapter, f"K={K} final")
        train_time = time.time() - t0
        results.append((f"K={K}", final_wer, adapter.num_params, train_time))
        print(f"\n  K={K} final: WER={final_wer:.1f}%, params={adapter.num_params}, time={train_time:.0f}s\n")

        # Save adapter weights
        torch.save(adapter.state_dict(), f"stt/features/adapter_K{K}.pt")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"{'FINAL RESULTS':^60}")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'WER':>8} {'Params':>10} {'Time':>10}")
    print("-" * 50)
    for name, wer, params, t in results:
        print(f"{name:<20} {wer:>7.1f}% {params:>10,} {t:>9.0f}s")
    print()

if __name__ == "__main__":
    main()
