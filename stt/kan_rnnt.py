#!/usr/bin/env python3
"""
KAN adapter with proper RNN-T loss.

RNN-T loss needs full logit grid: [B, T, U+1, V] where
  T = encoder time steps
  U = target length
  V = vocabulary size

For each (t, u) pair: joiner(encoder[t] + decoder[u]) → logits[V]

This gives proper alignment-aware gradients through the adapter.
"""
import os, csv, struct, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
from torchaudio.transforms import RNNTLoss

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

# --- Models ---

class FourierKANAdapter(nn.Module):
    def __init__(self, dim=512, K=3):
        super().__init__()
        self.dim = dim; self.K = K
        self.coeffs = nn.Parameter(torch.randn(dim, 2*K+1) * 0.001)

    def forward(self, x):
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

class DenseAdapter(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    def forward(self, x):
        return x + self.linear(x)

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
        y = y.clamp(min=0)
        mask = (y >= 0).unsqueeze(-1).float()
        emb = self.embedding(y) * mask
        x = emb.transpose(1, 2)
        x = self.conv(x)
        x = F.relu(x).squeeze(2)
        return self.proj(x)

class TransducerJoiner(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.linear = nn.Linear(512, 500)
        self.linear.weight.data = torch.tensor(weights['output_linear.weight'])
        self.linear.bias.data = torch.tensor(weights['output_linear.bias'])

    def forward(self, encoder_out, decoder_out):
        return self.linear(torch.tanh(encoder_out + decoder_out))

def compute_rnnt_logits(encoder_out, decoder, joiner, targets, blank_id=0):
    """
    Compute full RNN-T logit grid [1, T, U+1, V].
    encoder_out: [T, 512]
    targets: list of token IDs [U]
    """
    T = encoder_out.shape[0]
    U = len(targets)
    V = 500
    context_size = 2

    # Compute decoder output for each prefix [blank, blank], [blank, t0], [t0, t1], ...
    dec_inputs = []
    for u in range(U + 1):
        if u == 0:
            ctx = [blank_id, blank_id]
        elif u == 1:
            ctx = [blank_id, targets[0]]
        else:
            ctx = [targets[u-2], targets[u-1]]
        dec_inputs.append(ctx)

    dec_input_tensor = torch.tensor(dec_inputs, dtype=torch.long)  # [U+1, 2]
    dec_out = decoder(dec_input_tensor)  # [U+1, 512]

    # Compute joiner for all (t, u) pairs
    # encoder_out: [T, 512] → [T, 1, 512]
    # dec_out: [U+1, 512] → [1, U+1, 512]
    enc_expanded = encoder_out.unsqueeze(1)  # [T, 1, 512]
    dec_expanded = dec_out.unsqueeze(0)       # [1, U+1, 512]

    # Broadcast: [T, U+1, 512]
    logits = joiner.linear(torch.tanh(enc_expanded + dec_expanded))  # [T, U+1, V]

    return logits.unsqueeze(0)  # [1, T, U+1, V]

def main():
    print("=== KAN Adapter with RNN-T Loss ===\n")

    enc = ort.InferenceSession("stt/vosk-onnx/encoder.int8.onnx", providers=['CPUExecutionProvider'])

    import onnx
    from onnx import numpy_helper
    def get_weights(path):
        m = onnx.load(path)
        return {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}

    decoder = TransducerDecoder(get_weights("stt/vosk-onnx/decoder.onnx")).eval()
    joiner = TransducerJoiner(get_weights("stt/vosk-onnx/joiner.onnx")).eval()
    for p in decoder.parameters(): p.requires_grad = False
    for p in joiner.parameters(): p.requires_grad = False

    token2id, id2token = load_tokens("stt/vosk-onnx/tokens.txt")
    rnnt_loss = RNNTLoss(blank=0)

    # Prepare data
    samples = []
    with open("stt/test/farfield/manifest.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader: samples.append(row)

    N_train = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    N_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Preparing {N_train} samples...")
    data = []
    for i, s in enumerate(samples[:N_train]):
        wav_path = os.path.join("stt/test/farfield", s['path'])
        if not os.path.exists(wav_path): continue
        audio, sr = read_wav_float(wav_path)
        if audio is None: continue
        fbank = compute_fbank(audio, sr)
        if fbank.shape[0] < 50: continue  # skip very short audio
        x = fbank[np.newaxis,:,:]; xl = np.array([fbank.shape[0]], dtype=np.int64)
        try:
            eo, el = enc.run(None, {'x': x, 'x_lens': xl})
        except:
            continue
        feats = eo[0,:int(el[0]),:]

        text = s.get('transcription','').strip()
        token_ids = text_to_token_ids(text, token2id)
        if len(token_ids) == 0 or feats.shape[0] <= len(token_ids): continue

        data.append({
            'feats': torch.tensor(feats, dtype=torch.float32),
            'tokens': token_ids,
            'text': text,
        })
    print(f"Prepared {len(data)} samples\n")

    def train_adapter(adapter, name):
        optimizer = torch.optim.Adam(adapter.parameters(), lr=0.0001)

        for epoch in range(N_epochs):
            np.random.shuffle(data)
            total_loss = 0; n = 0

            for sample in data:
                feats = sample['feats']
                tokens = sample['tokens']

                adapted = adapter(feats)  # [T, 512]
                T = adapted.shape[0]
                U = len(tokens)

                # Compute full RNN-T logit grid
                logits = compute_rnnt_logits(adapted, decoder, joiner, tokens)
                # Log softmax over vocabulary dim
                log_probs = F.log_softmax(logits, dim=-1)

                targets = torch.tensor([tokens], dtype=torch.int32)
                logit_lengths = torch.tensor([T], dtype=torch.int32)
                target_lengths = torch.tensor([U], dtype=torch.int32)

                try:
                    loss = rnnt_loss(log_probs, targets, logit_lengths, target_lengths)
                except:
                    continue

                if not torch.isfinite(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n += 1

            avg = total_loss / max(n, 1)
            print(f"  [{name}] Epoch {epoch+1}/{N_epochs}: loss={avg:.3f} ({n} samples)")

        return adapter

    print("--- Training Fourier KAN adapter (K=3) ---")
    kan = train_adapter(FourierKANAdapter(512, K=3), "KAN")

    print("\n--- Training Dense adapter ---")
    dense = train_adapter(DenseAdapter(512), "Dense")

    # Test
    print(f"\n=== Testing on hard samples ===\n")
    hard = []
    with open("stt/features/hard_samples.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader: hard.append(row)

    dec_onnx = ort.InferenceSession("stt/vosk-onnx/decoder.onnx", providers=['CPUExecutionProvider'])
    join_onnx = ort.InferenceSession("stt/vosk-onnx/joiner.onnx", providers=['CPUExecutionProvider'])

    def decode_onnx(feats_np):
        T=feats_np.shape[0]; blank=0; ctx_sz=2
        ctx=[blank]*ctx_sz
        di=np.array([ctx],dtype=np.int64)
        do=dec_onnx.run(None,{'y':di})[0]
        toks=[]
        for t in range(T):
            logit=join_onnx.run(None,{'encoder_out':feats_np[t:t+1,:].astype(np.float32),'decoder_out':do})[0]
            tid=int(np.argmax(logit[0]))
            if tid!=blank:
                toks.append(tid)
                ctx=([blank]*ctx_sz+toks)[-ctx_sz:]
                di=np.array([ctx],dtype=np.int64)
                do=dec_onnx.run(None,{'y':di})[0]
        return ''.join(id2token.get(t,'') for t in toks).replace('▁',' ').strip()

    def decode_with_adapter(feats_np, adapter):
        with torch.no_grad():
            adapted = adapter(torch.tensor(feats_np, dtype=torch.float32)).numpy()
        return decode_onnx(adapted)

    n_test = 30
    base_err=kan_err=dense_err=0; total_w=0

    for i, s in enumerate(hard[:n_test]):
        wav_path=os.path.join("stt/test/farfield",s['path'])
        if not os.path.exists(wav_path): continue
        audio,sr=read_wav_float(wav_path)
        if audio is None: continue
        fbank=compute_fbank(audio,sr)
        x=fbank[np.newaxis,:,:]; xl=np.array([fbank.shape[0]],dtype=np.int64)
        eo,el=enc.run(None,{'x':x,'x_lens':xl})
        feats=eo[0,:int(el[0]),:]

        gt=s['transcription'].strip().lower()
        ref=gt.split(); total_w+=len(ref)

        t_base=decode_onnx(feats).lower()
        t_kan=decode_with_adapter(feats,kan).lower()
        t_dense=decode_with_adapter(feats,dense).lower()

        eb=edit_distance(ref,t_base.split())
        ek=edit_distance(ref,t_kan.split())
        ed=edit_distance(ref,t_dense.split())
        base_err+=eb; kan_err+=ek; dense_err+=ed

        if eb>0 and (ek!=eb or ed!=eb):
            mark_k = f"{'↓' if ek<eb else '↑' if ek>eb else '='}{abs(ek-eb)}" if ek!=eb else ""
            mark_d = f"{'↓' if ed<eb else '↑' if ed>eb else '='}{abs(ed-eb)}" if ed!=eb else ""
            print(f"[{i}] GT:    '{gt}'")
            print(f"     Base:  '{t_base}' (err={eb})")
            if ek!=eb: print(f"     KAN:   '{t_kan}' (err={ek}) {mark_k}")
            if ed!=eb: print(f"     Dense: '{t_dense}' (err={ed}) {mark_d}")
            print()

    print(f"\n=== WER on {n_test} hard samples ===")
    print(f"Baseline:      {base_err/total_w*100:.1f}% ({base_err}/{total_w})")
    print(f"Fourier KAN:   {kan_err/total_w*100:.1f}% ({kan_err}/{total_w})")
    print(f"Dense adapter: {dense_err/total_w*100:.1f}% ({dense_err}/{total_w})")

if __name__ == "__main__":
    main()
