#!/usr/bin/env python3
"""
Run vosk ONNX on hard samples (rare words) and measure WER.
"""
import csv
import os
import struct
import numpy as np
import sherpa_onnx

def read_wav_float(path):
    """Read IEEE Float WAV file."""
    with open(path, 'rb') as f:
        f.read(4)  # RIFF
        f.read(4)  # size
        f.read(4)  # WAVE
        sr = 16000
        audio = None
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]
            if chunk_id == b'fmt ':
                fmt = f.read(chunk_size)
                sr = struct.unpack('<I', fmt[4:8])[0]
            elif chunk_id == b'data':
                audio = np.frombuffer(f.read(chunk_size), dtype=np.float32)
            else:
                f.read(chunk_size)
    return audio, sr

def edit_distance(ref, hyp):
    """Compute word-level edit distance."""
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

def main():
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder="stt/vosk-onnx/encoder.int8.onnx",
        decoder="stt/vosk-onnx/decoder.onnx",
        joiner="stt/vosk-onnx/joiner.onnx",
        tokens="stt/vosk-onnx/tokens.txt",
        num_threads=4,
        decoding_method="greedy_search",
        sample_rate=16000,
    )

    # Load hard samples
    hard_samples = []
    with open("stt/features/hard_samples.tsv", 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            hard_samples.append(row)

    print(f"Testing {len(hard_samples)} hard samples...\n")

    total_ref_words = 0
    total_errors = 0
    error_samples = []

    for i, s in enumerate(hard_samples):
        wav_path = os.path.join("stt/test/farfield", s['path'])
        if not os.path.exists(wav_path):
            continue

        audio, sr = read_wav_float(wav_path)
        if audio is None:
            continue

        stream = recognizer.create_stream()
        stream.accept_waveform(sr, audio)
        recognizer.decode_stream(stream)
        pred = stream.result.text.strip().lower()
        gt = s['transcription'].strip().lower()

        ref_words = gt.split()
        hyp_words = pred.split()
        errors = edit_distance(ref_words, hyp_words)
        total_ref_words += len(ref_words)
        total_errors += errors

        if errors > 0:
            error_samples.append({
                'gt': gt,
                'pred': pred,
                'errors': errors,
                'ref_len': len(ref_words),
                'wer': errors / len(ref_words) * 100,
                'path': s['path'],
            })

        if i < 10 or (errors > 0 and len(error_samples) <= 30):
            marker = "✓" if errors == 0 else f"✗ ({errors} errors)"
            print(f"  [{i}] {marker}")
            if errors > 0:
                print(f"       GT:   '{gt}'")
                print(f"       PRED: '{pred}'")

    wer = total_errors / max(total_ref_words, 1) * 100
    print(f"\n=== Results on {len(hard_samples)} hard samples ===")
    print(f"WER: {wer:.1f}% ({total_errors} errors / {total_ref_words} words)")
    print(f"Samples with errors: {len(error_samples)}/{len(hard_samples)}")

    # Show worst errors
    error_samples.sort(key=lambda x: x['wer'], reverse=True)
    print(f"\n=== Top 20 worst errors ===\n")
    for e in error_samples[:20]:
        print(f"  WER={e['wer']:.0f}% ({e['errors']}/{e['ref_len']})")
        print(f"    GT:   '{e['gt']}'")
        print(f"    PRED: '{e['pred']}'")
        print()

if __name__ == "__main__":
    main()
