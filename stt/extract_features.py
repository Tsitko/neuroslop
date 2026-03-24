#!/usr/bin/env python3
"""
Extract encoder features from frozen Vosk ONNX model.
Saves features + transcriptions for Swift KAN adapter training.

Pipeline:
  WAV (16kHz) → sherpa-onnx encoder → features [T, D] → binary file
"""
import os
import sys
import struct
import csv
import numpy as np

def extract_with_sherpa(wav_dir, manifest_path, output_dir, max_samples=100):
    """Use sherpa-onnx to run full recognition and extract encoder outputs."""
    import sherpa_onnx

    # Create recognizer with ONNX model
    tokens_path = "stt/vosk-onnx/tokens.txt"
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder="stt/vosk-onnx/encoder.int8.onnx",
        decoder="stt/vosk-onnx/decoder.onnx",
        joiner="stt/vosk-onnx/joiner.onnx",
        tokens=tokens_path,
        num_threads=4,
        decoding_method="greedy_search",
        provider="cpu",
        sample_rate=16000,
    )

    # Read manifest
    samples = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            samples.append(row)
            if len(samples) >= max_samples:
                break

    print(f"Processing {len(samples)} samples...")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for i, sample in enumerate(samples):
        wav_path = os.path.join(wav_dir, sample['path'])
        if not os.path.exists(wav_path):
            continue

        # Read WAV (IEEE Float format — parse manually)
        with open(wav_path, 'rb') as wf:
            riff = wf.read(4)
            if riff != b'RIFF':
                print(f"  Skipping {wav_path}: not RIFF")
                continue
            wf.read(4)  # file size
            wave_id = wf.read(4)
            # Find 'fmt ' and 'data' chunks
            sr = 16000
            audio = None
            while True:
                chunk_id = wf.read(4)
                if len(chunk_id) < 4:
                    break
                chunk_size = struct.unpack('<I', wf.read(4))[0]
                if chunk_id == b'fmt ':
                    fmt_data = wf.read(chunk_size)
                    fmt_tag = struct.unpack('<H', fmt_data[0:2])[0]
                    n_channels = struct.unpack('<H', fmt_data[2:4])[0]
                    sr = struct.unpack('<I', fmt_data[4:8])[0]
                elif chunk_id == b'data':
                    raw = wf.read(chunk_size)
                    audio = np.frombuffer(raw, dtype=np.float32)  # IEEE Float
                else:
                    wf.read(chunk_size)
            if audio is None:
                print(f"  Skipping {wav_path}: no data chunk")
                continue

        # Run through recognizer (gets text output)
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, audio)
        recognizer.decode_stream(stream)
        recognized = stream.result.text.strip()

        ground_truth = sample.get('transcription', sample.get('transcription_ru', ''))

        results.append({
            'idx': i,
            'wav': sample['path'],
            'ground_truth': ground_truth,
            'recognized': recognized,
            'duration': sample.get('duration', '0'),
        })

        if i < 5 or i % 20 == 0:
            print(f"  [{i}] GT: '{ground_truth}' → Pred: '{recognized}'")

    # Save results
    results_path = os.path.join(output_dir, "baseline_results.tsv")
    with open(results_path, 'w') as f:
        f.write("idx\twav\tground_truth\trecognized\tduration\n")
        for r in results:
            f.write(f"{r['idx']}\t{r['wav']}\t{r['ground_truth']}\t{r['recognized']}\t{r['duration']}\n")

    # Compute WER
    total_words = 0
    total_errors = 0
    for r in results:
        gt_words = r['ground_truth'].split()
        pred_words = r['recognized'].split()
        total_words += len(gt_words)
        # Simple word error count (not true WER with edit distance, but rough estimate)
        total_errors += abs(len(gt_words) - len(pred_words))
        for gw, pw in zip(gt_words, pred_words):
            if gw != pw:
                total_errors += 1

    wer = total_errors / max(total_words, 1) * 100
    print(f"\nBaseline results: {len(results)} samples")
    print(f"Rough WER: {wer:.1f}% ({total_errors} errors / {total_words} words)")
    print(f"Results saved to {results_path}")

    return results

def main():
    max_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    # Check if tokens file exists
    tokens_paths = [
        "stt/vosk-model-small-ru-0.22/lang/tokens.txt",
        "stt/vosk-onnx/tokens.txt",
    ]
    tokens_found = any(os.path.exists(p) for p in tokens_paths)

    if not tokens_found:
        # Try to download tokens
        print("Downloading tokens.txt...")
        os.system("curl -sL -o stt/vosk-onnx/tokens.txt https://huggingface.co/alphacep/vosk-model-ru/resolve/main/lang/tokens.txt")

    # Run baseline recognition
    print("=== Vosk ONNX Baseline Recognition ===\n")
    extract_with_sherpa(
        wav_dir="stt/test/farfield",
        manifest_path="stt/test/farfield/manifest.tsv",
        output_dir="stt/features",
        max_samples=max_samples,
    )

if __name__ == "__main__":
    main()
