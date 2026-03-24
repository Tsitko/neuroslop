#!/usr/bin/env python3
"""
Find samples with rare words where vosk model is most likely to fail.
Strategy: build word frequency from all transcriptions, find samples with rarest words.
"""
import csv
import os
from collections import Counter

def main():
    manifest = "stt/test/farfield/manifest.tsv"

    # Also check train manifest if available
    train_manifests = []
    train_dir = "stt/train/farfield"
    if os.path.exists(train_dir):
        for f in os.listdir(train_dir):
            if f.endswith('.tsv'):
                train_manifests.append(os.path.join(train_dir, f))

    # Build word frequency from ALL available transcriptions
    word_freq = Counter()
    all_samples = []

    with open(manifest, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            text = row.get('transcription', row.get('transcription_ru', ''))
            words = text.lower().split()
            word_freq.update(words)
            all_samples.append({
                'path': row['path'],
                'text': text,
                'words': words,
                'duration': float(row.get('duration', 0)),
            })

    for mp in train_manifests:
        with open(mp, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                text = row.get('transcription', row.get('transcription_ru', ''))
                word_freq.update(text.lower().split())

    total_words = sum(word_freq.values())
    vocab_size = len(word_freq)
    print(f"Vocabulary: {vocab_size} unique words, {total_words} total")
    print(f"Test samples: {len(all_samples)}")

    # Show most common and rarest words
    print(f"\nTop 20 most common:")
    for word, count in word_freq.most_common(20):
        print(f"  {word}: {count}")

    print(f"\nTop 20 rarest (appearing once):")
    rare_words = [w for w, c in word_freq.items() if c == 1]
    print(f"  {len(rare_words)} words appear only once")
    for w in sorted(rare_words)[:20]:
        print(f"  {w}")

    # Score each sample by rarity of its words
    # Score = sum of 1/freq for each word (rarer words = higher score)
    for s in all_samples:
        rarity = sum(1.0 / word_freq[w] for w in s['words'] if w in word_freq)
        s['rarity'] = rarity
        s['min_freq'] = min((word_freq[w] for w in s['words']), default=0)
        s['has_rare'] = any(word_freq[w] <= 3 for w in s['words'])
        # Length bonus — longer utterances are harder
        s['length'] = len(s['words'])

    # Sort by rarity score (highest = hardest)
    hard_samples = sorted(all_samples, key=lambda x: x['rarity'], reverse=True)

    print(f"\n=== Top 50 hardest samples (rarest words) ===\n")
    for i, s in enumerate(hard_samples[:50]):
        rare = [f"{w}({word_freq[w]})" for w in s['words'] if word_freq[w] <= 5]
        print(f"  [{i}] rarity={s['rarity']:.1f} len={s['length']} "
              f"rare_words=[{', '.join(rare)}]")
        print(f"       \"{s['text']}\"")
        print(f"       {s['path']}")

    # Save hard samples list
    out_path = "stt/features/hard_samples.tsv"
    os.makedirs("stt/features", exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("idx\tpath\ttranscription\trarity\tmin_freq\tlength\n")
        for i, s in enumerate(hard_samples[:200]):
            f.write(f"{i}\t{s['path']}\t{s['text']}\t{s['rarity']:.2f}\t{s['min_freq']}\t{s['length']}\n")

    print(f"\nSaved top 200 hard samples to {out_path}")

    # Stats
    samples_with_rare = sum(1 for s in all_samples if s['has_rare'])
    print(f"\nSamples with rare words (freq≤3): {samples_with_rare}/{len(all_samples)} ({samples_with_rare*100//len(all_samples)}%)")

if __name__ == "__main__":
    main()
