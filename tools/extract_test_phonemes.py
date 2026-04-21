#!/usr/bin/env python3
"""
Extract phonemes from the HuggingFace muaalem-annotated-v3 dataset.

Downloads only the text columns (no audio) from one moshaf config
and saves ayah-level phonemes to a JSON file for testing.

Usage:
    pip install datasets
    python extract_test_phonemes.py [--config moshaf_0.0] [--output test_phonemes.json]
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
OUTPUT_PATH = SCRIPT_DIR / "test_phonemes.json"


def main():
    parser = argparse.ArgumentParser(description="Extract phonemes from HF dataset")
    parser.add_argument("--config", default="moshaf_0.0",
                        help="Dataset config(s), comma-separated or 'all' (default: moshaf_0.0)")
    parser.add_argument("--output", default=str(OUTPUT_PATH),
                        help="Output JSON path")
    parser.add_argument("-n", type=int, default=0,
                        help="Use first N configs (0=use --config as-is)")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("Please install pandas: pip install pandas pyarrow")
        sys.exit(1)

    # Download parquet files directly (text columns only, no audio)
    from huggingface_hub import HfApi
    api = HfApi()

    # Resolve config list
    all_files = api.list_repo_files("obadx/muaalem-annotated-v3", repo_type="dataset")
    all_configs = sorted(set(
        f.split("/")[0] for f in all_files
        if f.startswith("moshaf_") and f.endswith(".parquet") and "metadata" not in f
    ))

    if args.config == "all":
        configs = all_configs
    elif args.n > 0:
        configs = all_configs[:args.n]
    else:
        configs = [c.strip() for c in args.config.split(",")]

    print(f"Extracting from {len(configs)} config(s): {', '.join(configs)}")
    print()

    results = []
    seen = set()
    total = 0
    skipped = 0

    for config in configs:
        parquet_files = [f for f in all_files if f.startswith(config + "/") and f.endswith(".parquet")]
        print(f"Config '{config}': {len(parquet_files)} parquet files")

        for pf in sorted(parquet_files):
            url = f"https://huggingface.co/datasets/obadx/muaalem-annotated-v3/resolve/main/{pf}"
            print(f"  Reading {pf}...")

            columns = ["has_quran", "start_span", "end_span", "phonemes",
                        "match_ratio", "reciter_english_name"]
            try:
                df = pd.read_parquet(url, columns=columns)
            except Exception:
                df = pd.read_parquet(url)
                df = df[columns]

            for _, row in df.iterrows():
                total += 1

                if not row.get("has_quran", False):
                    skipped += 1
                    continue

                start_span = row.get("start_span")
                end_span = row.get("end_span")
                phonemes = row.get("phonemes", "")

                if start_span is None or end_span is None or not phonemes:
                    skipped += 1
                    continue

                surah = int(start_span["sura_idx"])
                ayah_start = int(start_span["aya_idx"])
                ayah_end = int(end_span["aya_idx"])
                match_ratio = float(row.get("match_ratio", 0))

                results.append({
                    "surah": surah,
                    "ayah_start": ayah_start,
                    "ayah_end": ayah_end,
                    "word_start": int(start_span["imlaey"]),
                    "word_end": int(end_span["imlaey"]),
                    "phonemes": phonemes,
                    "match_ratio": match_ratio,
                    "reciter": row.get("reciter_english_name", ""),
                    "config": config,
                })

    # Sort by config, surah, ayah for grouping
    results.sort(key=lambda r: (r["config"], r["surah"], r["ayah_start"]))

    # Concatenate segments that belong to the same (config, surah, ayah)
    # using word_start/word_end spans to place phonemes correctly
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        if r["ayah_start"] != r["ayah_end"]:
            continue
        key = (r["config"], r["surah"], r["ayah_start"])
        grouped[key].append(r)

    merged = []
    for key, segments in grouped.items():
        config, surah, ayah = key

        if len(segments) == 1:
            s = segments[0]
            merged.append({
                "surah": surah,
                "ayah_start": ayah,
                "ayah_end": ayah,
                "phonemes": s["phonemes"],
                "match_ratio": s["match_ratio"],
                "reciter": s["reciter"],
                "config": config,
            })
            continue

        # Sort segments by word_start position
        segments.sort(key=lambda s: s["word_start"])

        # Stitch segments together using spans to avoid overlap.
        # Each segment covers [word_start, word_end). When segments overlap,
        # use the later segment for the overlapping region (it's a fresh
        # recording of those words).
        #
        # Strategy: process segments in order, tracking the highest word_end
        # reached so far. For each segment, only take the portion that extends
        # beyond what we already have.
        combined_parts = []
        covered_up_to = 0  # word index we've covered so far

        for seg in segments:
            seg_words = seg["phonemes"].split()
            w_start = seg["word_start"]
            w_end = seg["word_end"]
            span_len = w_end - w_start

            if w_end <= covered_up_to:
                # Entirely within already-covered range, skip
                continue

            if w_start >= covered_up_to:
                # No overlap — take all words from this segment
                combined_parts.extend(seg_words)
            else:
                # Partial overlap — skip the words that overlap
                overlap = covered_up_to - w_start
                # The segment has seg_words for span_len ref words
                # Skip 'overlap' ref words worth of phoneme words
                if len(seg_words) == span_len:
                    # Perfect mapping: skip exactly 'overlap' phoneme words
                    combined_parts.extend(seg_words[overlap:])
                elif len(seg_words) < span_len:
                    # Fewer phoneme words (fusion) — estimate skip ratio
                    skip = int(overlap * len(seg_words) / span_len)
                    combined_parts.extend(seg_words[skip:])
                else:
                    # More words than span — skip proportionally
                    skip = int(overlap * len(seg_words) / span_len)
                    combined_parts.extend(seg_words[skip:])

            covered_up_to = max(covered_up_to, w_end)

        combined_phonemes = " ".join(combined_parts)
        avg_ratio = sum(s["match_ratio"] for s in segments) / len(segments)

        merged.append({
            "surah": surah,
            "ayah_start": ayah,
            "ayah_end": ayah,
            "phonemes": combined_phonemes,
            "match_ratio": avg_ratio,
            "reciter": segments[0]["reciter"],
            "config": config,
        })

    merged.sort(key=lambda r: (r["config"], r["surah"], r["ayah_start"]))

    # Deduplicate: entries with the same (surah, ayah, phonemes) across configs
    # are collapsed into a single entry with a configs list.
    dedup_key = defaultdict(list)
    for r in merged:
        key = (r["surah"], r["ayah_start"], r["ayah_end"], r["phonemes"])
        dedup_key[key].append(r["config"])

    # Output as position-based arrays: [surah, ayah_start, ayah_end, phonemes, configs]
    # Config values have the "moshaf_" prefix stripped.
    deduped = []
    for (surah, ayah_start, ayah_end, phonemes), cfgs in dedup_key.items():
        stripped = sorted(set(c.replace("moshaf_", "") for c in cfgs))
        deduped.append([surah, ayah_start, ayah_end, phonemes, stripped])

    deduped.sort(key=lambda r: (r[0], r[1], r[2]))

    print(f"\nProcessed {total} rows total from {len(configs)} config(s)")
    print(f"  Raw segments: {len(results)}")
    print(f"  After merging: {len(merged)} ayah entries")
    print(f"  After dedup:   {len(deduped)} unique phoneme entries")
    print(f"  Skipped: {skipped} (non-quran, missing data)")

    # Save as compact JSON (no indent — arrays are already compact)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False)

    print(f"Saved to {args.output}")

    # Show sample
    if merged:
        r = merged[0]
        print(f"\nSample: {r['surah']}:{r['ayah_start']} — {r['phonemes'][:60]}...")


if __name__ == "__main__":
    main()
