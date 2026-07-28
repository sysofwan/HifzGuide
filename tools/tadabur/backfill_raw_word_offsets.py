"""Backfill ``raw_word_offsets`` onto a segment manifest written before that field (ADR-0003).

``tadabur.segment_score`` originally emitted only ``word_offsets`` — per-word boundaries into
the ``.balanced``-normalized ``reference_phonemes``. Training slices *that* string, which is
vowel-stripped, so every fatha/damma/kasra vanished from the CTC target even though ADR-0003
names the tashkeel-bearing ``raw_reference_phonemes`` as *the* label. The scorer now emits
``raw_word_offsets`` (boundaries into the raw string) alongside, and
:func:`training.windowed_labels.read_segments` requires it.

Regenerating a corpus-scale manifest means re-running the GPU decode (hours). The offsets,
though, are a pure function of the segment's Uthmani words: re-running the phonetizer
(:func:`tadabur.waqf_segments.hafs_segment_reference`, CPU-only) reproduces both the realized
reference and its per-word offsets. This tool therefore re-derives the offsets and **verifies**
them against the stored string before writing: if the recomputed phonemes differ from the row's
``raw_reference_phonemes`` by even one character, the offsets would index a string the manifest
does not contain, so the row is reported and the backfill fails rather than writing a subtly
wrong label boundary. Rows that already carry ``raw_word_offsets`` are passed through unchanged.

Deterministic and idempotent: the phonetizer is a pure function, so re-running reproduces the
same file byte for byte. Writes to a new path; the input is never modified in place.

Usage:
  python -m tadabur.backfill_raw_word_offsets \\
      --manifest audit_run/seg_v21/manifest.jsonl \\
      --out audit_run/seg_v21/manifest.raw.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def backfill_row(row: dict, segment_reference) -> dict:
    """``row`` with ``raw_word_offsets`` added, verified against its stored raw reference.

    Raises ``ValueError`` when the recomputed phonetization does not reproduce the row's
    ``raw_reference_phonemes`` exactly — the offsets would then index a different string.
    """
    if "raw_word_offsets" in row:
        return row
    words = row["uthmani"].split()
    phonemes, offsets = segment_reference(words)
    stored = row["raw_reference_phonemes"]
    if phonemes != stored:
        raise ValueError(
            f"segment {row['clip_audio_filename']}#{row['segment_index']} "
            f"({row['surah_ayah']} words {row['word_start']}-{row['word_end']}): "
            f"recomputed phonetization {phonemes!r} != stored {stored!r}; the offsets "
            "would index a string this manifest does not contain."
        )
    if len(offsets) != len(words) + 1:
        raise ValueError(
            f"segment {row['clip_audio_filename']}#{row['segment_index']}: got "
            f"{len(offsets)} offsets for {len(words)} words (expected {len(words) + 1})."
        )
    # String equality alone does not prove the *boundaries* are sane: the phonetizer emits
    # the string and the mapping as separate artifacts, so a mapping change could reproduce
    # the same phonemes with different (or crossed) cuts. Assert the offsets really do
    # partition that exact string, left to right.
    if list(offsets) != sorted(offsets):
        raise ValueError(
            f"segment {row['clip_audio_filename']}#{row['segment_index']}: word offsets "
            f"{list(offsets)} are not monotonic; the word slices would cross."
        )
    if offsets[0] != 0 or offsets[-1] != len(phonemes):
        raise ValueError(
            f"segment {row['clip_audio_filename']}#{row['segment_index']}: offsets span "
            f"{offsets[0]}..{offsets[-1]} but the reference is {len(phonemes)} chars; the "
            "word slices would not cover the whole reference."
        )
    return {**row, "raw_word_offsets": list(offsets)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; refusing to overwrite.")

    from .waqf_segments import hafs_segment_reference

    segment_reference = hafs_segment_reference()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with open(args.manifest, encoding="utf-8") as src, \
            open(args.out, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "raw_word_offsets" in row:
                skipped += 1
            else:
                row = backfill_row(row, segment_reference)
                written += 1
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"backfilled {written} row(s), {skipped} already had offsets -> {args.out}")


if __name__ == "__main__":
    main()
