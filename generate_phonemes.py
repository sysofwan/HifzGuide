#!/usr/bin/env python3
"""
Generate ayah_phonemes.json using the quran-transcript library.

Uses quran_phonetizer() to produce phonetic transcriptions from Uthmani text
for all 6236 ayahs. Falls back to existing phonemes for ayahs where the
library has bugs (leen madd handling in 8 ayahs).

Requires: pip install quran-transcript

Usage:
  python3 generate_phonemes.py
"""

import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_PATH = DATA_DIR / "ayah_phonemes.json"


def main():
    from quran_transcript import Aya, quran_phonetizer
    from quran_transcript.phonetics.moshaf_attributes import MoshafAttributes

    moshaf = MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=4,
        madd_mottasel_len=4,
        madd_mottasel_waqf=6,
        madd_aared_len=2,
    )

    # Load existing phonemes as fallback for library bugs
    old_phonemes: dict[str, str] = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            old_phonemes = json.load(f)

    phonemes: dict[str, str] = {}
    failed: list[str] = []
    start = time.time()

    for sura in range(1, 115):
        seg0 = Aya(sura, 1).get()
        num_ayat = seg0.num_ayat_in_sura
        for ayah in range(1, num_ayat + 1):
            key = f"{sura}:{ayah}"
            try:
                seg = Aya(sura, ayah).get()
                result = quran_phonetizer(seg.uthmani, moshaf)
                phonemes[key] = result.phonemes
            except Exception:
                if key in old_phonemes:
                    phonemes[key] = old_phonemes[key]
                    failed.append(key)
                else:
                    print(f"  WARNING: {key} failed with no fallback!", file=sys.stderr)
        if sura % 20 == 0:
            print(f"  Processed sura {sura}...")

    elapsed = time.time() - start
    print(f"Generated {len(phonemes)} ayahs in {elapsed:.1f}s")
    if failed:
        print(f"Fallbacks ({len(failed)}): {failed}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(phonemes, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
