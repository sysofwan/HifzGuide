#!/usr/bin/env python3
"""
Generate ayah_phonemes.json using the quran-transcript library.

Uses quran_phonetizer() to produce phonetic transcriptions from Uthmani text
for all 6236 ayahs. Falls back to hardcoded phonemes for the 8 ayahs where the
library has a bug (leen madd handling on ayahs ending in sukoon).

Requires: pip install quran-transcript

Usage:
  python3 generate_phonemes.py
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_PATH = DATA_DIR / "ayah_phonemes.json"

# Hafs recitation, matching the reference labels the Muaalem model was trained on.
HAFS_MOSHAF = dict(
    rewaya="hafs",
    madd_monfasel_len=4,
    madd_mottasel_len=4,
    madd_mottasel_waqf=6,
    madd_aared_len=2,
)

# Ayahs where quran-transcript's phonetizer raises (leen madd on a final sukoon).
# Phonemes extracted from the muaalem-annotated-v3 dataset with word boundaries
# manually verified. This is the canonical source of the 8-ayah fallback; other
# generators (e.g. generate_quran_db.py) import it from here.
FALLBACK_PHONEMES: dict[str, str] = {
    "55:17": "رَببُ لمَشرِقَينِ وَرَببُ لمَغرِبَين",
    "90:8": "ءَلَم نَجڇعَللَهُۥۥ عَينَين",
    "90:9": "وَلِسَاانَوووَشَفَتَين",
    "90:10": "وَهَدَينَااهُ ننننَجڇدَين",
    "106:1": "لِءِۦۦلَاافِ قُرَيش",
    "106:2": "ءِۦۦلَاافِهِم رِحلَتَ ششِتَااااءِ وَصصَيف",
    "106:3": "فَليَعبُدُۥۥ رَببَ هَااذَ لبَيت",
    "106:4": "ءَللَذِۦۦ ءَطڇعَمَهُممممِںںںجُۥۥعِوووَءَاامَنَهُممممِن خَوف",
}


def generate_reference_phonemes() -> dict[str, str]:
    """Phonetize every ayah (1..6236) to its Hafs reference phoneme string.

    Returns a dict keyed by ``"surah:ayah"``. The 8 ayahs the phonetizer cannot
    handle fall back to FALLBACK_PHONEMES so the output stays complete.
    """
    from quran_transcript import Aya, quran_phonetizer
    from quran_transcript.phonetics.moshaf_attributes import MoshafAttributes

    moshaf = MoshafAttributes(**HAFS_MOSHAF)

    phonemes: dict[str, str] = {}
    for sura in range(1, 115):
        num_ayat = Aya(sura, 1).get().num_ayat_in_sura
        for ayah in range(1, num_ayat + 1):
            key = f"{sura}:{ayah}"
            try:
                seg = Aya(sura, ayah).get()
                phonemes[key] = quran_phonetizer(seg.uthmani, moshaf).phonemes
            except Exception:
                if key in FALLBACK_PHONEMES:
                    phonemes[key] = FALLBACK_PHONEMES[key]
                else:
                    print(f"  WARNING: {key} failed with no fallback!", file=sys.stderr)
    return phonemes


def main():
    phonemes = generate_reference_phonemes()
    fallbacks = [k for k in FALLBACK_PHONEMES if phonemes.get(k) == FALLBACK_PHONEMES[k]]
    print(f"Generated {len(phonemes)} ayahs (fallbacks: {fallbacks})")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(phonemes, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
