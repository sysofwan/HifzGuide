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
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_PATH = DATA_DIR / "ayah_phonemes.json"

# Ayah count per surah in the Hafs mushaf (index 0 == surah 1 .. index 113 ==
# surah 114). This is the canonical numbering of the mushaf and the single source
# of truth for the reference key set; derived from the committed Hafs word-by-word
# data (data/qpc-hafs-word-by-word.json) and fixed thereafter.
HAFS_AYAH_COUNTS: tuple[int, ...] = (
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128,
    111, 110, 98, 135, 112, 78, 118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73,
    54, 45, 83, 182, 88, 75, 85, 54, 53, 89, 59, 37, 35, 38, 29, 18, 45, 60,
    49, 62, 55, 78, 96, 29, 22, 24, 13, 14, 11, 11, 18, 12, 12, 30, 52, 52, 44,
    28, 28, 20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25, 22, 17, 19, 26, 30,
    20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3, 6, 3, 5, 4,
    5, 6,
)

# Total ayat in the mushaf (Hafs). A complete reference set must have exactly this
# many keys; a smaller count means an ayah was silently dropped.
TOTAL_AYAT = sum(HAFS_AYAH_COUNTS)


def expected_ayah_keys() -> frozenset[str]:
    """The canonical ``surah:ayah`` key set for the whole Hafs mushaf (6236 keys).

    Derived cheaply from ``HAFS_AYAH_COUNTS`` (no quran-transcript needed), so it
    can validate a warm cache's exact key set: a full-size cache that is missing a
    real ayah while carrying an extra key must be rejected, not trusted.
    """
    return frozenset(
        f"{sura}:{ayah}"
        for sura, count in enumerate(HAFS_AYAH_COUNTS, start=1)
        for ayah in range(1, count + 1)
    )

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
    handle raise ``KeyError`` and fall back to FALLBACK_PHONEMES; any other
    failure is unexpected and propagates so a partial cache is never produced.
    The result is asserted to be the exact canonical key set (``expected_ayah_keys``)
    before returning, so a dropped or spurious ayah fails loudly.
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
            except KeyError:
                # The phonetizer raises KeyError on the 8 leen-madd-on-sukoon
                # ayahs; anything else with this key is unexpected — surface it.
                if key not in FALLBACK_PHONEMES:
                    raise
                phonemes[key] = FALLBACK_PHONEMES[key]

    expected = expected_ayah_keys()
    if phonemes.keys() != expected:
        missing = sorted(expected - phonemes.keys())
        extra = sorted(phonemes.keys() - expected)
        raise RuntimeError(
            f"reference key set mismatch: missing={missing[:5]} "
            f"extra={extra[:5]} (missing {len(missing)}, extra {len(extra)})"
        )
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
