"""Parity tests for phoneme normalization, ported from Muraja's NormalizationTests.swift."""

import pytest

from tadabur.normalization import (
    map_to_original,
    map_to_original_end,
    normalize_phonemes,
)

# Tajweed variants that fold onto their base consonant before grouping.
_FOLD = {"\u06FE": "\u0645", "\u06BA": "\u0646"}


def _chunck_oracle(text: str) -> str:
    """Ground truth from quran-transcript: fold tajweed variants, then collapse
    each space-separated word with ``chunck_phonemes``. Used only to cross-check
    the cases where Swift's normalization *agrees* with quran-transcript (no
    shadda-expansion divergence); it deliberately collapses shadda runs, so it is
    NOT a parity oracle for the ``.balanced`` scorer. Skips if quran-transcript is
    not installed."""
    sifa = pytest.importorskip("quran_transcript.phonetics.sifa")
    folded = "".join(_FOLD.get(ch, ch) for ch in text)
    return " ".join(
        "".join(group[0] for group in sifa.chunck_phonemes(word))
        for word in folded.split(" ")
    )


def test_normalize_empty_string():
    result = normalize_phonemes("")
    assert result.normalized == ""
    assert result.offset_map == []


def test_normalize_simple_consonants():
    result = normalize_phonemes("بتث")
    assert result.normalized == "بتث"
    assert len(result.offset_map) == 3
    assert all(end - start == 1 for start, end in result.offset_map)


def test_normalize_repeated_consonants():
    result = normalize_phonemes("ببب")
    assert result.normalized == "ب"
    assert result.offset_map == [(0, 3)]


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Cases where Swift's group-collapse agrees with quran-transcript's
        # chunck_phonemes (no bare-core-then-same-core-with-combining pattern),
        # so both the hand-written expectation and the oracle must agree.
        # A diacritic between cores keeps them in separate groups: بَ | بُ → بب.
        ("بَبُ", "بب"),
        # ں folds to ن, the bare run collapses, madd ۥ stays: مِ | ںںں | جُ | ۥۥ.
        ("مِںںںجُۥۥ", "منجۥ"),
        # Non-combining ڇ acts as a trailing residual: جڇ → ج.
        ("نَجڇعَ", "نجع"),
    ],
)
def test_normalize_agrees_with_chunck_oracle(raw, expected):
    got = normalize_phonemes(raw).normalized
    assert got == expected
    assert got == _chunck_oracle(raw)


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Shadda-style expansion: a bare core followed by the SAME core carrying a
        # combining mark. Swift breaks before consuming the diacritic-bearing
        # cluster, so the doubled core is preserved — this is exactly where a
        # faithful port must diverge from chunck_phonemes (which would collapse to
        # رب / ب). The downstream word scorer's shaddahSuppression, not this
        # normalization, neutralises the extra core.
        ("رَببُ", "ربب"),   # رَ | ب | بُ → ربب  (not رب)
        ("ببَ", "بب"),       # ب | بَ → بب        (not ب)
        ("للَ", "لل"),       # ل | لَ → لل        (not ل)
        ("ءَننَ", "ءنن"),   # ءَ | ن | نَ → ءنن  (not ءن)
        ("هُممممِن", "هممن"),  # هُ | ممم | مِ | ن → هممن (not همن)
    ],
)
def test_normalize_keeps_shadda_expansion_doubled(raw, expected):
    # Swift-faithful: doubled cores are preserved (parity with .balanced scores).
    assert normalize_phonemes(raw).normalized == expected


def test_normalize_consonant_with_diacritic():
    # "بَ" is one grapheme cluster (base + combining fatha) → one group.
    result = normalize_phonemes("بَ")
    assert result.normalized == "ب"
    assert result.offset_map == [(0, 1)]


def test_normalize_spaces_preserved():
    result = normalize_phonemes("ب ت")
    assert result.normalized == "ب ت"
    assert len(result.offset_map) == 3
    assert result.offset_map[1] == (1, 2)


def test_normalize_tajweed_mim():
    # ۾ (ghunna mim, U+06FE) → م (regular mim).
    assert normalize_phonemes("\u06FE").normalized == "م"


def test_normalize_tajweed_nun():
    # ں (ghunna nun, U+06BA) → ن (regular nun).
    assert normalize_phonemes("\u06BA").normalized == "ن"


def test_normalize_tajweed_folds_into_group():
    # A ghunna mim adjacent to a regular mim collapses into one group.
    assert normalize_phonemes("م\u06FEم").normalized == "م"


def test_normalize_mixed_groups():
    result = normalize_phonemes("ببتتثث")
    assert result.normalized == "بتث"
    assert result.offset_map == [(0, 2), (2, 4), (4, 6)]


def test_normalize_stray_residual_skipped():
    assert normalize_phonemes("\u064E").normalized == ""


def test_normalize_madd_markers():
    result = normalize_phonemes("بۦۦۦت")
    assert result.normalized == "بۦت"
    assert len(result.offset_map) == 3


def test_normalize_word_boundaries():
    # ببَ → بب (ب | بَ) and تتِ → تت (ت | تِ): shadda expansion stays doubled on
    # both sides of the space, which is preserved.
    result = normalize_phonemes("ببَ تتِ")
    assert result.normalized == "بب تت"


def test_map_to_original_basic():
    offset_map = [(0, 2), (2, 4), (4, 6)]
    assert map_to_original(0, offset_map) == 0
    assert map_to_original(1, offset_map) == 2
    assert map_to_original(2, offset_map) == 4


def test_map_to_original_negative_index():
    assert map_to_original(-1, [(0, 2), (2, 4)]) == 0


def test_map_to_original_beyond_range():
    assert map_to_original(5, [(0, 2), (2, 4)]) == 4


def test_map_to_original_empty_map():
    assert map_to_original(0, []) == 0


def test_map_to_original_end_basic():
    offset_map = [(0, 2), (2, 4), (4, 6)]
    assert map_to_original_end(1, offset_map) == 2
    assert map_to_original_end(2, offset_map) == 4
    assert map_to_original_end(3, offset_map) == 6


def test_map_to_original_end_zero():
    assert map_to_original_end(0, [(0, 2), (2, 4)]) == 0


def test_map_to_original_end_beyond_range():
    assert map_to_original_end(5, [(0, 2), (2, 4)]) == 4
