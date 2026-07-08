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
    each space-separated word with ``chunck_phonemes`` (the ``.balanced`` scorer's
    canonical grouping), keeping the first (core) char of each group. Word
    boundaries are preserved. Skips if quran-transcript is not installed."""
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
        # Shadda-style run + trailing diacritic collapses to one core (the review's
        # ربب regression): رَ | ببُ → رب, not ربب.
        ("رَببُ", "رب"),
        ("ءَننَ", "ءن"),
        # A diacritic between cores keeps them in separate groups: بَ | بُ → بب.
        ("بَبُ", "بب"),
        # ں folds to ن, the run collapses, madd ۥ stays: مِ | ںںں | جُ | ۥۥ.
        ("مِںںںجُۥۥ", "منجۥ"),
        ("هُممممِن", "همن"),
        # Non-combining ڇ acts as a trailing residual: جڇ → ج.
        ("نَجڇعَ", "نجع"),
    ],
)
def test_normalize_collapses_repetition_like_scorer(raw, expected):
    got = normalize_phonemes(raw).normalized
    assert got == expected
    assert got == _chunck_oracle(raw)


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
    result = normalize_phonemes("ببَ تتِ")
    assert result.normalized != ""
    assert " " in result.normalized


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
