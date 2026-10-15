"""Tests for tadabur.qud_text: DigitalKhatt words onto quran-transcript's, and item references."""

from __future__ import annotations

import pytest

from tadabur.qud_text import Unmapped, item_reference, map_words, skeleton

CANONICAL = ["ٱلرَّحْمَـٰنُ", "عَلَى", "ٱلْعَرْشِ", "ٱسْتَوَىٰ"]


def test_skeleton_folds_the_two_scripts_spellings():
    assert skeleton("ٱلرَّحْمَٰنِ") == skeleton("ٱلرَّحْمَـٰنِ")  # tatweel before a dagger alif
    assert skeleton("يُؤْمِنُونَ") == skeleton("يُؤْمِنُونَ")  # ؤ, or و + hamza above
    assert skeleton("ٱلْحَيُّ") == skeleton("ٱلْحَىُّ")  # ي and alif maqsura
    assert skeleton("سَمْعِهِمْۖ") == skeleton("سَمْعِهِمْ")  # a waqf sign attached
    assert skeleton("بَرَآءَةࣱ") == skeleton("بَرَآءَةٌۭ")  # open tanween code points


def test_words_map_in_order_and_a_repeat_restarts_at_an_earlier_word():
    assert map_words(CANONICAL, CANONICAL).indices == (0, 1, 2, 3)
    recited = [CANONICAL[0], CANONICAL[1], CANONICAL[1], CANONICAL[2], CANONICAL[3]]
    mapped = map_words(recited, CANONICAL)
    assert mapped.indices == (0, 1, 1, 2, 3) and mapped.repeats == (2,)
    back = [*CANONICAL[:3], *CANONICAL[1:]]
    assert map_words(back, CANONICAL).indices == (0, 1, 2, 1, 2, 3)


def test_a_skipped_or_foreign_word_is_unmapped_never_guessed():
    with pytest.raises(Unmapped, match="never recited"):
        map_words(CANONICAL[:3], CANONICAL)
    with pytest.raises(Unmapped, match="matches no canonical word"):
        map_words([CANONICAL[0], "كَتَبَ", *CANONICAL[1:]], CANONICAL)


def test_an_item_is_a_run_of_canonical_words_whatever_was_repeated():
    """QUD's segments index the canonical words: a re-read is a new segment that restarts."""
    seen = []

    def reference_of(words):
        seen.append(words)
        return " ".join(words), [0, 1]

    item = item_reference([2, 3, 0, 1000], CANONICAL, reference_of)
    assert seen == [CANONICAL[1:3]] and item.word_indices == (1, 2)
    with pytest.raises(Unmapped):
        item_reference([4, 5, 0, 1], CANONICAL, reference_of)
