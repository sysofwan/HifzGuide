"""Tests for the counterfactual recording-sheet builder.

The sheet is recited by a human, so a bad item is expensive: it wastes a take and, worse,
can produce a recording that looks like evidence but is not. These tests pin the
constraints that make an item scorable.
"""

from collections import Counter

from training.counterfactual_script import (
    SWAP_DIRECTIONS,
    CounterfactualItem,
    candidate_items,
    select,
)
from training.minimal_pairs import TextPriors

FATHA, DAMMA, KASRA = "\u064e", "\u064f", "\u0650"
BA, LAM, MEEM, ALIF = "\u0628", "\u0644", "\u0645", "\u0627"


def _priors(context):
    return TextPriors(unigram={}, context={k: Counter(v) for k, v in context.items()})


def _segment(clip, uthmani, phonemes, offsets, surah_ayah="2:1"):
    return {
        "clip_audio_filename": clip,
        "audio_filename": f"{clip}_seg0.wav",
        "surah_ayah": surah_ayah,
        "uthmani": uthmani,
        "raw_reference_phonemes": phonemes,
        "raw_word_offsets": offsets,
    }


VAL = frozenset({"v.wav"})


def _one_word(uthmani, phonemes, surah_ayah="2:1"):
    return _segment("v.wav", uthmani, phonemes, [0, len(phonemes)], surah_ayah)


def _confident(*keys):
    return _priors({k: {v: 5} for k, v in keys})


def test_a_clean_single_vowel_word_is_a_candidate():
    word = f"{BA}{FATHA}{LAM}"
    priors = _confident((("^", BA + LAM, "$"), FATHA))
    items = candidate_items([_one_word(word, word)], VAL, priors)
    assert len(items) == 1
    assert items[0].target_word == word
    assert items[0].canonical_vowel == FATHA


def test_a_madd_vowel_is_excluded():
    """مَا cannot be said as مُا — the fatha opens an elongation, it is not a free choice."""
    word = f"{MEEM}{FATHA}{ALIF}"
    priors = _confident((("^", MEEM + ALIF, "$"), FATHA))
    assert candidate_items([_one_word(word, word)], VAL, priors) == []


def test_a_word_whose_context_is_not_deterministic_is_excluded():
    """If the text itself is unsure, a reconstructing model has no canonical vowel to
    fall back on, so the item cannot discriminate."""
    word = f"{BA}{FATHA}{LAM}"
    priors = _priors({("^", BA + LAM, "$"): {FATHA: 5, DAMMA: 5}})
    assert candidate_items([_one_word(word, word)], VAL, priors) == []


def test_a_barely_attested_context_is_excluded():
    word = f"{BA}{FATHA}{LAM}"
    priors = _priors({("^", BA + LAM, "$"): {FATHA: 1}})
    assert candidate_items([_one_word(word, word)], VAL, priors) == []


def test_a_written_form_with_extra_vowels_is_excluded():
    """The written and phoneme forms can disagree on vowel count.

    ``spoken_word`` rewrites the *written* form, so an item like ٱلْفَصْلِ would tell the
    reciter to change two vowels while only one is under test.
    """
    written = f"{BA}{FATHA}{LAM}{KASRA}"
    phonemes = f"{BA}{FATHA}{LAM}"
    priors = _confident((("^", BA + LAM, "$"), FATHA))
    assert candidate_items([_one_word(written, phonemes)], VAL, priors) == []


def test_training_clips_are_not_offered():
    word = f"{BA}{FATHA}{LAM}"
    priors = _confident((("^", BA + LAM, "$"), FATHA))
    assert candidate_items([_one_word(word, word)], frozenset({"other.wav"}), priors) == []


def _item(word_letters, vowel, surah_ayah, word=None):
    written = word or f"{word_letters[0]}{vowel}{word_letters[1:]}"
    return CounterfactualItem(
        item_id="",
        surah_ayah=surah_ayah,
        segment_text=written,
        word_index=0,
        target_word=written,
        canonical_vowel=vowel,
        spoken_vowel="",
        reference_phonemes=written,
        audio_filename="a.wav",
    )


def test_spoken_word_changes_exactly_one_character():
    item = _item(BA + LAM, FATHA, "2:1")
    chosen = select([item], 1)[0]
    changed = [a != b for a, b in zip(chosen.target_word, chosen.spoken_word)]
    assert sum(changed) == 1
    assert len(chosen.spoken_word) == len(chosen.target_word)
    assert chosen.spoken_vowel != chosen.canonical_vowel


def test_every_swap_direction_is_covered_when_one_vowel_is_scarce():
    """The greedy version starved the last direction and tested only five of six swaps.

    Damma-initial words are much rarer than fatha ones and words are deduplicated globally,
    so filling each direction in turn drains the shared pool before the last one is reached.
    """
    pool = [_item(BA + LAM, FATHA, f"2:{i}", word=f"{BA}{FATHA}{LAM}{'x' * i}") for i in range(20)]
    pool += [_item(BA + LAM, DAMMA, f"3:{i}", word=f"{BA}{DAMMA}{LAM}{'y' * i}") for i in range(2)]
    pool += [_item(BA + LAM, KASRA, f"4:{i}", word=f"{BA}{KASRA}{LAM}{'z' * i}") for i in range(2)]

    chosen = select(pool, 12)

    directions = {(c.canonical_vowel, c.spoken_vowel) for c in chosen}
    assert directions == set(SWAP_DIRECTIONS)


def test_no_ayah_or_word_is_used_twice():
    """Three takes of بَلْ probe one lexical item three times, not three."""
    pool = [_item(BA + LAM, FATHA, f"2:{i}") for i in range(5)]
    chosen = select(pool, 5)
    assert len(chosen) == 1, "the same word must not be offered repeatedly"

    varied = [_item(BA + LAM, FATHA, "2:1", word=f"{BA}{FATHA}{LAM}{c}") for c in "abc"]
    assert len(select(varied, 3)) == 1, "the same ayah must not be offered repeatedly"


def test_selection_is_deterministic():
    pool = [_item(BA + LAM, FATHA, f"2:{i}", word=f"{BA}{FATHA}{LAM}{'x' * i}") for i in range(20)]
    first = [(c.item_id, c.target_word, c.spoken_vowel) for c in select(pool, 6, seed=0)]
    second = [(c.item_id, c.target_word, c.spoken_vowel) for c in select(pool, 6, seed=0)]
    assert first == second
