"""Tests for the counterfactual scorer.

This is the measurement that decides #10, so the tests simulate each hypothesis end to end
and check the verdict separates them — and pin the guards that stop an uninformative
recording from being counted as evidence.
"""

import pytest

from training.counterfactual_eval import (
    FOLLOWED_AUDIO,
    FOLLOWED_TEXT,
    NO_VOWEL,
    OTHER_VOWEL,
    classify,
    score_item,
    substitute_vowel,
    summarize,
    verdict,
    vowel_in_span,
    wilson_interval,
)

FATHA, DAMMA, KASRA = "\u064e", "\u064f", "\u0650"
QAF, DAL, MEEM, NOON = "\u0642", "\u062f", "\u0645", "\u0646"

# "qad min" — a two word reference; the target is word 0, carrying a single fatha.
WORD = f"{QAF}{FATHA}{DAL}"
REFERENCE = f"{WORD} {MEEM}{KASRA}{NOON}"
OFFSETS = [0, len(WORD) + 1, len(REFERENCE)]


def _item(canonical=FATHA, spoken=DAMMA):
    return {
        "item_id": "cf000",
        "surah_ayah": "2:1",
        "word_index": 0,
        "target_word": WORD,
        "canonical_vowel": canonical,
        "spoken_vowel": spoken,
        "audio_filename": "a.wav",
    }


SEGMENT = {"raw_reference_phonemes": REFERENCE, "raw_word_offsets": OFFSETS}


def _decodes(control_vowel, counterfactual_vowel):
    def render(vowel):
        return f"{QAF}{vowel}{DAL} {MEEM}{KASRA}{NOON}" if vowel else f"{QAF}{DAL} {MEEM}{KASRA}{NOON}"

    return {"control": render(control_vowel), "counterfactual": render(counterfactual_vowel)}


def test_substitute_vowel_replaces_only_the_target_word():
    swapped = substitute_vowel(REFERENCE, 0, len(WORD), DAMMA)
    assert swapped == f"{QAF}{DAMMA}{DAL} {MEEM}{KASRA}{NOON}"
    assert swapped.count(KASRA) == 1, "the other word's vowel is untouched"


def test_vowel_in_span_reads_the_target_word_only():
    decode = f"{QAF}{DAMMA}{DAL} {MEEM}{KASRA}{NOON}"
    assert vowel_in_span(decode, REFERENCE, 0, len(WORD)) == DAMMA


def test_vowel_in_span_is_none_when_the_alignment_never_reached_the_word():
    """An untranscribed word is not evidence about which vowel the model heard."""
    assert vowel_in_span(f"{MEEM}{KASRA}{NOON}", REFERENCE, 0, len(WORD)) is None


@pytest.mark.parametrize(
    "vowel,expected",
    [(DAMMA, FOLLOWED_AUDIO), (FATHA, FOLLOWED_TEXT), (KASRA, OTHER_VOWEL),
     ("", NO_VOWEL), (None, NO_VOWEL)],
)
def test_classify_maps_each_rendering_to_its_hypothesis(vowel, expected):
    assert classify(vowel, FATHA, DAMMA) == expected


def test_a_hearing_model_is_scored_as_following_the_audio():
    result = score_item(_item(), SEGMENT, _decodes(FATHA, DAMMA))
    assert result["control_passed"] is True
    assert result["outcome"] == FOLLOWED_AUDIO
    assert result["scored"] is True


def test_a_reconstructing_model_is_scored_as_following_the_text():
    """It emits the canonical fatha even though a damma was spoken."""
    result = score_item(_item(), SEGMENT, _decodes(FATHA, FATHA))
    assert result["outcome"] == FOLLOWED_TEXT
    assert result["scored"] is True


def test_an_item_whose_control_take_fails_is_dropped_not_counted_as_a_failure():
    """The guard that keeps the denominator honest.

    If the model cannot render this word's vowel even when it is recited correctly, the
    counterfactual take measures general inaccuracy on this voice, not hearing. Counting it
    as "did not follow the audio" would understate a hearing model.
    """
    result = score_item(_item(), SEGMENT, _decodes(KASRA, FATHA))
    assert result["control_passed"] is False
    assert result["scored"] is False

    summary = summarize([result])
    assert summary["scored"] == 0
    assert summary["control_failures_dropped"] == 1


def test_alignment_stability_is_reported():
    """Both references must agree on what the model wrote.

    Projecting only through the canonical reference could bias the answer toward the
    canonical vowel — the very hypothesis under test.
    """
    result = score_item(_item(), SEGMENT, _decodes(FATHA, DAMMA))
    assert result["alignment_stable"] is True


def _results(followed_audio, followed_text, swap="a->b"):
    rows = []
    for i in range(followed_audio + followed_text):
        rows.append({
            "item_id": f"cf{i:03d}",
            "swap": swap,
            "outcome": FOLLOWED_AUDIO if i < followed_audio else FOLLOWED_TEXT,
            "control_passed": True,
            "alignment_stable": True,
            "scored": True,
        })
    return rows


def test_a_hearing_model_yields_a_conclusive_positive_verdict():
    summary = summarize(_results(43, 4))
    assert summary["followed_audio_rate"] == round(43 / 47, 4)
    ruling = verdict(summary)
    assert ruling["conclusive"] is True and ruling["hears_tashkeel"] is True


def test_a_reconstructing_model_yields_a_conclusive_negative_verdict():
    ruling = verdict(summarize(_results(4, 43)))
    assert ruling["conclusive"] is True and ruling["hears_tashkeel"] is False
    assert "cannot flag" in ruling["interpretation"]


def test_a_split_result_is_reported_as_inconclusive_rather_than_a_coin_flip():
    """A near-even outcome is the one case where the interval must not be read as an answer."""
    ruling = verdict(summarize(_results(24, 23)))
    assert ruling["conclusive"] is False
    assert "spans" in ruling["reason"]


def test_too_few_scorable_items_is_inconclusive_however_lopsided():
    ruling = verdict(summarize(_results(10, 0)))
    assert ruling["conclusive"] is False
    assert "too few" in ruling["reason"]


def test_wilson_interval_stays_in_range_at_the_extremes():
    """The normal approximation degenerates exactly where a decisive result lands."""
    low, high = wilson_interval(47, 47)
    assert 0.0 <= low <= 1.0 and high == 1.0
    assert low > 0.9
    low, high = wilson_interval(0, 47)
    assert low == 0.0 and high < 0.1


def test_per_swap_breakdown_exposes_deafness_to_one_vowel():
    """A model deaf to one colour must not be able to hide behind the average."""
    rows = _results(20, 0, swap="a->b") + _results(0, 20, swap="c->d")
    by_swap = summarize(rows)["by_swap"]
    assert by_swap["a->b"] == {"scored": 20, "followed_audio": 20, "followed_text": 0}
    assert by_swap["c->d"] == {"scored": 20, "followed_audio": 0, "followed_text": 20}
