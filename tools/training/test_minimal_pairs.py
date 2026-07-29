"""Tests for the tashkeel minimal-pair test.

The point of the module is to tell a model that *hears* a vowel from one that reconstructs
it from the canonical text, so the tests are built around simulating each of those two
models and checking the verdict separates them.
"""

from collections import Counter

import pytest

from training.minimal_pairs import (
    WordOccurrence,
    decoded_words,
    score,
    segment_audio_path,
    skeleton,
    text_prior,
    verdict,
    vowelization,
)

FATHA, DAMMA, KASRA = "\u064e", "\u064f", "\u0650"


def test_skeleton_and_vowelization_partition_the_word():
    word = f"\u0643{FATHA}\u062a{FATHA}\u0628{FATHA}"
    assert skeleton(word) == "\u0643\u062a\u0628"
    assert vowelization(word) == FATHA * 3
    assert len(skeleton(word)) + len(vowelization(word)) == len(word)


def _segment(clip, reference, offsets):
    return {
        "clip_audio_filename": clip,
        "raw_reference_phonemes": reference,
        "raw_word_offsets": offsets,
    }


def test_prior_counts_only_training_clips():
    word_a = f"\u0643{FATHA}\u062a{FATHA}"
    word_b = f"\u0643{DAMMA}\u062a{FATHA}"
    segments = [
        _segment("train.wav", word_a, [0, len(word_a)]),
        _segment("val.wav", word_b, [0, len(word_b)]),
    ]
    prior = text_prior(segments, frozenset({"train.wav"}))
    assert prior[skeleton(word_a)] == Counter({FATHA + FATHA: 1})


def test_prior_ignores_words_with_no_short_vowel():
    word = "\u0627\u0644\u0644\u0647"
    prior = text_prior([_segment("t.wav", word, [0, len(word)])], frozenset({"t.wav"}))
    assert prior == {}


def test_decoded_words_reads_the_vowels_the_model_produced():
    reference = f"\u0643{FATHA}\u062a{DAMMA}\u0628"
    decode = f"\u0643{FATHA}\u062a{KASRA}\u0628"
    pairs = decoded_words(decode, reference, [0, len(reference)])
    assert pairs == [(reference, FATHA + KASRA)]


def test_an_undecoded_word_yields_no_vowels_rather_than_being_dropped():
    """A word the model never reached is wrong, not absent.

    Silently skipping it would let a model raise its score by refusing to decode the hard
    words — the opposite of what this gate measures.
    """
    reference = f"\u0643{FATHA}\u062a{DAMMA}\u0628"
    pairs = decoded_words("\u0632\u0632\u0632\u0632", reference, [0, len(reference)])
    assert pairs == [(reference, "")]
    assert not WordOccurrence(skeleton(reference), FATHA + DAMMA, "").correct


def test_words_are_scored_independently_within_a_segment():
    first = f"\u0643{FATHA}\u062a{FATHA}"
    second = f"\u0628{DAMMA}\u0631{DAMMA}"
    reference = first + second
    pairs = decoded_words(reference, reference, [0, len(first), len(reference)])
    assert [vowels for _, vowels in pairs] == [FATHA + FATHA, DAMMA + DAMMA]


def _prior(counts: dict[str, dict[str, int]]) -> dict[str, Counter]:
    return {sk: Counter(v) for sk, v in counts.items()}


AMBIGUOUS = "\u0643\u062a"
UNAMBIGUOUS = "\u0628\u0631"
PRIOR = _prior(
    {
        AMBIGUOUS: {FATHA: 7, DAMMA: 3},
        UNAMBIGUOUS: {KASRA: 10},
    }
)


def _occurrences(truth_and_decode):
    return [
        WordOccurrence(sk, truth, decode) for sk, truth, decode in truth_and_decode
    ]


def test_the_prior_is_only_ambiguous_where_training_saw_two_vowelizations():
    report = score(_occurrences([(UNAMBIGUOUS, KASRA, KASRA)]), PRIOR)
    assert report["distinct_ambiguous_skeletons"] == 1
    assert report["ambiguous_skeletons"]["words"] == 0
    assert report["unambiguous_skeletons"]["words"] == 1


def test_a_text_inferring_model_does_not_beat_the_prior():
    """Simulate the failure mode: always emit the training majority vowelization."""
    truth = [FATHA] * 7 + [DAMMA] * 3
    occurrences = _occurrences([(AMBIGUOUS, t, FATHA) for t in truth] * 20)
    report = score(occurrences, PRIOR)
    amb = report["ambiguous_skeletons"]
    assert amb["model_accuracy"] == amb["text_prior_accuracy"] == 0.7
    assert amb["model_minus_prior"] == 0.0
    assert verdict(report)["hears_tashkeel"] is False


def test_a_hearing_model_beats_the_prior():
    truth = [FATHA] * 7 + [DAMMA] * 3
    occurrences = _occurrences([(AMBIGUOUS, t, t) for t in truth] * 20)
    report = score(occurrences, PRIOR)
    amb = report["ambiguous_skeletons"]
    assert amb["model_accuracy"] == 1.0
    assert amb["text_prior_accuracy"] == 0.7
    assert verdict(report)["hears_tashkeel"] is True


def test_overall_accuracy_can_hide_a_text_inferring_model():
    """The whole reason the gate exists: the headline number does not separate them.

    Unambiguous words dominate the corpus, so a model that only ever guesses the majority
    vowelization still posts a high overall accuracy.
    """
    guessed = [(AMBIGUOUS, DAMMA, FATHA)] * 300
    easy = [(UNAMBIGUOUS, KASRA, KASRA)] * 2700
    report = score(_occurrences(guessed + easy), PRIOR)
    assert report["all_words"]["model_accuracy"] == 0.9
    assert report["ambiguous_skeletons"]["model_accuracy"] == 0.0
    assert verdict(report)["hears_tashkeel"] is False


def test_verdict_is_inconclusive_when_the_ambiguous_slice_is_too_small():
    report = score(_occurrences([(AMBIGUOUS, FATHA, FATHA)] * 10), PRIOR)
    assert verdict(report)["conclusive"] is False


def test_verdict_is_inconclusive_when_the_text_prior_already_explains_the_words():
    """A skeleton with a 99:1 split is not a real minimal pair.

    Beating a prior that is already right ~99% of the time proves nothing, so the gate
    must decline to rule rather than report a false negative.
    """
    prior = _prior({AMBIGUOUS: {FATHA: 990, DAMMA: 10}})
    occurrences = _occurrences([(AMBIGUOUS, FATHA, FATHA)] * 990 + [(AMBIGUOUS, DAMMA, DAMMA)] * 10)
    report = score(occurrences, prior)
    assert report["ambiguous_skeletons"]["text_prior_accuracy"] == 0.99
    assert verdict(report) == {
        "conclusive": False,
        "reason": "the text prior alone already explains these words",
    }


@pytest.mark.parametrize("margin,expected", [(0.05, True), (0.30, False)])
def test_margin_controls_how_far_above_the_prior_counts_as_hearing(margin, expected):
    truth = [FATHA] * 8 + [DAMMA] * 2
    decodes = [FATHA] * 8 + [DAMMA, FATHA]
    occurrences = _occurrences(
        [(AMBIGUOUS, t, d) for t, d in zip(truth, decodes)] * 20
    )
    report = score(occurrences, PRIOR)
    assert verdict(report, margin)["hears_tashkeel"] is expected


def test_unstaged_segment_audio_resolves_to_none(tmp_path):
    """A partially staged segment directory is normal, so this reports rather than raises.

    The caller pre-filters on it and prints the skipped count, which keeps missing audio a
    visible reduction in sample size instead of a silent one.
    """
    assert segment_audio_path(tmp_path, "nope.wav") is None
    (tmp_path / "yes.wav").write_bytes(b"")
    assert segment_audio_path(tmp_path, "yes.wav") == tmp_path / "yes.wav"
