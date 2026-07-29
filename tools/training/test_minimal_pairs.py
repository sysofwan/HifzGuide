"""Tests for the tashkeel minimal-pair test.

The point of the module is to tell a model that *hears* a vowel from one that reconstructs
it from the canonical text, so the tests are built around simulating each of those two
models and checking the verdict separates them.
"""

from collections import Counter

import pytest

from training.minimal_pairs import (
    TextPriors,
    WordOccurrence,
    WordSite,
    ayah_overlap,
    decoded_words,
    reference_sites,
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
    priors = text_prior(segments, frozenset({"train.wav"}))
    assert priors.unigram[skeleton(word_a)] == Counter({FATHA + FATHA: 1})
    assert priors.ambiguous_skeletons == set()


def test_prior_ignores_words_with_no_short_vowel():
    word = "\u0627\u0644\u0644\u0647"
    priors = text_prior([_segment("t.wav", word, [0, len(word)])], frozenset({"t.wav"}))
    assert priors.unigram == {} and priors.context == {}


def test_decoded_words_reads_the_vowels_the_model_produced():
    reference = f"\u0643{FATHA}\u062a{DAMMA}\u0628"
    decode = f"\u0643{FATHA}\u062a{KASRA}\u0628"
    pairs = decoded_words(decode, reference, [0, len(reference)])
    assert [(site.reference_vowels, decoded) for site, decoded in pairs] == [
        (FATHA + DAMMA, FATHA + KASRA)
    ]


def test_an_undecoded_word_yields_no_vowels_rather_than_being_dropped():
    """A word the model never reached is wrong, not absent.

    Silently skipping it would let a model raise its score by refusing to decode the hard
    words — the opposite of what this gate measures.
    """
    reference = f"\u0643{FATHA}\u062a{DAMMA}\u0628"
    pairs = decoded_words("\u0632\u0632\u0632\u0632", reference, [0, len(reference)])
    assert [decoded for _, decoded in pairs] == [""]
    assert not WordOccurrence(pairs[0][0], "").correct


def test_words_are_scored_independently_within_a_segment():
    first = f"\u0643{FATHA}\u062a{FATHA}"
    second = f"\u0628{DAMMA}\u0631{DAMMA}"
    reference = first + second
    pairs = decoded_words(reference, reference, [0, len(first), len(reference)])
    assert [vowels for _, vowels in pairs] == [FATHA + FATHA, DAMMA + DAMMA]


AMBIGUOUS = "\u0643\u062a"
UNAMBIGUOUS = "\u0628\u0631"
LEFT, RIGHT = "\u0644", "\u0631"


def _site(sk, vowels, prev="^", nxt="$"):
    return WordSite(skeleton=sk, prev_skeleton=prev, next_skeleton=nxt, reference_vowels=vowels)


def _priors(unigram, context=None):
    return TextPriors(
        unigram={sk: Counter(v) for sk, v in unigram.items()},
        context={k: Counter(v) for k, v in (context or {}).items()},
    )


PRIOR = _priors({AMBIGUOUS: {FATHA: 7, DAMMA: 3}, UNAMBIGUOUS: {KASRA: 10}})


def _occ(sites_and_decodes):
    return [WordOccurrence(site, decode) for site, decode in sites_and_decodes]


def test_neighbours_are_recorded_for_each_word():
    first = f"\u0643{FATHA}"
    second = f"\u0628{DAMMA}"
    reference = first + second
    sites = [site for _, _, site in reference_sites(reference, [0, len(first), len(reference)])]
    assert [s.skeleton for s in sites] == ["\u0643", "\u0628"]
    assert sites[0].context_key == ("^", "\u0643", "\u0628")
    assert sites[1].context_key == ("\u0643", "\u0628", "$")


def test_the_prior_is_only_ambiguous_where_training_saw_two_vowelizations():
    report = score(_occ([(_site(UNAMBIGUOUS, KASRA), KASRA)]), PRIOR)
    assert report["distinct_ambiguous_skeletons"] == 1
    assert report["ambiguous_skeletons"]["words"] == 0
    assert report["unambiguous_skeletons"]["words"] == 1


def test_a_context_free_memorizer_does_not_beat_the_unigram_prior():
    truth = [FATHA] * 7 + [DAMMA] * 3
    occurrences = _occ([(_site(AMBIGUOUS, t), FATHA) for t in truth] * 20)
    amb = score(occurrences, PRIOR)["ambiguous_skeletons"]
    assert amb["model_accuracy"] == amb["unigram_prior_accuracy"] == 0.7
    assert amb["model_minus_unigram_prior"] == 0.0


def test_a_hearing_model_beats_both_priors():
    truth = [FATHA] * 7 + [DAMMA] * 3
    occurrences = _occ([(_site(AMBIGUOUS, t), t) for t in truth] * 20)
    report = score(occurrences, PRIOR)
    assert report["ambiguous_skeletons"]["model_accuracy"] == 1.0
    assert verdict(report)["hears_tashkeel"] is True


def test_a_context_memorizer_is_not_mistaken_for_a_hearing_model():
    """The defect that made the first version of this test a strawman.

    A model that never hears a harakah, but keys on the neighbouring words, scores far above
    the unigram prior. Judging against the unigram prior alone would call that "hearing".
    The verdict must be taken against the strongest text-only baseline, which this model
    exactly matches.
    """
    context = {
        (LEFT, AMBIGUOUS, "$"): {FATHA: 10},
        (RIGHT, AMBIGUOUS, "$"): {DAMMA: 10},
    }
    priors = _priors({AMBIGUOUS: {FATHA: 10, DAMMA: 10}}, context)
    sites = [_site(AMBIGUOUS, FATHA, prev=LEFT)] * 100 + [_site(AMBIGUOUS, DAMMA, prev=RIGHT)] * 100
    occurrences = _occ([(s, priors.guess_context(s)) for s in sites])

    report = score(occurrences, priors)
    amb = report["ambiguous_skeletons"]
    assert amb["model_accuracy"] == 1.0
    assert amb["unigram_prior_accuracy"] == 0.5
    assert amb["model_minus_unigram_prior"] == 0.5, "looks decisive against the weak prior"
    assert amb["context_prior_accuracy"] == 1.0
    assert amb["model_minus_context_prior"] == 0.0

    ruling = verdict(report)
    assert ruling["conclusive"] is False
    assert ruling["best_text_baseline"] == 1.0


def test_overall_accuracy_can_hide_a_text_inferring_model():
    guessed = [(_site(AMBIGUOUS, DAMMA), FATHA)] * 300
    easy = [(_site(UNAMBIGUOUS, KASRA), KASRA)] * 2700
    report = score(_occ(guessed + easy), PRIOR)
    assert report["all_words"]["model_accuracy"] == 0.9
    assert report["ambiguous_skeletons"]["model_accuracy"] == 0.0
    assert verdict(report)["hears_tashkeel"] is False


def test_verdict_is_inconclusive_when_the_ambiguous_slice_is_too_small():
    report = score(_occ([(_site(AMBIGUOUS, FATHA), FATHA)] * 10), PRIOR)
    assert verdict(report)["conclusive"] is False


def test_verdict_is_inconclusive_when_the_text_already_explains_the_words():
    priors = _priors({AMBIGUOUS: {FATHA: 990, DAMMA: 10}})
    occurrences = _occ(
        [(_site(AMBIGUOUS, FATHA), FATHA)] * 990 + [(_site(AMBIGUOUS, DAMMA), DAMMA)] * 10
    )
    ruling = verdict(score(occurrences, priors))
    assert ruling["conclusive"] is False
    assert "counterfactual" in ruling["reason"]


@pytest.mark.parametrize("margin,expected", [(0.05, True), (0.30, False)])
def test_margin_controls_how_far_above_the_prior_counts_as_hearing(margin, expected):
    truth = [FATHA] * 8 + [DAMMA] * 2
    decodes = [FATHA] * 8 + [DAMMA, FATHA]
    occurrences = _occ([(_site(AMBIGUOUS, t), d) for t, d in zip(truth, decodes)] * 20)
    assert verdict(score(occurrences, PRIOR), margin)["hears_tashkeel"] is expected


def test_ayah_overlap_reports_shared_text_between_the_splits():
    """The split separates reciters, not Quranic content — that limit must be visible."""
    segments = [
        {"clip_audio_filename": "t.wav", "surah_ayah": "2:1"},
        {"clip_audio_filename": "t.wav", "surah_ayah": "2:2"},
        {"clip_audio_filename": "v.wav", "surah_ayah": "2:1"},
    ]
    overlap = ayah_overlap(segments, frozenset({"t.wav"}), frozenset({"v.wav"}))
    assert overlap["shared_ayahs"] == 1
    assert overlap["val_ayahs_also_in_train"] == 1.0


def test_ayah_overlap_refuses_to_report_zero_when_the_key_is_missing():
    """Silently reporting no overlap would understate the very limit it exists to expose."""
    with pytest.raises(KeyError, match="surah_ayah"):
        ayah_overlap(
            [{"clip_audio_filename": "t.wav"}], frozenset({"t.wav"}), frozenset()
        )


def test_unstaged_segment_audio_resolves_to_none(tmp_path):
    """A partially staged segment directory is normal, so this reports rather than raises."""
    assert segment_audio_path(tmp_path, "nope.wav") is None
    (tmp_path / "yes.wav").write_bytes(b"")
    assert segment_audio_path(tmp_path, "yes.wav") == tmp_path / "yes.wav"
