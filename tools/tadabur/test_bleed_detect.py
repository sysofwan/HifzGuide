"""Tests for neighbour-ayah bleed detection.

The synthetic cases use invented "ayat" of distinct bare consonants so normalization
leaves them alone and the alignment's behaviour is the only thing under test. The last
test is the regression the issue's acceptance names: the nine human-adjudicated clips in
``eval_fixtures/reject_bleed_labels.jsonl``, re-scored from their committed decodes.
"""

from __future__ import annotations

import pytest

from tadabur.bleed_detect import (
    BAND_CLEAN,
    BAND_LOW,
    BAND_MARGINAL,
    BAND_NOT_REPEAT,
    BAND_SHADDA,
    BASMALLAH_KEY,
    TRUNCATION_PHONEMES,
    BleedDetector,
    Score,
    read_bleed_labels,
    reference_baseline,
    reject_band,
    trim_baseline,
)
from tadabur.rejects import RejectRecord

# Three consecutive invented ayat. Every phoneme is a bare core with no run, so
# ``normalize_phonemes`` is a no-op on them and word spaces are the only gaps.
PREVIOUS = "بتث جحخ دذر"
THIS = "زسش صضط ظعغ فقك"
NEXT = "لمن هوي ءبت"
REFERENCES = {"2:1": PREVIOUS, "2:2": THIS, "2:3": NEXT}
SPACELESS_THIS = THIS.replace(" ", "")


def detector() -> BleedDetector:
    return BleedDetector.of(REFERENCES)


def test_a_clean_recitation_of_the_ayah_shows_no_bleed():
    verdict = detector().detect(SPACELESS_THIS, "2:2")

    assert not verdict.detected
    assert verdict.leading.matched == 0
    assert verdict.trailing.matched == 0
    assert verdict.uncovered_head == 0
    assert verdict.uncovered_tail == 0


def test_both_edges_of_a_bled_clip_are_found_and_located():
    lead_in = "دذر"  # the previous ayah's last word
    tail = "لمن"  # the next ayah's first word
    verdict = detector().detect(lead_in + SPACELESS_THIS + tail, "2:2")

    assert verdict.leading.detected and verdict.trailing.detected
    assert verdict.leading.matched == len(lead_in)
    assert verdict.trailing.matched == len(tail)
    assert verdict.leading.purity == 1.0 and verdict.trailing.purity == 1.0
    # The query span is what a re-cut converts to time: the lead-in is the query's head
    # and the bleed tail is everything after this ayah's last phoneme.
    assert (verdict.leading.query_start, verdict.leading.query_end) == (0, len(lead_in))
    assert verdict.trailing.query_start == len(lead_in) + len(SPACELESS_THIS)
    assert verdict.trailing.query_end == len(lead_in) + len(SPACELESS_THIS) + len(tail)


def test_a_re_read_is_not_bleed():
    """The insertion a re-read makes is this ayah's own words, so no edge fires.

    This is the case the issue's per-ayah reference baseline gets wrong by construction:
    the repeated tail legitimately scores best against this ayah.
    """
    repeated = SPACELESS_THIS + "صضط" + "ظعغفقك"
    verdict = detector().detect(repeated, "2:2")

    assert not verdict.detected


def test_a_dragged_alignment_is_rejected_on_purity_not_span():
    """A garbled tail wanders into the next ayah's text but matches little of it."""
    garbled_tail = "بتثجحخدذر"  # neighbours' letters, but the *wrong* neighbour's
    verdict = detector().detect(SPACELESS_THIS + garbled_tail, "2:2")

    assert not verdict.trailing.detected
    assert verdict.trailing.purity < 0.6
    # ...and the same clip does fire once purity is no longer required, which is what
    # says purity — not span, and not score — is the term doing the work.
    lenient = detector().detect(
        SPACELESS_THIS + garbled_tail, "2:2", min_purity=0.0, min_phonemes=1
    )
    assert lenient.trailing.detected


def test_a_two_phoneme_bleed_still_fires():
    """The shortest real bleed in the labelled set is two phonemes (2:278, 12:48)."""
    verdict = detector().detect(SPACELESS_THIS + "لم", "2:2")

    assert verdict.trailing.detected
    assert verdict.trailing.matched == 2


def test_a_one_phoneme_edge_does_not_fire():
    verdict = detector().detect(SPACELESS_THIS + "ل", "2:2")

    assert not verdict.trailing.detected


def test_an_unfinished_ayah_is_reported_as_uncovered_tail():
    verdict = detector().detect("زسشصضط", "2:2")

    assert not verdict.detected
    assert verdict.uncovered_tail >= TRUNCATION_PHONEMES
    assert verdict.uncovered_head == 0


def test_an_unknown_ayah_yields_an_empty_verdict():
    verdict = detector().detect(SPACELESS_THIS, "99:99")

    assert not verdict.detected
    assert verdict.ref_length == 0


def test_an_empty_decode_yields_an_empty_verdict():
    verdict = detector().detect("", "2:2")

    assert not verdict.detected
    assert verdict.ref_length == len(THIS)


def test_the_basmallah_is_offered_as_a_lead_in_to_a_surah_opening():
    full = BleedDetector.load()

    assert full.leading_keys("2:1") == ("1:7", BASMALLAH_KEY)
    assert full.leading_keys("9:1") == ("8:75",)  # At-Tawbah opens without one
    assert full.leading_keys("1:1") == ()  # Al-Fatiha holds it as its own first ayah
    assert full.trailing_keys("114:6") == ()
    assert full.trailing_keys("2:1") == ("2:2",)


def test_a_surah_opening_bled_into_by_the_basmallah_is_found():
    full = BleedDetector.load()
    basmallah = full.references[BASMALLAH_KEY].replace(" ", "")
    ayah = full.references["2:1"].replace(" ", "")

    verdict = full.detect(basmallah + ayah, "2:1")

    assert verdict.leading.detected
    assert verdict.leading.query_start == 0


def _record(**kwargs) -> RejectRecord:
    defaults = dict(
        audio_filename="a.wav",
        surah_ayah="2:2",
        reciter_id=0,
        ayah_duration_s=1.0,
        match_ratio=0.8,
        max_insertion_run=8,
        leading_trim=0,
        trailing_trim=0,
        added_shadda=False,
        predicted_phonemes="",
    )
    return RejectRecord(**{**defaults, **kwargs})


def test_reject_bands_key_the_prevalence_table_the_way_the_audit_was_organised():
    assert reject_band(_record(max_insertion_run=0)) == BAND_NOT_REPEAT
    assert reject_band(_record(added_shadda=True)) == BAND_SHADDA
    assert reject_band(_record(match_ratio=0.80)) == BAND_CLEAN
    assert reject_band(_record(match_ratio=0.72)) == BAND_MARGINAL
    assert reject_band(_record(match_ratio=0.50)) == BAND_LOW


def test_trim_baseline_reads_the_gate_signals_verbatim():
    assert trim_baseline(_record(leading_trim=5, trailing_trim=4)) == (True, False)
    assert trim_baseline(_record(leading_trim=0, trailing_trim=0)) == (False, False)


def test_reference_baseline_is_blind_to_a_re_read():
    """Its documented failure: a re-read's repeated end is *this* ayah's own words.

    The baseline asks which ayah each end slice belongs to, and on a re-read the honest
    answer is "this one" at both ends — which is why it finds nothing on the very
    population the corpus is mined from.
    """
    re_read = SPACELESS_THIS + "صضطظعغفقك"

    assert reference_baseline(detector(), re_read, "2:2") == (False, False)


def test_score_tallies_both_directions():
    score = Score.of([(True, True), (True, False), (False, True), (False, False)])

    assert (score.true_positives, score.false_positives) == (1, 1)
    assert (score.false_negatives, score.true_negatives) == (1, 1)
    assert score.precision == 0.5
    assert score.recall == 0.5


def test_the_labelled_set_is_the_nine_adjudicated_clips():
    labels = read_bleed_labels()

    assert len(labels) == 9
    assert all(label.predicted_phonemes for label in labels)
    assert sum(label.leading_bleed for label in labels) == 2
    assert sum(label.trailing_bleed for label in labels) == 4
    assert sum(label.truncated for label in labels) == 1


def test_the_detector_clears_the_labelled_set_and_beats_both_baselines():
    """Issue #67's acceptance bar, re-scored from the committed decodes.

    A detector that fires on a clip the reviewer heard as clean is worse than none, so
    precision is asserted exactly, not as a bound.
    """
    full = BleedDetector.load()
    labels = read_bleed_labels()

    outcomes = {"detector": [], "trim_baseline": [], "reference_baseline": []}
    for label in labels:
        verdict = full.detect(label.predicted_phonemes, label.surah_ayah)
        actual = label.leading_bleed or label.trailing_bleed
        outcomes["detector"].append((verdict.detected, actual))
        outcomes["trim_baseline"].append((any(trim_baseline(label)), actual))
        outcomes["reference_baseline"].append(
            (any(reference_baseline(full, label.predicted_phonemes, label.surah_ayah)), actual)
        )

    scores = {name: Score.of(pairs) for name, pairs in outcomes.items()}
    assert scores["detector"].precision == 1.0
    assert scores["detector"].recall == 1.0
    for baseline in ("trim_baseline", "reference_baseline"):
        assert scores[baseline].recall < scores["detector"].recall


def test_the_truncated_clip_is_the_one_the_reviewer_heard_stop_early():
    full = BleedDetector.load()
    labels = {label.surah_ayah: label for label in read_bleed_labels()}
    truncated = [label for label in labels.values() if label.truncated]

    assert [label.surah_ayah for label in truncated] == ["40:16"]
    for label in labels.values():
        verdict = full.detect(label.predicted_phonemes, label.surah_ayah)
        assert (verdict.uncovered_tail >= TRUNCATION_PHONEMES) == label.truncated
