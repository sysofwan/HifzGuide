"""Tests for the tashkeel-sensitive eval (:mod:`training.tashkeel_eval`).

The property that matters: this gate must see what every other gate is blind to. The
`.balanced`/`.strict` scorers both normalize short vowels away, so a decode that emits no
tashkeel at all scores identically to a perfect one — which is how a fine-tune that destroyed
the capability passed all four #10 gates. These tests pin the opposite behaviour.
"""

from __future__ import annotations

import pytest

from training.tashkeel_eval import (
    DAMMA,
    FATHA,
    KASRA,
    TashkeelReport,
    VowelCounts,
    gate,
    score_vowels,
    score_windows,
)

# "مَلقَارِعَه" — three short vowels, one of each colour.
REFERENCE = f"\u0645{FATHA}\u0644\u0642{FATHA}\u0627\u0631{KASRA}\u0639{DAMMA}\u0647"


def _strip(text: str) -> str:
    return "".join(c for c in text if c not in {FATHA, DAMMA, KASRA})


def test_a_perfect_decode_scores_full_recall():
    counts = score_vowels(REFERENCE, REFERENCE)

    assert counts.matched == 4 and counts.swapped == 0 and counts.omitted == 0
    assert counts.recall == 1.0 and counts.precision == 1.0


def test_a_vowel_free_decode_scores_zero_recall():
    """The exact failure the normalized scorers cannot see.

    ``normalize_phonemes`` strips U+064E/64F/650 from both sides, so ``STRICT_SCORER``
    accepts this decode. This gate must not.
    """
    counts = score_vowels(_strip(REFERENCE), REFERENCE)

    assert counts.matched == 0
    assert counts.omitted == 4
    assert counts.recall == 0.0


def test_the_normalized_scorer_is_blind_to_what_this_gate_measures():
    """Pins the *reason* this module exists, so the redundancy claim stays falsifiable."""
    from tadabur.normalization import normalize_phonemes
    from tadabur.waqf_integration_eval import strict_accepts

    normalized_reference = normalize_phonemes(REFERENCE).normalized
    assert strict_accepts(REFERENCE, normalized_reference)
    assert strict_accepts(_strip(REFERENCE), normalized_reference)  # blind

    assert score_vowels(REFERENCE, REFERENCE).recall == 1.0
    assert score_vowels(_strip(REFERENCE), REFERENCE).recall == 0.0  # not blind


def test_a_swapped_vowel_is_distinguished_from_an_omitted_one():
    """A wrong i'raab is a different failure from declining to mark one."""
    swapped = REFERENCE.replace(f"\u0639{DAMMA}", f"\u0639{KASRA}")

    counts = score_vowels(swapped, REFERENCE)

    assert counts.swapped == 1 and counts.omitted == 0
    assert counts.matched == 3
    assert counts.swap_rate == pytest.approx(0.25)


def test_spurious_vowels_lower_precision_but_not_recall():
    hallucinated = REFERENCE.replace("\u0644", f"\u0644{FATHA}")

    counts = score_vowels(hallucinated, REFERENCE)

    assert counts.spurious == 1
    assert counts.recall == 1.0
    assert counts.precision < 1.0


def test_vowels_outside_the_local_alignment_count_as_omissions():
    """Smith-Waterman is local: a decode matching one fragment must not score full recall."""
    counts = score_vowels(f"\u0645{FATHA}", REFERENCE)

    assert counts.matched == 1
    assert counts.omitted == 3  # the three vowels the fragment never reached
    assert counts.recall == pytest.approx(0.25)


def test_empty_reference_scores_nothing_rather_than_dividing_by_zero():
    counts = score_vowels("anything", "")

    assert counts == VowelCounts()
    assert counts.recall == 0.0 and counts.f1 == 0.0


def test_per_vowel_breakdown_exposes_a_single_collapsed_colour():
    """A model that never marks kasra must not hide behind a good pooled number."""
    no_kasra = REFERENCE.replace(KASRA, "")

    report = score_windows([no_kasra], [REFERENCE], "candidate")

    assert report.per_vowel["fatha"].recall == 1.0
    assert report.per_vowel["damma"].recall == 1.0
    assert report.per_vowel["kasra"].recall == 0.0
    assert report.counts.recall < 1.0


def _report(recall_pairs: tuple[int, int], name: str) -> TashkeelReport:
    matched, omitted = recall_pairs
    counts = VowelCounts(matched=matched, omitted=omitted)
    return TashkeelReport(name, 1, counts, {})


def test_gate_fails_a_candidate_that_lost_the_capability_the_baseline_had():
    """The regression this gate exists to catch: base emits vowels, fine-tune emits none."""
    verdict = gate(_report((0, 10), "finetuned"), _report((8, 2), "base"))

    assert verdict["passed"] is False
    assert verdict["meets_floor"] is False
    assert verdict["regressed_vs_baseline"] is True
    assert verdict["recall_delta"] == pytest.approx(-0.8)


def test_gate_passes_a_candidate_that_matches_the_baseline():
    verdict = gate(_report((8, 2), "finetuned"), _report((8, 2), "base"))

    assert verdict["passed"] is True
    assert verdict["regressed_vs_baseline"] is False


def test_gate_allows_a_small_trade_within_tolerance():
    verdict = gate(_report((77, 23), "finetuned"), _report((80, 20), "base"), tolerance=0.05)

    assert verdict["passed"] is True
    assert verdict["regressed_vs_baseline"] is False


def test_gate_records_when_no_baseline_was_compared():
    """A missing baseline must weaken the verdict visibly, never pass silently."""
    verdict = gate(_report((8, 2), "finetuned"), None)

    assert verdict["baseline_compared"] is False
    assert verdict["baseline_recall"] is None
    assert verdict["passed"] is True  # floor only


def test_gate_fails_a_candidate_below_the_floor_even_without_a_baseline():
    verdict = gate(_report((1, 9), "finetuned"), None)

    assert verdict["passed"] is False and verdict["meets_floor"] is False
