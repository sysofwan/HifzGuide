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


def test_vowels_the_model_invents_outside_the_alignment_still_count_as_spurious():
    """The mirror of the omission case above.

    A hallucinated vowel is most likely at the trimmed ends, which is exactly where local
    alignment produces no column. Charging only interior insertions inflates precision.
    """
    leading = score_vowels(f"{FATHA}{REFERENCE}", REFERENCE)
    trailing = score_vowels(f"{REFERENCE}{KASRA}", REFERENCE)

    assert leading.spurious == 1, "a vowel before the aligned span is still emitted"
    assert trailing.spurious == 1, "a vowel after the aligned span is still emitted"
    assert leading.recall == 1.0 and trailing.recall == 1.0  # recall is untouched
    assert leading.precision < 1.0 and trailing.precision < 1.0


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


def test_gate_fails_a_candidate_that_traded_omissions_for_wrong_colours():
    """ADR-0003's named failure: pooled accuracy holds while discrimination collapses.

    The candidate matches the baseline's recall exactly, but every vowel it stopped
    omitting it now mis-colours -- asserting a wrong i'raab, the poisonous failure.
    """
    base = TashkeelReport("base", 1, VowelCounts(matched=8, omitted=2), {})
    swapper = TashkeelReport("swapper", 1, VowelCounts(matched=8, swapped=2), {})

    verdict = gate(swapper, base)

    assert verdict["recall_delta"] == pytest.approx(0.0), "pooled recall is unchanged"
    assert verdict["regressed_swap_rate"] is True
    assert verdict["passed"] is False


def test_gate_fails_a_candidate_that_bought_recall_by_voweling_everything():
    base = TashkeelReport("base", 1, VowelCounts(matched=8, omitted=2), {})
    spammer = TashkeelReport("spammer", 1, VowelCounts(matched=10, spurious=40), {})

    verdict = gate(spammer, base)

    assert verdict["candidate_recall"] == pytest.approx(1.0)  # recall improved
    assert verdict["regressed_precision"] is True
    assert verdict["passed"] is False


def test_gate_fails_a_candidate_that_sacrificed_one_colour():
    """Kasra is the weakest class, so fatha's larger count can mask its collapse."""
    per_vowel_base = {
        "fatha": VowelCounts(matched=80, omitted=20),
        "kasra": VowelCounts(matched=8, omitted=2),
    }
    per_vowel_cand = {
        "fatha": VowelCounts(matched=98, omitted=2),
        "kasra": VowelCounts(matched=0, omitted=10),
    }
    base = TashkeelReport("base", 1, VowelCounts(matched=88, omitted=22), per_vowel_base)
    cand = TashkeelReport("cand", 1, VowelCounts(matched=98, omitted=12), per_vowel_cand)

    verdict = gate(cand, base)

    assert verdict["recall_delta"] > 0, "pooled recall actually improved"
    assert verdict["collapsed_vowels"] == ["kasra"]
    assert verdict["passed"] is False


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


def test_a_vowels_only_decode_cannot_score_a_perfect_recall():
    """Smith-Waterman is local, so it will gap over every consonant.

    Before carrier-anchoring, a "decode" of nothing but the reference's vowel sequence --
    no consonants at all, an obviously worthless transcription -- scored recall 1.000 AND
    precision 1.000. The vowel must land on a consonant the model actually heard.
    """
    vowels_only = "".join(c for c in REFERENCE if c in {FATHA, DAMMA, KASRA})

    counts = score_vowels(vowels_only, REFERENCE)

    assert counts.matched == 0, "no vowel sat on a correctly heard carrier"
    assert counts.unanchored == 4
    assert counts.recall == 0.0


def test_a_vowel_on_a_misheard_consonant_is_not_credited():
    """The carrier is wrong, so the model did not put the right vowel in the right place."""
    misheard = REFERENCE.replace(f"\u0644\u0642{FATHA}", f"\u0644\u0643{FATHA}")

    counts = score_vowels(misheard, REFERENCE)

    assert counts.unanchored == 1
    assert counts.matched == 3


def test_the_per_vowel_breakdown_is_carrier_anchored_too():
    """The pooled gaming test above passed while the per-colour path stayed gameable.

    ``gate`` reads per-colour recall to detect the collapse ADR-0003 names, so an unanchored
    per-vowel path made the report's aggregate strict and its collapse check trivially
    fooled: the same vowels-only decode scored 0.0 pooled and 1.0 on every colour.
    """
    vowels_only = "".join(c for c in REFERENCE if c in {FATHA, DAMMA, KASRA})

    report = score_windows([vowels_only], [REFERENCE], "m")

    assert report.counts.recall == 0.0
    for colour in ("fatha", "damma", "kasra"):
        counts = report.per_vowel[colour]
        assert counts.matched == 0, f"{colour} credited a vowel with no carrier"
        assert counts.recall == 0.0


def test_per_vowel_matched_never_exceeds_the_pooled_matched_total():
    """The arithmetic that exposed the defect, pinned as an invariant.

    Both paths anchor on the same rule, so each colour's matches must partition the pooled
    matches. In the full artifact they differed by exactly ``unanchored``.
    """
    decode = REFERENCE.replace(f"\u0644\u0642{FATHA}", f"\u0644\u0643{FATHA}")

    report = score_windows([decode], [REFERENCE], "m")

    per_vowel_matched = sum(c.matched for c in report.per_vowel.values())
    assert per_vowel_matched == report.counts.matched
