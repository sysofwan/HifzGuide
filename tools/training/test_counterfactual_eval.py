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
    compare_to_baseline,
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


def test_a_model_that_never_silently_corrects_is_passed():
    summary = summarize(_results(80, 0))
    assert summary["silent_correction_rate"] == 0.0
    ruling = verdict(summary)
    assert ruling["conclusive"] is True and ruling["fit_to_flag_vowel_errors"] is True


def test_a_perfect_score_on_too_small_a_set_still_cannot_clear_the_tolerance():
    """The sample-size floor a 5% tolerance actually implies.

    Even a flawless 0-of-60 leaves a 95% upper bound above 5%, so the set must reach ~73
    scorable items before a clean sweep licenses a pass. This is why 47 items cannot settle
    the question however well the model does on them.
    """
    ruling = verdict(summarize(_results(60, 0)))
    assert ruling["conclusive"] is False
    assert verdict(summarize(_results(73, 0)))["conclusive"] is True


def test_a_reconstructing_model_yields_a_conclusive_negative_verdict():
    ruling = verdict(summarize(_results(4, 43)))
    assert ruling["conclusive"] is True and ruling["fit_to_flag_vowel_errors"] is False
    assert "flag a student's error" in ruling["interpretation"]


def test_following_the_audio_most_of_the_time_is_not_enough_to_pass():
    """The bar is the silent-correction rate, not beating a coin flip.

    A checker that misses 20% of vowel errors follows the audio on 80% of items -- a landslide
    against any majority test -- while being unfit for its actual job.
    """
    summary = summarize(_results(80, 20))
    assert summary["followed_audio_rate"] == 0.8
    ruling = verdict(summary)
    assert ruling["conclusive"] is True and ruling["fit_to_flag_vowel_errors"] is False


def test_a_rate_near_the_tolerance_is_inconclusive_at_this_sample_size():
    """47 items cannot resolve a 5% tolerance -- the verdict must say so, not guess."""
    ruling = verdict(summarize(_results(43, 4)))
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


ALEF, WAW, YEH_MAQSURA = "\u0627", "\u0648", "\u0649"
FEH, THAL = "\u0641", "\u0630"


@pytest.mark.parametrize(
    "word",
    [f"{MEEM}{FATHA}{ALEF}", f"{THAL}{DAMMA}{WAW}", f"{FEH}{KASRA}{YEH_MAQSURA}"],
)
def test_a_word_whose_vowel_is_held_long_is_excluded(word):
    """مَا cannot be recited as مُا, so the take never contains the substituted vowel.

    The generator checked only the phonetizer span, where the carrier is often absorbed, and
    its carrier set omitted alef maqsura — so five such words reached the recording sheet.
    """
    item = dict(_item(), target_word=word)
    segment = {
        "raw_reference_phonemes": word,
        "raw_word_offsets": [0, len(word)],
    }
    result = score_item(item, segment, _decodes(FATHA, DAMMA))
    assert result["excluded_madd"] is True
    assert result["scored"] is False
    assert summarize([result])["excluded_madd"] == 1


def test_a_word_with_a_freely_substitutable_vowel_is_not_excluded():
    assert score_item(_item(), SEGMENT, _decodes(FATHA, DAMMA))["excluded_madd"] is False


@pytest.mark.parametrize(
    "word",
    [
        "\u0671\u0628\u0652\u0646\u064f",  # ٱبْنُ  (43:57, followed by مَرْيَمَ)
        "\u0671\u062f\u0652\u0639\u064f",  # ٱدْعُ  (2:70, followed by لَنَا)
        "\u0671\u0633\u0652\u0645\u064e",  # ٱسْمَ  (87:15, followed by رَبِّهِۦ)
    ],
)
def test_an_ordinary_word_final_haraka_is_not_madd(word):
    """The end of a word is not a carrier — these are valid probes, not madd.

    Treating ``i + 1 >= len(word)`` as elongation discarded these three real items and
    shrank the scorable set from 42 to 39, weakening an already-underpowered comparison.
    Nothing holds a final damma long here; a reciter can say ٱبْنَ as easily as ٱبْنُ.
    """
    item = dict(_item(), target_word=word)
    segment = {"raw_reference_phonemes": word, "raw_word_offsets": [0, len(word)]}
    assert score_item(item, segment, _decodes(FATHA, DAMMA))["excluded_madd"] is False


def test_a_span_matched_on_too_little_of_the_word_is_refused():
    """A sliver of incidental agreement is not evidence about the target word.

    This is how a repeated word elsewhere in the ayah captures the span and hands back the
    wrong copy's vowel, so partial matches are refused rather than guessed at.
    """
    long_word = f"{QAF}{FATHA}{DAL}{MEEM}{NOON}"
    assert vowel_in_span(f"{QAF}{DAMMA}{DAL}{MEEM}{NOON}", long_word, 0, len(long_word)) == DAMMA
    assert vowel_in_span(f"{NOON}", long_word, 0, len(long_word)) is None


def test_an_ambiguous_span_carrying_two_vowels_is_refused():
    """Which vowel belongs to the target word is undecidable, so it is not evidence."""
    decode = f"{QAF}{DAMMA}{DAL}{KASRA}{MEEM}"
    assert vowel_in_span(decode, f"{QAF}{FATHA}{DAL}{MEEM}", 0, 4) is None


def test_the_cli_default_tolerance_is_the_product_tolerance():
    """A stale default here silently converts a failing model into a passing one.

    The CLI once still passed the old 0.5 coin-flip margin into this slot, which reported a
    10% silent-correction rate as fit.
    """
    import argparse
    from training.counterfactual_eval import MAX_SILENT_CORRECTION_RATE, build_parser

    args = build_parser().parse_args(
        ["--items", "i", "--manifest", "m", "--audio-dir", "a", "--model", "x", "--out", "o"]
    )
    assert args.max_silent_correction == MAX_SILENT_CORRECTION_RATE
    assert verdict(summarize(_results(35, 4)), args.max_silent_correction)["conclusive"] is False


def _outcome_rows(outcomes):
    return [
        {"item_id": f"cf{i:03d}", "swap": "a->b", "outcome": o,
         "control_passed": True, "alignment_stable": True, "scored": True}
        for i, o in enumerate(outcomes)
    ]


def test_relaxing_without_touching_tashkeel_is_not_a_regression():
    """The fine-tune's whole purpose is to relax phenomena Muraja ignores.

    Identical tashkeel behaviour must therefore pass, however much else changed.
    """
    rows = _outcome_rows([FOLLOWED_AUDIO] * 38 + [FOLLOWED_TEXT])
    comparison = compare_to_baseline(rows, rows)
    assert comparison["regressed"] == 0
    assert comparison["equality_finding"] == "no_evidence_of_regression"


def test_losing_vowel_errors_the_base_model_flagged_is_a_regression():
    """ADR-0003's named failure: accuracy rises while discrimination collapses."""
    base = _outcome_rows([FOLLOWED_AUDIO] * 40)
    tuned = _outcome_rows([FOLLOWED_TEXT] * 10 + [FOLLOWED_AUDIO] * 30)
    comparison = compare_to_baseline(tuned, base)
    assert comparison["regressed"] == 10 and comparison["recovered"] == 0
    assert comparison["equality_finding"] == "regression"
    assert comparison["mcnemar_exact_p"] < 0.01


def test_a_regression_offset_by_an_equal_recovery_is_not_significant():
    """Churn on individual items is not directional evidence."""
    base = _outcome_rows([FOLLOWED_AUDIO] * 4 + [FOLLOWED_TEXT] * 4 + [FOLLOWED_AUDIO] * 32)
    tuned = _outcome_rows([FOLLOWED_TEXT] * 4 + [FOLLOWED_AUDIO] * 4 + [FOLLOWED_AUDIO] * 32)
    comparison = compare_to_baseline(tuned, base)
    assert comparison["regressed"] == 4 and comparison["recovered"] == 4
    assert comparison["mcnemar_exact_p"] == 1.0
    assert comparison["equality_finding"] == "no_evidence_of_regression"


def test_a_directional_but_underpowered_difference_is_not_called_clean():
    """The rung3 case: 4 regressed vs 1 recovered is not significant, but not nothing."""
    base = _outcome_rows([FOLLOWED_AUDIO] * 38 + [FOLLOWED_TEXT])
    tuned = _outcome_rows([FOLLOWED_TEXT] * 4 + [FOLLOWED_AUDIO] * 34 + [FOLLOWED_AUDIO])
    comparison = compare_to_baseline(tuned, base)
    assert comparison["regressed"] == 4 and comparison["recovered"] == 1
    assert comparison["equality_finding"] == "inconclusive"


def test_only_items_both_models_scored_are_compared():
    """An item one model could not score is not evidence about the other."""
    base = _outcome_rows([FOLLOWED_AUDIO] * 3)
    tuned = _outcome_rows([FOLLOWED_AUDIO] * 3)
    tuned[2]["scored"] = False
    assert compare_to_baseline(tuned, base)["paired_items"] == 2


def test_finding_no_regression_is_not_the_same_as_certifying_non_inferiority():
    """The distinction that let me read an underpowered result as a pass on #10.

    Two identical models over 42 items produce zero regressions — there is genuinely no
    evidence of one — but the upper bound is still ~8%, above the 5% margin. Nothing is
    certified. Non-inferiority is a claim about the bound, never the point estimate.
    """
    rows = _outcome_rows([FOLLOWED_AUDIO] * 42)

    comparison = compare_to_baseline(rows, rows)

    assert comparison["equality_finding"] == "no_evidence_of_regression"
    assert comparison["regression_upper95"] > comparison["max_regression"]
    assert comparison["non_inferiority_certified"] is False


def test_churn_in_both_directions_certifies_nothing_even_at_zero_net():
    """5 regressed against 5 recovered nets to zero while the model churns a quarter of the set."""
    base = _outcome_rows([FOLLOWED_AUDIO] * 5 + [FOLLOWED_TEXT] * 5 + [FOLLOWED_AUDIO] * 32)
    tuned = _outcome_rows([FOLLOWED_TEXT] * 5 + [FOLLOWED_AUDIO] * 5 + [FOLLOWED_AUDIO] * 32)

    comparison = compare_to_baseline(tuned, base)

    assert comparison["regressed"] == comparison["recovered"] == 5
    assert comparison["non_inferiority_certified"] is False


def test_a_large_enough_clean_set_does_certify():
    """Only volume closes it: zero regressions over 200 items clears the 5% margin."""
    rows = _outcome_rows([FOLLOWED_AUDIO] * 200)

    comparison = compare_to_baseline(rows, rows)

    assert comparison["regression_upper95"] <= comparison["max_regression"]
    assert comparison["non_inferiority_certified"] is True


def test_an_item_the_two_alignments_disagree_about_is_not_scored(monkeypatch):
    """An unstable item's verdict depends on which reference the decode was aligned against.

    Those two references are the canonical and the spoken text -- the two hypotheses under
    test -- so scoring such an item would let the answer depend on the question. cf026 was
    exactly this: alignment_stable false, yet counted in the base model's headline. It must be
    reported for audit and excluded from the denominator, never silently resolved in favour of
    whichever reference happened to win.
    """
    import training.counterfactual_eval as module

    canonical, spoken = FATHA, DAMMA
    # The counterfactual take reads as the spoken vowel against one reference and the
    # canonical vowel against the other -- followed_audio or followed_text, take your pick.
    calls = iter([canonical, canonical, spoken, canonical])
    monkeypatch.setattr(module, "vowel_in_span", lambda *a, **k: next(calls))

    row = module.score_item(_item(canonical, spoken), SEGMENT, _decodes(canonical, spoken))

    assert row["control_passed"] is True, "the control take is clean; only stability is at issue"
    assert row["alignment_stable"] is False
    assert row["scored"] is False
    assert row["outcome"] == FOLLOWED_AUDIO, "still reported, so the exclusion can be audited"
