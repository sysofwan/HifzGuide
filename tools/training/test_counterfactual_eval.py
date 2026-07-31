"""Tests for the counterfactual scorer.

This is the measurement that decides #10, so the tests simulate each hypothesis end to end
and check the verdict separates them — and pin the guards that stop an uninformative
recording from being counted as evidence.
"""

import pytest

from training.counterfactual_eval import (
    FOLLOWED_AUDIO,
    FOLLOWED_TEXT,
    MAX_REGRESSION,
    NO_VOWEL,
    OTHER_VOWEL,
    classify,
    CERTIFICATION_POWER,
    certification_power,
    compare_to_baseline,
    paired_score_interval,
    required_items,
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
    """A net regression under the margin but too thinly sampled to certify.

    2 regressed against 1 recovered is a 2.6% net — inside the 5% margin, so volume could
    still settle it — but at 39 items the bound is nowhere near. The report must say how much
    volume, rather than the bare "more items needed" that sent #60 chasing an unreachable set.
    """
    base = _outcome_rows([FOLLOWED_AUDIO] * 38 + [FOLLOWED_TEXT])
    tuned = _outcome_rows([FOLLOWED_TEXT] * 2 + [FOLLOWED_AUDIO] * 36 + [FOLLOWED_AUDIO])
    comparison = compare_to_baseline(tuned, base)
    assert comparison["regressed"] == 2 and comparison["recovered"] == 1
    assert comparison["equality_finding"] == "inconclusive"
    assert comparison["non_inferiority_certified"] is False
    assert comparison["items_needed_at_observed_rate"] > 39


def test_a_checkpoint_more_of_the_same_audio_cannot_rescue_is_not_called_inconclusive():
    """The rung3_v2 shape: 4 regressed vs 1 recovered, a 7.7% net regression.

    "Inconclusive" invites more recording. Recording more *at this rate* cannot help, so the
    finding must say so and hand the decision to a human — while stopping short of calling
    the checkpoint inferior, which this set does not establish either.
    """
    base = _outcome_rows([FOLLOWED_AUDIO] * 38 + [FOLLOWED_TEXT])
    tuned = _outcome_rows([FOLLOWED_TEXT] * 4 + [FOLLOWED_AUDIO] * 34 + [FOLLOWED_AUDIO])

    comparison = compare_to_baseline(tuned, base)

    assert comparison["regressed"] == 4 and comparison["recovered"] == 1
    assert comparison["equality_finding"] == "above_margin"
    assert comparison["items_needed_at_observed_rate"] is None
    assert "cannot certify" in comparison["detail"]
    assert "unresolved" in comparison["detail"], "not a finding of inferiority"


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


def test_regressions_offset_by_recoveries_certify_once_the_set_is_large_enough():
    """ADR-0006's loosening, stated as a test.

    The rule this replaced also demanded ``b <= c``, so this set — 20 regressions against 10
    recoveries — could never certify however much audio was collected. Non-inferiority is a
    claim about the *net* difference: 1% net over 1,000 paired items is comfortably inside a
    5% margin, and refusing it was a zero-tolerance rule wearing non-inferiority's clothes.
    """
    base = _outcome_rows([FOLLOWED_AUDIO] * 20 + [FOLLOWED_TEXT] * 10 + [FOLLOWED_AUDIO] * 970)
    tuned = _outcome_rows([FOLLOWED_TEXT] * 20 + [FOLLOWED_AUDIO] * 10 + [FOLLOWED_AUDIO] * 970)

    comparison = compare_to_baseline(tuned, base)

    assert comparison["regressed"] == 20 and comparison["recovered"] == 10
    assert comparison["net_difference"] == 0.01
    assert comparison["non_inferiority_certified"] is True


def test_a_certified_set_is_never_described_as_needing_more_items():
    """A certified set that still has ``b > c`` must not be filed under "underpowered".

    The same 20-vs-10 set is *powered* — that is the point of measuring the net difference —
    yet the finding used to read "directional but underpowered; certifying at this rate would
    take 150 paired items" over a set of 1,000. Advice to collect more audio for an already
    settled result is the exact failure #60 was opened about, pointed the other way.
    """
    base = _outcome_rows([FOLLOWED_AUDIO] * 20 + [FOLLOWED_TEXT] * 10 + [FOLLOWED_AUDIO] * 970)
    tuned = _outcome_rows([FOLLOWED_TEXT] * 20 + [FOLLOWED_AUDIO] * 10 + [FOLLOWED_AUDIO] * 970)

    comparison = compare_to_baseline(tuned, base)

    assert comparison["non_inferiority_certified"] is True
    assert comparison["equality_finding"] == "within_margin"
    assert "underpowered" not in comparison["detail"]


def test_the_recorded_forty_one_item_set_still_certifies_nothing():
    """The rung1_v3 result — 2 regressed, 0 recovered on 41 paired items.

    The loosening above must not reach back and pass the sets that motivated it. With no
    recoveries the net-difference bound is just the Wilson bound on b/n, so it lands where it
    always did: 16%, three times the margin.
    """
    base = _outcome_rows([FOLLOWED_AUDIO] * 41)
    tuned = _outcome_rows([FOLLOWED_TEXT] * 2 + [FOLLOWED_AUDIO] * 39)

    comparison = compare_to_baseline(tuned, base)

    assert comparison["net_difference_ci95"][1] == pytest.approx(0.1614, abs=1e-4)
    assert comparison["non_inferiority_certified"] is False


def test_a_flawless_baseline_makes_recoveries_impossible():
    """Why the old ``b <= c`` clause could never be satisfied, not merely rarely.

    ``c`` counts items the base model got wrong and the fine-tune got right. Base silently
    corrects nothing on the recorded set, so ``c`` is pinned at zero and ``b <= c`` collapses
    to ``b == 0``. Concordant items move neither count, so no amount of extra recording could
    ever have changed it — the withdrawn "collect ~35 more items" advice was unreachable.
    """
    base = _outcome_rows([FOLLOWED_AUDIO] * 400)
    tuned = _outcome_rows([FOLLOWED_TEXT] * 2 + [FOLLOWED_AUDIO] * 398)

    comparison = compare_to_baseline(tuned, base)

    assert comparison["recovered"] == 0
    assert comparison["regressed"] == 2
    assert comparison["non_inferiority_certified"] is True


def test_a_paired_interval_without_discordant_pairs_still_costs_sample_size():
    """The trap a Wald interval would have walked into.

    Wald's width is proportional to ``b + c``, so with no discordant pairs it is zero and ten
    items would "certify" — looser than the rule being replaced, which is the opposite of the
    intent. The score interval reduces to the Wilson bound instead, so certifying zero
    regressions still costs 73 items.
    """
    assert paired_score_interval(0, 0, 10)[1] == pytest.approx(0.2775, abs=1e-4)
    assert paired_score_interval(0, 0, 72)[1] > MAX_REGRESSION
    assert paired_score_interval(0, 0, 73)[1] <= MAX_REGRESSION


def test_the_power_calculation_replaces_the_withdrawn_seventy_three():
    """~73 was the sample size for a *flawless* run, never for the observed regressions.

    Stating it per assumed discordance rate is what #60 asks for, because the cost explodes as
    the rate approaches the margin.
    """
    assert required_items(0.0) == 73
    assert required_items(0.01) == 173
    assert required_items(0.02) == 337
    assert required_items(0.03) == 826


def test_a_sample_size_is_quoted_at_a_stated_power_not_at_a_coin_flip():
    """Asking whether the single most likely table certifies is not a power calculation.

    Round the assumed rate to whole items, test that one table, take the smallest n that
    passes, and you get 202 items at a 2% rate — which actually certifies 62% of the time.
    Half of a recollection built on that number would come back unsettled, having spent the
    audio. The quoted size must clear a stated power.
    """
    assert certification_power(202, 0.02) == pytest.approx(0.62, abs=0.01)
    assert certification_power(required_items(0.02), 0.02) >= CERTIFICATION_POWER


def test_the_quoted_size_is_one_you_cannot_fall_off_by_recording_more():
    """Power is sawtoothed in n: the certifying threshold is an integer, so power leaps when
    it increments and decays until the next leap.

    At a 1% rate, 142 items reach 83% power and 160 items — MORE audio — fall back to 78%.
    Quoting 142 would send a recollection to a target it can miss by overshooting, so the
    answer must be a size the target holds from, not the first size to touch it.
    """
    assert certification_power(142, 0.01) > CERTIFICATION_POWER
    assert certification_power(160, 0.01) < CERTIFICATION_POWER, "more items, less power"

    n = required_items(0.01)
    assert n > 160
    assert all(
        certification_power(m, 0.01) >= CERTIFICATION_POWER
        for m in range(n, 2 * n + 1, max(1, n // 64))
    )


def test_the_search_does_not_step_over_its_own_limit():
    """Doubling from 1 lands on powers of two, which need not straddle the limit usefully.

    The last power of two under a limit can fail while the limit itself passes; stepping over
    it and giving up would report a reachable sample size as unreachable, and ``None`` is what
    the report renders as "recording more audio cannot certify it".
    """
    assert required_items(0.01, limit=400) == 173
    assert required_items(0.01, limit=128) is None


def test_a_net_regression_at_or_above_the_margin_is_unreachable_at_that_rate():
    """rung3_v2: 4 of 41, a 9.8% net regression. Recording more at that rate cannot rescue it.

    The claim is about the assumed rate, not the checkpoint: certification probability at a
    net difference above the margin tends to the one-sided alpha, never to the target power.
    What it does NOT establish is that the checkpoint is inferior — 4 of 41 is a loose
    estimate, and a fresh set drawn at a genuinely lower true rate certifies at 202 items.
    """
    assert required_items(4 / 41) is None
    assert required_items(0.05) is None
    assert required_items(0.049) is not None
    assert paired_score_interval(4, 0, 202)[1] <= MAX_REGRESSION, (
        "the same four regressions diluted by concordant items would certify -- 'no sample "
        "size can certify this checkpoint' would be an overclaim"
    )


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


def test_rescoring_refuses_to_silently_drop_a_baseline_comparison(tmp_path, monkeypatch):
    """``--rescore report --out report`` writes over its own input.

    A report decoded with ``--baseline`` carries the gate result in ``vs_baseline``. Rescoring
    without ``--baseline`` would rebuild the report from the stored per-item outcomes alone and
    overwrite the gate away — losing the only thing #10 reads — so refuse instead.
    """
    import json
    import sys

    import training.counterfactual_eval as module

    report = tmp_path / "counterfactual_rung.json"
    report.write_text(
        json.dumps({"model": "m", "items": _outcome_rows([FOLLOWED_AUDIO] * 3),
                    "vs_baseline": {"non_inferiority_certified": False}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys, "argv",
        ["counterfactual_eval", "--rescore", str(report), "--out", str(report)],
    )

    with pytest.raises(SystemExit, match="--baseline"):
        module.main()

    assert json.loads(report.read_text(encoding="utf-8"))["vs_baseline"] is not None


def test_rescoring_rejudges_stored_outcomes_without_a_model(tmp_path, monkeypatch):
    """The rule changed after the audio was decoded, and re-decoding needs a GPU.

    Rescoring must reproduce the summary and gate from the stored per-item outcomes alone, so
    a rule change never costs a decode pass.
    """
    import json
    import sys

    import training.counterfactual_eval as module

    rows = _outcome_rows([FOLLOWED_AUDIO] * 41)
    baseline = tmp_path / "base.json"
    baseline.write_text(json.dumps({"model": "base", "items": rows}), encoding="utf-8")
    report = tmp_path / "rung.json"
    report.write_text(
        json.dumps({"model": "rung", "items": _outcome_rows([FOLLOWED_TEXT] * 2 + [FOLLOWED_AUDIO] * 39)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys, "argv",
        ["counterfactual_eval", "--rescore", str(report), "--baseline", str(baseline),
         "--out", str(report)],
    )

    module.main()

    rescored = json.loads(report.read_text(encoding="utf-8"))
    assert rescored["model"] == "rung", "the decoding model is part of the result, not the rule"
    assert rescored["vs_baseline"]["regressed"] == 2
    assert rescored["vs_baseline"]["non_inferiority_certified"] is False


def test_rescoring_is_the_only_way_to_omit_the_decode_inputs(tmp_path, monkeypatch):
    """The four decode arguments stopped being argparse-required so ``--rescore`` could work.

    Without ``--rescore`` they are still mandatory; dropping them must fail on the argument,
    not later on a None path.
    """
    import sys

    import training.counterfactual_eval as module

    monkeypatch.setattr(sys, "argv", ["counterfactual_eval", "--out", str(tmp_path / "o.json")])

    with pytest.raises(SystemExit, match="--items"):
        module.main()


def test_the_certification_shortcut_agrees_with_the_full_interval():
    """``certifies`` skips the 60-step bisection by testing the margin directly.

    The power calculation asks the question millions of times, so the shortcut is what makes
    it usable — but it is only safe while it agrees exactly with the interval it stands in for.
    """
    import random

    from training.counterfactual_eval import certifies

    random.seed(0)
    for _ in range(2000):
        n = random.randint(1, 600)
        b = random.randint(0, n)
        c = random.randint(0, n - b)
        assert certifies(b, c, n) == (paired_score_interval(b, c, n)[1] <= MAX_REGRESSION), (
            f"disagreement at b={b} c={c} n={n}"
        )


def test_the_non_inferiority_level_is_one_sided_two_and_a_half_percent():
    """z=1.96 is a 95% TWO-sided interval, so the upper bound is a 97.5% one-sided one.

    Naming it matters because the looser conventional choice is a real one: a 95% one-sided
    bound (z=1.645) would certify a flawless run at 52 items rather than 73, and the margin
    would be doing correspondingly less work.
    """
    from training.counterfactual_eval import NON_INFERIORITY_Z

    assert NON_INFERIORITY_Z == 1.96
    assert paired_score_interval(0, 0, 73, 1.96)[1] <= MAX_REGRESSION
    assert paired_score_interval(0, 0, 72, 1.96)[1] > MAX_REGRESSION
    assert paired_score_interval(0, 0, 52, 1.645)[1] <= MAX_REGRESSION
