"""Tests for the confirmed-stream release gate's metrics.

The replay of the deployed protocol it scores is pinned in ``test_decoding``.
"""

from __future__ import annotations

import pytest

from training.decode_evalset import SCHEMA_VERSION

from training import distill_eval as de


# --- Stream comparison --------------------------------------------------------------


def test_levenshtein_basics():
    assert de.levenshtein([], []) == 0
    assert de.levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert de.levenshtein([1, 2, 3], []) == 3
    assert de.levenshtein([1, 2, 3], [1, 4, 3]) == 1       # substitution
    assert de.levenshtein([1, 2, 3], [1, 2]) == 1          # deletion
    assert de.levenshtein([1, 2], [1, 2, 3]) == 1          # insertion


def test_identical_streams_report_perfect_agreement():
    pairs = [([1, 2, 3], [1, 2, 3]), ([4, 5], [4, 5])]
    report = de.compare_streams(pairs)
    assert report.exact_match == pytest.approx(1.0)
    assert report.char_accuracy == pytest.approx(1.0)
    assert report.total_edits == 0


def test_char_accuracy_pools_edits_over_the_corpus():
    """Pooled, not averaged per clip -- so one short clip cannot swing the number.

    Two clips: 10 teacher tokens with 1 edit, and 2 tokens with 1 edit. Pooled accuracy is
    1 - 2/12 = 0.833; a per-clip mean would give 1 - (0.1 + 0.5)/2 = 0.70.
    """
    pairs = [
        (list(range(10)), [99] + list(range(1, 10))),
        ([1, 2], [1, 3]),
    ]
    report = de.compare_streams(pairs)
    assert report.total_teacher_tokens == 12
    assert report.total_edits == 2
    assert report.char_accuracy == pytest.approx(1 - 2 / 12)


def test_exact_match_is_all_or_nothing_per_clip():
    pairs = [([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [1, 2, 4])]
    assert de.compare_streams(pairs).exact_match == pytest.approx(0.5)


def test_an_empty_student_stream_is_scored_not_skipped():
    """A blank-collapsed student emits nothing; that must read as 0% accuracy, not 100%."""
    report = de.compare_streams([([1, 2, 3, 4], [])])
    assert report.exact_match == pytest.approx(0.0)
    assert report.char_accuracy == pytest.approx(0.0)
    assert report.mean_student_length == pytest.approx(0.0)


def test_an_empty_corpus_scores_zero_not_one():
    """Scoring nothing must not read as perfect agreement.

    This previously returned 1.0 -- `1 - 0/1` -- so a run that skipped every clip (wrong
    --audio-root, non-16 kHz staging directory, unreadable files) printed
    "clips evaluated 0 / char accuracy 100.00%". The old test asserted that value, pinning
    the deceptive behaviour instead of catching it.
    """
    report = de.compare_streams([])
    assert report.num_clips == 0
    assert report.char_accuracy == pytest.approx(0.0)


# --- Head health: is a stuck run a head problem or an encoder problem? ---------------


def _head(blank_bias, other_bias, blank_norm=0.4, other_norm=0.4):
    return de.HeadHealth(
        blank_bias=blank_bias,
        other_bias_mean=other_bias,
        other_bias_max=other_bias,
        blank_weight_norm=blank_norm,
        other_weight_norm_mean=other_norm,
        other_weight_norm_max=other_norm,
    )


def test_a_large_blank_bias_reads_as_a_head_shortcut():
    """The majority-class shortcut: more training will not fix this."""
    head = _head(blank_bias=4.0, other_bias=-0.2)
    assert head.bias_lead == pytest.approx(4.2)
    assert head.is_degenerate()


def test_the_measured_h384_head_is_not_degenerate():
    """Step 2000 of the real run: blank led by 0.004, so the collapse was upstream.

    Pinned because it is the observation that ruled out a whole class of interventions --
    if this ever reads degenerate, the diagnosis flips.
    """
    head = _head(blank_bias=0.011, other_bias=0.007)
    assert head.bias_lead == pytest.approx(0.004, abs=1e-6)
    assert not head.is_degenerate()


def test_bias_lead_is_measured_against_the_best_competitor():
    """Against the max, not the mean -- the mean would hide one strong rival class."""
    head = de.HeadHealth(
        blank_bias=1.0,
        other_bias_mean=-2.0,
        other_bias_max=0.9,
        blank_weight_norm=0.4,
        other_weight_norm_mean=0.4,
        other_weight_norm_max=0.4,
    )
    assert head.bias_lead == pytest.approx(0.1)
    assert not head.is_degenerate()


# --- Decode agreement: the distillation metric ---


def test_decode_agreement_pools_edits_and_reports_the_tail_beside_them():
    """The pooled number is dominated by long clips; the median is not, and they move apart."""
    from training.distill_eval import score_decode_agreement

    # Three short perfect clips and one long bad one, all from different reciters.
    per_clip = [(0, 10, 1), (0, 10, 2), (0, 10, 3), (60, 200, 4)]
    report = score_decode_agreement(per_clip)

    assert report.num_clips == 4
    assert report.num_reciters == 4
    assert report.char_accuracy == pytest.approx(1 - 60 / 230)
    assert report.exact_match == pytest.approx(0.75)
    # Median per-clip error is 0, while pooled accuracy is 74% -- the point of showing both.
    assert report.median_clip_error == pytest.approx(0.0)
    assert report.p90_clip_error > 0.0


def test_decode_agreement_of_an_empty_set_is_zero_not_one():
    """`1 - 0/1` is 1.0, which prints as perfect agreement produced by scoring nothing."""
    from training.distill_eval import score_decode_agreement

    report = score_decode_agreement([])
    assert report.char_accuracy == 0.0
    assert report.num_clips == 0


def test_decode_agreement_interval_widens_when_the_errors_cluster_by_reciter():
    from training.distill_eval import score_decode_agreement

    # Same pooled accuracy either way; only the clustering differs.
    clustered = []
    spread = []
    for reciter in range(20):
        for i in range(5):
            clustered.append((10 if reciter < 4 else 0, 50, reciter))
            spread.append((10 if i == 0 else 0, 50, reciter))
    assert sum(e for e, _, _ in clustered) == sum(e for e, _, _ in spread)

    wide = score_decode_agreement(clustered)
    narrow = score_decode_agreement(spread)
    assert wide.char_accuracy == pytest.approx(narrow.char_accuracy)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


def test_a_single_reciter_falls_back_to_the_unclustered_interval():
    from training.distill_eval import score_decode_agreement

    report = score_decode_agreement([(5, 100, 7), (5, 100, 7)])
    assert report.num_reciters == 1
    assert report.ci_low < report.char_accuracy < report.ci_high


def test_a_paired_comparison_reports_the_pooled_delta_not_a_sign_test():
    """Counting clips that improved is a sign test; the claim made from it is a rate change.

    Constructed so the two disagree: this checkpoint is worse on more clips, but by a little,
    while being much better on a few. The sign test would call that a regression.
    """
    from training.distill_eval import paired_reciter_bootstrap

    rows = []
    for reciter in range(30):
        # Three clips slightly worse ...
        for _ in range(3):
            rows.append((11, 10, 100, reciter))
        # ... and one hugely better.
        rows.append((5, 60, 100, reciter))
    result = paired_reciter_bootstrap(rows)

    assert result.clips_further > result.clips_closer  # the sign test's view
    assert result.delta > 0  # and the pooled rate says the opposite
    assert result.significant
    assert result.ci_low > 0


def test_the_paired_interval_widens_when_the_difference_clusters_by_reciter():
    from training.distill_eval import paired_reciter_bootstrap

    # Identical totals; only the arrangement across reciters differs.
    clustered, spread = [], []
    for reciter in range(20):
        for i in range(10):
            clustered.append((0 if reciter < 10 else 20, 10, 100, reciter))
            spread.append((0 if i < 5 else 20, 10, 100, reciter))
    assert sum(r[0] for r in clustered) == sum(r[0] for r in spread)
    wide = paired_reciter_bootstrap(clustered)
    narrow = paired_reciter_bootstrap(spread)
    assert wide.delta == pytest.approx(narrow.delta)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


def test_an_identical_pair_has_no_difference_and_no_significance():
    from training.distill_eval import paired_reciter_bootstrap

    rows = [(7, 7, 100, i // 3) for i in range(60)]
    result = paired_reciter_bootstrap(rows)
    assert result.delta == 0.0
    assert not result.significant
    assert paired_reciter_bootstrap([]).delta == 0.0


def test_the_two_halves_are_disjoint_and_selectable():
    """`--split dev` must not be able to see a test-half clip, and vice versa."""
    from training.decode_evalset import EvalClip, EvalSet, reciter_split

    clips = [
        EvalClip(f"c{i}.wav", "78:1", i, 20, 4.0, reciter_split(i), "abc")
        for i in range(200)
    ]
    evalset = EvalSet(SCHEMA_VERSION, tuple(clips), 200, 0, {})
    dev = {c.filename for c in evalset.subset("dev")}
    test = {c.filename for c in evalset.subset("test")}

    assert dev and test
    assert not (dev & test)
    assert dev | test == {c.filename for c in evalset.subset("both")}
    # And the split follows the reciter, not the clip.
    assert all(reciter_split(c.reciter_id) == c.split for c in evalset.clips)
