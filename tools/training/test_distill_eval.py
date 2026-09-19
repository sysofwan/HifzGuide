"""Tests for the confirmed-stream release gate.

The gate's job is to replay the deployed protocol exactly. If ``scan_ctc`` or the
``midpoint < 25`` split drift from what ``MuaalemInference.predictSplit`` does, the number
it reports stops predicting what Muraja will show and the whole measurement is worthless.
These pin the replay against the Swift semantics; the model-running half needs a GPU and is
exercised by the real eval run.
"""

from __future__ import annotations

import numpy as np
import pytest

from training import distill_eval as de
from training.distill_loss import BLANK_ID, CONFIRM_TIMESTEPS


def _ids(*tokens: int) -> np.ndarray:
    return np.array(tokens, dtype=np.int64)


# --- CTC collapse, mirroring Swift's scanCTC ----------------------------------------


def test_scan_ctc_merges_a_run_into_one_segment():
    segments = de.scan_ctc(_ids(0, 7, 7, 7, 0))
    assert len(segments) == 1
    assert segments[0].token_id == 7
    assert (segments[0].start_step, segments[0].end_step) == (1, 3)


def test_blank_separates_repeated_tokens():
    """The entire point of the CTC blank: 7 blank 7 is two tokens, not one."""
    segments = de.scan_ctc(_ids(7, 0, 7))
    assert [s.token_id for s in segments] == [7, 7]


def test_adjacent_identical_tokens_collapse_to_one():
    segments = de.scan_ctc(_ids(7, 7, 7))
    assert [s.token_id for s in segments] == [7]


def test_blank_runs_are_never_emitted():
    assert de.scan_ctc(_ids(0, 0, 0)) == []


def test_a_segment_running_to_the_end_is_closed():
    """A token still open at the last timestep must still be emitted."""
    segments = de.scan_ctc(_ids(0, 9, 9))
    assert [s.token_id for s in segments] == [9]
    assert segments[0].end_step == 2


def test_midpoint_is_the_centre_of_the_run():
    segments = de.scan_ctc(_ids(0, 5, 5, 5, 0))
    assert segments[0].midpoint == pytest.approx(2.0)


# --- The confirmation split ---------------------------------------------------------


def test_only_segments_before_the_split_are_confirmed():
    """`predictSplit` commits on `seg.midpoint < 25` -- the OLDEST second of the buffer."""
    ids = np.zeros(125, dtype=np.int64)
    ids[5:8] = 7        # midpoint 6 -> confirmed
    ids[60:63] = 9      # midpoint 61 -> not confirmed
    assert de.confirmed_tokens(ids) == [7]


def test_a_segment_straddling_the_split_goes_by_its_midpoint():
    """Not by its start or end -- the midpoint is what Swift compares."""
    ids = np.zeros(125, dtype=np.int64)
    ids[20:32] = 7      # spans the boundary; midpoint 25.5 -> NOT confirmed
    assert de.confirmed_tokens(ids) == []

    ids = np.zeros(125, dtype=np.int64)
    ids[18:30] = 7      # midpoint 23.5 -> confirmed
    assert de.confirmed_tokens(ids) == [7]


def test_confirmation_boundary_is_exclusive():
    ids = np.zeros(125, dtype=np.int64)
    ids[CONFIRM_TIMESTEPS] = 7          # midpoint exactly 25 -> not confirmed
    assert de.confirmed_tokens(ids) == []

    ids = np.zeros(125, dtype=np.int64)
    ids[CONFIRM_TIMESTEPS - 1] = 7      # midpoint 24 -> confirmed
    assert de.confirmed_tokens(ids) == [7]


def test_blank_is_never_confirmed():
    assert de.confirmed_tokens(np.zeros(125, dtype=np.int64)) == []
    assert BLANK_ID == 0


# --- Window scheduling --------------------------------------------------------------


def test_windows_advance_by_one_second():
    starts = de.clip_windows(de.WINDOW_SAMPLES + 3 * de.HOP_SAMPLES)
    assert starts == [0, de.HOP_SAMPLES, 2 * de.HOP_SAMPLES, 3 * de.HOP_SAMPLES]


def test_a_clip_shorter_than_one_window_yields_a_single_padded_pass():
    assert de.clip_windows(de.WINDOW_SAMPLES // 2) == [0]


def test_no_window_runs_off_the_end():
    starts = de.clip_windows(de.WINDOW_SAMPLES + 12345)
    assert all(s + de.WINDOW_SAMPLES <= de.WINDOW_SAMPLES + 12345 for s in starts)


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


# --- The silence flush (PROTOCOL_VERSION confirmed-stream-v2-flush) ---


def test_only_the_last_window_flushes():
    from training.distill_eval import CONFIRM_TIMESTEPS, DEPLOYED_LOGIT_FRAMES, confirm_split_for_window

    assert confirm_split_for_window(0, 4) == CONFIRM_TIMESTEPS
    assert confirm_split_for_window(3, 4) == CONFIRM_TIMESTEPS
    assert confirm_split_for_window(4, 4) == DEPLOYED_LOGIT_FRAMES


def test_a_single_window_clip_is_flushed_entirely():
    """A clip under 5 s is one padded window. Unflushed, it is gated on its first second."""
    from training.distill_eval import DEPLOYED_LOGIT_FRAMES, confirm_split_for_window

    assert confirm_split_for_window(0, 0) == DEPLOYED_LOGIT_FRAMES


def test_the_flush_can_be_turned_off_to_reproduce_the_old_protocol():
    from training.distill_eval import CONFIRM_TIMESTEPS, confirm_split_for_window

    assert confirm_split_for_window(4, 4, flush_tail=False) == CONFIRM_TIMESTEPS
    assert confirm_split_for_window(0, 0, flush_tail=False) == CONFIRM_TIMESTEPS


def test_flushing_adds_the_tail_of_its_own_window():
    """The flush adds this window's steps [25, 125), which no later window exists to decode.

    Scoped deliberately to one window. It does NOT show that no token is emitted twice across
    the corpus: a run straddling the confirmation boundary commits in one window by midpoint
    and again from the next window's opening steps, which predates the flush and is faithful
    to ``predictSplit``. See the module docstring.
    """
    import numpy as np

    from training.distill_eval import CONFIRM_TIMESTEPS, confirmed_tokens

    # A run in the confirmed region and a run in the flushed tail.
    ids = np.array([5] * 10 + [0] * 40 + [7] * 20 + [0] * 55)
    assert confirmed_tokens(ids, CONFIRM_TIMESTEPS) == [5]
    flushed = confirmed_tokens(ids, 125)
    assert flushed == [5, 7]
    assert flushed[: len(confirmed_tokens(ids, CONFIRM_TIMESTEPS))] == [5]


def test_a_segment_straddling_the_boundary_is_emitted_by_both_windows():
    """Documents a real double-emission, so nobody re-derives it as a surprise.

    A run at steps 18-29 has midpoint 23.5 and commits from this window. One second later the
    same audio sits at steps 0-4 of the next window and commits again. This predates the
    flush and is faithful to ``predictSplit``; it is pinned here so the behaviour is a
    recorded property rather than an assumed absence.
    """
    import numpy as np

    from training.distill_eval import CONFIRM_TIMESTEPS, confirmed_tokens

    window_k = np.array([0] * 18 + [9] * 12 + [0] * 95)
    assert confirmed_tokens(window_k, CONFIRM_TIMESTEPS) == [9]

    # Advance one second: 25 timesteps. The run's surviving portion opens the next window.
    window_k1 = np.array([9] * 5 + [0] * 120)
    assert confirmed_tokens(window_k1, CONFIRM_TIMESTEPS) == [9]


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


def test_the_test_half_is_not_scored_unless_it_is_asked_for():
    """Printing the held-out panel on every experiment is how it stops being held out."""
    import inspect

    from training import distill_eval

    source = inspect.getsource(distill_eval.run_evalset)
    assert "evalset.subset(args.split)" in source
    assert 'for split in ("both", "dev", "test")' not in source
