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


def test_compare_streams_survives_an_empty_corpus():
    report = de.compare_streams([])
    assert report.num_clips == 0
    assert report.char_accuracy == pytest.approx(1.0)


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
