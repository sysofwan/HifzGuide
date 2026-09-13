"""Tests for the excision differential — where the cut lands, and when the pair is kept.

Everything here is built by hand. A real clip would not let the pause-anchored and
onset-only branches, or the two refusals, be exercised separately, and the re-gate's
verdict is a function of two :class:`~tadabur.scorer.GateResult`\\ s rather than of audio.
"""

from __future__ import annotations

import numpy as np

from tadabur.excision import (
    EXCISION_MIN_RATIO,
    MIN_RETAINED_S,
    REASON_ACCEPTED,
    REASON_DEGENERATE_CUT,
    REASON_NO_SEAM,
    REASON_RATIO_TOO_LOW,
    REASON_REPEAT_REMAINS,
    cut_time,
    excise,
    plan_excision,
    validate_excision,
)
from tadabur.scorer import MAX_INSERTION_RUN, GateResult
from tadabur.seam import BoundaryPause, SeamCoverage

ANCHORED = BoundaryPause(pause_start_s=4.0, pause_end_s=4.5, nearest_gap_s=0.0)
UNANCHORED = BoundaryPause(nearest_gap_s=1.2)


def seam(start_s: float, end_s: float, *, anchored: bool = False, phonemes: int = 8):
    return SeamCoverage(
        audio_filename="a.wav",
        surah_ayah="2:2",
        reciter_id=1,
        run_index=0,
        repeat_phonemes=phonemes,
        query_start=10,
        query_end=10 + phonemes,
        ref_position=10,
        span_start_s=start_s,
        span_end_s=end_s,
        start=ANCHORED if anchored else UNANCHORED,
        end=UNANCHORED,
    )


# --- where a cut lands -------------------------------------------------------------


def test_an_unanchored_edge_cuts_at_the_ctc_onset():
    assert cut_time(UNANCHORED, 3.25) == 3.25


def test_an_anchored_edge_cuts_through_the_middle_of_its_silence():
    # The rim of a silence is adjacent to speech on one side; its middle is as far from
    # both as the cut can get.
    assert cut_time(ANCHORED, 3.25) == 4.25


# --- planning ----------------------------------------------------------------------


def test_a_clip_with_no_repeat_plans_nothing_and_says_so():
    plan = plan_excision([], duration_s=10.0)

    assert not plan.usable
    assert plan.reason == REASON_NO_SEAM
    assert plan.removed_s == 0.0


def test_every_repeat_is_cut_not_only_the_longest():
    # A second repeat left in the control clip would fail the re-gate on its own account,
    # so cutting one and keeping the pair would assert over a clip that still holds the
    # thing the control was supposed to remove.
    plan = plan_excision(
        [seam(6.0, 8.0, phonemes=6), seam(2.0, 3.0, phonemes=12)], duration_s=20.0
    )

    assert [(cut.start_s, cut.end_s) for cut in plan.cuts] == [(2.0, 3.0), (6.0, 8.0)]
    assert plan.removed_s == 3.0
    assert plan.retained_s == 17.0
    assert plan.reason == REASON_ACCEPTED


def test_overlapping_repeats_refuse_the_whole_plan_rather_than_half_of_it():
    plan = plan_excision([seam(2.0, 6.0), seam(5.0, 8.0)], duration_s=20.0)

    assert not plan.usable
    assert plan.reason == REASON_DEGENERATE_CUT


def test_a_cut_that_would_leave_a_fragment_is_refused():
    # Not a control clip, a fragment — and handing the gate a fragment earns an answer
    # about nothing.
    plan = plan_excision([seam(0.1, 9.8)], duration_s=10.0)

    assert not plan.usable
    assert plan.reason == REASON_DEGENERATE_CUT
    assert 10.0 - (9.8 - 0.1) < MIN_RETAINED_S


def test_a_zero_length_cut_is_refused():
    assert plan_excision([seam(4.0, 4.0)], duration_s=10.0).reason == REASON_DEGENERATE_CUT


# --- the audio ---------------------------------------------------------------------


def test_excising_removes_exactly_the_planned_spans_and_splices_the_rest():
    waveform = np.arange(100, dtype=np.float32)  # 10 s at 10 Hz
    plan = plan_excision([seam(3.0, 5.0), seam(7.0, 8.0)], duration_s=10.0)

    cut = excise(waveform, 10, plan)

    assert len(cut) == 70
    assert np.array_equal(cut, np.concatenate([waveform[:30], waveform[50:70], waveform[80:]]))


def test_a_clip_with_nothing_to_cut_comes_back_whole():
    waveform = np.arange(100, dtype=np.float32)

    assert np.array_equal(excise(waveform, 10, plan_excision([], 10.0)), waveform)


# --- the re-gate -------------------------------------------------------------------


def _gate(ratio: float, run: int) -> GateResult:
    return GateResult(passed=False, match_ratio=ratio, max_insertion_run=run)


def test_a_pair_is_kept_when_the_repeat_is_gone_and_what_is_left_gates_clean():
    validation = validate_excision(_gate(0.62, 15), _gate(0.93, 0))

    assert validation.accepted
    assert validation.reason == REASON_ACCEPTED
    assert (validation.match_ratio_before, validation.match_ratio_after) == (0.62, 0.93)
    assert (validation.max_insertion_run_before, validation.max_insertion_run_after) == (15, 0)


def test_a_cut_that_missed_the_repeat_is_discarded():
    # The control clip's whole purpose is to be the recitation without the repeat.
    validation = validate_excision(_gate(0.62, 15), _gate(0.95, MAX_INSERTION_RUN))

    assert not validation.accepted
    assert validation.reason == REASON_REPEAT_REMAINS


def test_a_cut_that_ate_real_recitation_is_discarded():
    # The repeat went, but so did something else: the ratio is the tell, and this is the
    # reason a non-pause-anchored cut is safe — it becomes yield loss, not a finding.
    validation = validate_excision(_gate(0.62, 15), _gate(EXCISION_MIN_RATIO - 0.01, 0))

    assert not validation.accepted
    assert validation.reason == REASON_RATIO_TOO_LOW


def test_the_ratio_bar_is_half_open_at_the_threshold():
    assert validate_excision(_gate(0.6, 9), _gate(EXCISION_MIN_RATIO, 0)).accepted
