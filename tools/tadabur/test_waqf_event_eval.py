"""Unit tests for the F2 event-level waqf eval + calibration (``tadabur.waqf_event_eval``).

Covers the per-clip lattice reconstruction, the F1 post-processing decision run through
``waqf_postprocess.waqf_events`` (word-edge snap + duration gate) at the calibrated silence
**posterior** threshold, the four ADR-0004 event metrics, the non-gated blank-run reference,
deterministic leak-free calibration (tuned on calibration, scored once on test), and a run
over the real frozen F0 partitions. All torch-free.
"""

from __future__ import annotations

import pytest

from .waqf_event_fixtures import (
    MID_WORD_CLOSURE,
    WAQF,
    WASL,
    WaqfEventEntry,
    _FIXTURE_DIR,
    load_waqf_events,
)
from .waqf_event_eval import (
    _PAD_FRAMES,
    _partition_lattices,
    blank_run_reference,
    calibrate,
    calibration_grid,
    compute_metrics,
    reconstruct_clip,
    run_eval,
)
from .waqf_postprocess import DEFAULT_SILENCE_THRESHOLD


def _entry(clip, idx, predicted, verdict, dur_s, start_s=0.0):
    """A candidate boundary whose silence span lasts ``dur_s`` seconds (0 for a wasl edge)."""
    return WaqfEventEntry(
        clip_id=clip,
        audio_ref=clip,
        surah_ayah="1:1",
        boundary_index=idx,
        word_index=0,
        start_s=start_s,
        end_s=start_s + dur_s,
        predicted=predicted,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Per-clip lattice reconstruction routed through F1's waqf_events.
# ---------------------------------------------------------------------------


def test_wasl_edge_carries_no_silence_and_never_fires():
    entries = [_entry("c", 0, WASL, WASL, 0.0, start_s=1.0)]
    lattice = reconstruct_clip(entries)
    assert lattice.candidate_spans == {}  # zero-width edge → no silence run
    m = compute_metrics(entries, {"c": lattice}, 0.5, label="t")
    assert m.false_waqf_at_wasl == 0


def test_waqf_gap_fires_as_a_stop_through_waqf_events():
    # A 0.6 s stop, well above the 300 ms gate, fires at the VAD's 0.5 argmax threshold.
    entries = [_entry("c", 0, WAQF, WAQF, 0.6, start_s=1.0)]
    m = compute_metrics(entries, _partition_lattices(entries), 0.5, label="t")
    assert m.true_positive == 1
    assert m.false_wasl == 0


def test_mid_word_closure_is_rejected_by_the_word_edge_snap_regardless_of_duration():
    # A long interior silence must NOT fire — waqf_events' snap rejects it as mid-word.
    entries = [_entry("c", 0, MID_WORD_CLOSURE, MID_WORD_CLOSURE, 2.0, start_s=1.0)]
    m = compute_metrics(entries, _partition_lattices(entries), 0.5, label="t")
    assert m.false_waqf_at_closure == 0
    assert m.mid_word_closure_rejection_rate == pytest.approx(1.0)


def test_silence_posterior_threshold_moves_the_duration_gate_at_the_frame_edge():
    # A 300 ms silence sits on the 300 ms gate: at 40 ms quantisation its trailing edge frame
    # is half-covered (P=0.5). Including it (threshold <= 0.5) clears the gate and fires;
    # excluding it (threshold > 0.5) drops one frame below the gate and suppresses the stop.
    entries = [_entry("c", 0, WAQF, WAQF, 0.30, start_s=1.0)]
    lattices = _partition_lattices(entries)
    lenient = compute_metrics(entries, lattices, 0.5, label="t")
    strict = compute_metrics(entries, lattices, 0.6, label="t")
    assert lenient.true_positive == 1
    assert strict.true_positive == 0  # the posterior threshold suppressed the borderline stop


# ---------------------------------------------------------------------------
# The four ADR-0004 event metrics.
# ---------------------------------------------------------------------------


def _mixed_clip():
    # One clip with well-separated boundaries so each silence is scored independently.
    return [
        _entry("c", 0, WAQF, WAQF, 0.6, start_s=1.0),  # true stop, edge, long → TP
        _entry("c", 1, WAQF, WAQF, 0.12, start_s=3.0),  # true stop but short → false-wasl
        _entry("c", 2, MID_WORD_CLOSURE, WAQF, 0.6, start_s=5.0),  # stop mis-snapped → miss + snap fail
        _entry("c", 3, WAQF, WASL, 0.6, start_s=7.0),  # true wasl, detector fired → false-waqf@wasl
        _entry("c", 4, WASL, WASL, 0.0, start_s=9.0),  # true wasl edge → correct not-fire
        _entry("c", 5, MID_WORD_CLOSURE, MID_WORD_CLOSURE, 0.6, start_s=11.0),  # closure rejected
        _entry("c", 6, WAQF, MID_WORD_CLOSURE, 0.6, start_s=13.0),  # closure detector missed → fires
    ]


def test_event_metrics_confusion_and_rates():
    entries = _mixed_clip()
    m = compute_metrics(entries, _partition_lattices(entries), 0.5, label="t")
    assert (m.waqf_total, m.wasl_total, m.closure_total) == (3, 2, 2)
    assert m.true_positive == 1
    assert m.false_wasl == 2  # the short stop + the mid-word-mis-snapped stop
    assert m.false_waqf_at_wasl == 1
    assert m.false_waqf_at_closure == 1  # the missed closure (predicted waqf) fired
    assert m.false_wasl_rate == pytest.approx(2 / 3)
    assert m.false_waqf_rate == pytest.approx(1 / 2)
    assert m.mid_word_closure_rejection_rate == pytest.approx(1 / 2)
    assert m.boundary_snap_accuracy == pytest.approx(2 / 3)
    assert m.precision == pytest.approx(1 / 3)
    assert m.recall == pytest.approx(1 / 3)


def test_snap_accuracy_is_threshold_independent():
    entries = _mixed_clip()
    lattices = _partition_lattices(entries)
    lo = compute_metrics(entries, lattices, 0.2, label="t")
    hi = compute_metrics(entries, lattices, 0.9, label="t")
    assert lo.boundary_snap_accuracy == hi.boundary_snap_accuracy


def test_rates_are_none_for_absent_classes():
    entries = [_entry("c", 0, WAQF, WAQF, 0.6, start_s=1.0)]
    m = compute_metrics(entries, _partition_lattices(entries), 0.5, label="t")
    assert m.false_waqf_rate is None  # no wasl boundaries
    assert m.mid_word_closure_rejection_rate is None


# ---------------------------------------------------------------------------
# Blank-run reference: recorded as explicitly unavailable (non-gated slot).
# ---------------------------------------------------------------------------


def test_blank_run_reference_is_recorded_unavailable_not_substituted():
    # The frozen fixtures carry no CTC decode, so the blank-run slot is recorded as
    # unavailable rather than back-filled with a different silence baseline (cycle-2 review).
    ref = blank_run_reference()
    assert ref.available is False
    assert ref.metrics is None
    assert "No CTC blank-run reference is available" in ref.reason
    doc = ref.to_json_dict()
    assert doc == {"available": False, "reason": ref.reason}
    assert "metrics" not in doc  # nothing substituted under the blank-run name


# ---------------------------------------------------------------------------
# Calibration: deterministic, data-derived, leak-free, posterior-threshold.
# ---------------------------------------------------------------------------


def test_calibration_grid_is_edge_coverages_plus_the_vad_argmax():
    entries = [_entry("c", 0, WAQF, WAQF, 0.30, start_s=1.0)]  # trailing edge frame P=0.5
    grid = calibration_grid(_partition_lattices(entries))
    assert DEFAULT_SILENCE_THRESHOLD in grid
    assert all(0.0 < t <= 1.0 for t in grid)
    assert grid == sorted(grid)  # deterministic ordering


def test_calibrate_picks_a_threshold_suppressing_borderline_false_waqf():
    # Genuine 0.6 s stops fire at any threshold; false-waqf@wasl candidates sit on the 300 ms
    # gate (trailing edge frame P=0.5), so raising the posterior threshold above 0.5 drops the
    # spurious fires while keeping the long stops → higher F1.
    entries = (
        [_entry("c", i, WAQF, WAQF, 0.6, start_s=1.0 + 2 * i) for i in range(4)]
        + [_entry("c", 10 + i, WAQF, WASL, 0.30, start_s=20.0 + 2 * i) for i in range(4)]
    )
    lattices = _partition_lattices(entries)
    threshold, sweep = calibrate(entries, lattices)
    chosen = compute_metrics(entries, lattices, threshold, label="t")
    assert chosen.false_waqf_rate == 0.0  # borderline false-waqf boundaries suppressed
    assert chosen.recall == 1.0  # every genuine stop retained
    assert threshold > DEFAULT_SILENCE_THRESHOLD
    assert calibrate(entries, lattices)[0] == threshold  # deterministic


def test_run_eval_rejects_clip_leakage_between_partitions():
    shared = [_entry("shared", 0, WAQF, WAQF, 0.6, start_s=1.0)]
    with pytest.raises(ValueError, match="leakage"):
        run_eval(shared, shared)


def test_run_eval_scores_test_once_at_the_calibration_threshold():
    calib = [_entry("c0", i, WAQF, WAQF, 0.6, start_s=1.0 + 2 * i) for i in range(3)]
    test = [
        _entry("c1", 0, WAQF, WAQF, 0.6, start_s=1.0),
        _entry("c1", 1, WAQF, WASL, 0.6, start_s=3.0),
    ]
    report = run_eval(calib, test)
    assert report.test.silence_threshold == report.calibrated_silence_threshold
    assert report.calibration.silence_threshold == report.calibrated_silence_threshold
    # The blank-run reference is recorded as unavailable (never gated, not substituted).
    assert report.blank_run_reference.available is False
    doc = report.to_json_dict()
    assert doc["blank_run_reference"] == {
        "available": False,
        "reason": report.blank_run_reference.reason,
    }
    assert "sanity check" in doc["teacher_circularity_note"]
    assert "blank-run" in doc["reference_note"].lower()


# ---------------------------------------------------------------------------
# Real frozen F0 partitions.
# ---------------------------------------------------------------------------


def test_run_eval_on_frozen_partitions_is_leak_free_and_reports_all_metrics():
    calibration = load_waqf_events(_FIXTURE_DIR / "waqf_events.calibration.jsonl")
    test = load_waqf_events(_FIXTURE_DIR / "waqf_events.test.jsonl")
    assert calibration and test
    report = run_eval(calibration, test)
    doc = report.to_json_dict()
    metrics = doc["test"]["metrics"]
    for key in (
        "false_waqf_at_wasl_rate",
        "false_wasl_at_genuine_stop_rate",
        "mid_word_closure_rejection_rate",
        "boundary_snap_accuracy",
    ):
        assert key in metrics
    # The frozen partitions carry the qalqala/hamza rejection set, so it is measurable.
    assert report.test.closure_total + report.calibration.closure_total > 0
    # The gate ran through real F1 post-processing and detected genuine stops.
    assert report.test.true_positive > 0
    # Determinism: identical inputs reproduce the report.
    assert run_eval(calibration, test).to_json_dict() == doc


def test_pad_frames_is_at_least_one_min_speech_span():
    assert _PAD_FRAMES >= 17  # 700 ms / 40 ms
