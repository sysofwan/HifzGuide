"""Unit tests for the F2 event-level waqf eval + calibration (``tadabur.waqf_event_eval``).

Covers the per-boundary F1 post-processing decision (word-edge snap + duration gate), the
four ADR-0004 event metrics, deterministic leak-free calibration (tuned on calibration,
scored once on test), and a run over the real frozen F0 partitions. All torch-free.
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
    calibrate,
    calibration_grid,
    compute_metrics,
    predict_waqf,
    run_eval,
    silence_ms,
    snapped_to_word_edge,
)
from .waqf_postprocess import DEFAULT_MIN_SILENCE_MS


def _entry(clip, idx, predicted, verdict, dur_s, word_index=0):
    """A candidate boundary whose silence span lasts ``dur_s`` seconds (0 for a wasl edge)."""
    return WaqfEventEntry(
        clip_id=clip,
        audio_ref=clip,
        surah_ayah="1:1",
        boundary_index=idx,
        word_index=word_index,
        start_s=0.0,
        end_s=dur_s,
        predicted=predicted,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Per-boundary F1 post-processing decision.
# ---------------------------------------------------------------------------


def test_silence_ms_quantises_to_40ms_lattice():
    # 0.5 s = 12.5 frames → 12 frames at nearest-frame rounding → 480 ms on the 40 ms lattice.
    assert silence_ms(_entry("c", 0, WAQF, WAQF, 0.5)) == 12 * 40


def test_wasl_edge_is_zero_duration_and_never_fires():
    edge = _entry("c", 0, WASL, WASL, 0.0)
    assert silence_ms(edge) == 0
    assert predict_waqf(edge, DEFAULT_MIN_SILENCE_MS) is False


def test_mid_word_closure_is_not_snapped_to_an_edge_and_never_fires():
    # A long interior silence still must not fire — the snap rejects it regardless of duration.
    closure = _entry("c", 0, MID_WORD_CLOSURE, MID_WORD_CLOSURE, 2.0)
    assert snapped_to_word_edge(closure) is False
    assert predict_waqf(closure, DEFAULT_MIN_SILENCE_MS) is False


def test_waqf_edge_fires_only_above_the_duration_threshold():
    stop = _entry("c", 0, WAQF, WAQF, 0.5)  # 480 ms
    assert predict_waqf(stop, 300) is True
    assert predict_waqf(stop, 500) is False  # threshold above the pause suppresses it


# ---------------------------------------------------------------------------
# The four ADR-0004 event metrics.
# ---------------------------------------------------------------------------


def _mixed_partition():
    return [
        _entry("c", 0, WAQF, WAQF, 0.6),  # true stop, edge, long → TP
        _entry("c", 1, WAQF, WAQF, 0.2),  # true stop but short → false-wasl (missed)
        _entry("c", 2, MID_WORD_CLOSURE, WAQF, 0.6),  # true stop mis-snapped mid-word → miss + snap fail
        _entry("c", 3, WAQF, WASL, 0.6),  # true wasl, detector fired → false-waqf@wasl
        _entry("c", 4, WASL, WASL, 0.0),  # true wasl edge → correct not-fire
        _entry("c", 5, MID_WORD_CLOSURE, MID_WORD_CLOSURE, 0.6),  # closure correctly rejected
        _entry("c", 6, WAQF, MID_WORD_CLOSURE, 0.6),  # closure detector missed → fires → not rejected
    ]


def test_event_metrics_confusion_and_rates():
    m = compute_metrics(
        _mixed_partition(), lambda e: predict_waqf(e, 300),
        label="t", min_silence_ms=300,
    )
    assert (m.waqf_total, m.wasl_total, m.closure_total) == (3, 2, 2)
    assert m.true_positive == 1
    assert m.false_wasl == 2  # the short stop + the mid-word-mis-snapped stop
    assert m.false_waqf_at_wasl == 1
    assert m.false_waqf_at_closure == 1  # the missed closure fired
    # false-wasl@genuine-stop = 2/3; false-waqf@wasl = 1/2.
    assert m.false_wasl_rate == pytest.approx(2 / 3)
    assert m.false_waqf_rate == pytest.approx(1 / 2)
    # mid-word-closure rejection: only 1 of 2 closures was rejected (the other fired).
    assert m.mid_word_closure_rejection_rate == pytest.approx(1 / 2)
    # boundary-snap accuracy: 2 of 3 genuine stops were placed at a word edge.
    assert m.boundary_snap_accuracy == pytest.approx(2 / 3)
    assert m.precision == pytest.approx(1 / 3)  # 1 TP of 3 fired
    assert m.recall == pytest.approx(1 / 3)


def test_snap_accuracy_is_threshold_independent():
    entries = _mixed_partition()
    lo = compute_metrics(entries, lambda e: predict_waqf(e, 300), label="t", min_silence_ms=300)
    hi = compute_metrics(entries, lambda e: predict_waqf(e, 900), label="t", min_silence_ms=900)
    assert lo.boundary_snap_accuracy == hi.boundary_snap_accuracy


def test_rates_are_none_for_absent_classes():
    m = compute_metrics(
        [_entry("c", 0, WAQF, WAQF, 0.6)], lambda e: predict_waqf(e, 300),
        label="t", min_silence_ms=300,
    )
    assert m.false_waqf_rate is None  # no wasl boundaries
    assert m.mid_word_closure_rejection_rate is None


# ---------------------------------------------------------------------------
# Calibration: deterministic, data-derived, leak-free.
# ---------------------------------------------------------------------------


def test_calibration_grid_is_edge_durations_plus_vad_anchor():
    entries = [
        _entry("c", 0, WAQF, WAQF, 0.5),  # 480 ms edge
        _entry("c", 1, WASL, WASL, 0.0),  # zero-width, excluded
        _entry("c", 2, MID_WORD_CLOSURE, WAQF, 2.0),  # interior, excluded from grid
    ]
    assert calibration_grid(entries) == sorted({480, DEFAULT_MIN_SILENCE_MS})


def test_calibrate_picks_threshold_separating_false_waqf():
    # Short (200 ms) wasl fires and long (600 ms) genuine stops fire at 300 ms; raising the
    # threshold to drop the short false-waqf while keeping the stops maximises F1.
    entries = [
        _entry("c", i, WAQF, WAQF, 0.6) for i in range(4)
    ] + [
        _entry("c", 10 + i, WAQF, WASL, 0.2) for i in range(4)
    ]
    threshold, sweep = calibrate(entries)
    chosen = compute_metrics(entries, lambda e: predict_waqf(e, threshold), label="t", min_silence_ms=threshold)
    assert chosen.false_waqf_rate == 0.0  # the short false-waqf boundaries are suppressed
    assert chosen.recall == 1.0  # every genuine stop retained
    # Deterministic: a rerun yields the identical operating point.
    assert calibrate(entries)[0] == threshold


def test_run_eval_rejects_clip_leakage_between_partitions():
    shared = [_entry("shared", 0, WAQF, WAQF, 0.6)]
    with pytest.raises(ValueError, match="leakage"):
        run_eval(shared, shared)


def test_run_eval_scores_test_once_at_the_calibration_threshold():
    calib = [_entry("c0", i, WAQF, WAQF, 0.6) for i in range(3)]
    test = [_entry("c1", 0, WAQF, WAQF, 0.6), _entry("c1", 1, WAQF, WASL, 0.6)]
    report = run_eval(calib, test)
    assert report.test.min_silence_ms == report.calibrated_min_silence_ms
    assert report.calibration.min_silence_ms == report.calibrated_min_silence_ms
    # The reference operating point is threshold-free and recorded (not gated).
    assert report.test_reference.min_silence_ms is None
    doc = report.to_json_dict()
    assert "teacher_circularity_note" in doc and "sanity check" in doc["teacher_circularity_note"]


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
    # Determinism: identical inputs reproduce the report.
    assert run_eval(calibration, test).to_json_dict() == doc
