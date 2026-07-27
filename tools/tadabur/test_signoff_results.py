"""Unit tests for the fine-tune sign-off results view (``tadabur.signoff_results``, #37).

Covers the three report sections (E ablation ladder, F2 event eval, H integration) — their
available / unavailable / malformed paths — and the readiness go/no-go the #10 human reads first.
All pure: reports are written to ``tmp_path`` JSON, no socket bound.
"""

from __future__ import annotations

import json

from .signoff_results import (
    build_signoff_view,
    event_eval_section,
    integration_section,
    ladder_section,
    readiness,
)


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _ladder_report(regressed=False, disc_delta=0.0, lever=None):
    return {
        "ladder": {
            "rungs": {
                "segmented_phoneme_only": {"should_accept_recall": 0.9, "should_reject_discrimination": 0.8},
                "whole_clip_phoneme_only": {"should_accept_recall": 0.91, "should_reject_discrimination": 0.8 + disc_delta},
                "whole_clip_joint_waqf": {"should_accept_recall": 0.91, "should_reject_discrimination": 0.8 + disc_delta},
            },
            "transitions": {
                "whole_clip_move": {"from_rung": "segmented_phoneme_only", "to_rung": "whole_clip_phoneme_only",
                                    "recall_delta": 0.01, "discrimination_delta": disc_delta},
                "waqf_head_addition": {"from_rung": "whole_clip_phoneme_only", "to_rung": "whole_clip_joint_waqf",
                                       "recall_delta": 0.0, "discrimination_delta": 0.0},
            },
        },
        "whole_clip_move_regressed": regressed,
        "recommended_lora_lever": lever,
    }


def _event_report():
    metrics = {
        "precision": 0.9, "recall": 0.85, "f1": 0.874,
        "false_waqf_at_wasl_rate": 0.02, "false_wasl_at_genuine_stop_rate": 0.1,
        "mid_word_closure_rejection_rate": 0.95, "boundary_snap_accuracy": 0.98,
    }
    counts = {"waqf_total": 20, "wasl_total": 30, "closure_total": 14, "true_positive": 17}
    partition = {"silence_threshold": 0.5, "counts": counts, "metrics": metrics}
    return {
        "calibrated_silence_threshold": 0.5,
        "duration_gate_ms": 300,
        "test": partition,
        "calibration": partition,
        "blank_run_reference": {"available": False, "reason": "no CTC decode"},
        "teacher_circularity_note": "frame-F1 is a sanity check only",
    }


# --- ladder section (E) ------------------------------------------------------


def test_ladder_section_reads_rungs_transitions_and_regression(tmp_path):
    section = ladder_section(_write(tmp_path / "e.json", _ladder_report(disc_delta=0.01)))
    assert section["available"] is True
    assert section["whole_clip_move_regressed"] is False
    assert section["transitions"]["whole_clip_move"]["discrimination_delta"] == 0.01
    assert "whole_clip_joint_waqf" in section["rungs"]


def test_ladder_section_missing_report_unavailable():
    section = ladder_section(None)
    assert section["available"] is False
    assert "not provided" in section["reason"]


def test_ladder_section_malformed_report_unavailable(tmp_path):
    section = ladder_section(_write(tmp_path / "bad.json", {"ladder": {"rungs": {}}}))
    assert section["available"] is False
    assert "malformed" in section["reason"]


def test_ladder_section_accepts_bare_ladder_dict(tmp_path):
    bare = _ladder_report(disc_delta=-0.05)["ladder"]
    section = ladder_section(_write(tmp_path / "bare.json", bare))
    assert section["available"] is True
    # No explicit flag → inferred from the negative discrimination delta.
    assert section["whole_clip_move_regressed"] is True


# --- event section (F2) ------------------------------------------------------


def test_event_section_surfaces_threshold_and_metrics(tmp_path):
    section = event_eval_section(_write(tmp_path / "f2.json", _event_report()))
    assert section["available"] is True
    assert section["calibrated_silence_threshold"] == 0.5
    assert section["duration_gate_ms"] == 300
    assert section["dangerous_metric"] == "false_waqf_at_wasl_rate"
    assert section["test"]["metrics"]["f1"] == 0.874
    assert section["blank_run_available"] is False


def test_event_section_malformed_unavailable(tmp_path):
    section = event_eval_section(_write(tmp_path / "bad.json", {"calibrated_silence_threshold": 0.5}))
    assert section["available"] is False
    assert "malformed" in section["reason"]


# --- integration section (H, forward-compatible) -----------------------------


def test_integration_section_missing_is_unavailable():
    section = integration_section(None)
    assert section["available"] is False


def test_integration_section_passed_true(tmp_path):
    report = {"passed": True, "summary": "regains i'raab discrimination", "cases": 12}
    section = integration_section(_write(tmp_path / "h.json", report))
    assert section["available"] is True
    assert section["passed"] is True
    assert section["report"]["cases"] == 12


def test_integration_section_missing_passed_field_is_unknown(tmp_path):
    section = integration_section(_write(tmp_path / "h.json", {"summary": "ran"}))
    assert section["available"] is True
    assert section["passed"] is None


# --- readiness (go/no-go) ----------------------------------------------------


def test_readiness_ready_when_all_present_and_h_passed():
    l = {"available": True, "whole_clip_move_regressed": False, "recommended_lora_lever": None}
    e = {"available": True}
    i = {"available": True, "passed": True}
    r = readiness(l, e, i)
    assert r["ready"] is True
    assert r["blocking"] == []
    assert r["pending"] == []


def test_readiness_blocks_on_whole_clip_regression():
    l = {"available": True, "whole_clip_move_regressed": True,
         "recommended_lora_lever": {"name": "lower_rank_alpha"}}
    e = {"available": True}
    i = {"available": True, "passed": True}
    r = readiness(l, e, i)
    assert r["ready"] is False
    assert any("lower_rank_alpha" in b for b in r["blocking"])


def test_readiness_blocks_on_failed_integration():
    l = {"available": True, "whole_clip_move_regressed": False, "recommended_lora_lever": None}
    e = {"available": True}
    i = {"available": True, "passed": False}
    r = readiness(l, e, i)
    assert r["ready"] is False
    assert any("integration" in b for b in r["blocking"])


def test_readiness_pending_when_reports_missing():
    l = {"available": False}
    e = {"available": False}
    i = {"available": False}
    r = readiness(l, e, i)
    assert r["ready"] is False
    assert len(r["pending"]) == 3
    assert r["blocking"] == []


# --- full view ---------------------------------------------------------------


def test_build_signoff_view_assembles_all_sections(tmp_path):
    view = build_signoff_view(
        _write(tmp_path / "e.json", _ladder_report()),
        _write(tmp_path / "f2.json", _event_report()),
        _write(tmp_path / "h.json", {"passed": True, "summary": "ok"}),
    )
    assert view["ladder"]["available"] is True
    assert view["event_eval"]["available"] is True
    assert view["integration"]["available"] is True
    assert view["readiness"]["ready"] is True


def test_build_signoff_view_h_absent_is_pending(tmp_path):
    view = build_signoff_view(
        _write(tmp_path / "e.json", _ladder_report()),
        _write(tmp_path / "f2.json", _event_report()),
        None,
    )
    assert view["integration"]["available"] is False
    assert view["readiness"]["ready"] is False
    assert any("integration eval (H)" in p for p in view["readiness"]["pending"])
