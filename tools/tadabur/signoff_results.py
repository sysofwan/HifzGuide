"""Assemble the fine-tune sign-off results view (issue #37, helper for #10).

The #10 HITL gate (``docs/adr/0004-waqf-head-and-joint-whole-clip-fine-tune.md``) is a human
**go/no-go** on the joint waqf fine-tune, and ADR-0004 pins exactly which signals it turns on —
*not* frame-F1, but three separate artifacts produced by three separate offline runs:

* **E — the ablation ladder** (:mod:`training.ablation_ladder`, #33): the should-accept recall /
  should-reject discrimination deltas across the three rungs (segmented → whole-clip → joint), plus
  whether the whole-clip move ``(1)→(2)`` regressed should-reject and which LoRA-native lever that
  triggers. The dangerous transition is ``(1)→(2)``; ``(2)→(3)`` is pinned near-zero by the
  identity check the ladder run asserts.
* **F2 — the event-level waqf eval** (:mod:`tadabur.waqf_event_eval`, #34): the calibrated silence
  posterior threshold and, on the once-scored ``test`` partition, the four ADR-named event metrics.
  The **false-waqf @ wasl** rate is the one ADR-0004 flags as dangerous (a spurious stop forgives a
  dropped final haraka / missed cross-word idgham), so the view highlights it.
* **H — the conditional-reference integration eval** (#35, the product gate): whether the full
  path (phoneme decode → snap → realized-reference selection → ``.strict`` scoring) actually regains
  the wasl-sensitive i'raab / idgham discrimination versus today's ignore-end-word-tashkeel
  baseline. **#35 is not built yet**, so its report may be absent; the view marks it *unavailable*
  rather than fabricating an outcome, and surfaces it once its report exists.

This module is the pure assembly layer: it reads the three report JSON artifacts and builds one
view dict (:func:`build_signoff_view`) that the audit UI serves read-only, so the human #10 sign-off
happens without leaving the UI. Each section is independent and marks its own availability — a
missing or malformed report leaves *that* section unavailable with a reason, never crashes the view.
Nothing is gated here: the numbers are surfaced, the two mechanical blockers ADR-0004 defines (a
whole-clip-move should-reject regression; a failed integration eval) are flagged, and the human
makes the call.

Torch-free and deterministic: the same three artifacts always yield the same view.

**H consumer contract.** Whatever schema #35 settles on, the sign-off gate reads two top-level
fields from its report: ``passed`` (bool — did the integration gate clear) and ``summary`` (str —
the one-line outcome). The full report is passed through for detail. A report missing ``passed``
leaves the outcome *unknown* (surfaced, not treated as a pass).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# ADR-0004 names false-waqf @ wasl the dangerous event error (a spurious stop forgives the very
# i'raab / idgham the fine-tune is regaining), so the view flags it for the human's eye. This is a
# display emphasis, not a gate — F2 defines no pass/fail threshold, so the sign-off never auto-fails
# on it.
DANGEROUS_EVENT_METRIC = "false_waqf_at_wasl_rate"


def _load_report(path: Optional[Path]) -> tuple[Optional[dict], str]:
    """Read a report JSON, returning ``(data, reason)`` — ``data`` is ``None`` when unavailable.

    ``reason`` explains an absent report (not provided, missing file, or unparseable) so the view
    can tell the human what to run rather than showing an empty section.
    """
    if path is None:
        return None, "report not provided"
    if not path.is_file():
        return None, f"report not found at {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"could not read report at {path}: {exc}"


def _unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


def ladder_section(path: Optional[Path]) -> dict:
    """The E ablation-ladder section from :mod:`training.ablation_ladder`'s ``report`` artifact.

    Surfaces the three rungs' headline numbers, the two transition deltas, and the whole-clip-move
    regression flag + the LoRA-native lever it recommends — ADR-0004's ``(1)→(2)`` go/no-go. The
    artifact is the CLI payload (``{"ladder": ..., "whole_clip_move_regressed": ...,
    "recommended_lora_lever": ...}``); a bare :meth:`AblationLadder.to_json_dict` (just ``rungs`` /
    ``transitions``) is also accepted, with the regression flag then read off the whole-clip-move
    discrimination delta.
    """
    report, reason = _load_report(path)
    if report is None:
        return _unavailable(reason)
    try:
        ladder = report["ladder"] if "ladder" in report else report
        transitions = ladder["transitions"]
        whole_clip_move = transitions["whole_clip_move"]
        regressed = report.get(
            "whole_clip_move_regressed",
            (whole_clip_move["discrimination_delta"] or 0.0) < 0.0,
        )
        return {
            "available": True,
            "rungs": ladder["rungs"],
            "transitions": transitions,
            "whole_clip_move_regressed": regressed,
            "recommended_lora_lever": report.get("recommended_lora_lever"),
        }
    except (KeyError, TypeError) as exc:
        return _unavailable(f"malformed ablation-ladder report: missing {exc}")


def event_eval_section(path: Optional[Path]) -> dict:
    """The F2 event-eval section from :meth:`tadabur.waqf_event_eval.EventEvalReport.to_json_dict`.

    Surfaces the calibrated posterior threshold, the fixed duration gate, and the ``test`` (the
    once-scored gate) and ``calibration`` partitions' event metrics + counts. Marks whether the
    non-gated blank-run reference is present, and carries the teacher-circularity note so the human
    reads frame-F1 as a sanity check, never the gate (ADR-0004).
    """
    report, reason = _load_report(path)
    if report is None:
        return _unavailable(reason)
    try:
        blank = report["blank_run_reference"]
        return {
            "available": True,
            "calibrated_silence_threshold": report["calibrated_silence_threshold"],
            "duration_gate_ms": report["duration_gate_ms"],
            "dangerous_metric": DANGEROUS_EVENT_METRIC,
            "test": _partition_view(report["test"]),
            "calibration": _partition_view(report["calibration"]),
            "blank_run_available": bool(blank["available"]),
            "blank_run_reason": blank.get("reason", ""),
            "teacher_circularity_note": report.get("teacher_circularity_note", ""),
        }
    except (KeyError, TypeError) as exc:
        return _unavailable(f"malformed event-eval report: missing {exc}")


def _partition_view(partition: dict) -> dict:
    """One event-eval partition's counts + metrics, as F2's ``to_json_dict`` shapes them."""
    return {
        "silence_threshold": partition["silence_threshold"],
        "counts": partition["counts"],
        "metrics": partition["metrics"],
    }


def integration_section(path: Optional[Path]) -> dict:
    """The H conditional-reference integration outcome (#35, the product gate).

    #35 is the end-to-end gate #10 must actually clear. Its report (from
    :mod:`tadabur.waqf_integration_eval`) is absent until the eval has been run, so the section is
    then *unavailable* with a reason (the human cannot sign off on H until it exists). When the
    report lands it is surfaced whole, with the consumer contract's ``passed`` / ``summary`` lifted
    out; a report that omits ``passed`` leaves the outcome unknown (``passed`` is ``None``) rather
    than being read as a pass.
    """
    report, reason = _load_report(path)
    if report is None:
        return _unavailable(reason)
    return {
        "available": True,
        "passed": report.get("passed"),
        "summary": report.get("summary", ""),
        "report": report,
    }


def readiness(ladder: dict, event: dict, integration: dict) -> dict:
    """The overall go/no-go: what still blocks sign-off and what is still pending.

    ``blocking`` holds the two mechanical no-go conditions ADR-0004 defines — a whole-clip-move
    should-reject regression (E) and a failed integration eval (H). ``pending`` holds signals not
    yet available (a report the human still needs before they can sign off). ``ready`` is the
    conjunction: nothing blocking, nothing pending, and H explicitly passed. The event eval (F2)
    defines no pass/fail threshold, so it never auto-blocks — its metrics inform the human directly.
    """
    blocking: list[str] = []
    pending: list[str] = []

    if not ladder["available"]:
        pending.append("ablation ladder (E) report missing — run training.ablation_ladder report")
    elif ladder["whole_clip_move_regressed"]:
        lever = ladder.get("recommended_lora_lever")
        lever_name = lever["name"] if lever else "a LoRA-native lever"
        blocking.append(
            "whole-clip move (1)→(2) regressed should-reject discrimination — "
            f"ADR-0004 response: re-run rung (2) with {lever_name}"
        )

    if not event["available"]:
        pending.append("event eval (F2) report missing — run tadabur.waqf_event_eval")

    if not integration["available"]:
        pending.append(
            "integration eval (H) report missing — the product gate (#35) must run before sign-off"
        )
    elif integration["passed"] is False:
        blocking.append("integration eval (H) did not pass — the .strict discrimination gate failed")

    ready = not blocking and not pending and integration.get("passed") is True
    return {"ready": ready, "blocking": blocking, "pending": pending}


def build_signoff_view(
    ladder_report: Optional[Path],
    event_eval_report: Optional[Path],
    integration_report: Optional[Path],
) -> dict:
    """Assemble the full sign-off view from the three report artifacts (any may be absent).

    Each section is read fresh and marks its own availability, so re-running an eval and refreshing
    the page shows the new numbers. The readiness block folds the three sections into the
    go/no-go summary the #10 human reads first.
    """
    ladder = ladder_section(ladder_report)
    event = event_eval_section(event_eval_report)
    integration = integration_section(integration_report)
    return {
        "ladder": ladder,
        "event_eval": event,
        "integration": integration,
        "readiness": readiness(ladder, event, integration),
    }
