"""P7.F2 event-level waqf eval + inference-threshold calibration (ADR-0004, #34).

A silence VAD detects *silence*, not whether a waqf-vs-wasl was correctly realized, so
ADR-0004 (``docs/adr/0004-waqf-head-and-joint-whole-clip-fine-tune.md``) makes the
scorer-facing gate **event-level**, measured *after* F1's post-processing
(:mod:`tadabur.waqf_postprocess`) on the frozen F0 fixtures
(:mod:`tadabur.waqf_event_fixtures`, split by :mod:`tadabur.waqf_freeze`):

* **false-waqf @ wasl** — a spurious stop fired at a true continuation. The dangerous
  error ADR-0004 calls out: a false waqf lets the scorer forgive a dropped final haraka
  or a missed cross-word idgham, the very ``.strict`` discrimination ADR-0001 is regaining.
* **false-wasl @ genuine-stop** — a real stop the pipeline missed (breath / filled pause /
  sub-threshold / madd-into-sukun the VAD under-fires).
* **mid-word-closure rejection** — the hard-negative set (qalqala on ق/ط, hamza in شَيء,
  madd): a silence the snap must **not** treat as a waqf. Measured as the fraction rejected.
* **boundary-snap accuracy** — of the genuine stops, how many the snap placed at a word
  edge rather than mis-snapping a real stop into a mid-word closure. Threshold-independent
  (it is the geometry step, not the duration gate).

**F1 post-processing on the frozen candidate fixtures.** F1's frame-level reference turns a
model's per-frame ``P(silence)`` lattice into snapped waqf events with two operative rules:
a **duration gate** (a silence run is a pause only at ≥ ``min_silence_ms``,
:func:`tadabur.waqf_postprocess.detect_pauses`) and a **word-edge snap** that rejects a
silence falling inside a word (:func:`tadabur.waqf_postprocess.snap_pauses`). The trained
waqf head does not exist yet, so there is no per-frame lattice to feed F1; what the frozen
fixtures carry is the torch-free candidate detector's output — each boundary's phoneme
-aligned snap class (``predicted``: ``mid_word_closure`` = interior, else a word edge) and
its **measured** silence span (``start_s``/``end_s``). F1's two rules therefore specialise,
on the fixtures, to one decision per candidate boundary
(:func:`predict_waqf`): a boundary fires a waqf iff it snapped to a word edge **and** its
silence run clears the duration threshold — one silence run per candidate, quantised on the
same 40 ms lattice F1 reasons over. When the trained model lands, the identical event
metrics consume :func:`tadabur.waqf_postprocess.waqf_events` over the model's real lattice;
only the source of the per-boundary decision changes, not the scoring.

**Calibration is leak-free by construction.** The inference threshold (the pause-duration
cut — the knob available on the frozen candidate silences) is tuned **only** on the
``calibration`` partition; the ``test`` partition is scored **once** at the chosen threshold
and is the reported gate. ``waqf_freeze`` already made the two reciter-disjoint, and
:func:`run_eval` additionally asserts they share no clip.

**References are recorded, never gated.** ADR-0004 keeps the blank-run + post-processing
number as a *documented reference point*, not a ship gate (CTC blank-runs are a known-
inadequate waqf signal — they over-split and fail on madd). The report carries a
``reference`` block for exactly that role; with no model decode yet it holds the candidate
detector's own operating point (``predicted`` as-is), and the ADR's blank-run number plugs
into the same non-gated slot once a decode exists.

**Beware teacher circularity.** Frame-F1 against the VAD teacher is a distillation *sanity
check only* (the VAD both labels the head and is the frame-F1 target, so a systematic VAD
error can pass frame-F1 while failing the recitation task). These event-level metrics — not
frame-F1 — are the product gate. Recorded verbatim on every report
(:data:`TEACHER_CIRCULARITY_NOTE`).

Everything here is pure, torch-free, deterministic: identical fixtures + threshold yield an
identical report.

Usage:
  python -m tadabur.waqf_event_eval \\
      --calibration waqf_event_fixtures/waqf_events.calibration.jsonl \\
      --test        waqf_event_fixtures/waqf_events.test.jsonl \\
      --out         waqf_event_fixtures/waqf_event_eval.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .waqf_event_fixtures import (
    MID_WORD_CLOSURE,
    WAQF,
    WASL,
    WaqfEventEntry,
    _FIXTURE_DIR,
    load_waqf_events,
)
from .waqf_postprocess import DEFAULT_MIN_SILENCE_MS, STUDENT_FRAME_MS, seconds_to_frame

# Frame-F1 vs the VAD teacher is a distillation sanity check, not the product gate — recorded
# verbatim on every report so a reader cannot mistake the two (ADR-0004 "Consequences").
TEACHER_CIRCULARITY_NOTE = (
    "Frame-F1 against the Recitation VAD teacher is a distillation sanity check only, never "
    "the product gate: the VAD both labels the waqf head and is the frame-F1 target, so a "
    "systematic VAD error in amateur audio can pass frame-F1 while failing the recitation "
    "task. The gate is the event-level metrics below (false-waqf@wasl, false-wasl@genuine-"
    "stop, mid-word-closure rejection, boundary-snap accuracy), computed after F1 post-"
    "processing on the human-adjudicated F0 fixtures."
)

# The candidate-detector operating point recorded (not gated) as the reference baseline, and
# the non-gated slot ADR-0004's blank-run + post-processing number occupies once a model
# decode exists (CTC blank-runs are a known-inadequate waqf signal — kept as a reference,
# never a ship gate).
REFERENCE_NOTE = (
    "Non-gated reference operating point: the torch-free candidate detector's own class "
    "(predicted == 'waqf'), scored against the same human verdicts. ADR-0004's blank-run + "
    "post-processing number occupies this same non-gated slot once a model decode exists."
)

# The calibration objective, recorded on the report so the chosen threshold is auditable.
CALIBRATION_OBJECTIVE = (
    "Tune the pause-duration threshold on the calibration partition to maximise waqf F1, "
    "tie-broken by the lower false-waqf@wasl rate (the more damaging error) then the higher "
    "threshold (fewer spurious fires). The test partition is scored once at the chosen "
    "threshold and is the reported gate."
)


def silence_ms(entry: WaqfEventEntry) -> int:
    """A candidate boundary's silence-run length in ms, quantised on the 40 ms lattice.

    Measured from the fixture's ``[start_s, end_s]`` span on the same 40 ms post-adapter
    lattice F1 reasons over (:func:`tadabur.waqf_postprocess.seconds_to_frame`), so the
    duration gate here matches F1's frame-level ``detect_pauses`` exactly. A ``wasl``
    candidate is a zero-width word edge, so its silence is 0 ms and it never fires.
    """
    frames = seconds_to_frame(entry.end_s) - seconds_to_frame(entry.start_s)
    return frames * STUDENT_FRAME_MS


def snapped_to_word_edge(entry: WaqfEventEntry) -> bool:
    """F1's snap decision on the fixture: the candidate sat at a word edge, not inside a word.

    ``mid_word_closure`` is precisely the class :func:`tadabur.waqf_postprocess.snap_pauses`
    rejects (a silence overlapping a word's interior span); ``waqf`` / ``wasl`` are word
    edges. The phoneme-aligned candidate detector already made this snap, carried as
    ``predicted``.
    """
    return entry.predicted != MID_WORD_CLOSURE


def predict_waqf(entry: WaqfEventEntry, min_silence_ms: int) -> bool:
    """F1 post-processing on one frozen candidate boundary: does it fire a waqf?

    The two operative F1 rules, specialised to the fixtures (one silence run per candidate):
    the silence must snap to a **word edge** (:func:`snapped_to_word_edge`) and its run must
    clear the **duration gate** (:func:`silence_ms` ≥ ``min_silence_ms``, the tuned knob).
    """
    return snapped_to_word_edge(entry) and silence_ms(entry) >= min_silence_ms


def _reference_predict(entry: WaqfEventEntry) -> bool:
    """The non-gated reference operating point: the detector's own ``predicted == waqf``."""
    return entry.predicted == WAQF


def _rate(numerator: int, denominator: int) -> float | None:
    """A rate, or ``None`` when the subset is empty (so an absent class never reads as 0.0)."""
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class EventMetrics:
    """Event-level waqf confusion for one partition under one prediction rule.

    Counts are split by the human ``verdict`` class so ADR-0004's four named metrics fall
    straight out (a false waqf is separated into its ``@wasl`` and ``@closure`` sub-counts
    because the two negatives — a true continuation vs a hard-negative closure — are the
    distinct errors the ADR names). ``label`` identifies the rule; ``min_silence_ms`` is the
    duration threshold, or ``None`` for the threshold-free reference operating point.
    """

    label: str
    min_silence_ms: int | None
    boundaries: int
    waqf_total: int
    wasl_total: int
    closure_total: int
    true_positive: int
    false_wasl: int
    false_waqf_at_wasl: int
    false_waqf_at_closure: int
    snap_correct: int

    @property
    def predicted_waqf(self) -> int:
        return self.true_positive + self.false_waqf_at_wasl + self.false_waqf_at_closure

    @property
    def false_positive(self) -> int:
        return self.false_waqf_at_wasl + self.false_waqf_at_closure

    @property
    def precision(self) -> float | None:
        return _rate(self.true_positive, self.predicted_waqf)

    @property
    def recall(self) -> float | None:
        return _rate(self.true_positive, self.waqf_total)

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if not p or not r:
            return None
        return 2 * p * r / (p + r)

    @property
    def false_waqf_rate(self) -> float | None:
        """False-waqf @ wasl: spurious stops fired at true continuations, over all wasl."""
        return _rate(self.false_waqf_at_wasl, self.wasl_total)

    @property
    def false_wasl_rate(self) -> float | None:
        """False-wasl @ genuine-stop: real stops missed, over all genuine stops."""
        return _rate(self.false_wasl, self.waqf_total)

    @property
    def mid_word_closure_rejection_rate(self) -> float | None:
        """Fraction of the hard-negative mid-word closures correctly not fired as a waqf."""
        return _rate(self.closure_total - self.false_waqf_at_closure, self.closure_total)

    @property
    def boundary_snap_accuracy(self) -> float | None:
        """Fraction of genuine stops the snap placed at a word edge (not mis-snapped)."""
        return _rate(self.snap_correct, self.waqf_total)

    def to_json_dict(self) -> dict:
        return {
            "label": self.label,
            "min_silence_ms": self.min_silence_ms,
            "counts": {
                "boundaries": self.boundaries,
                "waqf_total": self.waqf_total,
                "wasl_total": self.wasl_total,
                "closure_total": self.closure_total,
                "true_positive": self.true_positive,
                "false_wasl": self.false_wasl,
                "false_waqf_at_wasl": self.false_waqf_at_wasl,
                "false_waqf_at_closure": self.false_waqf_at_closure,
                "predicted_waqf": self.predicted_waqf,
                "snap_correct": self.snap_correct,
            },
            "metrics": {
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "false_waqf_at_wasl_rate": self.false_waqf_rate,
                "false_wasl_at_genuine_stop_rate": self.false_wasl_rate,
                "mid_word_closure_rejection_rate": self.mid_word_closure_rejection_rate,
                "boundary_snap_accuracy": self.boundary_snap_accuracy,
            },
        }


def compute_metrics(
    entries: list[WaqfEventEntry],
    predict: Callable[[WaqfEventEntry], bool],
    *,
    label: str,
    min_silence_ms: int | None,
) -> EventMetrics:
    """Confuse each boundary's ``predict`` outcome against its human verdict, once.

    ``predict`` is the prediction rule (a threshold-bound :func:`predict_waqf` or the
    reference :func:`_reference_predict`); the snap-accuracy count is a property of the snap
    alone (``verdict == waqf`` boundaries the detector placed at a word edge) and so is
    independent of ``predict``.
    """
    waqf_total = wasl_total = closure_total = 0
    true_positive = false_wasl = 0
    false_waqf_at_wasl = false_waqf_at_closure = 0
    snap_correct = 0
    for entry in entries:
        fired = predict(entry)
        if entry.verdict == WAQF:
            waqf_total += 1
            if fired:
                true_positive += 1
            else:
                false_wasl += 1
            if snapped_to_word_edge(entry):
                snap_correct += 1
        elif entry.verdict == WASL:
            wasl_total += 1
            if fired:
                false_waqf_at_wasl += 1
        elif entry.verdict == MID_WORD_CLOSURE:
            closure_total += 1
            if fired:
                false_waqf_at_closure += 1
    return EventMetrics(
        label=label,
        min_silence_ms=min_silence_ms,
        boundaries=len(entries),
        waqf_total=waqf_total,
        wasl_total=wasl_total,
        closure_total=closure_total,
        true_positive=true_positive,
        false_wasl=false_wasl,
        false_waqf_at_wasl=false_waqf_at_wasl,
        false_waqf_at_closure=false_waqf_at_closure,
        snap_correct=snap_correct,
    )


def calibration_grid(entries: list[WaqfEventEntry]) -> list[int]:
    """The pause-duration thresholds to sweep: every edge candidate's measured duration.

    A threshold is only distinguishable at a value where some candidate's silence run
    crosses it, so the grid is the distinct 40 ms-quantised durations of the word-edge
    candidates (the only ones that can fire), anchored on the VAD's own 300 ms waqf
    definition (:data:`tadabur.waqf_postprocess.DEFAULT_MIN_SILENCE_MS`). Data-derived from
    the given (calibration) entries only, so no test statistic can leak into the threshold.
    """
    durations = {
        silence_ms(e) for e in entries if snapped_to_word_edge(e) and silence_ms(e) > 0
    }
    durations.add(DEFAULT_MIN_SILENCE_MS)
    return sorted(durations)


def calibrate(entries: list[WaqfEventEntry]) -> tuple[int, list[EventMetrics]]:
    """Pick the pause-duration threshold on the calibration partition (:data:`CALIBRATION_OBJECTIVE`).

    Returns the chosen ``min_silence_ms`` and the full per-threshold sweep (for audit). The
    objective maximises waqf F1, tie-broken by the lower false-waqf@wasl rate then the higher
    threshold — deterministic, so a rerun reproduces the same operating point.
    """
    sweep = [
        compute_metrics(
            entries, lambda e, t=t: predict_waqf(e, t),
            label=f"calibration@{t}ms", min_silence_ms=t,
        )
        for t in calibration_grid(entries)
    ]

    def objective(metrics: EventMetrics) -> tuple[float, float, int]:
        return (
            metrics.f1 or 0.0,
            -(metrics.false_waqf_rate or 0.0),
            metrics.min_silence_ms or 0,
        )

    best = max(sweep, key=objective)
    return best.min_silence_ms, sweep


@dataclass(frozen=True)
class EventEvalReport:
    """The full F2 event-level eval: the calibrated threshold and the once-scored test gate."""

    calibrated_min_silence_ms: int
    calibration: EventMetrics
    test: EventMetrics
    calibration_reference: EventMetrics
    test_reference: EventMetrics
    sweep: list[EventMetrics]

    def to_json_dict(self) -> dict:
        return {
            "calibrated_min_silence_ms": self.calibrated_min_silence_ms,
            "calibration_objective": CALIBRATION_OBJECTIVE,
            "teacher_circularity_note": TEACHER_CIRCULARITY_NOTE,
            "reference_note": REFERENCE_NOTE,
            "test": self.test.to_json_dict(),
            "calibration": self.calibration.to_json_dict(),
            "reference": {
                "test": self.test_reference.to_json_dict(),
                "calibration": self.calibration_reference.to_json_dict(),
            },
            "calibration_sweep": [m.to_json_dict() for m in self.sweep],
        }


def run_eval(
    calibration_entries: list[WaqfEventEntry],
    test_entries: list[WaqfEventEntry],
) -> EventEvalReport:
    """Calibrate the threshold on ``calibration_entries``, score ``test_entries`` once.

    Asserts the two partitions share no clip (a belt-and-braces leak check on top of
    ``waqf_freeze``'s reciter-disjoint split) before the threshold is ever applied to test.
    """
    shared = {e.clip_id for e in calibration_entries} & {e.clip_id for e in test_entries}
    if shared:
        raise ValueError(
            f"{len(shared)} clip(s) appear in both calibration and test (leakage): "
            f"{sorted(shared)[:5]}"
        )

    threshold, sweep = calibrate(calibration_entries)
    calibration = compute_metrics(
        calibration_entries, lambda e: predict_waqf(e, threshold),
        label="calibration", min_silence_ms=threshold,
    )
    test = compute_metrics(
        test_entries, lambda e: predict_waqf(e, threshold),
        label="test", min_silence_ms=threshold,
    )
    calibration_reference = compute_metrics(
        calibration_entries, _reference_predict,
        label="calibration_reference", min_silence_ms=None,
    )
    test_reference = compute_metrics(
        test_entries, _reference_predict, label="test_reference", min_silence_ms=None,
    )
    return EventEvalReport(
        calibrated_min_silence_ms=threshold,
        calibration=calibration,
        test=test,
        calibration_reference=calibration_reference,
        test_reference=test_reference,
        sweep=sweep,
    )


def _print_summary(report: EventEvalReport) -> None:
    print(f"Calibrated pause-duration threshold: {report.calibrated_min_silence_ms} ms")
    for name, metrics in (("test (gate)", report.test), ("calibration", report.calibration)):
        print(
            f"{name}: waqf F1 {metrics.f1}  "
            f"false-waqf@wasl {metrics.false_waqf_rate}  "
            f"false-wasl@stop {metrics.false_wasl_rate}  "
            f"mwc-reject {metrics.mid_word_closure_rejection_rate}  "
            f"snap-acc {metrics.boundary_snap_accuracy}"
        )
    print(f"reference (not gated): test waqf F1 {report.test_reference.f1}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=_FIXTURE_DIR / "waqf_events.calibration.jsonl",
        help="Frozen calibration partition (waqf_events.calibration.jsonl).",
    )
    parser.add_argument(
        "--test", type=Path, default=_FIXTURE_DIR / "waqf_events.test.jsonl",
        help="Frozen test partition (waqf_events.test.jsonl), scored once.",
    )
    parser.add_argument(
        "--out", type=Path, default=_FIXTURE_DIR / "waqf_event_eval.json",
        help="Report output path (JSON).",
    )
    args = parser.parse_args()

    calibration_entries = load_waqf_events(args.calibration)
    test_entries = load_waqf_events(args.test)
    if not calibration_entries or not test_entries:
        raise SystemExit(
            "Empty calibration or test partition — run tadabur.waqf_freeze first "
            f"(calibration={len(calibration_entries)}, test={len(test_entries)})."
        )

    report = run_eval(calibration_entries, test_entries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report.to_json_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    _print_summary(report)
    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
