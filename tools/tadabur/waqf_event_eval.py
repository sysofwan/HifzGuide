"""P7.F2 event-level waqf eval + inference-threshold calibration (ADR-0004, #34).

A silence VAD detects *silence*, not whether a waqf-vs-wasl was correctly realized, so
ADR-0004 (``docs/adr/0004-waqf-head-and-joint-whole-clip-fine-tune.md``) makes the
scorer-facing gate **event-level**, measured *after* F1's post-processing
(:mod:`tadabur.waqf_postprocess`) on the frozen F0 fixtures
(:mod:`tadabur.waqf_event_fixtures`, split reciter-disjoint by :mod:`tadabur.waqf_freeze`):

* **false-waqf @ wasl** — a spurious stop fired at a true continuation. The dangerous
  error ADR-0004 calls out: a false waqf lets the scorer forgive a dropped final haraka
  or a missed cross-word idgham, the very ``.strict`` discrimination ADR-0001 is regaining.
* **false-wasl @ genuine-stop** — a real stop the pipeline missed (breath / filled pause /
  sub-threshold / madd-into-sukun the VAD under-fires).
* **mid-word-closure rejection** — the hard-negative set (qalqala on ق/ط, hamza in شَيء,
  madd): a silence the snap must **not** treat as a waqf. Measured as the fraction rejected.
* **boundary-snap accuracy** — of the genuine stops, how many the snap placed at a word
  edge rather than mis-snapping a real stop into a mid-word closure. Threshold-independent
  (it is the geometry step, not the posterior/duration gate).

**The eval input is per-clip frame scores, run through F1 unchanged.** ADR-0004's inference
knob is the **silence posterior threshold** — ``P(silence) >= threshold``, the argmax
boundary of the VAD's two-class softmax (:data:`tadabur.waqf_postprocess.DEFAULT_SILENCE_THRESHOLD`,
the ``threshold`` of :func:`tadabur.waqf_postprocess.detect_pauses` / ``waqf_events``). The
eval "only tunes the inference threshold" (ADR-0004). So this harness does not re-decide a
waqf itself: it **reconstructs each clip's silence-posterior lattice + word alignment** from
the frozen candidate fixtures (:func:`reconstruct_clip`) and feeds them straight through
:func:`tadabur.waqf_postprocess.waqf_events`, the identical F1 post-processing (300/700 ms
duration cleaning + word-edge snap). The calibrated number is therefore the real inference
threshold, and the event metrics are the real post-processing output. When the trained waqf
head lands, its per-frame ``P(silence)`` lattice replaces the reconstruction and the same
:func:`run_eval` calibrates the same threshold — no scoring code changes.

*How the lattice is reconstructed (torch-free, from the frozen candidates):* a clip's
silence-posterior track carries every candidate silence (``waqf`` **and**
``mid_word_closure`` spans), graded by each 40 ms frame's fractional silence coverage — so
the posterior threshold genuinely moves the silence-run edges, exactly as it would on a real
lattice. The **word** spans come from the phoneme alignment, not the silence, so they are
reconstructed from the ``waqf`` gaps alone (a ``mid_word_closure`` is *interior* speech the
word continues through); this is what lets :func:`snap_pauses` reject a long closure as a
mid-word silence while firing a genuine gap as a stop. The duration gate stays at F1's fixed
300/700 ms VAD definition (:data:`DEFAULT_MIN_SILENCE_MS`); the *tuned* knob is the posterior
threshold, per the ADR.

**Calibration is leak-free by construction.** The silence posterior threshold is tuned
**only** on the ``calibration`` partition; the ``test`` partition is scored **once** at the
chosen threshold and is the reported gate. ``waqf_freeze`` already made the two
reciter-disjoint, and :func:`run_eval` additionally asserts they share no clip.

**The blank-run baseline is a recorded reference, never a gate.** ADR-0004 keeps the
blank-run + post-processing number as a *documented reference point* only (CTC blank-runs
over-split and fail on madd — a known-inadequate waqf signal). The reference block scores the
**same** reconstructed lattice through F1's duration cleaning but **without** the phoneme
word-edge snap (a blank run has no word alignment to reject a closure with) — silence ⇒ stop
— so the number shows what the snap buys and holds the exact non-gated slot ADR-0004's
blank-run number occupies once a model decode exists. It is scored, recorded, and never gated.

**Beware teacher circularity.** Frame-F1 against the VAD teacher is a distillation *sanity
check only* (the VAD both labels the head and is the frame-F1 target, so a systematic VAD
error can pass frame-F1 while failing the recitation task). These event-level metrics — not
frame-F1 — are the product gate (:data:`TEACHER_CIRCULARITY_NOTE`).

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
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .waqf_event_fixtures import (
    MID_WORD_CLOSURE,
    WAQF,
    WASL,
    WaqfEventEntry,
    _FIXTURE_DIR,
    load_waqf_events,
)
from .waqf_postprocess import (
    DEFAULT_MIN_SILENCE_MS,
    DEFAULT_MIN_SPEECH_MS,
    DEFAULT_SILENCE_THRESHOLD,
    STUDENT_FRAME_MS,
    SilenceRun,
    WordSpan,
    detect_pauses,
    waqf_events,
)

# Frame-F1 vs the VAD teacher is a distillation sanity check, not the product gate — recorded
# verbatim on every report so a reader cannot mistake the two (ADR-0004 "Consequences").
TEACHER_CIRCULARITY_NOTE = (
    "Frame-F1 against the Recitation VAD teacher is a distillation sanity check only, never "
    "the product gate: the VAD both labels the waqf head and is the frame-F1 target, so a "
    "systematic VAD error in amateur audio can pass frame-F1 while failing the recitation "
    "task. The gate is the event-level metrics below (false-waqf@wasl, false-wasl@genuine-"
    "stop, mid-word-closure rejection, boundary-snap accuracy), computed by running F1 "
    "post-processing (tadabur.waqf_postprocess.waqf_events) on the reconstructed silence "
    "lattice of the human-adjudicated F0 fixtures."
)

# The blank-run baseline recorded (never gated) as ADR-0004's documented reference point: the
# same reconstructed lattice through F1's duration cleaning but WITHOUT the phoneme word-edge
# snap (a CTC blank run has no word alignment to reject a mid-word closure), so silence ⇒ stop.
# CTC blank-runs over-split and fail on madd; kept as a reference, never a ship gate.
REFERENCE_NOTE = (
    "Non-gated reference operating point (ADR-0004): the blank-run baseline — the same "
    "reconstructed silence lattice fed through F1's 300/700 ms duration cleaning but WITHOUT "
    "the phoneme word-edge snap, so every silence run fires as a stop (blank runs have no "
    "word alignment to reject a mid-word closure). Scored at the calibrated threshold and "
    "recorded, never gated: CTC blank-runs over-split and fail on madd. ADR-0004's model "
    "blank-run number occupies this same non-gated slot once a decode exists."
)

# The calibration objective, recorded on the report so the chosen threshold is auditable.
CALIBRATION_OBJECTIVE = (
    "Tune the silence posterior threshold (P(silence) >= threshold in waqf_postprocess) on "
    "the calibration partition to maximise waqf F1, tie-broken by the lower false-waqf@wasl "
    "rate (the more damaging error) then the threshold nearest the VAD's 0.5 argmax. The "
    "test partition is scored once at the chosen threshold through the same F1 post-"
    "processing and is the reported gate. The 300/700 ms duration gate stays at F1's fixed "
    "VAD definition; only the posterior threshold is tuned (ADR-0004)."
)

# Leading/trailing speech padding (in 40 ms frames) around a reconstructed clip's silences, so
# every candidate silence is scored as an *interior* pause flanked by >= min-speech frames
# rather than being trimmed as a clip edge. One min-speech span (700 ms) each side isolates
# F1's duration/snap decision from a synthetic clip boundary; inter-candidate speech is left
# untouched so F1's real short-speech merging still applies where two stops sit close together.
_PAD_FRAMES = math.ceil(DEFAULT_MIN_SPEECH_MS / STUDENT_FRAME_MS)


def _rate(numerator: int, denominator: int) -> float | None:
    """A rate, or ``None`` when the subset is empty (so an absent class never reads as 0.0)."""
    return numerator / denominator if denominator else None


def snapped_to_word_edge(entry: WaqfEventEntry) -> bool:
    """F1's snap geometry on the fixture: the candidate sat at a word edge, not inside a word.

    ``mid_word_closure`` is precisely the interior class :func:`tadabur.waqf_postprocess.snap_pauses`
    rejects; ``waqf`` / ``wasl`` are word edges. The phoneme-aligned candidate detector already
    made this snap (carried as ``predicted``), and the reconstruction reproduces it: only a
    ``waqf`` span opens a gap between reconstructed words, so a genuine stop the detector
    mis-snapped as ``mid_word_closure`` never reaches a word edge. Boundary-snap accuracy is a
    property of this geometry alone, so it is threshold-independent.
    """
    return entry.predicted != MID_WORD_CLOSURE


@dataclass(frozen=True)
class ClipLattice:
    """One clip's reconstructed F1 inputs: a silence-posterior track + word alignment.

    ``silence`` is the per-40 ms-frame ``P(silence)`` reconstructed from the clip's candidate
    silences (graded by fractional frame coverage, so the posterior threshold moves the run
    edges); ``words`` are the phoneme-alignment word spans (from the ``waqf`` gaps only, a
    ``mid_word_closure`` being interior speech); ``candidate_spans`` maps each silence-bearing
    candidate's ``boundary_index`` to its nominal frame span, so an F1 output pause can be
    attributed back to the boundary it fired (or failed to fire) at.
    """

    clip_id: str
    silence: np.ndarray
    words: list[WordSpan]
    candidate_spans: dict[int, SilenceRun]


def _fractional_frames(entry: WaqfEventEntry) -> tuple[float, float]:
    """A candidate silence's ``[start, end)`` extent in fractional 40 ms frames, padded."""
    scale = 1000.0 / STUDENT_FRAME_MS
    return _PAD_FRAMES + entry.start_s * scale, _PAD_FRAMES + entry.end_s * scale


def _add_coverage(track: np.ndarray, frame_start: float, frame_end: float) -> None:
    """Paint each frame's fractional silence coverage of ``[frame_start, frame_end)`` in place.

    A frame fully inside the span reads 1.0; an edge frame reads the fraction of its 40 ms it
    covers. Overlapping spans take the max (coverage is a probability, not additive), so the
    track stays in ``[0, 1]``. This is the honest 40 ms quantisation of a known silence
    interval — the finest per-frame ``P(silence)`` the frozen candidate carries — so the
    posterior threshold ``P(silence) >= t`` genuinely lengthens/shortens the run at its edges.
    """
    for frame in range(int(math.floor(frame_start)), int(math.ceil(frame_end))):
        coverage = min(frame + 1, frame_end) - max(frame, frame_start)
        if coverage > 0:
            track[frame] = max(track[frame], coverage)


def _word_spans(waqf_gaps: list[SilenceRun], n_frames: int) -> list[WordSpan]:
    """The word speech spans between the ``waqf`` gaps — the phoneme-alignment analogue.

    Words come from the phoneme head, not the silence VAD, so they break only at genuine
    stops (``waqf`` gaps); a ``mid_word_closure`` silence stays *inside* a word span, which is
    exactly what lets :func:`snap_pauses` reject its pause as a mid-word closure. Numbered in
    clip order; the index is only carried through to any fired :class:`WaqfEvent`.
    """
    is_gap = np.zeros(n_frames, dtype=bool)
    for gap in waqf_gaps:
        is_gap[gap.start_frame : gap.end_frame] = True
    spans: list[WordSpan] = []
    start: int | None = None
    for frame in range(n_frames + 1):
        speech = frame < n_frames and not is_gap[frame]
        if speech and start is None:
            start = frame
        elif not speech and start is not None:
            spans.append(WordSpan(len(spans), start, frame))
            start = None
    return spans


def reconstruct_clip(entries: list[WaqfEventEntry]) -> ClipLattice:
    """Reconstruct one clip's F1 inputs (silence lattice + word spans) from its candidates.

    ``entries`` are all of one clip's adjudicated candidate boundaries. Every ``waqf`` and
    ``mid_word_closure`` candidate contributes a graded silence run to the posterior track;
    only the ``waqf`` gaps break the word spans (a ``mid_word_closure`` is interior speech,
    a ``wasl`` a zero-width contiguous edge). The track is padded with leading/trailing
    speech (:data:`_PAD_FRAMES`) so each silence is scored as an interior pause.
    """
    clip_id = entries[0].clip_id
    silences = [e for e in entries if e.end_s > e.start_s]
    if not silences:
        return ClipLattice(clip_id, np.zeros(_PAD_FRAMES, dtype=np.float32), [], {})

    n_frames = math.ceil(max(_fractional_frames(e)[1] for e in silences)) + _PAD_FRAMES
    track = np.zeros(n_frames, dtype=np.float32)
    waqf_gaps: list[SilenceRun] = []
    candidate_spans: dict[int, SilenceRun] = {}
    for entry in silences:
        frame_start, frame_end = _fractional_frames(entry)
        _add_coverage(track, frame_start, frame_end)
        span = SilenceRun(round(frame_start), round(frame_end))
        candidate_spans[entry.boundary_index] = span
        if entry.predicted == WAQF:
            waqf_gaps.append(span)
    return ClipLattice(clip_id, track, _word_spans(waqf_gaps, n_frames), candidate_spans)


def _overlaps(pause: SilenceRun, span: SilenceRun) -> bool:
    """True if two half-open frame runs share any frame."""
    return pause.start_frame < span.end_frame and span.start_frame < pause.end_frame


def _fired_boundaries(lattice: ClipLattice, threshold: float, *, snap: bool) -> set[int]:
    """The clip's candidate boundaries that fire a stop under F1 at ``threshold``.

    With ``snap`` (the gated path) each candidate fires iff a :func:`waqf_events` waqf pause
    overlaps its span — the full F1 post-processing (duration gate + word-edge snap). Without
    ``snap`` (the blank-run reference) it fires iff any :func:`detect_pauses` silence run
    overlaps its span — duration cleaning only, no phoneme snap, so a mid-word closure fires.
    """
    if snap:
        pauses = [
            event.pause
            for event in waqf_events(lattice.silence, lattice.words, threshold=threshold).waqf
        ]
    else:
        pauses = detect_pauses(lattice.silence, threshold=threshold)
    return {
        boundary_index
        for boundary_index, span in lattice.candidate_spans.items()
        if any(_overlaps(pause, span) for pause in pauses)
    }


@dataclass(frozen=True)
class EventMetrics:
    """Event-level waqf confusion for one partition under one prediction rule.

    Counts are split by the human ``verdict`` class so ADR-0004's four named metrics fall
    straight out (a false waqf is separated into its ``@wasl`` and ``@closure`` sub-counts
    because the two negatives — a true continuation vs a hard-negative closure — are the
    distinct errors the ADR names). ``label`` identifies the rule; ``silence_threshold`` is
    the ``P(silence)`` posterior threshold the F1 pass ran at.
    """

    label: str
    silence_threshold: float
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
            "silence_threshold": self.silence_threshold,
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


def _partition_lattices(entries: list[WaqfEventEntry]) -> dict[str, ClipLattice]:
    """Group a partition's boundaries by clip and reconstruct each clip's F1 inputs once."""
    by_clip: dict[str, list[WaqfEventEntry]] = {}
    for entry in entries:
        by_clip.setdefault(entry.clip_id, []).append(entry)
    return {clip: reconstruct_clip(rows) for clip, rows in by_clip.items()}


def compute_metrics(
    entries: list[WaqfEventEntry],
    lattices: dict[str, ClipLattice],
    threshold: float,
    *,
    label: str,
    snap: bool,
) -> EventMetrics:
    """Confuse F1's fired stops (run through :func:`waqf_events`) against the human verdicts.

    Each clip's lattice is run through F1 once at ``threshold`` (``snap`` selects the gated
    word-edge-snapped path or the blank-run reference); a boundary is a fired stop iff F1's
    output overlaps it. The snap-accuracy count is a property of the reconstruction geometry
    (``verdict == waqf`` boundaries that reached a word edge) and so is threshold-independent.
    """
    fired: set[tuple[str, int]] = {
        (clip, boundary_index)
        for clip, lattice in lattices.items()
        for boundary_index in _fired_boundaries(lattice, threshold, snap=snap)
    }
    waqf_total = wasl_total = closure_total = 0
    true_positive = false_wasl = 0
    false_waqf_at_wasl = false_waqf_at_closure = 0
    snap_correct = 0
    for entry in entries:
        is_fired = (entry.clip_id, entry.boundary_index) in fired
        if entry.verdict == WAQF:
            waqf_total += 1
            true_positive += is_fired
            false_wasl += not is_fired
            snap_correct += snapped_to_word_edge(entry)
        elif entry.verdict == WASL:
            wasl_total += 1
            false_waqf_at_wasl += is_fired
        elif entry.verdict == MID_WORD_CLOSURE:
            closure_total += 1
            false_waqf_at_closure += is_fired
    return EventMetrics(
        label=label,
        silence_threshold=threshold,
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


def calibration_grid(lattices: dict[str, ClipLattice]) -> list[float]:
    """The silence posterior thresholds to sweep: every distinct frame-coverage flip point.

    A threshold changes the binarisation only where some frame's fractional silence coverage
    crosses it, so the grid is the distinct partial-coverage values on the calibration
    lattices' edge frames, anchored on the VAD's own 0.5 argmax
    (:data:`tadabur.waqf_postprocess.DEFAULT_SILENCE_THRESHOLD`). Derived from the calibration
    lattices only, so no test statistic can leak into the threshold.
    """
    thresholds = {
        round(float(value), 6)
        for lattice in lattices.values()
        for value in lattice.silence
        if 0.0 < value <= 1.0
    }
    thresholds.add(DEFAULT_SILENCE_THRESHOLD)
    return sorted(thresholds)


def calibrate(
    entries: list[WaqfEventEntry], lattices: dict[str, ClipLattice]
) -> tuple[float, list[EventMetrics]]:
    """Pick the silence posterior threshold on calibration (:data:`CALIBRATION_OBJECTIVE`).

    Returns the chosen threshold and the full per-threshold sweep (for audit). The objective
    maximises waqf F1, tie-broken by the lower false-waqf@wasl rate then the threshold nearest
    the VAD's 0.5 argmax — deterministic, so a rerun reproduces the same operating point.
    """
    sweep = [
        compute_metrics(
            entries, lattices, threshold, label=f"calibration@{threshold:.6g}", snap=True
        )
        for threshold in calibration_grid(lattices)
    ]

    def objective(metrics: EventMetrics) -> tuple[float, float, float]:
        return (
            metrics.f1 or 0.0,
            -(metrics.false_waqf_rate or 0.0),
            -abs(metrics.silence_threshold - DEFAULT_SILENCE_THRESHOLD),
        )

    best = max(sweep, key=objective)
    return best.silence_threshold, sweep


@dataclass(frozen=True)
class EventEvalReport:
    """The full F2 event-level eval: the calibrated threshold and the once-scored test gate."""

    calibrated_silence_threshold: float
    calibration: EventMetrics
    test: EventMetrics
    calibration_reference: EventMetrics
    test_reference: EventMetrics
    sweep: list[EventMetrics]

    def to_json_dict(self) -> dict:
        return {
            "calibrated_silence_threshold": self.calibrated_silence_threshold,
            "duration_gate_ms": DEFAULT_MIN_SILENCE_MS,
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
    """Calibrate the posterior threshold on ``calibration_entries``, score ``test_entries`` once.

    Reconstructs each partition's per-clip silence lattices, sweeps the threshold on
    calibration through F1's :func:`waqf_events`, then scores test once at the chosen
    threshold. Asserts the two partitions share no clip (a belt-and-braces leak check on top
    of ``waqf_freeze``'s reciter-disjoint split) before the threshold is ever applied to test.
    The blank-run reference is scored on both partitions at the same threshold and recorded.
    """
    shared = {e.clip_id for e in calibration_entries} & {e.clip_id for e in test_entries}
    if shared:
        raise ValueError(
            f"{len(shared)} clip(s) appear in both calibration and test (leakage): "
            f"{sorted(shared)[:5]}"
        )

    calibration_lattices = _partition_lattices(calibration_entries)
    test_lattices = _partition_lattices(test_entries)
    threshold, sweep = calibrate(calibration_entries, calibration_lattices)

    return EventEvalReport(
        calibrated_silence_threshold=threshold,
        calibration=compute_metrics(
            calibration_entries, calibration_lattices, threshold,
            label="calibration", snap=True,
        ),
        test=compute_metrics(
            test_entries, test_lattices, threshold, label="test", snap=True,
        ),
        calibration_reference=compute_metrics(
            calibration_entries, calibration_lattices, threshold,
            label="calibration_blank_run_reference", snap=False,
        ),
        test_reference=compute_metrics(
            test_entries, test_lattices, threshold,
            label="test_blank_run_reference", snap=False,
        ),
        sweep=sweep,
    )


def _print_summary(report: EventEvalReport) -> None:
    print(f"Calibrated silence posterior threshold: {report.calibrated_silence_threshold}")
    for name, metrics in (("test (gate)", report.test), ("calibration", report.calibration)):
        print(
            f"{name}: waqf F1 {metrics.f1}  "
            f"false-waqf@wasl {metrics.false_waqf_rate}  "
            f"false-wasl@stop {metrics.false_wasl_rate}  "
            f"mwc-reject {metrics.mid_word_closure_rejection_rate}  "
            f"snap-acc {metrics.boundary_snap_accuracy}"
        )
    print(f"blank-run reference (not gated): test waqf F1 {report.test_reference.f1}")


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
