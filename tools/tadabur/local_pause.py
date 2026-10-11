"""The local pause check (#134): did the reciter stop at one word boundary?

The segmenter (:mod:`tadabur.waqf_detect`) cuts a clip at the recitation VAD's pauses of at
least 300 ms that land on a word edge. Against the owner's adjudicated waqf fixtures
(:mod:`tadabur.waqf_event_fixtures`) it misses about one real pause in five. At a boundary
whose sites only exist in wasl (:mod:`tadabur.waqf_guard`), a missed pause leaves a wrong
reference, while a false alarm costs one excluded site. So this check reads the same VAD
**more sensitively, and only locally**:

* **Signal.** The VAD's (``obadx/recitation-segmenter-v2``, :mod:`tadabur.vad`) probability
  of silence for every 20 ms frame, before any of the segmenter's cleaning (its 300 ms
  minimum silence and 700 ms minimum speech). The clip is cut into the chunks
  ``recitations_segmenter.segment_recitations`` uses, so each frame is the one it labels.
* **Window.** A boundary is read between the two words' aligned times: from the onset of
  the word before it to the onset of the word after it (the clip's whole-clip alignment,
  ``ClipStatus.word_times``, the times every pool excerpt is cut by), plus
  :data:`WINDOW_PAD_S` after, for a late CTC onset.
* **Score.** Per probability threshold in :data:`THRESHOLDS`, the longest run of frames in
  the window whose silence probability reaches it, in seconds (:func:`longest_silence`).

The check fires when the score at a threshold ``τ`` reaches a duration ``d``.
:func:`calibrate` sweeps every ``(τ, d)`` of :data:`THRESHOLDS` x :data:`DURATIONS` over the
**calibration** fixture boundaries the segmenter did not cut (the only ones the guard applies
the check to) and picks the pre-stated operating point: the highest recall with precision at
least :data:`MIN_PRECISION` (ties: higher precision, then the longer ``d``, then the higher
``τ``). The curve over every calibration boundary is kept for reference. The **test** fixtures
are only reported. Each fixture boundary is one unit: a pause when the owner's verdict on any
of its rows is ``waqf``.

The two errors cost differently. A false pause takes one site out of scoring; a missed pause
leaves a reference that only exists in wasl, so a correct waqf is scored as a mistake.

Usage (Linux + CUDA, from ``tools/``; reads the windows :mod:`tadabur.waqf_guard` lists)::

  python -m tadabur.local_pause score --windows windows.jsonl \\
      --audio-dir stage/clips --audio-dir ../issue-126/stage/clips \\
      --out tadabur/ship_a/waqf_guard.boundaries.jsonl \\
      --provenance tadabur/ship_a/waqf_guard.vad.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import TARGET_SAMPLE_RATE

#: One VAD frame: the feature extractor's hop (160) times its stride (2), 20 ms.
FRAME_SAMPLES = 320
#: The longest span the VAD reads at once (``segment_recitations``' ``max_duration_ms``).
CHUNK_SAMPLES = 19995 * TARGET_SAMPLE_RATE // 1000
#: The VAD's silence class.
SILENCE_LABEL = 0
#: Silence probabilities a frame is counted at; 0.5 is the VAD's own argmax.
THRESHOLDS = (0.5, 0.3, 0.2, 0.1, 0.05)
#: Durations (seconds) a longest silence run is compared with: 20 ms to 1 s.
DURATIONS = tuple(round(0.02 * k, 2) for k in range(1, 51))
#: Seconds the window runs past the next word's aligned onset (a CTC spike fires late).
WINDOW_PAD_S = 0.04
#: The operating-point rule's precision floor (pre-stated by the coordinator, 2026-10-10).
MIN_PRECISION = 0.90

CALIBRATION = "calibration"
TEST = "test"
SPLITS = (CALIBRATION, TEST)


def longest_silence(probabilities: np.ndarray, start_s: float, end_s: float,
                    threshold: float) -> float:
    """The longest run of frames starting in ``[start_s, end_s)`` whose silence probability
    is at least ``threshold``, in seconds."""
    seconds_per_frame = FRAME_SAMPLES / TARGET_SAMPLE_RATE
    lo = max(0, math.ceil(round(start_s / seconds_per_frame, 6)))
    hi = min(len(probabilities), math.ceil(round(end_s / seconds_per_frame, 6)))
    longest = run = 0
    for silent in probabilities[lo:hi] >= threshold:
        run = run + 1 if silent else 0
        longest = max(longest, run)
    return round(longest * seconds_per_frame, 2)


@dataclass(frozen=True)
class Window:
    """Where one boundary is read: its clip and the seconds between the two words' onsets."""

    key: str
    audio_filename: str
    start_s: float
    end_s: float


def boundary_window(word_times: Sequence[float], word: int) -> list[float]:
    """The seconds a boundary before ayah word ``word`` is read in: from the onset of the
    word before it to the onset of ``word``, plus :data:`WINDOW_PAD_S`."""
    return [word_times[word - 1], word_times[word] + WINDOW_PAD_S]


def window_scores(probabilities: np.ndarray, window: Window) -> dict[str, float]:
    """The window's longest silence run at every threshold, keyed by the threshold."""
    return {str(t): longest_silence(probabilities, window.start_s, window.end_s, t)
            for t in THRESHOLDS}


def fires(scores: dict[str, float], threshold: float, duration: float) -> bool:
    return scores[str(threshold)] >= duration


# --- calibration -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """A detector's precision and recall on one set of boundaries."""

    true_positives: int
    false_positives: int
    positives: int

    @property
    def precision(self) -> float | None:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else None

    @property
    def recall(self) -> float | None:
        return self.true_positives / self.positives if self.positives else None

    def as_dict(self) -> dict:
        return {"tp": self.true_positives, "fp": self.false_positives,
                "positives": self.positives,
                "precision": _round(self.precision), "recall": _round(self.recall)}


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def point(predicted: Iterable[bool], truth: Iterable[bool]) -> Point:
    pairs = list(zip(predicted, truth, strict=True))
    return Point(sum(p and t for p, t in pairs), sum(p and not t for p, t in pairs),
                 sum(t for _, t in pairs))


def operating_point(curve: Sequence[dict]) -> dict | None:
    """The pre-stated rule over a calibration curve: the highest recall with precision at
    least :data:`MIN_PRECISION`; ties go to higher precision, then the longer duration, then
    the higher threshold. ``None`` when no point meets the precision floor."""
    eligible = [p for p in curve
                if p["precision"] is not None and p["precision"] >= MIN_PRECISION]
    if not eligible:
        return None
    return max(eligible, key=lambda p: (p["recall"], p["precision"], p["duration"],
                                        p["threshold"]))


def curve_over(rows: Sequence[dict]) -> list[dict]:
    """The local check's precision and recall on ``rows`` at every threshold and duration."""
    truth = [b["waqf"] for b in rows]
    return [{"threshold": t, "duration": d,
             **point((fires(b["scores"], t, d) for b in rows), truth).as_dict()}
            for t in THRESHOLDS for d in DURATIONS]


def calibrate(boundaries: Sequence[dict]) -> dict:
    """The local check's calibration curve on the calibration boundaries the segmenter did
    not cut (where the guard applies it), the operating point the rule picks on it, the curve
    over every calibration boundary for reference, and, at that point, every split's
    precision and recall for the check alone, for the segmenter, for both (a pause when
    either fires: the reference the guard leaves) and for the check where the segmenter did
    not cut. ``boundaries`` are fixture rows of :mod:`tadabur.waqf_guard` with ``split``,
    ``waqf`` (the owner's verdict), ``segmenter`` and ``scores``."""
    by_split = {s: [b for b in boundaries if b["split"] == s] for s in SPLITS}
    calibration = by_split[CALIBRATION]
    curve = curve_over([b for b in calibration if not b["segmenter"]])
    chosen = operating_point(curve)

    def report(rows: list[dict]) -> dict:
        truth = [b["waqf"] for b in rows]
        segmenter = [b["segmenter"] for b in rows]
        out = {"boundaries": len(rows), "segmenter": point(segmenter, truth).as_dict()}
        if chosen is None:
            return out
        check = [fires(b["scores"], chosen["threshold"], chosen["duration"]) for b in rows]
        residual = [(c, t) for c, s, t in zip(check, segmenter, truth) if not s]
        return {**out,
                "local_check": point(check, truth).as_dict(),
                "segmenter_or_local_check": point(
                    (c or s for c, s in zip(check, segmenter)), truth).as_dict(),
                "local_check_where_segmenter_did_not_cut": point(
                    (c for c, _ in residual), (t for _, t in residual)).as_dict()}

    return {"rule": f"highest calibration recall with precision >= {MIN_PRECISION} on the "
                    "boundaries the segmenter did not cut; ties: higher precision, longer "
                    "duration, higher threshold",
            "selected_on": "calibration boundaries the segmenter did not cut",
            "operating_point": None if chosen is None else
            {"threshold": chosen["threshold"], "duration": chosen["duration"]},
            "splits": {s: report(rows) for s, rows in by_split.items()},
            "calibration_curve": curve,
            "calibration_curve_every_boundary": curve_over(calibration)}


# --- the VAD pass (GPU) ----------------------------------------------------------------------


def silence_probabilities(waveform: np.ndarray, model, processor, device, dtype) -> np.ndarray:
    """The VAD's silence probability for every 20 ms frame of a 16 kHz clip, read in the
    chunks ``segment_recitations`` reads (frame ``k`` of chunk ``c`` starts at sample
    ``c * CHUNK_SAMPLES + k * FRAME_SAMPLES``)."""
    import torch

    frames = []
    for start in range(0, len(waveform), CHUNK_SAMPLES):
        chunk = waveform[start:start + CHUNK_SAMPLES]
        inputs = processor([chunk], sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = model(input_features=inputs["input_features"].to(device, dtype=dtype),
                           attention_mask=inputs["attention_mask"].to(device, dtype=dtype),
                           return_dict=False)[0][0]
        probabilities = torch.softmax(logits.float(), dim=-1)[:, SILENCE_LABEL].cpu().numpy()
        expected = -(-len(chunk) // FRAME_SAMPLES)
        frames.append(probabilities[:expected])
    return np.concatenate(frames) if frames else np.zeros(0)


def read_windows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="score every listed window with the VAD")
    score.add_argument("--windows", type=Path, required=True)
    score.add_argument("--audio-dir", type=Path, action="append", required=True,
                       help="a directory of staged clips (repeatable; the first holding a "
                            "clip is used)")
    score.add_argument("--out", type=Path, required=True,
                       help="the windows with their scores (null where unplaced)")
    score.add_argument("--provenance", type=Path, required=True,
                       help="where to write the VAD pass's provenance (JSON)")
    score.add_argument("--device", default="cuda")
    score.add_argument("--vad-dtype", default="bfloat16")
    args = parser.parse_args()

    import torch

    from training.tashkeel_eval import write_text_atomically

    from .audio import read_audio
    from .staged_audio import load_staged_clips, verify_staged
    from .vad import _load_vad

    rows = read_windows(args.windows)
    registry = load_staged_clips()
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["window"] is not None:
            by_clip[row["audio_filename"]].append(row)
    device, dtype = torch.device(args.device), getattr(torch, args.vad_dtype)
    model, processor = _load_vad(device, dtype)
    scores: dict[str, dict[str, float]] = {}
    for name in sorted(by_clip):
        audio_dir = next(d for d in args.audio_dir if (d / name).exists())
        verify_staged(registry[name], audio_dir)
        waveform = read_audio(audio_dir / name, dtype="float32")[0]
        probabilities = silence_probabilities(waveform, model, processor, device, dtype)
        for row in by_clip[name]:
            window = Window(row["key"], name, *row["window"])
            scores[row["key"]] = window_scores(probabilities, window)
    write_text_atomically(args.out, "".join(
        json.dumps({**row, "scores": scores.get(row["key"])}, ensure_ascii=False,
                   sort_keys=True) + "\n" for row in rows))
    provenance = {"vad_model": model.config._name_or_path,
                  "vad_revision": getattr(model.config, "_commit_hash", None),
                  "vad_dtype": args.vad_dtype, "frame_samples": FRAME_SAMPLES,
                  "chunk_samples": CHUNK_SAMPLES, "window_pad_s": WINDOW_PAD_S,
                  "thresholds": list(THRESHOLDS), "clips": len(by_clip),
                  "windows": len(scores), "unplaced": len(rows) - len(scores)}
    write_text_atomically(args.provenance, json.dumps(provenance, indent=1, sort_keys=True)
                          + "\n")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
