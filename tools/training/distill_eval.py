"""Student-vs-teacher agreement on the stream Muraja actually shows the user.

Frame-level agreement (``training.distill_loss.agreement_stats``) is cheap and smooth, which
makes it a good training signal, but it is not the release gate. It averages over all 125
timesteps of every window, and 100 of those never reach the user. A student could look
excellent there and still produce a visibly different transcript.

This module measures the thing that decides: the **confirmed phoneme stream** produced by
replaying the deployed sliding-window protocol.

The protocol, replicated from ``MuaalemInference.predictSplit`` and ``RealtimeTranscriber``:

* A 5 s window of audio is feature-extracted on its own (per-window normalization) and run
  to 125 CTC timesteps.
* ``scanCTC`` collapses the greedy argmax into contiguous runs of the same **non-blank**
  token -- each run is one segment with a midpoint.
* A segment is **confirmed** when ``midpoint < 25``, i.e. it sits in the oldest second of
  the buffer, having accumulated the full 4 s of right context. Everything later stays
  provisional and is re-decoded by the next window.
* The window advances by 1 s and the next pass confirms the next second.

Concatenating the confirmed segments across a clip gives the transcript the user sees. We
build that stream for both models and compare them.

Two deliberate simplifications, neither of which favours the student. The silence flush
(which confirms whatever is pending when speech stops) is not replayed, so we measure the
steady-state stream; and the VAD gate that skips inference during silence is ignored,
because it gates *both* models identically and so cannot move the agreement. The preview
inferences are likewise skipped -- they are provisional and never enter the transcript,
though they are why per-window cost matters so much.

Usage::

    python -m training.distill_eval --checkpoint runs/h384/checkpoint.pt \\
        --audio-root ../tadabur/audit_run/clips_v2 --num-clips 200

    python -m training.distill_eval --checkpoint runs/h384/checkpoint.pt \\
        --audio-root <dir> --num-clips 200 --json > agreement.json

Linux + CUDA (both models must be resident).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from training.distill_data import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    discover_clips,
    split_clips,
)
from training.distill_loss import BLANK_ID, CONFIRM_TIMESTEPS
from training.distill_student import PRESETS, build_student

# The device advances its buffer by 1 s per confirmed pass; at 125 timesteps per 5 s window
# that is 25 timesteps, which is also ``CONFIRM_TIMESTEPS``.
HOP_SAMPLES = SAMPLE_RATE
LOGIT_FRAMES = 125


@dataclass(frozen=True)
class Segment:
    """A contiguous run of one non-blank CTC token -- Swift's ``CTCSegment``."""

    token_id: int
    start_step: int
    end_step: int  # inclusive

    @property
    def midpoint(self) -> float:
        return (self.start_step + self.end_step) / 2.0


def scan_ctc(class_ids: np.ndarray) -> list[Segment]:
    """Collapse per-timestep argmax into non-blank runs, mirroring Swift's ``scanCTC``.

    A run ends when the token changes. Blank runs are tracked (they separate repeated
    tokens, which is the whole point of the CTC blank) but never emitted.
    """
    segments: list[Segment] = []
    current_token = -1
    current_start = 0

    for step, token in enumerate(int(t) for t in class_ids):
        if token == current_token:
            continue
        if current_token != BLANK_ID and current_token != -1:
            segments.append(Segment(current_token, current_start, step - 1))
        current_token = token
        current_start = step

    if current_token != BLANK_ID and current_token != -1:
        segments.append(Segment(current_token, current_start, len(class_ids) - 1))

    return segments


def confirmed_tokens(
    class_ids: np.ndarray, confirm_timesteps: int = CONFIRM_TIMESTEPS
) -> list[int]:
    """The tokens one window commits to the transcript: segments with midpoint < split."""
    return [
        seg.token_id
        for seg in scan_ctc(class_ids)
        if seg.midpoint < float(confirm_timesteps)
    ]


def levenshtein(a: list[int], b: list[int]) -> int:
    """Edit distance between two token streams."""
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (token_a != token_b),
                )
            )
        previous = current
    return previous[-1]


def clip_windows(num_samples: int, hop_samples: int = HOP_SAMPLES) -> list[int]:
    """Window starts for the deployed protocol: advance 1 s while audio remains.

    Only full windows confirm in steady state, so a clip shorter than one window yields a
    single (padded) pass and the tail past the last full window is not replayed -- the same
    place the real pipeline hands over to the silence flush.
    """
    if num_samples < WINDOW_SAMPLES:
        return [0]
    return list(range(0, num_samples - WINDOW_SAMPLES + 1, hop_samples))


@torch.no_grad()
def confirmed_stream(
    model,
    extractor,
    samples: np.ndarray,
    device: torch.device,
    batch_size: int = 16,
) -> list[int]:
    """Replay the deployed protocol over one clip and return its confirmed tokens."""
    starts = clip_windows(len(samples))

    windows = []
    for start in starts:
        chunk = samples[start : start + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        windows.append(chunk)

    stream: list[int] = []
    for offset in range(0, len(windows), batch_size):
        batch = windows[offset : offset + batch_size]
        extracted = extractor(
            batch, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )
        features = extracted.input_features.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(features, return_dict=True)["logits"]["phonemes"]
        ids = logits.float().argmax(dim=-1).cpu().numpy()
        for row in ids:
            stream.extend(confirmed_tokens(row[:LOGIT_FRAMES]))

    return stream


@dataclass
class AgreementReport:
    """Decoded-stream agreement, in the same shape as the quantization table's columns.

    ``exact_match`` and ``char_accuracy`` deliberately mirror
    ``ml-model-transformation.md`` section 1.3 so a distilled student can be read straight
    against the INT8 / 6-bit / 4-bit rows already measured there.
    """

    num_clips: int
    exact_match: float
    char_accuracy: float
    mean_teacher_length: float
    mean_student_length: float
    total_edits: int
    total_teacher_tokens: int

    def as_dict(self) -> dict:
        return {
            "num_clips": self.num_clips,
            "exact_match": round(self.exact_match, 4),
            "char_accuracy": round(self.char_accuracy, 4),
            "mean_teacher_length": round(self.mean_teacher_length, 1),
            "mean_student_length": round(self.mean_student_length, 1),
            "total_edits": self.total_edits,
            "total_teacher_tokens": self.total_teacher_tokens,
        }


def compare_streams(pairs: list[tuple[list[int], list[int]]]) -> AgreementReport:
    """Aggregate per-clip (teacher, student) streams into the report.

    ``char_accuracy`` is edit distance pooled over the corpus rather than averaged
    per-clip, so one short clip cannot swing it the way a per-clip mean would.
    """
    exact = 0
    edits = 0
    teacher_tokens = 0
    teacher_lengths = []
    student_lengths = []

    for teacher_stream, student_stream in pairs:
        if teacher_stream == student_stream:
            exact += 1
        edits += levenshtein(teacher_stream, student_stream)
        teacher_tokens += len(teacher_stream)
        teacher_lengths.append(len(teacher_stream))
        student_lengths.append(len(student_stream))

    count = max(1, len(pairs))
    return AgreementReport(
        num_clips=len(pairs),
        exact_match=exact / count,
        char_accuracy=1.0 - (edits / max(1, teacher_tokens)),
        mean_teacher_length=sum(teacher_lengths) / count,
        mean_student_length=sum(student_lengths) / count,
        total_edits=edits,
        total_teacher_tokens=teacher_tokens,
    )


def load_student_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Rebuild the student described by a checkpoint and load its weights."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    preset = state["config"]["preset"]
    student = build_student(PRESETS[preset])
    student.load_state_dict(state["student"])
    student = student.to(device)
    student.eval()
    return student, preset, state["step"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirmed-stream agreement between a distilled student and the teacher"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--num-clips", type=int, default=200, help="held-out clips to evaluate"
    )
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")

    import soundfile as sf
    from transformers import SeamlessM4TFeatureExtractor

    from training.distill_train import load_teacher

    student, preset, step = load_student_from_checkpoint(args.checkpoint, device)
    teacher = load_teacher(device)
    extractor = SeamlessM4TFeatureExtractor.from_pretrained("obadx/muaalem-model-v3_2")

    # Evaluate on the *validation* clips only -- the same hash split training used, so no
    # clip the student was fit on can inflate the number.
    _, val_clips = split_clips(discover_clips(args.audio_root), args.val_fraction)
    clips = val_clips[: args.num_clips]
    if not clips:
        raise SystemExit("no validation clips found")

    pairs: list[tuple[list[int], list[int]]] = []
    for index, path in enumerate(clips, start=1):
        try:
            samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            continue
        if rate != SAMPLE_RATE:
            continue
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        teacher_stream = confirmed_stream(
            teacher, extractor, samples, device, args.batch_size
        )
        student_stream = confirmed_stream(
            student, extractor, samples, device, args.batch_size
        )
        pairs.append((teacher_stream, student_stream))

        if not args.json and index % 25 == 0:
            print(f"  {index}/{len(clips)} clips", flush=True)

    report = compare_streams(pairs)
    payload = {
        "checkpoint": str(args.checkpoint),
        "preset": preset,
        "step": step,
        **report.as_dict(),
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"\nConfirmed-stream agreement -- {preset} @ step {step}")
    print(f"  clips evaluated      {report.num_clips}")
    print(f"  exact match          {report.exact_match:.1%}")
    print(f"  char accuracy        {report.char_accuracy:.2%}")
    print(f"  mean tokens/clip     teacher {report.mean_teacher_length:.1f}, "
          f"student {report.mean_student_length:.1f}")
    print(f"  edits / tokens       {report.total_edits} / {report.total_teacher_tokens}")


if __name__ == "__main__":
    main()
