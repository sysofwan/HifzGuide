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
from training.distill_loss import BLANK_ID, CONFIRM_TIMESTEPS, breakout_stats
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
    # An empty corpus scored 1.0 here, because `1 - 0/1` is 1.0. That prints
    # "clips evaluated 0 / char accuracy 100.00%", which is the most dangerous possible
    # reading: a headline of perfect agreement produced by scoring nothing. It happens
    # whenever every clip is skipped -- a wrong --audio-root, a non-16 kHz staging
    # directory, unreadable files. Report 0.0 so the failure looks like a failure.
    return AgreementReport(
        num_clips=len(pairs),
        exact_match=exact / count,
        char_accuracy=(1.0 - edits / teacher_tokens) if teacher_tokens else 0.0,
        mean_teacher_length=sum(teacher_lengths) / count,
        mean_student_length=sum(student_lengths) / count,
        total_edits=edits,
        total_teacher_tokens=teacher_tokens,
    )


def check_split_matches_checkpoint(saved: dict, val_fraction: float) -> None:
    """Refuse to score a split the checkpoint was not trained under.

    ``clip_split`` is a monotone hash bucket, so a *larger* ``--val-fraction`` is a
    superset: evaluating at 0.1 a run trained at 0.02 puts ~80% training clips into the
    set the report labels ``[val]``. The number looks like held-out agreement, is inflated
    by memorised clips, and nothing in the output says so -- exactly the class of silent
    wrongness this tool exists to avoid producing.
    """
    trained = saved.get("val_fraction")
    if trained is not None and trained != val_fraction:
        raise SystemExit(
            f"--val-fraction {val_fraction} does not match the checkpoint's {trained}. "
            f"The hash split is monotone, so this would score clips the student trained "
            f"on and label them held-out. Pass --val-fraction {trained}."
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


@dataclass(frozen=True)
class HeadHealth:
    """Whether a blank-collapsed student's CTC head has gone degenerate.

    There are two very different reasons a student can emit blank everywhere, and they call
    for opposite responses. Either the **head** has learned a large blank bias -- the
    classic majority-class shortcut, since blank is ~67% of frames -- in which case the fix
    is a loss or initialisation change and more training will not help. Or the head is
    fine and the **encoder** has not yet learned frame-level phoneme discrimination, in
    which case the only fix is more training and changing the loss is wasted effort.

    Measured on the h384 run at step 2000, blank led the next-highest bias by **0.004** and
    its weight-row norm sat inside the non-blank spread -- i.e. no head pathology at all,
    and the collapse was entirely upstream. That ruled out a whole class of interventions.
    """

    blank_bias: float
    other_bias_mean: float
    other_bias_max: float
    blank_weight_norm: float
    other_weight_norm_mean: float
    other_weight_norm_max: float

    @property
    def bias_lead(self) -> float:
        """How far blank's bias exceeds the best non-blank one. Large => head shortcut."""
        return self.blank_bias - self.other_bias_max

    def is_degenerate(self, threshold: float = 1.0) -> bool:
        """A blank bias this far ahead is a head problem, not a representation problem."""
        return self.bias_lead > threshold

    def as_dict(self) -> dict:
        return {
            "blank_bias": round(self.blank_bias, 4),
            "other_bias_mean": round(self.other_bias_mean, 4),
            "other_bias_max": round(self.other_bias_max, 4),
            "bias_lead": round(self.bias_lead, 4),
            "blank_weight_norm": round(self.blank_weight_norm, 4),
            "other_weight_norm_mean": round(self.other_weight_norm_mean, 4),
            "other_weight_norm_max": round(self.other_weight_norm_max, 4),
        }


def head_health(student) -> HeadHealth:
    """Blank-vs-rest statistics of the student's phoneme CTC head."""
    head = student.level_to_lm_head["phonemes"]
    bias = head.bias.detach().float().cpu()
    norms = head.weight.detach().float().cpu().norm(dim=1)

    return HeadHealth(
        blank_bias=bias[BLANK_ID].item(),
        other_bias_mean=bias[BLANK_ID + 1 :].mean().item(),
        other_bias_max=bias[BLANK_ID + 1 :].max().item(),
        blank_weight_norm=norms[BLANK_ID].item(),
        other_weight_norm_mean=norms[BLANK_ID + 1 :].mean().item(),
        other_weight_norm_max=norms[BLANK_ID + 1 :].max().item(),
    )


def run_breakout_diagnostic(
    checkpoint: Path, audio_root: Path, val_fraction: float, num_windows: int
) -> dict:
    """Is a blank-collapsed student converging or stuck? Measured, not guessed.

    Exists because argmax agreement is useless inside the all-blank basin: it reads a flat
    0 for thousands of steps whether the correct class holds 40% of the student's mass or
    0.1%. This reports the continuous quantities instead -- see
    :class:`training.distill_loss.BreakoutStats`.
    """
    import torch as _torch
    from torch.utils.data import DataLoader

    from training.distill_data import DistillWindowDataset, build_window_index
    from training.distill_train import load_teacher

    device = _torch.device("cuda")
    student, preset, step = load_student_from_checkpoint(checkpoint, device)
    teacher = load_teacher(device)

    _, val_clips = split_clips(discover_clips(audio_root), val_fraction)
    refs = build_window_index(val_clips)[:num_windows]
    loader = DataLoader(DistillWindowDataset(refs), batch_size=16, num_workers=4)

    totals: dict[str, float] = {}
    batches = 0
    for features in loader:
        features = features.to(device)
        with _torch.no_grad(), _torch.autocast("cuda", dtype=_torch.bfloat16):
            teacher_logits = teacher(features, return_dict=True)["logits"]["phonemes"]
            student_logits = student(features, return_dict=True)["logits"]["phonemes"]
        for key, value in breakout_stats(student_logits, teacher_logits).as_dict().items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1

    averaged = {k: round(v / max(1, batches), 4) for k, v in totals.items()}
    return {
        "preset": preset,
        "step": step,
        "windows": len(refs),
        **averaged,
        "head": head_health(student).as_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirmed-stream agreement between a distilled student and the teacher"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--breakout",
        action="store_true",
        help="report distance-from-breakout instead of stream agreement; use while the "
        "student is still blank-collapsed, when argmax agreement is uninformative",
    )
    parser.add_argument("--num-windows", type=int, default=320)
    parser.add_argument(
        "--split",
        choices=("val", "train"),
        default="val",
        help="which side of the clip split to score. Running both separates a "
        "generalisation gap from a ceiling: if TRAIN agreement is also stuck at the val "
        "number, more data cannot be the fix and the objective or capacity is the limit.",
    )
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

    if args.breakout:
        report = run_breakout_diagnostic(
            args.checkpoint, args.audio_root, args.val_fraction, args.num_windows
        )
        if args.json:
            print(json.dumps(report, indent=2))
            return
        print(f"\nBreakout diagnostic -- {report['preset']} @ step {report['step']}")
        print(f"  windows              {report['windows']}")
        print(f"  P(teacher's class)   {report['target_prob']:.4f}")
        print(f"  rank of that class   {report['target_rank']:.2f}   (1.0 = agrees)")
        print(f"  P(blank)             {report['blank_prob']:.4f}")
        print(f"  margin blank-target  {report['prob_margin']:+.4f}  (<=0 means escaped)")
        print(f"  top-5 agreement      {report['top5_agreement']:.4f}")
        head = report["head"]
        verdict = (
            "head shortcut -- change the loss, not the step count"
            if head["bias_lead"] > 1.0
            else "head is clean -- the collapse is upstream, in the encoder"
        )
        print(f"  blank bias lead      {head['bias_lead']:+.4f}   ({verdict})")
        return

    import soundfile as sf
    from transformers import SeamlessM4TFeatureExtractor

    from training.distill_train import load_teacher

    student, preset, step = load_student_from_checkpoint(args.checkpoint, device)
    teacher = load_teacher(device)
    extractor = SeamlessM4TFeatureExtractor.from_pretrained("obadx/muaalem-model-v3_2")

    # Default is the *validation* side -- the same hash split training used, so no clip the
    # student was fit on can inflate the number. --split train scores seen clips instead,
    # which is only useful as the paired comparison described in the flag's help.
    train_clips, val_clips = split_clips(discover_clips(args.audio_root), args.val_fraction)
    clips = (train_clips if args.split == "train" else val_clips)[: args.num_clips]
    if not clips:
        raise SystemExit(f"no {args.split} clips found")

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
        "split": args.split,
        **report.as_dict(),
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"\nConfirmed-stream agreement -- {preset} @ step {step} [{args.split}]")
    print(f"  clips evaluated      {report.num_clips}")
    print(f"  exact match          {report.exact_match:.1%}")
    print(f"  char accuracy        {report.char_accuracy:.2%}")
    print(f"  mean tokens/clip     teacher {report.mean_teacher_length:.1f}, "
          f"student {report.mean_student_length:.1f}")
    print(f"  edits / tokens       {report.total_edits} / {report.total_teacher_tokens}")


if __name__ == "__main__":
    main()
