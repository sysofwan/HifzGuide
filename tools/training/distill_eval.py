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

**The final window is flushed.** Confirmation commits only the oldest second of each
window, so without a flush the last 4 s of every clip -- and for a clip under 5 s, everything
past the first second -- is decoded and then thrown away. Muraja does not do that: it flushes
whatever is pending when speech stops. Leaving the flush out was defensible while this module
only reported *character* agreement, where both models lose the same tail; it is not
defensible for :mod:`training.distill_gate`, where the tail is missing phonemes in a
``match_ratio`` computed against the whole ayah, and a 3 s clip was being gated on one second
of audio. ``flush_tail`` is therefore on by default and :data:`PROTOCOL_VERSION` records it,
because it moves every number this module and the gate produce. Pass ``flush_tail=False``
only to reproduce a pre-flush measurement.

Three gaps to the deployed protocol remain, and none of them is established to be harmless.
The VAD gate that skips inference during silence is ignored: it treats both models
identically, but that is not the same as not moving the agreement -- it selects which regions
are scored, and the two models need not disagree at the same rate inside and outside them.
The preview inferences are skipped, which is safe in that they never enter the transcript.
And a tail of under one second past the last full window is never decoded at all -- a 5.9 s
clip is one window covering its first 5 s, and no confirmation rule can recover audio the
model never saw. Closing these needs a deployment replay fixture (fractional endings, short
clips, seam-spanning runs, silence), not a choice between policies by which scores better.

One thing the flush does **not** fix, because it predates it: a segment straddling the
confirmation boundary can be emitted twice. A run at steps 18-29 of one window has midpoint
23.5 and commits; the same audio lands at steps 0-4 of the next window and commits again.
``confirmed_stream`` concatenates without reconciliation, faithfully to ``predictSplit``.
Whether the device dedupes is unverified here.

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
from training.distill_student import DEPLOYED_LOGIT_FRAMES, PRESETS, build_student

# The device advances its buffer by 1 s per confirmed pass; at 125 timesteps per 5 s window
# that is 25 timesteps, which is also ``CONFIRM_TIMESTEPS``.
HOP_SAMPLES = SAMPLE_RATE

# Bumped whenever the replayed protocol changes what a clip decodes to. Cached teacher
# decodes carry it (``training.decode_evalset``) so a manifest built under one protocol cannot
# be silently scored under another. v1 was the unflushed stream; v2 flushes the last window.
PROTOCOL_VERSION = "confirmed-stream-v2-flush"


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


def confirm_split_for_window(
    position: int, last_index: int, flush_tail: bool = True
) -> int:
    """How many of a window's 125 timesteps commit to the transcript.

    Every window commits its oldest second (``CONFIRM_TIMESTEPS``), because the next window
    will re-decode the rest with more right context. The **last** window has no next window,
    so its remaining timesteps are either flushed or silently discarded -- and discarding
    them drops the last 4 s of every clip, or all but the first second of a clip shorter than
    one window. Muraja flushes them when speech stops; so does this, unless ``flush_tail`` is
    off for a pre-``PROTOCOL_VERSION`` comparison.
    """
    if flush_tail and position == last_index:
        return DEPLOYED_LOGIT_FRAMES
    return CONFIRM_TIMESTEPS


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
    flush_tail: bool = True,
) -> list[int]:
    """Replay the deployed protocol over one clip and return its confirmed tokens.

    Every window commits the segments in its oldest second. The **last** window additionally
    commits everything still pending, which is the silence flush: no later window exists to
    re-decode those timesteps, so they are either flushed or lost. Without the flush a clip
    is transcribed only up to its last 4 seconds, and a clip shorter than one window is
    transcribed from its first second alone.
    """
    starts = clip_windows(len(samples))

    windows = []
    for start in starts:
        chunk = samples[start : start + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        windows.append(chunk)

    last_index = len(windows) - 1
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
        for position, row in enumerate(ids, start=offset):
            stream.extend(
                confirmed_tokens(
                    row[:DEPLOYED_LOGIT_FRAMES],
                    confirm_split_for_window(position, last_index, flush_tail),
                )
            )

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


def load_student_from_checkpoint(
    checkpoint_path: Path, device: torch.device, use_ema: bool = False
):
    """Rebuild the student described by a checkpoint and load its weights.

    Returns the run's persisted config alongside the model: the eval tools need it to
    refuse a split the checkpoint was not trained under.

    ``use_ema`` selects the averaged weights a run with ``--ema-decay`` stored beside the
    live ones. It raises rather than falling back when they are absent: silently scoring the
    live weights under an ``--ema`` flag would report the wrong model's number, and the two
    are meant to be compared.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = state["config"]
    student = build_student(PRESETS[config["preset"]])
    if use_ema:
        # Reconstructed from the averager's shadow rather than read from a second stored
        # copy: the shadow holds only the floating-point tensors, so the live state supplies
        # the remaining buffers. Older checkpoints carry a materialised ``student_ema`` and
        # are still read directly.
        if "student_ema" in state:
            student.load_state_dict(state["student_ema"])
        elif "ema_state" in state:
            live = state["student"]
            shadow = state["ema_state"]["shadow"]
            student.load_state_dict(
                {
                    name: (shadow[name].to(value.dtype) if name in shadow else value)
                    for name, value in live.items()
                }
            )
        else:
            raise SystemExit(
                f"{checkpoint_path} carries no averaged weights -- it was trained without "
                f"--ema-decay. Drop --ema, or train a run that keeps an average."
            )
    else:
        student.load_state_dict(state["student"])
    student = student.to(device)
    student.eval()
    return student, config, state["step"]


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
    from torch.utils.data import DataLoader

    from training.distill_data import DistillWindowDataset, build_window_index
    from training.distill_train import _init_worker, load_teacher

    device = torch.device("cuda")
    student, preset, step = load_student_from_checkpoint(checkpoint, device)
    teacher = load_teacher(device)

    _, val_clips = split_clips(discover_clips(audio_root), val_fraction)
    refs = build_window_index(val_clips)[:num_windows]
    loader = DataLoader(
        DistillWindowDataset(refs),
        batch_size=16,
        num_workers=4,
        worker_init_fn=_init_worker,
    )

    totals: dict[str, float] = {}
    batches = 0
    for features in loader:
        features = features.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
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


# --- Scoring a student against a frozen set with the teacher's decode cached ---


@dataclass(frozen=True)
class DecodeAgreement:
    """Teacher decode against student decode, and nothing downstream of them.

    This is the distillation metric. Size distillation is behavioural cloning of the
    teacher's phoneme stream, so the measurement is that stream against the student's --
    no ayah reference, no Smith-Waterman alignment, no threshold. Those belong to the
    ADR-0001 corpus filter and to Muraja's follow-along grading, which are different
    questions on different tracks; ADR-0008 records that the gate "should not be the
    headline metric at all", and for a distillation it is not even the right *kind* of
    number.

    ``char_accuracy`` pools edits over the corpus rather than averaging per clip, so one
    short clip cannot swing it. ``median_clip_error`` is reported beside it because the
    pooled figure is dominated by long clips and the two move apart: a model can improve
    the median while a heavy tail holds the pooled number down.
    """

    num_clips: int
    num_reciters: int
    char_accuracy: float
    ci_low: float
    ci_high: float
    exact_match: float
    median_clip_error: float
    p90_clip_error: float
    total_edits: int
    total_teacher_phonemes: int

    def as_dict(self) -> dict:
        return {
            "num_clips": self.num_clips,
            "num_reciters": self.num_reciters,
            "char_accuracy": round(self.char_accuracy, 4),
            "char_accuracy_ci95_clustered": [round(self.ci_low, 4), round(self.ci_high, 4)],
            "exact_match": round(self.exact_match, 4),
            "median_clip_error": round(self.median_clip_error, 4),
            "p90_clip_error": round(self.p90_clip_error, 4),
            "total_edits": self.total_edits,
            "total_teacher_phonemes": self.total_teacher_phonemes,
        }


def score_decode_agreement(per_clip: list[tuple[int, int, int]]) -> DecodeAgreement:
    """Aggregate ``(edits, teacher_phonemes, reciter_id)`` triples.

    The interval is bootstrapped over **reciters**. The clips are not independent -- 2,000 of
    them come from 286 voices and agreement correlates within one -- so an independent-sample
    interval is too narrow on exactly the question a checkpoint comparison asks.
    """
    import random
    import statistics

    from training.decode_evalset import wilson_interval

    if not per_clip:
        # 1 - 0/0 has no answer and 1 - 0/1 is 1.0, which would print as perfect agreement
        # produced by scoring nothing. Report zero so a failure looks like a failure.
        return DecodeAgreement(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    edits = sum(e for e, _, _ in per_clip)
    phonemes = sum(t for _, t, _ in per_clip)
    rates = sorted(e / max(1, t) for e, t, _ in per_clip)

    grouped: dict[int, list[tuple[int, int]]] = {}
    for edit_count, tokens, reciter in per_clip:
        grouped.setdefault(reciter, []).append((edit_count, tokens))
    keys = list(grouped)
    if len(keys) < 2:
        low, high = wilson_interval(phonemes - edits, max(1, phonemes))
    else:
        rng = random.Random(7)
        draws = []
        for _ in range(4000):
            drawn_edits = drawn_tokens = 0
            for _ in keys:
                for edit_count, tokens in grouped[keys[rng.randrange(len(keys))]]:
                    drawn_edits += edit_count
                    drawn_tokens += tokens
            draws.append(1 - drawn_edits / max(1, drawn_tokens))
        draws.sort()
        low, high = draws[100], draws[3899]

    return DecodeAgreement(
        num_clips=len(per_clip),
        num_reciters=len(keys),
        char_accuracy=1.0 - edits / max(1, phonemes),
        ci_low=low,
        ci_high=high,
        exact_match=sum(1 for e, _, _ in per_clip if e == 0) / len(per_clip),
        median_clip_error=statistics.median(rates),
        p90_clip_error=rates[min(len(rates) - 1, int(0.90 * len(rates)))],
        total_edits=edits,
        total_teacher_phonemes=phonemes,
    )


@dataclass(frozen=True)
class PairedDelta:
    """Two checkpoints' pooled accuracy difference, with a paired reciter-clustered interval.

    The obvious comparison -- count clips where this checkpoint is closer than the other --
    is a **sign test**. It asks whether more clips improved than worsened, which is not the
    claim anyone makes from it: the headline is a pooled edit-rate difference, and a sign
    test neither weights by how much a clip moved nor accounts for the clips of one reciter
    not being independent. Both errors push the p-value the same way, toward significance.

    This resamples whole reciters and recomputes the pooled difference inside each draw, so
    the pairing (both checkpoints see the same clips), the clustering and the magnitude all
    survive.
    """

    delta: float
    ci_low: float
    ci_high: float
    clips_closer: int
    clips_further: int

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero -- the actual claim, not the sign test's."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def as_dict(self) -> dict:
        return {
            "pooled_accuracy_delta": round(self.delta, 5),
            "delta_ci95_clustered": [round(self.ci_low, 5), round(self.ci_high, 5)],
            "clips_closer": self.clips_closer,
            "clips_further": self.clips_further,
            "significant": self.significant,
        }


def paired_reciter_bootstrap(
    rows: list[tuple[int, int, int, int]], iterations: int = 4000, seed: int = 11
) -> PairedDelta:
    """``(edits_this, edits_other, teacher_phonemes, reciter_id)`` -> pooled delta and interval."""
    import random

    if not rows:
        return PairedDelta(0.0, 0.0, 0.0, 0, 0)

    def pooled(sample):
        this = sum(r[0] for r in sample)
        other = sum(r[1] for r in sample)
        tokens = max(1, sum(r[2] for r in sample))
        return (other - this) / tokens  # positive = this checkpoint is closer

    grouped: dict[int, list] = {}
    for row in rows:
        grouped.setdefault(row[3], []).append(row)
    keys = list(grouped)

    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sample = []
        for _ in keys:
            sample.extend(grouped[keys[rng.randrange(len(keys))]])
        draws.append(pooled(sample))
    draws.sort()
    return PairedDelta(
        delta=pooled(rows),
        ci_low=draws[int(0.025 * iterations)],
        ci_high=draws[int(0.975 * iterations)],
        clips_closer=sum(1 for r in rows if r[0] < r[1]),
        clips_further=sum(1 for r in rows if r[0] > r[1]),
    )


def run_evalset(args, device) -> None:
    """Score one checkpoint's decode against a frozen set's cached teacher decode."""
    import soundfile as sf
    from transformers import SeamlessM4TFeatureExtractor

    from training.decode_evalset import (
        CLIPS_DIRNAME,
        check_provenance,
        load_manifest,
        paired_comparison,
    )
    from training.distill_gate import tokens_to_phonemes
    from training.distill_student import TEACHER_MODEL_ID

    evalset = load_manifest(args.eval_set)
    check_provenance(evalset, TEACHER_MODEL_ID)
    # Inherit the manifest's batch size unless told otherwise. bf16 accumulation makes the
    # decode depend on it -- bit-identical at the same batch, 0.17% adrift at batch 4 -- and
    # that is the same order as a real gain, so letting it float would let a rerun look like
    # progress.
    batch_size = args.batch_size or evalset.provenance.get("batch_size", 16)
    if args.batch_size and args.batch_size != evalset.provenance.get("batch_size"):
        print(
            f"[warn] scoring at batch {args.batch_size}, manifest built at "
            f"{evalset.provenance.get('batch_size')}: expect ~0.2% of characters to move "
            f"for that reason alone. Comparisons across batch sizes are refused.",
            flush=True,
        )
    student, state_config, step = load_student_from_checkpoint(
        args.checkpoint, device, use_ema=args.ema
    )
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(TEACHER_MODEL_ID)
    clips_dir = Path(args.eval_set) / CLIPS_DIRNAME

    decodes: dict[str, str] = {}
    for index, clip in enumerate(evalset.clips, start=1):
        samples, rate = sf.read(str(clips_dir / clip.filename), dtype="float32")
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{clip.filename} is {rate} Hz, not {SAMPLE_RATE}")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        decodes[clip.filename] = tokens_to_phonemes(
            confirmed_stream(student, extractor, samples, device, batch_size)
        )
        if index % 200 == 0:
            print(f"  {index}/{len(evalset.clips)} clips", flush=True)

    # dev by default. A reciter-disjoint test half is only a held-out panel while nothing
    # has been chosen by looking at it, and printing it on every experiment is how it stops
    # being one. --split test is an explicit act.
    scored_clips = evalset.subset(args.split)
    if not scored_clips:
        raise SystemExit(f"no clips in split {args.split!r}")
    reports = {
        args.split: score_decode_agreement(
            [
                (
                    levenshtein(list(c.teacher_text), list(decodes[c.filename])),
                    len(c.teacher_text),
                    c.reciter_id,
                )
                for c in scored_clips
            ]
        )
    }

    comparison = None
    if args.compare_decodes:
        previous = json.loads(Path(args.compare_decodes).read_text(encoding="utf-8"))
        mismatched = [
            f"  {key}: this run={mine!r} comparison file={previous.get(key)!r}"
            for key, mine in (
                ("evalset_fingerprint", evalset.fingerprint()),
                ("protocol_version", PROTOCOL_VERSION),
                ("batch_size", batch_size),
            )
            if previous.get(key) != mine
        ]
        if mismatched:
            raise SystemExit(
                "refusing to compare: those decodes were produced against a different "
                "evaluation.\n" + "\n".join(mismatched)
            )
        other = previous["decodes"]
        comparison = paired_reciter_bootstrap(
            [
                (
                    levenshtein(list(c.teacher_text), list(decodes[c.filename])),
                    levenshtein(list(c.teacher_text), list(other[c.filename])),
                    len(c.teacher_text),
                    c.reciter_id,
                )
                for c in scored_clips
                if c.filename in other
            ]
        ).as_dict()

    payload = {
        "checkpoint": str(args.checkpoint),
        "preset": state_config["preset"],
        "step": step,
        "weights": "ema" if args.ema else "live",
        "protocol_version": PROTOCOL_VERSION,
        "eval_set": str(args.eval_set),
        "evalset_fingerprint": evalset.fingerprint(),
        "batch_size": batch_size,
        "split": args.split,
        "splits": {name: report.as_dict() for name, report in reports.items()},
        "comparison": comparison,
    }
    if args.save_decodes:
        Path(args.save_decodes).write_text(
            json.dumps({**payload, "decodes": decodes}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"\nPhoneme-decode agreement -- {payload['preset']} @ step {step} "
          f"({payload['weights']} weights) vs the cached teacher")
    for name, report in reports.items():
        print(
            f"  [{name}] {report.num_clips} clips / {report.num_reciters} reciters\n"
            f"    character accuracy  {report.char_accuracy:.2%}  "
            f"95% CI [{report.ci_low:.2%}, {report.ci_high:.2%}] over reciters\n"
            f"    exact-match clips   {report.exact_match:.1%}\n"
            f"    per-clip error      median {report.median_clip_error:.2%}, "
            f"p90 {report.p90_clip_error:.2%}\n"
            f"    edits / phonemes    {report.total_edits:,} / "
            f"{report.total_teacher_phonemes:,}"
        )
    if comparison:
        low, high = comparison["delta_ci95_clustered"]
        print(
            f"  paired vs {args.compare_decodes}:\n"
            f"    pooled accuracy delta {comparison['pooled_accuracy_delta']:+.2%}  "
            f"95% CI [{low:+.2%}, {high:+.2%}] over reciters"
            f"  {'(excludes zero)' if comparison['significant'] else '(includes zero)'}\n"
            f"    clips closer {comparison['clips_closer']}, "
            f"further {comparison['clips_further']}  "
            f"-- descriptive only; the delta above is the claim"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirmed-stream agreement between a distilled student and the teacher"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--eval-set",
        type=Path,
        help="a frozen set from training.decode_evalset. Preferred: the teacher's decode is "
        "cached in it, so only the student runs and two checkpoints are scored against "
        "identical targets.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="decode BOTH models over a clip directory instead (the older path)",
    )
    parser.add_argument(
        "--save-decodes", type=Path, help="write this student's per-clip decodes"
    )
    parser.add_argument(
        "--compare-decodes",
        type=Path,
        help="another run's saved decodes; adds a paired test over per-clip edit distance",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="score the averaged weights a --ema-decay run stored beside the live ones",
    )
    parser.add_argument(
        "--breakout",
        action="store_true",
        help="report distance-from-breakout instead of stream agreement; use while the "
        "student is still blank-collapsed, when argmax agreement is uninformative",
    )
    parser.add_argument("--num-windows", type=int, default=320)
    parser.add_argument(
        "--split",
        default="dev",
        help="with --eval-set: dev (default), test or both. The reciter-disjoint test half "
        "is a held-out panel only while nothing has been chosen by looking at it, so asking "
        "for it is an explicit act. With --audio-root: val or train -- running both "
        "separates a generalisation gap from a ceiling.",
    )
    parser.add_argument(
        "--num-clips", type=int, default=200, help="held-out clips to evaluate"
    )
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 inherits the evaluation set's own batch size, which is what keeps two "
        "checkpoints comparable; the decode is bf16 and moves ~0.2%% of characters between "
        "batch sizes. Only the --audio-root path needs this set explicitly.",
    )
    parser.add_argument(
        "--no-flush-tail",
        dest="flush_tail",
        action="store_false",
        help="drop the silence flush, i.e. transcribe only up to the last 4 s of each clip. "
        "Only for reproducing a pre-" + PROTOCOL_VERSION + " measurement; the numbers are "
        "not comparable to a flushed run.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")

    if args.eval_set:
        if args.split not in ("dev", "test", "both"):
            raise SystemExit("--split must be dev, test or both when using --eval-set")
        run_evalset(args, device)
        return
    if not args.audio_root:
        raise SystemExit("pass --eval-set (preferred) or --audio-root")
    if args.split == "dev":
        args.split = "val"
    if args.split not in ("val", "train"):
        raise SystemExit("--split must be val or train when using --audio-root")

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

    student, state_config, step = load_student_from_checkpoint(
        args.checkpoint, device, use_ema=args.ema
    )
    preset = state_config["preset"]
    check_split_matches_checkpoint(state_config, args.val_fraction)
    teacher = load_teacher(device)
    extractor = SeamlessM4TFeatureExtractor.from_pretrained("obadx/muaalem-model-v3_2")

    # Default is the *validation* side -- the same hash split training used, so no clip the
    # student was fit on can inflate the number. --split train scores seen clips instead,
    # which is only useful as the paired comparison described in the flag's help.
    train_clips, val_clips = split_clips(discover_clips(args.audio_root), args.val_fraction)
    if args.split == "train" and state_config.get("stream_shards"):
        raise SystemExit(
            "--split train is meaningless for this checkpoint: it was trained with "
            f"--stream-shards {state_config['stream_shards']!r}, so every clip under "
            "--audio-root is held out and the 'train' side was never seen. Comparing the "
            "two splits would show no gap for a reason that has nothing to do with "
            "generalisation."
        )
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
            teacher, extractor, samples, device, args.batch_size or 16, args.flush_tail
        )
        student_stream = confirmed_stream(
            student, extractor, samples, device, args.batch_size or 16, args.flush_tail
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
