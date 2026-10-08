"""Does the protocol commit the *right* second of each window?

The deployed protocol commits frames 0-24 -- a window's **first** second, which has at most
0.96 s of left context and 4 s of right context. The edit decomposition found the committed
region carries roughly twice the error rate of the flushed region, and reweighting the
objective toward it (confirm_weight 2 -> 4) did not move it. One explanation the objective
cannot reach: the committed region is not harder *audio*, it is audio decoded from the one
window position with no left context.

That is a protocol question, not a training question, so this tool asks it by inference
alone. For every absolute second of every clip it collects the tokens produced at each of
the five within-window positions that can cover it:

    block b of window w  covers absolute [w+b, w+b+1)

so the same second is decoded five different ways -- b=0 with no left context and 4 s of
right, through b=4 with 4 s of left and none of right. Comparing student-teacher agreement
across b says whether committing a later block (a "lookback" protocol, at +b seconds of
latency) would reproduce the teacher better, for no training at all.

Two things are measured, because they answer different questions:

* **student-teacher agreement per block** -- the actionable one. If b=1 beats b=0 by more
  than the metric's noise, the protocol is leaving accuracy on the table.
* **teacher self-consistency per block**, against the teacher's own b=0 output. ADR-0010
  records that the teacher agrees with itself only 79-82% across window *phases*; if it is
  also inconsistent across *positions*, then "which block to commit" is partly arbitrary and
  a gain at b=1 may be the teacher moving rather than the student improving. Reading the
  first number without the second would repeat the mistake of scoring a decode against a
  reference that is itself unstable.

**This does not isolate left context.** Block b differs from block b+1 in left context *and*
right context at once. It answers "which position commits best", not "why". Isolating the
mechanism would need a window whose right context is held fixed, which the frozen
(1, 250, 160) contract cannot express.

Usage::

    python -m training.window_position --checkpoint runs/h384_unseen/checkpoint.pt \\
        --eval-set tadabur/gate_eval --out runs/h384_unseen/positions.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from training.decoding import NUM_BLOCKS, block_bounds, segments_between
from training.distill_eval import levenshtein


def block_tokens(class_ids, block: int) -> list[int]:
    """Tokens this window would commit if it committed block ``block`` instead of block 0.

    The deployed commit rule -- a segment belongs to the block its **midpoint** falls in --
    at a later offset, so block 0 reproduces :func:`training.decoding.confirmed_tokens`
    exactly. Using the midpoint (rather than the start) is what keeps a run that straddles a
    boundary owned by exactly one block. This is one window's view; a whole clip streamed
    at block ``b``, with its startup and flush, is :meth:`training.decoding.Decoder.emissions`.
    """
    return [seg.token_id for seg in segments_between(class_ids, *block_bounds(block))]


@dataclass
class PositionTally:
    """Pooled edits and reference length for one within-window position."""

    edits: int = 0
    reference: int = 0
    blocks: int = 0

    def add(self, reference: list[int], hypothesis: list[int]) -> None:
        self.edits += levenshtein(reference, hypothesis)
        self.reference += len(reference)
        self.blocks += 1

    @property
    def accuracy(self) -> float:
        return 1.0 - self.edits / self.reference if self.reference else 0.0

    def as_dict(self) -> dict:
        return {
            "edits": self.edits,
            "reference_tokens": self.reference,
            "blocks": self.blocks,
            "accuracy": round(self.accuracy, 6),
        }


def tally_positions(
    teacher_rows: list, student_rows: list
) -> tuple[list[PositionTally], list[PositionTally]]:
    """Per-block student-teacher agreement, and teacher self-consistency vs its own block 0.

    ``*_rows`` are one ``(125,)`` argmax row per window, in window order. For the
    self-consistency tally, block ``b`` of window ``w`` is compared against block 0 of window
    ``w + b`` -- the two decodes of the same absolute second -- so it is only counted where
    that later window exists.
    """
    agreement = [PositionTally() for _ in range(NUM_BLOCKS)]
    consistency = [PositionTally() for _ in range(NUM_BLOCKS)]

    teacher_blocks = [
        [block_tokens(row, b) for b in range(NUM_BLOCKS)] for row in teacher_rows
    ]
    student_blocks = [
        [block_tokens(row, b) for b in range(NUM_BLOCKS)] for row in student_rows
    ]

    for window in range(len(teacher_rows)):
        for b in range(NUM_BLOCKS):
            agreement[b].add(teacher_blocks[window][b], student_blocks[window][b])
            # The same absolute second, as the deployed protocol would have decoded it.
            deployed = window + b
            if deployed < len(teacher_rows):
                consistency[b].add(
                    teacher_blocks[deployed][0], teacher_blocks[window][b]
                )

    return agreement, consistency


def main() -> None:
    import torch
    from transformers import SeamlessM4TFeatureExtractor

    from training.decode_evalset import (
        CLIPS_DIRNAME,
        check_provenance,
        load_manifest,
        read_clip_audio,
        scoring_batch_size,
    )
    from training.decoding import Decoder, load_student_from_checkpoint, window_audio
    from training.distill_student import TEACHER_MODEL_ID
    from training.distill_train import load_teacher

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test", "both"), default="dev")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evalset = load_manifest(args.eval_set)
    check_provenance(evalset, TEACHER_MODEL_ID)
    batch_size = scoring_batch_size(evalset, args.batch_size)

    student, _, step = load_student_from_checkpoint(
        args.checkpoint, device, use_ema=args.ema
    )
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(TEACHER_MODEL_ID)
    teacher_decoder = Decoder(load_teacher(device), extractor, device, batch_size)
    student_decoder = Decoder(student, extractor, device, batch_size)
    clips_dir = Path(args.eval_set) / CLIPS_DIRNAME
    clips = evalset.subset(args.split)
    if args.limit:
        clips = clips[: args.limit]
    print(f"[setup] {len(clips)} {args.split} clips, batch {batch_size}, step {step}")

    agreement = [PositionTally() for _ in range(NUM_BLOCKS)]
    consistency = [PositionTally() for _ in range(NUM_BLOCKS)]

    for index, clip in enumerate(clips, start=1):
        windows = window_audio(read_clip_audio(clips_dir, clip.filename))
        if len(windows) < 2:
            continue  # a single-window clip cannot compare positions

        a, c = tally_positions(
            teacher_decoder.window_rows(windows), student_decoder.window_rows(windows)
        )
        for b in range(NUM_BLOCKS):
            for dst, src in ((agreement, a), (consistency, c)):
                dst[b].edits += src[b].edits
                dst[b].reference += src[b].reference
                dst[b].blocks += src[b].blocks
        if index % 100 == 0:
            print(f"  {index}/{len(clips)} clips", flush=True)

    report = {
        "student_teacher_agreement": [t.as_dict() for t in agreement],
        "teacher_self_consistency": [t.as_dict() for t in consistency],
        "provenance": {
            "checkpoint": str(args.checkpoint),
            "step": step,
            "ema": args.ema,
            "eval_set": str(args.eval_set),
            "evalset_fingerprint": evalset.fingerprint(),
            "split": args.split,
            "batch_size": batch_size,
        },
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n  block   left ctx   student-teacher   teacher vs its own block 0")
    for b in range(NUM_BLOCKS):
        print(
            f"    b={b}      {b}.0 s      {agreement[b].accuracy:7.4%}"
            f"            {consistency[b].accuracy:7.4%}"
        )
    deployed, best = agreement[0].accuracy, max(t.accuracy for t in agreement)
    print(
        f"\n  deployed block 0 {deployed:.4%}; best block {best:.4%} "
        f"(+{100 * (best - deployed):.3f} points if the protocol committed it)"
    )
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
