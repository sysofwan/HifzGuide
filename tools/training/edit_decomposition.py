"""Where the student's remaining disagreement actually lives, by protocol region.

The headline metric says the student reproduces 93.09% of the teacher's decoded characters.
It cannot say *which part of the streaming protocol* the missing 6.91% comes from, and that
is the question every remaining objective experiment turns on:

- Training weights the confirmed region (the oldest second of each window) at 2x, but about
  two fifths of the **scored** characters are committed by the flushed final window, which
  training weights at 1x. If the flush region carries a disproportionate share of the edits,
  a tail weight is the indicated change; if it does not, that arm is not worth 4 GPU-hours.
- A run straddling the confirmation boundary is committed by whichever window sees its
  midpoint. A student whose run ends one frame early is charged an insertion or a deletion
  for a token it got right. If seam-straddling emissions are not over-represented in the
  edits, the seam-weight arm is likewise not indicated.

So this tool answers one question: **per teacher character, is the error rate higher in the
flush region than in the committed region, and higher at seams than away from them?**

Two methodological guards, both bought with prior mistakes:

1. The teacher is re-decoded here rather than read from the manifest's cached text, because
   the cache has no provenance. The re-decode is then asserted to equal the cache. At a
   fixed batch size the teacher is bit-identical, so any mismatch means the batch size or
   the protocol has drifted -- exactly the silent confound that makes a comparison
   meaningless.
2. Levenshtein admits many optimal alignments, and which one is chosen decides whether a
   duplicated token reads as an insertion before or after its twin -- which in turn decides
   the region it is attributed to. Every figure here is therefore computed under **three**
   backtrace orders that reach genuinely different members of the optimal-alignment set, and
   the report carries the spread between them. A region gap smaller than its own tie-break
   spread is an artefact of the measurement, not a finding about the model.

Usage::

    python -m training.edit_decomposition --checkpoint runs/h384_wsd_probe/checkpoint.pt \\
        --eval-set tadabur/gate_eval --out runs/h384_wsd_probe/edits.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from training.distill_eval import CONFIRM_TIMESTEPS, Emission

# Which branch the backtrace tries first when several are equally optimal.
#
# ``diagonal_first`` is the canonical alignment and the headline: preferring the diagonal
# keeps a matching character a match rather than an equal-cost insertion-plus-deletion. But
# it is also so dominant that it pins the walk almost everywhere, which makes it useless as
# its own robustness check -- two variants that both prefer the diagonal produce the same
# alignment and would report a reassuring zero spread for no reason. The other two orders
# deliberately take an indel where the diagonal was also optimal, which is what actually
# reaches a *different* member of the optimal-alignment set. A conclusion that survives all
# three is a property of the streams; one that does not is a property of the backtrace.
TIE_BREAKS = ("diagonal_first", "deletion_first", "insertion_first")


@dataclass(frozen=True)
class Edit:
    """One charged edit, tagged with the teacher emission it is attributed to.

    Substitutions and deletions have a teacher emission of their own. An **insertion** does
    not -- the student produced a character the teacher never emitted -- so it is attributed
    to the teacher emission it sits against in the alignment (the next one, or the last one
    when the insertion trails the stream). That is an attribution rule, not a fact about the
    audio, which is why the tie-break sensitivity in the report matters most for insertions.
    """

    kind: str  # "substitution" | "insertion" | "deletion"
    emission: Emission
    teacher_token: int | None
    student_token: int | None

    @property
    def is_adjacent_duplicate(self) -> bool:
        """Split/merge shaped: the charged token is the same as the one it sits beside.

        A CTC run that the student breaks into two (or fuses into one) shows up here. This
        is the shape a seam or a spike-timing disagreement produces, as opposed to the
        student simply hearing a different phoneme.
        """
        if self.kind == "insertion":
            return self.student_token == self.emission.token_id
        if self.kind == "deletion":
            return self.teacher_token == self.emission.token_id
        return False


def align(teacher: list[int], student: list[int], tie_break: str = "diagonal_first"):
    """Levenshtein backtrace as a list of ``(op, teacher_index, student_index)``.

    ``op`` is one of ``match``/``substitution``/``deletion``/``insertion``. Indices are into
    the respective streams, and are ``None`` on the side the op does not consume.

    The DP is the same recurrence :func:`training.distill_eval.levenshtein` scores with, and
    a branch is only ever taken when it is cost-optimal, so the edit *count* equals the
    metric's under every ``tie_break``. Only the attribution of those edits differs.
    """
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"unknown tie_break {tie_break!r}, expected one of {TIE_BREAKS}")

    rows, cols = len(teacher) + 1, len(student) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i][j] = min(
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
                cost[i - 1][j - 1] + (teacher[i - 1] != student[j - 1]),
            )

    # Walk back from the corner, trying branches in the order this tie-break asks for and
    # taking the first that is cost-optimal.
    order = {
        "diagonal_first": ("diagonal", "deletion", "insertion"),
        "deletion_first": ("deletion", "insertion", "diagonal"),
        "insertion_first": ("insertion", "deletion", "diagonal"),
    }[tie_break]

    ops: list[tuple[str, int | None, int | None]] = []
    i, j = len(teacher), len(student)
    while i > 0 or j > 0:
        for branch in order:
            if branch == "diagonal" and i > 0 and j > 0:
                same = teacher[i - 1] == student[j - 1]
                if cost[i][j] == cost[i - 1][j - 1] + (not same):
                    ops.append(("match" if same else "substitution", i - 1, j - 1))
                    i, j = i - 1, j - 1
                    break
            elif branch == "deletion" and i > 0 and cost[i][j] == cost[i - 1][j] + 1:
                ops.append(("deletion", i - 1, None))
                i -= 1
                break
            elif branch == "insertion" and j > 0 and cost[i][j] == cost[i][j - 1] + 1:
                ops.append(("insertion", None, j - 1))
                j -= 1
                break
        else:  # pragma: no cover - the DP guarantees one branch is always available
            raise AssertionError(f"no backtrace step from ({i}, {j})")

    ops.reverse()
    return ops


def decompose(
    emissions: list[Emission], student: list[int], tie_break: str = "diagonal_first"
) -> list[Edit]:
    """Charge every edit between the two streams to a teacher emission."""
    teacher = [e.token_id for e in emissions]
    if not emissions:
        return []

    edits: list[Edit] = []
    # Forward cursor over the teacher stream: the index of the next emission still to be
    # consumed. An insertion is attributed to that emission -- the one it sits against --
    # and to the final emission when it trails the whole stream.
    cursor = 0
    for kind, teacher_index, student_index in align(teacher, student, tie_break):
        if kind == "insertion":
            anchor = min(cursor, len(emissions) - 1)
            edits.append(
                Edit("insertion", emissions[anchor], None, student[student_index])
            )
            continue
        cursor = teacher_index + 1
        if kind == "match":
            continue
        emission = emissions[teacher_index]
        edits.append(
            Edit(
                kind,
                emission,
                emission.token_id,
                student[student_index] if student_index is not None else None,
            )
        )
    return edits


def _region(emission: Emission) -> str:
    return "flush" if emission.is_flush else "committed"


def summarise(
    per_clip: list[tuple[list[Emission], list[int]]], tie_break: str = "diagonal_first"
) -> dict:
    """Error rates by protocol region, pooled over clips.

    The denominators are teacher characters emitted in each region, so the rates are
    directly comparable: "of the characters the flush commits, what fraction are charged an
    edit" against the same for the committed region. That ratio is the decision.
    """
    exposure: Counter[str] = Counter()
    seam_exposure = Counter({"seam": 0, "interior": 0})
    charged: Counter[str] = Counter()
    seam_charged = Counter({"seam": 0, "interior": 0})
    by_kind: Counter[str] = Counter()
    duplicates: Counter[str] = Counter()
    total_edits = 0
    total_reference = 0

    for emissions, student in per_clip:
        total_reference += len(emissions)
        for emission in emissions:
            exposure[_region(emission)] += 1
            seam_exposure["seam" if emission.straddles_seam else "interior"] += 1
        for edit in decompose(emissions, student, tie_break):
            total_edits += 1
            region = _region(edit.emission)
            charged[region] += 1
            by_kind[f"{region}:{edit.kind}"] += 1
            by_kind[edit.kind] += 1
            seam_charged["seam" if edit.emission.straddles_seam else "interior"] += 1
            if edit.is_adjacent_duplicate:
                duplicates[region] += 1
                duplicates["total"] += 1

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    return {
        "tie_break": tie_break,
        "reference_characters": total_reference,
        "edits": total_edits,
        "character_error_rate": rate(total_edits, total_reference),
        "region": {
            name: {
                "reference_characters": exposure[name],
                "share_of_reference": rate(exposure[name], total_reference),
                "edits": charged[name],
                "share_of_edits": rate(charged[name], total_edits),
                "error_rate": rate(charged[name], exposure[name]),
                "substitutions": by_kind[f"{name}:substitution"],
                "insertions": by_kind[f"{name}:insertion"],
                "deletions": by_kind[f"{name}:deletion"],
                "adjacent_duplicates": duplicates[name],
            }
            for name in ("committed", "flush")
        },
        "seam": {
            name: {
                "reference_characters": seam_exposure[name],
                "edits": seam_charged[name],
                "error_rate": rate(seam_charged[name], seam_exposure[name]),
            }
            for name in ("seam", "interior")
        },
        "kinds": {
            "substitution": by_kind["substitution"],
            "insertion": by_kind["insertion"],
            "deletion": by_kind["deletion"],
        },
        "adjacent_duplicates": duplicates["total"],
    }


def tie_break_spread(per_clip: list[tuple[list[Emission], list[int]]]) -> dict:
    """Both summaries plus the gap between them, so an artefact cannot pass as a finding."""
    reports = {tb: summarise(per_clip, tb) for tb in TIE_BREAKS}

    def spread(read) -> float:
        values = [read(reports[tb]) for tb in TIE_BREAKS]
        return round(max(values) - min(values), 6)

    lift = {
        tb: (
            reports[tb]["region"]["flush"]["error_rate"]
            / reports[tb]["region"]["committed"]["error_rate"]
            if reports[tb]["region"]["committed"]["error_rate"]
            else 0.0
        )
        for tb in TIE_BREAKS
    }
    return {
        "by_tie_break": reports,
        "flush_over_committed_error_ratio": {k: round(v, 4) for k, v in lift.items()},
        # The headline figures, worst case over the optimal-alignment set. Compare each
        # against the effect an arm would have to show: a region gap smaller than its own
        # tie-break spread is not a finding about the model.
        "max_abs_difference": {
            "flush_error_rate": spread(lambda r: r["region"]["flush"]["error_rate"]),
            "committed_error_rate": spread(
                lambda r: r["region"]["committed"]["error_rate"]
            ),
            "seam_error_rate": spread(lambda r: r["seam"]["seam"]["error_rate"]),
            "flush_over_committed_ratio": round(
                max(lift.values()) - min(lift.values()), 4
            ),
        },
    }


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import SeamlessM4TFeatureExtractor

    from training.decode_evalset import CLIPS_DIRNAME, check_provenance, load_manifest
    from training.distill_eval import (
        SAMPLE_RATE,
        confirmed_emissions,
        confirmed_stream,
        load_student_from_checkpoint,
    )
    from training.distill_gate import tokens_to_phonemes
    from training.distill_student import TEACHER_MODEL_ID
    from training.distill_train import load_teacher

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test", "both"), default="dev")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, default=0, help="score only the first N clips (smoke runs)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evalset = load_manifest(args.eval_set)
    check_provenance(evalset, TEACHER_MODEL_ID)
    # Same rule as distill_eval: the manifest's batch size, because the teacher's decode
    # depends on it and a re-decode at another batch would fail the cache assertion below
    # for a reason that has nothing to do with the student.
    batch_size = args.batch_size or evalset.provenance.get("batch_size", 16)

    teacher = load_teacher(device)
    student, _, step = load_student_from_checkpoint(
        args.checkpoint, device, use_ema=args.ema
    )
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(TEACHER_MODEL_ID)
    clips_dir = Path(args.eval_set) / CLIPS_DIRNAME

    clips = evalset.subset(args.split)
    if args.limit:
        clips = clips[: args.limit]
    print(f"[setup] {len(clips)} {args.split} clips, batch {batch_size}, step {step}")

    per_clip: list[tuple[list[Emission], list[int]]] = []
    for index, clip in enumerate(clips, start=1):
        samples, rate = sf.read(str(clips_dir / clip.filename), dtype="float32")
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{clip.filename} is {rate} Hz, not {SAMPLE_RATE}")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        emissions = confirmed_emissions(teacher, extractor, samples, device, batch_size)
        # The guard: provenance is only meaningful if this decode is the decode the frozen
        # set was built from. The teacher is bit-identical at a fixed batch size, so this is
        # an equality, not a tolerance.
        redecoded = tokens_to_phonemes([e.token_id for e in emissions])
        if redecoded != clip.teacher_text:
            raise SystemExit(
                f"teacher re-decode differs from the cached decode on {clip.filename}.\n"
                f"  cached:    {clip.teacher_text!r}\n"
                f"  re-decode: {redecoded!r}\n"
                "The provenance would describe a different decode than the metric scores. "
                "Check --batch-size against the manifest and PROTOCOL_VERSION."
            )
        per_clip.append(
            (
                emissions,
                confirmed_stream(student, extractor, samples, device, batch_size),
            )
        )
        if index % 100 == 0:
            print(f"  {index}/{len(clips)} clips", flush=True)

    report = tie_break_spread(per_clip)
    report["provenance"] = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "ema": args.ema,
        "eval_set": str(args.eval_set),
        "evalset_fingerprint": evalset.fingerprint(),
        "split": args.split,
        "batch_size": batch_size,
        "num_clips": len(clips),
        "confirm_timesteps": CONFIRM_TIMESTEPS,
    }
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    headline = report["by_tie_break"]["diagonal_first"]
    print(f"\n[edits] CER {headline['character_error_rate']:.4%} over "
          f"{headline['reference_characters']:,} characters")
    for name in ("committed", "flush"):
        block = headline["region"][name]
        print(
            f"  {name:<10} {block['share_of_reference']:6.1%} of characters, "
            f"{block['share_of_edits']:6.1%} of edits, "
            f"error rate {block['error_rate']:.4%} "
            f"(sub {block['substitutions']}, ins {block['insertions']}, "
            f"del {block['deletions']}, dup {block['adjacent_duplicates']})"
        )
    ratio = report["flush_over_committed_error_ratio"]
    print(f"  flush/committed error ratio: {ratio}")
    for name in ("seam", "interior"):
        block = headline["seam"][name]
        print(
            f"  {name:<10} {block['reference_characters']:,} characters, "
            f"error rate {block['error_rate']:.4%}"
        )
    print(f"  tie-break spread: {report['max_abs_difference']}")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
