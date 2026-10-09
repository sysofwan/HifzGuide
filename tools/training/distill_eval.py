"""Student-vs-teacher agreement on the stream Muraja actually shows the user.

Frame-level agreement (``training.distill_loss.agreement_stats``) is cheap and smooth, which
makes it a good training signal, but it is not the release gate. It averages over all 125
timesteps of every window, and 100 of those never reach the user. A student could look
excellent there and still produce a visibly different transcript.

This module measures the thing that decides: the **confirmed phoneme stream** produced by
replaying the deployed sliding-window protocol, for both models, and compares them. The
protocol itself -- windowing, ``scanCTC``, the commit rule, the silence flush -- lives in
:mod:`training.decoding`, which every tool that replays it shares; this module scores what
it commits at the deployed block.

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

import torch

from training.decode_evalset import levenshtein  # noqa: F401  (its callers import it from here)
from training.decoding import (
    PROTOCOL_VERSION,
    Decoder,
    load_student_from_checkpoint,
)
from training.distill_data import (
    SAMPLE_RATE,
    discover_clips,
    split_clips,
)
from tadabur.panel_seal import refuse_sealed
from training.distill_loss import BLANK_ID, breakout_stats

# Default for the older --audio-root path. Named so the --eval-set path can tell "the user
# passed --num-clips" from "the user did not", and refuse the former.
DEFAULT_AUDIO_ROOT_CLIPS = 200


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
    import statistics

    from training.decode_evalset import cluster_bootstrap, wilson_interval

    if not per_clip:
        # 1 - 0/0 has no answer and 1 - 0/1 is 1.0, which would print as perfect agreement
        # produced by scoring nothing. Report zero so a failure looks like a failure.
        return DecodeAgreement(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    edits = sum(e for e, _, _ in per_clip)
    phonemes = sum(t for _, t, _ in per_clip)
    rates = sorted(e / max(1, t) for e, t, _ in per_clip)
    reciters = {r for _, _, r in per_clip}

    if len(reciters) < 2:
        low, high = wilson_interval(phonemes - edits, max(1, phonemes))
    else:
        low, high = cluster_bootstrap(
            per_clip,
            cluster_of=lambda row: row[2],
            statistic=lambda rows: 1
            - sum(e for e, _, _ in rows) / max(1, sum(t for _, t, _ in rows)),
            seed=7,
        )

    return DecodeAgreement(
        num_clips=len(per_clip),
        num_reciters=len(reciters),
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

    The obvious comparison -- count clips where this checkpoint is closer -- is a **sign
    test**. It asks whether more clips improved than worsened, which is not the claim anyone
    makes from it: the headline is a pooled edit-rate difference, and a sign test neither
    weights by how much a clip moved nor accounts for one reciter's clips not being
    independent. Both errors push the p-value the same way, toward significance.
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
    from training.decode_evalset import cluster_bootstrap

    if not rows:
        return PairedDelta(0.0, 0.0, 0.0, 0, 0)

    def pooled(sample):
        this = sum(row[0] for row in sample)
        other = sum(row[1] for row in sample)
        tokens = max(1, sum(row[2] for row in sample))
        return (other - this) / tokens  # positive = this checkpoint is closer

    low, high = cluster_bootstrap(
        rows,
        cluster_of=lambda row: row[3],
        statistic=pooled,
        seed=seed,
        iterations=iterations,
    )
    return PairedDelta(
        delta=pooled(rows),
        ci_low=low,
        ci_high=high,
        clips_closer=sum(1 for row in rows if row[0] < row[1]),
        clips_further=sum(1 for row in rows if row[0] > row[1]),
    )


def run_evalset(args, device) -> None:
    """Score one checkpoint's decode against a frozen set's cached teacher decode."""
    from transformers import SeamlessM4TFeatureExtractor

    from training.decode_evalset import (
        CLIPS_DIRNAME,
        check_provenance,
        load_manifest,
        read_clip_audio,
        scoring_batch_size,
    )
    from training.distill_student import TEACHER_MODEL_ID

    evalset = load_manifest(args.eval_set)
    check_provenance(evalset, TEACHER_MODEL_ID)
    batch_size = scoring_batch_size(evalset, args.batch_size)
    student, state_config, step = load_student_from_checkpoint(
        args.checkpoint, device, use_ema=args.ema
    )
    decoder = Decoder(
        args.checkpoint,
        student,
        SeamlessM4TFeatureExtractor.from_pretrained(TEACHER_MODEL_ID),
        device,
        batch_size,
    )
    clips_dir = Path(args.eval_set) / CLIPS_DIRNAME

    decodes: dict[str, str] = {}
    for index, clip in enumerate(evalset.clips, start=1):
        samples = read_clip_audio(clips_dir, clip.filename)
        decodes[clip.filename] = decoder.decode_stream(samples)
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
        missing = [c.filename for c in scored_clips if c.filename not in other]
        if missing:
            raise SystemExit(
                f"the comparison file is missing {len(missing)} of {len(scored_clips)} "
                f"clips in this split (first: {missing[0]}). Comparing the intersection "
                f"would report a paired delta over an unannounced subset."
            )
        comparison = paired_reciter_bootstrap(
            [
                (
                    levenshtein(list(c.teacher_text), list(decodes[c.filename])),
                    levenshtein(list(c.teacher_text), list(other[c.filename])),
                    len(c.teacher_text),
                    c.reciter_id,
                )
                for c in scored_clips
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
        "--num-clips",
        type=int,
        default=DEFAULT_AUDIO_ROOT_CLIPS,
        help="held-out clips to evaluate (--audio-root path only)",
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
        # Accepting a flag and ignoring it is the one behaviour that cannot be right: the
        # protocol and the clip set are pinned by the manifest, and --num-clips/--breakout
        # belong to the --audio-root path. Refuse rather than silently do something else.
        ignored = [
            name
            for name, used in (
                ("--no-flush-tail", not args.flush_tail),
                ("--num-clips", args.num_clips != DEFAULT_AUDIO_ROOT_CLIPS),
                ("--breakout", args.breakout),
            )
            if used
        ]
        if ignored:
            raise SystemExit(
                f"{', '.join(ignored)} has no meaning with --eval-set: the evaluation set "
                f"pins its own protocol and clip list. Drop the flag, or use --audio-root."
            )
        run_evalset(args, device)
        return
    if not args.audio_root:
        raise SystemExit("pass --eval-set (preferred) or --audio-root")
    if args.split == "dev":
        args.split = "val"
    if args.split not in ("val", "train"):
        raise SystemExit("--split must be val or train when using --audio-root")
    # Listed (and checked against the sealed panel) before any model loads.
    listing = discover_clips(args.audio_root)

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

    from training.distill_student import TEACHER_MODEL_ID
    from training.distill_train import load_teacher

    student, state_config, step = load_student_from_checkpoint(
        args.checkpoint, device, use_ema=args.ema
    )
    preset = state_config["preset"]
    check_split_matches_checkpoint(state_config, args.val_fraction)
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(TEACHER_MODEL_ID)
    batch_size = args.batch_size or 16
    teacher_decoder = Decoder(TEACHER_MODEL_ID, load_teacher(device), extractor, device, batch_size)
    student_decoder = Decoder(args.checkpoint, student, extractor, device, batch_size)

    # Default is the *validation* side -- the same hash split training used, so no clip the
    # student was fit on can inflate the number. --split train scores seen clips instead,
    # which is only useful as the paired comparison described in the flag's help.
    train_clips, val_clips = split_clips(listing, args.val_fraction)
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
            samples, rate = sf.read(str(refuse_sealed(path)), dtype="float32", always_2d=False)
        except Exception:
            continue
        if rate != SAMPLE_RATE:
            continue
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        teacher_stream, student_stream = (
            [e.token_id for e in decoder.emissions(samples, flush_tail=args.flush_tail)]
            for decoder in (teacher_decoder, student_decoder)
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
