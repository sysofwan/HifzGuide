"""Does swapping the teacher for the student change what Muraja *decides*?

``training.distill_eval`` measures whether the student reproduces the teacher's phoneme
**string**. That is the right target for distillation, but it is not the product question,
and it is a harsher bar than the product actually imposes. Muraja never compares strings:
it runs Smith-Waterman with soft pairs against the ayah reference to locate the reciter and
produce a ``match_ratio``, then gates on it (``tadabur.scorer``, ported verbatim from
Muraja's ``.balanced`` ``ScoringParameters``). A student whose decode differs from the
teacher's in ways the aligner absorbs -- a boundary shifted by a frame, a soft-pair
substitution -- changes the string while changing no decision at all.

So this module scores both decodes through the **real gate** and asks how often they
disagree about the outcome. That is the number a ship decision should turn on. An 84%
character agreement that preserves 99% of gate decisions is a different proposition from
one that flips them.

Both models are decoded through the **deployed** protocol
(``training.distill_eval.confirmed_stream``: 5 s window, 1 s hop, ``midpoint < 25``
confirmation), not the whole-clip pass ``tadabur.filter`` uses, because the confirmed
stream is what the device actually produces and therefore what the scorer actually sees.

One indexing trap, inherited from ``tadabur.filter.canonical_surah_ayah``: Tadabur numbers
surahs **0-indexed** in the filename (``S77`` is Al-Naba, the 78th), while ayah is already
1-indexed. Get it wrong and every clip gates against the wrong reference and nothing passes.

**Prefer ``--eval-set`` over ``--audio-root``.** A random clip directory cannot answer the
question: 88% of ``clips_v2`` passes the gate, so a student that rubber-stamps everything
scores 88% and any real number is a few points of daylight on top of that.
``training.gate_evalset`` builds a frozen set with a population view and a boundary view, and
caches the teacher's decode per clip -- so this tool runs **only the student**, at roughly
twice the speed, against decisions that cannot drift between two checkpoints being compared.

What that path reports, and why raw agreement is not the headline:

* **flip rate with a Wilson interval**, split by direction. A false *rejection* (teacher
  passed, student failed) is what a reciter feels; a false *acceptance* is what silently
  degrades the product. The same aggregate agreement can be either.
* **the population-weighted estimate**, reconstructed from the scan counts the manifest
  carries, so the boundary view's enriched errors are diluted back to what a user would meet.
* **``match_ratio`` error** -- RMSE, p95 and mean offset. This is continuous, uses every clip,
  and moves while the flip rate is still quantised; rank checkpoints on it, ship on flips.
* **paired McNemar against another checkpoint's saved decisions** (``--save-decisions`` /
  ``--compare-decisions``). Against pass-everything, any competent student wins once the
  boundary view is balanced, so that test is printed as a guard, not as evidence.

Usage::

    # The frozen set (preferred)
    python -m training.distill_gate --checkpoint runs/h384/checkpoint.pt \\
        --eval-set ../tadabur/gate_eval --save-decisions runs/h384/gate_dev.json

    # ... and the next checkpoint, compared against it on the same clips
    python -m training.distill_gate --checkpoint runs/h384_v2/checkpoint.pt \\
        --eval-set ../tadabur/gate_eval --compare-decisions runs/h384/gate_dev.json

    # Continuity panel on the staged corpus (decodes both models)
    python -m training.distill_gate --checkpoint runs/h384/checkpoint.pt \\
        --audio-root ../tadabur/audit_run/clips_v2 --num-clips 100

Linux + CUDA.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from training.distill_data import SAMPLE_RATE, discover_clips, split_clips
from training.distill_eval import (
    check_split_matches_checkpoint,
    confirmed_stream,
    load_student_from_checkpoint,
)

# ``tadabur_spk0039_S5_A31_60e0c708_000042.wav`` -> surah index 5 (0-based), ayah 31.
CLIP_NAME_PATTERN = re.compile(r"_S(\d+)_A(\d+)_")


def parse_surah_ayah(filename: str) -> str | None:
    """Canonical ``"surah:ayah"`` for a staged clip filename, or None if unparseable.

    Applies the same +1 surah shift as ``tadabur.filter.canonical_surah_ayah``: the
    filename carries Tadabur's 0-indexed surah array position, the reference cache is keyed
    by the canonical 1-indexed number.
    """
    match = CLIP_NAME_PATTERN.search(filename)
    if match is None:
        return None
    surah_index, ayah = int(match.group(1)), int(match.group(2))
    return f"{surah_index + 1}:{ayah}"


def tokens_to_phonemes(token_ids: list[int]) -> str:
    """Map already-collapsed CTC token ids to their phoneme characters.

    ``PHONEME_ID_TO_CHAR`` is a **tuple indexed by class id**, not a mapping. Testing
    ``id in PHONEME_ID_TO_CHAR`` therefore asks whether the *integer* is one of the
    characters, which is never true -- it silently yields an empty string, every gate
    scores 0.0, and the result reads as "the model produces nothing" rather than "the
    lookup is wrong". Guard the range explicitly instead.
    """
    from tadabur.phoneme_vocab import PHONEME_ID_TO_CHAR, PHONEME_PAD_ID

    return "".join(
        PHONEME_ID_TO_CHAR[t]
        for t in token_ids
        if t != PHONEME_PAD_ID and 0 <= t < len(PHONEME_ID_TO_CHAR)
    )


@dataclass
class GateAgreement:
    """How often teacher and student lead the scorer to the same conclusion."""

    num_clips: int
    same_decision: int
    both_passed: int
    both_failed: int
    teacher_only_passed: int
    student_only_passed: int
    mean_abs_ratio_delta: float
    mean_teacher_ratio: float
    mean_student_ratio: float
    ratio_correlation: float = 0.0
    best_threshold: float = 0.0
    best_threshold_agreement: float = 0.0
    # Agreement from the trivial "pass everything" policy, i.e. the teacher's own pass
    # rate. A recalibrated threshold only means something if it beats this.
    always_pass_agreement: float = 0.0

    @property
    def decision_agreement(self) -> float:
        return self.same_decision / max(1, self.num_clips)

    def as_dict(self) -> dict:
        return {
            "num_clips": self.num_clips,
            "decision_agreement": round(self.decision_agreement, 4),
            "ratio_correlation": round(self.ratio_correlation, 4),
            "best_threshold": round(self.best_threshold, 3),
            "best_threshold_agreement": round(self.best_threshold_agreement, 4),
            "always_pass_agreement": round(self.always_pass_agreement, 4),
            "recalibration_beats_trivial": bool(
                self.best_threshold_agreement > self.always_pass_agreement + 0.01
            ),
            "both_passed": self.both_passed,
            "both_failed": self.both_failed,
            "teacher_only_passed": self.teacher_only_passed,
            "student_only_passed": self.student_only_passed,
            "mean_abs_ratio_delta": round(self.mean_abs_ratio_delta, 4),
            "mean_teacher_ratio": round(self.mean_teacher_ratio, 4),
            "mean_student_ratio": round(self.mean_student_ratio, 4),
        }


def recalibrated_agreement(
    pairs: list[tuple[bool, float, bool, float]], threshold: float
) -> float:
    """Decision agreement if the student were gated at ``threshold`` instead of the default.

    The teacher keeps the shipped ``.balanced`` bar; only the student's threshold moves.
    This asks whether the student is simply *offset* from the teacher -- in which case a
    recalibrated bar restores the decisions -- or genuinely noisier, in which case no
    single threshold helps and the disagreement is irreducible.

    **This models only the match_ratio condition.** The real gate
    (``tadabur.scorer.Scorer.gate``) also rejects on ``max_insertion_run`` and on
    ``added_shadda``, and ``pairs`` does not carry either, so a clip the real gate fails at
    a high ratio is counted here as passing at any threshold below it. An over-emitting
    student -- long interior insertion runs -- would therefore be scored optimistically, and
    the sweep can report a "recalibration win" that the shipped gate would not deliver.
    Treat the result as an upper bound on what moving the bar could buy, never as a
    replacement for measuring the real gate at that bar.
    """
    same = sum(
        1
        for teacher_passed, _, _, student_ratio in pairs
        if teacher_passed == (student_ratio >= threshold)
    )
    return same / max(1, len(pairs))


def compare_gates(pairs: list[tuple[bool, float, bool, float]]) -> GateAgreement:
    """Aggregate per-clip ``(teacher_passed, teacher_ratio, student_passed, student_ratio)``."""
    same = both_pass = both_fail = teacher_only = student_only = 0
    deltas: list[float] = []
    teacher_ratios: list[float] = []
    student_ratios: list[float] = []

    for teacher_passed, teacher_ratio, student_passed, student_ratio in pairs:
        if teacher_passed == student_passed:
            same += 1
            both_pass += int(teacher_passed)
            both_fail += int(not teacher_passed)
        elif teacher_passed:
            teacher_only += 1
        else:
            student_only += 1
        deltas.append(abs(teacher_ratio - student_ratio))
        teacher_ratios.append(teacher_ratio)
        student_ratios.append(student_ratio)

    count = max(1, len(pairs))

    # Is the student a shifted copy of the teacher, or a noisier one? A high correlation
    # with a large mean offset is recoverable by moving the student's threshold; a low
    # correlation is not, whatever the threshold.
    correlation = 0.0
    if len(pairs) > 1:
        import statistics

        try:
            correlation = statistics.correlation(teacher_ratios, student_ratios)
        except statistics.StatisticsError:
            correlation = 0.0

    # The trivial baseline the search must beat: gate nothing, pass everything. It scores
    # exactly the teacher's pass rate, so a "best threshold" near zero is not a
    # recalibration win -- it is the search rediscovering pass-everything.
    always_pass = sum(1 for teacher_passed, _, _, _ in pairs if teacher_passed) / count

    best_threshold, best_agreement = 0.0, 0.0
    for step in range(0, 101):
        threshold = step / 100
        agreement = recalibrated_agreement(pairs, threshold)
        if agreement > best_agreement:
            best_threshold, best_agreement = threshold, agreement

    return GateAgreement(
        num_clips=len(pairs),
        same_decision=same,
        both_passed=both_pass,
        both_failed=both_fail,
        teacher_only_passed=teacher_only,
        student_only_passed=student_only,
        mean_abs_ratio_delta=sum(deltas) / count,
        mean_teacher_ratio=sum(teacher_ratios) / count,
        mean_student_ratio=sum(student_ratios) / count,
        ratio_correlation=correlation,
        best_threshold=best_threshold,
        best_threshold_agreement=best_agreement,
        always_pass_agreement=always_pass,
    )


# --- Scoring a student against a frozen evaluation set (the preferred path) ---


@dataclass(frozen=True)
class StudentDecision:
    """What one student made of one frozen clip, alongside the cached teacher decision."""

    filename: str
    student_text: str
    student_ratio: float
    student_passed: bool

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "student_text": self.student_text,
            "student_ratio": round(self.student_ratio, 6),
            "student_passed": self.student_passed,
        }


def score_student_on_evalset(
    student, extractor, evalset, clips_dir: Path, device, batch_size: int
) -> dict[str, StudentDecision]:
    """Decode and gate every clip in a frozen set with the student alone.

    The teacher is not loaded: its decode is in the manifest, cached by
    ``training.gate_evalset``. Besides halving the work, this makes two checkpoints
    comparable by construction -- they are scored against the *same* teacher decisions, not
    against two re-runs of a model that is only deterministic if nothing about the box
    changed.
    """
    import soundfile as sf

    from tadabur.reference_phonemes import load_reference_phonemes
    from tadabur.scorer import BALANCED_SCORER
    from training.distill_eval import confirmed_stream

    references = load_reference_phonemes()
    decisions: dict[str, StudentDecision] = {}

    for index, clip in enumerate(evalset.clips, start=1):
        reference = references.get(clip.surah_ayah)
        if reference is None:
            raise SystemExit(
                f"{clip.filename} references {clip.surah_ayah}, which is not in the "
                f"reference cache. The manifest and the cache disagree; rebuild one."
            )
        samples, rate = sf.read(str(clips_dir / clip.filename), dtype="float32")
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{clip.filename} is {rate} Hz, not {SAMPLE_RATE}")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        text = tokens_to_phonemes(
            confirmed_stream(student, extractor, samples, device, batch_size)
        )
        result = BALANCED_SCORER.gate(text, reference)
        decisions[clip.filename] = StudentDecision(
            filename=clip.filename,
            student_text=text,
            student_ratio=result.match_ratio,
            student_passed=result.passed,
        )
        if index % 100 == 0:
            print(f"  {index}/{len(evalset.clips)} clips", flush=True)

    return decisions


@dataclass(frozen=True)
class ViewReport:
    """One view of the frozen set (population or boundary), scored for one student.

    ``agreement`` is the fraction of clips on which the student reaches the teacher's
    decision; ``1 - agreement`` is the flip rate, which is the number to quote, because a
    flip is the event with a product consequence. The interval is Wilson, because at 95% the
    normal approximation runs off the end of the scale.
    """

    view: str
    split: str
    num_clips: int
    same_decision: int
    ci_low: float
    ci_high: float
    always_pass_agreement: float
    errors: object
    ratio_rmse: float
    ratio_p95_abs_delta: float
    ratio_offset: float
    per_stratum_agreement: dict[str, float]
    reweighted_agreement: float
    trivial_guard: object

    @property
    def agreement(self) -> float:
        return self.same_decision / max(1, self.num_clips)

    @property
    def flip_rate(self) -> float:
        return 1.0 - self.agreement

    def as_dict(self) -> dict:
        return {
            "view": self.view,
            "split": self.split,
            "num_clips": self.num_clips,
            "agreement": round(self.agreement, 4),
            "flip_rate": round(self.flip_rate, 4),
            "agreement_ci95": [round(self.ci_low, 4), round(self.ci_high, 4)],
            "always_pass_agreement": round(self.always_pass_agreement, 4),
            "population_reweighted_agreement": round(self.reweighted_agreement, 4),
            "ratio_rmse": round(self.ratio_rmse, 4),
            "ratio_p95_abs_delta": round(self.ratio_p95_abs_delta, 4),
            "ratio_offset": round(self.ratio_offset, 4),
            "per_stratum_agreement": {
                name: round(value, 4) for name, value in self.per_stratum_agreement.items()
            },
            **self.errors.as_dict(),
            "guard_vs_always_pass": self.trivial_guard.as_dict(),
        }


def build_view_report(
    view: str,
    split: str,
    clips,
    decisions: dict[str, StudentDecision],
    scanned_by_stratum: dict[str, int],
) -> ViewReport:
    """Aggregate one view's clips into the report a ship decision can be read off."""
    import math

    from training.gate_evalset import (
        STRATA,
        directional_errors,
        paired_comparison,
        reweighted_agreement,
        wilson_interval,
    )

    pairs = [(clip.teacher_passed, decisions[clip.filename].student_passed) for clip in clips]
    deltas = [
        decisions[clip.filename].student_ratio - clip.teacher_ratio for clip in clips
    ]
    same = sum(1 for teacher, student in pairs if teacher == student)
    count = max(1, len(clips))
    absolute = sorted(abs(delta) for delta in deltas)

    per_stratum: dict[str, float] = {}
    for name in STRATA:
        members = [clip for clip in clips if clip.stratum == name]
        if members:
            per_stratum[name] = sum(
                1
                for clip in members
                if clip.teacher_passed == decisions[clip.filename].student_passed
            ) / len(members)

    low, high = wilson_interval(same, len(clips))
    return ViewReport(
        view=view,
        split=split,
        num_clips=len(clips),
        same_decision=same,
        ci_low=low,
        ci_high=high,
        always_pass_agreement=sum(1 for teacher, _ in pairs if teacher) / count,
        errors=directional_errors(pairs),
        ratio_rmse=math.sqrt(sum(delta**2 for delta in deltas) / count),
        ratio_p95_abs_delta=(
            absolute[min(len(absolute) - 1, int(0.95 * len(absolute)))] if absolute else 0.0
        ),
        ratio_offset=sum(deltas) / count,
        per_stratum_agreement=per_stratum,
        reweighted_agreement=reweighted_agreement(per_stratum, scanned_by_stratum),
        # A guard, not evidence: on a view balanced around the bar, any competent student
        # beats pass-everything. It stays printed because a student that does NOT beat it is
        # a rubber stamp, and that has happened.
        trivial_guard=paired_comparison(
            [teacher == student for teacher, student in pairs],
            [teacher for teacher, _ in pairs],
            "student",
            "always_pass",
        ),
    )


def format_view_report(report: ViewReport) -> str:
    """The human rendering. Flip rate leads; agreement follows it."""
    errors = report.errors
    lines = [
        f"  [{report.view}/{report.split}] {report.num_clips} clips",
        f"    FLIP RATE           {report.flip_rate:.2%}  "
        f"(agreement {report.agreement:.2%}, 95% CI "
        f"[{report.ci_low:.2%}, {report.ci_high:.2%}])",
        f"    population estimate {1 - report.reweighted_agreement:.2%} flips "
        f"-- the same student on an unstratified sample",
        f"    false rejections    {errors.false_fails}/{errors.teacher_passes} "
        f"({errors.false_fail_rate:.2%}) -- teacher passed, student failed",
        f"    false acceptances   {errors.false_passes}/{errors.teacher_fails} "
        f"({errors.false_pass_rate:.2%}) -- teacher failed, student passed",
        f"    ratio error         rmse {report.ratio_rmse:.4f}, "
        f"p95 {report.ratio_p95_abs_delta:.4f}, offset {report.ratio_offset:+.4f}",
        f"    pass-everything     {report.always_pass_agreement:.2%} agreement "
        f"-- the floor this must clear",
    ]
    if report.per_stratum_agreement:
        by_stratum = "  ".join(
            f"{name} {value:.1%}" for name, value in report.per_stratum_agreement.items()
        )
        lines.append(f"    by stratum          {by_stratum}")
    if not report.trivial_guard.as_dict()["a_beats_b"]:
        lines.append(
            "    => NOT significantly better than pass-everything on this view."
        )
    return "\n".join(lines)


def run_evalset(args, device) -> None:
    """Score one checkpoint against a frozen evaluation set and report every view."""
    from transformers import SeamlessM4TFeatureExtractor

    from training.gate_evalset import (
        CLIPS_DIRNAME,
        check_provenance,
        load_manifest,
        paired_comparison,
    )
    from training.distill_eval import PROTOCOL_VERSION
    from training.distill_student import TEACHER_MODEL_ID
    from tadabur.scorer import BALANCED

    evalset = load_manifest(args.eval_set)
    check_provenance(evalset, TEACHER_MODEL_ID, BALANCED.correct_threshold)
    manifest_protocol = evalset.provenance.get("protocol_version")
    if manifest_protocol != PROTOCOL_VERSION:
        raise SystemExit(
            f"the evaluation set was built under protocol {manifest_protocol!r} and this "
            f"code decodes under {PROTOCOL_VERSION!r}. The cached teacher decodes would be "
            f"compared against a student decoded differently. Rebuild the set."
        )

    student, state_config, step = load_student_from_checkpoint(
        args.checkpoint, device, use_ema=args.ema
    )
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(TEACHER_MODEL_ID)
    decisions = score_student_on_evalset(
        student, extractor, evalset, Path(args.eval_set) / CLIPS_DIRNAME, device,
        args.batch_size,
    )

    reports = []
    for view in ("population", "boundary"):
        for split in (("dev", "test") if args.split == "both" else (args.split,)):
            clips = evalset.subset(view, split)
            if clips:
                reports.append(
                    build_view_report(view, split, clips, decisions, evalset.scanned_by_stratum)
                )

    comparisons = []
    if args.compare_decisions:
        previous = json.loads(Path(args.compare_decisions).read_text(encoding="utf-8"))
        other = previous["decisions"]
        for report in reports:
            clips = [
                clip
                for clip in evalset.subset(report.view, report.split)
                if clip.filename in other
            ]
            if not clips:
                continue
            comparisons.append(
                {
                    "view": report.view,
                    "split": report.split,
                    "against": previous.get("checkpoint", args.compare_decisions),
                    "num_clips": len(clips),
                    **paired_comparison(
                        [
                            clip.teacher_passed == decisions[clip.filename].student_passed
                            for clip in clips
                        ],
                        [
                            clip.teacher_passed == other[clip.filename]["student_passed"]
                            for clip in clips
                        ],
                        "this",
                        "other",
                    ).as_dict(),
                }
            )

    payload = {
        "checkpoint": str(args.checkpoint),
        "preset": state_config["preset"],
        "step": step,
        "weights": "ema" if args.ema else "live",
        "protocol_version": PROTOCOL_VERSION,
        "eval_set": str(args.eval_set),
        "views": [report.as_dict() for report in reports],
        "comparisons": comparisons,
    }

    if args.save_decisions:
        Path(args.save_decisions).write_text(
            json.dumps(
                {**payload, "decisions": {
                    name: decision.as_dict() for name, decision in decisions.items()
                }},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"\nGate decisions -- {payload['preset']} @ step {step} vs cached teacher")
    for report in reports:
        print(format_view_report(report))
    for comparison in comparisons:
        print(
            f"  [{comparison['view']}/{comparison['split']}] paired vs "
            f"{comparison['against']}: this-only-correct "
            f"{comparison['this_only_correct']}, other-only-correct "
            f"{comparison['other_only_correct']}, p {comparison['p_value']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate-decision agreement between a distilled student and the teacher"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--eval-set",
        type=Path,
        help="a frozen set built by training.gate_evalset. Preferred: it carries the "
        "teacher's cached decisions, a population view and a boundary view, so the number "
        "is sensitive AND the teacher is not re-run per checkpoint.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="continuity panel: decode BOTH models over a clip directory. On clips_v2 the "
        "trivial pass-everything policy already scores 88%%, so read this number next to "
        "always_pass_agreement, never alone.",
    )
    parser.add_argument("--num-clips", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument(
        "--split",
        default="dev",
        help="with --eval-set: dev, test or both (default dev -- keep test for finalists, "
        "or tuning turns it into another dev set). With --audio-root: val or train.",
    )
    parser.add_argument(
        "--save-decisions",
        type=Path,
        help="write this student's per-clip decisions, for a later --compare-decisions",
    )
    parser.add_argument(
        "--compare-decisions",
        type=Path,
        help="another run's saved decisions; adds a paired McNemar on the shared clips, "
        "which is the test that should decide whether an intervention helped",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="windows per forward. Lower it (4 or less) to run alongside a training job; "
        "both models must be resident and a training run may leave under 2 GB free.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="score the averaged weights a --ema-decay run stored beside the live ones",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")

    if bool(args.eval_set) == bool(args.audio_root):
        raise SystemExit("pass exactly one of --eval-set (preferred) or --audio-root")
    if args.eval_set:
        if args.split not in ("dev", "test", "both"):
            raise SystemExit("--split must be dev, test or both when using --eval-set")
        run_evalset(args, device)
        return
    if args.split not in ("val", "train"):
        raise SystemExit("--split must be val or train when using --audio-root")

    import soundfile as sf
    from transformers import SeamlessM4TFeatureExtractor

    from tadabur.reference_phonemes import load_reference_phonemes
    from tadabur.scorer import BALANCED_SCORER
    from training.distill_train import load_teacher

    student, state_config, step = load_student_from_checkpoint(
        args.checkpoint, device, use_ema=args.ema
    )
    preset = state_config["preset"]
    check_split_matches_checkpoint(state_config, args.val_fraction)
    teacher = load_teacher(device)
    extractor = SeamlessM4TFeatureExtractor.from_pretrained("obadx/muaalem-model-v3_2")
    references = load_reference_phonemes()

    train_clips, val_clips = split_clips(discover_clips(args.audio_root), args.val_fraction)
    clips = (train_clips if args.split == "train" else val_clips)[: args.num_clips]

    pairs: list[tuple[bool, float, bool, float]] = []
    skipped = 0
    for index, path in enumerate(clips, start=1):
        key = parse_surah_ayah(path.name)
        reference = references.get(key) if key else None
        if reference is None:
            skipped += 1
            continue
        try:
            samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            skipped += 1
            continue
        if rate != SAMPLE_RATE:
            skipped += 1
            continue
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        teacher_text = tokens_to_phonemes(
            confirmed_stream(teacher, extractor, samples, device, args.batch_size)
        )
        student_text = tokens_to_phonemes(
            confirmed_stream(student, extractor, samples, device, args.batch_size)
        )
        teacher_gate = BALANCED_SCORER.gate(teacher_text, reference)
        student_gate = BALANCED_SCORER.gate(student_text, reference)
        pairs.append(
            (
                teacher_gate.passed,
                teacher_gate.match_ratio,
                student_gate.passed,
                student_gate.match_ratio,
            )
        )
        if not args.json and index % 25 == 0:
            print(f"  {index}/{len(clips)} clips", flush=True)

    report = compare_gates(pairs)
    payload = {"preset": preset, "step": step, "skipped": skipped, **report.as_dict()}

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"\nGate-decision agreement -- {preset} @ step {step} [{args.split}]")
    print(f"  clips scored          {report.num_clips} ({skipped} skipped)")
    print(f"  SAME decision         {report.decision_agreement:.1%}")
    print(f"    both passed         {report.both_passed}")
    print(f"    both failed         {report.both_failed}")
    print(f"  teacher passed only   {report.teacher_only_passed}")
    print(f"  student passed only   {report.student_only_passed}")
    print(f"  mean |ratio delta|    {report.mean_abs_ratio_delta:.4f}")
    print(
        f"  mean match_ratio      teacher {report.mean_teacher_ratio:.4f}, "
        f"student {report.mean_student_ratio:.4f}"
    )
    print(f"  ratio correlation     {report.ratio_correlation:.4f}")
    from tadabur.scorer import BALANCED

    print(
        f"  best student bar      {report.best_threshold:.2f} -> "
        f"{report.best_threshold_agreement:.1%} agreement (ratio condition only; "
        f"shipped bar is {BALANCED.correct_threshold:.2f} -> "
        f"{report.decision_agreement:.1%} on the FULL gate)"
    )
    print(
        f"  pass-everything       {report.always_pass_agreement:.1%} "
        f"-- the trivial baseline any recalibration must beat"
    )
    if report.best_threshold_agreement <= report.always_pass_agreement + 0.01:
        print(
            "  => recalibration is NOT a fix: the best bar merely rediscovers "
            "pass-everything, so the student is noisier than the teacher, not offset "
            f"from it (ratio correlation {report.ratio_correlation:.2f})."
        )


if __name__ == "__main__":
    main()
