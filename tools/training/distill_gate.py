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

Usage::

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate-decision agreement between a distilled student and the teacher"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--num-clips", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="windows per forward. Lower it (4 or less) to run alongside a training job; "
        "both models must be resident and a training run may leave under 2 GB free.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")

    import soundfile as sf
    from transformers import SeamlessM4TFeatureExtractor

    from tadabur.reference_phonemes import load_reference_phonemes
    from tadabur.scorer import BALANCED_SCORER
    from training.distill_train import load_teacher

    student, state_config, step = load_student_from_checkpoint(args.checkpoint, device)
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
