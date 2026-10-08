"""What does exporting and palettizing the student cost the metric we actually target?

``ml-model-transformation.md`` and ADR-0010 answer this with **frame argmax agreement** --
98.91% at 6-bit, 99.85% at 8-bit, measured on 11 windows. That number cannot be converted
into the one that matters. A flipped frame may leave the decoded string untouched (it is
inside a run that collapses to the same token) or it may cost several edits (it splits a run
in two). The two quantities are not on the same scale and subtracting one from a character
accuracy is meaningless.

So this replays the **deployed protocol through the CoreML model itself** and compares the
resulting phoneme string to the teacher's cached decode, which is what
``training.decode_evalset`` froze. It reports two different things and both are needed:

* **agreement with the teacher** -- the distillation metric, before and after export. This is
  what tells you whether the export costs anything you care about.
* **drift from the PyTorch student** -- how many characters moved at all. Drift is an upper
  bound on the damage, and the gap between the two is informative: measured here, 6-bit moved
  0.75% of characters while costing only 0.10 points against the teacher, because the moves
  are roughly orthogonal to the teacher rather than away from it.

Measured on a mid-run ``h384`` (300 held-out dev clips, 24,390 teacher phonemes):

    pytorch fp32   90.94%                  drift 0.00%
    coreml fp16    90.87%  (-0.07)         drift 0.32%
    coreml 8-bit   90.91%  (-0.03)         drift 0.31%
    coreml 6-bit   90.84%  (-0.10)         drift 0.75%

**Palettization is not a constraint on the training target.** 6-bit costs a tenth of a point,
so a student that reaches X in PyTorch ships at about X. The frame-level table reads far more
alarming than the string-level truth, and had it been trusted the 20.5 MB for 8-bit would have
been bought for nothing. Note also that the conversion itself (fp16) accounts for most of the
drift that 8-bit shows, so 8-bit adds essentially no error of its own.

Two things this does **not** establish: the numbers come from a mid-training checkpoint (a
converged model is expected to be *more* robust, so read them as an upper bound), and it runs
on a Mac's ANE via ``CPU_AND_NE``, not an iPhone 13's. Single-chunk acceptance on device is
still enforced at ``MLModel`` load and still unproven.

Usage::

    python export_student_coreml.py --checkpoint runs/h384/checkpoint.pt \\
        --output-dir student_coreml --nbits 6
    python verify_student_export.py --checkpoint runs/h384/checkpoint.pt \\
        --eval-set ../tadabur/gate_eval --export student_coreml --num-clips 300

macOS only (coremltools). The evaluation set and checkpoint come from the Linux box.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def window_features(extractor, samples, frames: int = 250):
    """The deployed protocol's windows, feature-extracted one at a time.

    Per-window extraction, not whole-clip-then-slice: the extractor normalises over whatever
    it is given and the device normalises over the 5 s window, so slicing afterwards trains
    and measures on a distribution the device never produces.
    """
    from training.decoding import window_audio
    from training.distill_data import SAMPLE_RATE

    out = []
    for chunk in window_audio(samples):
        extracted = extractor(
            chunk, sampling_rate=SAMPLE_RATE, return_tensors="np", padding=False
        )
        features = extracted.input_features[0]
        if features.shape[0] < frames:
            features = np.pad(features, ((0, frames - features.shape[0]), (0, 0)))
        out.append(features[:frames].astype("float32"))
    return out


def decode_windows(logits_for_window, windows) -> str:
    """Stream the windows through the deployed protocol and map to phonemes."""
    from training.decoding import stream_emissions, tokens_to_phonemes
    from training.distill_student import DEPLOYED_LOGIT_FRAMES

    rows = [
        logits_for_window(features)[:DEPLOYED_LOGIT_FRAMES].argmax(-1) for features in windows
    ]
    return tokens_to_phonemes(e.token_id for e in stream_emissions(rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode agreement of the exported student, in string space"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument(
        "--export",
        type=Path,
        required=True,
        help="directory from export_student_coreml.py (its .mlpackage files are scored)",
    )
    parser.add_argument("--num-clips", type=int, default=300)
    parser.add_argument("--split", default="dev", choices=("dev", "test", "both"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    import coremltools as ct
    import torch
    from transformers import SeamlessM4TFeatureExtractor

    from training.decode_evalset import (
        CLIPS_DIRNAME,
        check_provenance,
        load_manifest,
        read_clip_audio,
    )
    from training.distill_data import SAMPLE_RATE
    from training.distill_eval import levenshtein, score_decode_agreement
    from training.distill_student import PRESETS, TEACHER_MODEL_ID, build_student

    evalset = load_manifest(args.eval_set)
    # This is the one tool that runs on a different machine, against assets copied across a
    # boundary, so it is the one that most needs the guard every other consumer applies. A
    # manifest built under a different teacher or a pre-flush protocol would otherwise print
    # a plausible absolute number next to figures from the Linux path.
    check_provenance(evalset, TEACHER_MODEL_ID)
    clips = evalset.subset(args.split)[: args.num_clips]
    if not clips:
        raise SystemExit(f"no clips in split {args.split!r}")
    clips_dir = Path(args.eval_set) / CLIPS_DIRNAME
    extractor = SeamlessM4TFeatureExtractor.from_pretrained("obadx/muaalem-model-v3_2")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    student = build_student(PRESETS[state["config"]["preset"]])
    student.load_state_dict(state["student"])
    student.eval()

    def torch_logits(features):
        with torch.no_grad():
            out = student(torch.from_numpy(features)[None, ...], return_dict=True)
        return out["logits"]["phonemes"][0].numpy()

    scorers = {"pytorch fp32": torch_logits}
    for package in sorted(Path(args.export).glob("*.mlpackage")):
        model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
        label = f"coreml {package.stem.rsplit('_', 1)[-1].lower()}"
        if label in scorers:
            raise SystemExit(
                f"two exports in {args.export} reduce to the label {label!r} "
                f"({package.name}); one would silently replace the other."
            )

        def coreml_logits(features, model=model):
            return model.predict({"input_features": features[None, ...]})["phoneme_logits"][0]

        scorers[label] = coreml_logits

    decodes: dict[str, dict[str, str]] = {name: {} for name in scorers}
    started = time.time()
    for index, clip in enumerate(clips, start=1):
        samples = read_clip_audio(clips_dir, clip.filename)
        windows = window_features(extractor, samples)
        for name, logits_for_window in scorers.items():
            decodes[name][clip.filename] = decode_windows(logits_for_window, windows)
        if index % 50 == 0:
            print(f"  {index}/{len(clips)}  {time.time() - started:.0f}s", flush=True)

    total = sum(len(clip.teacher_text) for clip in clips)
    reference = None
    rows = []
    for name in scorers:
        report = score_decode_agreement(
            [
                (
                    levenshtein(list(c.teacher_text), list(decodes[name][c.filename])),
                    len(c.teacher_text),
                    c.reciter_id,
                )
                for c in clips
            ]
        )
        drift = sum(
            levenshtein(list(decodes["pytorch fp32"][c.filename]), list(decodes[name][c.filename]))
            for c in clips
        )
        if reference is None:
            reference = report.char_accuracy
        rows.append(
            {
                "model": name,
                "char_accuracy": round(report.char_accuracy, 5),
                "delta_vs_pytorch": round(report.char_accuracy - reference, 5),
                "exact_match": round(report.exact_match, 4),
                "drift_from_pytorch": round(drift / max(1, total), 5),
            }
        )

    if args.json:
        print(json.dumps({"num_clips": len(clips), "phonemes": total, "rows": rows}, indent=2))
        return

    print(f"\n{len(clips)} {args.split} clips, {total:,} teacher phonemes\n")
    for row in rows:
        print(
            f"  {row['model']:<16} vs teacher {row['char_accuracy']:>7.2%}  "
            f"({row['delta_vs_pytorch']:+.2%})   exact {row['exact_match']:>5.1%}   "
            f"drift from fp32 {row['drift_from_pytorch']:>6.2%}"
        )
    print(
        "\ndrift is an upper bound on the damage: characters that moved, whether toward the "
        "teacher or away.\nThe gap between drift and delta is how much of the movement was "
        "orthogonal rather than harmful."
    )


if __name__ == "__main__":
    main()
