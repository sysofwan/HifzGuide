"""Teacher agreement of hub-format students, and the protocol every scorer shares.

A hub card's headline ("94.690% against the teacher") is reproducible only if the windowing,
the 125-frame slice, the streamed CTC collapse and the reciter-clustered interval are the same
in the tool as in the card. This module owns that protocol: ``verify_student_export`` scores a
CoreML export through these helpers, and the CLI below scores published repositories, so the
two numbers sit on one scale.

The protocol, in order: per-window feature extraction (never whole-clip-then-slice -- the
extractor normalises over what it is given, and the device normalises over the 5 s window),
the first :data:`DEPLOYED_LOGIT_FRAMES` logit rows per window, the windows streamed as one
sequence, and greedy CTC over the concatenated rows.

Usage::

    python -m training.hub_agreement --eval-set tadabur/gate_eval --num-clips 300 \\
        --model mini=/path/to/hifzguide-muaalem-mini \\
        --model v2=/path/to/hifzguide-muaalem-mini-v2

Prints one row per model: teacher agreement with its reciter-clustered 95% CI, exact-match
rate, and -- against the model named first -- the fraction of characters that differ. A hub
directory is loaded the way its card documents: ``AutoModel`` and ``AutoFeatureExtractor``
with ``trust_remote_code=True``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.decode_evalset import (
    CLIPS_DIRNAME,
    check_provenance,
    load_manifest,
    read_clip_audio,
)
from training.decoding import stream_emissions, tokens_to_phonemes
from training.distill_eval import levenshtein, score_decode_agreement
from training.distill_student import DEPLOYED_LOGIT_FRAMES

DEFAULT_EXTRACTOR = "obadx/muaalem-model-v3_2"


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
    rows = [
        logits_for_window(features)[:DEPLOYED_LOGIT_FRAMES].argmax(-1) for features in windows
    ]
    return tokens_to_phonemes(e.token_id for e in stream_emissions(rows))


@dataclass(frozen=True)
class Row:
    """One model's line in the comparison."""

    model: str
    char_accuracy: float
    ci_low: float
    ci_high: float
    exact_match: float
    drift_from_reference: float

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "char_accuracy": round(self.char_accuracy, 5),
            "ci95_clustered": [round(self.ci_low, 5), round(self.ci_high, 5)],
            "exact_match": round(self.exact_match, 4),
            "drift_from_reference": round(self.drift_from_reference, 5),
        }


def agreement_rows(clips, decodes: dict[str, dict[str, str]]) -> list[Row]:
    """Score each model's decodes against the clips' teacher text.

    ``drift_from_reference`` is measured against the first model's decodes: characters that
    moved at all, whether toward the teacher or away, so it reads as an upper bound on the
    difference between two publications rather than a verdict.
    """
    total = sum(len(clip.teacher_text) for clip in clips)
    reference = next(iter(decodes))
    rows = []
    for model, texts in decodes.items():
        report = score_decode_agreement(
            [
                (
                    levenshtein(list(clip.teacher_text), list(texts[clip.filename])),
                    len(clip.teacher_text),
                    clip.reciter_id,
                )
                for clip in clips
            ]
        )
        drift = sum(
            levenshtein(
                list(decodes[reference][clip.filename]), list(texts[clip.filename])
            )
            for clip in clips
        )
        rows.append(
            Row(
                model=model,
                char_accuracy=report.char_accuracy,
                ci_low=report.ci_low,
                ci_high=report.ci_high,
                exact_match=report.exact_match,
                drift_from_reference=drift / max(1, total),
            )
        )
    return rows


def load_hub_scorer(directory: Path):
    """A hub directory's logits function, loaded as its card documents."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(directory), trust_remote_code=True).eval()

    def logits_for_window(features):
        import torch

        with torch.no_grad():
            out = model(torch.from_numpy(features)[None, ...], return_dict=True)
        return out.logits["phonemes"][0].numpy()

    return logits_for_window


def score(
    eval_set: Path,
    models: dict[str, Path],
    split: str = "dev",
    num_clips: int = 300,
    extractor=None,
    scorer=load_hub_scorer,
) -> list[Row]:
    """Decode every model on the same clips and return the comparison table."""
    from transformers import SeamlessM4TFeatureExtractor

    evalset = load_manifest(eval_set)
    # The cached teacher text is the truth this table measures against; a manifest built by a
    # different teacher, protocol or confirmation split would produce a plausible number
    # against the wrong reference. `verify_student_export` refuses exactly here.
    check_provenance(evalset, DEFAULT_EXTRACTOR)
    clips = evalset.subset(split)[:num_clips]
    if not clips:
        raise SystemExit(f"no clips in split {split!r}")
    clips_dir = Path(eval_set) / CLIPS_DIRNAME
    extractor = extractor or SeamlessM4TFeatureExtractor.from_pretrained(DEFAULT_EXTRACTOR)

    decodes: dict[str, dict[str, str]] = {}
    for label, directory in models.items():
        logits_for_window = scorer(directory)
        texts = {}
        for clip in clips:
            samples = read_clip_audio(clips_dir, clip.filename)
            windows = window_features(extractor, samples)
            texts[clip.filename] = decode_windows(logits_for_window, windows)
        decodes[label] = texts
    return agreement_rows(clips, decodes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--split", default="dev", choices=("dev", "test", "both"))
    parser.add_argument("--num-clips", type=int, default=300)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="a hub model directory and the label to print it under; repeatable, and the "
        "first is the drift reference",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    models = {}
    for item in args.model:
        label, _, directory = item.partition("=")
        if not directory:
            raise SystemExit(f"--model wants LABEL=DIR, got {item!r}")
        models[label] = Path(directory)

    rows = score(args.eval_set, models, args.split, args.num_clips)
    if args.json:
        import json

        print(json.dumps([row.as_dict() for row in rows], indent=2))
        return

    reference = rows[0].model
    for row in rows:
        print(
            f"{row.model:<12} vs teacher {row.char_accuracy:>7.3%}  "
            f"95% CI [{row.ci_low:.3%}, {row.ci_high:.3%}]  exact {row.exact_match:>5.2%}  "
            f"drift vs {reference} {row.drift_from_reference:>6.3%}"
        )


if __name__ == "__main__":
    main()
