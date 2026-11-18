"""Stage a hub release: the student's weights and the configs a reader needs to load them.

The GitHub release ships compiled CoreML; a Hugging Face model repo ships the PyTorch
weights and the settings they were trained against, so a reader can inspect, re-export or
continue from them. The two must carry the *same* weight set: which one a release carries is
a scored decision (each round's ``choice.json`` records ``weights: ema``), so ``--ema`` goes
through the loader every eval tool uses (``training.decoding``) instead of re-reading
``state["student"]``. That loader raises when a checkpoint has no average, so an ``--ema``
staging can never silently ship the live weights under the averaged name.

The staging is deterministic -- a rebuild from the same checkpoint reproduces the same
bytes -- and it refuses a non-empty output directory rather than merging into an old stage.

Usage::

    python3 stage_hf_model.py --checkpoint runs/consonant3_candidate/checkpoint.pt --ema \\
        --processor-config preprocessor_config.json \\
        --output-dir ../hf-stage/hifzguide-muaalem-student-h448

What it writes:

    student_state_dict.pt     the chosen weight set, fp32 on CPU
    config.json               preset geometry, deployed shapes, vocabulary size, provenance
    phoneme_vocab.json        the 43-class CTC vocabulary, id -> character
    preprocessor_config.json  the teacher's SeamlessM4TFeatureExtractor settings, verbatim
    sha256sums.txt            every file above, as ``sha256  name``

The directory is a **draft**. Publishing it, and the repo it goes to, is the owner's call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

STATE_DICT_NAME = "student_state_dict.pt"
CONFIG_NAME = "config.json"
VOCAB_NAME = "phoneme_vocab.json"
PROCESSOR_NAME = "preprocessor_config.json"
CHECKSUMS_NAME = "sha256sums.txt"


def sha256_file(path: Path) -> str:
    """A staged file's identity, recorded in ``sha256sums.txt`` and the release note."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Indented UTF-8 JSON with a trailing newline, so a diff of two stages is readable."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def staged_config(model, preset: str, run_config: dict, step: int, checkpoint: Path,
                  use_ema: bool) -> dict:
    """The load-side description of the staged weights.

    Everything here is either needed to rebuild the module (``preset`` and its geometry,
    the feature and logit shapes) or provenance a reader cannot recover from the tensor
    file alone (which weight set, which checkpoint, which teacher).
    """
    from training.distill_student import (
        DEPLOYED_FEATURE_FRAMES,
        DEPLOYED_LOGIT_FRAMES,
        FEATURE_INPUT_DIM,
        NUM_PHONEME_CLASSES,
        PRESETS,
        TEACHER_MODEL_ID,
        count_parameters,
    )

    spec = PRESETS[preset]
    return {
        "architectures": ["MuaalemStudent"],
        "model_type": "muaalem-student",
        "preset": preset,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_attention_heads": spec.num_attention_heads,
        "num_hidden_layers": spec.num_hidden_layers,
        "position_embeddings_type": spec.position_embeddings_type,
        "num_parameters": count_parameters(model),
        "phoneme_classes": NUM_PHONEME_CLASSES,
        "feature_extractor": "SeamlessM4TFeatureExtractor",
        "feature_input": {
            "name": "input_features",
            "shape": [DEPLOYED_FEATURE_FRAMES, FEATURE_INPUT_DIM],
            "sampling_rate": 16000,
            "mel_bins": 80,
            "stride": 2,
        },
        "logit_frames": DEPLOYED_LOGIT_FRAMES,
        "weights": "ema" if use_ema else "live",
        "step": step,
        "checkpoint_sha256": sha256_file(checkpoint),
        "ema_decay": run_config.get("ema_decay"),
        "teacher": {
            "model_id": TEACHER_MODEL_ID,
            "revision": run_config.get("teacher_revision"),
        },
        "augmentation": (run_config.get("augment") or {}).get("version"),
    }


def stage(checkpoint: Path, use_ema: bool, processor_config: Path, output_dir: Path) -> dict:
    """Write the staging directory and return its checksum table."""
    import torch

    from training.decoding import load_student_from_checkpoint
    from tadabur.phoneme_vocab import NUM_PHONEME_CLASSES, PHONEME_ID_TO_CHAR, PHONEME_PAD_ID

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"{output_dir} is not empty: a stage is written once, from one checkpoint, so a "
            f"merge into an older stage would mix two weight sets. Point --output-dir at a "
            f"new directory."
        )
    if not processor_config.is_file():
        raise SystemExit(f"{processor_config}: no such file (the teacher's preprocessor config)")
    if len(PHONEME_ID_TO_CHAR) != NUM_PHONEME_CLASSES:
        raise SystemExit("phoneme vocabulary and class count disagree")

    model, run_config, step = load_student_from_checkpoint(checkpoint, "cpu", use_ema=use_ema)
    weights = "ema" if use_ema else "live"
    print(f"Loaded {run_config['preset']} from {checkpoint} @ step {step} ({weights})")

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / STATE_DICT_NAME)
    write_json(
        output_dir / CONFIG_NAME,
        staged_config(model, run_config["preset"], run_config, step, checkpoint, use_ema),
    )
    write_json(
        output_dir / VOCAB_NAME,
        {
            "pad_id": PHONEME_PAD_ID,
            "classes": NUM_PHONEME_CLASSES,
            "id_to_char": list(PHONEME_ID_TO_CHAR),
        },
    )
    shutil.copyfile(processor_config, output_dir / PROCESSOR_NAME)

    names = [STATE_DICT_NAME, CONFIG_NAME, VOCAB_NAME, PROCESSOR_NAME]
    checksums = {name: sha256_file(output_dir / name) for name in names}
    (output_dir / CHECKSUMS_NAME).write_text(
        "".join(f"{checksums[name]}  {name}\n" for name in names), encoding="utf-8"
    )
    return checksums


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage a student checkpoint as a Hugging Face model directory"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ema",
        action="store_true",
        help="stage the averaged weights a --ema-decay run stored beside the live ones",
    )
    parser.add_argument(
        "--processor-config",
        type=Path,
        required=True,
        help="the teacher's preprocessor_config.json (SeamlessM4TFeatureExtractor settings)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checksums = stage(args.checkpoint, args.ema, args.processor_config, args.output_dir)
    if args.json:
        print(json.dumps({"output_dir": str(args.output_dir), "sha256": checksums}, indent=2))
        return

    print(f"\nstaged {args.output_dir}")
    for name, digest in checksums.items():
        print(f"  {name:<26} {digest}")
    print(f"\nchecksum table: {args.output_dir / CHECKSUMS_NAME}")


if __name__ == "__main__":
    main()
