"""Stage a hub release in the layout `sysofwan/hifzguide-muaalem-mini` publishes, from a checkpoint.

The published repository is a `transformers` checkpoint, not a bare state dict: `config.json`
carries the `auto_map` that makes `trust_remote_code=True` work, the vendored
`configuration_`/`modeling_multi_level_ctc.py` sit beside it, and `model.safetensors` holds the
weights under the same names the module assigns them. A reader then needs nothing from this
repository: `AutoModel.from_pretrained(...)` and `AutoFeatureExtractor.from_pretrained(...)`.

The weight set is a scored decision (each round's ``choice.json`` records ``weights: ema``), so
``--ema`` goes through the loader every eval tool uses (``training.decoding``) instead of
re-reading ``state["student"]``. That loader raises when a checkpoint has no average, so an
``--ema`` staging can never silently publish the live weights under the averaged name.

The stage is deterministic -- the same checkpoint reproduces the same file contents -- and it
refuses a non-empty output directory rather than merging into an older stage. `LICENSE`,
`NOTICE.md`, `README.md` and `.gitattributes` are authored files, not derived from the
checkpoint, and are placed by hand (they are what the release note records).

Usage::

    python3 stage_hf_model.py --checkpoint runs/consonant3_candidate/checkpoint.pt --ema \\
        --output-dir ../hf-stage/hifzguide-muaalem-mini-v2

Writes:

    model.safetensors                    the chosen weight set, fp32 (save_pretrained)
    config.json                          geometries, phoneme labels, auto_map
    configuration_multi_level_ctc.py     vendored verbatim from ``tadabur/muaalem``
    modeling_multi_level_ctc.py          the same
    phoneme_vocab.json                   id -> character, blank/pad named
    preprocessor_config.json             the teacher's feature-extractor settings

The directory is a **draft**. Publishing it, and the repository name, are the owner's call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

STATE_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
VOCAB_FILE = "phoneme_vocab.json"
PROCESSOR_FILE = "preprocessor_config.json"
CONFIGURATION_FILE = "configuration_multi_level_ctc.py"
MODELING_FILE = "modeling_multi_level_ctc.py"

# What `config.auto_map` must say for `trust_remote_code=True` to find the vendored classes in
# this directory. The published repository pins exactly these two entries.
AUTO_MAP = {
    "AutoConfig": f"{CONFIGURATION_FILE[:-3]}.Wav2Vec2BertForMultilevelCTCConfig",
    "AutoModel": f"{MODELING_FILE[:-3]}.Wav2Vec2BertForMultilevelCTC",
}


def sha256_file(path: Path) -> str:
    """A staged file's identity, for the release note's checksum table."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_phoneme_vocab() -> tuple[tuple[str, ...], int]:
    """The 43-class vocabulary and its CTC blank id, from the canonical table."""
    from tadabur.phoneme_vocab import (
        NUM_PHONEME_CLASSES,
        PHONEME_ID_TO_CHAR,
        PHONEME_PAD_ID,
    )

    if len(PHONEME_ID_TO_CHAR) != NUM_PHONEME_CLASSES:
        raise SystemExit("phoneme vocabulary and class count disagree")
    return PHONEME_ID_TO_CHAR, PHONEME_PAD_ID


def configure_for_hub(model, characters: tuple[str, ...]) -> None:
    """Make the checkpoint's config loadable outside this repository.

    ``architectures`` and ``auto_map`` are what ``trust_remote_code`` reads; ``id2label`` and
    ``label2id`` are the phoneme vocabulary the vendored head emits, in the same form the
    published repository carries them. Without the labels the config would ship
    ``LABEL_0``/``LABEL_1``, i.e. a decoder that maps every id to nothing.
    """
    config = model.config
    config.architectures = ["Wav2Vec2BertForMultilevelCTC"]
    config.auto_map = dict(AUTO_MAP)
    config.id2label = {index: char for index, char in enumerate(characters)}
    config.label2id = {char: index for index, char in enumerate(characters)}


def stage(checkpoint: Path, use_ema: bool, output_dir: Path, teacher_revision: str | None) -> dict:
    """Write the staging directory and return ``{filename: sha256}``."""
    from transformers import SeamlessM4TFeatureExtractor

    from tadabur import muaalem
    from training.decoding import load_student_from_checkpoint
    from training.distill_student import TEACHER_MODEL_ID

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"{output_dir} is not empty: a stage is written once, from one checkpoint, so a "
            f"merge into an older stage would mix two weight sets. Point --output-dir at a "
            f"new directory."
        )

    characters, pad_id = load_phoneme_vocab()
    model, run_config, step = load_student_from_checkpoint(checkpoint, "cpu", use_ema=use_ema)
    weights = "ema" if use_ema else "live"
    print(f"Loaded {run_config['preset']} from {checkpoint} @ step {step} ({weights})")
    configure_for_hub(model, characters)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    vendored = Path(muaalem.__file__).parent
    for name in (CONFIGURATION_FILE, MODELING_FILE):
        shutil.copyfile(vendored / name, output_dir / name)

    (output_dir / VOCAB_FILE).write_text(
        json.dumps(
            {
                "id_to_char": list(characters),
                "pad_id": pad_id,
                "pad_token": characters[pad_id],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        TEACHER_MODEL_ID, revision=teacher_revision
    )
    extractor.save_pretrained(output_dir)

    names = [
        STATE_FILE,
        CONFIG_FILE,
        CONFIGURATION_FILE,
        MODELING_FILE,
        VOCAB_FILE,
        PROCESSOR_FILE,
    ]
    return {name: sha256_file(output_dir / name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage a student checkpoint as a Hub repository in the published layout"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ema",
        action="store_true",
        help="stage the averaged weights a --ema-decay run stored beside the live ones",
    )
    parser.add_argument(
        "--teacher-revision",
        help="pin the teacher revision the feature-extractor settings are read from "
        "(the checkpoint's run config records the one training used)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checksums = stage(args.checkpoint, args.ema, args.output_dir, args.teacher_revision)
    if args.json:
        print(json.dumps({"output_dir": str(args.output_dir), "sha256": checksums}, indent=2))
        return

    print(f"\nstaged {args.output_dir}")
    for name, digest in checksums.items():
        print(f"  {name:<30} {digest}")


if __name__ == "__main__":
    main()
