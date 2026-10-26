"""Tarteel v1's look / confirm split, by clip hash (#160, pre-registered in #128 on 2026-10-15).

The augmentation experiment of the #77 re-plan reads Tarteel v1 app-user audio
(:data:`tadabur.nonpro_sources.TARTEEL_V1`) twice: the **look** half may be read while the
recipe is chosen, and the **confirm** half is read once, for the final candidate (#128's
"Discipline"). The halves are fixed before any candidate is decoded, by the clip alone:

    look  when  int.from_bytes(sha256("tarteel-v1-split-2026-10:" + clip_id)[:8], "big") < 2**63
    confirm otherwise

``clip_id`` is the clip's own id in the dataset, its ``audio.path`` (``100_10_4115049010.wav``;
``audio_filename`` in #156's decodes), never the decode's item key. The rule depends on nothing
else, so every reader (the owner's listening, #160; the experiment's readout) splits alike.

``tadabur/nonpro_eval/tarteel_v1/split.json`` freezes the rule applied to #156's decoded
clips: the salt, the cutoff, the count per half and each half's clip ids.

Usage (from ``tools/``; torch-free, a second)::

  python -m tadabur.tarteel_split
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

SALT = "tarteel-v1-split-2026-10"
CUTOFF = 2**63
LOOK, CONFIRM = "look", "confirm"
TARTEEL_DIR = Path(__file__).parent / "nonpro_eval" / "tarteel_v1"
SPLIT_PATH = TARTEEL_DIR / "split.json"
#: The base teacher's whole-clip decodes of the corpus: the one definition of this path.
DECODES_PATH = TARTEEL_DIR / "decodes.base.jsonl.gz"


def clip_hash(clip_id: str) -> int:
    """The 64-bit value the split compares to :data:`CUTOFF`: the first 8 bytes, big-endian,
    of SHA-256 over ``"<salt>:<clip id>"``."""
    if not clip_id or "#" in clip_id or "/" in clip_id:
        raise ValueError(f"a clip id is the dataset's audio path, got {clip_id!r}")
    return int.from_bytes(hashlib.sha256(f"{SALT}:{clip_id}".encode("utf-8")).digest()[:8], "big")


def half(clip_id: str) -> str:
    """:data:`LOOK` or :data:`CONFIRM` for one Tarteel v1 clip (module docstring)."""
    return LOOK if clip_hash(clip_id) < CUTOFF else CONFIRM


def split_record(clip_ids: Iterable[str]) -> dict:
    """The rule and its result over ``clip_ids``: counts and each half's sorted ids."""
    ids = sorted(set(clip_ids))
    halves = {h: [c for c in ids if half(c) == h] for h in (LOOK, CONFIRM)}
    return {
        "issue": 160,
        "rule": f'{LOOK} when the first 8 bytes (big-endian) of sha256("{SALT}:" + clip_id) '
                f"are below 2**63, else {CONFIRM}; clip_id is the dataset's audio path",
        "salt": SALT,
        "cutoff": CUTOFF,
        "clips": len(ids),
        "counts": {h: len(v) for h, v in halves.items()},
        **halves,
    }


def decoded_clips() -> list[str]:
    """Every Tarteel v1 clip #156 decoded (its committed base decodes)."""
    from training.external_corpus import read_decode_file
    from training.nonpro_eval import decodes_file

    from .nonpro_sources import TARTEEL_V1

    return [line["audio_filename"] for line in read_decode_file(
        decodes_file(TARTEEL_V1, "base")).values()]


def load_split(path: Path = SPLIT_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    from training.tashkeel_eval import write_text_atomically

    record = split_record(decoded_clips())
    write_text_atomically(SPLIT_PATH, json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {SPLIT_PATH}: {record['counts']}")


if __name__ == "__main__":
    main()
