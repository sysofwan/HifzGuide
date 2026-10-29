"""The frame manifest: what a streamed run's input is, so another run can replay it exactly.

Same seed is not enough. A run's seed orders its shards
(:func:`training.consumed_prefix.shard_permutation`) and a *different* derived seed drives the
window shuffle (:func:`training.distill_stream.window_shuffle_seed`), while augmentation is
keyed by the source row, not by the stream (:func:`training.augmentation.draw`). Round 2's
candidate takes its Tadabur substream from this arm's frame, so the launcher writes the frame's
ordered shard list and its digest, both seeds and the recipe record into the run directory,
beside the checkpoint and the trainer's ``run_config.json``; the run note copies it.

Usage::

    python -m training.frame_manifest --shards "$(cat logs/spec.txt)" \\
        --shard-order-seed 2026 --exposure-use consonant2.control.training \\
        --run-dir runs/consonant2_control
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from tadabur.shard_reader import parse_shard_spec
from training.augmentation import recipe_record
from training.consumed_prefix import shard_permutation
from training.distill_stream import window_shuffle_seed

#: Written into the run directory, beside the checkpoint and ``run_config.json``.
FILENAME = "frame_manifest.json"


def shards_digest(shards: Iterable[int]) -> str:
    """SHA-256 of the shard list as given: one decimal number per line.

    Order-sensitive, so it can pin either the frame (the sorted set the run draws from,
    ``shards_sha256``) or the order it actually streams (``stream_order_sha256``).
    """
    text = "".join(f"{shard}\n" for shard in shards)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frame_manifest(
    shards: Iterable[int], *, shard_order_seed: int, exposure_use: str
) -> dict:
    """The manifest of a run streaming ``shards`` with ``shard_order_seed``.

    ``shards_sha256`` is the frame's membership digest (over the sorted list, so it is the
    same for any run drawing from the same set); ``stream_order_sha256`` is over the order
    the run actually streams, which is what a replay has to reproduce.
    """
    frame = sorted(shards)
    order = shard_permutation(frame, shard_order_seed)
    return {
        "shards": frame,
        "shards_sha256": shards_digest(frame),
        "stream_order": order,
        "stream_order_sha256": shards_digest(order),
        "shard_order_seed": shard_order_seed,
        "window_shuffle_seed": window_shuffle_seed(shard_order_seed),
        "augmentation": recipe_record(),
        "exposure_use": exposure_use,
    }


def write_frame_manifest(run_dir: Path, manifest: dict) -> Path:
    """Write ``manifest`` into ``run_dir`` as one sorted-key JSON line."""
    path = Path(run_dir) / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", required=True,
                        help="the run's --stream-shards spec, as the stream reads it")
    parser.add_argument("--shard-order-seed", type=int, required=True,
                        help="the run seed: it orders the shards, and seed + 1 the windows")
    parser.add_argument("--exposure-use", required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help=f"the run's output directory; {FILENAME} goes in it")
    args = parser.parse_args()
    manifest = frame_manifest(
        parse_shard_spec(args.shards),
        shard_order_seed=args.shard_order_seed,
        exposure_use=args.exposure_use,
    )
    path = write_frame_manifest(args.run_dir, manifest)
    print(f"frame manifest: {len(manifest['shards'])} shards, shards_sha256 "
          f"{manifest['shards_sha256']}, stream_order_sha256 "
          f"{manifest['stream_order_sha256']}, stream-order seed "
          f"{manifest['shard_order_seed']}, window-shuffle seed "
          f"{manifest['window_shuffle_seed']}, recipe "
          f"{manifest['augmentation']['version']} -> {path}")


if __name__ == "__main__":
    main()
