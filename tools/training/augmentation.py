"""Training-time acoustic augmentation for #94: the student hears a perturbed clip, the teacher
the clean one.

``h448`` was distilled with every augmentation off (:mod:`training.distill_student`), and
#159's D2 found it drifting further than the teacher under noise (consonant excess drift +2.9
points at 10 dB SNR, +6.1 at 5 dB; ``docs/nonpro-gap.md``). #94's augmentation arm continues
``h448`` with the **student** reading augmented audio and the **teacher** the clean clip, under
the existing frame-weighted KL; its control arm is the same run with augmentation off.

**One condition per clip, drawn before windowing.** :class:`Augmenter` perturbs a whole clip,
and :class:`training.distill_stream.StreamingWindowDataset` cuts the clean and the augmented
clip at the same window starts, so window ``k`` of each covers the same samples. Every
condition keeps the clip's length and timing within
:data:`training.perturbation.MAX_RESIDUAL_LAG` samples (under 1% of a 20 ms frame), so a
teacher frame on the clean window pairs with the student frame on the augmented one.

**The mixture** (:data:`MIXTURE`; no stacking, no gain jitter):

=================  =====  =================================================================
condition          share  parameters
=================  =====  =================================================================
clean              0.30   --
noise              0.35   SNR ~ U(3, 20) dB, one pool recording at a uniform offset, looped
reverb             0.15   RT60 ~ U(0.2, 1.0) s, DRR 3 dB (:func:`perturbation.synthetic_rir`)
8 kHz band limit   0.10   down to 8 kHz and back (:func:`perturbation.band_limit`)
Opus               0.10   voip, narrowband, 8 / 12 / 16 kbit/s, each its own calibrated lag
=================  =====  =================================================================

**Draws are a pure function of the run's seed and the clip's Tadabur row** (:func:`draw`):
each clip gets its own ``random.Random`` seeded by ``seed + 2`` and its ``(shard, row)``, so
which condition a clip gets does not depend on the shuffle (``seed + 1``), the worker count or
the order clips arrive in.

**The noise pool is disjoint from D2's.** It is built from two shards of the same Freesound
collection that D2 did not read (:data:`TRAINING_NOISE`), and any recording whose
``freesound_id`` is in D2's pool (``tadabur/nonpro_gap/noise_pool.json``) is dropped
(:func:`build_training_pool`). Its provenance is committed (:data:`POOL_PATH`); its audio stays
on the GPU box's scratch, checked against that record when loaded (:func:`load_training_pool`).

Usage (from ``tools/``; the codec needs ``ffmpeg`` with ``libopus`` on the path)::

  python -m training.augmentation noise-pool --cache-dir <scratch> --out <scratch>/pool.npz
  python -m training.augmentation record     # the recipe a run's config carries

Torch-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tadabur.audio import TARGET_SAMPLE_RATE
from tadabur.nonpro_sources import Dataset
from training.perturbation import (
    BAND_LIMIT,
    CLEAN,
    CODEC,
    DRR_DB,
    NARROWBAND_RATE,
    NOISE,
    NOISE_KIND,
    OPUS_CONFIGS,
    REVERB,
    NoiseRecording,
    OpusConfig,
    add_noise,
    band_limit,
    build_pool,
    codec_lag,
    load_pool,
    noise_segment,
    opus_round_trip,
    reverberate,
    save_pool,
    synthetic_rir,
)

RECIPE_VERSION = "aug94-mixture-v1"

#: Each condition kind and its share of clips; the shares sum to one.
MIXTURE = ((CLEAN, 0.30), (NOISE_KIND, 0.35), (REVERB, 0.15), (BAND_LIMIT, 0.10), (CODEC, 0.10))
SNR_RANGE_DB = (3.0, 20.0)
RT60_RANGE_S = (0.2, 1.0)
#: The run seed's offset for the augmentation draws: ``seed`` orders the shards and
#: ``seed + 1`` drives the shuffle buffer (:mod:`training.distill_stream`).
DRAW_SEED_OFFSET = 2

#: Two shards of D2's noise collection, at its pinned revision, that D2 never read (it read
#: 00000 and 00052), far apart in the collection's order.
TRAINING_NOISE = Dataset(
    "freesound_noise_training", NOISE.repo, NOISE.revision,
    ("data/train-00026-of-00104.parquet", "data/train-00078-of-00104.parquet"),
    NOISE.licence, False)

AUGMENTATION_DIR = Path(__file__).parent.parent / "tadabur" / "augmentation"
#: The training pool's provenance: every recording kept, and what was left out and why.
POOL_PATH = AUGMENTATION_DIR / "noise_pool.json"
#: D2's pool, which the training pool must not share a recording with.
D2_POOL_PATH = Path(__file__).parent.parent / "tadabur" / "nonpro_gap" / "noise_pool.json"

assert abs(sum(share for _, share in MIXTURE) - 1.0) < 1e-12


@dataclass(frozen=True)
class Draw:
    """What one clip gets: its condition kind and level (SNR in dB, RT60 in s, bitrate in
    bit/s; none for clean and the band limit), and for noise the pool recording and its start
    (a fraction of its length), for reverb the impulse response's seed."""

    kind: str
    level: float | None = None
    noise_index: int | None = None
    noise_offset: float | None = None
    rir_seed: int | None = None


def draw(seed: int, shard: int, row: int, pool_size: int) -> Draw:
    """The draw for Tadabur row ``row`` of ``shard`` in a run seeded ``seed``."""
    rng = random.Random(f"{seed + DRAW_SEED_OFFSET}:{shard}:{row}")
    kinds, shares = zip(*MIXTURE)
    kind = rng.choices(kinds, weights=shares)[0]
    if kind == NOISE_KIND:
        if pool_size < 1:
            raise ValueError("the noise pool is empty")
        return Draw(kind, rng.uniform(*SNR_RANGE_DB), rng.randrange(pool_size), rng.random())
    if kind == REVERB:
        return Draw(kind, rng.uniform(*RT60_RANGE_S), rir_seed=rng.getrandbits(64))
    if kind == CODEC:
        return Draw(kind, float(rng.choice(OPUS_CONFIGS).bitrate))
    return Draw(kind)


def apply(clean: np.ndarray, chosen: Draw, pool: tuple[NoiseRecording, ...]) -> np.ndarray:
    """``clean`` (float32, 16 kHz) under ``chosen``: the same length and timing."""
    if chosen.kind == CLEAN:
        return clean
    if chosen.kind == NOISE_KIND:
        recording = pool[chosen.noise_index]
        return add_noise(clean, noise_segment(recording.samples, chosen.noise_offset, len(clean)),
                         chosen.level)
    if chosen.kind == REVERB:
        return reverberate(clean, synthetic_rir(chosen.level,
                                                np.random.default_rng(chosen.rir_seed)))
    if chosen.kind == BAND_LIMIT:
        return band_limit(clean, NARROWBAND_RATE)
    if chosen.kind == CODEC:
        return opus_round_trip(clean, OpusConfig(bitrate=int(chosen.level)))[0]
    raise ValueError(f"unknown condition kind {chosen.kind!r}")


class Augmenter:
    """The #94 mixture as the callable a streaming dataset takes:
    ``augmenter(clean, shard, row)`` is the student's clip.

    The codec's lag is calibrated for every Opus configuration on construction, so a missing
    or broken ``ffmpeg`` stops the run before its first step rather than an hour in, and
    DataLoader workers forked afterwards inherit the calibration.
    """

    def __init__(self, seed: int, pool: tuple[NoiseRecording, ...]) -> None:
        if not pool:
            raise ValueError("the noise pool is empty")
        self.seed = seed
        self.pool = pool
        self.codec_lags = {config.label: codec_lag(config) for config in OPUS_CONFIGS}

    def __call__(self, clean: np.ndarray, shard: int, row: int) -> np.ndarray:
        # A clip of digital silence has no level to set an SNR or a reverberant level
        # against; it is the only clip that stays clean whatever it drew.
        if not np.any(clean):
            return clean
        return apply(clean, draw(self.seed, shard, row, len(self.pool)), self.pool)


def recipe_record() -> dict:
    """The recipe as a run's config carries it (``TrainConfig.augment``, resume-critical):
    every constant a draw or a condition reads, and the committed pool's checksum, so a
    resume under a changed recipe or pool is refused."""
    return {
        "version": RECIPE_VERSION,
        "mixture": {kind: share for kind, share in MIXTURE},
        "snr_db": list(SNR_RANGE_DB),
        "rt60_s": list(RT60_RANGE_S),
        "drr_db": DRR_DB,
        "band_limit_rate": NARROWBAND_RATE,
        "opus": [config.label for config in OPUS_CONFIGS],
        "draw_seed": f"random.Random('<seed + {DRAW_SEED_OFFSET}>:<shard>:<row>')",
        "noise_pool_sha256": hashlib.sha256(POOL_PATH.read_bytes()).hexdigest(),
    }


# --- the training noise pool ---------------------------------------------------------------


def d2_pool_ids() -> frozenset[int]:
    """The ``freesound_id`` of every recording in D2's pool."""
    pool = json.loads(D2_POOL_PATH.read_text(encoding="utf-8"))
    return frozenset(r["freesound_id"] for r in pool["recordings"])


def build_training_pool(cache_dir: Path) -> tuple[tuple[NoiseRecording, ...], dict]:
    """The eligible recordings of :data:`TRAINING_NOISE` that D2's pool does not hold, ordered
    by Freesound id, and the provenance record committed at :data:`POOL_PATH`."""
    pool, left_out = build_pool(cache_dir, TRAINING_NOISE)
    barred = d2_pool_ids()
    kept = tuple(r for r in pool if r.freesound_id not in barred)
    ids = [r.freesound_id for r in kept]
    if len(set(ids)) != len(ids):
        raise ValueError("a freesound_id appears twice in the training shards")
    record = {
        "dataset": {"repo": TRAINING_NOISE.repo, "revision": TRAINING_NOISE.revision,
                    "shards": list(TRAINING_NOISE.shards), "licence": TRAINING_NOISE.licence},
        "left_out": {**left_out, "in_d2_pool": len(pool) - len(kept)},
        "hours": round(sum(len(r.samples) for r in kept) / TARGET_SAMPLE_RATE / 3600, 2),
        "licences": dict(sorted(Counter(r.licence for r in kept).items())),
        "recordings": [r.as_dict() for r in kept],
    }
    return kept, record


def load_training_pool(path: Path) -> tuple[NoiseRecording, ...]:
    """The pool saved on scratch, checked against its committed record and against D2's."""
    pool = load_pool(path, json.loads(POOL_PATH.read_text(encoding="utf-8"))["recordings"])
    if shared := sorted({r.freesound_id for r in pool} & d2_pool_ids()):
        raise ValueError(f"the training pool shares recordings with D2's: {shared[:5]}")
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    pooled = commands.add_parser("noise-pool", help="download, decode, filter and save the "
                                 "training pool; commit its provenance")
    pooled.add_argument("--cache-dir", type=Path, required=True)
    pooled.add_argument("--out", type=Path, required=True, help="the pool's .npz (scratch)")
    commands.add_parser("record", help="print the recipe a run's config carries")
    args = parser.parse_args()

    if args.command == "record":
        print(json.dumps(recipe_record(), indent=1, ensure_ascii=False))
        return
    pool, record = build_training_pool(args.cache_dir)
    save_pool(pool, args.out)
    AUGMENTATION_DIR.mkdir(parents=True, exist_ok=True)
    POOL_PATH.write_text(json.dumps(record, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                         encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "recordings"}, indent=1))


if __name__ == "__main__":
    main()
