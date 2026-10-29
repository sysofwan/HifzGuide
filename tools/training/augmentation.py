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

**Draws are a pure function of the run's seed and the clip's Tadabur row** (:func:`draw`;
a noise segment too quiet to scale is re-drawn from the same keyed stream, :func:`audible_noise`):
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
  python -m training.augmentation dry-run --pool <npz> --index <full index> \
      --exposure-use aug.training --seed 9494 --steps 10000   # silent draws a run would hit

Torch-free, but ``dry-run``, which cuts windows as the training stream does
(:mod:`training.distill_stream`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, replace
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
    NOISE_MIN_RMS,
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

#: v2 (after the first candidate run crashed on a silent noise segment, before any result):
#: a noise segment at or below the pool's silence floor is re-drawn (:func:`audible_noise`);
#: every other clip's draw is v1's.
RECIPE_VERSION = "aug94-mixture-v2"

#: Each condition kind and its share of clips; the shares sum to one.
MIXTURE = ((CLEAN, 0.30), (NOISE_KIND, 0.35), (REVERB, 0.15), (BAND_LIMIT, 0.10), (CODEC, 0.10))
SNR_RANGE_DB = (3.0, 20.0)
RT60_RANGE_S = (0.2, 1.0)
#: The run seed's offset for the augmentation draws: ``seed`` orders the shards and
#: ``seed + 1`` drives the shuffle buffer (:mod:`training.distill_stream`).
DRAW_SEED_OFFSET = 2
#: How many recordings and offsets a noise clip may draw (its own and up to seven re-draws)
#: before it gives up and stays clean. A
#: pool recording can hold a long silent stretch (one holds 17.6 s of exact zeros), so a
#: segment as long as a clip can be silent although the recording passed the pool's checks.
NOISE_DRAWS = 8

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
    (a fraction of its length), for reverb the impulse response's seed. ``noise_redraws``
    counts the silent noise segments drawn before this one (:func:`audible_noise`); a clean
    draw with :data:`NOISE_DRAWS` of them is a noise clip that found no audible segment."""

    kind: str
    level: float | None = None
    noise_index: int | None = None
    noise_offset: float | None = None
    rir_seed: int | None = None
    noise_redraws: int = 0


def draw(seed: int, shard: int | str, row: int, pool_size: int) -> Draw:
    """The draw for row ``row`` of ``shard`` in a run seeded ``seed``.

    ``shard`` is a Tadabur shard's index or a tlog clip's ``f"{split}:{shard_file}"`` key
    (:class:`training.distill_mixed.MixedWindowDataset`); it is formatted into the seed
    string, so the two sources cannot share a draw.
    """
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


def audible_noise(chosen: Draw, seed: int, shard: int | str, row: int,
                  pool: tuple[NoiseRecording, ...], length: int) -> Draw:
    """``chosen`` (a noise draw) if its segment of ``length`` samples is audible -- its RMS
    above the pool's silence floor, :data:`perturbation.NOISE_MIN_RMS` (-80 dBFS), below which
    no SNR can be set from it; else the first audible re-draw of the recording and offset,
    attempt ``k`` from ``random.Random('<seed + 2>:<shard>:<row>:<k>')`` (the SNR kept); else,
    when all :data:`NOISE_DRAWS` segments are silent, a clean draw. A clip whose first segment
    is audible keeps its draw exactly."""
    candidate = chosen
    for attempt in range(NOISE_DRAWS):
        if attempt:
            rng = random.Random(f"{seed + DRAW_SEED_OFFSET}:{shard}:{row}:{attempt}")
            candidate = replace(chosen, noise_index=rng.randrange(len(pool)),
                                noise_offset=rng.random(), noise_redraws=attempt)
        segment = noise_segment(pool[candidate.noise_index].samples, candidate.noise_offset,
                                length)
        if np.sqrt(np.mean(np.square(segment, dtype=np.float64))) > NOISE_MIN_RMS:
            return candidate
    return Draw(CLEAN, noise_redraws=NOISE_DRAWS)


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
        #: Noise clips whose first segment was silent: re-drawn, and left clean (this worker).
        self.noise_redrawn = 0
        self.noise_left_clean = 0

    def resolve(self, clean: np.ndarray, shard: int | str, row: int) -> Draw:
        """The draw a clip is augmented under: its keyed draw, a silent noise segment
        re-drawn (:func:`audible_noise`). Logs each re-draw; never raises for one."""
        chosen = draw(self.seed, shard, row, len(self.pool))
        if chosen.kind != NOISE_KIND:
            return chosen
        resolved = audible_noise(chosen, self.seed, shard, row, self.pool, len(clean))
        if resolved.noise_redraws:
            left_clean = resolved.kind == CLEAN
            self.noise_left_clean += left_clean
            self.noise_redrawn += not left_clean
            print(f"[augment] silent noise segment at ({shard}, {row}): "
                  f"{'left clean after ' if left_clean else ''}{resolved.noise_redraws} "
                  f"re-draw(s); this worker: {self.noise_redrawn} re-drawn, "
                  f"{self.noise_left_clean} left clean", flush=True)
        return resolved

    def __call__(self, clean: np.ndarray, shard: int | str, row: int) -> np.ndarray:
        # A clip of digital silence has no level to set an SNR or a reverberant level
        # against; it is the only clip that stays clean whatever it drew.
        if not np.any(clean):
            return clean
        return apply(clean, self.resolve(clean, shard, row), self.pool)


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
        "silent_noise": {"floor_rms": NOISE_MIN_RMS, "draws": NOISE_DRAWS,
                         "redraw_seed": f"random.Random('<seed + {DRAW_SEED_OFFSET}>:<shard>:"
                                        "<row>:<attempt>')", "exhausted": "clean"},
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


# --- a dry run of a run's noise draws ------------------------------------------------------


def dry_run(seed: int, pool: tuple[NoiseRecording, ...], rows, shard_order: list[int],
            windows_per_step: int, steps: int) -> dict:
    """How a run's noise draws resolve (:func:`audible_noise`), without its audio: ``rows``
    are ``(shard, row, samples)`` of the admitted clips (lengths from a shard index), read in
    ``shard_order``. Counts over the shards a run of ``steps`` reaches (its windows, as the
    stream cuts them) and over all of them; the first silent draws are listed."""
    from training.distill_stream import row_window_starts

    by_shard: dict[int, list[tuple[int, int]]] = {}
    for shard, row, samples in rows:
        by_shard.setdefault(shard, []).append((row, samples))
    counts = Counter()
    reached, windows, first = None, 0, []
    for position, shard in enumerate(shard_order, 1):
        for row, samples in sorted(by_shard.get(shard, ())):
            windows += len(row_window_starts(samples))
            chosen = draw(seed, shard, row, len(pool))
            counts["clips"] += 1
            if chosen.kind != NOISE_KIND:
                continue
            counts["noise"] += 1
            resolved = audible_noise(chosen, seed, shard, row, pool, samples)
            if resolved.noise_redraws:
                counts["left_clean" if resolved.kind == CLEAN else "redrawn"] += 1
                if len(first) < 10:
                    first.append({"shard": shard, "row": row,
                                  "freesound_id": pool[chosen.noise_index].freesound_id,
                                  "redraws": resolved.noise_redraws,
                                  "step": windows // windows_per_step})
        if reached is None and windows >= steps * windows_per_step:
            reached = {"shards": position, **counts}
    return {"steps": steps, "reached": reached, "all_shards": dict(counts), "first": first}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    pooled = commands.add_parser("noise-pool", help="download, decode, filter and save the "
                                 "training pool; commit its provenance")
    pooled.add_argument("--cache-dir", type=Path, required=True)
    pooled.add_argument("--out", type=Path, required=True, help="the pool's .npz (scratch)")
    commands.add_parser("record", help="print the recipe a run's config carries")
    dry = commands.add_parser("dry-run", help="count how a run's noise draws resolve, from a "
                              "shard index and the pool (no other audio, no GPU)")
    dry.add_argument("--pool", type=Path, required=True)
    dry.add_argument("--index", type=Path, required=True,
                     help="a full shard index (tadabur.staged_audio index)")
    dry.add_argument("--exposure-use", required=True, help="the run's exposure use")
    dry.add_argument("--seed", type=int, required=True)
    dry.add_argument("--steps", type=int, required=True)
    dry.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.command == "record":
        print(json.dumps(recipe_record(), indent=1, ensure_ascii=False))
        return
    if args.command == "dry-run":
        from tadabur.exposure import RowExclusion
        from tadabur.shard_reader import parse_shard_spec
        from tadabur.staged_audio import read_shard_index
        from training.decode_evalset import training_shard_spec

        shards = parse_shard_spec(training_shard_spec())
        order = list(shards)
        random.Random(args.seed).shuffle(order)  # one worker's order (distill_stream)
        exclusion = RowExclusion.for_use(args.exposure_use)
        index = sorted(read_shard_index(args.index).values(), key=lambda r: (r.shard, r.row_index))
        rows = [(r.shard, r.row_index, int(round(r.duration_s * TARGET_SAMPLE_RATE)))
                for r in index if r.shard in set(shards) and r.duration_s > 0
                and exclusion.keeps((r.shard, r.row_index), {"reciter_id": r.reciter_id})]
        report = dry_run(args.seed, load_training_pool(args.pool), rows, order,
                         args.batch_size, args.steps)
        print(json.dumps({**report, "no_duration": sum(1 for r in index if r.shard in
                                                        set(shards) and r.duration_s <= 0)},
                         indent=1))
        return
    pool, record = build_training_pool(args.cache_dir)
    save_pool(pool, args.out)
    AUGMENTATION_DIR.mkdir(parents=True, exist_ok=True)
    POOL_PATH.write_text(json.dumps(record, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                         encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "recordings"}, indent=1))


if __name__ == "__main__":
    main()
