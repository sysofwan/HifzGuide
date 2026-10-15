"""Decode #159's Tadabur dev clips, clean and under the committed grid, with both models.

One condition per run (:func:`training.nonpro_gap.decode_plan`): ``clean`` decodes the whole
``decode_evalset`` dev half (D1, and D2's reference), every other condition the committed D2
sample, each clip perturbed by :func:`training.perturbation.perturb` as it is read. Both models
decode at ``stream_b0`` only, through :mod:`training.external_decode`'s loop (the truth
baseline's protocol and identity checks, resumable, timed). Each line is the clip's provenance
(its name, reciter, the SHA-256 of its clean float32 samples, the condition and what was drawn
for it) and its decode. Decodes go to ``<out>/<condition>.<model>.jsonl``, each model's identity
to ``<out>/<model>.identity.json``, the run's timing to ``<out>/<condition>.timing.jsonl``.

Refuses to run unless the committed grid is the grid the code applies
(:func:`training.nonpro_gap.committed_grid`) and, for a noise condition, the pool on scratch is
the committed one.

Usage (on the GPU box, from a copy of ``tools/``)::

  flock /root/scratch/gpu.lock env OMP_NUM_THREADS=1 python -m training.nonpro_gap_decode \\
      --condition noise_snr10 --clips-dir /root/repos/HifzGuide/tools/tadabur/gate_eval/clips \\
      --pool /root/scratch/issue-159/noise_pool.npz --out-dir /root/scratch/issue-159/decodes
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from training.decode_evalset import read_clip_audio
from training.external_decode import Recording, append_timing, decode_recordings, load_decoders
from training.nonpro_gap import POOL_PATH, STREAM, DevClip, committed_grid, decode_plan
from training.perturbation import (
    CONDITIONS,
    NOISE_KIND,
    Condition,
    NoiseRecording,
    ffmpeg_version,
    load_pool,
    perturb,
)
from training.truth_baseline import DECODE_STORE, PROTOCOLS, Store


def _stream(decoder, span) -> dict:
    return {STREAM: decoder.decode_stream(span, block=0, flush_tail=True)}


#: The deployed b = 0 stream alone: the one protocol D1 and D2 read on Tadabur.
STREAM_STORE = Store(DECODE_STORE.directory, {STREAM: PROTOCOLS[STREAM]}, _stream)


def recordings(condition: Condition, clips: Sequence[DevClip], clips_dir: Path,
               pool: tuple[NoiseRecording, ...]) -> Iterator[Recording]:
    for clip in clips:
        def load(clip=clip):
            clean = read_clip_audio(clips_dir, clip.filename)
            samples, draw = perturb(clean, condition, clip.filename, pool)
            return samples, [{"key": clip.filename, "reciter_id": clip.reciter_id,
                              "clean_sha256": hashlib.sha256(clean.tobytes()).hexdigest(),
                              "condition": condition.name, **draw,
                              "start_sample": 0, "end_sample": len(samples)}]

        yield Recording((clip.filename,), load)


def run(condition: Condition, clips_dir: Path, pool_path: Path | None, out_dir: Path) -> None:
    committed_grid()
    pool: tuple[NoiseRecording, ...] = ()
    if condition.kind == NOISE_KIND:
        pool = load_pool(pool_path, json.loads(POOL_PATH.read_text(encoding="utf-8"))
                         ["recordings"])
    out_dir.mkdir(parents=True, exist_ok=True)
    decoders = load_decoders(lambda model: out_dir / f"{model}.identity.json")
    paths = {model: out_dir / f"{condition.name}.{model}.jsonl" for model in decoders}
    timing = decode_recordings(recordings(condition, decode_plan(condition), clips_dir, pool),
                               decoders, paths, condition.name, {STREAM: STREAM_STORE})
    append_timing(out_dir / f"{condition.name}.timing.jsonl",
                  {"condition": condition.name, "ffmpeg": ffmpeg_version(), **timing})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--clips-dir", type=Path, required=True,
                        help="the decode_evalset's clips (gate_eval/clips)")
    parser.add_argument("--pool", type=Path, help="the noise pool's .npz (noise conditions)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(CONDITIONS[args.condition], args.clips_dir, args.pool, args.out_dir)


if __name__ == "__main__":
    main()
