"""Decode the non-professional corpora with the base teacher and ``h448``, every arm (#156).

Each selected clip (:func:`tadabur.nonpro_sources.select`: every Tarteel v1 clip, RetaSy's Hafs
clips labelled ``correct`` or ``in_correct`` whose text is Quran, every qdat clip) is one item,
the whole clip, decoded by :mod:`training.external_decode`, the loop every external corpus
shares (``spans``, ``stream_b0`` and ``checks``, the truth baseline's protocol, resumable).
Each line is the clip's provenance (its id, the SHA-256 of its audio as published, the canonical
words it recites, its reciter and the dataset's labels kept with it) and its decodes. Decodes go
to ``<out>/<dataset>.<shard>.<model>.jsonl``, each model's identity to
``<out>/<dataset>.<model>.identity.json``, the shard's timing to
``<out>/<dataset>.<shard>.timing.jsonl``. One shard is read per run (downloaded into
``--cache-dir`` and deleted when read), so shards are decoded side by side.

Usage (on the GPU box, from a copy of ``tools/``)::

  flock /root/scratch/gpu.lock env OMP_NUM_THREADS=1 python -m training.nonpro_decode \\
      --dataset qdat --shard 0 --out-dir /root/scratch/issue-156/decodes \\
      --cache-dir /root/scratch/issue-156/hf_cache
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from tadabur.audio import decode_to_mono_16k
from tadabur.nonpro_sources import DATASETS, Dataset, LeftOut, audio_sha256, iter_rows, select
from training.external_decode import Recording, append_timing, decode_recordings, load_decoders


def item_key(audio_filename: str) -> str:
    """A clip's one item, named as a QUD segment is (``<clip>#<segment>``)."""
    return f"{audio_filename}#0"


def decodes_path(out_dir: Path, dataset: str, shard_index: int, model: str) -> Path:
    return out_dir / f"{dataset}.{shard_index}.{model}.jsonl"


def identity_path(out_dir: Path, dataset: str, model: str) -> Path:
    return out_dir / f"{dataset}.{model}.identity.json"


def provenance(dataset: Dataset, row: dict) -> dict | None:
    """A row's provenance record without its sample span, or ``None`` for a row not selected."""
    clip = select(dataset, row)
    if isinstance(clip, LeftOut):
        return None
    return {"key": item_key(clip.audio_filename), "audio_filename": clip.audio_filename,
            "audio_sha256": audio_sha256(clip.audio), "surah_ayah": clip.surah_ayah,
            "word_from": clip.word_from, "word_to": clip.word_to, "reciter": clip.reciter,
            "labels": clip.labels}


def recordings(dataset: Dataset, shard: str, cache_dir: Path) -> Iterator[Recording]:
    """Each selected clip of one shard, its one item the whole clip."""
    for row in iter_rows(dataset, shard, cache_dir):
        record = provenance(dataset, row)
        if record is None:
            continue

        def load(record=record, audio=row["audio"]["bytes"]):
            samples = decode_to_mono_16k(audio)
            return samples, [{**record, "start_sample": 0, "end_sample": len(samples)}]

        yield Recording((record["key"],), load)


def run_shard(name: str, shard_index: int, out_dir: Path, cache_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = DATASETS[name]
    shard = dataset.shards[shard_index]
    decoders = load_decoders(lambda model: identity_path(out_dir, name, model))
    paths = {model: decodes_path(out_dir, name, shard_index, model) for model in decoders}
    timing = decode_recordings(recordings(dataset, shard, cache_dir), decoders, paths,
                               f"{name} {shard}")
    append_timing(out_dir / f"{name}.{shard_index}.timing.jsonl",
                  {"dataset": name, "shard": shard, **timing})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["tarteel_v1", "retasy", "qdat"])
    parser.add_argument("--shard", type=int, required=True, help="index into its shards")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    run_shard(args.dataset, args.shard, args.out_dir, args.cache_dir)


if __name__ == "__main__":
    main()
