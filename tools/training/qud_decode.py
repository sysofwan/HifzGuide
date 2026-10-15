"""Decode the QUD pilot mushafs with the base teacher and ``h448``, every arm (#144).

The pilot's items are QUD's own waqf-aware segments (``segments`` of each ayah row,
:mod:`tadabur.qud_source`): the span of the ayah's 16 kHz decode between a segment's start and
end. Each is decoded by :mod:`training.external_decode`, the loop every external corpus shares
(``spans``, ``stream_b0`` and ``checks``, the truth baseline's protocol, resumable); each line
carries the ayah's MP3 checksum and the item's sample span. Decodes go to
``<out>/<slug>.<shard>.<model>.jsonl``, each model's identity to
``<out>/<slug>.<model>.identity.json``. One shard of the mushaf is read per run (downloaded into
``--cache-dir`` and deleted when read), so shards can be decoded side by side; the shard's wall
time per model and arm is appended to ``<out>/<slug>.<shard>.timing.jsonl``.

Usage (on the GPU box, from a copy of ``tools/``)::

  flock /root/scratch/gpu.lock bash -c 'for s in 0 1 2 3 4 5; do OMP_NUM_THREADS=2 \\
      python -m training.qud_decode --slug anas_almiman_yt --shard $s \\
      --out-dir /root/scratch/issue-144/decodes --cache-dir /root/scratch/issue-144/hf_cache & \\
      done; wait'
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from tadabur.qud_selection import SELECTION_PATH
from tadabur.qud_source import decode_ayah, iter_mushaf_rows, mp3_sha256, segment_span
from training.external_decode import Recording, append_timing, decode_recordings, load_decoders


def item_key(audio_filename: str, segment_index: int) -> str:
    return f"{audio_filename}#{segment_index}"


def decodes_path(out_dir: Path, slug: str, shard_index: int, model: str) -> Path:
    """One shard's decodes by one model, so shards can be decoded side by side."""
    return out_dir / f"{slug}.{shard_index}.{model}.jsonl"


def identity_path(out_dir: Path, slug: str, model: str) -> Path:
    return out_dir / f"{slug}.{model}.identity.json"


def mushaf_shards(slug: str) -> list[str]:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    (mushaf,) = [m for m in selection["mushafs"] if m["slug"] == slug]
    return mushaf["shards"]


def recordings(slug: str, shard: str, cache_dir: Path) -> Iterator[Recording]:
    """Each ayah of one shard, its items QUD's segments of it."""
    for row in iter_mushaf_rows(slug, [shard], cache_dir):
        keys = tuple(item_key(row.audio_filename, k) for k in range(len(row.segments)))

        def load(row=row, keys=keys):
            samples = decode_ayah(row)
            records = []
            for k, segment in enumerate(row.segments):
                start, end = segment_span(segment, len(samples))
                records.append({"key": keys[k], "audio_filename": row.audio_filename,
                                "mp3_sha256": mp3_sha256(row.mp3), "surah_ayah": row.surah_ayah,
                                "segment_index": k, "segment": list(segment),
                                "start_sample": start, "end_sample": end})
            return samples, records

        yield Recording(keys, load)


def run_shard(slug: str, shard_index: int, out_dir: Path, cache_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = mushaf_shards(slug)[shard_index]
    decoders = load_decoders(lambda model: identity_path(out_dir, slug, model))
    paths = {name: decodes_path(out_dir, slug, shard_index, name) for name in decoders}
    timing = decode_recordings(recordings(slug, shard, cache_dir), decoders, paths,
                               f"{slug} {shard}")
    append_timing(out_dir / f"{slug}.{shard_index}.timing.jsonl",
                  {"slug": slug, "shard": shard, **timing})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--shard", type=int, required=True, help="index into the mushaf's shards")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    run_shard(args.slug, args.shard, args.out_dir, args.cache_dir)


if __name__ == "__main__":
    main()
