"""Decode the QUD pilot mushafs with the base teacher and ``h448``, every arm (#144).

The pilot's items are QUD's own waqf-aware segments (``segments`` of each ayah row,
:mod:`tadabur.qud_source`): the span of the ayah's 16 kHz decode between a segment's start and
end. Each item is decoded by both models exactly as the truth baseline decodes its items
(:mod:`training.truth_baseline`: bf16 weights, batch size 1, CUDA), so the two are comparable:

* ``spans``: the segment decoded whole;
* ``stream_b0``: the deployed b = 0 stream over the segment, tail flushed;
* ``checks``: the segment's recorded check sequence (:meth:`training.decoding.Decoder.app_checks`),
  what Muraja's transcriber would hand its engine, for the Muraja harness.

Each model's identity and fingerprints are checked against the truth baseline's frozen ones
(:func:`training.truth_baseline.expected_fingerprint`, :func:`~training.truth_baseline.model_identity`)
and written beside its decodes. Decodes are appended one item per line to
``<out>/<slug>.<shard>.<model>.jsonl`` as they are made, so a run resumes where it stopped;
each line carries the ayah's MP3 checksum and the item's sample span. One shard of the mushaf
is read per run (downloaded into ``--cache-dir`` and deleted when read), so shards can be
decoded side by side; the shard's wall time per model and arm is appended to
``<out>/<slug>.<shard>.timing.jsonl``. The work is CPU-bound (one feature extraction and one
batch-1 forward per check), so several runs share the GPU, each with few CPU threads.

Usage (on the GPU box, from a copy of ``tools/``)::

  flock /root/scratch/gpu.lock bash -c 'for s in 0 1 2 3 4 5; do OMP_NUM_THREADS=2 \\
      python -m training.qud_decode --slug anas_almiman_yt --shard $s \\
      --out-dir /root/scratch/issue-144/decodes --cache-dir /root/scratch/issue-144/hf_cache & \\
      done; wait'
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from tadabur.audio import TARGET_SAMPLE_RATE
from tadabur.qud_selection import SELECTION_PATH
from tadabur.qud_source import decode_ayah, iter_mushaf_rows, mp3_sha256, segment_span
from training.truth_baseline import (
    BATCH_SIZE,
    CHECK_PROTOCOLS,
    CHECK_STORE,
    CHECKS,
    DECODE_STORE,
    DEFAULT_MODELS,
    DEVICE_TYPE,
    PROTOCOLS,
    WEIGHTS_DTYPE,
    expected_fingerprint,
    model_identity,
)

MODES = {**PROTOCOLS, **CHECK_PROTOCOLS}
#: The truth baseline's two stores' decode functions: ``spans`` and ``stream_b0`` together,
#: then the check sequence; each is timed on its own.
STORES = {"spans+stream_b0": DECODE_STORE, CHECKS: CHECK_STORE}
#: Items between progress lines.
PROGRESS_EVERY = 200


def item_key(audio_filename: str, segment_index: int) -> str:
    return f"{audio_filename}#{segment_index}"


def decodes_path(out_dir: Path, slug: str, shard_index: int, model: str) -> Path:
    """One shard's decodes by one model, so shards can be decoded side by side."""
    return out_dir / f"{slug}.{shard_index}.{model}.jsonl"


def identity_path(out_dir: Path, slug: str, model: str) -> Path:
    return out_dir / f"{slug}.{model}.identity.json"


def done_keys(path: Path) -> set[str]:
    """The items a decodes file already holds (a torn last line is dropped and redone)."""
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            keys.add(json.loads(line)["key"])
        except json.JSONDecodeError:
            continue
    return keys


def mushaf_shards(slug: str) -> list[str]:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    (mushaf,) = [m for m in selection["mushafs"] if m["slug"] == slug]
    return mushaf["shards"]


def load_decoders(out_dir: Path, slug: str) -> dict:
    """Both models, checked against the truth baseline's frozen protocol and against the
    identity a previous run of this mushaf recorded."""
    from training.decoding import Decoder

    decoders = {}
    for name, ref in DEFAULT_MODELS.items():
        decoder = Decoder.load(ref, DEVICE_TYPE, weights_dtype=WEIGHTS_DTYPE,
                               batch_size=BATCH_SIZE)
        record = {
            "model": name,
            "identity": model_identity(ref, decoder),
            "fingerprints": {arm: decoder.fingerprint(mode).as_dict()
                             for arm, mode in MODES.items()},
        }
        for arm, mode in MODES.items():
            if record["fingerprints"][arm] != expected_fingerprint(ref, mode):
                raise ValueError(f"{name} decodes {arm} as {record['fingerprints'][arm]}")
        path = identity_path(out_dir, slug, name)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != record:
                raise ValueError(f"{path} records other weights or settings than {name} has now")
        else:  # shards run side by side: each writes through its own temporary file
            temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, path)
        decoders[name] = decoder
    return decoders


def run_shard(slug: str, shard_index: int, out_dir: Path, cache_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = mushaf_shards(slug)[shard_index]
    decoders = load_decoders(out_dir, slug)
    paths = {name: decodes_path(out_dir, slug, shard_index, name) for name in decoders}
    done = {name: done_keys(path) for name, path in paths.items()}
    seconds: dict[str, dict[str, float]] = {name: defaultdict(float) for name in decoders}
    audio_seconds = 0.0
    items = 0
    started = time.time()
    files = {name: open(path, "a", encoding="utf-8") for name, path in paths.items()}
    try:
        for row in iter_mushaf_rows(slug, [shard], cache_dir):
            keys = [item_key(row.audio_filename, k) for k in range(len(row.segments))]
            if all(key in done[name] for key in keys for name in decoders):
                continue
            samples = decode_ayah(row)
            for k, segment in enumerate(row.segments):
                start, end = segment_span(segment, len(samples))
                span = samples[start:end]
                audio_seconds += len(span) / TARGET_SAMPLE_RATE
                items += 1
                for name, decoder in decoders.items():
                    if keys[k] in done[name]:
                        continue
                    line = {"key": keys[k], "audio_filename": row.audio_filename,
                            "mp3_sha256": mp3_sha256(row.mp3), "surah_ayah": row.surah_ayah,
                            "segment_index": k, "segment": list(segment),
                            "start_sample": start, "end_sample": end}
                    for label, store in STORES.items():
                        tick = time.time()
                        line.update(store.decode(decoder, span))
                        seconds[name][label] += time.time() - tick
                    files[name].write(json.dumps(line, ensure_ascii=False,
                                                 separators=(",", ":")) + "\n")
                    files[name].flush()
                if items % PROGRESS_EVERY == 0:
                    print(f"{slug} {shard}: {items} items, {audio_seconds:.0f} s of audio in "
                          f"{time.time() - started:.0f} s", flush=True)
    finally:
        for handle in files.values():
            handle.close()
    timing = {"slug": slug, "shard": shard, "items": items,
              "audio_seconds": round(audio_seconds, 1),
              "wall_seconds": round(time.time() - started, 1),
              "decode_seconds": {n: {a: round(s, 1) for a, s in arms.items()}
                                 for n, arms in seconds.items()}}
    with open(out_dir / f"{slug}.{shard_index}.timing.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(timing) + "\n")
    print(json.dumps(timing), flush=True)


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
