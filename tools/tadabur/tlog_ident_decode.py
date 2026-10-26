"""Whole-clip base-teacher decodes of S5's calibration populations (#190), on cuda-dev.

Two populations, one model, one mode — the base teacher at ``01a1ef9`` decoding a whole clip in
one pass (``whole-spans``), which is the mode the Tarteel v1 look decodes are in and the mode the
S5 amendment requires:

* the 52 clips holding the 56 owner-verified real-mistake sites
  (``tlog/ident/real_mistake_clips.jsonl``), read from the staged clip directories and checked
  against each row's ``audio_sha256`` before decoding;
* the Freesound noise pool (``nonpro_gap/noise_pool.json``), rebuilt on scratch when it is not
  there, verified against the committed provenance, and decoded recording by recording.

Outputs are the committed stores ``tlog/ident/decodes/real_mistake.base.json`` and
``tlog/ident/decodes/noise.base.json``: the weights' identity, the decode fingerprint, and one
``spans`` string per item. No C95, student or candidate decode is made or read.

Usage (on the GPU box, from a copy of ``tools/``)::

  flock /root/scratch/gpu.lock python -u -m tadabur.tlog_ident_decode \\
      --clips-dir /root/scratch/issue-83/stage/clips \\
      --clips-dir /root/scratch/issue-126/stage/clips \\
      --cache-dir /root/scratch/tlog/s5/freesound --pool /root/scratch/tlog/s5/noise_pool.npz \\
      --out-dir tools/tadabur/tlog/ident/decodes

``scripts/issue-190-ident-decode.sh`` stages the repo, rebuilds the pool and starts this detached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from training.decoding import SPANS, Decoder
from training.decode_evalset import read_clip_audio
from training.perturbation import build_pool, load_pool, save_pool
from training.tashkeel_eval import write_text_atomically
from training.truth_baseline import (
    BATCH_SIZE,
    DEVICE_TYPE,
    WEIGHTS_DTYPE,
    model_identity,
)

from .tlog_ident import (
    BASE_MODEL,
    BASE_REVISION,
    NOISE_DECODES_PATH,
    NOISE_POOL_PATH,
    REAL_MISTAKE_DECODES_PATH,
    real_mistake_clips,
)

SCHEMA_VERSION = 1


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_store(path: Path, identity: dict, fingerprint: dict, items: dict) -> None:
    write_text_atomically(Path(path), json.dumps(
        {"schema_version": SCHEMA_VERSION, "model": "base", "identity": identity,
         "fingerprints": {SPANS: fingerprint}, "items": dict(sorted(items.items()))},
        ensure_ascii=False, indent=1, sort_keys=True) + "\n")


def _decoder():
    decoder = Decoder.load(BASE_MODEL, DEVICE_TYPE, weights_dtype=WEIGHTS_DTYPE,
                           batch_size=BATCH_SIZE, revision=BASE_REVISION)
    identity = model_identity(BASE_MODEL, decoder)
    if identity != {"model_ref": BASE_MODEL, "hub_revision": BASE_REVISION}:
        raise ValueError(f"the base teacher resolved to {identity}, not the pinned revision")
    return decoder, identity, decoder.fingerprint(SPANS).as_dict()


def decode_clips(decoder, clips: list[dict], clips_dirs: list[Path], out: Path,
                 identity: dict, fingerprint: dict) -> dict:
    """The 52 real-mistake clips, whole, each checked against its recorded audio checksum."""
    items: dict[str, dict] = {}
    started = time.time()
    for position, clip in enumerate(clips, start=1):
        path = next((directory / clip["audio_filename"] for directory in clips_dirs
                     if (directory / clip["audio_filename"]).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"{clip['audio_filename']} is in no staged clip directory")
        checksum = _sha256_bytes(Path(path).read_bytes())
        if checksum != clip["audio_sha256"]:
            raise ValueError(f"{clip['audio_filename']}: {checksum} is not the recorded "
                             f"{clip['audio_sha256']}")
        samples = read_clip_audio(Path(path).parent, clip["audio_filename"])
        items[clip["audio_filename"]] = {
            "audio_sha256": checksum,
            "spans": decoder.decode_spans([samples])[0],
        }
        print(f"clip {position}/{len(clips)} {clip['audio_filename']}", flush=True)
    _write_store(out, identity, fingerprint, items)
    print(f"wrote {out} after {time.time() - started:.0f} s", flush=True)
    return items


def decode_pool(decoder, pool_path: Path, cache_dir: Path, out: Path, identity: dict,
                fingerprint: dict) -> dict:
    """The Freesound pool, whole, after checking it against the committed provenance."""
    records = json.loads(NOISE_POOL_PATH.read_text(encoding="utf-8"))["recordings"]
    if Path(pool_path).is_file():
        pool = load_pool(Path(pool_path), records)
    else:
        built, left_out = build_pool(Path(cache_dir))
        if [recording.as_dict() for recording in built] != records:
            raise ValueError(f"the rebuilt pool is not the committed one (left out {left_out})")
        Path(pool_path).parent.mkdir(parents=True, exist_ok=True)
        save_pool(built, Path(pool_path))
        pool = built
    items: dict[str, dict] = {}
    started = time.time()
    for position, recording in enumerate(pool, start=1):
        items[str(recording.freesound_id)] = {
            "samples_sha256": _sha256_bytes(recording.samples.tobytes()),
            "spans": decoder.decode_spans([recording.samples])[0],
        }
        if position % 25 == 0:
            print(f"pool {position}/{len(pool)}", flush=True)
    _write_store(out, identity, fingerprint, items)
    print(f"wrote {out} after {time.time() - started:.0f} s", flush=True)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips-dir", action="append", type=Path, required=True,
                        help="a staged clip directory (repeatable)")
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="scratch the Freesound shards are downloaded into and deleted from")
    parser.add_argument("--pool", type=Path, required=True,
                        help="the pool .npz on scratch (built when absent)")
    parser.add_argument("--out-dir", type=Path, default=NOISE_DECODES_PATH.parent)
    parser.add_argument("--only", choices=("clips", "pool"), default=None)
    args = parser.parse_args()

    decoder, identity, fingerprint = _decoder()
    if args.only in (None, "clips"):
        decode_clips(decoder, real_mistake_clips(), args.clips_dir,
                     args.out_dir / REAL_MISTAKE_DECODES_PATH.name, identity, fingerprint)
    if args.only in (None, "pool"):
        decode_pool(decoder, args.pool, args.cache_dir,
                    args.out_dir / NOISE_DECODES_PATH.name, identity, fingerprint)


if __name__ == "__main__":
    main()
