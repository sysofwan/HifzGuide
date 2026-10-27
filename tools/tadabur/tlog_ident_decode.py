"""Whole-clip base-teacher decodes of S5's calibration populations (#190), on cuda-dev.

Two populations, one model, one mode — the base teacher at ``01a1ef9`` decoding a whole clip in
one pass (``whole-spans``), which is the mode the Tarteel v1 look decodes are in and the mode the
S5 amendment requires:

* the 52 clips holding the 56 owner-verified real-mistake sites
  (``tlog/ident/real_mistake_clips.jsonl``), read from the staged clip directories and checked
  against each row's ``audio_sha256`` before decoding;
* the Freesound noise pool (``nonpro_gap/noise_pool.json``), rebuilt on scratch when it is not
  there, verified against the committed provenance, and decoded recording by recording. A pool
  recording is up to 174 s, and a whole-recording pass overflows the card's 16 GB, so a
  recording is decoded with the deployed ``stream_b0`` protocol (5 s windows, 1 s hop, tail
  flushed) — the repo's canonical long-audio decode, recorded with the store. The floor itself
  is measured on Tarteel v1 look's whole-clip ``spans``, as pre-registered.

The S5b confirmation (``--only confirm``) decodes the two fresh sets from the committed draws:
the 1,000 Freesound recordings with ``stream_b0`` (the 445 pool's mode) and the 1,000 Common
Voice clips with ``whole-spans`` (the S6/S7 mode). Each store cites the pre-registration and the
manifest it was drawn from (path, SHA-256, commit), and the script refuses to run unless both
are committed and clean.

Outputs are the committed stores ``tlog/ident/decodes/real_mistake.base.json`` and
``tlog/ident/decodes/noise.base.json``: the weights' identity, the decode fingerprint, and one
``spans`` string per item. No C95, student or candidate decode is made or read.

Usage (on the GPU box, from a copy of ``tools/``)::

  flock /root/scratch/gpu.lock python -u -m tadabur.tlog_ident_decode \\
      --clips-dir /root/scratch/issue-83/stage/clips \\
      --clips-dir /root/scratch/issue-126/stage/clips \\
      --cache-dir /root/scratch/tlog/s5/freesound --pool /root/scratch/tlog/s5/noise_pool.npz \\
      --out-dir tools/tadabur/tlog/ident/decodes

``scripts/ident-decode.sh s5 launch`` (or ``s5b``) stages the repo, rebuilds the pool and
starts this detached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from training.decoding import SPANS, Decoder
from training.decode_evalset import read_clip_audio
from training.perturbation import (NOISE_LICENCES, NoiseRecording, build_pool, load_pool,
                                   save_pool)
from training.tashkeel_eval import write_text_atomically
from training.truth_baseline import (
    BATCH_SIZE,
    DEVICE_TYPE,
    PROTOCOLS,
    WEIGHTS_DTYPE,
    model_identity,
)

from .audio import decode_to_mono_16k, decode_to_mono_16k_with_rate_and_decoder
from .nonpro_sources import Dataset, iter_rows
from .tlog_ident import (
    BASE_MODEL,
    BASE_REVISION,
    FLOOR_PREREGISTRATION_PATH,
    NOISE_DECODE_MODE,
    NOISE_DECODES_PATH,
    NOISE_POOL_PATH,
    REAL_MISTAKE_DECODES_PATH,
    real_mistake_clips,
)
from .tlog_ident_confirm import (
    CONFIRM_DECODES,
    CV_MANIFEST_PATH,
    FREESOUND_MANIFEST_PATH,
    require_committed,
)
from .tlog_index import record_of

SCHEMA_VERSION = 1
#: The three S5b confirmation stores, and the arms a rerun may select.
CONFIRM_ARMS = ("freesound", "cv_ar", "cv_other")
#: The pool's decode mode: the deployed stream, which is the only VRAM-safe whole-recording pass.
STREAM_MODE = PROTOCOLS[NOISE_DECODE_MODE]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_store(path: Path, identity: dict, mode: str, fingerprint: dict,
                 items: dict, provenance: dict | None = None) -> None:
    payload = {"schema_version": SCHEMA_VERSION, "model": "base", "identity": identity,
               "fingerprints": {mode: fingerprint}, "items": dict(sorted(items.items()))}
    if provenance is not None:
        payload["provenance"] = provenance
    write_text_atomically(Path(path), json.dumps(
        payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n")


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
    _write_store(out, identity, SPANS, fingerprint, items)
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
            "spans": decoder.decode_stream(recording.samples, block=0, flush_tail=True),
        }
        if position % 25 == 0:
            print(f"pool {position}/{len(pool)}", flush=True)
    _write_store(out, identity, STREAM_MODE,
                 decoder.fingerprint(STREAM_MODE).as_dict(), items)
    print(f"wrote {out} after {time.time() - started:.0f} s", flush=True)
    return items


# --- the S5b confirmation stores (#204) --------------------------------------------------------


def _confirm_chain() -> dict:
    """The doc -> manifests chain every confirmation store cites."""
    return {"doc": record_of(FLOOR_PREREGISTRATION_PATH),
            "manifests": {"freesound": record_of(FREESOUND_MANIFEST_PATH),
                          "common_voice": record_of(CV_MANIFEST_PATH)},
            "pull_request": None, "pull_request_ref": None}


def _store_provenance(chain: dict, manifest: str) -> dict:
    return {"doc": chain["doc"], "manifest": chain["manifests"][manifest],
            "pull_request": chain["pull_request"],
            "pull_request_ref": chain["pull_request_ref"]}


def confirm_cv_samples(path: Path) -> np.ndarray:
    """A Common Voice clip at 16 kHz mono, through the loader the pre-registration pins.

    CV publishes 48 kHz MP3s, and the frozen sets' ``read_clip_audio`` refuses anything but
    16 kHz. The doc names ``tadabur.audio:decode_to_mono_16k_with_rate_and_decoder`` for the
    confirmation rows: soundfile (PyAV when the container needs it), downmix, then
    ``librosa.resample`` (``orig_sr`` -> 16 kHz) — the same loader and resampler the S6/S7
    tlog rows use.
    """
    samples, _rate, _decoder = decode_to_mono_16k_with_rate_and_decoder(path.read_bytes())
    return samples


def _confirm_recordings(manifest: dict, cache_dir: Path) -> dict[int, NoiseRecording]:
    """Stream the drawn shards and return the 1,000 recordings, each checked to its row.

    The check is identity, licence and duration, not the manifest's ``samples_sha256``: the
    float samples are a function of the host's MP3 decoder and resampler, so the Mac draw's
    hash cannot equal the GPU box's reading of the same recording. The id is the draw key;
    the metadata and duration (within one MP3 frame) pin the row, and each store records the
    hash of the samples it actually decoded.
    """
    wanted = {row["freesound_id"]: row for row in manifest["recordings"]}
    found: dict[int, NoiseRecording] = {}
    for shard in manifest["drawn_shards"]:
        dataset = Dataset("freesound_confirm", manifest["dataset"]["repo"],
                          manifest["dataset"]["revision"], (shard,), "per recording", False)
        for row in iter_rows(dataset, shard, cache_dir):
            freesound_id = int(row["freesound_id"])
            if freesound_id not in wanted:
                continue
            samples = decode_to_mono_16k(row["audio"]["bytes"])
            recording = NoiseRecording(freesound_id, NOISE_LICENCES[row["license"]],
                                       row["username"], samples)
            expected = wanted[freesound_id]
            actual = recording.as_dict()
            if actual["licence"] != expected["licence"] or \
                    actual["username"] != expected["username"] or \
                    abs(actual["seconds"] - expected["seconds"]) > 0.1:
                raise ValueError(f"{freesound_id} is not the drawn recording: "
                                 f"{actual} vs {expected}")
            found[freesound_id] = recording
    missing = set(wanted) - set(found)
    if missing:
        raise ValueError(f"{len(missing)} drawn recordings are not in the drawn shards")
    return found


def decode_confirm(decoder, cache_dir: Path, cv_clips_dir: Path, out_dir: Path,
                   identity: dict, arms: set[str] | None = None) -> None:
    """The S5b confirmation stores, each citing the doc and its manifest.

    ``arms`` selects any of ``freesound``, ``cv_ar`` and ``cv_other``; ``None`` is all three.
    A rerun after one arm failed redoes only that arm, never the finished store.
    """
    require_committed([FLOOR_PREREGISTRATION_PATH, FREESOUND_MANIFEST_PATH,
                       CV_MANIFEST_PATH])
    chain = _confirm_chain()
    selected = set(CONFIRM_ARMS if arms is None else arms)
    if "freesound" in selected:
        _decode_confirm_freesound(decoder, cache_dir, out_dir, identity, chain)
    rows = [json.loads(line) for line in
            CV_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()]
    for arm, key in (("ar", "cv_ar"), ("other", "cv_other")):
        if key in selected:
            _decode_confirm_cv(decoder, rows, arm, key, Path(cv_clips_dir), out_dir,
                               identity, chain)


def _decode_confirm_freesound(decoder, cache_dir: Path, out_dir: Path, identity: dict,
                              chain: dict) -> None:
    manifest = json.loads(FREESOUND_MANIFEST_PATH.read_text(encoding="utf-8"))
    recordings = _confirm_recordings(manifest, cache_dir)
    items: dict[str, dict] = {}
    started = time.time()
    for position, freesound_id in enumerate(sorted(recordings), start=1):
        recording = recordings[freesound_id]
        items[str(freesound_id)] = {
            "samples_sha256": _sha256_bytes(recording.samples.tobytes()),
            "spans": decoder.decode_stream(recording.samples, block=0, flush_tail=True),
        }
        if position % 25 == 0:
            print(f"freesound {position}/{len(recordings)}", flush=True)
    out = out_dir / CONFIRM_DECODES["freesound"].name
    _write_store(out, identity, NOISE_DECODE_MODE,
                 decoder.fingerprint(STREAM_MODE).as_dict(),
                 items, _store_provenance(chain, "freesound"))
    print(f"wrote {out} after {time.time() - started:.0f} s", flush=True)


def _decode_confirm_cv(decoder, rows: list[dict], arm: str, key: str, cv_clips_dir: Path,
                       out_dir: Path, identity: dict, chain: dict) -> None:
    started = time.time()
    arm_items: dict[str, dict] = {}
    arm_rows = [row for row in rows if row["arm"] == arm]
    for position, row in enumerate(arm_rows, start=1):
        path = Path(cv_clips_dir) / row["locale"] / row["path"]
        checksum = _sha256_bytes(path.read_bytes())
        if checksum != row["audio_sha256"]:
            raise ValueError(f"{row['path']}: {checksum} is not the recorded "
                             f"{row['audio_sha256']}")
        samples = confirm_cv_samples(path)
        arm_items[row["path"]] = {
            "audio_sha256": checksum,
            "spans": decoder.decode_spans([samples])[0],
        }
        if position % 50 == 0:
            print(f"{arm} {position}/{len(arm_rows)}", flush=True)
    out = out_dir / CONFIRM_DECODES[key].name
    _write_store(out, identity, SPANS, decoder.fingerprint(SPANS).as_dict(), arm_items,
                 _store_provenance(chain, "common_voice"))
    print(f"wrote {out} after {time.time() - started:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips-dir", action="append", type=Path, default=None,
                        help="a staged clip directory (repeatable; for --only clips)")
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="scratch the Freesound shards are downloaded into and deleted from")
    parser.add_argument("--pool", type=Path, default=None,
                        help="the pool .npz on scratch (built when absent; for --only pool)")
    parser.add_argument("--cv-clips-dir", type=Path, default=None,
                        help="the drawn Common Voice clips (for --only confirm)")
    parser.add_argument("--out-dir", type=Path, default=NOISE_DECODES_PATH.parent)
    parser.add_argument("--only", choices=("clips", "pool", "confirm"), default=None)
    parser.add_argument("--confirm-arms", nargs="+", choices=CONFIRM_ARMS, default=None,
                        help="which confirmation stores to write (default: all three)")
    args = parser.parse_args()
    if args.only in (None, "clips") and not args.clips_dir:
        parser.error("--clips-dir is required for the real-mistake clips")
    if args.only in (None, "pool") and args.pool is None:
        parser.error("--pool is required for the noise pool")
    if args.only in (None, "confirm") and args.cv_clips_dir is None:
        parser.error("--cv-clips-dir is required for the confirmation stores")

    decoder, identity, fingerprint = _decoder()
    if args.only in (None, "clips"):
        decode_clips(decoder, real_mistake_clips(), args.clips_dir,
                     args.out_dir / REAL_MISTAKE_DECODES_PATH.name, identity, fingerprint)
    if args.only in (None, "pool"):
        decode_pool(decoder, args.pool, args.cache_dir,
                    args.out_dir / NOISE_DECODES_PATH.name, identity, fingerprint)
    if args.only in (None, "confirm"):
        decode_confirm(decoder, args.cache_dir, args.cv_clips_dir, args.out_dir, identity,
                       set(args.confirm_arms) if args.confirm_arms else None)


if __name__ == "__main__":
    main()
