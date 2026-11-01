"""Decode #245's comparison sets with the Quran-Lab zipformer, and time the batch first.

The screen compares zipformer v3.2 against the base teacher, the round-2 candidate and C95 on
sets those three never trained on. :mod:`training.zip_screen_decode` is our models' side;
this module is zipformer's, through the arms :mod:`training.zipformer_decode` defines:

* **the single stream** (``spans`` and ``checks``) on every set: the benchmark's 577 allowed
  clips, the 300 synthetic edits, qdat's 1,505 clips, Ship A's 530 verified sites and the unread
  QUD half;
* **the warm-started per-check arms** (``checks-w1`` and ``checks-w1-windows``) only where
  Muraja's engine grades the set — qdat's false flags and Ship A's catches. Those are the only
  app-outcome verdict metrics (#245, architect r2); the remaining sets are report-only from the
  single stream.

Each set's items come from the set's own source, exactly as our models' decode reads it
(:func:`training.zip_screen_decode.recordings` for the benchmark, the edits and qdat;
:func:`training.truth_baseline.items_of` and the staged registry for Ship A;
:func:`training.qud_decode.recordings` for QUD's unread half). Every line is the shared
provenance plus its arms, one item per line, resumable by key. The identity the decodes rest
on is the gate-pinned one (:data:`training.zipformer_decode.IDENTITY_PATH`), written beside
each set's decodes by the shared loader.

Whole items are independent, so the batch decodes them in worker processes, one ONNX session
and one thread each, while the parent streams the audio (a QUD or qdat shard is downloaded,
read and deleted once — never staged).

The batch's first step is the timing probe (#245 architect r2, AC 6): 20 qdat items through
the ``checks-w1`` store at the batch's worker count, projected over every set's items. A
projection above 12 h stops the batch before any screen decode; the record says why, and the
module exits 3.

Usage (on the run box, from ``tools/``)::

    # the batch: the probe over 20 qdat items, then every set and store (exit 3 = past 12 h)
    python -m training.zipformer_screen_decode --batch \\
        --probe-record /root/scratch/zipformer/screen/probe.json \\
        --out-dir /root/scratch/zipformer/screen/decodes \\
        --work-dir /root/scratch/zipformer/data \\
        --edits-dir /root/scratch/issue-117/stage/edits/audio \\
        --qdat-cache /root/scratch/zipformer/screen/qdat_cache \\
        --ship-a-dir /root/scratch/issue-90/clips \\
        --qud-cache /root/scratch/zipformer/screen/qud_cache

    # one job, resumed by key (a smoke, or a relaunch after a stop)
    python -m training.zipformer_screen_decode --set qdat --store w1 \\
        --work-dir /root/scratch/zipformer/data --qdat-cache ... \\
        --out-dir /root/scratch/zipformer/screen/decodes --limit 2
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from tadabur.audio import TARGET_SAMPLE_RATE
from tadabur.tlog_index import write_json
from training.external_decode import PROGRESS_EVERY, Recording, append_timing, resume_keys
from training.truth_baseline import Store
from training.zipformer_decode import (
    CHECKS_W1,
    CHECKS_W1_WINDOWS,
    IDENTITY_PATH,
    ZIPFORMER_DIR,
    ZIPFORMER_STORES,
    ZIPFORMER_W1_STORES,
    arms_decoded_samples,
    load_zipformer,
)

#: The sets this batch decodes, and the two stores: the single stream, and the warm-started
#: per-check arms Muraja's engine grades.
BENCHMARK = "benchmark"
EDITS = "edits"
QDAT = "qdat"
SHIP_A = "ship-a"
QUD_OTHER = "qud-other"
DECODE = "decode"
W1 = "w1"

#: The sets, in the order the batch decodes them (:data:`JOBS` walks this order).
SETS = (BENCHMARK, EDITS, QDAT, SHIP_A, QUD_OTHER)
#: Which stores a set decodes: the per-check arms only on the sets the Muraja harness grades —
#: qdat's all-cause false flags and Ship A's catches (owner-confirmed amendment 2026-10-31).
SET_STORES = {
    BENCHMARK: (DECODE,),
    EDITS: (DECODE,),
    QDAT: (DECODE, W1),
    SHIP_A: (DECODE, W1),
    QUD_OTHER: (DECODE,),
}
STORE_MAPS: dict[str, Mapping[str, Store]] = {DECODE: ZIPFORMER_STORES, W1: ZIPFORMER_W1_STORES}

#: Every decode job: a set and the store it writes, in batch order.
JOBS = tuple((set_name, store) for set_name in SETS for store in SET_STORES[set_name])
#: The flag each set's own source directory arrives by. A set never defaults another's
#: directory: a wrong one decodes the wrong clips instead of failing.
SET_DIRECTORY = {BENCHMARK: "work_dir", EDITS: "edits_dir", QDAT: "qdat_cache",
                 SHIP_A: "ship_a_dir", QUD_OTHER: "qud_cache"}

#: The timing probe: qdat items through the verdict arm, and the batch's horizon (AC 6).
PROBE_ITEMS = 20
HORIZON_HOURS = 12.0
#: The probe's exit when the projection exceeds the horizon: the batch must not decode.
PROBE_STOP_EXIT = 3

#: The committed single-stream decodes the probe reads item lengths from, under the card's own
#: committed files. The sets are frozen, so a missing one is a broken checkout, not a skip.
COMMITTED_DECODES = ZIPFORMER_DIR / "decodes"
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)


# --- each set's items, from the set's own source -------------------------------------------


def ship_a_recordings(audio_dir: Path) -> Iterator[Recording]:
    """Ship A's verified sites, grouped into their distinct items, from the staged clips.

    One item is a clip and a sample span; two sites on one span (a mark read as two directions)
    are one item, as :func:`training.truth_baseline.items_of` takes them. Every clip is
    verified against the registry before it is read.
    """
    from tadabur.staged_audio import load_staged_clips, verify_staged
    from tadabur.truth_sites import TRUTH_SITES_DIR, load_truth_sites
    from training.decode_evalset import read_clip_audio
    from training.truth_baseline import items_of
    from training.truth_scorer import item_key

    path = TRUTH_SITES_DIR / "ship_a_verified.jsonl"
    sites = load_truth_sites(path)
    registry = load_staged_clips()
    items = items_of(sites, registry)
    site_ids: dict[str, list[str]] = {}
    for site in sites:
        site_ids.setdefault(item_key(site), []).append(site.site_id)
    by_clip: dict[str, list[tuple[str, object]]] = {}
    for key, site in sorted(items.items()):
        by_clip.setdefault(site.audio_filename, []).append((key, site))
    for clip, clip_items in sorted(by_clip.items()):
        keys = tuple(key for key, _ in clip_items)

        def load(clip=clip, clip_items=clip_items):
            verify_staged(registry[clip], audio_dir)
            samples = read_clip_audio(audio_dir, clip)
            return samples, [{
                "key": key, "audio_filename": clip, "audio_sha256": site.audio_sha256,
                "start_sample": site.start_sample, "end_sample": site.end_sample,
                "site_ids": site_ids[key],
            } for key, site in clip_items]

        yield Recording(keys, load)


def qud_recordings(cache_dir: Path) -> Iterator[Recording]:
    """Every QUD segment of the pilot's two mushafs, shard by shard, as the pilot decodes read it.

    The unread half (:func:`training.consonant_readout.load_qud_complement`) is the batch's
    keep set; the recording covers every segment of an ayah, as the source yields it.
    """
    from tadabur.qud_overlap import PILOT
    from training.qud_decode import mushaf_shards, recordings

    for slug in PILOT:
        for shard in mushaf_shards(slug):
            yield from recordings(slug, shard, cache_dir)


def recordings(set_name: str, directory: Path) -> Iterator[Recording]:
    """The set's items, from its own source directory."""
    from training import zip_screen_decode

    if set_name == BENCHMARK:
        return zip_screen_decode.benchmark_recordings(directory)
    if set_name == EDITS:
        return zip_screen_decode.edit_recordings(directory)
    if set_name == QDAT:
        return zip_screen_decode.qdat_recordings(directory)
    if set_name == SHIP_A:
        return ship_a_recordings(directory)
    return qud_recordings(directory)


def keep_keys(set_name: str) -> set[str] | None:
    """The items of a set that are part of it, when its source yields more: QUD's other half."""
    if set_name != QUD_OTHER:
        return None
    from training.consonant_readout import load_qud_complement

    return load_qud_complement()


# --- the batch's arithmetic ----------------------------------------------------------------


def arms_of(set_name: str, store: str) -> list[str]:
    """The arms one job decodes."""
    return sorted({arm for held in STORE_MAPS[store].values() for arm in held.protocols})


def decode_seconds(set_name: str, store: str, lengths: Sequence[int]) -> float:
    """The seconds of audio the job's arms hand the model, over items of ``lengths`` samples."""
    arms = arms_of(set_name, store)
    return (sum(arms_decoded_samples(arms, length) for length in lengths)
            / TARGET_SAMPLE_RATE)


def _decoded_lengths(path: Path, keep: set[str] | None = None) -> list[int]:
    """Each committed decode line's item length, in file order; QUD's half filtered by key."""
    lengths: list[int] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if keep is not None and record["key"] not in keep:
                continue
            lengths.append(record["end_sample"] - record["start_sample"])
    return lengths


def item_lengths(set_name: str) -> list[int]:
    """Every item of a setup set's length in samples, from a committed decode of the same items
    (the probe has no audio to measure and must not download a shard to project)."""
    if set_name == BENCHMARK:
        return _decoded_lengths(COMMITTED_DECODES / "benchmark.base.jsonl.gz")
    if set_name == EDITS:
        return _decoded_lengths(COMMITTED_DECODES / "edits.base.jsonl.gz")
    if set_name == QDAT:
        # base's #156 qdat decode: the same clips the batch reads, with their sample spans
        from tadabur.nonpro_sources import QDAT as QDAT_DATASET
        from training.nonpro_eval import decodes_file

        return _decoded_lengths(decodes_file(QDAT_DATASET, "base"))
    if set_name == SHIP_A:
        from tadabur.truth_sites import TRUTH_SITES_DIR, load_truth_sites
        from training.truth_scorer import item_key

        spans = {item_key(site): site.end_sample - site.start_sample
                 for site in load_truth_sites(TRUTH_SITES_DIR / "ship_a_verified.jsonl")}
        return [spans[key] for key in sorted(spans)]
    from tadabur.qud_overlap import PILOT
    from training.qud_pilot import decodes_file

    keep = keep_keys(set_name)
    return [length for slug in PILOT
            for length in _decoded_lengths(decodes_file(slug, "base"), keep)]


def project(timing: Mapping, decoded_by_job: Mapping[str, float],
            *, horizon_hours: float = HORIZON_HOURS) -> dict:
    """The batch's projection from one measured job's throughput.

    ``timing`` is the probe's run record (decoded audio seconds and wall seconds); each job's
    decoded audio is in seconds too, and its projected wall is that divided by the measured
    throughput, at the same worker count.
    """
    if timing["wall_seconds"] <= 0 or timing["decoded_audio_seconds"] <= 0:
        raise ValueError("the probe measured no decode: no projection from an empty run")
    rate = timing["decoded_audio_seconds"] / timing["wall_seconds"]
    jobs = {name: {"decoded_audio_seconds": round(seconds, 1),
                   "projected_seconds": round(seconds / rate, 1)}
            for name, seconds in decoded_by_job.items()}
    total = round(sum(job["projected_seconds"] for job in jobs.values()), 1)
    return {
        "decoded_audio_per_second": rate,
        "jobs": jobs,
        "projected_seconds": total,
        "projected_hours": round(total / 3600, 2),
        "horizon_hours": horizon_hours,
        "stop": total > horizon_hours * 3600,
    }


def probe_projection(timing: Mapping, *, horizon_hours: float = HORIZON_HOURS) -> dict:
    """Every job's decoded audio from its set's items, and the projection over the batch."""
    return project(timing, {f"{set_name}/{store}": decode_seconds(set_name, store,
                                                                  item_lengths(set_name))
                            for set_name, store in JOBS},
                   horizon_hours=horizon_hours)


# --- the decode loop -----------------------------------------------------------------------


def pending_items(source: Iterable[Recording], done: set[str],
                  keep: set[str] | None = None) -> Iterator[tuple[dict, np.ndarray]]:
    """Each item still to decode, as its provenance record and its sample span.

    A recording with no pending item is never loaded (its shard is never downloaded). ``keep``
    selects the items of a source that holds more than the set (QUD's unread half), and is
    applied before the recording is loaded: a source whose every segment sits in the read half
    or is already decoded is skipped whole, never downloaded and filtered afterwards.
    """
    try:
        for recording in source:
            wanted = set(recording.keys) if keep is None else set(recording.keys) & keep
            if not wanted - done:
                continue
            samples, records = recording.load()
            for record in records:
                key = record["key"]
                if key in done or (keep is not None and key not in keep):
                    continue
                yield record, samples[record["start_sample"]:record["end_sample"]]
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()  # a stopped stream deletes the shard it was reading


_WORKER: dict[str, object] = {}


def start_worker(files, scale: float, stores: Mapping[str, Store]) -> None:
    """One worker's session and decoder: every decode job starts its processes here."""
    import torch

    torch.set_num_threads(1)
    _WORKER["decoder"] = load_zipformer(files, scale=scale)
    _WORKER["stores"] = stores


def _decode_item(record: dict, span: np.ndarray) -> dict:
    """One item's line: its provenance and the store's arms."""
    decoder = _WORKER["decoder"]
    line = dict(record)
    samples = np.ascontiguousarray(span, dtype=np.float32)
    for store in _WORKER["stores"].values():
        line.update(store.decode(decoder, samples))
    return line


def pinned(work_dir: Path):
    """The fetched files at the identity's pinned scale, refusing weights it did not pin."""
    from training.zipformer_run import files_from_record, load_fetch

    files = files_from_record(work_dir, load_fetch(work_dir))
    identity = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))["identity"]
    scale = identity["sample_scale"]
    if files.identity(scale=scale)["files"] != identity["files"]:
        raise ValueError(f"{IDENTITY_PATH} pinned other weights than the fetched files")
    return files, scale


def decodes_path(out_dir: Path, set_name: str, store: str) -> Path:
    suffix = "" if store == DECODE else f".{store}"
    return out_dir / f"{set_name}.zipformer{suffix}.jsonl"


def identity_path(out_dir: Path, set_name: str, store: str) -> Path:
    suffix = "" if store == DECODE else f".{store}"
    return out_dir / f"{set_name}.zipformer{suffix}.identity.json"


def decode(source: Iterable[Recording], *, set_name: str, store: str, out_dir: Path,
           files, scale: float, workers: int, keep: set[str] | None = None,
           limit: int | None = None) -> dict:
    """Decode every pending item of ``source`` into the job's file, resuming by key."""
    path = decodes_path(out_dir, set_name, store)
    done = resume_keys(path)
    stores = STORE_MAPS[store]
    items = 0
    audio_seconds = decoded_seconds = 0.0
    started = time.time()
    source = pending_items(source, done, keep)
    with ProcessPoolExecutor(max_workers=workers, initializer=start_worker,
                             initargs=(files, scale, stores)) as pool, \
            open(path, "a", encoding="utf-8") as out:

        def collect(inflight):
            finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in finished:
                line = future.result()
                inflight.discard(future)
                out.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
                out.flush()

        inflight = set()
        try:
            for record, span in itertools.islice(source, limit):
                items += 1
                audio_seconds += len(span) / TARGET_SAMPLE_RATE
                decoded_seconds += decode_seconds(set_name, store, [len(span)])
                inflight.add(pool.submit(_decode_item, record, span))
                while len(inflight) > 2 * workers:
                    collect(inflight)
                if items % PROGRESS_EVERY == 0:
                    print(f"{set_name} {store}: {items} items, {audio_seconds:.0f} s of audio "
                          f"in {time.time() - started:.0f} s", flush=True)
            while inflight:
                collect(inflight)
        finally:
            source.close()  # a stopped stream deletes the shard it was reading
    return {"items": items, "audio_seconds": round(audio_seconds, 1),
            "decoded_audio_seconds": round(decoded_seconds, 1),
            "wall_seconds": round(time.time() - started, 1), "workers": workers}


def run(set_name: str, store: str, out_dir: Path, *, directory: Path, work_dir: Path,
        workers: int, limit: int | None = None) -> dict:
    """One decode job: the set's items through its store, its identity and its timing."""
    if store not in SET_STORES[set_name]:
        raise ValueError(f"{set_name} does not decode the {store!r} store")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files, scale = pinned(work_dir)
    load_zipformer(files, scale=scale, identity_path=identity_path(out_dir, set_name, store))
    timing = decode(recordings(set_name, directory),
                    set_name=set_name, store=store, out_dir=out_dir, files=files, scale=scale,
                    workers=workers, keep=keep_keys(set_name), limit=limit)
    suffix = "" if store == DECODE else f".{store}"
    append_timing(out_dir / f"{set_name}.zipformer{suffix}.timing.jsonl",
                  {"set": set_name, "store": store, "model": "zipformer",
                   "decoded_audio_unit": "seconds", **timing})
    print(json.dumps({"set": set_name, "store": store, **timing}), flush=True)
    return timing


def probe(work_dir: Path, cache_dir: Path, record_path: Path, *,
          workers: int, limit: int = PROBE_ITEMS) -> dict:
    """The timing probe: ``limit`` qdat items through the verdict arm, and the projection.

    The probe decodes into a scratch directory, never the batch's own decode files: a relaunch
    whose qdat job is already complete must still measure throughput, where resuming onto the
    finished file would leave a zero-item run with nothing to project. The probe's decodes are
    discarded, which is why a stopped batch has decoded no set.
    """
    files, scale = pinned(work_dir)
    with tempfile.TemporaryDirectory(prefix="zipformer-probe-") as scratch:
        load_zipformer(files, scale=scale)
        timing = decode(recordings(QDAT, cache_dir),
                        set_name=QDAT, store=W1, out_dir=Path(scratch), files=files,
                        scale=scale, workers=workers, limit=limit)
    record = {
        "issue": 245,
        "step": "timing-probe",
        "rule": "stop before any screen decode if the projected decode of every set exceeds "
                "12 h (architect r2, AC 6); the rate is the checks-w1 store at this worker count",
        "decoded_audio_unit": "seconds",
        "probe": {"set": QDAT, "store": W1, **timing},
        **probe_projection(timing),
    }
    Path(record_path).parent.mkdir(parents=True, exist_ok=True)
    write_json(record, record_path)
    print(json.dumps({key: record[key] for key in
                      ("decoded_audio_per_second", "projected_hours", "horizon_hours", "stop")}),
          flush=True)
    return record


def directories(args: argparse.Namespace) -> dict[str, Path]:
    """Each set's own source directory from the run's flags; a missing one stops the batch."""
    dirs = {set_name: getattr(args, flag) for set_name, flag in SET_DIRECTORY.items()}
    missing = [f"--{SET_DIRECTORY[set_name].replace('_', '-')}"
               for set_name, path in dirs.items() if path is None]
    if missing:
        raise SystemExit(f"every set needs its own directory: pass {' '.join(missing)}")
    return dirs


def batch(out_dir: Path, *, dirs: Mapping[str, Path], probe_record: Path,
          workers: int) -> int:
    """The probe, then every job; a projection past the horizon stops before any decode.

    The probe runs in a scratch directory, so it neither resumes a finished qdat job nor adds
    its items to the batch's files; a stop at that point has decoded no set.
    """
    record = probe(dirs[BENCHMARK], dirs[QDAT], probe_record, workers=workers)
    if record["stop"]:
        print(f"STOP: the probe projects {record['projected_hours']:.1f} h of decode, past the "
              f"{record['horizon_hours']:.0f} h horizon; no set was decoded", flush=True)
        return PROBE_STOP_EXIT
    for set_name, store in JOBS:
        run(set_name, store, out_dir, directory=dirs[set_name], work_dir=dirs[BENCHMARK],
            workers=workers)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the batch, or one job; the batch's probe stops it with :data:`PROBE_STOP_EXIT`."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", action="store_true",
                        help="the timing probe, then every set and store")
    parser.add_argument("--set", choices=SETS, help="the set to decode")
    parser.add_argument("--store", choices=tuple(STORE_MAPS), help="the arms to decode")
    parser.add_argument("--out-dir", type=Path, required=True, help="the decodes' directory")
    parser.add_argument("--work-dir", type=Path, help="the benchmark fetch's work directory")
    parser.add_argument("--edits-dir", type=Path, help="the synthetic edits' rendered audio")
    parser.add_argument("--qdat-cache", type=Path, help="qdat's shard cache")
    parser.add_argument("--ship-a-dir", type=Path, help="Ship A's staged clips")
    parser.add_argument("--qud-cache", type=Path, help="QUD's shard cache")
    parser.add_argument("--probe-record", type=Path, help="the batch's timing-probe record")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, help="decode only the first items of one job")
    args = parser.parse_args(argv)
    if args.batch:
        if args.probe_record is None:
            parser.error("the batch writes its timing probe: pass --probe-record")
        return batch(args.out_dir, dirs=directories(args), probe_record=args.probe_record,
                     workers=args.workers)
    if args.set is None or args.store is None:
        parser.error("pass --batch, or --set and --store")
    directory = getattr(args, SET_DIRECTORY[args.set])
    if directory is None:
        parser.error(f"{args.set} reads --{SET_DIRECTORY[args.set].replace('_', '-')}")
    if args.work_dir is None:
        parser.error("every decode reads the pinned fetch: pass --work-dir")
    run(args.set, args.store, args.out_dir, directory=directory, work_dir=args.work_dir,
        workers=args.workers, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
