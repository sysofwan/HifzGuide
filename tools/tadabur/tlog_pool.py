"""The tlog pool pipeline (#192): select rows, decode whole-clip up to D_max, identify, tree IO.

S6 (#191) is the first caller: it decodes the whole look tranche and a salted 2,000-row training
sample, identifies them and audits them. S7 decodes the whole pool through the same functions
and **reuses S6's committed decodes verbatim**, via ``decode_rows(..., already=...)``: a row
decoded again on another GPU can differ in the last bits, and the pool tree must agree with the
rows the audit heard. One tree per population, read back by name with :func:`read_ident_tree`.

The population, the sampling salt and the 5% rule belong to S6 and live in
:mod:`tadabur.tlog_ident_run`; the audit belongs to :mod:`tadabur.tlog_audit`. Nothing here
knows either.

Every tree is written by :func:`tadabur.tlog_index.write_shard` (sorted keys, ``mtime`` 0) and
every meta by :func:`tree_provenance`, so the cited commit survives the squash merge under
``refs/pull/<N>/head``. No audio is kept: a shard is streamed once and dropped.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.tashkeel_eval import write_text_atomically

from . import tlog_ident
from .tlog_ident import BASE_MODEL, BASE_REVISION, NONE, identification_record, identify
from .tlog_index import head_sha, provenance, sha256_file, write_shard
from .tlog_pass1 import read_jsonl_gz, shard_rows

SCHEMA_VERSION = 1
#: The two decode modes a row may carry (r20 Q1).
WHOLE_CLIP = "whole_clip"
NOT_DECODED = "not_decoded:over_d_max"
#: The synthetic-audio ladder D_max is measured on, and the headroom every rung must keep.
LADDER = (60, 90, 120, 150, 180, 240, 300)
HEADROOM_GIB = 1.5
MEASURE_SEED = 19_100_620

ROOT = Path(__file__).resolve().parents[2]
IDENT_DIR = Path(__file__).parent / "tlog" / "ident"
DECODES_DIR = IDENT_DIR / "decodes"
DECODE_LIMIT_PATH = IDENT_DIR / "decode_limit.json"


def row_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    """The immutable row identity: ``(split, shard_file, row)``."""
    return row["split"], row["shard_file"], row["row"]


def rank(salt: str, *parts: object) -> bytes:
    """One row's salted rank: the draw order every S6/S7 sample and stratum uses."""
    payload = f"{salt}:" + ":".join(map(str, parts))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def decode_limit(path: Path | None = None) -> dict:
    """The committed ``decode_limit.json``, or a refusal.

    ``d_max_seconds`` must be a ladder rung, the rung must keep ``HEADROOM_GIB`` free, and no
    larger measured rung may keep that much: D_max is the largest rung that fits.
    """
    path = DECODE_LIMIT_PATH if path is None else Path(path)
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    d_max = record.get("d_max_seconds")
    if not isinstance(d_max, (int, float)) or isinstance(d_max, bool) or d_max <= 0:
        raise ValueError(f"{path} records no positive d_max_seconds")
    if d_max not in LADDER:
        raise ValueError(f"{path} records D_max {d_max}, not a ladder rung {LADDER}")
    if not isinstance(record.get("gpu"), str) or not record["gpu"]:
        raise ValueError(f"{path} does not record the measured GPU")
    rungs = record.get("rungs")
    if not isinstance(rungs, list) or [rung.get("seconds") for rung in rungs] != list(LADDER):
        raise ValueError(f"{path} does not measure the ladder {LADDER}")
    for rung in rungs:
        for field in ("peak_gib", "headroom_gib"):
            if not isinstance(rung.get(field), (int, float)) or isinstance(rung[field], bool):
                raise ValueError(f"{path} rung {rung.get('seconds')} records no {field}")
    chosen = next(rung for rung in rungs if rung["seconds"] == d_max)
    if chosen["headroom_gib"] < HEADROOM_GIB:
        raise ValueError(f"{path} D_max {d_max} keeps {chosen['headroom_gib']} GiB, "
                         f"under {HEADROOM_GIB}")
    if any(rung["headroom_gib"] >= HEADROOM_GIB
           for rung in rungs if rung["seconds"] > d_max):
        raise ValueError(f"{path} D_max {d_max} is not the largest rung with "
                         f"{HEADROOM_GIB} GiB headroom")
    return record


# --- selection ---------------------------------------------------------------------------------


def select_rows(rows: Iterable[Mapping[str, Any]], *, roles: Sequence[str],
                d_max: float) -> tuple[list[dict], list[dict]]:
    """The usable rows of ``roles`` at or below D_max, and the rows above it.

    A row above D_max comes back with ``decode_mode = NOT_DECODED`` and ``label_source = none``:
    it is never decoded, carries no label and belongs to no duration band.
    """
    selected: list[dict] = []
    over: list[dict] = []
    for row in rows:
        if row["role"] not in roles or not row["usable"]:
            continue
        if row["duration_s"] <= d_max:
            selected.append(dict(row))
        else:
            over.append({**dict(row), "decode_mode": NOT_DECODED, "label_source": NONE})
    return selected, over


def sample_rows(rows: Sequence[Mapping[str, Any]], *, count: int, salt: str) -> list[dict]:
    """The first ``count`` rows by the salted rank of their key: the seeded sample."""
    ranked = sorted(rows, key=lambda row: rank(salt, *row_key(row)))
    return [dict(row) for row in ranked[:count]]


# --- decode records and the one streaming pass -------------------------------------------------


def decode_record(*, split: str, shard_file: str, row: int, audio_path: str,
                  audio_sha256: str, duration_s: float, decode_mode: str,
                  spans: str | None, filename_label: str | None = None) -> dict:
    """One committed decode row; ``decode_mode`` is required, never defaulted."""
    if decode_mode not in (WHOLE_CLIP, NOT_DECODED):
        raise ValueError(f"{decode_mode!r} is not a decode mode")
    return {"schema_version": SCHEMA_VERSION, "split": split, "shard_file": shard_file,
            "row": row, "audio_path": audio_path, "audio_sha256": audio_sha256,
            "duration_s": duration_s, "decode_mode": decode_mode, "spans": spans,
            "filename_label": filename_label}


def _not_decoded_template() -> dict:
    """The identification-record key set for a row with no decode (one tiny database)."""
    reference = tlog_ident.AyahReference(1, 1, "بجح", (0, 3))
    database = tlog_ident.ReferenceDatabase.build([reference])
    return identification_record(identify("", database=database), decode_mode=NOT_DECODED)


def not_decoded_record(row: Mapping[str, Any], *, d_max: float) -> dict:
    """The identification-tree row for a clip above D_max: no decode, no label.

    The key set is exactly :func:`tadabur.tlog_ident.identification_record`'s; the caller adds
    the row's ``(split, shard_file, row)`` and provenance fields.
    """
    if row["duration_s"] <= d_max:
        raise ValueError(f"{row_key(row)} is within D_max {d_max}; it was decoded")
    record = _not_decoded_template()
    record.update({
        "decode_mode": NOT_DECODED, "filename_label": row.get("filename_label"),
        "label_source": NONE, "assigned_label": None, "id_floor_ok": None,
        "floor_rule": tlog_ident.floor_rule(),
    })
    return record


def default_audio(data: bytes):
    """The clip's mono 16 kHz samples, by the pinned decoder."""
    from .audio import decode_to_mono_16k_with_rate_and_decoder

    return decode_to_mono_16k_with_rate_and_decoder(data)[0]


def decode_shard(decoder, shard: str, wanted: Sequence[Mapping[str, Any]], *, d_max: float,
                 stream, load_audio=None) -> list[dict]:
    """Decode one shard's selected rows whole-clip, checking each audio SHA-256 first.

    ``wanted`` is this shard's selection rows. A wanted row above D_max raises before the
    stream is read, so a bad selection can never spend GPU time.
    """
    from .nonpro_sources import audio_sha256

    for row in wanted:
        if row["duration_s"] > d_max:
            raise ValueError(f"{row_key(row)} is {row['duration_s']} s, above D_max {d_max}")
    by_index = {row["row"]: row for row in wanted}
    load = default_audio if load_audio is None else load_audio
    records: dict[int, dict] = {}
    for index, item in enumerate(stream(shard)):
        row = by_index.get(index)
        if row is None:
            continue
        data = item["audio"]["bytes"]
        checksum = audio_sha256(data)
        if checksum != row["audio_sha256"]:
            raise ValueError(f"{row_key(row)} is {checksum}, not the recorded "
                             f"{row['audio_sha256']}")
        samples = load(data)
        records[index] = decode_record(
            split=row["split"], shard_file=row["shard_file"], row=row["row"],
            audio_path=row["audio_path"], audio_sha256=checksum,
            duration_s=row["duration_s"], decode_mode=WHOLE_CLIP,
            spans=decoder.decode_spans([samples])[0],
            filename_label=row.get("filename_label"))
    missing = set(by_index) - set(records)
    if missing:
        raise ValueError(f"{shard}: {sorted(missing)} were not in the streamed shard")
    return [records[row["row"]] for row in wanted]


def decoder():
    """The pinned base teacher, loaded through S5's own loader."""
    from .tlog_ident_decode import load_decoder

    return load_decoder()


def decode_rows(rows: Iterable[Mapping[str, Any]], *, d_max: float, cache_dir: Path,
                scratch: Path, already: Iterable[Mapping[str, Any]] = (),
                teacher=None) -> list[dict]:
    """Decode ``rows`` whole-clip, streaming each shard once; return them in row-key order.

    ``already`` is committed decode rows to reuse verbatim: a row S6 decoded is never decoded
    again, so a pool tree agrees bit for bit with the rows the audit heard. ``scratch`` keeps one
    file per streamed shard, stamped with the exact wanted keys and D_max, so a rerun reads what
    a previous one wrote instead of spending GPU time again; a stale scratch is refused.
    ``teacher`` is the model :func:`decoder` returned, when the caller loaded it itself (S7
    records its identity in the meta): loaded here otherwise, never a second copy.
    """
    wanted = sorted(rows, key=row_key)
    for row in wanted:
        if row["duration_s"] > d_max:
            raise ValueError(f"{row_key(row)} is {row['duration_s']} s, above D_max {d_max}")
    reused = {row_key(row): dict(row) for row in already}
    keys = [row_key(row) for row in wanted]
    by_shard: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in wanted:
        if row_key(row) not in reused:
            by_shard[(row["split"], row["shard_file"])].append(row)
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    stamp = {"d_max": d_max, "rows": len(keys),
             "keys_sha256": hashlib.sha256("".join(
                 f"{split}:{shard}:{number}\n" for split, shard, number in keys
             ).encode("utf-8")).hexdigest()}
    stamp_path = scratch / "selection.json"
    if stamp_path.is_file():
        if json.loads(stamp_path.read_text(encoding="utf-8")) != stamp:
            raise ValueError(f"{scratch} was built for a different selection; remove it")
    else:
        write_text_atomically(stamp_path, json.dumps(stamp, ensure_ascii=False, indent=1,
                                                     sort_keys=True) + "\n")
    decoded: dict[tuple[str, str, int], dict] = {}
    if by_shard:
        if teacher is None:
            teacher, _identity, _fingerprint = decoder()
        started = time.time()
        for position, ((split, stem), shard_wanted) in enumerate(sorted(by_shard.items()),
                                                                 start=1):
            target = scratch / split / f"{stem}.jsonl.gz"
            if target.is_file():
                for row in read_jsonl_gz(target):
                    decoded[row_key(row)] = row
                continue
            records = decode_shard(teacher, f"data/{stem}.parquet", shard_wanted, d_max=d_max,
                                   stream=lambda name: shard_rows(name, Path(cache_dir)))
            write_shard(records, target)
            for row in records:
                decoded[row_key(row)] = row
            print(f"[{position}/{len(by_shard)}] {split}/{stem}: {len(records)} rows, "
                  f"{time.time() - started:.0f} s", flush=True)
    decoded.update(reused)  # a committed decode is authoritative over a scratch copy
    missing = [key for key in keys if key not in decoded]
    if missing:
        raise ValueError(f"{len(missing)} wanted rows have no decode, e.g. {missing[0]}")
    return [decoded[key] for key in keys]


# --- identification ----------------------------------------------------------------------------


def with_keys(record: Mapping[str, Any], row: Mapping[str, Any]) -> dict:
    """S5's record plus the row's key, audio and duration facts: one ident-tree row."""
    return {**record, "split": row["split"], "shard_file": row["shard_file"],
            "row": row["row"], "audio_path": row["audio_path"],
            "audio_sha256": row["audio_sha256"], "duration_s": row["duration_s"],
            "filename_label": row.get("filename_label")}


def identified_row(database, bars, k: int, row: Mapping[str, Any]) -> dict:
    """Identify one decode row with S5's engine and key the result to it."""
    identification = identify(row["spans"], database=database,
                              filename_ayah=row.get("filename_label"), bars=bars, k=k)
    return with_keys(identification_record(identification,
                                           decode_mode=row["decode_mode"]), row)


def identify_rows(rows: Iterable[Mapping[str, Any]], *, database, bars,
                  k: int) -> list[dict]:
    """Identify decode rows with S5's engine, in the order given."""
    return [identified_row(database, bars, k, row) for row in rows]


def is_non_recitation(row: Mapping[str, Any]) -> bool:
    """Whether a decoded row fails the recitation rule ("speech present and the rule failed").

    The one definition of the S6 flag, from the ident row's ``id_floor_ok``: a clip above D_max
    is not decoded, so its verdict is ``None`` and it is neither flagged nor assigned a label.
    S7 writes this field, ``label_source`` and ``assigned_label`` to its tranche manifest.
    """
    return row.get("id_floor_ok") is False


def read_ident_tree(population: str, *, ident_dir: Path = IDENT_DIR) -> list[dict]:
    """One committed identification tree, by population name.

    A population with splits is a directory of one file per split (``look/``); a single-file
    population is ``<name>.jsonl.gz`` (``train_sample``). S7 reuses this reader for its tranche
    manifest: one tree per population, read by name.
    """
    base = Path(ident_dir) / population
    paths = (sorted(base.glob("*.jsonl.gz")) if base.is_dir()
             else [Path(ident_dir) / f"{population}.jsonl.gz"])
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"no identification tree for {population} under {ident_dir}")
    return [row for path in paths for row in read_jsonl_gz(path)]


# --- the per-tree census and provenance --------------------------------------------------------


def count_by_shard(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], int]:
    """Rows per ``(split, shard)``, the input of :func:`tree_census`."""
    counts: Counter = Counter()
    for row in rows:
        counts[(row["split"], row["shard_file"])] += 1
    return dict(counts)


def tree_census(counts: Mapping[tuple[str, str], int]) -> dict:
    """One tree's census over every tlog shard, zero-row shards included."""
    from .nonpro_sources import TLOG, tlog_split

    shards = [shard for shard in TLOG.shards
              if tlog_split(shard) in ("clean", "unclean")]
    return {f"{tlog_split(shard)}/{Path(shard).stem}": counts.get(
        (tlog_split(shard), Path(shard).stem), 0) for shard in shards}


def tree_provenance(pull_request: int | None = None) -> dict:
    """The model, revision and commit a tree was written from, with its PR ref (I4.2).

    The branch is squash-merged, so the cited ``code_sha`` survives only under
    ``refs/pull/<N>/head``; :func:`tadabur.tlog_index.provenance` records both.
    """
    return {"model": BASE_MODEL, "revision": BASE_REVISION,
            **provenance(head_sha(ROOT), pull_request),
            "decode_limit_sha256": sha256_file(DECODE_LIMIT_PATH),
            "calibration_sha256": sha256_file(tlog_ident.CALIBRATION_PATH)}


# --- measuring D_max on synthetic audio --------------------------------------------------------


def measure_decode_limit(*, seed: int = MEASURE_SEED) -> dict:
    """Decode seeded synthetic audio whole-clip at the production settings, rung by rung."""
    import numpy as np
    import torch

    teacher, _identity, _fingerprint = decoder()
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    rng = np.random.default_rng(seed)
    rungs = []
    for seconds in LADDER:
        samples = rng.standard_normal(int(seconds * 16_000)).astype(np.float32) * 0.01
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            teacher.decode_spans([samples])
            peak = torch.cuda.max_memory_reserved() / 2**30
            headroom = total_gib - peak
            oom = False
        except torch.cuda.OutOfMemoryError:
            peak, headroom, oom = total_gib, 0.0, True
            torch.cuda.empty_cache()
        rungs.append({"seconds": seconds, "peak_gib": round(peak, 3),
                      "headroom_gib": round(headroom, 3), "oom": oom})
        print(f"rung {seconds}s: peak {peak:.2f} GiB, headroom {headroom:.2f} GiB, "
              f"oom {oom}", flush=True)
    fitting = [rung for rung in rungs if rung["headroom_gib"] >= HEADROOM_GIB]
    if not fitting:
        raise RuntimeError("no ladder rung keeps the required headroom; stop for the owner")
    d_max = max(rung["seconds"] for rung in fitting)
    chosen = next(rung for rung in rungs if rung["seconds"] == d_max)
    return {"schema_version": SCHEMA_VERSION, "issue": 191, "gpu": torch.cuda.get_device_name(0),
            "measured_on": time.strftime("%Y-%m-%d"), "seed": seed, "dtype": "bf16",
            "batch_size": 1, "model": BASE_MODEL, "revision": BASE_REVISION,
            "ladder": list(LADDER), "rungs": rungs, "d_max_seconds": d_max,
            "headroom_gib": chosen["headroom_gib"]}
