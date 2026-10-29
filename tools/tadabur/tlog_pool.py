"""The tlog pool pipeline (#192): select rows, decode whole-clip up to D_max, identify, tree IO.

S6 (#191) is the first caller: it decodes the whole look tranche and a salted 2,000-row training
sample, identifies them and audits them. S7 decodes the whole pool through the same functions
and **reuses S6's committed decodes verbatim**, via ``decode_rows(..., already=...)``: a row
decoded again on another GPU can differ in the last bits, and the pool tree must agree with the
rows the audit heard. One tree per population, read back by name with :func:`read_ident_tree`.

The population, the sampling salt and the 5% rule belong to S6 and live in
:mod:`tadabur.tlog_ident_run`; the audit belongs to :mod:`tadabur.tlog_audit`. Nothing here
knows either.

S8 (#193) adds the pool's training side: :data:`TRAINING_PREDICATE` selects the usable
``train`` rows, :func:`load_training_rows` re-materialises them from the committed overlay and
verifies the row-set use's count and hash, and :func:`iter_training_rows` streams their audio
shard by shard, deleting each shard after use. The predicate has no recitation term: the
owner-approved gating amendment (2026-10-28 §4) leaves the training pool unfiltered by S6's
rule, so nothing decodes the pool to select it.

#224 gives the reader its order: :mod:`tadabur.tlog_schedule` freezes a seeded, 8-way
interleaved shuffle of both splits' shards, and :func:`iter_training_rows` requires it, so a
prefix of the stream is representative of ``tlog.training`` and the sorted walk cannot return.

Every tree is written by :func:`tadabur.tlog_index.write_shard` (sorted keys, ``mtime`` 0) and
every meta by :func:`tree_provenance`, so the cited commit survives the squash merge under
``refs/pull/<N>/head``. No audio is kept: a shard is streamed once and dropped.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.tashkeel_eval import write_text_atomically

from . import exposure, tlog_ident, tlog_schedule
from .nonpro_sources import audio_sha256
from .tlog_ident import BASE_MODEL, BASE_REVISION, NONE, identification_record, identify
from .tlog_index import head_sha, provenance, sha256_file, write_json, write_shard
from .tlog_pass1 import read_jsonl_gz, shard_rows

SCHEMA_VERSION = 1
#: The two decode modes a row may carry (r20 Q1).
WHOLE_CLIP = "whole_clip"
NOT_DECODED = "not_decoded:over_d_max"
#: The synthetic-audio ladder D_max is measured on, and the headroom every rung must keep.
LADDER = (60, 90, 120, 150, 180, 240, 300)
HEADROOM_GIB = 1.5
MEASURE_SEED = 19_100_620
#: The amendment's stop: more than this share of a population's usable rows above D_max halts.
D_MAX_STOP_SHARE = 0.01

#: The training reader's default run log, inside the cache directory (#224): one JSON line per
#: consumed row key, appended across restarts, so a run's actual prefix can be compared with
#: the plan. A run that must keep that evidence passes ``run_log`` beside the run; the cache
#: directory is scratch.
RUN_LOG_NAME = "consumed.jsonl.gz"

ROOT = Path(__file__).resolve().parents[2]
IDENT_DIR = Path(__file__).parent / "tlog" / "ident"
DECODES_DIR = IDENT_DIR / "decodes"
DECODE_LIMIT_PATH = IDENT_DIR / "decode_limit.json"

#: The tlog overlay's data (#193): the four per-row trees and the silence floors. A row-set
#: use cites the commit that last touched these, and the loader refuses an overlay that has
#: moved from it -- the overlay is the training pool's input, not a scratch tree.
OVERLAY_PATHS = (
    "tools/tadabur/tlog/index",
    "tools/tadabur/tlog/pass1",
    "tools/tadabur/tlog/split",
    "tools/tadabur/tlog/overlap",
    "tools/tadabur/tlog/silence_floors.json",
)
#: The training use's predicate, as committed in its row-set file: what
#: :func:`training_rows` applies and what the loader checks the file against. It has no
#: recitation term -- the owner-approved gating amendment (2026-10-28 §4) leaves the training
#: pool unfiltered by S6's ``non_recitation`` rule, so the pool is never decoded to filter it.
TRAINING_PREDICATE = {
    "text": "role == train and usable: not excluded (codec_ok and not silence, no S1 "
            "tarteel_v1 id overlap, no S4 eval_overlap and no S4 duplicate), the pool view of "
            "tadabur.tlog_overlap.iter_pool_rows",
    "code": "tadabur.tlog_pool.training_rows",
}


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


def require_d_max_share(over_d_max: int, total: int, label: str) -> None:
    """Stop when strictly more than :data:`D_MAX_STOP_SHARE` of ``label``'s rows exceed D_max.

    One definition for S6's whole pool and S7's per-tranche recheck (r24 item 6): a slice may
    not slip past the amendment's stop on the other slices' rows.
    """
    share = over_d_max / total if total else 0.0
    if share > D_MAX_STOP_SHARE:
        raise RuntimeError(f"{label}: {over_d_max} of {total} usable rows ({share:.2%}) are "
                           f"above D_max, over the {D_MAX_STOP_SHARE:.0%} stop; ask the owner")


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


def write_tree(directory: Path, splits: Mapping[str, Sequence[Mapping[str, Any]]],
               base: Mapping[str, Any]) -> None:
    """Write one population's tree: a shard per split, sorted by row key, and its meta.

    ``base`` is the meta's provenance; ``rows`` and ``census`` are computed from the rows
    written here, so the census of one tree is never read off another's. S6's look tree and
    S7's tranche trees share this, so the meta's shape changes in one place.
    """
    for split, rows in splits.items():
        write_shard(sorted(rows, key=row_key), Path(directory) / f"{split}.jsonl.gz")
    written = [row for rows in splits.values() for row in rows]
    write_json({**base, "census": tree_census(count_by_shard(written)),
                "rows": {split: len(rows) for split, rows in splits.items()}},
               Path(directory) / "meta.json")


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


# --- the training pool (#193, S8) --------------------------------------------------------------


def training_rows(pool_rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    """:data:`TRAINING_PREDICATE` applied: the usable ``train`` rows, in row-key order.

    ``usable`` is :func:`tadabur.tlog_overlap.iter_pool_rows`' flag -- no exclusion at all
    (codec, silence, S1's tarteel_v1 id overlap, S4's eval overlap or duplicate) -- so the
    predicate is one field, not a re-derivation of the union. Every other role is a sealed
    evaluation tranche and never trains.
    """
    return sorted((row for row in pool_rows if row["role"] == "train" and row["usable"]),
                  key=row_key)


def rows_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    """The row-set use's ``rows_sha256``: the SHA-256 of the sorted keys, ``split:shard:row``
    per line. Sorting the keys makes it independent of the input's order.
    """
    text = "".join(f"{split}:{shard}:{row}\n"
                   for split, shard, row in sorted(row_key(row) for row in rows))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def overlay_data_commit(repo: Path = ROOT) -> str:
    """The last commit touching :data:`OVERLAY_PATHS`: what a row-set use cites.

    The overlay's code SHAs (``tlog_tranches.overlay_commit``) say which builders wrote the
    trees; this says which commit holds the data, which is what a row can be re-materialised
    from. The branch is squash-merged, so the commit survives on ``main``.
    """
    result = _git(repo, "log", "-1", "--format=%H", "--", *OVERLAY_PATHS)
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        raise ValueError(f"{repo} has no commit touching the tlog overlay data")
    return commit


def require_overlay_at(commit: str, repo: Path = ROOT) -> None:
    """Refuse unless the working tree's overlay is exactly ``commit``'s.

    A modified, deleted or untracked overlay file refuses, as does any later commit that
    touched it: the rows a use file names are the rows of the overlay it cites.
    """
    if _git(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        raise ValueError(f"{repo} has no overlay commit {commit}")
    if _git(repo, "diff", "--quiet", commit, "--", *OVERLAY_PATHS).returncode != 0:
        raise ValueError(f"the tlog overlay has moved from {commit}")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *OVERLAY_PATHS)
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError(f"the tlog overlay has uncommitted changes from {commit}:\n"
                         f"{status.stdout}")


def empty_training_use(repo: Path = ROOT) -> exposure.RowSetExposure:
    """A row-set use with no rows: the shape a registry records for an unused use."""
    return exposure.RowSetExposure(overlay_data_commit(repo), dict(TRAINING_PREDICATE), 0,
                                   rows_digest([]))


def load_training_rows(registry: exposure.ExposureRegistry | None = None, *,
                       pool_rows: Iterable[Mapping[str, Any]] | None = None,
                       repo: Path = ROOT) -> list[dict]:
    """The committed ``tlog.training`` rows, re-materialised from the overlay and verified.

    The use file records the overlay commit, the predicate, the count and the hash; here the
    overlay of that commit is checked to be the working tree's, the predicate to be
    :data:`TRAINING_PREDICATE`, and the rows the predicate selects to have exactly the recorded
    count and hash. ``pool_rows`` is the pool read (default
    :func:`tadabur.tlog_overlap.iter_pool_rows`, one full pass); a use with no file raises
    :class:`tadabur.exposure.ExposureIncomplete` -- absent evidence is not an empty pool.
    """
    registry = exposure.load_registry() if registry is None else registry
    use = registry.row_sets.get(exposure.TLOG_TRAINING)
    if use is None:
        raise exposure.ExposureIncomplete(
            f"{exposure.TLOG_TRAINING} has no file in the registry: write it (empty if the "
            "training pool has no rows) before reading it")
    require_overlay_at(use.overlay_commit, repo)
    if dict(use.predicate) != TRAINING_PREDICATE:
        raise ValueError(f"{exposure.TLOG_TRAINING}: the recorded predicate is "
                         f"{use.predicate}, not {TRAINING_PREDICATE}")
    if pool_rows is None:
        from .tlog_overlap import iter_pool_rows

        pool_rows = iter_pool_rows()
    rows = training_rows(pool_rows)
    if len(rows) != use.count:
        raise ValueError(f"{exposure.TLOG_TRAINING}: the overlay has {len(rows)} training rows, "
                         f"not the recorded count {use.count}")
    if (digest := rows_digest(rows)) != use.rows_sha256:
        raise ValueError(f"{exposure.TLOG_TRAINING}: rows_sha256 is {digest}, not the recorded "
                         f"{use.rows_sha256}")
    return rows


def iter_training_rows(rows: Iterable[Mapping[str, Any]] | None = None, *,
                       cache_dir: Path, schedule: tlog_schedule.Schedule, stream=None,
                       load_audio=None, run_log: Path | None = None
                       ) -> Iterator[tuple[tuple[str, str, int], Any]]:
    """Stream the training rows' audio in the frozen shard schedule (#224).

    Yields ``((split, shard_file, row), samples)`` for exactly ``rows`` (default:
    :func:`load_training_rows`, the verified committed use), in
    :func:`tadabur.tlog_schedule.interleave_order`: the schedule's shards in its order,
    ``schedule.interleave_width`` of them open at once and one row per open shard per round, so
    consecutive rows come from different shards while K are open. Row order within a shard is
    kept; the downstream shuffle buffer does the rest. The schedule is a seeded draw over both
    splits, not a stratified one, so how early a split first appears is the draw's: on the
    committed file the first 10 scheduled shards are ``clean`` and the first ``unclean`` row is
    the 5,292nd (the guarantee is the interleave, not a first-round mix).

    ``schedule`` is required: a training call without one refuses, so the sorted
    ``(split, shard_file)`` walk cannot come back unnoticed. The schedule must cover exactly
    the shards of ``rows``, and its count and ``rows_sha256`` must be theirs -- a missing,
    duplicated or extra shard, or a schedule built for another overlay, refuses before any
    download.

    Each shard is read once and dropped -- the disk budget is at most ``interleave_width``
    shards -- and every open shard is closed, and so deleted, when the reader is exhausted, when
    it is closed (:meth:`close`), and when it raises (a checksum mismatch, a wanted row the
    shard lacks); a consumer that raises inside its own loop never resumes the reader, so it
    must hold it in :func:`contextlib.closing` (or close it in a ``finally``) or the download is
    dropped only when the generator is finalised. ``stream`` is the row source (default
    :func:`tadabur.tlog_pass1.shard_rows` at the pinned revision, downloading into
    ``cache_dir``) and ``load_audio`` the decoder (default :func:`default_audio`, 16 kHz mono
    float32). A row outside :data:`TRAINING_PREDICATE` refuses before any shard is read; a
    checksum mismatch refuses before that row is decoded, and a wanted row the shard lacks
    refuses after it.

    Every consumed row key is appended as one JSON line to ``run_log`` (default
    ``cache_dir / RUN_LOG_NAME``, a scratch directory -- a run that must keep its prefix
    evidence passes ``run_log`` beside the run), opened for append so a resumed run keeps its
    earlier lines. The summary -- one line per split with its rows, clips, windows and
    real-audio seconds -- is printed when the stream ends, when the reader is closed, and when
    it raises, so a run that stops at its window budget still reports the prefix it consumed.
    The first log line reports the rows, their windows per clip at the round-2 stream's 5 s
    window and 2.5 s hop, and the total, so a run's example count is visible before the first
    window is cut.
    """
    if schedule is None:
        raise ValueError("a training run needs the frozen shard schedule "
                         "(tadabur.tlog_schedule); the sorted walk is gone (#224)")
    rows = list(load_training_rows() if rows is None else rows)
    if outside := next((row for row in rows
                        if row["role"] != "train" or not row["usable"]), None):
        raise ValueError(f"{row_key(outside)} is not a training row: "
                         f"{TRAINING_PREDICATE['text']}")
    by_shard = tlog_schedule.rows_by_shard(rows, schedule)
    digest = rows_digest(rows)
    if (len(rows), digest) != (schedule.count, schedule.rows_sha256):
        raise ValueError(f"the schedule is for {schedule.count} rows with rows_sha256 "
                         f"{schedule.rows_sha256}, not these {len(rows)} rows with {digest}")
    windows = [_windows_per_clip(row) for row in rows]
    print(f"[training] {len(rows)} rows, windows per clip: "
          f"{sum(windows) / len(windows) if windows else 0.0:.1f} at a 5.0 s window and a "
          f"2.5 s hop, {sum(windows)} windows", flush=True)
    open_shard = (lambda shard: shard_rows(shard, Path(cache_dir))) if stream is None else stream
    load = default_audio if load_audio is None else load_audio

    def shard_stream(shard: tuple[str, str]):
        yield from _training_shard_rows(f"data/{shard[1]}.parquet", by_shard[shard],
                                        open_shard=open_shard, load=load)

    # The shard generators are created up front but start nothing until their first ``next()``,
    # so the downloads happen in slot order and at most ``interleave_width`` are open.
    shard_streams = {shard: shard_stream(shard) for shard in schedule.shards}
    log_path = Path(cache_dir) / RUN_LOG_NAME if run_log is None else Path(run_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = gzip.open(log_path, "at", encoding="utf-8")
    totals: dict[str, dict[str, int | float]] = {}
    try:
        for row, audio in tlog_schedule.interleave_streams(shard_streams, schedule):
            log.write(json.dumps({"split": row["split"], "shard_file": row["shard_file"],
                                  "row": row["row"]}) + "\n")
            tlog_schedule.add_split_row(totals, row)
            yield row_key(row), audio
    finally:
        log.close()
        for stream_ in shard_streams.values():
            stream_.close()
        # Printed here, not after the loop: a run that stops at its window budget never
        # exhausts the stream, and its consumed prefix is what the plan is compared with.
        for split, total in sorted(totals.items()):
            print(f"[training] {split}: {total['rows']} rows, {total['clips']} clips, "
                  f"{total['windows']} windows, {round(total['audio_seconds'], 3)} s of real "
                  f"audio", flush=True)


def _windows_per_clip(row: Mapping[str, Any]) -> int:
    """The round-2 stream's windows for one clip: 5 s windows at a 2.5 s hop."""
    from training.distill_data import SAMPLE_RATE, window_starts

    return len(window_starts(int(round(row["duration_s"] * SAMPLE_RATE))))


def _training_shard_rows(shard: str, wanted: Sequence[Mapping[str, Any]], *, open_shard,
                         load) -> Iterator[tuple[Mapping[str, Any], Any]]:
    """One shard's wanted rows, in row order, each with its decoded audio; the shard source is
    closed even on a raise."""
    by_index = {row["row"]: row for row in wanted}
    found: set[int] = set()
    items = open_shard(shard)
    try:
        for index, item in enumerate(items):
            row = by_index.get(index)
            if row is None:
                continue
            data = item["audio"]["bytes"]
            checksum = audio_sha256(data)
            if checksum != row["audio_sha256"]:
                raise ValueError(f"{row_key(row)} is {checksum}, not the recorded "
                                 f"{row['audio_sha256']}")
            found.add(index)
            yield row, load(data)
    finally:
        # The shard reader is a generator that deletes its download in a ``finally``, so the
        # close is what drops the shard when the consumer stops early or a row raises.
        if (close := getattr(items, "close", None)) is not None:
            close()
    if missing := sorted(set(by_index) - found):
        raise ValueError(f"{shard}: rows {missing} were not in the streamed shard")
