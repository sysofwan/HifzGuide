"""tlog S6 (#191): the look and training-sample decodes, identification, 5% rule and audit.

This is the first content read of tlog. It follows the frozen
``docs/tlog-holdout-preregistration.md`` as amended by ``docs/tlog-s6-amendment.md``:
whole-clip decodes only, up to a GPU limit measured on synthetic audio and committed in
``tlog/ident/decode_limit.json`` before the first tlog decode; a salted 2,000-row training
sample; the 5% rule; and the owner's stratified listening audit.

The committed trees are one JSONL.gz per population and split (r20 Q2)::

    tools/tadabur/tlog/ident/decodes/look/{clean,unclean}.jsonl.gz
    tools/tadabur/tlog/ident/decodes/train_sample.jsonl.gz
    tools/tadabur/tlog/ident/look/{clean,unclean}.jsonl.gz
    tools/tadabur/tlog/ident/train_sample.jsonl.gz
    tools/tadabur/tlog/audit/worklist.jsonl
    tools/tadabur/tlog/audit/verdicts.jsonl

Every row is keyed ``(split, shard_file, row)`` and sorted by that key. The decode meta records
the model, revision, code SHA, calibration, decode-limit and amendment SHAs; the identification
tree mirrors it and adds the 5% rule. No tlog audio is committed.

Usage, from ``tools/`` (GPU box for ``limit`` and ``decode``)::

    python -m tadabur.tlog_ident_run limit --out tadabur/tlog/ident/decode_limit.json
    python -m tadabur.tlog_ident_run decode --cache-dir /root/scratch/tlog/s6/cache
    python -m tadabur.tlog_ident_run identify
    python -m tadabur.tlog_ident_run audit
    python -m tadabur.tlog_ident_run supplement  # only if verdicts predate the correction
    python -m tadabur.tlog_ident_run stage --audio-dir /root/scratch/tlog/s6/audio \\
        --cache-dir /root/scratch/tlog/s6/cache
    python -m tadabur.tlog_ident_run report
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.counterfactual_eval import wilson_interval
from training.tashkeel_eval import write_text_atomically

from . import tlog_ident
from .tlog_ident import (
    BASE_MODEL,
    BASE_REVISION,
    BOTH,
    FILENAME,
    IDENTIFIED,
    NONE,
    identification_record,
    identify,
    load_identifier,
)
from .tlog_index import head_sha, repo_relative, require_clean_tree, sha256_file
from .tlog_overlap import iter_pool_rows
from .tlog_pass1 import read_jsonl_gz, shard_rows

SCHEMA_VERSION = 1
#: The two decode modes a row may carry (r20 Q1).
WHOLE_CLIP = "whole_clip"
NOT_DECODED = "not_decoded:over_d_max"
#: The 5% rule and the measured-limit stop.
SAMPLE_SIZE = 2_000
FIVE_PERCENT_LIMIT = 0.05
D_MAX_STOP_SHARE = 0.01
HEADROOM_GIB = 1.5
#: The synthetic-audio ladder D_max is measured on (r20 Q1).
LADDER = (60, 90, 120, 150, 180, 240, 300)
MEASURE_SEED = 19_100_620
#: The draw salt for the training sample and the audit (pinned in the amendment).
SALT = "tlog-s6-2026-10-27-v1"
#: The owner's approval of the S6 amendment. The frozen amendment says "requested"; the
#: approval given on #191 is recorded in the ident metas and the generated audit doc.
AMENDMENT_APPROVAL = {"status": "given", "date": "2026-10-27",
                      "url": "https://github.com/sysofwan/HifzGuide/issues/191"
                             "#issuecomment-6434980858"}
#: The pre-registered audit strata and sizes (frozen table; architect r20 Q3).
AUDIT_SIZES = {"both": 40, "relabelled": 30, "non_recitation": 25, "silence": 15,
               "unclean identified": 25, "below-bar disagreement": 15}
#: At least this many silence items come from the unclean split (architect r13).
SILENCE_UNCLEAN = 8
#: The frozen verdict vocabulary (pre-registration "Audit (S6)"), and r20 Q4's table.
RIGHT = "right ayah"
OTHER = "other ayah"
SILENCE = "silence"
NOT_RECITATION = "not recitation"
UNCLEAR = "unclear"
VERDICTS = (RIGHT, OTHER, SILENCE, NOT_RECITATION, UNCLEAR)
#: Per-band reporting over the frozen durations; 60 s belongs to the third band (r20 Q1).
DURATION_BANDS = ("0-20", "20-60", "60-d_max")
#: The overlap disclosure (r20 Q6), read from ``overlap/meta.json`` at render time.
OUTSIDE_ASR_LIMIT = 0.02
#: Below this many decided items an interval is not read (amendment §4).
MIN_READABLE_N = 10
#: The S5b §9 rule, pre-stated before any verdict: at this many audited
#: ``non_recitation`` clips heard as recitation, the 5% consequence pauses.
FLOOR_PAUSE_AT = 3

ROOT = Path(__file__).resolve().parents[2]
IDENT_DIR = Path(__file__).parent / "tlog" / "ident"
DECODES_DIR = IDENT_DIR / "decodes"
AUDIT_DIR = Path(__file__).parent / "tlog" / "audit"
DECODE_LIMIT_PATH = IDENT_DIR / "decode_limit.json"
AMENDMENT_PATH = ROOT / "docs" / "tlog-s6-amendment.md"
DOC_PATH = ROOT / "docs" / "tlog-audit.md"
OVERLAP_META_PATH = Path(__file__).parent / "tlog" / "overlap" / "meta.json"
#: The dated clarification of the strata overlap (architect r22), cited by the report.
OVERLAP_NOTE_PATH = ROOT / "docs" / "tlog-s6-amendment-note-overlap.md"


@dataclass(frozen=True)
class Selection:
    """The look population, the salted training sample, and the rows above D_max."""

    look: tuple[dict, ...]
    train_sample: tuple[dict, ...]
    over_d_max: tuple[dict, ...]
    counts: dict[str, int]


@dataclass(frozen=True)
class AuditDraw:
    """The drawn audit items and each stratum's target/eligible/drawn/shortfall census."""

    items: tuple[dict, ...]
    strata: dict[str, dict[str, int]]


# --- the committed-limit and order guards ------------------------------------------------------


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


def require_decode_preconditions() -> dict:
    """Refuse a decode until the amendment and the measured limit are committed.

    The order is r20's: synthetic measurement, the amendment, then the decode. The decode script
    additionally refuses a dirty checkout, so the committed files are the ones cited.
    """
    if not AMENDMENT_PATH.is_file():
        raise FileNotFoundError(f"the S6 amendment {repo_relative(AMENDMENT_PATH)} is absent; "
                                f"commit it before any tlog decode")
    if not DECODE_LIMIT_PATH.is_file():
        raise FileNotFoundError(f"{repo_relative(DECODE_LIMIT_PATH)} is absent; measure and "
                                f"commit D_max before any tlog decode")
    return decode_limit()


# --- selection ---------------------------------------------------------------------------------


def _key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return row["split"], row["shard_file"], row["row"]


def _rank(salt: str, *parts: object) -> bytes:
    payload = f"{salt}:" + ":".join(map(str, parts))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _duration_map(pass1_dir: Path) -> dict[tuple[str, str, int], float]:
    durations: dict[tuple[str, str, int], float] = {}
    for path in sorted(Path(pass1_dir).glob("*/*.jsonl.gz")):
        split, stem = path.parent.name, path.name[: -len(".jsonl.gz")]
        for row in read_jsonl_gz(path):
            durations[(split, stem, row["row"])] = row["duration_s"]
    return durations


def pool_rows(index_dir=None, pass1_dir=None) -> Iterator[dict]:
    """Every frozen row joined to its pass-1 duration, shard by shard."""
    from .tlog_overlap import INDEX_DIR, OVERLAP_DIR, PASS1_DIR
    from .tlog_split import FLOORS_PATH

    index_dir = INDEX_DIR if index_dir is None else Path(index_dir)
    pass1_dir = PASS1_DIR if pass1_dir is None else Path(pass1_dir)
    durations = _duration_map(pass1_dir)
    for row in iter_pool_rows(index_dir=index_dir, pass1_dir=pass1_dir, floors_path=FLOORS_PATH):
        key = _key(row)
        if key not in durations:
            raise ValueError(f"{key} has no pass-1 duration")
        yield {**row, "duration_s": durations[key]}


def _not_decoded(row: Mapping[str, Any]) -> dict:
    return {**row, "decode_mode": NOT_DECODED, "label_source": NONE}


def select(*, d_max: float, count: int = SAMPLE_SIZE, salt: str | None = None,
           rows: Iterable[Mapping[str, Any]] | None = None) -> Selection:
    """The look population and the salted 2,000-row training sample, both bounded by D_max.

    Look is every usable ``look`` row; the sample is the first ``count`` usable ``train`` rows
    by salted rank of the row key. Rows above D_max are listed separately, carry
    ``decode_mode = NOT_DECODED`` and ``label_source = none``, and are never decoded.
    """
    salt = SALT if salt is None else salt
    source = pool_rows() if rows is None else rows
    look, train, over = [], [], []
    for row in source:
        if row["role"] not in ("look", "train"):
            continue
        if not row["usable"]:
            continue
        if row["duration_s"] <= d_max:
            (look if row["role"] == "look" else train).append(dict(row))
        else:
            over.append(_not_decoded(row))
    sample = sorted(train, key=lambda row: _rank(salt, *_key(row)))[:count]
    counts = {
        "look_rows": sum(1 for row in look) + sum(1 for row in over if row["role"] == "look"),
        "look_selected": len(look),
        "look_over_d_max": sum(1 for row in over if row["role"] == "look"),
        "train_rows": len(train) + sum(1 for row in over if row["role"] == "train"),
        "train_selected": len(sample),
        "train_over_d_max": sum(1 for row in over if row["role"] == "train"),
    }
    return Selection(look=tuple(sorted(look, key=_key)),
                     train_sample=tuple(sorted(sample, key=_key)),
                     over_d_max=tuple(sorted(over, key=_key)), counts=counts)


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
        raise ValueError(f"{_key(row)} is within D_max {d_max}; it was decoded")
    record = _not_decoded_template()
    record.update({
        "decode_mode": NOT_DECODED, "filename_label": row.get("filename_label"),
        "label_source": NONE, "assigned_label": None, "id_floor_ok": None,
        "floor_rule": tlog_ident.floor_rule(),
    })
    return record


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _default_audio(data: bytes):
    from .audio import decode_to_mono_16k_with_rate_and_decoder

    return decode_to_mono_16k_with_rate_and_decoder(data)[0]


def decode_shard(decoder, shard: str, wanted: Sequence[Mapping[str, Any]], *, d_max: float,
                 stream, load_audio=None) -> list[dict]:
    """Decode one shard's selected rows whole-clip, checking each audio SHA-256 first.

    ``wanted`` is this shard's selection rows. A wanted row above D_max raises before the
    stream is read, so a bad selection can never spend GPU time.
    """
    for row in wanted:
        if row["duration_s"] > d_max:
            raise ValueError(f"{_key(row)} is {row['duration_s']} s, above D_max {d_max}")
    by_index = {row["row"]: row for row in wanted}
    load = _default_audio if load_audio is None else load_audio
    records: dict[int, dict] = {}
    for index, item in enumerate(stream(shard)):
        row = by_index.get(index)
        if row is None:
            continue
        data = item["audio"]["bytes"]
        checksum = _sha256_bytes(data)
        if checksum != row["audio_sha256"]:
            raise ValueError(f"{_key(row)} is {checksum}, not the recorded "
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


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """One JSONL-gzip file with no timestamp: the same rows give the same bytes, always."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(gzip.compress(body, mtime=0))
    os.replace(temporary, path)


def _census(counts: Mapping[tuple[str, str], int]) -> dict:
    from .nonpro_sources import TLOG, tlog_split

    shards = [shard for shard in TLOG.shards
              if tlog_split(shard) in ("clean", "unclean")]
    return {f"{tlog_split(shard)}/{Path(shard).stem}": counts.get(
        (tlog_split(shard), Path(shard).stem), 0) for shard in shards}


def _provenance() -> dict:
    return {"model": BASE_MODEL, "revision": BASE_REVISION,
            "code_sha": head_sha(ROOT),
            "decode_limit_sha256": sha256_file(DECODE_LIMIT_PATH),
            "calibration_sha256": sha256_file(tlog_ident.CALIBRATION_PATH),
            "amendment_sha256": sha256_file(AMENDMENT_PATH)}


def _decoder():
    from .tlog_ident_decode import _decoder as load_decoder

    return load_decoder()


def _require_d_max_share(counts: Mapping[str, int]) -> None:
    """Stop before S7 when more than 1% of the usable pool is above D_max (r20 Q1)."""
    total = counts["look_rows"] + counts["train_rows"]
    over = counts["look_over_d_max"] + counts["train_over_d_max"]
    if total and over / total > D_MAX_STOP_SHARE:
        raise RuntimeError(f"{over} of {total} usable rows ({over / total:.2%}) are above "
                           f"D_max, over the {D_MAX_STOP_SHARE:.0%} stop; ask the owner")


def decode(*, cache_dir: Path, out_dir: Path = DECODES_DIR,
           scratch: Path | None = None, d_max: float | None = None) -> dict:
    """Stream every selected shard once, decode whole-clip, delete the shard, merge."""
    require_clean_tree(ROOT)
    limit = require_decode_preconditions()
    d_max = float(limit["d_max_seconds"] if d_max is None else d_max)
    selection = select(d_max=d_max)
    _require_d_max_share(selection.counts)
    sample_keys = {_key(row) for row in selection.train_sample}
    by_shard: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in (*selection.look, *selection.train_sample):
        by_shard[(row["split"], row["shard_file"])].append(row)
    scratch = Path(scratch) if scratch is not None else Path(cache_dir).parent / "decoded"
    scratch.mkdir(parents=True, exist_ok=True)
    stamp = {"salt": SALT, "d_max": d_max, **selection.counts}
    stamp_path = scratch / "selection.json"
    if stamp_path.is_file():
        if json.loads(stamp_path.read_text(encoding="utf-8")) != stamp:
            raise ValueError(f"{scratch} was built for a different selection; remove it")
    else:
        write_text_atomically(stamp_path,
                              json.dumps(stamp, ensure_ascii=False, indent=1,
                                         sort_keys=True) + "\n")
    decoder, _identity, _fingerprint = _decoder()
    written: dict[tuple[str, str], int] = {}
    started = time.time()
    for position, ((split, stem), wanted) in enumerate(sorted(by_shard.items()), start=1):
        target = scratch / split / f"{stem}.jsonl.gz"
        if target.is_file():
            written[(split, stem)] = len(read_jsonl_gz(target))
            continue
        shard = f"data/{stem}.parquet"
        records = decode_shard(decoder, shard, wanted, d_max=d_max,
                               stream=lambda name: shard_rows(name, Path(cache_dir)))
        _write_jsonl_gz(target, records)
        written[(split, stem)] = len(records)
        print(f"[{position}/{len(by_shard)}] {split}/{stem}: {len(records)} rows, "
              f"{time.time() - started:.0f} s", flush=True)
    look = {split: [] for split in ("clean", "unclean")}
    sample = []
    for (split, stem), _count in sorted(written.items()):
        for row in read_jsonl_gz(scratch / split / f"{stem}.jsonl.gz"):
            (sample if _key(row) in sample_keys else look[split]).append(row)
    for split, rows in look.items():
        _write_jsonl_gz(out_dir / "look" / f"{split}.jsonl.gz", sorted(rows, key=_key))
    _write_jsonl_gz(out_dir / "train_sample.jsonl.gz", sorted(sample, key=_key))
    # Each meta's census counts the rows written to its own tree: ``rows`` and the census
    # must agree, so the provenance of one tree is never read off another's.
    base = {**_provenance(), "schema_version": SCHEMA_VERSION,
            "selection": {"salt": SALT, **selection.counts}}
    look_census = _census(_count_by_shard(
        [row for split in ("clean", "unclean") for row in look[split]]))
    write_text_atomically(out_dir / "look" / "meta.json",
                          json.dumps({**base, "census": look_census,
                                      "rows": {split: len(rows)
                                               for split, rows in look.items()}},
                                     ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    write_text_atomically(out_dir / "train_sample.meta.json",
                          json.dumps({**base, "census": _census(_count_by_shard(sample)),
                                      "rows": len(sample)},
                                     ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    return {"look": {split: len(rows) for split, rows in look.items()},
            "train_sample": len(sample), "seconds": round(time.time() - started, 1)}


# --- duration bands and the 5% rule ------------------------------------------------------------


def band_of(duration_s: float, d_max: float) -> str | None:
    """The pre-registered band of one duration: ``0-20``, ``20-60``, ``60-d_max``."""
    if duration_s > d_max:
        return None
    if duration_s <= 20.0:
        return "0-20"
    if duration_s < 60.0:
        return "20-60"
    return "60-d_max"


def _non_recitation(row: Mapping[str, Any]) -> bool:
    if "non_recitation" in row:
        return bool(row["non_recitation"])
    return row.get("id_floor_ok") is False


def band_report(rows: Iterable[Mapping[str, Any]], d_max: float) -> dict[str, dict[str, int]]:
    """Rows and ``non_recitation`` counts per band; rows above D_max belong to none."""
    report = {name: {"rows": 0, "non_recitation": 0} for name in DURATION_BANDS}
    for row in rows:
        band = band_of(row["duration_s"], d_max)
        if band is None:
            continue
        report[band]["rows"] += 1
        if _non_recitation(row):
            report[band]["non_recitation"] += 1
    return report


def five_percent_rule(rows: Iterable[Mapping[str, Any]]) -> dict:
    """The sample's ``non_recitation`` rate and Wilson 95% outcome, triggered above 5%."""
    rows = list(rows)
    if not rows:
        raise ValueError("the 5% rule needs at least one sampled row")
    bad = sum(1 for row in rows if _non_recitation(row))
    low, high = wilson_interval(bad, len(rows))
    return {"n": len(rows), "non_recitation": bad, "rate": bad / len(rows),
            "wilson": [low, high], "limit": FIVE_PERCENT_LIMIT,
            "triggered": high > FIVE_PERCENT_LIMIT}


# --- identification ----------------------------------------------------------------------------


def identify_trees(*, decodes_dir: Path = DECODES_DIR, out_dir: Path = IDENT_DIR,
                   d_max: float | None = None) -> dict:
    """Identify the committed decodes with S5's engine and write the ident tree."""
    require_clean_tree(ROOT)
    limit = decode_limit()
    d_max = float(limit["d_max_seconds"] if d_max is None else d_max)
    database, bars, k = load_identifier()
    provenance = {**_provenance(), "amendment_approval": dict(AMENDMENT_APPROVAL)}
    selection = select(d_max=d_max)
    look_rows: dict[str, list[dict]] = {"clean": [], "unclean": []}
    for split in ("clean", "unclean"):
        for row in read_jsonl_gz(decodes_dir / "look" / f"{split}.jsonl.gz"):
            look_rows[split].append(_identified_row(database, bars, k, row))
    for row in selection.over_d_max:
        if row["role"] == "look":
            look_rows[row["split"]].append(
                _with_keys(not_decoded_record(row, d_max=d_max), row))
    sample = [_identified_row(database, bars, k, row)
              for row in read_jsonl_gz(decodes_dir / "train_sample.jsonl.gz")]
    for split in ("clean", "unclean"):
        _write_jsonl_gz(out_dir / "look" / f"{split}.jsonl.gz",
                        sorted(look_rows[split], key=_key))
    count_by_shard: Counter = Counter()
    for rows in look_rows.values():
        count_by_shard.update(_count_by_shard(rows))
    write_text_atomically(out_dir / "look" / "meta.json",
                          json.dumps({**provenance, "schema_version": SCHEMA_VERSION,
                                      "rows": {split: len(rows)
                                               for split, rows in look_rows.items()},
                                      "census": _census(count_by_shard)},
                                     ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    _write_jsonl_gz(out_dir / "train_sample.jsonl.gz", sorted(sample, key=_key))
    rule = five_percent_rule(sample)
    decode_meta = json.loads((decodes_dir / "train_sample.meta.json").read_text(encoding="utf-8"))
    meta = {**provenance, "schema_version": SCHEMA_VERSION, "rows": len(sample),
            "selection": decode_meta["selection"],
            "five_percent_rule": rule,
            "bands": band_report(sample, d_max),
            "census": _census(_count_by_shard(sample))}
    write_text_atomically(out_dir / "train_sample.meta.json",
                          json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    return {"look": {split: len(rows) for split, rows in look_rows.items()},
            "train_sample": len(sample), "five_percent_rule": rule}


def _count_by_shard(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], int]:
    counts: Counter = Counter()
    for row in rows:
        counts[(row["split"], row["shard_file"])] += 1
    return dict(counts)


def _with_keys(record: Mapping[str, Any], row: Mapping[str, Any]) -> dict:
    return {**record, "split": row["split"], "shard_file": row["shard_file"],
            "row": row["row"], "audio_path": row["audio_path"],
            "audio_sha256": row["audio_sha256"], "duration_s": row["duration_s"],
            "filename_label": row.get("filename_label")}


def _identified_row(database, bars, k: int, row: Mapping[str, Any]) -> dict:
    identification = identify(row["spans"], database=database,
                              filename_ayah=row.get("filename_label"), bars=bars, k=k)
    return _with_keys(identification_record(identification,
                                            decode_mode=row["decode_mode"]), row)


# --- the audit draw ----------------------------------------------------------------------------


def _floor_ok(row: Mapping[str, Any]) -> bool | None:
    """The recitation verdict on a committed ident row (``None`` off the decoded population)."""
    return row["id_floor_ok"]


#: Amendment §4 defines each stratum by its own predicate and sets no precedence between them: a
#: row satisfying two predicates is drawn in both and served once
#: (``docs/tlog-s6-amendment-note-overlap.md``).
def _matching_strata(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Every stratum predicate this row satisfies, in :data:`AUDIT_SIZES` order.

    ``below-bar disagreement`` needs ``label_source == "filename"``: S5 records ``both``
    exactly when the best class and the filename class are one text class, so a ``both`` row is
    never a disagreement even when its two label strings differ (repeated ayat, e.g. ``26:190``
    and ``26:8`` are class 2925), and ``identified`` belongs to the relabel strata.
    """
    if row.get("decode_mode") == NOT_DECODED:
        return ()
    floor_ok = _floor_ok(row)
    matched = set()
    if tuple(row.get("exclusions") or ()) == ("silence",):
        matched.add(SILENCE)
    if row.get("label_source") == BOTH:
        matched.add("both")
    if row["split"] == "clean" and row.get("filename_label") and \
            row.get("label_source") == IDENTIFIED:
        matched.add("relabelled")
    if row["split"] == "unclean" and row.get("label_source") == IDENTIFIED:
        matched.add("unclean identified")
    if row.get("decode_mode") == WHOLE_CLIP and floor_ok is False:
        matched.add("non_recitation")
    if (floor_ok is True and row.get("label_source") == FILENAME
            and row.get("class_label") != row.get("filename_label")):
        matched.add("below-bar disagreement")
    return tuple(stratum for stratum in AUDIT_SIZES if stratum in matched)


def _stratum_label(stratum: str, row: Mapping[str, Any]) -> str | None:
    """The ayah the stratum scores against (amendment §5)."""
    if stratum in ("relabelled", "unclean identified"):
        return row.get("assigned_label")
    if stratum in ("both", "below-bar disagreement"):
        return row.get("filename_label")
    return None


def _page_stratum(strata: Sequence[str], row: Mapping[str, Any]) -> str:
    """The one stratum whose label the served page treats as right for a shared row.

    The report never reads this field: it scores a row in every stratum it was drawn in. The
    page needs a single label under test for its right/other choice, and a labelled stratum's is
    the one the owner hears the filename with.
    """
    labelled = [stratum for stratum in strata if _stratum_label(stratum, row) is not None]
    return (labelled or list(strata))[0]


def _blocking_reason(row: Mapping[str, Any]) -> str:
    return "overlap" if row.get("overlaps_filename") else "delta"


def _stratum_members(records: Iterable[Mapping[str, Any]],
                     salt: str) -> dict[str, list[dict]]:
    """Each stratum's full eligible set, sorted by that stratum's salted rank.

    A row satisfying two predicates is in both lists: the strata overlap, and each draws from
    its own set.
    """
    members: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        strata = _matching_strata(row)
        if not strata:
            continue
        item = {**dict(row),
                "blocking_reason": (_blocking_reason(row)
                                    if "below-bar disagreement" in strata else None)}
        for stratum in strata:
            members[stratum].append(item)
    return {stratum: sorted(rows, key=lambda r: _rank(salt, stratum, *_key(r)))
            for stratum, rows in members.items()}


def _served_item(item: Mapping[str, Any], drawn_in: Iterable[str]) -> dict:
    """One worklist row carrying the strata it was drawn in, in the frozen table's order.

    ``floor_sample`` marks the §9 sample: the ``non_recitation`` draw is exactly the frozen
    full-frame prefix, and the fallback's supplement adds the prefix rows an earlier draw
    lacked, so every row this module draws in that stratum is a sample row. A row an earlier
    draw kept only to fill the 25 is never marked, and the report reports it without counting it.
    """
    strata = [stratum for stratum in AUDIT_SIZES if stratum in set(drawn_in)]
    return {**dict(item), "strata": strata, "stratum": _page_stratum(strata, item),
            "floor_sample": "non_recitation" in strata}


def audit_draw(records: Iterable[Mapping[str, Any]], *, sizes: Mapping[str, int] | None = None,
               salt: str | None = None) -> AuditDraw:
    """Draw each stratum's salted prefix; a shortfall is recorded and its slots go to ``both``.

    Each stratum draws from its own full eligible set, so one row can be drawn in two strata and
    is served once.
    """
    salt = SALT if salt is None else salt
    sizes = dict(AUDIT_SIZES if sizes is None else sizes)
    if set(sizes) - set(AUDIT_SIZES):
        raise ValueError(f"sizes name unknown strata: {sorted(set(sizes) - set(AUDIT_SIZES))}")
    members = _stratum_members(records, salt)
    drawn: dict[str, list[dict]] = {}
    census: dict[str, dict[str, int]] = {}
    shortfall = 0
    for stratum in AUDIT_SIZES:
        if stratum == "both":
            continue
        eligible = members.get(stratum, [])
        target = sizes.get(stratum, 0)
        chosen = (_draw_silence(eligible, target, salt) if stratum == SILENCE
                  else eligible[:target])
        drawn[stratum] = chosen
        missing = max(0, target - len(chosen))
        shortfall += missing
        census[stratum] = {"target": target, "eligible": len(eligible),
                           "drawn": len(chosen), "shortfall": missing}
    both_target = sizes.get("both", 0) + shortfall
    eligible = members.get("both", [])
    drawn["both"] = eligible[:both_target]
    census["both"] = {"target": sizes.get("both", 0), "eligible": len(eligible),
                      "drawn": len(drawn["both"]),
                      "shortfall": max(0, both_target - len(drawn["both"]))}
    served: dict[str, dict] = {}
    drawn_in: dict[str, list[str]] = defaultdict(list)
    for stratum in AUDIT_SIZES:
        for item in drawn[stratum]:
            drawn_in[_item_key(item)].append(stratum)
    for stratum in AUDIT_SIZES:
        for item in drawn[stratum]:
            key = _item_key(item)
            if key not in served:
                served[key] = _served_item(item, drawn_in[key])
    return AuditDraw(items=tuple(served.values()), strata=census)


def with_floor_supplement(records: Iterable[Mapping[str, Any]],
                          worklist: Sequence[Mapping[str, Any]],
                          *, salt: str | None = None) -> tuple[dict, ...]:
    """The architect r22 fallback: the worklist with its §9 sample marked and completed.

    The frozen prefix's rows the worklist already holds are marked, whatever stratum they were
    drawn for. The prefix rows it lacks are appended in salted rank order, for a short second
    sitting; the union is the same 25 as the corrected draw, so no frame weighting is needed.
    Rows an earlier draw kept only to fill the 25 stay unmarked: reported, never counted.
    """
    salt = SALT if salt is None else salt
    prefix = _stratum_members(records, salt).get("non_recitation", [])[
        :AUDIT_SIZES["non_recitation"]]
    sample = {_item_key(row) for row in prefix}
    present = {_item_key(row) for row in worklist}
    marked = [{**dict(row), "floor_sample": _item_key(row) in sample} for row in worklist]
    missing = [_served_item(row, ["non_recitation"]) for row in prefix
               if _item_key(row) not in present]
    return tuple(marked + missing)


def _draw_silence(members: list[dict], target: int, salt: str) -> list[dict]:
    """At least ``SILENCE_UNCLEAN`` unclean, then clean, then unclean again if clean is short."""
    if target == 0:
        return []
    unclean = sorted((r for r in members if r["split"] == "unclean"),
                     key=lambda r: _rank(salt, "silence", *_key(r)))
    clean = sorted((r for r in members if r["split"] != "unclean"),
                   key=lambda r: _rank(salt, "silence", *_key(r)))
    first = unclean[:SILENCE_UNCLEAN]
    remaining = target - len(first)
    rest_clean = clean[:remaining]
    remaining -= len(rest_clean)
    extra = unclean[len(first):len(first) + remaining]
    return first + rest_clean + extra


def label_under_test(row: Mapping[str, Any]) -> str | None:
    """The ayah ``row["stratum"]`` scores against."""
    return _stratum_label(row["stratum"], row)


def verdict_outcome(item: Mapping[str, Any], verdict: Mapping[str, Any], *,
                    canonical: Callable[[str], str | None] | None = None) -> dict:
    """Score one verdict against the stratum's label under test (r20 Q4).

    ``unclear`` is an undecided outcome for every stratum: not hearing a clip is not a measured
    wrong ayah. When ``canonical`` is given (the report has the reference database), the heard
    ayah is compared by text class, so an identical recitation of a repeated ayah counts as the
    same ayah; without it, labels are compared literally.
    """
    heard = verdict.get("verdict")
    if heard not in VERDICTS:
        raise ValueError(f"{heard!r} is not one of {VERDICTS}")
    recitation = heard in (RIGHT, OTHER)
    label = label_under_test(item)
    heard_ayah = verdict.get("ayah")

    def same(first: object, second: object) -> bool | None:
        if first is None or second is None:
            return None
        if canonical is None:
            return first == second
        one, two = canonical(str(first)), canonical(str(second))
        return None if one is None or two is None else one == two

    if heard == UNCLEAR:
        return {"label_under_test": label, "correct": None, "best_correct": None,
                "recitation": False}
    if label is not None:
        matched = same(heard_ayah, label)
        correct = (heard == RIGHT) if matched is None else (recitation and matched)
        best_correct = None
        if item["stratum"] == "below-bar disagreement":
            best = same(heard_ayah, item.get("class_label"))
            best_correct = False if best is None else (recitation and best)
        return {"label_under_test": label, "correct": correct,
                "best_correct": best_correct, "recitation": recitation}
    if item["stratum"] == "non_recitation":
        return {"label_under_test": None, "correct": not recitation, "best_correct": None,
                "recitation": recitation}
    if item["stratum"] == SILENCE:
        return {"label_under_test": None, "correct": heard == SILENCE,
                "best_correct": None, "recitation": recitation}
    raise ValueError(f"no verdict rule for stratum {item['stratum']!r}")


def _item_key(row: Mapping[str, Any]) -> str:
    return _sha256_bytes(f"{row['split']}:{row['shard_file']}:{row['row']}".encode())[:16]


def audit_payload(draw: AuditDraw, *, texts: Mapping[str, str] | None = None,
                  verdicts: Mapping[str, Mapping[str, Any]] | None = None,
                  canonical: Callable[[str], str | None] | None = None) -> dict:
    """The served payload: anonymous candidate texts, no stratum or attribution.

    With ``canonical``, labels that are the same text class are one candidate (the class's
    canonical label), so an identical recitation is not offered twice and either copy scores the
    same.
    """
    verdicts = verdicts or {}
    texts = texts or {}
    items = []
    for row in draw.items:
        key = _item_key(row)
        labels: list[str] = []
        for label in (row.get("filename_label"), row.get("class_label")):
            if not label:
                continue
            name = label if canonical is None else canonical(label)
            if name is not None and name not in labels:
                labels.append(name)
        labels.sort(key=lambda label: _rank(SALT, "candidate", key, label))
        candidates = [{"ayah": label, "text": texts.get(label, "")} for label in labels]
        saved = verdicts.get(key)
        items.append({"key": key, "candidates": candidates,
                      "choices": [RIGHT, OTHER, SILENCE, NOT_RECITATION, UNCLEAR],
                      "heard": saved.get("verdict") if saved else None,
                      "heard_ayah": (saved.get("heard_ayah", saved.get("ayah"))
                                     if saved else None),
                      "note": saved.get("note", "") if saved else ""})
    answered = sum(1 for item in items if item["heard"])
    return {"items": items, "progress": {"answered": answered, "total": len(items)}}


# --- staging, verdicts and the generated page --------------------------------------------------


def _silence_candidates() -> list[dict]:
    rows = []
    for row in pool_rows():
        if row["role"] == "look" and tuple(row["exclusions"]) == ("silence",):
            rows.append({**row, "label_source": NONE, "id_floor_ok": None,
                         "class_label": None, "assigned_label": None,
                         "overlaps_filename": None, "decode_mode": None})
    return rows


def audit_candidates(*, ident_dir: Path = IDENT_DIR) -> list[dict]:
    """The drawn population: ident look rows, ident sample rows, and silence-only look rows."""
    rows: list[dict] = []
    for split in ("clean", "unclean"):
        rows += [{**row, "population": "look", "role": "look"}
                 for row in read_jsonl_gz(ident_dir / "look" / f"{split}.jsonl.gz")]
    rows += [{**row, "population": "train_sample", "role": "train"}
             for row in read_jsonl_gz(ident_dir / "train_sample.jsonl.gz")]
    rows += [{**row, "population": "look"} for row in _silence_candidates()]
    return rows


def _wav_bytes(data: bytes) -> bytes:
    """The clip as 16 kHz mono PCM WAV, the format the listening page serves."""
    import io

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, _default_audio(data), 16_000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def stage(draw: AuditDraw, *, audio_dir: Path, cache_dir: Path) -> None:
    """Stream the drawn rows' shards, verify each audio SHA-256 and keep the drawn clips.

    A clip the previous worklist staged and this one does not is deleted, so a redraw leaves no
    orphan audio behind.
    """
    wanted: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for position, row in enumerate(draw.items):
        wanted[(row["split"], row["shard_file"])][row["row"]] = {**row, "index": position}
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    worklist: list[dict] = []
    for (split, stem), rows in sorted(wanted.items()):
        for index, item in enumerate(shard_rows(f"data/{stem}.parquet", Path(cache_dir))):
            row = rows.get(index)
            if row is None:
                continue
            data = item["audio"]["bytes"]
            checksum = _sha256_bytes(data)
            if checksum != row["audio_sha256"]:
                raise ValueError(f"{_key(row)} is {checksum}, not {row['audio_sha256']}")
            name = f"{_item_key(row)}_{checksum[:8]}.wav"
            (audio_dir / name).write_bytes(_wav_bytes(data))
            worklist.append({**{key: row[key] for key in (
                "population", "split", "shard_file", "row", "role", "audio_path",
                "audio_sha256", "duration_s", "filename_label", "assigned_label",
                "class_label", "label_source", "id_floor_ok", "overlaps_filename",
                "blocking_reason", "strata", "stratum", "floor_sample")},
                "local_audio": name})
    if len(worklist) != len(draw.items):
        raise ValueError(f"staged {len(worklist)} of {len(draw.items)} drawn items")
    kept = {row["local_audio"] for row in worklist}
    for previous in _read_jsonl(Path(AUDIT_DIR) / "worklist.jsonl"):
        if previous["local_audio"] not in kept:
            (audio_dir / previous["local_audio"]).unlink(missing_ok=True)
    _write_jsonl(Path(AUDIT_DIR) / "worklist.jsonl", worklist)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_verdicts(path: Path = AUDIT_DIR / "verdicts.jsonl") -> dict[str, dict]:
    path = Path(path)
    if not path.is_file():
        return {}
    verdicts: dict[str, dict] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = _item_key(row)
        if key in verdicts:
            raise ValueError(f"{path}:{lineno}: a second verdict for {key}")
        verdicts[key] = row
    return verdicts


def _precision(rows: Sequence[Mapping[str, Any]], verdicts: Mapping[str, Mapping[str, Any]],
               canonical: Callable[[str], str | None] | None = None) -> dict:
    scored = []
    for row in rows:
        verdict = verdicts.get(_item_key(row))
        if verdict is None:
            continue
        outcome = verdict_outcome(row, verdict, canonical=canonical)
        scored.append(outcome)
    decided = [outcome for outcome in scored if outcome["correct"] is not None]
    correct = sum(1 for outcome in decided if outcome["correct"])
    interval = list(wilson_interval(correct, len(decided))) if decided else None
    return {"n": len(scored), "decided": len(decided), "correct": correct,
            "precision": (correct / len(decided)) if decided else None,
            "wilson": interval,
            "recitation": sum(1 for outcome in scored if outcome["recitation"]),
            "best_correct": sum(1 for outcome in scored if outcome["best_correct"])}


def _interval_text(correct: int, n: int) -> str:
    """The amendment §4 rule: an interval at any n, unreadable below ten decided items."""
    if n == 0:
        return "—"
    if n < MIN_READABLE_N:
        return "n too small to read"
    low, high = wilson_interval(correct, n)
    return f"{low:.2%}–{high:.2%}"


def _wilson_cell(stats: Mapping[str, Any]) -> str:
    return _interval_text(stats["correct"], stats["decided"])


def _band_cell(stats: Mapping[str, Any]) -> str:
    if not stats["decided"]:
        return "—"
    if stats["decided"] < MIN_READABLE_N:
        return f"{stats['correct']}/{stats['decided']} (n too small to read)"
    return f"{stats['correct']}/{stats['decided']} ({stats['precision']:.0%})"


def _best_class_line(stats: Mapping[str, Any]) -> str:
    """Best-class precision over decided answers (review R2.2): unclear is not a miss."""
    return (f"Best-class suggestions heard as the recitation: **{stats['best_correct']} of "
            f"{stats['decided']} decided** ({stats['n']} answered; Wilson 95% "
            f"{_interval_text(stats['best_correct'], stats['decided'])}).")


def _stratum_views(worklist: Sequence[Mapping[str, Any]]) -> dict[str, list[dict]]:
    """Each stratum's rows as single-stratum views, so a shared row is scored in every one."""
    return {stratum: [{**row, "stratum": stratum} for row in worklist
                      if stratum in row["strata"]]
            for stratum in AUDIT_SIZES}


def _by_band(rows: Iterable[Mapping[str, Any]], d_max: float) -> dict[str, list[dict]]:
    bands: dict[str, list[dict]] = {band: [] for band in DURATION_BANDS}
    for row in rows:
        band = band_of(row["duration_s"], d_max)
        if band is not None:
            bands[band].append(dict(row))
    return bands


def _within_ayat(first: object, second: object, distance: int) -> bool:
    """Whether two ``surah:ayah`` labels are within ``distance`` ayat of one another."""
    try:
        surah_a, ayah_a = (int(part) for part in str(first).split(":"))
        surah_b, ayah_b = (int(part) for part in str(second).split(":"))
    except ValueError:
        return False
    return surah_a == surah_b and 0 < abs(ayah_a - ayah_b) <= distance


def _ident_rows(ident_dir: Path) -> list[dict]:
    """The committed ident trees, each row tagged with the population it came from."""
    look = [{**row, "population": "look"} for split in ("clean", "unclean")
            for row in read_jsonl_gz(Path(ident_dir) / "look" / f"{split}.jsonl.gz")]
    sample = [{**row, "population": "train_sample"}
              for row in read_jsonl_gz(Path(ident_dir) / "train_sample.jsonl.gz")]
    return look + sample


def _floor_frame(ident_dir: Path) -> dict[str, int]:
    """The frame the ``non_recitation`` stratum samples.

    Every decoded row that fails the floor is in the stratum (architect r22); those that also
    match their filename class are drawn in ``both`` too and scored in both, so the count here
    is disclosed next to the precision rather than removed from it.
    """
    failing = [row for row in _ident_rows(ident_dir)
               if row["decode_mode"] == WHOLE_CLIP and row["id_floor_ok"] is False]
    return {"failing": len(failing),
            "both": sum(1 for row in failing if row.get("label_source") == BOTH)}


def _floor_prefix(ident_dir: Path) -> list[dict]:
    """The §9 sample: the first 25 floor-failing rows by the frozen salt's rank (r22).

    The draw and the fallback write the worklist's ``floor_sample`` marker; this derivation
    from the committed ident trees is what binds that marker to the frozen rule
    (``test_the_committed_worklist_marks_the_frozen_floor_sample``).
    """
    failing = [row for row in _ident_rows(ident_dir)
               if row["decode_mode"] == WHOLE_CLIP and row["id_floor_ok"] is False]
    ranked = sorted(failing, key=lambda row: _rank(SALT, "non_recitation", *_key(row)))
    return ranked[:AUDIT_SIZES["non_recitation"]]


def _label_independent(ident_dir: Path) -> dict[str, float]:
    """r21 §2a/§2d: how the floor behaves on tlog, from the committed ident trees.

    The shortlist is label-free, so a best match landing on the labelled ayah is not caused by
    the label: at about 6,053 text classes that is not chance. Retention is the share of
    decoded clean look rows the floor keeps; the calibration's own retention comes from
    ``ident_calibration.json``.
    """
    decoded = [row for row in _ident_rows(ident_dir)
               if row["population"] == "look" and row["decode_mode"] == WHOLE_CLIP]
    clean = [row for row in decoded if row["split"] == "clean"]
    labelled = [row for row in clean if row.get("filename_label")]
    flagged = [row for row in labelled if row["id_floor_ok"] is False]
    if not flagged:
        raise SystemExit("no flagged clean look row: the §2a disclosure would be empty")
    same = sum(1 for row in flagged if row.get("class_label") == row["filename_label"])
    near = sum(1 for row in flagged
               if _within_ayat(row.get("class_label"), row["filename_label"], 2))
    calibration = json.loads(Path(tlog_ident.CALIBRATION_PATH).read_text(encoding="utf-8"))
    return {"flagged": len(flagged), "same": same, "near": near,
            "retention": sum(1 for row in clean if row["id_floor_ok"] is True) / len(clean),
            "calibration_retention": calibration["look"]["retention"]}


def _require_verdicts(worklist: Sequence[Mapping[str, Any]],
                      verdicts: Mapping[str, Any], path: Path) -> None:
    """Refuse a document built from no verdicts, or from verdicts off another worklist."""
    if not verdicts:
        raise SystemExit(f"{path} has no verdicts: commit the owner's sitting before reporting")
    stray = sorted(set(verdicts) - {_item_key(row) for row in worklist})
    if stray:
        raise SystemExit(f"{path}: {len(stray)} verdicts name items that are not in the "
                         f"worklist ({', '.join(stray[:3])})")


def render_audit(*, worklist_path: Path = AUDIT_DIR / "worklist.jsonl",
                 verdicts_path: Path = AUDIT_DIR / "verdicts.jsonl",
                 ident_dir: Path = IDENT_DIR) -> str:
    """Render ``docs/tlog-audit.md`` from the committed worklist and verdicts.

    A render from zero verdicts would assert ``trigger fired: False`` from no evidence, so it
    refuses instead; a partly answered sitting still renders.
    """
    worklist = [dict(row) for row in _read_jsonl(worklist_path)]
    if not worklist:
        raise SystemExit(f"{worklist_path} is empty: there is no audit to report")
    verdicts = read_verdicts(verdicts_path)
    _require_verdicts(worklist, verdicts, verdicts_path)
    d_max = float(decode_limit()["d_max_seconds"])
    database, _bars, _k = load_identifier()
    canonical = lambda label: tlog_ident.canonical_label(database, label)
    by_stratum = _stratum_views(worklist)
    # The floor table and §9 read the frozen sample only (architect r22): a fallback's
    # restricted-only extras are reported, never counted. An older worklist carries no marker
    # at all, which would silently count extras, so it is refused.
    floor_rows = [{**row, "stratum": "non_recitation"} for row in worklist
                  if row.get("floor_sample")]
    extras = [{**row, "stratum": "non_recitation"} for row in worklist
              if "non_recitation" in row["strata"] and not row.get("floor_sample")]
    if extras and not floor_rows:
        raise SystemExit(f"{worklist_path} marks no §9 sample row: the floor outcome cannot "
                         "tell the frozen sample from an earlier draw's extras")
    by_stratum["non_recitation"] = floor_rows
    lines = ["# tlog S6 audit (#191)", "",
             "Generated by `python -m tadabur.tlog_ident_run report` from",
             "`tools/tadabur/tlog/audit/{worklist,verdicts}.jsonl`. Never hand-edited.", "",
             f"The S6 amendment (`docs/tlog-s6-amendment.md`, SHA-256 "
             f"`{sha256_file(AMENDMENT_PATH)}`) was approved by the owner on "
             f"{AMENDMENT_APPROVAL['date']} ({AMENDMENT_APPROVAL['url']}).", "",
             f"{len(verdicts)} of {len(worklist)} drawn items answered. A Wilson 95% interval "
             f"is reported at any n; below {MIN_READABLE_N} decided items it reads "
             "`n too small to read` (amendment §4).", "",
             "## Verdicts", "",
             "| stratum | drawn | answered | decided | right | precision | Wilson 95% |",
             "|---|---|---|---|---|---|---|"]
    for stratum in AUDIT_SIZES:
        rows = by_stratum.get(stratum, [])
        stats = _precision(rows, verdicts, canonical)
        unreadable = bool(stats["decided"]) and stats["decided"] < MIN_READABLE_N
        precision = ("n too small to read" if unreadable else
                     "—" if stats["precision"] is None else f"{stats['precision']:.2%}")
        lines.append(f"| {stratum} | {len(rows)} | {stats['n']} | {stats['decided']} | "
                     f"{stats['correct']} | {precision} | {_wilson_cell(stats)} |")
    lines += ["", "## Precision by duration band", "",
              "| stratum | 0-20 s | 20-60 s | 60-d_max |", "|---|---|---|---|"]
    for stratum in AUDIT_SIZES:
        bands = _by_band(by_stratum.get(stratum, []), d_max)
        cells = [_band_cell(_precision(bands[band], verdicts, canonical))
                 for band in DURATION_BANDS]
        lines.append(f"| {stratum} | " + " | ".join(cells) + " |")
    both = _precision(by_stratum.get("both", []), verdicts, canonical)
    wrong = 1 - both["precision"] if both["precision"] is not None else None
    trigger = wrong is not None and wrong > OUTSIDE_ASR_LIMIT
    lines += ["", "## Outside-ASR trigger", "",
              f"- Residual wrong-ayah rate in `both`: "
              f"{'—' if wrong is None else f'{wrong:.2%}'} (limit "
              f"{OUTSIDE_ASR_LIMIT:.0%}).",
              f"- Trigger fired: **{trigger}**."
              + (" A follow-up issue is opened for an outside-family ASR cross-check."
                 if trigger else "")]
    drawn_floor = len(by_stratum["non_recitation"])
    floor = _precision(by_stratum["non_recitation"], verdicts, canonical)
    extras_stats = _precision(extras, verdicts, canonical)
    paused = floor["recitation"] >= FLOOR_PAUSE_AT
    lines += ["", "## The S5b §9 trigger", "",
              f"- Sample: the first {drawn_floor} rows by the frozen salt's rank among all "
              "decoded rows failing the floor, whatever other stratum they also belong to "
              f"(`{OVERLAP_NOTE_PATH.relative_to(ROOT).as_posix()}` §3).",
              f"- Audited `non_recitation` clips heard as real recitation: "
              f"**{floor['recitation']} of {drawn_floor} drawn** ({floor['n']} answered).",
              f"- Restricted-only extras kept from an earlier draw: {len(extras)} "
              f"({extras_stats['n']} answered, {extras_stats['recitation']} heard as real "
              "recitation). Reported, never counted in §9.",
              f"- Pre-stated before any verdict: {FLOOR_PAUSE_AT} or more suspends the 5% "
              "rule's consequence pending a dated floor revision the owner decides.",
              f"- **Suspended: {paused}.**"
              + (" The whole-pool decode and the `not non_recitation` filter do not start."
                 if paused else " They proceed as frozen, through the follow-up issue."),
              "- The follow-up issue for the whole-pool decode and the filter is drafted, not "
              "filed: it is blocked on the owner's decision and on this outcome.", ""]
    below = _precision(by_stratum["below-bar disagreement"], verdicts, canonical)
    below_precision = ("—" if below["precision"] is None
                       else f"{below['precision']:.2%}")
    lines += ["", "## The best class in the below-bar stratum (reported, never gating)", "",
              f"- {_best_class_line(below)}",
              f"- Filename-label precision in the same items: {below_precision}.", ""]
    silence = _precision(by_stratum.get("silence", []), verdicts, canonical)
    unclean = [row for row in by_stratum.get("silence", []) if row["split"] == "unclean"]
    lines += ["## The recitation floor in the field", "",
              f"- `non_recitation` items heard as real recitation: {floor['recitation']} of "
              f"{drawn_floor} drawn ({floor['n']} answered).",
              f"- silence items heard as real recitation: {silence['recitation']} of "
              f"{len(by_stratum.get('silence', []))} drawn; unclean alone {len(unclean)} "
              "drawn.", ""]
    meta_path = Path(ident_dir) / "train_sample.meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"{meta_path} is absent: the 5% rule has not been computed")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rule = meta["five_percent_rule"]
    lines += ["## The 5% rule", "",
              f"- sample n = {rule['n']}, non_recitation = {rule['non_recitation']}, "
              f"Wilson 95% upper = {rule['wilson'][1]:.2%}, triggered = "
              f"**{rule['triggered']}**.", "",
              "| band | rows | non_recitation |", "|---|---|---|"]
    for band in DURATION_BANDS:
        counts = meta["bands"][band]
        lines.append(f"| {band} | {counts['rows']} | {counts['non_recitation']} |")
    lines.append("")
    if not OVERLAP_META_PATH.is_file():
        raise SystemExit(f"{OVERLAP_META_PATH} is absent: the S4 disclosure cannot be skipped")
    overlap = json.loads(OVERLAP_META_PATH.read_text(encoding="utf-8"))
    words = overlap["controls"]["stop_word_matches"]["accepted"]
    survivors = overlap["duplicates"]["survivor_unusable"]
    lines += ["## Disclosure: the S4 exclusions (not audited here)", "",
              f"- {words['near_duplicate']['share']:.2%} of accepted near-duplicate matches are "
              f"dominated by stop-words; {words['all']['share']:.2%} combined, above the 1% "
              "warning.",
              f"- {survivors['eval']['total']} eval-role and {survivors['train']['total']} "
              "train usable rows were lost as `survivor_unusable`.",
              "- No audited item: those rows are excluded.", ""]
    field = _label_independent(Path(ident_dir))
    frame = _floor_frame(Path(ident_dir))
    lines += ["## Disclosure: the floor's calibration population", "",
              f"- The floor and its 0.30 decoy margin were calibrated on Tarteel v1, whose "
              "clips were filtered upstream (`cer_wav <= 0.299`, completeness >= 0.85); tlog "
              f"is unfiltered. Tarteel look retention was "
              f"{field['calibration_retention']:.2%}; tlog clean look retention is "
              f"{field['retention']:.2%}.",
              f"- The shortlist is label-free, so a best match on the labelled ayah is not "
              f"driven by the label: of {field['flagged']} flagged clean look rows, "
              f"{field['same']} ({field['same'] / field['flagged']:.1%}) match exactly the "
              f"labelled ayah and {field['near']} ({field['near'] / field['flagged']:.1%}) a "
              "neighbour within two ayat. Many flagged rows are recitation the teacher decodes "
              "poorly, not non-recitation.",
              f"- Sampling frame of the `non_recitation` stratum: all {frame['failing']} decoded "
              "rows that fail the floor. "
              f"{frame['both']} of them also match their filename class and are drawn in `both` "
              "as well; per "
              f"`{OVERLAP_NOTE_PATH.relative_to(ROOT).as_posix()}` each such row is heard once "
              "and scored in both strata, so the precision above covers the whole flag rather "
              "than a subset of it.", ""]
    lines += ["## Served order", "",
              "- The page serves the committed worklist in its written (shard) order rather "
              "than the amendment §5 salted shuffle: the worklist was frozen once verdicts "
              "began, and the order carries no stratum information.", ""]
    return "\n".join(lines).rstrip() + "\n"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


# --- measuring D_max on synthetic audio --------------------------------------------------------


def measure_decode_limit(*, seed: int = MEASURE_SEED) -> dict:
    """Decode seeded synthetic audio whole-clip at the production settings, rung by rung."""
    import numpy as np
    import torch

    from .tlog_ident_decode import _decoder as load_decoder

    decoder, _identity, _fingerprint = load_decoder()
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    rng = np.random.default_rng(seed)
    rungs = []
    for seconds in LADDER:
        samples = rng.standard_normal(int(seconds * 16_000)).astype(np.float32) * 0.01
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            decoder.decode_spans([samples])
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


# --- CLI ---------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    limit = commands.add_parser("limit", help="measure D_max on synthetic audio (GPU box)")
    limit.add_argument("--out", type=Path, default=DECODE_LIMIT_PATH)
    decode_ = commands.add_parser("decode", help="stream and decode the selected rows")
    decode_.add_argument("--cache-dir", type=Path, required=True)
    decode_.add_argument("--out-dir", type=Path, default=DECODES_DIR)
    decode_.add_argument("--scratch", type=Path, default=None)
    identify_ = commands.add_parser("identify", help="identify the committed decodes")
    identify_.add_argument("--decodes-dir", type=Path, default=DECODES_DIR)
    identify_.add_argument("--out-dir", type=Path, default=IDENT_DIR)
    commands.add_parser("audit", help="draw the stratified audit worklist")
    commands.add_parser("supplement", help="complete an earlier worklist's §9 sample in place")
    stage_ = commands.add_parser("stage", help="stage the drawn clips for listening")
    stage_.add_argument("--audio-dir", type=Path, required=True)
    stage_.add_argument("--cache-dir", type=Path, required=True)
    stage_.add_argument("--draw", type=Path, default=AUDIT_DIR / "draw.jsonl",
                        help="the draw to stage (default: the box's audit/draw.jsonl)")
    report_ = commands.add_parser("report", help="render docs/tlog-audit.md")
    report_.add_argument("--audio-dir", type=Path, default=None,
                         help="the session-only staged audio to delete after the report")
    args = parser.parse_args()
    if args.command == "limit":
        record = measure_decode_limit()
        write_text_atomically(args.out, json.dumps(record, ensure_ascii=False, indent=1,
                                                   sort_keys=True) + "\n")
        print(f"wrote {args.out}: D_max {record['d_max_seconds']} s", flush=True)
    elif args.command == "decode":
        print(json.dumps(decode(cache_dir=args.cache_dir, out_dir=args.out_dir,
                                scratch=args.scratch), sort_keys=True), flush=True)
    elif args.command == "identify":
        print(json.dumps(identify_trees(decodes_dir=args.decodes_dir, out_dir=args.out_dir),
                         sort_keys=True), flush=True)
    elif args.command == "audit":
        draw = audit_draw(audit_candidates())
        _write_jsonl(AUDIT_DIR / "draw.jsonl", draw.items)
        print(json.dumps(draw.strata, indent=1, sort_keys=True), flush=True)
    elif args.command == "supplement":
        corrected = with_floor_supplement(audit_candidates(),
                                          _read_jsonl(AUDIT_DIR / "worklist.jsonl"))
        _write_jsonl(AUDIT_DIR / "worklist.jsonl", corrected)
        print(f"{len(corrected)} worklist rows, "
              f"{sum(1 for row in corrected if row['floor_sample'])} in the §9 sample", flush=True)
    elif args.command == "stage":
        draw = AuditDraw(items=tuple(_read_jsonl(args.draw)), strata={})
        stage(draw, audio_dir=args.audio_dir, cache_dir=args.cache_dir)
    elif args.command == "report":
        write_text_atomically(DOC_PATH, render_audit())
        if args.audio_dir is not None:
            shutil.rmtree(args.audio_dir, ignore_errors=True)
            print(f"deleted the staged audio in {args.audio_dir}", flush=True)
        print(f"wrote {DOC_PATH}", flush=True)


if __name__ == "__main__":
    main()
