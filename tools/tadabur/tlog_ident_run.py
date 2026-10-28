"""tlog S6 (#191): the look and training-sample decodes, identification and the 5% rule.

This is the first content read of tlog. It follows the frozen
``docs/tlog-holdout-preregistration.md`` as amended by ``docs/tlog-s6-amendment.md``:
whole-clip decodes only, up to a GPU limit measured on synthetic audio and committed in
``tlog/ident/decode_limit.json`` before the first tlog decode; a salted 2,000-row training
sample; the 5% rule. The pool pipeline it drives is :mod:`tadabur.tlog_pool` (S7 reuses it); the
audit is :mod:`tadabur.tlog_audit`.

The committed trees are one JSONL.gz per population and split (r20 Q2)::

    tools/tadabur/tlog/ident/decodes/look/{clean,unclean}.jsonl.gz
    tools/tadabur/tlog/ident/decodes/train_sample.jsonl.gz
    tools/tadabur/tlog/ident/look/{clean,unclean}.jsonl.gz
    tools/tadabur/tlog/ident/train_sample.jsonl.gz
    tools/tadabur/tlog/audit/worklist.jsonl
    tools/tadabur/tlog/audit/verdicts.jsonl

Every row is keyed ``(split, shard_file, row)`` and sorted by that key. The decode meta records
the model, revision, code SHA and its PR ref, calibration, decode-limit and amendment SHAs; the
identification tree mirrors it and adds the 5% rule. No tlog audio is committed.

Usage, from ``tools/`` (GPU box for ``limit`` and ``decode``; ``--pr`` keeps the cited commit
resolvable after the squash merge)::

    python -m tadabur.tlog_ident_run limit --out tadabur/tlog/ident/decode_limit.json
    python -m tadabur.tlog_ident_run decode --cache-dir /root/scratch/tlog/s6/cache --pr 207
    python -m tadabur.tlog_ident_run identify --pr 207
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.counterfactual_eval import wilson_interval

from . import tlog_pool
from .tlog_ident import load_identifier
from .tlog_index import (REPO_ROOT, repo_relative, require_clean_tree, sha256_file, write_json,
                         write_shard)
from .tlog_overlap import iter_pool_rows
from .tlog_pass1 import read_jsonl_gz
from .tlog_pool import (
    DECODES_DIR,
    IDENT_DIR,
    SCHEMA_VERSION,
    count_by_shard,
    decode_limit,
    decode_rows,
    identify_rows,
    is_non_recitation,
    measure_decode_limit,
    not_decoded_record,
    row_key,
    sample_rows,
    select_rows,
    tree_census,
    tree_provenance,
    with_keys,
)

#: The seeded sample's size and the 5% rule's limit (the pre-registration's "Training pool").
SAMPLE_SIZE = 2_000
FIVE_PERCENT_LIMIT = 0.05
#: The amendment's stop: more than this share of the usable pool above D_max halts before S7.
D_MAX_STOP_SHARE = 0.01
#: The draw salt for the training sample and the audit (pinned in the amendment).
SALT = "tlog-s6-2026-10-27-v1"
#: The owner's approval of the S6 amendment. The frozen amendment says "requested"; the
#: approval given on #191 is recorded in the ident metas and the generated audit doc.
AMENDMENT_APPROVAL = {"status": "given", "date": "2026-10-27",
                      "url": "https://github.com/sysofwan/HifzGuide/issues/191"
                             "#issuecomment-6434980858"}
#: Per-band reporting over the frozen durations; 60 s belongs to the third band (r20 Q1).
DURATION_BANDS = ("0-20", "20-60", "60-d_max")

AMENDMENT_PATH = REPO_ROOT / "docs" / "tlog-s6-amendment.md"


@dataclass(frozen=True)
class Selection:
    """The look population, the salted training sample, and the rows above D_max."""

    look: tuple[dict, ...]
    train_sample: tuple[dict, ...]
    over_d_max: tuple[dict, ...]
    counts: dict[str, int]


def require_decode_preconditions() -> dict:
    """Refuse a decode until the amendment and the measured limit are committed.

    The order is r20's: synthetic measurement, the amendment, then the decode. The decode script
    additionally refuses a dirty checkout, so the committed files are the ones cited.
    """
    if not AMENDMENT_PATH.is_file():
        raise FileNotFoundError(f"the S6 amendment {repo_relative(AMENDMENT_PATH)} is absent; "
                                f"commit it before any tlog decode")
    if not tlog_pool.DECODE_LIMIT_PATH.is_file():
        raise FileNotFoundError(f"{repo_relative(tlog_pool.DECODE_LIMIT_PATH)} is absent; "
                                f"measure and commit D_max before any tlog decode")
    return decode_limit()


def select(*, d_max: float, count: int = SAMPLE_SIZE, salt: str | None = None,
           rows: Iterable[Mapping[str, Any]] | None = None) -> Selection:
    """The look population and the salted 2,000-row training sample, both bounded by D_max.

    Look is every usable ``look`` row; the sample is the first ``count`` usable ``train`` rows
    by salted rank of the row key. Rows above D_max are listed separately, carry
    ``decode_mode = NOT_DECODED`` and ``label_source = none``, and are never decoded.
    """
    salt = SALT if salt is None else salt
    selected, over = select_rows(iter_pool_rows() if rows is None else rows,
                                 roles=("look", "train"), d_max=d_max)
    look = sorted((row for row in selected if row["role"] == "look"), key=row_key)
    train = sorted((row for row in selected if row["role"] == "train"), key=row_key)
    sample = sample_rows(train, count=count, salt=salt)
    look_over = sum(1 for row in over if row["role"] == "look")
    train_over = sum(1 for row in over if row["role"] == "train")
    counts = {
        "look_rows": len(look) + look_over,
        "look_selected": len(look),
        "look_over_d_max": look_over,
        "train_rows": len(train) + train_over,
        "train_selected": len(sample),
        "train_over_d_max": train_over,
    }
    return Selection(look=tuple(look), train_sample=tuple(sorted(sample, key=row_key)),
                     over_d_max=tuple(sorted(over, key=row_key)), counts=counts)


def _provenance(pull_request: int | None = None) -> dict:
    """The pool provenance plus the S6 amendment's SHA."""
    return {**tree_provenance(pull_request),
            "amendment_sha256": sha256_file(AMENDMENT_PATH)}


def _require_d_max_share(counts: Mapping[str, int]) -> None:
    """Stop before S7 when more than 1% of the usable pool is above D_max (r20 Q1)."""
    total = counts["look_rows"] + counts["train_rows"]
    over = counts["look_over_d_max"] + counts["train_over_d_max"]
    if total and over / total > D_MAX_STOP_SHARE:
        raise RuntimeError(f"{over} of {total} usable rows ({over / total:.2%}) are above "
                           f"D_max, over the {D_MAX_STOP_SHARE:.0%} stop; ask the owner")


def decode(*, cache_dir: Path, out_dir: Path = DECODES_DIR,
           scratch: Path | None = None, d_max: float | None = None,
           pull_request: int | None = None) -> dict:
    """Stream every selected shard once, decode whole-clip, delete the shard, merge."""
    require_clean_tree(REPO_ROOT)
    limit = require_decode_preconditions()
    d_max = float(limit["d_max_seconds"] if d_max is None else d_max)
    selection = select(d_max=d_max)
    _require_d_max_share(selection.counts)
    scratch = Path(scratch) if scratch is not None else Path(cache_dir).parent / "decoded"
    started = time.time()
    records = decode_rows([*selection.look, *selection.train_sample], d_max=d_max,
                          cache_dir=cache_dir, scratch=scratch)
    sample_keys = {row_key(row) for row in selection.train_sample}
    look: dict[str, list[dict]] = {split: [] for split in ("clean", "unclean")}
    sample = []
    for row in records:
        (sample if row_key(row) in sample_keys else look[row["split"]]).append(row)
    for split, rows in look.items():
        write_shard(rows, out_dir / "look" / f"{split}.jsonl.gz")
    write_shard(sample, out_dir / "train_sample.jsonl.gz")
    # Each meta's census counts the rows written to its own tree: ``rows`` and the census
    # must agree, so the provenance of one tree is never read off another's.
    base = {**_provenance(pull_request), "schema_version": SCHEMA_VERSION,
            "selection": {"salt": SALT, **selection.counts}}
    look_census = tree_census(count_by_shard(
        [row for split in ("clean", "unclean") for row in look[split]]))
    write_json({**base, "census": look_census,
                "rows": {split: len(rows) for split, rows in look.items()}},
               out_dir / "look" / "meta.json")
    write_json({**base, "census": tree_census(count_by_shard(sample)),
                "rows": len(sample)}, out_dir / "train_sample.meta.json")
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


def band_report(rows: Iterable[Mapping[str, Any]], d_max: float) -> dict[str, dict[str, int]]:
    """Rows and ``non_recitation`` counts per band; rows above D_max belong to none."""
    report = {name: {"rows": 0, "non_recitation": 0} for name in DURATION_BANDS}
    for row in rows:
        band = band_of(row["duration_s"], d_max)
        if band is None:
            continue
        report[band]["rows"] += 1
        if is_non_recitation(row):
            report[band]["non_recitation"] += 1
    return report


def five_percent_rule(rows: Iterable[Mapping[str, Any]]) -> dict:
    """The sample's ``non_recitation`` rate and Wilson 95% outcome, triggered above 5%."""
    rows = list(rows)
    if not rows:
        raise ValueError("the 5% rule needs at least one sampled row")
    bad = sum(1 for row in rows if is_non_recitation(row))
    low, high = wilson_interval(bad, len(rows))
    return {"n": len(rows), "non_recitation": bad, "rate": bad / len(rows),
            "wilson": [low, high], "limit": FIVE_PERCENT_LIMIT,
            "triggered": high > FIVE_PERCENT_LIMIT}


# --- identification ----------------------------------------------------------------------------


def identify_trees(*, decodes_dir: Path = DECODES_DIR, out_dir: Path = IDENT_DIR,
                   d_max: float | None = None, pull_request: int | None = None) -> dict:
    """Identify the committed decodes with S5's engine and write the ident tree."""
    require_clean_tree(REPO_ROOT)
    d_max = float(decode_limit()["d_max_seconds"] if d_max is None else d_max)
    database, bars, k = load_identifier()
    provenance = {**_provenance(pull_request), "amendment_approval": dict(AMENDMENT_APPROVAL)}
    selection = select(d_max=d_max)
    look_rows: dict[str, list[dict]] = {"clean": [], "unclean": []}
    for split in ("clean", "unclean"):
        look_rows[split] = identify_rows(
            read_jsonl_gz(decodes_dir / "look" / f"{split}.jsonl.gz"),
            database=database, bars=bars, k=k)
    for row in selection.over_d_max:
        if row["role"] == "look":
            look_rows[row["split"]].append(
                with_keys(not_decoded_record(row, d_max=d_max), row))
    sample = identify_rows(read_jsonl_gz(decodes_dir / "train_sample.jsonl.gz"),
                           database=database, bars=bars, k=k)
    for split in ("clean", "unclean"):
        write_shard(sorted(look_rows[split], key=row_key),
                    out_dir / "look" / f"{split}.jsonl.gz")
    write_json({**provenance, "schema_version": SCHEMA_VERSION,
                "rows": {split: len(rows) for split, rows in look_rows.items()},
                "census": tree_census(count_by_shard(
                    [row for rows in look_rows.values() for row in rows]))},
               out_dir / "look" / "meta.json")
    write_shard(sorted(sample, key=row_key), out_dir / "train_sample.jsonl.gz")
    rule = five_percent_rule(sample)
    decode_meta = json.loads((decodes_dir / "train_sample.meta.json").read_text(encoding="utf-8"))
    meta = {**provenance, "schema_version": SCHEMA_VERSION, "rows": len(sample),
            "selection": decode_meta["selection"],
            "five_percent_rule": rule,
            "bands": band_report(sample, d_max),
            "census": tree_census(count_by_shard(sample))}
    write_json(meta, out_dir / "train_sample.meta.json")
    return {"look": {split: len(rows) for split, rows in look_rows.items()},
            "train_sample": len(sample), "five_percent_rule": rule}


# --- CLI ---------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    limit = commands.add_parser("limit", help="measure D_max on synthetic audio (GPU box)")
    limit.add_argument("--out", type=Path, default=tlog_pool.DECODE_LIMIT_PATH)
    decode_ = commands.add_parser("decode", help="stream and decode the selected rows")
    decode_.add_argument("--cache-dir", type=Path, required=True)
    decode_.add_argument("--out-dir", type=Path, default=DECODES_DIR)
    decode_.add_argument("--scratch", type=Path, default=None)
    decode_.add_argument("--pr", type=int, default=None,
                         help="the PR whose refs/pull/<N>/head keeps the cited code SHA")
    identify_ = commands.add_parser("identify", help="identify the committed decodes")
    identify_.add_argument("--decodes-dir", type=Path, default=DECODES_DIR)
    identify_.add_argument("--out-dir", type=Path, default=IDENT_DIR)
    identify_.add_argument("--pr", type=int, default=None,
                           help="the PR whose refs/pull/<N>/head keeps the cited code SHA")
    args = parser.parse_args()
    if args.command == "limit":
        record = measure_decode_limit()
        write_json(record, args.out)
        print(f"wrote {args.out}: D_max {record['d_max_seconds']} s", flush=True)
    elif args.command == "decode":
        print(json.dumps(decode(cache_dir=args.cache_dir, out_dir=args.out_dir,
                                scratch=args.scratch, pull_request=args.pr),
                         sort_keys=True), flush=True)
    else:
        print(json.dumps(identify_trees(decodes_dir=args.decodes_dir, out_dir=args.out_dir,
                                        pull_request=args.pr), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
