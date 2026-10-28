"""tlog S7 (#192): the four sealed ``confirm`` tranches, their manifests and the tlog eval uses.

S5's frozen identification is applied mechanically to ``confirm.1`` … ``confirm.4`` with the
base teacher only, then every role (``look`` and the four confirm tranches) is written as one
manifest under the owner-approved gating rule of ``docs/tlog-gating-amendment.md`` (2026-10-28;
it replaces the pre-registration's gating rule and is never edited in place). The three
commands are one pass each, driven by ``scripts/issue-192-ident.sh`` on the GPU box:

    decode      stream and decode every usable confirm row whole-clip, up to D_max, after a
                timed ~500-row slice; ``look`` is NOT decoded again (S6's trees are reused);
    identify    identify the committed decodes with S5's engine (``load_identifier``); a row
                above D_max is recorded ``not_decoded:over_d_max`` and carries no label;
    manifests   join each role's committed ident tree to the split tree's ``group_key``,
                write ``tranches/<role>.jsonl``, ``tranches/meta.json`` and the five
                ``ClipExposure`` uses ``tlog.look`` … ``tlog.confirm.4``.

The model is the pool's pinned base teacher, reached only through
:func:`tadabur.tlog_pool.decoder`; no student, C95 or candidate model decodes a confirm row
(a locked AST test checks the imports). The tranches are sealed: logs and messages carry
counts and row keys only, never a label or a decode, and nobody listens to a tranche's audio.

Decode trees follow S6's layout, one file per split plus a meta::

    tools/tadabur/tlog/ident/decodes/confirm.k/{clean,unclean}.jsonl.gz
    tools/tadabur/tlog/ident/confirm.k/{clean,unclean}.jsonl.gz
    tools/tadabur/tlog/tranches/{look,confirm.1,...,confirm.4}.jsonl
    tools/tadabur/tlog/tranches/meta.json

Usage, from ``tools/`` (GPU box for ``decode`` and ``identify``; ``--pr`` keeps the cited
commit resolvable after the squash merge)::

    python -m tadabur.tlog_tranches decode --cache-dir /root/scratch/tlog/s7/cache --pr 211
    python -m tadabur.tlog_tranches identify --pr 211
    python -m tadabur.tlog_tranches manifests --pr 211
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import tlog_pool
from .exposure import EXPOSURE_DIR, ClipExposure, write_clip_use
from .nonpro_sources import TLOG
from .tlog_ident import BOTH, FILENAME, IDENTIFIED, NONE, load_identifier
from .tlog_ident_run import D_MAX_STOP_SHARE
from .tlog_index import (require_clean_tree, sha256_file, write_json, write_jsonl, write_shard)
from .tlog_overlap import iter_pool_rows
from .tlog_pass1 import read_jsonl_gz

SCHEMA_VERSION = tlog_pool.SCHEMA_VERSION
#: The five pre-registered roles S7 writes a manifest for: look (S6's tree) and the four
#: separately sealed confirm tranches.
ROLES = ("look", "confirm.1", "confirm.2", "confirm.3", "confirm.4")
CONFIRM_ROLES = ROLES[1:]
GATING = "gating"
#: One stratum per manifest row: the six names in **summary** order (``gating`` first, the
#: primary count). This is not the precedence — that is the branch order in
#: :func:`stratum_of` (r24 item 2), and rebuilding the rule from this tuple would gate
#: ``unclean`` rows.
STRATA = (GATING, "reported:unclean", "reported:undecided", "reported:straddles",
          "reported:identified", "reported:below_bar")
#: The label sources an ident row may carry; anything else is a broken tree, not a stratum.
LABEL_SOURCES = (BOTH, IDENTIFIED, FILENAME, NONE)
#: The manifest row's fields, in written order (r24 item 2). Every one is copied from the
#: ident tree except ``group_key`` (the split tree) and ``stratum`` (the rule).
MANIFEST_FIELDS = ("split", "shard_file", "row", "audio_path", "audio_sha256", "group_key",
                   "filename_label", "assigned_label", "label_source", "straddles",
                   "decode_mode", "id_floor_ok", "stratum")
#: A claim record's schema (r24 item 4). S7 writes none: the round that opens a tranche does.
CLAIM_FIELDS = ("role", "issue", "commit", "date")
#: The rows the timed slice decodes before the full run (r24 item 6).
SLICE_ROWS = 500

TLOG_DIR = Path(__file__).parent / "tlog"
TRANCHES_DIR = TLOG_DIR / "tranches"
SPLIT_DIR = TLOG_DIR / "split"
PREREGISTRATION_PATH = tlog_pool.ROOT / "docs" / "tlog-holdout-preregistration.md"
GATING_AMENDMENT_PATH = tlog_pool.ROOT / "docs" / "tlog-gating-amendment.md"


# --- the gating rule (docs/tlog-gating-amendment.md §1, r24 item 2) ------------------------------


def stratum_of(record: Mapping[str, Any]) -> str:
    """One ident row's stratum: the first matching row of r24's precedence table.

    Step 4 of the table is the amendment's gating rule — clean, not ``none``, non-straddling
    and ``label_source == both`` — so a row reaching ``gating`` gates by construction. The
    recitation floor and the Δ relabel bar decide nothing here (amendment §1, §2).
    """
    source = record["label_source"]
    if source not in LABEL_SOURCES:
        raise ValueError(f"unknown label_source {source!r}; the tree carries "
                         f"{list(LABEL_SOURCES)}")
    if record["split"] == "unclean":
        return "reported:unclean"
    if source == NONE:
        return "reported:undecided"
    if record["straddles"]:
        return "reported:straddles"
    return {BOTH: GATING, IDENTIFIED: "reported:identified",
            FILENAME: "reported:below_bar"}[source]


def manifest_rows(ident_rows: Iterable[Mapping[str, Any]],
                  group_keys: Mapping[tuple[str, str, int], str]) -> list[dict]:
    """One manifest row per ident row, sorted by row key, carrying exactly :data:`MANIFEST_FIELDS`.

    ``group_keys`` is the split tree's ``(split, shard_file, row) -> group_key``: the S1 id
    group's smallest ``audio_path``. Under the owner's S2 fallback the bootstrap cluster is the
    **surah** — ``assigned_label``'s, counted as ``gating_surahs`` — not ``group_key``
    (``docs/tlog-holdout-preregistration.md``, ``docs/tlog-speaker-clustering.md``). A row the
    split tree does not cover is refused rather than written without its group.
    """
    rows = []
    for record in sorted(ident_rows, key=tlog_pool.row_key):
        key = tlog_pool.row_key(record)
        if key not in group_keys:
            raise ValueError(f"{key} has no group key; the split tree does not cover it")
        derived = {"group_key": group_keys[key], "stratum": stratum_of(record)}
        rows.append({field: derived[field] if field in derived else record[field]
                     for field in MANIFEST_FIELDS})
    return rows


def tranche_summary(rows: Iterable[Mapping[str, Any]]) -> dict:
    """One manifest's counts: its rows and clip entries, every stratum, and the §5 lines.

    ``entries`` is what the exposure use holds, so a dropped duplicate row is visible;
    ``gating_id_floor_ok`` is amendment §5's sensitivity line and ``gating_surahs`` the
    distinct surahs among the gating rows — the bootstrap cluster count (r24 item 1).
    """
    rows = list(rows)
    strata = Counter(row["stratum"] for row in rows)
    gating = [row for row in rows if row["stratum"] == GATING]
    return {"rows": len(rows),
            "entries": len(exposure_rows(rows)),
            "strata": {name: strata.get(name, 0) for name in STRATA},
            "gating": len(gating),
            "gating_id_floor_ok": sum(1 for row in gating if row["id_floor_ok"] is True),
            "gating_surahs": len({row["assigned_label"].split(":")[0] for row in gating})}


# --- selection ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """One confirm role's usable rows at or below D_max, and those above it (not decoded)."""

    role: str
    selected: tuple[dict, ...]
    over_d_max: tuple[dict, ...]


def select(role: str, *, d_max: float,
           rows: Iterable[Mapping[str, Any]] | None = None) -> Selection:
    """The usable rows of one confirm role, split at D_max and in row-key order.

    ``look`` is refused: S6 decoded it and S7 reuses that tree verbatim (r24 item 6).
    """
    if role not in CONFIRM_ROLES:
        raise ValueError(f"{role!r} is not a sealed confirm tranche {CONFIRM_ROLES}")
    selected, over = tlog_pool.select_rows(
        iter_pool_rows() if rows is None else rows, roles=(role,), d_max=d_max)
    return Selection(role=role,
                     selected=tuple(sorted(selected, key=tlog_pool.row_key)),
                     over_d_max=tuple(sorted(over, key=tlog_pool.row_key)))


def require_d_max_share(selection: Selection) -> None:
    """Refuse a tranche with strictly more than 1% of its usable rows above D_max.

    The check is per tranche (r24 item 6): the amendment's stop was measured on the pool, and a
    single tranche must not slip past it on the others' rows.
    """
    total = len(selection.selected) + len(selection.over_d_max)
    share = len(selection.over_d_max) / total if total else 0.0
    if share > D_MAX_STOP_SHARE:
        raise RuntimeError(f"{selection.role}: {len(selection.over_d_max)} of {total} usable "
                           f"rows ({share:.2%}) are above D_max, over the "
                           f"{D_MAX_STOP_SHARE:.0%} stop")


# --- the eval uses (r24 item 3) -----------------------------------------------------------------


def exposure_rows(rows: Iterable[Mapping[str, Any]]) -> list[ClipExposure]:
    """The clip exposures of manifest rows: every row of the role, whatever its stratum.

    The whole role is reserved for evaluation, so an unclean or undecoded row is listed too,
    with ``surah_ayah = unknown``. One clip of the dataset (path and checksum) carrying two
    ayahs inside a use is refused: the registry has no row key and would silently collapse it.
    """
    ayahs: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (str(row["audio_path"]), str(row["audio_sha256"]))
        ayah = row["assigned_label"] or "unknown"
        known = ayahs.setdefault(key, ayah)
        if known != ayah:
            raise ValueError(f"clip {key[0]} ({key[1][:12]}…) carries two ayahs in one use")
    return sorted({ClipExposure(TLOG.repo, TLOG.revision, path, ayah, checksum, "unknown")
                   for (path, checksum), ayah in ayahs.items()})


def require_disjoint_uses(uses: Mapping[str, Iterable[ClipExposure]]) -> None:
    """Refuse a clip in two uses: cross-tranche leakage that S4's overlap screen excludes."""
    owner: dict[tuple[str, str], str] = {}
    for use, clips in uses.items():
        for clip in clips:
            key = (clip.audio_filename, clip.audio_sha256)
            other = owner.setdefault(key, use)
            if other != use:
                raise ValueError(f"clip {key[0]} ({key[1][:12]}…) is in {other} and {use}: "
                                 f"cross-tranche leakage")


# --- claim order (r24 item 4) -------------------------------------------------------------------


def require_claim_order(role: str, tranches_dir: Path = TRANCHES_DIR) -> None:
    """Refuse a tranche whose predecessor no round has claimed yet.

    ``confirm.1`` needs no claim. A round may claim ``confirm.(k-1)`` and ``confirm.k``
    together, in that order, by writing each claim before opening the next tranche. S7 writes
    no claim: the file is the opening round's, with exactly :data:`CLAIM_FIELDS`.
    """
    if role not in CONFIRM_ROLES:
        raise ValueError(f"{role!r} is not claimable; only {CONFIRM_ROLES} have claims")
    position = CONFIRM_ROLES.index(role)
    if position == 0:
        return
    previous = CONFIRM_ROLES[position - 1]
    path = Path(tranches_dir) / f"{previous}.claim.json"
    if not path.is_file():
        raise ValueError(f"{role} needs {previous} claimed first; {path} is absent")
    record = json.loads(path.read_text(encoding="utf-8"))
    if set(record) != set(CLAIM_FIELDS) or record["role"] != previous:
        raise ValueError(f"{path} is not a {previous} claim: it needs exactly the fields "
                         f"{list(CLAIM_FIELDS)} and role {previous}")


# --- the committed tree readers -----------------------------------------------------------------


def split_group_keys(roles: Sequence[str] = ROLES,
                     split_dir: Path = SPLIT_DIR) -> dict[tuple[str, str, int], str]:
    """Every manifest role's ``group_key`` from the committed split tree, by row key."""
    keys: dict[tuple[str, str, int], str] = {}
    for path in sorted(Path(split_dir).glob("*/*.jsonl.gz")):
        for row in read_jsonl_gz(path):
            if row["role"] in roles:
                keys[tlog_pool.row_key(row)] = row["group_key"]
    return keys


def overlay_commit() -> dict:
    """The split and overlap overlay commits these manifests were built from (r24 item 7)."""
    split = json.loads((TLOG_DIR / "split" / "meta.json").read_text(encoding="utf-8"))
    overlap = json.loads((TLOG_DIR / "overlap" / "meta.json").read_text(encoding="utf-8"))
    return {"split_code_sha": split["code_sha"], "overlap_code_sha": overlap["code_sha"]}


# --- decode, identify, manifests ----------------------------------------------------------------


def decode(*, cache_dir: Path, out_dir: Path = tlog_pool.DECODES_DIR,
           scratch: Path | None = None, roles: Sequence[str] = CONFIRM_ROLES,
           d_max: float | None = None, slice_rows: int = SLICE_ROWS,
           pull_request: int | None = None) -> dict:
    """Decode every usable row of ``roles`` whole-clip, after timing a ~500-row slice.

    The slice runs first through the same path as the full pass, and the full pass reuses its
    rows via ``already``: no row is decoded or downloaded twice for the timing. Its record
    (``rows``, ``seconds``) is written to ``<out_dir>/slice_timing.json`` for the manifests'
    meta (r24 item 6).
    """
    require_clean_tree()
    d_max = float(tlog_pool.decode_limit()["d_max_seconds"] if d_max is None else d_max)
    if slice_rows <= 0:
        raise ValueError("the decode times a slice before the full run; --slice-rows is positive")
    selections = [select(role, d_max=d_max) for role in roles]
    for selection in selections:
        require_d_max_share(selection)
    rows = [row for selection in selections for row in selection.selected]
    role_of = {tlog_pool.row_key(row): selection.role
               for selection in selections for row in selection.selected}
    scratch = Path(scratch) if scratch is not None else Path(cache_dir).parent / "decoded"
    teacher, _identity, _fingerprint = tlog_pool.decoder()
    slice_started = time.time()
    slice_records = tlog_pool.decode_rows(rows[:slice_rows], d_max=d_max, cache_dir=cache_dir,
                                          scratch=scratch / "slice", teacher=teacher)
    slice_timing = {"rows": len(slice_records), "seconds": round(time.time() - slice_started, 1)}
    print(f"slice: {slice_timing['rows']} rows in {slice_timing['seconds']} s", flush=True)
    records = tlog_pool.decode_rows(rows, d_max=d_max, cache_dir=cache_dir, scratch=scratch,
                                    already=slice_records, teacher=teacher)
    trees = {role: {split: [] for split in ("clean", "unclean")} for role in roles}
    for record in records:
        trees[role_of[tlog_pool.row_key(record)]][record["split"]].append(record)
    base = {**tlog_pool.tree_provenance(pull_request), "schema_version": SCHEMA_VERSION}
    for role, splits in trees.items():
        for split, split_rows in splits.items():
            write_shard(split_rows, out_dir / role / f"{split}.jsonl.gz")
        write_json({**base, "census": tlog_pool.tree_census(tlog_pool.count_by_shard(
            [row for split_rows in splits.values() for row in split_rows])),
            "rows": {split: len(split_rows) for split, split_rows in splits.items()}},
            out_dir / role / "meta.json")
    write_json(slice_timing, out_dir / "slice_timing.json")
    counts = {role: {split: len(split_rows) for split, split_rows in splits.items()}
              for role, splits in trees.items()}
    return {"slice": slice_timing, "roles": counts}


def identify(*, decodes_dir: Path = tlog_pool.DECODES_DIR,
             out_dir: Path = tlog_pool.IDENT_DIR, roles: Sequence[str] = CONFIRM_ROLES,
             d_max: float | None = None, pull_request: int | None = None) -> dict:
    """Identify the tranches' committed decodes with S5's engine, one tree per role.

    A row above D_max was never decoded: it is written from the frozen pool row as
    ``not_decoded:over_d_max`` with ``label_source = none`` and no floor verdict.
    """
    require_clean_tree()
    d_max = float(tlog_pool.decode_limit()["d_max_seconds"] if d_max is None else d_max)
    database, bars, k = load_identifier()
    base = {**tlog_pool.tree_provenance(pull_request), "schema_version": SCHEMA_VERSION}
    counts = {}
    for role in roles:
        selection = select(role, d_max=d_max)
        splits: dict[str, list[dict]] = {split: tlog_pool.identify_rows(
            read_jsonl_gz(decodes_dir / role / f"{split}.jsonl.gz"), database=database,
            bars=bars, k=k) for split in ("clean", "unclean")}
        for row in selection.over_d_max:
            splits[row["split"]].append(tlog_pool.with_keys(
                tlog_pool.not_decoded_record(row, d_max=d_max), row))
        for split, split_rows in splits.items():
            write_shard(sorted(split_rows, key=tlog_pool.row_key),
                        out_dir / role / f"{split}.jsonl.gz")
        write_json({**base, "census": tlog_pool.tree_census(tlog_pool.count_by_shard(
            [row for split_rows in splits.values() for row in split_rows])),
            "rows": {split: len(split_rows) for split, split_rows in splits.items()}},
            out_dir / role / "meta.json")
        counts[role] = {split: len(split_rows) for split, split_rows in splits.items()}
    return counts


def manifests(*, ident_dir: Path = tlog_pool.IDENT_DIR, tranches_dir: Path = TRANCHES_DIR,
              exposure_dir: Path = EXPOSURE_DIR,
              slice_timing: Path = tlog_pool.DECODES_DIR / "slice_timing.json",
              pull_request: int | None = None) -> dict:
    """Write the five manifests, their eval uses and ``tranches/meta.json``.

    The look tree is S6's committed one, joined — never re-decoded. The confirm trees are
    S7's. Each row keeps ``group_key``, its label facts and ``stratum``, so any re-slice can
    be recomputed from the committed files.
    """
    require_clean_tree()
    timing = json.loads(Path(slice_timing).read_text(encoding="utf-8"))
    group_keys = split_group_keys()
    summaries = {}
    for role in ROLES:
        rows = manifest_rows(tlog_pool.read_ident_tree(role, ident_dir=ident_dir), group_keys)
        write_jsonl(rows, Path(tranches_dir) / f"{role}.jsonl", sort_keys=True)
        write_clip_use(f"tlog.{role}", exposure_rows(rows), Path(exposure_dir))
        summaries[role] = tranche_summary(rows)
    write_json({**tlog_pool.tree_provenance(pull_request), "schema_version": SCHEMA_VERSION,
                "preregistration_sha256": sha256_file(PREREGISTRATION_PATH),
                "gating_amendment_sha256": sha256_file(GATING_AMENDMENT_PATH),
                "overlay": overlay_commit(),
                "slice_timing": {"rows": timing["rows"], "seconds": timing["seconds"]},
                "tranches": summaries},
               Path(tranches_dir) / "meta.json")
    return summaries


# --- CLI ----------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    decode_ = commands.add_parser("decode", help="decode the confirm tranches on the GPU box")
    decode_.add_argument("--cache-dir", type=Path, required=True)
    decode_.add_argument("--out-dir", type=Path, default=tlog_pool.DECODES_DIR)
    decode_.add_argument("--scratch", type=Path, default=None)
    decode_.add_argument("--slice-rows", type=int, default=SLICE_ROWS)
    decode_.add_argument("--roles", nargs="+", default=list(CONFIRM_ROLES))
    decode_.add_argument("--pr", type=int, default=None,
                         help="the PR whose refs/pull/<N>/head keeps the cited code SHA")
    identify_ = commands.add_parser("identify", help="identify the committed decodes")
    identify_.add_argument("--decodes-dir", type=Path, default=tlog_pool.DECODES_DIR)
    identify_.add_argument("--out-dir", type=Path, default=tlog_pool.IDENT_DIR)
    identify_.add_argument("--roles", nargs="+", default=list(CONFIRM_ROLES))
    identify_.add_argument("--pr", type=int, default=None,
                           help="the PR whose refs/pull/<N>/head keeps the cited code SHA")
    manifests_ = commands.add_parser("manifests", help="write the tranche manifests and uses")
    manifests_.add_argument("--ident-dir", type=Path, default=tlog_pool.IDENT_DIR)
    manifests_.add_argument("--tranches-dir", type=Path, default=TRANCHES_DIR)
    manifests_.add_argument("--exposure-dir", type=Path, default=EXPOSURE_DIR)
    manifests_.add_argument("--slice-timing", type=Path,
                            default=tlog_pool.DECODES_DIR / "slice_timing.json")
    manifests_.add_argument("--pr", type=int, default=None,
                            help="the PR whose refs/pull/<N>/head keeps the cited code SHA")
    args = parser.parse_args()
    if args.command == "decode":
        result = decode(cache_dir=args.cache_dir, out_dir=args.out_dir, scratch=args.scratch,
                        roles=args.roles, slice_rows=args.slice_rows,
                        pull_request=args.pr)
    elif args.command == "identify":
        result = identify(decodes_dir=args.decodes_dir, out_dir=args.out_dir, roles=args.roles,
                          pull_request=args.pr)
    else:
        result = manifests(ident_dir=args.ident_dir, tranches_dir=args.tranches_dir,
                           exposure_dir=args.exposure_dir, slice_timing=args.slice_timing,
                           pull_request=args.pr)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
