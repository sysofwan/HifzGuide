"""The tlog clip-level split freeze (#188): every row's group key and pre-registered role.

The owner accepted the S2 fallback on 2026-10-24: no speaker embedding, no clustering, so a
row's group is its S1 **id group** and ``group_key`` is that group's smallest ``audio_path``
(:func:`tadabur.tlog_index.id_groups`). ``speaker_cluster`` is therefore a documented ``null``
on every row; the schema keeps the column so it still matches the pre-registration. The role is
the pre-registered hash of ``group_key`` (:func:`tadabur.tlog_index.role`), so the freeze is
reproducible from the committed group keys alone.

``tools/tadabur/tlog/split/<split>/<shard>.jsonl.gz``
    one row per pass-1 row: ``split``, ``shard_file``, ``row``, ``audio_path``,
    ``speaker_cluster`` (always null), ``group_key`` and ``role``. Keys sorted, ``mtime`` 0 and
    written atomically by the index writer, so a rerun reproduces every byte. A row's key is
    ``(split, shard_file, row)`` — that is the tuple the index, pass 1 and the freeze are
    aligned on. ``audio_path`` is **not** a key: 4,052 paths appear in 8,104 rows across shards,
    and two such rows can carry different facts, so a consumer that joins on the path merges
    rows that only share a group.
``tools/tadabur/tlog/split/meta.json``
    the code SHA with its PR number and ``refs/pull/<N>/head``, the pre-registration's and
    S2 report's SHA-256, the pass-1 meta's SHA-256, the floors and their ``f_s``/``d_s``, the
    ``"embeddings": "none (owner fallback 2026-10-24)"`` record, the counts per role and split,
    the largest group's share (the pre-registration's 1% cap), each role's share, every role's
    surah/ayah distribution from filename labels (the training pool included), the yield after
    the floors, and each shard's producing code SHA + PR number as the pass-1 meta recorded it.

``tools/tadabur/tlog/split/pool.json``
    the gating population S4 consumes, which ``meta.json`` does not carry: per split and per
    role, ``rows``, ``usable`` after the codec and silence floors,
    ``eval_overlap_tarteel_v1`` (the share of Tarteel v1's own eval clips) and
    ``usable_after_eval_overlap``. ``meta.json`` keeps the floors-only yield and names the
    overlap as deferred; this file makes it countable without regenerating the freeze.

The build refuses a pass-1 shard whose rows are not exactly the index shard's rows
(:func:`tadabur.tlog_pass1.require_pass1_alignment`), so the freeze cannot cite a shard a
resumed run would have refused. :func:`iter_frozen_rows` is the one join of the three trees:
it yields each row's merged view (freeze fields, index label and overlap flags, pass-1 facts
and floor reasons), and :func:`build` iterates it, so S4 reads the same join the freeze wrote.

Usage (from ``tools/``, offline; no audio is read)::

  python -m tadabur.tlog_split build --pr <N>

A later HEAD changes ``meta.json``'s ``code_sha`` and the producers it cites, so a faithful
rebuild of the committed freeze pins both: ``build --code-sha 7380e1c --pr 199``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping
from itertools import groupby
from pathlib import Path
from typing import Any

from training.tashkeel_eval import write_text_atomically

from .nonpro_sources import TLOG, TLOG_SPLITS, tlog_split
from .tlog_index import (INDEX_DIR, ROLES, head_sha, provenance, repo_relative, role,
                         sha256_file, write_shard)
from .tlog_pass1 import (FLOORS_PATH, PASS1_DIR, exclusion_reasons, read_jsonl_gz,
                         require_pass1_alignment)

SCHEMA_VERSION = 1
#: The freeze's committed location (``tools/tadabur/tlog/split``).
SPLIT_DIR = Path(__file__).parent / "tlog" / "split"
PREREGISTRATION_PATH = (Path(__file__).resolve().parents[2]
                        / "docs" / "tlog-holdout-preregistration.md")
S2_REPORT_PATH = Path(__file__).parent / "tlog" / "s2" / "report.json"
#: The owner's fallback, recorded verbatim: no embedding was computed, not "deleted later".
EMBEDDINGS_NOTE = "none (owner fallback 2026-10-24)"
#: Why every committed row still carries the pre-registration's cluster column.
SPEAKER_CLUSTER_NOTE = ("null (S2 fallback 2026-10-24: no speaker embeddings or clustering; "
                        "the group is the S1 id group)")
#: ``audio_sha256`` of the zero-byte rows the unclean split is mostly made of.
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
#: The tuple that identifies one row across the index, pass 1 and the freeze.
ROW_KEY = ("split", "shard_file", "row")
#: The committed freeze fields, and nothing else.
FREEZE_KEYS = ("split", "shard_file", "row", "audio_path", "speaker_cluster", "group_key",
               "role")
#: The gating population's committed location (``tools/tadabur/tlog/split/pool.json``).
POOL_NAME = "pool.json"


def freeze_row(index_row: Mapping[str, Any]) -> dict:
    """One committed freeze row from one S1 index row (the fallback's group and role)."""
    group_key = str(index_row["id_group"])
    return {
        "split": index_row["split"],
        "shard_file": index_row["shard_file"],
        "row": index_row["row"],
        "audio_path": index_row["audio_path"],
        "speaker_cluster": None,
        "group_key": group_key,
        "role": role(group_key),
    }


def iter_frozen_rows(*, index_dir: Path = INDEX_DIR, pass1_dir: Path = PASS1_DIR,
                     floors_path: Path = FLOORS_PATH) -> Iterator[dict]:
    """Every row's merged view of the three committed trees, shard by shard in index order.

    One row per :data:`ROW_KEY` ``(split, shard_file, row)`` — never per ``audio_path``, which
    is not unique across shards — carrying the committed freeze fields, the index row's
    ``filename_label``, ``recording_id`` and Tarteel v1 overlap flags, the pass-1 facts
    (``audio_sha256``, ``codec_ok``, ``speech_s``, ``speech_ratio``) and the row's floor
    ``exclusion_reasons``. Rows are aligned by
    :func:`tadabur.tlog_pass1.require_pass1_alignment`, the same rule the resume guard applies,
    and arrive shard by shard so a consumer can group on ``(split, shard_file)``.
    """
    floors = json.loads(Path(floors_path).read_text(encoding="utf-8"))
    index_dir, pass1_dir = Path(index_dir), Path(pass1_dir)
    for index_path in sorted(index_dir.glob("*/*.jsonl.gz")):
        split = index_path.parent.name
        stem = index_path.name[: -len(".jsonl.gz")]
        if f"data/{stem}.parquet" not in TLOG.shards or tlog_split(f"data/{stem}.parquet") != split:
            raise ValueError(f"{index_path} is not a tlog index shard")
        pass1_path = pass1_dir / split / index_path.name
        if not pass1_path.exists():
            raise ValueError(f"pass 1 has no output for {split}/{stem}")
        index_rows = read_jsonl_gz(index_path)
        facts_rows = read_jsonl_gz(pass1_path)
        require_pass1_alignment(facts_rows, index_rows, pass1_path)
        for index_row, facts in zip(index_rows, facts_rows, strict=True):
            reasons = exclusion_reasons(facts, floors)
            yield {
                **freeze_row(index_row),
                "filename_label": index_row["filename_label"],
                "recording_id": index_row["recording_id"],
                "eval_overlap": bool(index_row["tarteel_v1_eval_overlap"]),
                "any_overlap": bool(index_row["tarteel_v1_any_overlap"]),
                "audio_sha256": facts["audio_sha256"],
                "codec_ok": facts["codec_ok"],
                "speech_s": facts["speech_s"],
                "speech_ratio": facts["speech_ratio"],
                "exclusion_reasons": reasons,
            }


def _pool_record(counts: Counter) -> dict:
    """One population's usable count after the floors, and after the Tarteel v1 overlap too."""
    return {"rows": counts["rows"], "usable": counts["usable"],
            "eval_overlap_tarteel_v1": counts["eval_overlap"],
            "usable_after_eval_overlap": counts["usable_after_eval_overlap"]}


def _yield_record(counts: Counter) -> dict:
    """One split's usable-row report, after the ``codec`` and ``silence`` floors.

    ``usable`` is not the pre-registration's whole exclusion set: ``eval_overlap`` and
    ``duplicate`` are S4's, and ``meta["exclusions"]`` records that they are deferred.
    """
    rows = counts["rows"]
    return {
        "rows": rows,
        "codec_failed": counts["codec_failed"],
        "empty_bytes": counts["empty_bytes"],
        "silence": counts["silence"],
        "usable": counts["usable"],
        "usable_share": round(counts["usable"] / rows, 6) if rows else 0.0,
    }


def build(split_dir: Path = SPLIT_DIR, *, index_dir: Path = INDEX_DIR,
          pass1_dir: Path = PASS1_DIR, floors_path: Path = FLOORS_PATH,
          code_sha: str | None = None,
          pull_request: int | None = None) -> dict:
    """Write every shard's freeze rows, ``meta.json`` and ``pool.json``; return the meta.

    :func:`iter_frozen_rows` is the one join of the index, pass 1 and the freeze, so the row
    counters here and S4's gating population come from the same alignment. The index is the
    row universe; pass 1 must have an aligned output for every index shard.
    A shard's producing SHA and PR come from the pass-1 meta when it records them (the three
    S3a tracer shards were verified, not re-measured), and from this build's own provenance
    only when no pass-1 meta exists at all: a shard a present meta omits is refused, because
    this build did not measure those rows either.
    """
    index_dir, pass1_dir = Path(index_dir), Path(pass1_dir)
    split_dir, floors_path = Path(split_dir), Path(floors_path)
    floors = json.loads(floors_path.read_text(encoding="utf-8"))
    pass1_meta_path = pass1_dir / "meta.json"
    pass1_meta = (json.loads(pass1_meta_path.read_text(encoding="utf-8"))
                  if pass1_meta_path.exists() else {})
    producers = pass1_meta.get("shards", {})
    current = provenance(code_sha or head_sha(), pull_request)

    by_split: Counter = Counter()
    by_role = Counter({name: 0 for name in ROLES})
    groups: Counter = Counter()
    distribution = {name: {"rows": 0, "labelled": 0,
                           "surah": Counter(), "ayah": Counter()} for name in ROLES}
    yields = {split: Counter() for split in TLOG_SPLITS}
    pool = {split: Counter() for split in TLOG_SPLITS}
    pool_by_role = {name: Counter() for name in ROLES}
    paths: Counter = Counter()
    shard_producers = {}

    index_files = sorted(index_dir.glob("*/*.jsonl.gz"))
    if not index_files:
        raise ValueError(f"no index shards under {index_dir}")
    index_meta_path = index_dir / "meta.json"
    if index_meta_path.exists():
        declared = json.loads(index_meta_path.read_text(encoding="utf-8"))["source"]["splits"]
        found = Counter(path.parent.name for path in index_files)
        if dict(found) != declared:
            raise ValueError(f"the index holds {dict(found)} shards; its meta declares "
                             f"{declared}: a whole index is required for the freeze")
    rows = iter_frozen_rows(index_dir=index_dir, pass1_dir=pass1_dir, floors_path=floors_path)
    for key, shard_rows in groupby(rows, key=lambda row: f"{row['split']}/{row['shard_file']}"):
        shard = list(shard_rows)
        split = shard[0]["split"]
        write_shard([{field: row[field] for field in FREEZE_KEYS} for row in shard],
                    split_dir / split / f"{shard[0]['shard_file']}.jsonl.gz")
        producer = producers.get(key)
        if producer is None:
            if pass1_meta:
                raise ValueError(
                    f"pass 1 meta has no producing record for {key}: the freeze would cite "
                    f"this build's SHA for rows it never produced")
            producer = current
        shard_producers[key] = {"code_sha": producer["code_sha"],
                                "pull_request": producer["pull_request"]}

        for row in shard:
            by_split[split] += 1
            by_role[row["role"]] += 1
            groups[row["group_key"]] += 1
            bucket = distribution[row["role"]]
            bucket["rows"] += 1
            label = row["filename_label"]
            if label is not None:
                bucket["labelled"] += 1
                bucket["surah"][label.split(":", 1)[0]] += 1
                bucket["ayah"][label] += 1
            reasons = row["exclusion_reasons"]
            report = yields[split]
            report["rows"] += 1
            report["codec_failed"] += "codec" in reasons
            report["empty_bytes"] += ("codec" in reasons
                                      and row["audio_sha256"] == EMPTY_SHA256)
            report["silence"] += "silence" in reasons
            report["usable"] += not reasons
            paths[row["audio_path"]] += 1
            usable, overlap = not reasons, row["eval_overlap"]
            for population in (pool[split], pool_by_role[row["role"]]):
                population["rows"] += 1
                population["usable"] += usable
                population["eval_overlap"] += overlap
                population["usable_after_eval_overlap"] += usable and not overlap

    total = sum(by_split.values())
    largest_key, largest_rows = groups.most_common(1)[0] if groups else ("", 0)
    yield_report = {split: _yield_record(yields[split]) for split in TLOG_SPLITS}
    yield_report["total"] = _yield_record(sum(yields.values(), Counter()))
    pool_record = {
        "schema_version": SCHEMA_VERSION,
        **current,
        "row_key": list(ROW_KEY),
        "floors": {"path": repo_relative(floors_path), "sha256": sha256_file(floors_path),
                   "f_s": floors["f_s"], "d_s": floors["d_s"]},
        "exclusions": {"applied": ["codec", "silence", "eval_overlap_tarteel_v1"],
                       "deferred": ["duplicate"]},
        "splits": {split: _pool_record(pool[split]) for split in TLOG_SPLITS},
        "roles": {name: _pool_record(pool_by_role[name]) for name in ROLES},
        "total": _pool_record(sum(pool.values(), Counter())),
        # ``audio_path`` is not a row key: these are the paths that several rows share.
        "audio_paths": {"distinct": len(paths),
                        "shared_by_several_rows": sum(1 for count in paths.values()
                                                      if count > 1),
                        "rows_covered_by_shared_paths": sum(count for count in paths.values()
                                                            if count > 1)},
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source": {"repo": TLOG.repo, "revision": TLOG.revision, "licence": TLOG.licence,
                   "splits": {split: sum(1 for shard in TLOG.shards
                                         if tlog_split(shard) == split)
                              for split in TLOG_SPLITS}},
        **current,
        "preregistration": {"path": repo_relative(PREREGISTRATION_PATH),
                            "sha256": sha256_file(PREREGISTRATION_PATH)},
        "s2_report": {"path": repo_relative(S2_REPORT_PATH),
                      "sha256": sha256_file(S2_REPORT_PATH)},
        "pass1": {"path": repo_relative(pass1_dir),
                  "meta_sha256": (sha256_file(pass1_meta_path)
                                  if pass1_meta_path.exists() else None)},
        "floors": {"path": repo_relative(floors_path),
                   "sha256": sha256_file(floors_path),
                   "f_s": floors["f_s"], "d_s": floors["d_s"]},
        "embeddings": EMBEDDINGS_NOTE,
        "speaker_cluster": SPEAKER_CLUSTER_NOTE,
        # The pre-registration's exclusion set is codec, silence, eval_overlap and duplicate.
        # The yield report decides the first two; eval_overlap:tarteel_v1 sits on the S1 index
        # row and duplicate needs S4's fingerprints, so a training pool must apply both before
        # it trusts the yield's ``usable`` count.
        "exclusions": {"applied": ["codec", "silence"],
                       "deferred": ["eval_overlap", "duplicate"]},
        "counts": {
            "rows": {"by_role": {name: by_role[name] for name in ROLES},
                     "by_split": {**{split: by_split[split] for split in TLOG_SPLITS},
                                  "total": total},
                     "total": total},
            "roles": {name: {"rows": by_role[name],
                             "share": round(by_role[name] / total, 6) if total else 0.0}
                      for name in ROLES},
            "groups": {"count": len(groups), "largest": largest_key,
                       "largest_rows": largest_rows,
                       "largest_share": (round(largest_rows / total, 6)
                                         if total else 0.0)},
        },
        "distribution": {name: {"rows": distribution[name]["rows"],
                                "labelled": distribution[name]["labelled"],
                                "surah": dict(sorted(distribution[name]["surah"].items())),
                                "ayah": dict(sorted(distribution[name]["ayah"].items()))}
                         for name in ROLES},
        "yield": yield_report,
        "shards": shard_producers,
    }
    write_text_atomically(
        split_dir / "meta.json",
        json.dumps(meta, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    write_text_atomically(
        split_dir / POOL_NAME,
        json.dumps(pool_record, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser(
        "build", help="write the freeze rows and meta.json from the committed index and pass 1")
    builder.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    builder.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    builder.add_argument("--pass1-dir", type=Path, default=PASS1_DIR)
    builder.add_argument("--floors", type=Path, default=FLOORS_PATH)
    builder.add_argument("--code-sha", default=None)
    builder.add_argument("--pr", type=int, required=True,
                         help="the PR number whose refs/pull/<N>/head keeps code_sha reachable "
                              "after the squash merge")
    args = parser.parse_args()
    meta = build(args.split_dir, index_dir=args.index_dir, pass1_dir=args.pass1_dir,
                 floors_path=args.floors, code_sha=args.code_sha, pull_request=args.pr)
    print(json.dumps({"counts": meta["counts"]["rows"],
                      "groups": meta["counts"]["groups"],
                      "yield": meta["yield"]}, indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
