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
    written atomically by the index writer, so a rerun reproduces every byte.
``tools/tadabur/tlog/split/meta.json``
    the code SHA with its PR number and ``refs/pull/<N>/head``, the pre-registration's and
    S2 report's SHA-256, the pass-1 meta's SHA-256, the floors and their ``f_s``/``d_s``, the
    ``"embeddings": "none (owner fallback 2026-10-24)"`` record, the counts per role and split,
    the largest group's share (the pre-registration's 1% cap), each role's share, every role's
    surah/ayah distribution from filename labels (the training pool included), the yield after
    the floors, and each shard's producing code SHA + PR number as the pass-1 meta recorded it.

The build refuses a pass-1 shard whose rows are not exactly the index shard's rows
(:func:`tadabur.tlog_pass1.require_pass1_alignment`), so the freeze cannot cite a shard a
resumed run would have refused.

Usage (from ``tools/``, offline; no audio is read)::

  python -m tadabur.tlog_split build --pr <N>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from training.tashkeel_eval import write_text_atomically

from .nonpro_sources import TLOG, TLOG_SPLITS, tlog_split
from .tlog_index import (INDEX_DIR, ROLES, head_sha, role, sha256_file, write_shard)
from .tlog_pass1 import (FLOORS_PATH, PASS1_DIR, _repo_relative, exclusion_reasons,
                         read_jsonl_gz, require_pass1_alignment)

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
    """Write every shard's freeze rows and ``meta.json``; return the meta record.

    The index is the row universe; pass 1 must have an aligned output for every index shard.
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
    current = {"code_sha": code_sha or head_sha(), "pull_request": pull_request,
               "pull_request_ref": (f"refs/pull/{pull_request}/head"
                                    if pull_request is not None else None)}

    by_split: Counter = Counter()
    by_role = Counter({name: 0 for name in ROLES})
    groups: Counter = Counter()
    distribution = {name: {"rows": 0, "labelled": 0,
                           "surah": Counter(), "ayah": Counter()} for name in ROLES}
    yields = {split: Counter() for split in TLOG_SPLITS}
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
    for index_path in index_files:
        split = index_path.parent.name
        stem = index_path.name[: -len(".jsonl.gz")]
        shard = f"data/{stem}.parquet"
        if shard not in TLOG.shards or tlog_split(shard) != split:
            raise ValueError(f"{index_path} is not a tlog index shard")
        key = f"{split}/{stem}"
        pass1_path = pass1_dir / split / index_path.name
        if not pass1_path.exists():
            raise ValueError(f"pass 1 has no output for {key}")
        index_rows = read_jsonl_gz(index_path)
        pass1_rows = read_jsonl_gz(pass1_path)
        require_pass1_alignment(pass1_rows, index_rows, pass1_path)
        rows = [freeze_row(index_row) for index_row in index_rows]
        write_shard(rows, split_dir / split / index_path.name)
        producer = producers.get(key)
        if producer is None:
            if pass1_meta:
                raise ValueError(
                    f"pass 1 meta has no producing record for {key}: the freeze would cite "
                    f"this build's SHA for rows it never produced")
            producer = current
        shard_producers[key] = {"code_sha": producer["code_sha"],
                                "pull_request": producer["pull_request"]}

        for frozen, index_row, facts in zip(rows, index_rows, pass1_rows, strict=True):
            by_split[split] += 1
            by_role[frozen["role"]] += 1
            groups[frozen["group_key"]] += 1
            bucket = distribution[frozen["role"]]
            bucket["rows"] += 1
            label = index_row["filename_label"]
            if label is not None:
                bucket["labelled"] += 1
                bucket["surah"][label.split(":", 1)[0]] += 1
                bucket["ayah"][label] += 1
            reasons = exclusion_reasons(facts, floors)
            report = yields[split]
            report["rows"] += 1
            report["codec_failed"] += "codec" in reasons
            report["empty_bytes"] += ("codec" in reasons
                                      and facts["audio_sha256"] == EMPTY_SHA256)
            report["silence"] += "silence" in reasons
            report["usable"] += not reasons

    total = sum(by_split.values())
    largest_key, largest_rows = groups.most_common(1)[0] if groups else ("", 0)
    yield_report = {split: _yield_record(yields[split]) for split in TLOG_SPLITS}
    yield_report["total"] = _yield_record(sum(yields.values(), Counter()))
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source": {"repo": TLOG.repo, "revision": TLOG.revision, "licence": TLOG.licence,
                   "splits": {split: sum(1 for shard in TLOG.shards
                                         if tlog_split(shard) == split)
                              for split in TLOG_SPLITS}},
        **current,
        "preregistration": {"path": _repo_relative(PREREGISTRATION_PATH),
                            "sha256": sha256_file(PREREGISTRATION_PATH)},
        "s2_report": {"path": _repo_relative(S2_REPORT_PATH),
                      "sha256": sha256_file(S2_REPORT_PATH)},
        "pass1": {"path": _repo_relative(pass1_dir),
                  "meta_sha256": (sha256_file(pass1_meta_path)
                                  if pass1_meta_path.exists() else None)},
        "floors": {"path": _repo_relative(floors_path),
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
