"""The tlog metadata index (#185): which clip is which before any audio is read.

``tarteel-ai/tlog`` is 719,853 amateur app clips (``clean`` 398,490, ``unclean`` 321,363) at the
revision :data:`tadabur.nonpro_sources.TLOG` pins. This module reads **only** the ``audio.path``
column of each shard and commits, beside it, the metadata every later slice keys on:

``tools/tadabur/tlog/index/<split>/<shard>.jsonl.gz``
    one row per shard row: ``schema_version``, ``split``, ``shard_file``, ``row``, ``audio_path``,
    ``filename_label``, ``recording_id``, ``id_group``, ``tarteel_v1_eval_overlap`` and
    ``tarteel_v1_any_overlap``. Keys sorted, ``mtime`` 0, written atomically, so a rerun with the
    same code SHA on the **build interpreter** reproduces every byte. gzip's header carries an
    OS byte (index 9) that the zlib binding writes per platform: cuda-dev's Python 3.12.3 wrote
    ``0x03`` into the committed files, and a macOS Python 3.13 writes ``0xff`` with the payload
    otherwise identical. A rerun elsewhere therefore changes that one byte in all 743 files.
``tools/tadabur/tlog/index/meta.json``
    the source pin, the code SHA and the counts the issue posts on #173. ``code_sha`` names the
    commit that built the index, never HEAD: a rerun writes its own, so ``meta.json`` differs by
    that field alone even when every shard is identical. The shard files are the reproducibility
    witness; ``meta.json`` is not.

**The filename is the only label.** A clip's stem splits on ``_``: fields 1-2 are its surah and
ayah, kept as ``"surah:ayah"`` only when ``quran.db`` gives that ayah range; field 3 is the
recording id only when it is all digits (the uuid family's third field is the audio spec,
``float64``, and must not become an id). The dataset's own ``label`` column is never read: the
amateur app's labels are the thing the hold-out is meant to test, not its key.

Rows sharing an ``audio_path``, or a parsed recording id, across both splits are one **id group**,
named by its lexicographically smallest ``audio_path``. A group is the S3b hold-out's unit.

**Tarteel v1 overlap.** ``tarteel_v1_eval_overlap`` is true when the recording id is one of the
8,950 decoded Tarteel v1 clips' ids (``nonpro_eval/tarteel_v1/split.json``, both halves);
``tarteel_v1_any_overlap`` is the same test against all of Tarteel v1's metadata. They coincide
today: #156 decoded every one of its 8,950 rows. Both are reported, never used to drop a row.

**The role** is pre-registered (``docs/tlog-holdout-preregistration.md``) and computed by
:func:`role`; the index does not carry it, the S3b freeze does.

The dataset is gated: the build needs a Hub token that accepted its access terms, so it runs where
that token lives. No audio bytes are read.

Usage (from ``tools/``)::

  python -m tadabur.tlog_index build                 # every shard, into tadabur/tlog/index/
  python -m tadabur.tlog_index build --workers 8     # fewer concurrent shard reads
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import qud_source
from .nonpro_sources import TARTEEL_V1, TLOG, TLOG_SPLITS, Dataset, tlog_split
from .tarteel_split import load_split

SCHEMA_VERSION = 1
#: The checkout whose commit every tlog writer cites.
REPO_ROOT = Path(__file__).resolve().parents[2]
#: Where the committed index lives (``tools/tadabur/tlog/index``).
INDEX_DIR = Path(__file__).parent / "tlog" / "index"
#: The repo's canonical ``quran.db``: the ayah counts the filename labels are checked against.
DEFAULT_QURAN_DB = REPO_ROOT / "data" / "quran.db"

# --- the pre-registered role (docs/tlog-holdout-preregistration.md) --------------------------

ROLE_SALT = "tlog-holdout-2026-10"
LOOK, CONFIRM_1, CONFIRM_2, CONFIRM_3, CONFIRM_4, TRAIN = (
    "look", "confirm.1", "confirm.2", "confirm.3", "confirm.4", "train")
ROLES = (LOOK, CONFIRM_1, CONFIRM_2, CONFIRM_3, CONFIRM_4, TRAIN)
#: The role boundaries: ``u`` of 0.02, 0.04, 0.06, 0.08 and 0.10, as the integers its 64-bit
#: numerator is compared against (``u = value / 2**64``).
ROLE_CUTOFFS = tuple(2**64 * percent // 100 for percent in (2, 4, 6, 8, 10))


def group_hash(group_key: str) -> int:
    """The 64-bit numerator of a group's ``u``: the first 8 bytes, big-endian, of
    ``sha256("tlog-holdout-2026-10:" + group_key)``."""
    return int.from_bytes(
        hashlib.sha256(f"{ROLE_SALT}:{group_key}".encode("utf-8")).digest()[:8], "big")


def role_from_hash(value: int) -> str:
    """The role a 64-bit numerator takes (:data:`ROLE_CUTOFFS`, half-open ranges)."""
    if not 0 <= value < 2**64:
        raise ValueError(f"a role value is a 64-bit numerator, got {value}")
    for name, cutoff in zip(ROLES, ROLE_CUTOFFS):
        if value < cutoff:
            return name
    return TRAIN


def role(group_key: str) -> str:
    """The pre-registered role of an id group, which is also its S3b group key."""
    return role_from_hash(group_hash(group_key))


# --- the filename rule -----------------------------------------------------------------------


def ayah_counts(quran_db: Path) -> dict[int, int]:
    """Every surah's ayah count from ``quran.db`` (read-only; 114 rows)."""
    with sqlite3.connect(f"file:{quran_db}?mode=ro", uri=True) as connection:
        return dict(connection.execute("SELECT id, verses_count FROM surahs"))


def _stem(audio_path: str) -> str:
    """The file name without its directory or extension (tlog's paths are flat)."""
    return audio_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def filename_label(audio_path: str, counts: Mapping[int, int]) -> str | None:
    """``"surah:ayah"`` from a clip's name, or ``None`` when the name gives no valid ayah.

    The name is ``<surah>_<ayah>[_<recording id>][_<suffix>].<ext>``; the ayah must be inside its
    surah per ``counts``. ``unclean``'s ``-1_-1_<id>.wav`` and the uuid family carry none.
    """
    fields = _stem(audio_path).split("_")
    if len(fields) < 2 or not (fields[0].isdigit() and fields[1].isdigit()):
        return None
    surah, ayah = int(fields[0]), int(fields[1])
    if surah not in counts or not 1 <= ayah <= counts[surah]:
        return None
    return f"{surah}:{ayah}"


def recording_id(audio_path: str) -> str | None:
    """The name's third field when the name has fields and it is all digits, else ``None``.

    ``null``/``undefined`` rows lost it; the uuid family's third field is ``float64``, part of the
    audio spec, never an id. A name with a leading empty field (``__0.wav``) has no fields at all.
    """
    fields = _stem(audio_path).split("_")
    if len(fields) < 3 or not (fields[0] and fields[1]) or not fields[2].isdigit():
        return None
    return fields[2]


def clip_ids(clip_paths: Iterable[str]) -> frozenset[str]:
    """The recording id of every named clip (Tarteel v1's clip ids are its audio paths)."""
    return frozenset(ids for ids in (recording_id(path) for path in clip_paths) if ids)


def tarteel_v1_ids(record: Mapping[str, object]) -> frozenset[str]:
    """The ids of Tarteel v1's clips in a frozen split record (both halves)."""
    return clip_ids([*record["look"], *record["confirm"]])


def overlap_flags(recording_id: str | None, eval_ids: frozenset[str],
                  any_ids: frozenset[str]) -> tuple[bool, bool]:
    """``(in the decoded eval clips, in any Tarteel v1 clip)`` for one recording id."""
    return (recording_id is not None and recording_id in eval_ids,
            recording_id is not None and recording_id in any_ids)


# --- id groups -------------------------------------------------------------------------------


def id_groups(rows: Iterable[Mapping[str, object]]) -> dict[str, str]:
    """Each ``audio_path`` mapped to its id group's key.

    Rows sharing an ``audio_path``, or a parsed ``recording_id``, are one component across both
    splits; the key is the component's lexicographically smallest ``audio_path``.
    """
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    paths = []
    for row in rows:
        path = str(row["audio_path"])
        node = ("path", path)
        find(node)
        if row.get("recording_id") is not None:
            recording = ("rid", str(row["recording_id"]))
            left, right = find(node), find(recording)
            if left != right:
                parent[left] = right
        paths.append(path)
    smallest: dict[tuple[str, str], str] = {}
    for path in set(paths):
        root = find(("path", path))
        if root not in smallest or path < smallest[root]:
            smallest[root] = path
    return {path: smallest[find(("path", path))] for path in paths}


# --- reading and writing ---------------------------------------------------------------------


def fetch_paths(shard: str, dataset: Dataset = TLOG) -> list[str]:
    """A shard's ``audio.path`` values, in row order, and nothing else of it.

    Only the path column is requested: the dataset's ``label`` column is never read.
    """
    table = qud_source.parquet_table(dataset.repo, dataset.revision, shard, ["audio.path"])
    return [row["audio"]["path"] for row in table.to_pylist()]


def write_shard(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write one shard's rows as JSON lines, gzipped without a timestamp.

    The bytes are deterministic on one interpreter: the same rows written twice are identical,
    but gzip's OS header byte is platform-dependent (module docstring), so another platform
    rewrites that byte and nothing else.
    """
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode("utf-8")
    _write_atomically(path, gzip.compress(text, mtime=0))


def _write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def head_sha(repo: Path = REPO_ROOT) -> str:
    """The code SHA ``meta.json`` records, unless the caller pins one."""
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def provenance(code_sha: str, pull_request: int | None = None) -> dict:
    """The commit a measurement ran from, and the PR ref that keeps it after the squash merge.

    One definition for the index, pass 1 and the freeze: the branch is squash-merged, so a
    cited SHA survives only under ``refs/pull/<N>/head`` and is recorded with its PR number.
    The caller resolves the SHA (:func:`head_sha`, or a pinned ``--code-sha``).
    """
    return {"code_sha": code_sha, "pull_request": pull_request,
            "pull_request_ref": (f"refs/pull/{pull_request}/head"
                                 if pull_request is not None else None)}


def repo_relative(path: Path, repo: Path = REPO_ROOT) -> str:
    """``path`` inside the checkout as a repo-relative string; a file outside it unchanged."""
    try:
        return str(Path(path).resolve().relative_to(repo))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    """The SHA-256 of a file's bytes — the one definition every tlog writer cites."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def installed_versions(names: Iterable[str]) -> dict[str, str | None]:
    """The installed versions of ``names``, ``None`` where a package is absent.

    Shared by the index/S2 metadata and the pass-1 ``decoders`` record, so one provenance
    idiom reports what ran instead of three.
    """
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str | None] = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def require_clean_tree(repo: Path = Path(__file__).resolve().parents[2]) -> None:
    """Refuse a checkout with uncommitted or untracked changes.

    The committed metadata cites the commit the measurement ran from; a dirty tree would make
    that provenance a lie, so the measuring commands check this before they download anything.
    Generated outputs therefore go to scratch (``--out-dir``), never into the measured tree.
    """
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                            capture_output=True, text=True)
    if status.returncode != 0:
        raise RuntimeError(f"{repo} is not a readable git checkout: {status.stderr.strip()}")
    if status.stdout:
        raise RuntimeError("the checkout is dirty; commit or remove these paths first:\n"
                           + status.stdout)


# --- the build -------------------------------------------------------------------------------


def build(index_dir: Path = INDEX_DIR, quran_db: Path = DEFAULT_QURAN_DB,
          code_sha: str | None = None, workers: int = 16) -> dict:
    """Read every shard's paths, group them, and write the index; return its ``meta.json``."""
    counts = ayah_counts(quran_db)
    with ThreadPoolExecutor(workers) as pool:
        shard_paths = list(pool.map(fetch_paths, TLOG.shards))
    # (split, shard file stem, row, audio_path, filename_label, recording_id), in shard order
    records = [(tlog_split(shard), Path(shard).stem, index, path,
                filename_label(path, counts), recording_id(path))
               for shard, paths in zip(TLOG.shards, shard_paths, strict=True)
               for index, path in enumerate(paths)]
    groups = id_groups({"audio_path": record[3], "recording_id": record[5]}
                       for record in records)
    eval_ids = tarteel_v1_ids(load_split())
    any_ids = clip_ids(path for shard in TARTEEL_V1.shards
                       for path in fetch_paths(shard, TARTEEL_V1))

    totals: Counter = Counter()
    overlaps: Counter = Counter()
    labels = with_id = 0
    cursor = 0
    for shard, paths in zip(TLOG.shards, shard_paths, strict=True):
        split, stem = tlog_split(shard), Path(shard).stem
        rows = []
        for split_, shard_, index, path, label, recording in records[cursor:cursor + len(paths)]:
            eval_overlap, any_overlap = overlap_flags(recording, eval_ids, any_ids)
            rows.append({
                "schema_version": SCHEMA_VERSION, "split": split_, "shard_file": shard_,
                "row": index, "audio_path": path, "filename_label": label,
                "recording_id": recording, "id_group": groups[path],
                "tarteel_v1_eval_overlap": eval_overlap,
                "tarteel_v1_any_overlap": any_overlap,
            })
            labels += label is not None
            with_id += recording is not None
            overlaps["eval"] += eval_overlap
            overlaps["any"] += any_overlap
        cursor += len(paths)
        write_shard(rows, index_dir / split / f"{stem}.jsonl.gz")
        totals[split] += len(rows)
        print(f"{split}/{stem}: {len(rows)} rows", flush=True)

    # Row-level, not path-level: rows repeating an audio_path are rows on a duplicated id too.
    row_groups = Counter(groups[record[3]] for record in records)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source": {"repo": TLOG.repo, "revision": TLOG.revision,
                   "splits": {split: sum(1 for shard in TLOG.shards
                                         if tlog_split(shard) == split)
                              for split in TLOG_SPLITS},
                   "licence": TLOG.licence},
        "code_sha": code_sha or head_sha(),
        "counts": {
            "rows": {**{split: totals[split] for split in TLOG_SPLITS},
                     "total": sum(totals.values())},
            "filename_labels": labels,
            "recording_ids": with_id,
            "id_groups": len(row_groups),
            "rows_on_duplicated_ids": sum(size for size in row_groups.values() if size > 1),
            "tarteel_v1_eval_overlap": overlaps["eval"],
            "tarteel_v1_any_overlap": overlaps["any"],
        },
    }
    _write_atomically(index_dir / "meta.json",
                      (json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True)
                       + "\n").encode("utf-8"))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build", help="read every shard's paths and write the index")
    builder.add_argument("--index-dir", type=Path, default=INDEX_DIR,
                         help="where the index is written (default: %(default)s)")
    builder.add_argument("--quran-db", type=Path, default=DEFAULT_QURAN_DB,
                         help="quran.db for the ayah counts (default: %(default)s)")
    builder.add_argument("--code-sha", default=None,
                         help="the code SHA meta.json records (default: git rev-parse HEAD)")
    builder.add_argument("--workers", type=int, default=16,
                         help="concurrent shard reads (default: %(default)s)")
    args = parser.parse_args()
    meta = build(args.index_dir, args.quran_db, args.code_sha, args.workers)
    print(json.dumps(meta["counts"], indent=1), flush=True)
    if meta["counts"]["rows"]["total"] == 0:
        sys.exit("no rows read")


if __name__ == "__main__":
    main()
