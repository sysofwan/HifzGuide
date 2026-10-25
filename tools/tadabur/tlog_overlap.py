"""Audio overlap against every frozen eval set, and tlog-internal duplicates (#189).

The pre-registration's ``eval_overlap:<set>`` and ``duplicate`` exclusions, decided by audio
rather than by name. S1 removed literal recording-id matches with Tarteel v1; this stage
fingerprints every frozen evaluation set (:data:`SETS`) and every tlog row, looks candidates
up in a Haitsma–Kalker inverted index of the 32-bit sub-fingerprints
(:mod:`tadabur.qud_fingerprint`), and verifies each candidate with ``best_match`` at the
committed threshold (:data:`LINE`).

**External sets.** One fingerprint store per set on scratch (``<fp-dir>/<set>/<unit>.npz``,
one npz per source shard), resumable unit by unit, with a control-candidate subdirectory of
the same units (:func:`external`). The audio is streamed and deleted; only a fingerprint and
the clip's id are kept. Every set's clip ids come from the committed checksum lists where
one exists — ``exposure/nonpro.evaluation_only.clips.jsonl`` (Tarteel v1, qdat, qdat_bench,
RetaSy) and ``exposure/qud.correct_recitation.qud.jsonl`` (the two QUD pilot mushafs) — so
the store is checked against a committed population, not against itself. The frame sets have
no source-byte checksums: every row of ``held_out_shards()`` is fingerprinted, and a row is
labelled ``gate_eval`` when it is a ``decode_evalset.{dev,test}`` row, else ``tadabur_frame``.
The 197 sealed-panel clips cannot be read (the seal's two authorisations are pinned), so they
are skipped and recorded as a blind spot.

**Threshold.** :data:`LINE` is the committed line from ``qud_eval/overlap_rules.json`` (0.238,
halfway between the controls' highest synthetic positive and lowest hard negative), reused
with its file SHA-256 recorded in ``overlap/meta.json``. It is not re-derived: these sets offer
no clean hard negatives. The positives are re-checked instead, by a required run-level control
:func:`synthetic_recall_control` — at least :data:`CONTROL_COPIES` re-encoded, re-cut copies of
external clips through the real index; the run refuses to write anything unless ≥ 99% are found
and every copy's BER ≤ the line.

**Silence.** Digital silence makes every band difference zero, so the fingerprint emits the
word ``0``; two silent clips then match at BER 0. :data:`STOP_WORDS` (``0`` and ``0xFFFFFFFF``)
are therefore never indexed and never queried, and every verified match reports the share of
its overlap window that is a stop-word in either fingerprint.

**Duplicates.** Exact duplicates are equal ``audio_sha256`` across all rows (empty bytes are
never a key). Near duplicates index the frozen eval-role rows only and query every row; a
train–train near pair is not searched and that gap is recorded. Resolution is by component:
one survivor, earliest-opened eval role first (``look`` < ``confirm.1`` … < ``confirm.4`` <
``train``), then the smallest ``(audio_path, split, shard_file, row)``; every other member
names that survivor. No eval row can lose to a training copy, and train–train exact duplicates
are recorded too (the pre-registration's within-role rule applies to every role).

``tools/tadabur/tlog/overlap/<split>/<shard>.jsonl.gz``
    one record per frozen row, keyed ``(split, shard_file, row)``: ``eval_overlap`` (every set
    hit, the best match per set, by :data:`SETS` order), ``duplicate_of`` (the survivor's row
    key or null), ``duplicate_ber`` (0 for an exact checksum) and ``excluded`` (S4's own
    primary reason or null). S4 never rewrites another stage's facts: a row excluded by codec,
    silence or S1's id overlap still gets its S4 record.
``tools/tadabur/tlog/overlap/meta.json``
    the set census and provenance, the threshold and its source SHA-256, the vote threshold
    and stop-words, both run controls, hits per set and role, duplicates by role pair, the
    net-size tables, the blind spots, and the frozen inputs' digests.

Usage (from ``tools/``; the external pass and the build are separate, resumable steps)::

  python -m tadabur.tlog_overlap external --fp-dir /root/scratch/tlog/overlap/fp
  python -m tadabur.tlog_overlap build --pr <N> --fp-dir /root/scratch/tlog/overlap/fp

``scripts/issue-189-overlap.sh`` stages the branch on cuda-dev, launches both detached and
fetches the committed tree back. The tlog pass-1 fingerprint store
(``/root/scratch/tlog/fp``) is only read, never written; no audio is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np

from training.tashkeel_eval import write_text_atomically

from .audio import decode_to_mono_16k
from .dataset_source import DATASET_ID
from .nonpro_sources import (QDAT, QDAT_BENCH, RETASY, TARTEEL_V1, LeftOut, iter_rows,
                             select)
from .panel_seal import is_panel_name, panel_provenance
from .qud_fingerprint import best_match, fingerprint
from .qud_source import QUD_REVISION, QUD_DATASET, audio_filename as qud_audio_filename
from .tlog_index import (INDEX_DIR, ROLES, TRAIN, head_sha, provenance, repo_relative,
                         sha256_file, write_shard)
from .tlog_pass1 import FLOORS_PATH, PASS1_DIR, read_jsonl_gz
from .tlog_split import EMPTY_SHA256, ROW_KEY, iter_frozen_rows

SCHEMA_VERSION = 1
#: The committed tree this module writes (``tools/tadabur/tlog/overlap``).
OVERLAP_DIR = Path(__file__).parent / "tlog" / "overlap"
META_NAME = "meta.json"
#: The set codes, in the order that breaks a tie for the primary ``excluded`` reason.
SETS = ("tarteel_v1", "qud", "gate_eval", "tadabur_frame", "qdat", "qdat_bench", "retasy")
#: The frame sets: one audio pass over ``held_out_shards()``, labelled by gate membership.
FRAME_SETS = ("gate_eval", "tadabur_frame")
#: Digital silence's word, and the all-ones word: never indexed, never queried.
STOP_WORDS = frozenset({0, 0xFFFFFFFF})
RULES_PATH = Path(__file__).parent / "qud_eval" / "overlap_rules.json"
RULES = json.loads(RULES_PATH.read_text(encoding="utf-8"))
#: ``qud_overlap``'s committed line, reused (the module docstring says why).
LINE = float(RULES["lines"]["fingerprint"]["line"])
#: The controls' highest synthetic-positive BER: matches above it are the ambiguous zone.
SYNTHETIC_POSITIVE_MAX = float(RULES["lines"]["fingerprint"]["synthetic_positive_max"])

EXPOSURE_DIR = Path(__file__).parent / "exposure"
NONPRO_CHECKSUMS_PATH = EXPOSURE_DIR / "nonpro.evaluation_only.clips.jsonl"
QUD_CHECKSUMS_PATH = EXPOSURE_DIR / "qud.correct_recitation.qud.jsonl"
GATE_EVAL_PATHS = tuple(EXPOSURE_DIR / f"decode_evalset.{half}.jsonl"
                        for half in ("dev", "test"))
SELECTION_PATH = Path(__file__).parent / "qud_eval" / "selection.json"
#: cuda-dev scratch: the external store, S3b's tlog fingerprints, and the audio cache.
FP_DIR = Path("/root/scratch/tlog/overlap/fp")
STORE_DIR = Path("/root/scratch/tlog/fp")
CACHE_DIR = Path("/root/scratch/tlog/cache")

#: Candidate generation: exact 32-bit key votes at one aligned offset before ``best_match``.
MIN_VOTES = 8
#: Synthetic-positive copies per set in the run's recall control (7 sets, so 210 copies).
CONTROL_COPIES = 30
CONTROL_SALT = "tlog-s4-control"
#: A verified match is "mostly stop-word" at this share of its overlap window.
MOSTLY_STOP_WORD = 0.5
#: The panel's committed size (its clips are sealed; the frame fingerprint pass skips them).
SEALED_PANEL = 197
#: Offset encoding of a (clip, time) vote: shifts are frame indices, bounded by ±:data:`OFFSET`.
OFFSET = 1 << 17
MOD = 1 << 18
#: Bindings of the two committed dataset checksum sources to their set codes.
NONPRO_SET_CODES = {TARTEEL_V1.repo: "tarteel_v1", QDAT.repo: "qdat",
                    QDAT_BENCH.repo: "qdat_bench", RETASY.repo: "retasy"}
_NONPRO_DATASETS = {dataset.repo: dataset for dataset in (TARTEEL_V1, QDAT, QDAT_BENCH, RETASY)}
#: The frame rows' source: Tadabur's full-config shards, at whatever the reader fetches.
TADABUR_SOURCE = DATASET_ID

_ROW_SEP = "\x00"
_STOP = np.array(sorted(STOP_WORDS), dtype=np.uint32)
BER_BIN = 0.05
_BINS = tuple(round(BER_BIN * index, 2) for index in range(20))


# --- the index ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    """One verified match: the indexed clip, its BER at the best offset, and the overlap's
    share of stop-word frames in either fingerprint."""

    clip: str
    ber: float
    offset: int
    overlap: int
    stop_word_share: float


def _stop_share(query: np.ndarray, reference: np.ndarray, offset: int) -> float:
    """The share of ``best_match``'s overlap window that is a stop-word in either fingerprint."""
    low, high = max(0, offset), min(len(query), offset + len(reference))
    if high <= low:
        return 0.0
    window = (np.isin(query[low:high], _STOP)
              | np.isin(reference[low - offset:high - offset], _STOP))
    return float(window.mean())


class EvalIndex:
    """A Haitsma–Kalker inverted index over 32-bit sub-fingerprints.

    Every frame of every indexed clip is one posting ``(key, clip, time)``, sorted by key, so a
    query frame's postings are a ``searchsorted`` range. Postings whose words are stop-words
    are dropped on both sides: silence would otherwise dominate both the index and the votes.
    Candidate pairs need :data:`MIN_VOTES` frames bit-identical at the same time offset; each
    surviving clip is verified with :func:`best_match`, never accepted on votes alone.
    """

    def __init__(self, clips: tuple[str, ...], fps: dict[str, np.ndarray],
                 keys: np.ndarray, clip_idx: np.ndarray, times: np.ndarray,
                 unique: np.ndarray, starts: np.ndarray) -> None:
        self._clips = clips
        self._fps = fps
        self._keys = keys
        self._clip_idx = clip_idx
        self._times = times
        self._unique = unique
        self._starts = starts
        self._ends = np.append(starts[1:], len(keys))

    @classmethod
    def build(cls, prints: Mapping[str, np.ndarray], *,
              stop_words: Iterable[int] = STOP_WORDS) -> EvalIndex:
        """The index of ``clip -> uint32 fingerprint`` (clips sorted, for a stable build)."""
        items = [(clip, np.asarray(prints[clip], dtype=np.uint32))
                 for clip in sorted(prints) if len(prints[clip])]
        if not items:
            empty_u32 = np.zeros(0, dtype=np.uint32)
            return cls((), {}, empty_u32, empty_u32, empty_u32, empty_u32,
                       np.zeros(0, dtype=np.int64))
        keys = np.concatenate([fp for _, fp in items])
        clip_idx = np.concatenate([np.full(len(fp), index, dtype=np.uint32)
                                   for index, (_, fp) in enumerate(items)])
        times = np.concatenate([np.arange(len(fp), dtype=np.uint32) for _, fp in items])
        keep = ~np.isin(keys, np.asarray(sorted(stop_words), dtype=np.uint32))
        keys, clip_idx, times = keys[keep], clip_idx[keep], times[keep]
        order = np.argsort(keys, kind="stable")
        keys, clip_idx, times = keys[order], clip_idx[order], times[order]
        unique, starts = (np.unique(keys, return_index=True) if len(keys)
                          else (np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.int64)))
        return cls(tuple(clip for clip, _ in items), {clip: fp for clip, fp in items},
                   keys, clip_idx, times, unique, starts)

    def __len__(self) -> int:
        """The number of indexed frames (after the stop-word filter)."""
        return int(len(self._keys))

    def query(self, prints: np.ndarray, *, line: float = LINE, min_votes: int = MIN_VOTES,
              exclude: str | None = None) -> list[Hit]:
        """Every clip verified within ``line``, best BER first; stops below ``min_votes``.

        ``exclude`` is one clip id never verified — the eval-row index's own query would
        otherwise spend a full ``best_match`` on the row against itself.
        """
        query = np.asarray(prints, dtype=np.uint32)
        if not len(self._keys) or not len(self._unique) or query.size == 0 or min_votes < 1:
            return []
        frame = np.flatnonzero(~np.isin(query, _STOP))
        if not frame.size:
            return []
        keys = query[frame]
        low = np.searchsorted(self._unique, keys, side="left")
        present = low < len(self._unique)
        present[present] = self._unique[low[present]] == keys[present]
        low, frame = low[present], frame[present]
        counts = self._ends[low] - self._starts[low]
        total = int(counts.sum())
        if not total:
            return []
        position = np.repeat(self._starts[low], counts) + (
            np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts))
        clip = self._clip_idx[position].astype(np.int64)
        shift = np.repeat(frame, counts).astype(np.int64) - self._times[position]
        if np.any(np.abs(shift) >= OFFSET):
            raise ValueError("a fingerprint is longer than the vote encoding supports")
        votes, count = np.unique(clip * MOD + (shift + OFFSET), return_counts=True)
        chosen = count >= min_votes
        best: dict[int, int] = {}
        for encoded, number in zip(votes[chosen], count[chosen].tolist()):
            index = int(encoded) // MOD
            if number > best.get(index, 0):
                best[index] = number
        hits = []
        for index in sorted(best):
            clip_id = self._clips[index]
            if clip_id == exclude:
                continue
            reference = self._fps[clip_id]
            match = best_match(query, reference)
            if match is None or match.ber > line:
                continue
            hits.append(Hit(clip_id, match.ber, match.offset, match.overlap,
                            _stop_share(query, reference, match.offset)))
        hits.sort(key=lambda hit: (hit.ber, hit.clip))
        return hits


# --- duplicate resolution ----------------------------------------------------------------------


def _key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (row["split"], row["shard_file"], row["row"])


def _row_clip(key: tuple[str, str, int]) -> str:
    return _ROW_SEP.join((key[0], key[1], str(key[2])))


def _clip_row(clip: str) -> tuple[str, str, int]:
    split, shard, row = clip.split(_ROW_SEP)
    return split, shard, int(row)


def _role_rank(role: str) -> int:
    return ROLES.index(role)


def _exact_links(groups: Mapping[str, Sequence[tuple[str, str, int]]]
                 ) -> list[tuple[tuple[str, str, int], tuple[str, str, int]]]:
    """One star per equal-checksum group, from its smallest row key."""
    pairs = []
    for digest in sorted(groups):
        keys = sorted(groups[digest])
        pairs.extend((keys[0], key) for key in keys[1:])
    return pairs


def exact_duplicate_pairs(rows: Iterable[Mapping[str, Any]]
                          ) -> list[tuple[tuple[str, str, int], tuple[str, str, int]]]:
    """Every pair inside an equal non-empty ``audio_sha256`` group, a star from the smallest
    row key; ``EMPTY_SHA256`` and null checksums are never keys (empty bytes are not audio)."""
    groups: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for row in rows:
        digest = row.get("audio_sha256")
        if digest and digest != EMPTY_SHA256:
            groups[digest].append(_key(row))
    return _exact_links(groups)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Any, Any] = {}

    def find(self, item: Any) -> Any:
        parent = self.parent.setdefault(item, item)
        while parent != item:
            grandparent = self.parent[parent]
            self.parent[item] = grandparent
            item, parent = parent, grandparent
        return item

    def union(self, first: Any, second: Any) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[right] = left

    def components(self, members: Iterable[Any]) -> list[list[Any]]:
        grouped: dict[Any, list[Any]] = defaultdict(list)
        for member in members:
            grouped[self.find(member)].append(member)
        return list(grouped.values())


def resolve_duplicates(rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
                       links: Iterable[tuple[Any, Any]]
                       ) -> dict[Any, Any]:
    """Every non-survivor's survivor, by the pre-registered precedence.

    One survivor per component: the earliest-opened eval role (``look``, then ``confirm.1`` …
    ``confirm.4``, then ``train``), then the smallest ``(audio_path, split, shard_file, row)``.
    ``audio_path`` is not unique, so the full row key breaks its ties. The result is a forest:
    every loser names the survivor directly, never an intermediate member.
    """
    union = _UnionFind()
    members = []
    for first, second in links:
        union.union(first, second)
        members.extend((first, second))
    resolved = {}
    for component in union.components(members):
        survivor = min(component, key=lambda key: (_role_rank(rows[key]["role"]),
                                                  rows[key]["audio_path"], key))
        resolved.update({key: survivor for key in component if key != survivor})
    return resolved


# --- the per-row record ----------------------------------------------------------------------


def overlap_record(row: Mapping[str, Any], hits: Iterable[Mapping[str, Any]], *,
                   duplicate_of: Mapping[str, Any] | None = None,
                   duplicate_ber: float | None = None) -> dict[str, Any]:
    """One committed S4 record: the best hit per set (by :data:`SETS` order), one primary
    reason, and the duplicate fact. Every set hit is kept; ``excluded`` is only a label."""
    best: dict[str, dict[str, Any]] = {}
    for hit in hits:
        code = hit["set"]
        if code not in best or hit["ber"] < best[code]["ber"]:
            best[code] = {"set": code, "clip": hit["clip"], "ber": hit["ber"]}
    ordered = [best[code] for code in SETS if code in best]
    escaped = (f"eval_overlap:{ordered[0]['set']}" if ordered
               else ("duplicate" if duplicate_of is not None else None))
    return {
        "schema_version": SCHEMA_VERSION,
        "split": row["split"],
        "shard_file": row["shard_file"],
        "row": row["row"],
        "eval_overlap": ordered,
        "duplicate_of": dict(duplicate_of) if duplicate_of is not None else None,
        "duplicate_ber": duplicate_ber,
        "excluded": escaped,
    }


# --- external sets: sources, checksums, store -------------------------------------------------


@dataclass(frozen=True)
class SetSource:
    code: str
    source: str
    revision: str | None
    checksums: Path | None


def set_sources() -> dict[str, SetSource]:
    """Every set's pinned source and, where it exists, its committed checksum list."""
    sources = {
        "tarteel_v1": SetSource("tarteel_v1", TARTEEL_V1.repo, TARTEEL_V1.revision,
                                NONPRO_CHECKSUMS_PATH),
        "qud": SetSource("qud", QUD_DATASET, QUD_REVISION, QUD_CHECKSUMS_PATH),
        "gate_eval": SetSource("gate_eval", TADABUR_SOURCE, None, None),
        "tadabur_frame": SetSource("tadabur_frame", TADABUR_SOURCE, None, None),
        "qdat": SetSource("qdat", QDAT.repo, QDAT.revision, NONPRO_CHECKSUMS_PATH),
        "qdat_bench": SetSource("qdat_bench", QDAT_BENCH.repo, QDAT_BENCH.revision,
                                NONPRO_CHECKSUMS_PATH),
        "retasy": SetSource("retasy", RETASY.repo, RETASY.revision, NONPRO_CHECKSUMS_PATH),
    }
    if set(sources) != set(SETS):
        raise ValueError(f"set registry {sorted(sources)} is not {list(SETS)}")
    return sources


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def committed_clips() -> dict[str, dict[str, str]]:
    """``set -> clip -> audio_sha256`` from the committed checksum lists."""
    clips: dict[str, dict[str, str]] = {code: {} for code in SETS}
    for record in _read_jsonl(NONPRO_CHECKSUMS_PATH):
        code = NONPRO_SET_CODES.get(record["dataset"])
        if code is None:
            raise ValueError(f"unexpected checksum dataset {record['dataset']}")
        clips[code][record["audio_filename"]] = record["audio_sha256"]
    for record in _read_jsonl(QUD_CHECKSUMS_PATH):
        surah, ayah = (int(part) for part in record["surah_ayah"].split(":"))
        name = qud_audio_filename(record["slug"], surah, ayah)
        clips["qud"][name] = record["mp3_sha256"]
    return clips


def _sha_map(clips: Mapping[str, Mapping[str, str]]
             ) -> dict[str, list[tuple[str, str]]]:
    """``sha256 -> [(set, clip)]``, the smallest clip per set, in :data:`SETS` order."""
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for code in SETS:
        for clip, digest in clips.get(code, {}).items():
            if digest not in grouped[code] or clip < grouped[code][digest]:
                grouped[code][digest] = clip
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for code in SETS:
        for digest, clip in grouped[code].items():
            out[digest].append((code, clip))
    return out


def gate_membership() -> dict[tuple[int, int], str]:
    """The ``decode_evalset.{dev,test}`` rows: ``(shard, row) -> audio_filename``.

    The filename is carried so the frame pass can refuse a row-index drift between the
    committed exposure file and the shard reader, rather than mislabel a frame row.
    """
    rows = _read_jsonl(GATE_EVAL_PATHS[0]) + _read_jsonl(GATE_EVAL_PATHS[1])
    return {(int(row["shard"]), int(row["row_index"])): row["audio_filename"] for row in rows}


def _external_wanted(clips: Mapping[str, Mapping[str, str]]) -> dict[str, set[str]]:
    return {code: set(clips[code]) for code in SETS if code not in FRAME_SETS}


@dataclass
class _Unit:
    """One source shard's fingerprints-to-be: its clips, and the panel rows it must skip."""

    name: str
    clips: Iterator[tuple[str, np.ndarray]]
    skipped: list[str]


def _nonpro_unit(dataset, shard: str, cache_dir: Path, wanted: set[str]) -> Iterator[
        tuple[str, np.ndarray]]:
    for record in iter_rows(dataset, shard, cache_dir, strict_delete=True):
        if dataset is QDAT_BENCH:
            clip, raw = record["id"], record["audio"]["bytes"]
        else:
            clip = select(dataset, record)
            if isinstance(clip, LeftOut):
                continue
            clip, raw = clip.audio_filename, clip.audio
        if clip in wanted:
            yield clip, decode_to_mono_16k(raw)


def _nonpro_units(code: str, cache_dir: Path, wanted: set[str]) -> Iterator[_Unit]:
    dataset = _NONPRO_DATASETS[set_sources()[code].source]
    for shard in dataset.shards:
        yield _Unit(Path(shard).stem, _nonpro_unit(dataset, shard, cache_dir, wanted), [])


def _qud_unit(slug: str, shard: str, cache_dir: Path,
              wanted: set[str]) -> Iterator[tuple[str, np.ndarray]]:
    from .qud_source import decode_ayah, iter_mushaf_rows

    for row in iter_mushaf_rows(slug, [shard], cache_dir):
        name = row.audio_filename
        if name in wanted:
            yield name, decode_ayah(row)


def _qud_units(cache_dir: Path, wanted: set[str]) -> Iterator[_Unit]:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    shards_by_slug = {mushaf["slug"]: mushaf["shards"] for mushaf in selection["mushafs"]}
    slugs = sorted({record["slug"] for record in _read_jsonl(QUD_CHECKSUMS_PATH)})
    for slug in slugs:
        if slug not in shards_by_slug:
            raise ValueError(f"{slug} is not in {SELECTION_PATH}")
        for shard in shards_by_slug[slug]:
            yield _Unit(f"{slug}-{Path(shard).stem}",
                        _qud_unit(slug, shard, cache_dir, wanted), [])


def _frame_unit(shard: int, cache_dir: Path, skipped: list[str], panel,
                gate: Mapping[tuple[int, int], str]) -> Iterator[tuple[str, np.ndarray]]:
    from .shard_reader import iter_shard_rows

    for position, row in enumerate(iter_shard_rows([shard], cache_dir=cache_dir,
                                                   delete_after=True)):
        path = (row.get("audio") or {}).get("path") or ""
        expected = gate.get((shard, position))
        if expected is not None and Path(path).name != expected:
            raise ValueError(f"shard {shard} row {position} is {Path(path).name}, "
                             f"not gate_eval's {expected}: the frame rows drifted")
        if is_panel_name(path, panel):
            skipped.append(path)
            continue
        yield f"{shard}:{position}", decode_to_mono_16k(row["audio"]["bytes"])


def external_units(code: str, *, cache_dir: Path,
                   wanted: Mapping[str, set[str]]) -> Iterator[_Unit]:
    """Every unit of one non-frame set, its clips decoded to 16 kHz mono."""
    if code == "qud":
        return _qud_units(cache_dir, wanted[code])
    return _nonpro_units(code, cache_dir, wanted[code])


def _control_rank(code: str, clip: str) -> int:
    digest = hashlib.sha256(f"{CONTROL_SALT}{_ROW_SEP}{code}{_ROW_SEP}{clip}".encode())
    return int.from_bytes(digest.digest()[:8], "big")


def _save_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """One npz written atomically: a visible file is always a complete unit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def _members(path: Path) -> int:
    """The unit's member count, without decompressing its arrays."""
    with np.load(path) as data:
        return len(data.files)


class _ControlPicker:
    """A unit's lowest-ranked control candidates, bounded to :data:`CONTROL_COPIES`."""

    def __init__(self, code: str) -> None:
        self.code = code
        self._heap: list[tuple[int, str]] = []
        self._waves: dict[str, np.ndarray] = {}

    def add(self, clip: str, waveform: np.ndarray) -> None:
        rank = _control_rank(self.code, clip)
        if len(self._heap) < CONTROL_COPIES:
            heapq.heappush(self._heap, (-rank, clip))
            self._waves[clip] = waveform
        elif -rank > self._heap[0][0]:
            _, dropped = heapq.heapreplace(self._heap, (-rank, clip))
            self._waves.pop(dropped, None)
            self._waves[clip] = waveform

    def arrays(self) -> dict[str, np.ndarray]:
        return {clip: waveform.astype(np.float32)
                for clip, waveform in self._waves.items()}


def _write_unit(set_dir: Path, unit: _Unit, code: str) -> tuple[int, int]:
    """Fingerprint one unit, save it with its control candidates; ``(members, skipped)``."""
    prints: dict[str, np.ndarray] = {}
    picker = _ControlPicker(code)
    for clip, waveform in unit.clips:
        prints[clip] = fingerprint(waveform)
        picker.add(clip, waveform)
    members = int(sum(1 for fp in prints.values() if len(fp)))
    _save_arrays(set_dir / f"{unit.name}.npz", {clip: fp for clip, fp in prints.items()
                                                if len(fp)})
    _save_arrays(set_dir / "control" / f"{unit.name}.npz", picker.arrays())
    print(f"[external] {code}/{unit.name}: {members} fingerprints", flush=True)
    return members, len(unit.skipped)


def _frame_units(cache_dir: Path, panel,
                 gate: Mapping[tuple[int, int], str]) -> Iterator[tuple[int, _Unit]]:
    from training.distill_stream import held_out_shards

    for shard in sorted(held_out_shards()):
        skipped: list[str] = []
        yield shard, _Unit(f"{shard:03d}",
                           _frame_unit(shard, cache_dir, skipped, panel, gate), skipped)


def external(fp_dir: Path = FP_DIR, *, cache_dir: Path = CACHE_DIR,
             codes: Sequence[str] = SETS) -> dict[str, dict]:
    """Fingerprint every frozen set's audio into ``fp_dir``, unit by unit, resuming.

    The frame sets share one pass over ``held_out_shards()``: a row lands in ``gate_eval``
    when it is a ``decode_evalset.{dev,test}`` row, else in ``tadabur_frame``; sealed-panel
    rows are skipped. Existing unit files are verified by nothing but their own completeness
    (atomic write), so a resumed run reuses them.
    """
    fp_dir, cache_dir = Path(fp_dir), Path(cache_dir)
    clips = committed_clips()
    wanted = _external_wanted(clips)
    census = {}
    for code in codes:
        if code in FRAME_SETS:
            continue
        expected = len(wanted[code])
        seen = 0
        for unit in external_units(code, cache_dir=cache_dir, wanted=wanted):
            path = fp_dir / code / f"{unit.name}.npz"
            if path.exists():
                members = _members(path)
                print(f"[external] {code}/{unit.name}: reused {members}", flush=True)
            else:
                members, _ = _write_unit(fp_dir / code, unit, code)
            seen += members
        if seen != expected:
            raise ValueError(f"{code}: {seen} fingerprints for {expected} committed clips")
        census[code] = _set_census(fp_dir / code)
    if set(codes) & set(FRAME_SETS):
        census.update(_frame_pass(fp_dir, cache_dir))
    return census


def _frame_pass(fp_dir: Path, cache_dir: Path) -> dict[str, dict]:
    gate = gate_membership()
    panel = panel_provenance()
    seen_gate = 0
    skipped_total = 0
    for shard, unit in _frame_units(cache_dir, panel, gate):
        paths = {code: fp_dir / code / f"{unit.name}.npz" for code in FRAME_SETS}
        if all(path.exists() for path in paths.values()):
            reused = {code: _members(path) for code, path in paths.items()}
            print(f"[external] {shard:03d}: reused gate_eval {reused['gate_eval']}, "
                  f"tadabur_frame {reused['tadabur_frame']}", flush=True)
            seen_gate += reused["gate_eval"]
            continue
        prints = {code: {} for code in FRAME_SETS}
        picker = {code: _ControlPicker(code) for code in FRAME_SETS}
        for clip, waveform in unit.clips:
            row_index = int(clip.split(":")[1])
            code = "gate_eval" if (shard, row_index) in gate else "tadabur_frame"
            prints[code][clip] = fingerprint(waveform)
            picker[code].add(clip, waveform)
        for code in FRAME_SETS:
            _save_arrays(paths[code], {clip: fp for clip, fp in prints[code].items() if len(fp)})
            _save_arrays(fp_dir / code / "control" / paths[code].name, picker[code].arrays())
        seen_gate += sum(1 for fp in prints["gate_eval"].values() if len(fp))
        skipped_total += len(unit.skipped)
        print(f"[external] {shard:03d}: gate_eval {len(prints['gate_eval'])}, "
              f"tadabur_frame {len(prints['tadabur_frame'])}, "
              f"panel rows skipped {len(unit.skipped)}", flush=True)
    if seen_gate != len(gate):
        raise ValueError(f"gate_eval store holds {seen_gate} clips for {len(gate)} committed rows")
    print(f"[external] frame pass: {skipped_total} sealed-panel rows skipped", flush=True)
    return {code: _set_census(fp_dir / code) for code in FRAME_SETS}


def _set_census(set_dir: Path) -> dict:
    """One set's store census: every unit file's SHA-256, member count and the set digest."""
    units = sorted(path for path in Path(set_dir).glob("*.npz"))
    digest = hashlib.sha256()
    members = 0
    for path in units:
        count = _members(path)
        digest.update(f"{sha256_file(path)}  {path.name}\n".encode())
        members += count
    return {"path": str(set_dir), "sha256": digest.hexdigest(), "members": members,
            "units": len(units)}


def load_external(fp_dir: Path = FP_DIR) -> dict[str, dict[str, np.ndarray]]:
    """Every set's ``clip -> fingerprint`` from the scratch store."""
    store = {}
    for code in SETS:
        directory = Path(fp_dir) / code
        if not directory.is_dir():
            raise ValueError(f"the external store has no {code} under {fp_dir}: run `external`")
        prints: dict[str, np.ndarray] = {}
        for path in sorted(directory.glob("*.npz")):
            prints.update(_load_arrays(path))
        store[code] = prints
    return store


def _control_waves(fp_dir: Path, code: str) -> list[tuple[str, np.ndarray]]:
    """The set's :data:`CONTROL_COPIES` lowest-ranked candidate waveforms, from its units."""
    candidates: list[tuple[int, Path, str]] = []
    for path in sorted((Path(fp_dir) / code / "control").glob("*.npz")):
        with np.load(path) as data:
            candidates.extend((_control_rank(code, clip), path, clip) for clip in data.files)
    chosen = heapq.nsmallest(CONTROL_COPIES, candidates, key=lambda item: item[0])
    waves = []
    by_path: dict[Path, dict[str, np.ndarray]] = {}
    for _, path, clip in chosen:
        if path not in by_path:
            by_path[path] = _load_arrays(path)
        waves.append((clip, by_path[path][clip]))
    return waves


def synthetic_recall_control(index: EvalIndex, fp_dir: Path) -> dict:
    """Control A: re-encode and re-cut copies of external clips through the real index.

    Required, not advisory: fewer than :data:`CONTROL_COPIES` copies per set, any copy whose
    BER exceeds :data:`LINE`, or fewer than 99% found raises — the run must not write a tree
    from an index that cannot find a copy of a clip it indexed.
    """
    from .qud_overlap import synthetic_copy

    copies = found = 0
    max_ber = 0.0
    for code in SETS:
        waves = _control_waves(fp_dir, code)
        if len(waves) < CONTROL_COPIES:
            raise ValueError(f"{code}: {len(waves)} control candidates, "
                             f"need {CONTROL_COPIES}")
        for clip, waveform in waves:
            cut_ms = 100 + _control_rank(code, clip) % 401
            probe = fingerprint(synthetic_copy(waveform, cut_ms))
            copies += 1
            hits = index.query(probe)
            match = next((hit for hit in hits
                          if hit.clip.split(_ROW_SEP, 1) == [code, clip]), None)
            if match is None:
                continue
            found += 1
            max_ber = max(max_ber, match.ber)
    share = found / copies if copies else 0.0
    print(f"[build] synthetic recall: {found}/{copies} found, max BER {max_ber}", flush=True)
    if share < 0.99 or max_ber > LINE:
        raise ValueError(f"synthetic-positive recall failed: {found}/{copies}, "
                         f"max BER {max_ber} (line {LINE})")
    return {"copies": copies, "found": found, "found_share": round(share, 6),
            "max_ber": max_ber}


# --- the tlog side -----------------------------------------------------------------------------


def _load_tlog_prints(path: Path) -> dict[str, np.ndarray]:
    """One tlog shard's ``audio_path -> fingerprint`` (the store's ``<path>|<sha>`` keys)."""
    with np.load(path) as data:
        return {name.rsplit("|", 1)[0]: data[name] for name in data.files}


def _eval_index(store_dir: Path, split_dir: Path) -> EvalIndex:
    """The frozen eval-role rows' fingerprints, keyed by their row key."""
    prints: dict[str, np.ndarray] = {}
    for split_path in sorted(Path(split_dir).glob("*/*.jsonl.gz")):
        rows = read_jsonl_gz(split_path)
        if not any(row["role"] != TRAIN for row in rows):
            continue
        store = _load_tlog_prints(Path(store_dir) / split_path.name)
        for row in rows:
            if row["role"] == TRAIN:
                continue
            prints[_row_clip(_key(row))] = store.get(row["audio_path"], np.zeros(0, np.uint32))
    return EvalIndex.build(prints)


def _external_hits(digest, prints, index, sha_map, stats) -> list[dict]:
    """The row's best hit per external set: committed checksums first, then the index."""
    best: dict[str, dict] = {}
    if digest and digest != EMPTY_SHA256:
        for code, clip in sha_map.get(digest, ()):
            best[code] = {"set": code, "clip": clip, "ber": 0.0}
    if prints is not None and len(prints):
        for hit in index.query(prints):
            code, clip = hit.clip.split(_ROW_SEP, 1)
            stats["verified"] += 1
            if hit.stop_word_share >= MOSTLY_STOP_WORD:
                stats["mostly_stop_word"] += 1
            stats["histogram"][_bin(hit.ber)] += 1
            if hit.ber > SYNTHETIC_POSITIVE_MAX:
                stats["ambiguous"].append({"set": code, "clip": clip, "ber": hit.ber})
            if code not in best or hit.ber < best[code]["ber"]:
                best[code] = {"set": code, "clip": clip, "ber": hit.ber}
    return list(best.values())


def _bin(ber: float) -> int:
    return min(int(ber / BER_BIN), len(_BINS) - 1)


def _scan(index: EvalIndex, eval_index: EvalIndex, sha_map, store_dir: Path,
          index_dir: Path, pass1_dir: Path, floors_path: Path):
    """One pass: external hits, duplicate links, exact-checksum groups and row metadata."""
    hits: dict[tuple, list[dict]] = {}
    edges: dict[tuple, float] = {}
    groups: dict[str, list[tuple]] = defaultdict(list)
    metadata: dict[tuple, tuple[str, str]] = {}
    stats = {"verified": 0, "mostly_stop_word": 0, "histogram": Counter(), "ambiguous": []}
    rows = iter_frozen_rows(index_dir=index_dir, pass1_dir=pass1_dir, floors_path=floors_path)
    for (split, stem), shard_rows in groupby(rows, key=lambda row: (row["split"],
                                                                    row["shard_file"])):
        shard = list(shard_rows)
        prints = _load_tlog_prints(Path(store_dir) / f"{stem}.npz")
        decodable = sum(1 for row in shard if row["codec_ok"])
        if len(prints) != decodable:
            raise ValueError(f"{store_dir}/{stem}.npz holds {len(prints)} fingerprints; "
                             f"pass 1 has {decodable} codec_ok rows")
        for row in shard:
            key = _key(row)
            metadata[key] = (row["role"], row["audio_path"])
            digest = row["audio_sha256"]
            if digest and digest != EMPTY_SHA256:
                groups[digest].append(key)
            found = _external_hits(digest, prints.get(row["audio_path"]), index, sha_map, stats)
            if found:
                hits[key] = found
            own = prints.get(row["audio_path"])
            if own is None or not len(own):
                continue
            for hit in eval_index.query(own, exclude=_row_clip(key)):
                other = _clip_row(hit.clip)
                stats["verified"] += 1
                if hit.stop_word_share >= MOSTLY_STOP_WORD:
                    stats["mostly_stop_word"] += 1
                stats["histogram"][_bin(hit.ber)] += 1
                pair = (key, other) if key < other else (other, key)
                if hit.ber < edges.get(pair, 1.0):
                    edges[pair] = hit.ber
        print(f"[build] scan {split}/{stem}: {len(shard)} rows", flush=True)
    return hits, edges, groups, metadata, stats


def _near_ber(ber: float) -> float:
    """A verified near duplicate's recorded BER: 0 is reserved for an exact checksum."""
    return ber if ber > 0 else 1e-4


def _duplicate_ber(key, survivor, digest_of, near_by_survivor) -> float:
    """0 when the loser's bytes are the survivor's, else the component's smallest near BER."""
    digest = digest_of.get(key)
    if digest and digest == digest_of.get(survivor):
        return 0.0
    try:
        return near_by_survivor[survivor]
    except KeyError:
        raise ValueError(f"{key} is linked to {survivor} without a verified near pair") from None


def _role_pair(first: str, second: str) -> str:
    left, right = sorted((first, second), key=_role_rank)
    return f"{left}:{right}"


def _frozen_inputs(split_dir: Path, pass1_dir: Path) -> dict:
    return {"split_tree_sha256": _tree_sha256(split_dir),
            "pass1_meta_sha256": sha256_file(Path(pass1_dir) / "meta.json")}


def _tree_sha256(root: Path) -> str:
    """One digest over a tree: ``sha256(file)  <posix relpath>`` lines, path-sorted."""
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest.update(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n".encode())
    return digest.hexdigest()


# --- the build ---------------------------------------------------------------------------------


def build(overlap_dir: Path = OVERLAP_DIR, *, fp_dir: Path = FP_DIR, store_dir: Path = STORE_DIR,
          index_dir: Path = INDEX_DIR, pass1_dir: Path = PASS1_DIR,
          floors_path: Path = FLOORS_PATH, split_dir: Path | None = None,
          code_sha: str | None = None, pull_request: int | None = None) -> dict:
    """Write the committed overlap tree and ``meta.json``; return the meta.

    ``fp_dir`` is the external store :func:`external` wrote (and keeps the control candidates).
    ``store_dir`` is S3b's tlog fingerprint store, read only. Rows come from
    :func:`tadabur.tlog_split.iter_frozen_rows`, so the key is the frozen
    ``(split, shard_file, row)`` on both passes.
    """
    split_dir = Path(split_dir or Path(__file__).parent / "tlog" / "split")
    overlap_dir = Path(overlap_dir)
    current = provenance(code_sha or head_sha(), pull_request)
    clips = committed_clips()
    sources = set_sources()

    external = load_external(fp_dir)
    census = {code: _set_census(Path(fp_dir) / code) for code in SETS}
    for code, store in clips.items():
        expected = len(store)
        if census[code]["members"] != expected:
            raise ValueError(f"{code}: the store holds {census[code]['members']} fingerprints "
                             f"for {expected} committed clips")
    gate = gate_membership()
    if census["gate_eval"]["members"] != len(gate):
        raise ValueError(f"gate_eval: {census['gate_eval']['members']} fingerprints "
                         f"for {len(gate)} committed rows")

    index = EvalIndex.build({f"{code}{_ROW_SEP}{clip}": external[code][clip]
                             for code in SETS for clip in sorted(external[code])})
    control = synthetic_recall_control(index, Path(fp_dir))
    eval_index = _eval_index(Path(store_dir), split_dir)
    sha_map = _sha_map(clips)
    hits, edges, groups, metadata, stats = _scan(
        index, eval_index, sha_map, Path(store_dir), Path(index_dir), Path(pass1_dir),
        Path(floors_path))

    exact = _exact_links(groups)
    links = dict(edges)
    for pair in exact:
        links[pair] = 0.0
    rows_map = {key: {"role": metadata[key][0], "audio_path": metadata[key][1]}
                for key in {member for pair in links for member in pair}}
    resolved = resolve_duplicates(rows_map, links)
    digest_of = {key: digest for digest, keys in groups.items() for key in keys}
    near_by_survivor: dict[tuple, float] = {}
    for pair, ber in edges.items():
        value = _near_ber(ber)
        for member in pair:
            survivor = resolved.get(member, member)
            if value < near_by_survivor.get(survivor, 1.0):
                near_by_survivor[survivor] = value

    hits_by_set: Counter = Counter()
    hits_by_set_and_role: dict[str, Counter] = defaultdict(Counter)
    duplicates = {"exact": Counter(), "near": Counter()}
    net = {base: {role: Counter() for role in ROLES}
           for base in ("usable_before_s4", "gating_eligible")}
    written = 0
    rows = iter_frozen_rows(index_dir=index_dir, pass1_dir=pass1_dir, floors_path=floors_path)
    for (split, stem), shard_rows in groupby(rows, key=lambda row: (row["split"],
                                                                    row["shard_file"])):
        records = []
        for row in shard_rows:
            key = _key(row)
            survivor = resolved.get(key)
            duplicate_ber = None
            if survivor is not None:
                duplicate_ber = _duplicate_ber(key, survivor, digest_of, near_by_survivor)
                duplicates["exact" if duplicate_ber == 0 else "near"][
                    _role_pair(row["role"], metadata[survivor][0])] += 1
            record = overlap_record(
                row, hits.get(key, []),
                duplicate_of=({"split": survivor[0], "shard_file": survivor[1],
                               "row": survivor[2]} if survivor is not None else None),
                duplicate_ber=duplicate_ber)
            for hit in record["eval_overlap"]:
                hits_by_set[hit["set"]] += 1
                hits_by_set_and_role[hit["set"]][row["role"]] += 1
            usable = not row["exclusion_reasons"] and not row["eval_overlap"]
            before = row["role"] != TRAIN and usable
            gating = before and split == "clean" and row["filename_label"] is not None
            for base, applies in (("usable_before_s4", before),
                                  ("gating_eligible", gating)):
                if applies:
                    net[base][row["role"]]["before"] += 1
                    net[base][row["role"]]["lost"] += record["excluded"] is not None
            records.append(record)
        write_shard(records, overlap_dir / split / f"{stem}.jsonl.gz")
        written += len(records)
        print(f"[build] wrote {split}/{stem}: {len(records)} rows", flush=True)

    net_sizes = {base: {role: {"before": counts["before"], "lost": counts["lost"],
                               "remaining": counts["before"] - counts["lost"]}
                        for role, counts in roles.items() if role != TRAIN}
                 for base, roles in net.items()}
    meta = {
        "schema_version": SCHEMA_VERSION,
        **current,
        "row_key": list(ROW_KEY),
        "threshold": {"line": LINE, "reused": True,
                      "source": {"path": repo_relative(RULES_PATH),
                                 "sha256": sha256_file(RULES_PATH)}},
        "vote_threshold": MIN_VOTES,
        "stop_words": sorted(STOP_WORDS),
        "sets": {code: {
            "members": census[code]["members"],
            "source": sources[code].source,
            "revision": sources[code].revision,
            "checksums": ({"path": repo_relative(sources[code].checksums),
                           "sha256": sha256_file(sources[code].checksums)}
                          if sources[code].checksums is not None else None),
            "fingerprints": {"path": census[code]["path"],
                             "sha256": census[code]["sha256"],
                             "members": census[code]["members"]},
            "units": census[code]["units"]}
            for code in SETS},
        "hits": {"by_set": dict(sorted(hits_by_set.items())),
                 "by_set_and_role": {code: dict(sorted(roles.items()))
                                     for code, roles in sorted(hits_by_set_and_role.items())}},
        "duplicates": {kind: dict(sorted(counts.items()))
                       for kind, counts in duplicates.items()},
        "net_sizes": net_sizes,
        "controls": {
            "synthetic_recall": control,
            "stop_word_matches": {
                "verified": stats["verified"],
                "mostly_stop_word": stats["mostly_stop_word"],
                "share": round(stats["mostly_stop_word"] / stats["verified"], 6)
                if stats["verified"] else 0.0,
                "mostly_stop_word_share": MOSTLY_STOP_WORD}},
        "ber_histogram": {"bins": list(_BINS),
                          "counts": [stats["histogram"][index]
                                     for index in range(len(_BINS))]},
        "ambiguous_zone": sorted(stats["ambiguous"],
                                 key=lambda hit: (hit["set"], hit["clip"], hit["ber"])),
        "blind_spots": {
            "sealed_panel": SEALED_PANEL,
            "train_near_duplicates_searched": False,
            "external_without_checksums": sorted(FRAME_SETS),
            "why": ("the sealed panel cannot be read (the seal pins two authorisations); "
                    "train-train near duplicates were scoped out; frame and gate_eval have "
                    "no source-byte checksums, so they are matched by fingerprint only")},
        "frozen": _frozen_inputs(split_dir, Path(pass1_dir)),
        "counts": {"rows": written},
    }
    write_text_atomically(overlap_dir / META_NAME,
                          json.dumps(meta, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    return meta


# --- the command line --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fingerprint_sets = commands.add_parser(
        "external", help="fingerprint the frozen eval sets into the scratch store")
    fingerprint_sets.add_argument("--fp-dir", type=Path, default=FP_DIR)
    fingerprint_sets.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    fingerprint_sets.add_argument("--set", action="append", dest="sets", choices=SETS,
                                  help="only these sets (default: all)")
    builder = commands.add_parser(
        "build", help="write the committed overlap tree from both fingerprint stores")
    builder.add_argument("--pr", type=int, required=True,
                         help="the PR number whose refs/pull/<N>/head keeps code_sha reachable")
    builder.add_argument("--out", type=Path, default=OVERLAP_DIR)
    builder.add_argument("--fp-dir", type=Path, default=FP_DIR)
    builder.add_argument("--store-dir", type=Path, default=STORE_DIR)
    builder.add_argument("--code-sha", default=None)
    args = parser.parse_args()
    if args.command == "external":
        census = external(args.fp_dir, cache_dir=args.cache_dir, codes=args.sets or SETS)
        print(json.dumps(census, indent=1, sort_keys=True), flush=True)
        return
    meta = build(args.out, fp_dir=args.fp_dir, store_dir=args.store_dir,
                 code_sha=args.code_sha, pull_request=args.pr)
    print(json.dumps({"counts": meta["counts"], "hits": meta["hits"],
                      "duplicates": meta["duplicates"], "net_sizes": meta["net_sizes"]},
                     indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
