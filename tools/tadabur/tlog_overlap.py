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

**Revision.** The frame sets share one audio pass over a re-uploadable Hub dataset, so the pass
resolves that dataset's commit once (:func:`_resolve_frame_revision`) and stamps it into
``<fp-dir>/<set>/revision.txt`` before any unit is written: a resume under another revision, or
a store whose units carry no stamp, is refused. The stamped commit is the two frame sets'
``revision`` in ``overlap/meta.json``, so every committed ``eval_overlap:gate_eval`` /
``eval_overlap:tadabur_frame`` fact names the audio it was measured against.

**Undecodable rows.** One row that will not decode is a fact, not a dead multi-hour pass: the
row is skipped and the committed clips it leaves without a fingerprint are listed in
``overlap/meta.json``'s ``unfingerprinted`` blind spot. A ``gate_eval`` row is the exception —
its 2,000-clip population is an invariant, so it fails loudly. The frame pass writes one
disposition file per held-out shard (``<fp-dir>/frames/<NNN>.json``): the rows it read and every
row's fate — fingerprinted into ``gate_eval`` or ``tadabur_frame``, sealed panel, undecodable,
or too short to fingerprint — plus ``frames/index.json``, the shard list the store covers. The
build refuses a store whose declared shards and dispositions do not match, or whose two frame
stores hold anything but the fingerprints those dispositions claim, and it lists the frame rows
it could not fingerprint in the same blind spot.

**Threshold.** :data:`LINE` is the committed line from ``qud_eval/overlap_rules.json`` (0.238,
halfway between the controls' highest synthetic positive and lowest hard negative), reused
with its file SHA-256 recorded in ``overlap/meta.json``. It is not re-derived: these sets offer
no clean hard negatives. The positives are re-checked instead, by a required run-level control
:func:`synthetic_recall_control` — at least :data:`CONTROL_COPIES` re-encoded, re-cut copies of
external clips through the real index; the run refuses to write anything unless ≥ 99% of those
copies are verified candidates and every candidate's BER is within the line. The line filter
inside :meth:`EvalIndex.query` hides the band above it, so the control, the BER histogram and
the stop-word report read :meth:`EvalIndex.verify` — every candidate the votes propose.

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
    the set census and provenance (source, pinned revision, checksums, fingerprint digest),
    the threshold and its source SHA-256, the vote threshold and stop-words, both run controls,
    the BER histogram of every verified candidate — inside the line and rejected above it —
    with the ambiguous zone, hits per set and role, duplicates by role pair, the net-size
    tables, the blind spots, the tlog store's census and the frozen inputs' digests.

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
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np

from training.tashkeel_eval import write_text_atomically

from .audio import DecodeError, decode_to_mono_16k
from .dataset_source import DATASET_ID
from .nonpro_sources import (QDAT, QDAT_BENCH, RETASY, TARTEEL_V1, LeftOut, iter_rows,
                             select)
from .panel_seal import is_panel_name, panel_provenance
from .qud_fingerprint import best_match, fingerprint
from .qud_source import QUD_REVISION, QUD_DATASET, audio_filename as qud_audio_filename
from .tlog_index import (INDEX_DIR, ROLES, TRAIN, head_sha, provenance, recording_id,
                         repo_relative, sha256_file, write_shard)
from .tlog_pass1 import FLOORS_PATH, FP_RECORD_PATH, PASS1_DIR, read_jsonl_gz
from .tlog_split import EMPTY_SHA256, ROW_KEY, iter_frozen_rows

SCHEMA_VERSION = 1
#: The committed tree this module writes (``tools/tadabur/tlog/overlap``).
OVERLAP_DIR = Path(__file__).parent / "tlog" / "overlap"
META_NAME = "meta.json"
#: The set codes, in the order that breaks a tie for the primary ``excluded`` reason.
SETS = ("tarteel_v1", "qud", "gate_eval", "tadabur_frame", "qdat", "qdat_bench", "retasy")
#: The frozen exclusion union, in order: the floors, S4's set reasons in :data:`SETS` order
#: (which starts with ``tarteel_v1``, so S1's id reason appears once), then ``duplicate``.
EXCLUSION_ORDER = ("codec", "silence", *(f"eval_overlap:{code}" for code in SETS),
                   "duplicate")
assert len(EXCLUSION_ORDER) == len(set(EXCLUSION_ORDER))
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
#: The frame rows' source: Tadabur's full-config shards, at whatever :func:`_frame_pass` pins.
TADABUR_SOURCE = DATASET_ID
#: The frame stores' revision stamp: the Tadabur commit their fingerprints were taken at.
REVISION_NAME = "revision.txt"
#: The frame pass's per-shard disposition records (``frames/<NNN>.json``) and the shard list
#: they must cover (``frames/index.json``), which the build reconciles.
FRAMES_DIR = "frames"
FRAME_INDEX = "index.json"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

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
        """Every :meth:`verify` candidate within ``line``, best BER first.

        ``exclude`` is one clip id never verified — the eval-row index's own query would
        otherwise spend a full ``best_match`` on the row against itself.
        """
        return [hit for hit in self.verify(prints, min_votes=min_votes, exclude=exclude)
                if hit.ber <= line]

    def verify(self, prints: np.ndarray, *, min_votes: int = MIN_VOTES,
               exclude: str | None = None) -> list[Hit]:
        """Every clip the votes propose, verified with :func:`best_match`, best BER first.

        Unfiltered by any line: the run's recall control, its BER histogram and its stop-word
        report need the candidates the line rejects, which :meth:`query` would hide.
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
            if match is None:
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


def _row_exclusions(row: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[str, ...]:
    """One row's frozen exclusion union, in :data:`EXCLUSION_ORDER` and deduplicated."""
    found = set(row["exclusion_reasons"])
    if row["s1_eval_overlap"]:
        found.add("eval_overlap:tarteel_v1")
    if record["excluded"]:
        found.add(record["excluded"])
    unknown = found - set(EXCLUSION_ORDER)
    if unknown:
        raise ValueError(f"{_key(row)} has exclusions outside the union: {sorted(unknown)}")
    exclusions = tuple(reason for reason in EXCLUSION_ORDER if reason in found)
    if len(exclusions) != len(found):
        raise AssertionError(f"{_key(row)} emitted a duplicated exclusion: {exclusions}")
    return exclusions


def iter_pool_rows(*, overlap_dir: Path = OVERLAP_DIR, index_dir: Path = INDEX_DIR,
                   pass1_dir: Path = PASS1_DIR, floors_path: Path = FLOORS_PATH) -> Iterator[dict]:
    """Every row's S1 + S4 view: one ``exclusions`` tuple and one ``usable`` flag.

    The frozen row (with S1's flag renamed ``s1_eval_overlap``) is joined to its committed
    overlap record by :data:`ROW_KEY`, shard by shard, and the two trees must cover exactly the
    same rows. The S4 record's fields are then attached under their own names — its
    ``eval_overlap`` list no longer collides with S1's boolean — and the ordered union is
    ``exclusions``: ``codec``, ``silence``, S1's ``eval_overlap:tarteel_v1``, S4's primary
    ``excluded``, then ``usable``. This is the one reader S6/S7 merge, so no caller has to
    repeat the union or the alignment.
    """
    rows = iter_frozen_rows(index_dir=index_dir, pass1_dir=pass1_dir, floors_path=floors_path)
    for (split, stem), shard_rows in groupby(rows, key=lambda row: (row["split"],
                                                                    row["shard_file"])):
        shard = list(shard_rows)
        path = Path(overlap_dir) / split / f"{stem}.jsonl.gz"
        if not path.exists():
            raise ValueError(f"the overlap tree has no {split}/{stem}.jsonl.gz")
        records = read_jsonl_gz(path)
        if len(records) != len(shard):
            raise ValueError(f"{split}/{stem}: {len(records)} overlap records for {len(shard)} "
                             f"frozen rows")
        for row, record in zip(shard, records, strict=True):
            if _key(row) != _key(record):
                raise ValueError(f"{split}/{stem}: {_key(record)} is not frozen row {_key(row)}")
            exclusions = _row_exclusions(row, record)
            yield {**row,
                   "eval_overlap": record["eval_overlap"],
                   "duplicate_of": record["duplicate_of"],
                   "duplicate_ber": record["duplicate_ber"],
                   "excluded": record["excluded"],
                   "exclusions": exclusions,
                   "usable": not exclusions}


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
    """One set's upstream identity: its Hub dataset, the pinned commit and, where one exists,
    its committed checksum list. The frame sets' revision is ``None`` here because they share
    a pass that resolves it: :func:`_frame_pass` stamps the commit in their stores and the
    build reads it back from there.
    """

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
    unresolved = {code for code, source in sources.items() if source.revision is None}
    if unresolved != set(FRAME_SETS):
        raise ValueError(f"sets without a pinned revision: {sorted(unresolved)}; "
                         f"only the frame sets resolve theirs at run time")
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
    """One source shard's fingerprints-to-be, and the rows it could not fingerprint:
    ``skipped`` (sealed-panel clips) and ``broken`` (undecodable, recorded per set).

    ``rows`` is filled by the clip generator once it is exhausted — the shard's source row
    count, so a completed unit's dispositions are reconciled against the shard rather than
    against themselves. It stays empty for a unit whose disposition is not recorded.
    """

    name: str
    clips: Iterator[tuple[str, np.ndarray]]
    skipped: list[str]
    broken: list[str]
    rows: list[int]


def _nonpro_unit(dataset, shard: str, cache_dir: Path, wanted: set[str],
                 broken: list[str]) -> Iterator[tuple[str, np.ndarray]]:
    for record in iter_rows(dataset, shard, cache_dir, strict_delete=True):
        if dataset is QDAT_BENCH:
            clip, raw = record["id"], record["audio"]["bytes"]
        else:
            clip = select(dataset, record)
            if isinstance(clip, LeftOut):
                continue
            clip, raw = clip.audio_filename, clip.audio
        if clip not in wanted:
            continue
        try:
            yield clip, decode_to_mono_16k(raw)
        except DecodeError:
            broken.append(clip)


def _nonpro_units(code: str, cache_dir: Path, wanted: set[str]) -> Iterator[_Unit]:
    dataset = _NONPRO_DATASETS[set_sources()[code].source]
    for shard in dataset.shards:
        broken: list[str] = []
        yield _Unit(Path(shard).stem, _nonpro_unit(dataset, shard, cache_dir, wanted, broken),
                    [], broken, [])


def _qud_unit(slug: str, shard: str, cache_dir: Path, wanted: set[str],
              broken: list[str]) -> Iterator[tuple[str, np.ndarray]]:
    from .qud_source import decode_ayah, iter_mushaf_rows

    for row in iter_mushaf_rows(slug, [shard], cache_dir):
        name = row.audio_filename
        if name not in wanted:
            continue
        try:
            yield name, decode_ayah(row)
        except DecodeError:
            broken.append(name)


def _qud_units(cache_dir: Path, wanted: set[str]) -> Iterator[_Unit]:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    shards_by_slug = {mushaf["slug"]: mushaf["shards"] for mushaf in selection["mushafs"]}
    slugs = sorted({record["slug"] for record in _read_jsonl(QUD_CHECKSUMS_PATH)})
    for slug in slugs:
        if slug not in shards_by_slug:
            raise ValueError(f"{slug} is not in {SELECTION_PATH}")
        for shard in shards_by_slug[slug]:
            broken: list[str] = []
            yield _Unit(f"{slug}-{Path(shard).stem}",
                        _qud_unit(slug, shard, cache_dir, wanted, broken), [], broken, [])


def _frame_unit(shard: int, cache_dir: Path, skipped: list[str], broken: list[str],
                rows: list[int], panel, gate: Mapping[tuple[int, int], str],
                revision: str) -> Iterator[tuple[str, np.ndarray]]:
    from .shard_reader import iter_shard_rows

    seen = 0
    for position, row in enumerate(iter_shard_rows([shard], cache_dir=cache_dir,
                                                   delete_after=True, revision=revision)):
        seen = position + 1
        path = (row.get("audio") or {}).get("path") or ""
        expected = gate.get((shard, position))
        if expected is not None and Path(path).name != expected:
            raise ValueError(f"shard {shard} row {position} is {Path(path).name}, "
                             f"not gate_eval's {expected}: the frame rows drifted")
        if is_panel_name(path, panel):
            skipped.append(path)
            continue
        try:
            waveform = decode_to_mono_16k(row["audio"]["bytes"])
        except DecodeError:
            if expected is not None:
                raise ValueError(f"gate_eval row {shard}:{position} ({expected}) "
                                 f"does not decode") from None
            broken.append(f"{shard}:{position}")
            continue
        yield f"{shard}:{position}", waveform
    rows.append(seen)


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


def _indexable(prints: np.ndarray) -> bool:
    """Whether the index will hold any of this fingerprint's frames.

    An empty fingerprint (shorter than two frames) and an all-stop-word one (digital silence)
    both index nothing, so neither may become a control candidate: the control must measure
    the index the build uses.
    """
    return bool((~np.isin(np.asarray(prints), _STOP)).any())


def _write_unit(set_dir: Path, unit: _Unit, code: str) -> None:
    """Fingerprint one unit and save it with its control candidates.

    Every decoded fingerprint is saved, but only an indexable one (:func:`_indexable`) is
    offered as a control candidate: a copy of a clip that the index cannot hold could never be
    found, and one such miss per set would fail the run's required control.
    """
    prints: dict[str, np.ndarray] = {}
    picker = _ControlPicker(code)
    for clip, waveform in unit.clips:
        prints[clip] = fingerprint(waveform)
        if _indexable(prints[clip]):
            picker.add(clip, waveform)
    saved = {clip: fp for clip, fp in prints.items() if len(fp)}
    broken = len(unit.broken)
    # The control archive first: the fingerprint archive is what resume keys on, so a
    # process killed between the two writes never leaves a unit that looks complete.
    _save_arrays(set_dir / "control" / f"{unit.name}.npz", picker.arrays())
    _save_arrays(set_dir / f"{unit.name}.npz", saved)
    print(f"[external] {code}/{unit.name}: {len(saved)} fingerprints"
          + (f", {broken} undecodable" if broken else ""), flush=True)


def _resolve_frame_revision(source: str = TADABUR_SOURCE) -> str:
    """The frame source's current commit, resolved once per pass and then pinned in the store."""
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(source).sha
    if not _COMMIT.match(revision):
        raise ValueError(f"{source} resolved to {revision!r}, not a commit")
    return revision


def _recorded_revision(set_dir: Path) -> str | None:
    path = Path(set_dir) / REVISION_NAME
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def _pin_frame_revision(fp_dir: Path, revision: str) -> None:
    """Stamp ``revision`` into both frame stores, refusing a unit of unknown or other vintage."""
    for code in FRAME_SETS:
        set_dir = Path(fp_dir) / code
        recorded = _recorded_revision(set_dir)
        if recorded is not None and recorded != revision:
            raise ValueError(f"{code}: the store was fingerprinted at {recorded}; "
                             f"the source now resolves {revision}")
        if recorded is None and any(set_dir.glob("*.npz")):
            raise ValueError(f"{code}: the store holds fingerprints but no {REVISION_NAME}, "
                             f"so the revision they were taken at cannot be verified")
    for code in FRAME_SETS:
        set_dir = Path(fp_dir) / code
        if _recorded_revision(set_dir) is None:
            write_text_atomically(set_dir / REVISION_NAME, revision + "\n")


def _frame_revision(fp_dir: Path) -> str:
    """The commit both frame stores were fingerprinted at; the build refuses a mismatch."""
    recorded = {code: _recorded_revision(Path(fp_dir) / code) for code in FRAME_SETS}
    if len(set(recorded.values())) != 1 or None in recorded.values():
        raise ValueError(f"the frame stores' {REVISION_NAME} is {recorded}: run `external` "
                         f"first, and never mix revisions")
    return str(recorded[FRAME_SETS[0]])


def _frame_shards() -> list[int]:
    """Every held-out shard the frame pass covers (the torch-backed module is imported here)."""
    from training.distill_stream import held_out_shards

    return sorted(held_out_shards())


def _frame_units(cache_dir: Path, panel, gate: Mapping[tuple[int, int], str],
                 revision: str) -> Iterator[tuple[int, _Unit]]:
    for shard in _frame_shards():
        skipped: list[str] = []
        broken: list[str] = []
        rows: list[int] = []
        yield shard, _Unit(f"{shard:03d}",
                           _frame_unit(shard, cache_dir, skipped, broken, rows, panel, gate,
                                       revision),
                           skipped, broken, rows)


def _frame_record_path(fp_dir: Path, shard: int) -> Path:
    return Path(fp_dir) / FRAMES_DIR / f"{shard:03d}.json"


def _write_frame_record(fp_dir: Path, record: Mapping[str, Any]) -> None:
    """Publish one shard's dispositions — the last write of its unit, so it is the marker."""
    write_text_atomically(_frame_record_path(fp_dir, record["shard"]),
                          json.dumps(record, sort_keys=True) + "\n")


def _read_frame_record(fp_dir: Path, shard: int) -> dict:
    return json.loads(_frame_record_path(fp_dir, shard).read_text(encoding="utf-8"))


def _declare_frames(fp_dir: Path, shards: Sequence[int]) -> None:
    """Declare (or verify) the held-out shard list this store covers; a change is refused."""
    directory = Path(fp_dir) / FRAMES_DIR
    path = directory / FRAME_INDEX
    declared = ([int(shard) for shard in json.loads(path.read_text(encoding="utf-8"))["shards"]]
                if path.exists() else None)
    if declared is None:
        if any(directory.glob("[0-9][0-9][0-9].json")):
            raise ValueError(f"{directory} holds dispositions but no {FRAME_INDEX}")
        write_text_atomically(path, json.dumps({"shards": list(shards)}) + "\n")
    elif declared != list(shards):
        raise ValueError(f"the frame store covers shards {declared}, the run now declares "
                         f"{list(shards)}: start a new store")


def _frame_dispositions(shard: int, unit: _Unit, saved: Mapping[str, Mapping[str, Any]],
                        short: Sequence[str]) -> dict:
    """One shard's dispositions: every row it read is fingerprinted, skipped or accounted for."""
    covered = (len(unit.skipped) + len(unit.broken) + len(short)
               + len(saved["gate_eval"]) + len(saved["tadabur_frame"]))
    rows = unit.rows[-1] if unit.rows else 0
    if covered != rows:
        raise ValueError(f"shard {shard}: dispositions cover {covered} of {rows} rows")
    return {"shard": shard, "rows": rows,
            "gate_eval": len(saved["gate_eval"]), "tadabur_frame": len(saved["tadabur_frame"]),
            "panel": len(unit.skipped), "broken": sorted(unit.broken), "short": sorted(short)}


def _frame_unit_complete(fp_dir: Path, shard: int, name: str) -> bool:
    """Whether one frame shard published everything, dispositions last."""
    return (all((Path(fp_dir) / code / f"{name}.npz").exists() for code in FRAME_SETS)
            and all((Path(fp_dir) / code / "control" / f"{name}.npz").exists()
                    for code in FRAME_SETS)
            and _frame_record_path(fp_dir, shard).exists())


def external(fp_dir: Path = FP_DIR, *, cache_dir: Path = CACHE_DIR,
             codes: Sequence[str] = SETS) -> dict[str, dict]:
    """Fingerprint every frozen set's audio into ``fp_dir``, unit by unit, resuming.

    The frame sets share one pass over ``held_out_shards()``: a row lands in ``gate_eval``
    when it is a ``decode_evalset.{dev,test}`` row, else in ``tadabur_frame``; sealed-panel
    rows are skipped and undecodable rows recorded. A unit is reused only when its fingerprint
    archive and its control archive are both on disk (the archive is published last), so an
    interrupted unit is recomputed; the build then checks the store against the committed
    populations.
    """
    fp_dir, cache_dir = Path(fp_dir), Path(cache_dir)
    wanted = _external_wanted(committed_clips())
    census = {}
    for code in codes:
        if code in FRAME_SETS:
            continue
        for unit in external_units(code, cache_dir=cache_dir, wanted=wanted):
            path = fp_dir / code / f"{unit.name}.npz"
            control = fp_dir / code / "control" / path.name
            if path.exists() and control.exists():
                print(f"[external] {code}/{unit.name}: reused {_members(path)}", flush=True)
            else:
                _write_unit(fp_dir / code, unit, code)
        census[code] = _set_census(fp_dir / code)
    if set(codes) & set(FRAME_SETS):
        census.update(_frame_pass(fp_dir, cache_dir))
    return census


def _frame_pass(fp_dir: Path, cache_dir: Path, *, revision: str | None = None) -> dict[str, dict]:
    """One pass over the frame shards, pinned to ``revision`` (resolved when None).

    Each shard publishes its controls, then its two fingerprint archives, then its
    disposition record (:func:`_frame_dispositions`); a unit is reused only when all five are
    on disk, so an interrupted shard is recomputed rather than trusted.
    """
    revision = revision or _resolve_frame_revision()
    _pin_frame_revision(fp_dir, revision)
    _declare_frames(fp_dir, _frame_shards())
    gate = gate_membership()
    panel = panel_provenance()
    seen_gate = 0
    skipped_total = broken_total = 0
    for shard, unit in _frame_units(cache_dir, panel, gate, revision):
        paths = {code: fp_dir / code / f"{unit.name}.npz" for code in FRAME_SETS}
        if _frame_unit_complete(fp_dir, shard, unit.name):
            record = _read_frame_record(fp_dir, shard)
            print(f"[external] {shard:03d}: reused gate_eval {record['gate_eval']}, "
                  f"tadabur_frame {record['tadabur_frame']}", flush=True)
            seen_gate += record["gate_eval"]
            skipped_total += record["panel"]
            broken_total += len(record["broken"]) + len(record["short"])
            continue
        prints = {code: {} for code in FRAME_SETS}
        picker = {code: _ControlPicker(code) for code in FRAME_SETS}
        short: list[str] = []
        for clip, waveform in unit.clips:
            row_index = int(clip.split(":")[1])
            code = "gate_eval" if (shard, row_index) in gate else "tadabur_frame"
            prints[code][clip] = fingerprint(waveform)
            if not len(prints[code][clip]):
                if code == "gate_eval":
                    raise ValueError(f"gate_eval row {clip} is shorter than two frames")
                short.append(clip)
            elif _indexable(prints[code][clip]):
                picker[code].add(clip, waveform)
        saved = {code: {clip: fp for clip, fp in prints[code].items() if len(fp)}
                 for code in FRAME_SETS}
        record = _frame_dispositions(shard, unit, saved, short)
        for code in FRAME_SETS:
            _save_arrays(fp_dir / code / "control" / paths[code].name, picker[code].arrays())
        for code in FRAME_SETS:
            _save_arrays(paths[code], saved[code])
        _write_frame_record(fp_dir, record)
        seen_gate += record["gate_eval"]
        skipped_total += record["panel"]
        broken_total += len(record["broken"]) + len(record["short"])
        print(f"[external] {shard:03d}: gate_eval {record['gate_eval']}, "
              f"tadabur_frame {record['tadabur_frame']}, "
              f"panel rows skipped {record['panel']}, rows {record['rows']}"
              + (f", undecodable {len(record['broken'])}" if record["broken"] else "")
              + (f", short {len(record['short'])}" if record["short"] else ""), flush=True)
    if seen_gate != len(gate):
        raise ValueError(f"gate_eval store holds {seen_gate} clips for {len(gate)} committed rows")
    print(f"[external] frame pass at {revision}: {skipped_total} sealed-panel rows skipped, "
          f"{broken_total} undecodable or too short", flush=True)
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

    Required, not advisory: fewer than :data:`CONTROL_COPIES` copies per set, fewer than 99%
    verified as candidates, or any candidate whose BER exceeds :data:`LINE` raises — the run
    must not write a tree from an index that cannot find a copy of a clip it indexed.

    The copy's BER is read from :meth:`EvalIndex.verify`, not :meth:`EvalIndex.query`: a copy
    the index would reject must be visible as a high BER rather than counted as found.
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
            match = next((hit for hit in index.verify(probe)
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
        stem = split_path.name[: -len(".jsonl.gz")]
        store = _load_tlog_prints(Path(store_dir) / f"{stem}.npz")
        for row in rows:
            if row["role"] == TRAIN:
                continue
            prints[_row_clip(_key(row))] = store.get(row["audio_path"], np.zeros(0, np.uint32))
    return EvalIndex.build(prints)


def _count_hit(stats: dict, hit: Hit, *, kind: str) -> bool:
    """One verified candidate's facts; whether it is inside :data:`LINE`.

    ``kind`` separates the two questions: ``external`` candidates answer how much of an
    external-set exclusion rests on a silence-dominated overlap, ``near_duplicate`` the same
    for a duplicate link. Both accepted buckets are reported beside the all-candidate figure.
    """
    stats["verified"] += 1
    mostly = hit.stop_word_share >= MOSTLY_STOP_WORD
    if mostly:
        stats["mostly_stop_word"] += 1
    accepted = hit.ber <= LINE
    stats["histogram" if accepted else "rejected"][_bin(hit.ber)] += 1
    if accepted:
        bucket = stats["accepted"][kind]
        bucket["hits"] += 1
        bucket["mostly_stop_word"] += mostly
    return accepted


def _count_s1_recall(stats: dict, row: Mapping[str, Any], prints, found: list[dict],
                     s1_clips: Mapping[str, list[str]],
                     tarteel_prints: Mapping[str, np.ndarray]) -> None:
    """Compare one S1 id-flagged row with the Tarteel v1 clips that share its recording id.

    S1's flag is an id match, not an audio match; this is the only real-positive recall the
    run can measure without audio. A row S4 already found counts as found. A miss is verified
    directly with :func:`best_match` against every clip sharing its id (ids are not unique in
    Tarteel v1), and its BER and stop-word share are recorded instead of assumed.
    """
    recall = stats["s1_recall"]
    recall["rows"] += 1
    if any(hit["set"] == "tarteel_v1" for hit in found):
        recall["found"] += 1
        return
    own = prints.get(row["audio_path"]) if prints is not None else None
    if own is None or not len(own):
        recall["unfingerprinted"] += 1
        return
    matches = []
    for clip in s1_clips.get(row["recording_id"] or "", ()):
        reference = tarteel_prints[clip]
        match = best_match(own, reference)
        if match is not None:
            matches.append((match.ber, match, reference))
    if not matches:
        recall["missing_clip"] += 1
        return
    _, match, reference = min(matches, key=lambda item: item[0])
    recall["misses"] += 1
    recall["miss_ber"][_bin(match.ber)] += 1
    recall["miss_stop_word"] += _stop_share(own, reference, match.offset) >= MOSTLY_STOP_WORD


def _external_hits(digest, prints, index, sha_map, stats) -> list[dict]:
    """The row's best hit per external set: committed checksums first, then the index."""
    best: dict[str, dict] = {}
    if digest and digest != EMPTY_SHA256:
        for code, clip in sha_map.get(digest, ()):
            best[code] = {"set": code, "clip": clip, "ber": 0.0}
    if prints is not None and len(prints):
        for hit in index.verify(prints):
            if not _count_hit(stats, hit, kind="external"):
                continue
            code, clip = hit.clip.split(_ROW_SEP, 1)
            if hit.ber > SYNTHETIC_POSITIVE_MAX:
                stats["ambiguous"].append({"set": code, "clip": clip, "ber": hit.ber})
            if code not in best or hit.ber < best[code]["ber"]:
                best[code] = {"set": code, "clip": clip, "ber": hit.ber}
    return list(best.values())


def _bin(ber: float) -> int:
    return min(int(ber / BER_BIN), len(_BINS) - 1)


def _scan(index: EvalIndex, eval_index: EvalIndex, sha_map, store_dir: Path,
          index_dir: Path, pass1_dir: Path, floors_path: Path,
          tarteel_prints: Mapping[str, np.ndarray]):
    """One pass: external hits, duplicate links, exact-checksum groups and row metadata."""
    hits: dict[tuple, list[dict]] = {}
    edges: dict[tuple, float] = {}
    groups: dict[str, list[tuple]] = defaultdict(list)
    metadata: dict[tuple, tuple[str, str]] = {}
    s1_clips: dict[str, list[str]] = defaultdict(list)
    for clip in sorted(tarteel_prints):
        identifier = recording_id(clip)
        if identifier is not None:
            s1_clips[identifier].append(clip)
    stats = {"verified": 0, "mostly_stop_word": 0, "histogram": Counter(),
             "rejected": Counter(), "ambiguous": [],
             "accepted": {"external": Counter(), "near_duplicate": Counter()},
             "s1_recall": {"rows": 0, "found": 0, "unfingerprinted": 0, "missing_clip": 0,
                           "misses": 0, "miss_ber": Counter(), "miss_stop_word": 0}}
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
            if row["s1_eval_overlap"]:
                _count_s1_recall(stats, row, prints, found, s1_clips, tarteel_prints)
            if found:
                hits[key] = found
            own = prints.get(row["audio_path"])
            if own is None or not len(own):
                continue
            for hit in eval_index.verify(own, exclude=_row_clip(key)):
                if not _count_hit(stats, hit, kind="near_duplicate"):
                    continue
                other = _clip_row(hit.clip)
                pair = (key, other) if key < other else (other, key)
                if hit.ber < edges.get(pair, 1.0):
                    edges[pair] = hit.ber
        print(f"[build] scan {split}/{stem}: {len(shard)} rows", flush=True)
    return hits, edges, groups, metadata, stats


def _stop_word_record(counts: Counter) -> dict:
    hits = counts["hits"]
    mostly = counts["mostly_stop_word"]
    return {"hits": hits, "mostly_stop_word": mostly,
            "share": round(mostly / hits, 6) if hits else 0.0}


def _s1_recall_record(stats: dict) -> dict:
    recall = stats["s1_recall"]
    rows = recall["rows"]
    return {"rows": rows, "found": recall["found"],
            "found_share": round(recall["found"] / rows, 6) if rows else 0.0,
            "unfingerprinted": recall["unfingerprinted"],
            "missing_clip": recall["missing_clip"],
            "misses": recall["misses"],
            "miss_ber": {"bins": list(_BINS),
                         "counts": [recall["miss_ber"][index] for index in range(len(_BINS))]},
            "miss_stop_word": {
                "mostly_stop_word": recall["miss_stop_word"],
                "share": (round(recall["miss_stop_word"] / recall["misses"], 6)
                          if recall["misses"] else 0.0)}}


def _survivor_reason(row: Mapping[str, Any]) -> str | None:
    """Why an unusable duplicate survivor is unusable, grouped for the committed report."""
    if row["exclusion_reasons"]:
        return "floors"
    if row["s1_eval_overlap"] or row["eval_overlap"]:
        return "eval_overlap"
    return None


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


def _frame_inventory(fp_dir: Path) -> tuple[list[str], dict[str, int]]:
    """Reconcile the frame pass's inventory; return its misfingerprinted ids and its counts.

    Every declared held-out shard must have a disposition record — a shard without one never
    completed — and every record's buckets must cover the rows it read. A frame row the pass
    could not fingerprint (undecodable, or shorter than two frames) is returned as a blind
    spot with its identity, never dropped: the build must be able to say which clip it could
    not have matched.
    """
    directory = Path(fp_dir) / FRAMES_DIR
    index_path = directory / FRAME_INDEX
    if not index_path.exists():
        raise ValueError(f"the frame store has no {FRAMES_DIR}/{FRAME_INDEX}: run `external`")
    declared = [int(shard) for shard in
                json.loads(index_path.read_text(encoding="utf-8"))["shards"]]
    found = sorted(int(path.stem) for path in directory.glob("[0-9][0-9][0-9].json"))
    if found != declared:
        raise ValueError(f"the frame store's dispositions are for shards {found}, not the "
                         f"declared {declared}: the pass did not complete")
    missing: list[str] = []
    counts = {code: 0 for code in FRAME_SETS}
    for shard in declared:
        record = _read_frame_record(fp_dir, shard)
        covered = (record["gate_eval"] + record["tadabur_frame"] + record["panel"]
                   + len(record["broken"]) + len(record["short"]))
        if covered != record["rows"]:
            raise ValueError(f"shard {shard}: dispositions cover {covered} of "
                             f"{record['rows']} rows")
        missing.extend(record["broken"])
        missing.extend(record["short"])
        for code in FRAME_SETS:
            counts[code] += record[code]
    return sorted(missing), counts


def _verify_tlog_store(store_dir: Path) -> str:
    """Refuse a tlog fingerprint store that is not the committed census's; return its digest.

    S3b's store lives on scratch and costs a full re-stream to rebuild, so the committed
    census (``tlog/fingerprints.json``) is the only thing that proves the build queries the
    store S3b wrote. A missing or different npz would silently change every duplicate, so it
    raises; the census's own SHA-256 is recorded in ``meta.json``.
    """
    census = json.loads(FP_RECORD_PATH.read_text(encoding="utf-8"))
    for key, record in sorted(census["shards"].items()):
        stem = key.split("/", 1)[1]
        path = Path(store_dir) / f"{stem}.npz"
        if not path.exists():
            raise ValueError(f"the tlog fingerprint store has no {stem}.npz ({key})")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"{stem}.npz is not the fingerprint file the committed census "
                             f"names for {key}")
    return sha256_file(FP_RECORD_PATH)


# --- the build ---------------------------------------------------------------------------------


def build(overlap_dir: Path = OVERLAP_DIR, *, fp_dir: Path = FP_DIR, store_dir: Path = STORE_DIR,
          index_dir: Path = INDEX_DIR, pass1_dir: Path = PASS1_DIR,
          floors_path: Path = FLOORS_PATH, split_dir: Path | None = None,
          code_sha: str | None = None, pull_request: int | None = None) -> dict:
    """Write the committed overlap tree and ``meta.json``; return the meta.

    ``fp_dir`` is the external store :func:`external` wrote (and keeps the control candidates;
    its frame sets' ``revision.txt`` is the pinned Tadabur commit the build records, and
    ``frames/`` its per-shard dispositions, which must reconcile).
    ``store_dir`` is S3b's tlog fingerprint store, read only and verified npz by npz against
    the committed census. Rows come from
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
    missing: dict[str, list[str]] = {}
    for code in SETS:
        if code in FRAME_SETS:
            continue
        unexpected = sorted(set(external[code]) - set(clips[code]))
        if unexpected:
            raise ValueError(f"{code}: the store holds {len(unexpected)} clips the committed "
                             f"checksums do not name, e.g. {unexpected[0]}")
        missing[code] = sorted(set(clips[code]) - set(external[code]))
    gate = gate_membership()
    if census["gate_eval"]["members"] != len(gate):
        raise ValueError(f"gate_eval: {census['gate_eval']['members']} fingerprints "
                         f"for {len(gate)} committed rows")
    frame_missing, frame_counts = _frame_inventory(Path(fp_dir))
    for code in FRAME_SETS:
        if census[code]["members"] != frame_counts[code]:
            raise ValueError(f"{code}: the store holds {census[code]['members']} fingerprints; "
                             f"the frame inventory claims {frame_counts[code]}")
    missing["tadabur_frame"] = frame_missing
    frame_revision = _frame_revision(Path(fp_dir))
    store_census_sha = _verify_tlog_store(Path(store_dir))

    index = EvalIndex.build({f"{code}{_ROW_SEP}{clip}": external[code][clip]
                             for code in SETS for clip in sorted(external[code])})
    control = synthetic_recall_control(index, Path(fp_dir))
    eval_index = _eval_index(Path(store_dir), split_dir)
    sha_map = _sha_map(clips)
    hits, edges, groups, metadata, stats = _scan(
        index, eval_index, sha_map, Path(store_dir), Path(index_dir), Path(pass1_dir),
        Path(floors_path), external["tarteel_v1"])

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
            records.append(record)
        write_shard(records, overlap_dir / split / f"{stem}.jsonl.gz")
        written += len(records)
        print(f"[build] wrote {split}/{stem}: {len(records)} rows", flush=True)

    net = {base: {role: Counter() for role in ROLES}
           for base in ("usable_before_s4", "gating_eligible")}
    usable_after_s4: Counter = Counter({role: 0 for role in ROLES})
    survivor_keys = set(resolved.values())
    survivor_facts: dict[tuple, tuple[bool, str | None]] = {}
    losers: list[tuple[tuple, dict]] = []
    for row in iter_pool_rows(overlap_dir=overlap_dir, index_dir=index_dir,
                              pass1_dir=pass1_dir, floors_path=floors_path):
        role, key = row["role"], _key(row)
        if row["usable"]:
            usable_after_s4[role] += 1
        before = (role != TRAIN and not row["exclusion_reasons"]
                  and not row["s1_eval_overlap"])
        gating = before and row["split"] == "clean" and row["filename_label"] is not None
        for base, applies in (("usable_before_s4", before),
                              ("gating_eligible", gating)):
            if applies:
                net[base][role]["before"] += 1
                net[base][role]["lost"] += not row["usable"]
        if key in survivor_keys:
            survivor_facts[key] = (row["usable"], _survivor_reason(row))
        if row["duplicate_of"] is not None:
            losers.append((key, row))

    survivor_unusable: dict[str, Counter] = {"eval": Counter(), "train": Counter()}
    for key, row in losers:
        if row["exclusion_reasons"] or row["s1_eval_overlap"] or row["eval_overlap"]:
            continue
        target = row["duplicate_of"]
        usable, reason = survivor_facts[(target["split"], target["shard_file"],
                                         target["row"])]
        if usable:
            continue
        group = "train" if row["role"] == TRAIN else "eval"
        survivor_unusable[group][reason] += 1

    net_sizes = {base: {role: {"before": counts["before"], "lost": counts["lost"],
                               "remaining": counts["before"] - counts["lost"]}
                        for role, counts in roles.items() if role != TRAIN}
                 for base, roles in net.items()}
    duplicates_meta = {kind: dict(sorted(counts.items()))
                       for kind, counts in duplicates.items()}
    duplicates_meta["survivor_unusable"] = {
        group: {"total": sum(counts.values()), **dict(sorted(counts.items()))}
        for group, counts in survivor_unusable.items()}
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
            "revision": (frame_revision if code in FRAME_SETS else sources[code].revision),
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
        "duplicates": duplicates_meta,
        "net_sizes": net_sizes,
        "usable_after_s4": {role: usable_after_s4[role] for role in ROLES},
        "controls": {
            "synthetic_recall": control,
            "s1_id_overlap_recall": _s1_recall_record(stats),
            "stop_word_matches": {
                "verified": stats["verified"],
                "mostly_stop_word": stats["mostly_stop_word"],
                "share": round(stats["mostly_stop_word"] / stats["verified"], 6)
                if stats["verified"] else 0.0,
                "mostly_stop_word_share": MOSTLY_STOP_WORD,
                "accepted": {
                    "all": _stop_word_record(
                        stats["accepted"]["external"] + stats["accepted"]["near_duplicate"]),
                    "external": _stop_word_record(stats["accepted"]["external"]),
                    "near_duplicate": _stop_word_record(
                        stats["accepted"]["near_duplicate"])}}},
        "ber_histogram": {"bins": list(_BINS),
                          "counts": [stats["histogram"][index]
                                     for index in range(len(_BINS))],
                          "rejected": [stats["rejected"][index]
                                       for index in range(len(_BINS))]},
        "ambiguous_zone": sorted(stats["ambiguous"],
                                 key=lambda hit: (hit["set"], hit["clip"], hit["ber"])),
        "blind_spots": {
            "sealed_panel": SEALED_PANEL,
            "train_near_duplicates_searched": False,
            "external_without_checksums": sorted(FRAME_SETS),
            "unfingerprinted": {code: ids for code, ids in missing.items() if ids},
            "why": ("the sealed panel cannot be read (the seal pins two authorisations); "
                    "train-train near duplicates were scoped out; frame and gate_eval have "
                    "no source-byte checksums, so they are matched by fingerprint only; a "
                    "committed clip with no fingerprint (undecodable, or shorter than two "
                    "frames) is listed in unfingerprinted")},
        "frozen": _frozen_inputs(split_dir, Path(pass1_dir)),
        "tlog_store": {"path": str(store_dir),
                       "census": {"path": repo_relative(FP_RECORD_PATH),
                                  "sha256": store_census_sha}},
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
