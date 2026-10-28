"""The exposure registry: every canonical reciter and recording, with each use it has had.

Acceptance rules §6 ask for **one committed registry** listing every canonical reciter id,
source recording (Tadabur ``audio_filename``), span and checksum, with each use it has had,
so that a held-out claim can be checked rather than asserted: the sealed panel (#89) must be
reciter-disjoint from every other use, a bias score half from every tuning use, and so on.
This module owns that record (``exposure/`` next to it, one file per use) and the check.

**Reciters are canonical.** ``reciter_id`` is Tadabur's own ``reciter_id`` column, never the
filename's ``spkNNNN`` (they disagree on ~1% of rows, :mod:`tadabur.staged_audio`). Every
recording is identified by its parquet ``shard`` and ``row_index`` as well as its filename,
and the loader refuses a registry in which one filename names two rows or two reciters.

**Two shapes of use.** Most uses touched known recordings and are listed row by row in
``exposure/<use>.jsonl`` (:class:`Exposure`: the recording, an optional sample span, and the
staged file's SHA-256 when the recording was re-staged). A use that consumed whole shards,
``h448``'s streaming training, is a :class:`ShardExposure` in ``exposure/<use>.shards.json``:
its shards and how many rows each reciter has in them, counted from a full shard index.
A recording overlaps a shard use when its shard is one of the use's shards.

**One file per use** means each issue writes only its own files (:func:`write_use`,
:func:`write_shard_use`), so parallel work never edits a shared table, and a use is
re-derived by re-running the builder that owns it.

The check is :func:`check_disjoint`, one use against each of the others::

    from tadabur.exposure import (
        SEALED_PANEL, SYNTHETIC_EDIT_SOURCE, TRUTH_SITE_USES, check_disjoint,
    )
    check_disjoint(SYNTHETIC_EDIT_SOURCE, SEALED_PANEL, *TRUTH_SITE_USES.values())

``by="reciter"`` compares canonical reciter ids, ``by="source"`` the recordings themselves
(shard and row, or the same audio checksum); :class:`ExposureOverlap` lists what they share.

**Duplicate recordings.** Tadabur holds byte-identical clips under more than one file name
and shard row: under two filename speaker ids, or for two ayahs of identical text. Two
committed records say which rows are one recording, and ``by="source"`` treats them as one:

* **aliases** (``exposure/recording_aliases.jsonl``, :func:`write_recording_aliases`): rows
  confirmed identical by the checksum of their staged audio. The file only grows, so a
  recording removed from a use (a pruned panel clip) keeps its evidence;
* **probable copies** (``exposure/probable_copies.json``, :func:`copy_groups`): rows of any
  shard, checksummed or not, that the screen :data:`COPY_SCREEN` puts together. A probable
  copy is a conflict on suspicion, never a pass, against shard uses too.

The screen finds what it was measured to find (:func:`screen_agreement`), no more: a check
that passes means **no detected overlap** under it, not proven recording disjointness.
``python -m tadabur.exposure duplicates`` reports both records.

The frozen inputs the indexed uses come from are committed in ``exposure/sources/``: the
``decode_evalset`` manifest and the list of clips shipped to Muraja.

Usage (from ``tools/``; ``--index`` is ``tadabur.staged_audio index --shards 0-384``)::

  python -m tadabur.exposure build --index stage/full_index.jsonl
  python -m tadabur.exposure staged     # after a truth-site file changes (no index needed)
  python -m tadabur.exposure describe
  python -m tadabur.exposure copies --index stage/full_index.jsonl   # the two records alone
  python -m tadabur.exposure duplicates
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .staged_audio import (
    MINING_POOL,
    SEALED_PANEL,
    IndexRow,
    StagedClip,
    load_staged_clips,
    read_shard_index,
)
from .truth_sites import NUM_SHARDS, TADABUR_SOURCES, TRUTH_SITES_DIR, load_truth_sites

EXPOSURE_DIR = Path(__file__).parent / "exposure"
#: The frozen inputs the indexed uses are derived from, committed beside them.
SOURCES_DIRNAME = "sources"
EVALSET_MANIFEST_PATH = EXPOSURE_DIR / SOURCES_DIRNAME / "decode_evalset.manifest.json"
MURAJA_CLIPS_PATH = EXPOSURE_DIR / SOURCES_DIRNAME / "muraja_clips.json"
#: The probable-copy screen's groups (:func:`write_probable_copies`), beside the use files.
PROBABLE_COPIES_NAME = "probable_copies.json"
#: Rows confirmed one recording by checksum (:func:`write_recording_aliases`).
RECORDING_ALIASES_NAME = "recording_aliases.jsonl"
_RELATION_FILES = (PROBABLE_COPIES_NAME, RECORDING_ALIASES_NAME)

# --- the vocabulary of uses ------------------------------------------------------------

#: A truth site of each Tadabur source (``tadabur.truth_sites.TADABUR_SOURCES``) sits on its
#: recording. QUD's sites sit on QUD ayat, a use of their own (:data:`QUD_CORRECT_RECITATION`).
TRUTH_SITE_USES = {source: f"truth_site.{source}" for source in sorted(TADABUR_SOURCES)}
#: Clip-level human verdicts in committed label files that are not truth sites.
LABEL_P35_FIXTURE = "human_label.p35_fixture"
LABEL_REJECT_REREAD = "human_label.reject_reread"
LABEL_REJECT_BLEED = "human_label.reject_bleed"
LABEL_WAQF_EVENTS = "human_label.waqf_events"
LABEL_TASHKEEL_COUNTERFACTUAL = "human_label.tashkeel_counterfactual"
#: The frozen ``decode_evalset``: its reciter halves, and the legacy ratio-stratified
#: sample older manifests still carry (scored by earlier gates, so it is exposed too).
EVALSET_DEV = "decode_evalset.dev"
EVALSET_TEST = "decode_evalset.test"
EVALSET_LEGACY = "decode_evalset.legacy_stratified"
#: The shipped student's streaming training (a shard use).
H448_TRAINING = "h448.training"
#: ``h448_init``'s teacher-init calibration and ``h448``'s validation windows: drawn from
#: the ``clips_v2`` corpus (the filter's passes over shards 0-19), lost with ``audit_run/``,
#: so which rows is unknown (a shard use with uncertain membership).
H448_INIT_VALIDATION = "h448.init_validation"
#: Shards 0-19, the ``clips_v2`` corpus's source.
CLIPS_V2_SHARDS = range(0, 20)
#: The re-read corpus and scenario bundles shipped to Muraja for follow-along tuning.
MURAJA_REREAD_CORPUS = "muraja.reread_corpus"
#: Synthetic edits (#88): the clip an edit was made on, and the clip a swap took audio from.
SYNTHETIC_EDIT_SOURCE = "synthetic_edit.source"
SYNTHETIC_EDIT_DONOR = "synthetic_edit.donor"
#: Ship A (#126): every recording its mining extension decoded (h448-unseen; hit or not),
#: and the recordings of its verified-label halves (the reciter split, D11): the train half's
#: sites (training labels) and the held-out half's sites and sampled words (evaluation).
SHIP_A_MINING = "ship_a.mining"
SHIP_A_TRAIN = "ship_a.verified_train"
SHIP_A_HELD_OUT = "ship_a.verified_held_out"
#: The QUD ayat #144's correct-recitation set is built on: QUD audio, not Tadabur rows, so a
#: use of its own shape (:class:`QudExposure`, ``exposure/<use>.qud.jsonl``).
QUD_CORRECT_RECITATION = "qud.correct_recitation"
QUD_USES = frozenset({QUD_CORRECT_RECITATION})
#: The non-professional corpora's clips #156 decoded (Tarteel v1, RetaSy, qdat, qdat_bench):
#: evaluation only. Hub dataset clips, a use of their own shape (:class:`ClipExposure`,
#: ``exposure/<use>.clips.jsonl``).
NONPRO_EVALUATION = "nonpro.evaluation_only"
#: The tlog hold-out's five roles (#192, S7): ``look`` and the four sealed ``confirm``
#: tranches, each a use of its whole role — every usable row, whatever its stratum — so a
#: future consonant round can open one tranche knowing what was reserved. A tranche's rows are
#: committed in ``tlog/tranches/<role>.jsonl`` and written here by
#: ``python -m tadabur.tlog_tranches manifests``.
TLOG_USES = frozenset({f"tlog.{role}" for role in
                       ("look", "confirm.1", "confirm.2", "confirm.3", "confirm.4")})
CLIP_USES = frozenset({NONPRO_EVALUATION}) | TLOG_USES
#: The tlog training pool (#193, S8): a **row-set use**, not a clip or shard use. tlog shards
#: hold training and evaluation rows side by side, so this use names its rows exactly, by a
#: predicate over the overlay at ``overlay_commit`` and the count and SHA-256 of their sorted
#: row keys (:class:`RowSetExposure`, ``exposure/<use>.rows.json``). It is the one external
#: use that may train; :func:`check_disjoint` still refuses it in every evaluation context and
#: :func:`check_row_set_disjoint` checks it against every evaluation use.
TLOG_TRAINING = "tlog.training"
ROW_SET_USES = frozenset({TLOG_TRAINING})
#: Every use of audio from outside Tadabur: evaluation only, except :data:`TLOG_TRAINING`, and
#: none holds a Tadabur recording, so :func:`check_disjoint` refuses to certify anything with
#: one.
EXTERNAL_USES = QUD_USES | CLIP_USES | ROW_SET_USES
#: The uses acceptance rules §6 names that no issue has made yet. Each has an empty file
#: until its issue writes it, so "unused" is recorded rather than inferred from absence.
PROBE_TRAINING = "probe.training"
PROBE_KL_CONTROL = "probe.kl_control"
BIAS_TUNE = "bias.tune"
BIAS_SCORE = "bias.score"
SHADDAH_PROBE_TUNING = "shaddah_probe.tuning"
#: #94's augmentation experiment: ``h448`` continued with augmented student input, and its
#: control, the same run with augmentation off. Streaming training over ``h448``'s shards
#: (shard uses), written when the arms launch.
AUG_TRAINING = "aug.training"
AUG_CONTROL = "aug.control"
#: #178's consonant round: C95 continued toward the teacher on the shards #94's run left --
#: `h448`'s 345 `--stream-shards` minus the prefix #94 streamed (derived by
#: ``training.consumed_prefix`` from the seed-9494 permutation and its log), warm-started
#: from C95's averaged weights. A shard use, membership `exact`, written before launch.
CONSONANT_TRAINING = "consonant1.training"

USES = frozenset({
    *TRUTH_SITE_USES.values(),
    LABEL_P35_FIXTURE, LABEL_REJECT_REREAD, LABEL_REJECT_BLEED, LABEL_WAQF_EVENTS,
    LABEL_TASHKEEL_COUNTERFACTUAL,
    MINING_POOL, EVALSET_DEV, EVALSET_TEST, EVALSET_LEGACY, H448_TRAINING, H448_INIT_VALIDATION,
    MURAJA_REREAD_CORPUS, SYNTHETIC_EDIT_SOURCE, SYNTHETIC_EDIT_DONOR, SEALED_PANEL,
    PROBE_TRAINING, PROBE_KL_CONTROL, BIAS_TUNE, BIAS_SCORE, SHADDAH_PROBE_TUNING,
    SHIP_A_MINING, SHIP_A_TRAIN, SHIP_A_HELD_OUT, AUG_TRAINING, AUG_CONTROL,
    CONSONANT_TRAINING,
})

#: ``h448``'s ``--stream-shards``, verbatim from ``runs/h448_stream/run_config.json`` on the
#: GPU box: every shard but the held-out block 0-20 and the strided reserve.
H448_STREAM_SHARDS = (
    "21-38,40-57,59-76,78-95,97-114,116-133,135-152,154-171,173-190,192-209,211-228,"
    "230-247,249-266,268-285,287-304,306-323,325-342,344-361,363-380,382-384"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


# --- the record ------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Exposure:
    """One recording (or one span of it) that a use touched. ``start_sample`` and
    ``end_sample`` are ``None`` together when the use took the whole recording;
    ``audio_sha256`` is the staged 16 kHz PCM_16 WAV's when the recording was re-staged."""

    audio_filename: str
    shard: int
    row_index: int
    reciter_id: int
    start_sample: int | None
    end_sample: int | None
    audio_sha256: str | None

    @property
    def source(self) -> tuple[int, int]:
        """The Tadabur recording: ``(shard, row_index)``."""
        return (self.shard, self.row_index)


@dataclass(frozen=True, order=True)
class QudExposure:
    """One QUD ayah clip a use touched (#144): the QUD revision it was read at, the mushaf
    (config slug), the ayah, the SHA-256 of its MP3 and QUD's reciter id. It is no Tadabur
    recording; that it overlaps none is the audio check's finding
    (``qud_eval/overlap_results.json``), not this record's."""

    qud_revision: str
    slug: str
    surah_ayah: str
    mp3_sha256: str
    qud_reciter: str


_QUD_FIELDS = {f.name for f in fields(QudExposure)}


def parse_qud_exposure(data: dict, where: str) -> QudExposure:
    if set(data) != _QUD_FIELDS or not all(type(v) is str and v for v in data.values()):
        raise ValueError(f"{where}: a QUD exposure has the non-empty string fields "
                         f"{sorted(_QUD_FIELDS)}")
    row = QudExposure(**data)
    if not _SHA256.fullmatch(row.mp3_sha256) or not re.fullmatch(r"[0-9a-f]{40}", row.qud_revision):
        raise ValueError(f"{where}: mp3_sha256 or qud_revision is malformed")
    return row


@dataclass(frozen=True, order=True)
class ClipExposure:
    """One clip of a Hub dataset a use touched (#156): the dataset and the revision it was read
    at, the clip's id in it, the canonical ayah it recites, the SHA-256 of its audio as
    published and the dataset's reciter id (``unknown`` where it has none). Like a QUD ayah it
    is no Tadabur recording; whether it overlaps one is an audio check's finding."""

    dataset: str
    revision: str
    audio_filename: str
    surah_ayah: str
    audio_sha256: str
    reciter: str


_CLIP_EXPOSURE_FIELDS = {f.name for f in fields(ClipExposure)}


def parse_clip_exposure(data: dict, where: str) -> ClipExposure:
    if set(data) != _CLIP_EXPOSURE_FIELDS or not all(type(v) is str and v
                                                     for v in data.values()):
        raise ValueError(f"{where}: a clip exposure has the non-empty string fields "
                         f"{sorted(_CLIP_EXPOSURE_FIELDS)}")
    row = ClipExposure(**data)
    if not _SHA256.fullmatch(row.audio_sha256) or not re.fullmatch(r"[0-9a-f]{40}", row.revision):
        raise ValueError(f"{where}: audio_sha256 or revision is malformed")
    return row


@dataclass(frozen=True)
class RowSetExposure:
    """A use that names its rows by predicate rather than listing them (#193): the overlay
    commit the predicate was evaluated at, the predicate itself (the rule's text and the code
    that applies it), how many rows it selected, and the SHA-256 of their sorted row keys
    (``split:shard_file:row`` newline-joined). The loader re-materialises the rows and
    verifies the count and the hash, so a use file cannot drift from the overlay it cites.
    """

    overlay_commit: str
    predicate: Mapping[str, str]
    count: int
    rows_sha256: str


_ROW_SET_FIELDS = {f.name for f in fields(RowSetExposure)}
_PREDICATE_FIELDS = {"text", "code"}


def parse_row_set_exposure(data: dict, where: str) -> RowSetExposure:
    """One validated :class:`RowSetExposure` from its JSON object; ``where`` prefixes errors."""
    if set(data) != _ROW_SET_FIELDS:
        raise ValueError(f"{where}: fields {sorted(data)} are not the row-set use's "
                         f"{sorted(_ROW_SET_FIELDS)}")
    commit = data["overlay_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"{where}: overlay_commit is not a 40-character lowercase hex commit")
    predicate = data["predicate"]
    if (not isinstance(predicate, Mapping) or set(predicate) != _PREDICATE_FIELDS
            or not all(type(value) is str and value for value in predicate.values())):
        raise ValueError(f"{where}: predicate is an object with exactly the non-empty strings "
                         f"{sorted(_PREDICATE_FIELDS)}")
    if type(data["count"]) is not int or data["count"] < 0:
        raise ValueError(f"{where}: count is not a non-negative integer")
    digest = data["rows_sha256"]
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError(f"{where}: rows_sha256 is not a SHA-256")
    return RowSetExposure(commit, dict(predicate), data["count"], digest)


@dataclass(frozen=True)
class ShardExposure:
    """A use that consumed whole shards: the shards, how many rows each canonical reciter
    has in them, and the SHA-256 of the shard index the rows were counted from.

    ``membership`` says what is known of the rows inside those shards: ``"exact"`` when
    the shards are the use's exact input (a streaming run over them), ``"uncertain"`` when
    the use took some unknown subset of their rows, so every row counts as possibly
    exposed. ``shared_baseline`` marks an exposure of ``h448`` itself: the baseline and
    every candidate warm-started from it share it, so a paired claim (acceptance rules §6,
    owner amendment) may overlap it by reciter, and by recording only where membership is
    uncertain. ``note`` says where the use's shards come from.
    """

    shards: tuple[int, ...]
    rows_per_reciter: Mapping[int, int]
    index_sha256: str
    membership: str
    shared_baseline: bool
    note: str


_EXPOSURE_TYPES = {
    "audio_filename": (str,),
    "shard": (int,),
    "row_index": (int,),
    "reciter_id": (int,),
    "start_sample": (int, type(None)),
    "end_sample": (int, type(None)),
    "audio_sha256": (str, type(None)),
}
assert set(_EXPOSURE_TYPES) == {f.name for f in fields(Exposure)}


def parse_exposure(data: dict, where: str) -> Exposure:
    """One validated :class:`Exposure` from its JSON object; ``where`` prefixes errors."""
    if set(data) != set(_EXPOSURE_TYPES):
        raise ValueError(f"{where}: fields {sorted(data)} do not match {sorted(_EXPOSURE_TYPES)}")
    for name, allowed in _EXPOSURE_TYPES.items():
        if type(data[name]) not in allowed:  # exact: a bool is not an int
            raise ValueError(f"{where}: {name} has type {type(data[name]).__name__}")
    row = Exposure(**data)
    if not row.audio_filename.endswith(".wav") or "/" in row.audio_filename:
        raise ValueError(f"{where}: audio_filename must be a bare .wav name")
    if not 0 <= row.shard < NUM_SHARDS or row.row_index < 0 or row.reciter_id < 0:
        raise ValueError(f"{where}: shard, row_index or reciter_id is out of range")
    if (row.start_sample is None) != (row.end_sample is None):
        raise ValueError(f"{where}: start_sample and end_sample are null together or not at all")
    if row.start_sample is not None and not 0 <= row.start_sample < row.end_sample:
        raise ValueError(f"{where}: the span [{row.start_sample}, {row.end_sample}) is empty")
    if row.audio_sha256 is not None and not _SHA256.fullmatch(row.audio_sha256):
        raise ValueError(f"{where}: audio_sha256 is not 64 lowercase hex characters")
    return row


@dataclass(frozen=True)
class ExposureRegistry:
    """Every use's exposures, recording-level and shard-level, as loaded and validated, and
    which rows are one recording, by ``(shard, row_index)``: each screened row's probable
    copies and each row's checksum-confirmed aliases (:func:`relations`)."""

    recordings: Mapping[str, tuple[Exposure, ...]]
    shard_uses: Mapping[str, ShardExposure]
    copies: Mapping[tuple[int, int], tuple[tuple[int, int], ...]] = field(default_factory=dict)
    aliases: Mapping[tuple[int, int], tuple[tuple[int, int], ...]] = field(default_factory=dict)
    qud: Mapping[str, tuple[QudExposure, ...]] = field(default_factory=dict)
    clips: Mapping[str, tuple[ClipExposure, ...]] = field(default_factory=dict)
    row_sets: Mapping[str, RowSetExposure] = field(default_factory=dict)

    def same_recording(self, source: tuple[int, int]) -> set[tuple[int, int]]:
        """``source`` and every row that is, or probably is, the same recording."""
        return {source, *self.copies.get(source, ()), *self.aliases.get(source, ())}

    def related_sources(self) -> set[tuple[int, int]]:
        """Every row known or screened to share its recording with another."""
        return set(self.copies) | set(self.aliases)

    def missing(self) -> list[str]:
        """Declared uses (:data:`USES`, :data:`EXTERNAL_USES`) with no file: evidence that is
        absent, not empty."""
        return sorted((USES | EXTERNAL_USES) - set(self.recordings) - set(self.shard_uses)
                      - set(self.qud) - set(self.clips) - set(self.row_sets))

    def shared_baseline_uses(self) -> list[str]:
        """The shard uses marked as exposures shared by the baseline and its candidates."""
        return sorted(u for u, e in self.shard_uses.items() if e.shared_baseline)

    def uses(self) -> list[str]:
        """Every Tadabur use with a file, sorted (external uses are in :attr:`qud`,
        :attr:`clips` and :attr:`row_sets`)."""
        return sorted({*self.recordings, *self.shard_uses})

    def reciters(self, use: str) -> frozenset[int]:
        """The canonical reciter ids a use touched. A use with no file is an error."""
        self._require(use)
        if use in self.shard_uses:
            return frozenset(self.shard_uses[use].rows_per_reciter)
        return frozenset(row.reciter_id for row in self.recordings[use])

    def _require(self, use: str) -> None:
        if _known(use) not in self.recordings and use not in self.shard_uses:
            raise ExposureIncomplete(
                f"{use} has no file in the registry: write it (empty if the use has no "
                "exposures) before checking against it"
            )

    def uses_of(self, reciter_id: int) -> list[str]:
        """Every use that touched one reciter, sorted."""
        return [use for use in self.uses() if reciter_id in self.reciters(use)]


def _known(use: str) -> str:
    if use not in USES:
        raise ValueError(f"unknown use {use!r}; the vocabulary is tadabur.exposure.USES")
    return use


def _recording_path(use: str, directory: Path) -> Path:
    return directory / f"{_known(use)}.jsonl"


def _shard_path(use: str, directory: Path) -> Path:
    return directory / f"{_known(use)}.shards.json"


def load_registry(directory: Path = EXPOSURE_DIR) -> ExposureRegistry:
    """Every use file in ``directory``, validated one by one and then together: a file's
    rows sorted and unique, every file named for a known use, no use in both shapes, and
    one filename always the same shard, row, reciter and (where recorded) checksum. The
    recording records (aliases, probable copies) are loaded with it when present; an
    alias must name a recording the uses name the same way, with the same checksum."""
    recordings: dict[str, tuple[Exposure, ...]] = {}
    shard_uses: dict[str, ShardExposure] = {}
    qud: dict[str, tuple[QudExposure, ...]] = {}
    clips: dict[str, tuple[ClipExposure, ...]] = {}
    row_sets: dict[str, RowSetExposure] = {}
    for path in sorted(directory.iterdir()):
        if path.name in ("README.md", SOURCES_DIRNAME, *_RELATION_FILES) or path.is_dir():
            continue
        if path.name.endswith(".qud.jsonl"):
            use = path.name.removesuffix(".qud.jsonl")
            if use not in QUD_USES:
                raise ValueError(f"unknown QUD use {use!r}; the vocabulary is QUD_USES")
            qud[use] = _read_external_use(path, parse_qud_exposure)
        elif path.name.endswith(".clips.jsonl"):
            use = path.name.removesuffix(".clips.jsonl")
            if use not in CLIP_USES:
                raise ValueError(f"unknown clip use {use!r}; the vocabulary is CLIP_USES")
            clips[use] = _read_external_use(path, parse_clip_exposure)
        elif path.name.endswith(".rows.json"):
            use = path.name.removesuffix(".rows.json")
            if use not in ROW_SET_USES:
                raise ValueError(f"unknown row-set use {use!r}; the vocabulary is ROW_SET_USES")
            row_sets[use] = parse_row_set_exposure(
                json.loads(path.read_text(encoding="utf-8")), str(path))
        elif path.name.endswith(".shards.json"):
            use = _known(path.name.removesuffix(".shards.json"))
            shard_uses[use] = _parse_shard_use(json.loads(path.read_text(encoding="utf-8")), path)
        elif path.suffix == ".jsonl":
            recordings[_known(path.stem)] = _read_use(path)
        else:
            raise ValueError(f"{path}: not a use file")
    if both := set(recordings) & set(shard_uses):
        raise ValueError(f"{directory}: {sorted(both)} are recorded in both shapes")
    _check_consistent(recordings)
    copies_path, aliases_path = (directory / name for name in _RELATION_FILES)
    copies = relations(read_probable_copies(copies_path)) if copies_path.exists() else {}
    aliases: dict = {}
    if aliases_path.exists():
        groups = read_recording_aliases(aliases_path)
        _check_consistent({**recordings, RECORDING_ALIASES_NAME: [
            Exposure(r.audio_filename, r.shard, r.row_index, r.reciter_id, None, None,
                     r.audio_sha256) for r in groups]})
        aliases = relations(_alias_groups(groups))
    return ExposureRegistry(recordings, shard_uses, copies, aliases, qud, clips, row_sets)


def _read_use(path: Path) -> tuple[Exposure, ...]:
    with open(path, encoding="utf-8") as f:
        rows = [
            parse_exposure(json.loads(raw), f"{path}:{lineno}")
            for lineno, raw in enumerate(f, 1) if raw.strip()
        ]
    if rows != sorted(set(rows)):
        raise ValueError(f"{path}: rows must be sorted and unique")
    return tuple(rows)


def _read_external_use(path: Path, parse) -> tuple:
    """An external use's rows, each validated by ``parse``, sorted and unique."""
    with open(path, encoding="utf-8") as f:
        rows = [parse(json.loads(raw), f"{path}:{lineno}")
                for lineno, raw in enumerate(f, 1) if raw.strip()]
    if rows != sorted(set(rows)):
        raise ValueError(f"{path}: rows must be sorted and unique")
    return tuple(rows)


def write_qud_use(use: str, rows: Iterable[QudExposure], directory: Path = EXPOSURE_DIR) -> None:
    """Atomically write one QUD use, sorted and de-duplicated, every row validated."""
    if use not in QUD_USES:
        raise ValueError(f"unknown QUD use {use!r}; the vocabulary is QUD_USES")
    _write_external_use(directory / f"{use}.qud.jsonl", rows, parse_qud_exposure)


def write_clip_use(use: str, rows: Iterable[ClipExposure], directory: Path = EXPOSURE_DIR) -> None:
    """Atomically write one clip use, sorted and de-duplicated, every row validated."""
    if use not in CLIP_USES:
        raise ValueError(f"unknown clip use {use!r}; the vocabulary is CLIP_USES")
    _write_external_use(directory / f"{use}.clips.jsonl", rows, parse_clip_exposure)


def write_row_set_use(use: str, row_set: RowSetExposure,
                      directory: Path = EXPOSURE_DIR) -> None:
    """Atomically write the one row-set use, validated first."""
    if use not in ROW_SET_USES:
        raise ValueError(f"unknown row-set use {use!r}; the vocabulary is ROW_SET_USES")
    data = asdict(row_set)
    path = directory / f"{use}.rows.json"
    parse_row_set_exposure(data, str(path))
    _write_atomic(path, json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n")


def _write_external_use(path: Path, rows: Iterable, parse) -> None:
    ordered = sorted(set(rows))
    for number, row in enumerate(ordered):
        parse(asdict(row), f"{path.name}[{number}]")
    _write_atomic(path, "".join(
        json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in ordered))


def write_empty(use: str, directory: Path = EXPOSURE_DIR) -> None:
    """Record a declared use as unused: an empty file of its shape."""
    if use in QUD_USES:
        write_qud_use(use, [], directory)
    elif use in CLIP_USES:
        write_clip_use(use, [], directory)
    elif use in ROW_SET_USES:
        from .tlog_pool import empty_training_use

        write_row_set_use(use, empty_training_use(), directory)
    else:
        write_use(use, [], directory)


_SHARD_USE_FIELDS = {
    "shards", "rows_per_reciter", "index_sha256", "membership", "shared_baseline", "note",
}
MEMBERSHIPS = ("exact", "uncertain")


def _parse_shard_use(data: dict, where: Path) -> ShardExposure:
    if set(data) != _SHARD_USE_FIELDS:
        raise ValueError(f"{where}: fields {sorted(data)} are not a shard use")
    if data["membership"] not in MEMBERSHIPS or type(data["shared_baseline"]) is not bool:
        raise ValueError(f"{where}: membership must be one of {MEMBERSHIPS}, shared_baseline a bool")
    shards = tuple(data["shards"])
    if list(shards) != sorted(set(shards)) or not all(0 <= s < NUM_SHARDS for s in shards):
        raise ValueError(f"{where}: shards must be sorted, unique and in range")
    if not _SHA256.fullmatch(data["index_sha256"]):
        raise ValueError(f"{where}: index_sha256 is not a SHA-256")
    rows = {int(reciter): count for reciter, count in data["rows_per_reciter"].items()}
    if not all(type(count) is int and count > 0 for count in rows.values()):
        raise ValueError(f"{where}: rows_per_reciter counts must be positive integers")
    return ShardExposure(
        shards, rows, data["index_sha256"], data["membership"], data["shared_baseline"],
        data["note"],
    )


def _check_consistent(recordings: Mapping[str, Iterable[Exposure]]) -> None:
    identity: dict[str, tuple[int, int, int]] = {}
    sources: dict[tuple[int, int], str] = {}
    checksums: dict[str, str] = {}
    for use, rows in sorted(recordings.items()):
        for row in rows:
            name = row.audio_filename
            if identity.setdefault(name, (row.shard, row.row_index, row.reciter_id)) != (
                row.shard, row.row_index, row.reciter_id
            ):
                raise ValueError(f"{use}: {name} disagrees with another use on its row or reciter")
            if sources.setdefault(row.source, name) != name:
                raise ValueError(f"{use}: {name} and {sources[row.source]} claim one shard row")
            if row.audio_sha256 is not None and (
                checksums.setdefault(name, row.audio_sha256) != row.audio_sha256
            ):
                raise ValueError(f"{use}: {name} is recorded with two checksums")


def write_use(use: str, rows: Iterable[Exposure], directory: Path = EXPOSURE_DIR) -> None:
    """Atomically write one recording-level use, sorted and de-duplicated, after validating
    every row and checking it against the rest of the registry."""
    ordered = sorted(set(rows))
    for row in ordered:
        parse_exposure(asdict(row), f"write_use({use})[{row.audio_filename}]")
    others = load_registry(directory).recordings if directory.exists() else {}
    _check_consistent({**others, use: ordered})
    _write_atomic(
        _recording_path(use, directory),
        "".join(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in ordered),
    )


def write_shard_use(use: str, exposure: ShardExposure, directory: Path = EXPOSURE_DIR) -> None:
    """Atomically write one shard-level use."""
    data = {
        "shards": list(exposure.shards),
        "rows_per_reciter": {str(k): v for k, v in sorted(exposure.rows_per_reciter.items())},
        "index_sha256": exposure.index_sha256,
        "membership": exposure.membership,
        "shared_baseline": exposure.shared_baseline,
        "note": exposure.note,
    }
    path = _shard_path(use, directory)
    _parse_shard_use(data, path)
    _write_atomic(path, json.dumps(data, indent=1, sort_keys=True) + "\n")


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


# --- the check -------------------------------------------------------------------------


class ExposureOverlap(ValueError):
    """Two uses that must be disjoint share reciters or recordings."""


class ExposureIncomplete(ValueError):
    """A declared use has no file, so its exposures are unknown rather than empty."""


def require_complete(registry: ExposureRegistry) -> None:
    """Raise :class:`ExposureIncomplete` unless every declared use has a file: what a
    certification (the panel's disjointness, a held-out half) must check first."""
    if missing := registry.missing():
        raise ExposureIncomplete(f"the registry has no file for {missing}")


def check_disjoint(
    use: str, *others: str, by: str = "reciter", registry: ExposureRegistry | None = None
) -> None:
    """Raise :class:`ExposureOverlap` unless ``use`` is disjoint from each of ``others``.

    ``by="reciter"`` compares canonical reciter ids; ``by="source"`` compares recordings:
    ``(shard, row_index)``, the audio checksum where both rows carry one, a confirmed alias
    or a probable copy (:meth:`ExposureRegistry.same_recording`), so a copy of a recording
    under another row is the same recording. A shard use covers every row of its shards, so
    a recording with a copy in one of them overlaps it, and two shard uses overlap when
    they share a shard or a recording. A pass means no overlap **detected**: the screen's
    recall is measured, not perfect (:func:`screen_agreement`). Only ``use`` is
    compared with each other use, never the others among themselves (two truth-site
    sources may share a clip). A named use with no file raises :class:`ExposureIncomplete`
    (absent evidence is not an empty use), and an unknown use name is an error. The
    registry is loaded from ``exposure/`` unless one is passed.
    """
    if by not in ("reciter", "source"):
        raise ValueError(f"by must be 'reciter' or 'source', not {by!r}")
    registry = registry or load_registry()
    if external := {use, *others} & EXTERNAL_USES:
        raise ValueError(f"{sorted(external)}: an external use (QUD, #156's corpora, #192's "
                         "tlog tranches) is evaluation only and holds no Tadabur recording, so "
                         "it certifies no training use; its overlap with Tadabur is an audio "
                         "check's (qud_eval/overlap_results.json, nonpro_eval/overlap.json, "
                         "tlog/overlap/). #193's tlog.training is a row-set use, checked by "
                         "check_row_set_disjoint")
    for name in (use, *others):
        registry._require(name)
    problems = []
    for other in sorted(set(others) - {use}):
        shared = (
            sorted(registry.reciters(use) & registry.reciters(other))
            if by == "reciter" else _shared_sources(registry, use, other)
        )
        if shared:
            problems.append(f"{use} and {other} share {len(shared)} {by}s, e.g. {shared[:5]}")
    if problems:
        raise ExposureOverlap("; ".join(problems))


def _shared_sources(registry: ExposureRegistry, a: str, b: str) -> list:
    if a in registry.shard_uses and b in registry.shard_uses:
        shards_a, shards_b = set(registry.shard_uses[a].shards), set(registry.shard_uses[b].shards)
        linked = {s for s in registry.related_sources() if s[0] in shards_a and any(
            c[0] in shards_b for c in registry.same_recording(s) - {s})}
        return sorted(shards_a & shards_b) + sorted(linked)
    if b in registry.shard_uses:
        a, b = b, a
    if a in registry.shard_uses:
        shards = set(registry.shard_uses[a].shards)
        return sorted({r.source for r in registry.recordings.get(b, ())
                       if any(shard in shards for shard, _ in registry.same_recording(r.source))})
    rows_b = registry.recordings.get(b, ())
    sources_b = {s for r in rows_b for s in registry.same_recording(r.source)}
    checksums_b = {r.audio_sha256 for r in rows_b} - {None}
    return sorted({r.source for r in registry.recordings.get(a, ())
                   if r.source in sources_b or r.audio_sha256 in checksums_b})


def check_row_set_disjoint(
    rows: Iterable[Mapping[str, object]], *, registry: ExposureRegistry | None = None,
    eval_groups: Mapping[str, Iterable[str]] | None = None,
) -> None:
    """Raise :class:`ExposureOverlap` unless a row-set use's rows are disjoint from every
    evaluation use (#193).

    ``rows`` are pool rows (:func:`tadabur.tlog_overlap.iter_pool_rows`' records). A collision,
    checked in that order, is:

    * ``audio_path``: a clip id of any clip use (``registry.clips``, the non-professional
      corpora's and each tlog role's), or the ``audio_filename`` of a registered Tadabur
      recording of a use in :data:`TRAINING_BARRED_USES`, equals a row's ``audio_path``;
    * ``audio_sha256``: a clip's checksum, a QUD ayah's MP3 checksum, or a registered Tadabur
      recording's (non-null) ``audio_sha256`` of a :data:`TRAINING_BARRED_USES` use, equals a
      row's. The S4 ``eval_overlap`` screen cannot stand in for this: a use registered after
      the screen holds recordings the row's frozen record never saw;
    * ``fingerprint``: a row carries an S4 ``eval_overlap`` hit -- the live case. Also a row
      whose ``duplicate_of`` names a row outside this set, which no usable pool row can carry
      (S4 excludes a duplicate: ``usable`` is false when ``duplicate_of`` is set), so this
      branch fires only on a pool bug and refuses it rather than training the audio twice;
    * ``group_key``: a row's ``group_key`` is one of a tlog evaluation use's
      (``eval_groups``, a ``use -> {group_key}`` map; by default the committed split tree's
      groups of each ``tlog.<role>``, :func:`tadabur.tlog_tranches.split_groups`).

    :data:`TRAINING_BARRED_USES` is the canonical barred vocabulary -- the frozen evalsets, the
    sealed panel, the truth sites, the human-label fixtures and both Ship A halves -- so a
    training or mining use is not matched, and a tlog row's missing Tadabur ``(shard, row)``
    identity leaves the file name and the checksum as the applicable direct checks. A barred
    use with no file contributes nothing here: :func:`require_complete` is the completeness
    check, and a use's absence is not evidence of disjointness.

    ``eval_groups`` defaults to the committed split tree, so the real registry is checked
    without passing anything. The registry is loaded from ``exposure/`` unless one is passed.
    """
    registry = registry or load_registry()
    rows = list(rows)
    keys = {_row_key(row) for row in rows}
    paths: dict[str, tuple[str, str]] = {}
    for use, clips in sorted(registry.clips.items()):
        for clip in clips:
            paths.setdefault(clip.audio_filename, ("clip", use))
    checksums: dict[str, tuple[str, str]] = {}
    for use, clips in sorted(registry.clips.items()):
        for clip in clips:
            checksums.setdefault(clip.audio_sha256, ("clip", use))
    for use, ayat in sorted(registry.qud.items()):
        for ayah in ayat:
            checksums.setdefault(ayah.mp3_sha256, ("QUD ayah", use))
    for use in TRAINING_BARRED_USES:
        for recording in registry.recordings.get(use, ()):
            paths.setdefault(recording.audio_filename, ("Tadabur recording", use))
            if recording.audio_sha256 is not None:
                checksums.setdefault(recording.audio_sha256, ("Tadabur recording", use))
    groups = _tlog_eval_groups() if eval_groups is None else {
        use: set(members) for use, members in eval_groups.items()}
    for row in rows:
        key = _row_key(row)
        if (owner := paths.get(row["audio_path"])) is not None:
            kind, use = owner
            raise ExposureOverlap(f"{key}: audio_path {row['audio_path']} is a {kind} of the "
                                  f"evaluation use {use}")
        if (owner := checksums.get(row["audio_sha256"])) is not None:
            kind, use = owner
            raise ExposureOverlap(f"{key}: audio_sha256 {row['audio_sha256']} is a {kind}'s, "
                                  f"in the evaluation use {use}")
        if row["eval_overlap"]:
            hit = row["eval_overlap"][0]
            raise ExposureOverlap(f"{key}: fingerprint match to {hit['set']} clip {hit['clip']} "
                                  f"at BER {hit['ber']} (S4 eval_overlap)")
        duplicate = row["duplicate_of"]
        if duplicate is not None and _row_key(duplicate) not in keys:
            raise ExposureOverlap(f"{key}: fingerprint of {_row_key(duplicate)}, a row outside "
                                  "this training set")
        for use, members in sorted(groups.items()):
            if row["group_key"] in members:
                raise ExposureOverlap(f"{key}: group_key {row['group_key']} is in the tlog "
                                      f"evaluation use {use}")


def _row_key(row: Mapping[str, object]) -> tuple[str, str, int]:
    return row["split"], row["shard_file"], row["row"]


def _tlog_eval_groups() -> dict[str, set[str]]:
    """Each tlog evaluation use's group keys, from the committed split tree."""
    from .tlog_tranches import split_groups

    return split_groups()


# --- duplicate recordings --------------------------------------------------------------


def duplicate_recordings(
    registry: ExposureRegistry,
) -> dict[str, dict[tuple[int, int], tuple[str, ...]]]:
    """Every checksum that two or more recordings (shard rows) carry, with each row's uses."""
    groups: dict[str, dict[tuple[int, int], set[str]]] = {}
    for use, rows in registry.recordings.items():
        for row in rows:
            if row.audio_sha256 is not None:
                groups.setdefault(row.audio_sha256, {}).setdefault(row.source, set()).add(use)
    return {sha: {source: tuple(sorted(uses)) for source, uses in sorted(group.items())}
            for sha, group in sorted(groups.items()) if len(group) > 1}


#: The probable-copy screen's rule. A copy keeps its reciter and its length; the file name
#: is no evidence either way (a copy may be filed under another speaker id or ayah).
COPY_SCREEN = "copy-screen-v2: the same canonical reciter and the same duration to the ms"

Source = tuple[int, int]


def copy_groups(index: Iterable[IndexRow]) -> list[tuple[Source, ...]]:
    """The probable-copy screen (:data:`COPY_SCREEN`) over a shard index: every group of two
    or more rows ``(shard, row_index)`` with one canonical reciter and one duration to the
    millisecond, each sorted, in order.

    A screen, not proof: it is measured against the checksums by :func:`screen_agreement`.
    """
    groups: dict[tuple[int, int], list[Source]] = {}
    for row in index:
        key = (row.reciter_id, round(row.duration_s * 1000))
        groups.setdefault(key, []).append((row.shard, row.row_index))
    return sorted(tuple(sorted(g)) for g in groups.values() if len(g) > 1)


def relations(groups: Iterable[Iterable[Source]]) -> dict[Source, tuple[Source, ...]]:
    """Each grouped row's partners in its group."""
    related = {}
    for group in groups:
        group = tuple(group)
        for source in group:
            related[source] = tuple(s for s in group if s != source)
    return related


def screen_agreement(
    groups: Iterable[Iterable[Source]], checksums: Mapping[Source, str]
) -> dict[str, int]:
    """How the screen agrees with the checksums, over the rows that have one: of the
    checksum-confirmed groups (two or more rows, one checksum), how many it puts wholly in
    one group (recall); and of the pairs of checksummed rows it groups, how many have one
    checksum and how many two (false positives)."""
    group_of = {s: i for i, group in enumerate(groups) for s in group}
    confirmed: dict[str, list[Source]] = {}
    for source, sha in checksums.items():
        confirmed.setdefault(sha, []).append(source)
    confirmed_groups = [g for g in confirmed.values() if len(g) > 1]
    found = sum(len({group_of.get(s, s) for s in g}) == 1 for g in confirmed_groups)
    members: dict[int, list[Source]] = {}
    for source in checksums:
        if source in group_of:
            members.setdefault(group_of[source], []).append(source)
    pairs = [(a, b) for group in members.values() for i, a in enumerate(group)
             for b in group[i + 1:]]
    same = sum(checksums[a] == checksums[b] for a, b in pairs)
    return {"checksummed_rows": len(checksums), "confirmed_groups": len(confirmed_groups),
            "confirmed_groups_found": found, "grouped_pairs_same_checksum": same,
            "grouped_pairs_different_checksum": len(pairs) - same}


def write_probable_copies(
    groups: list[tuple[Source, ...]], index_sha256: str, directory: Path = EXPOSURE_DIR
) -> None:
    """Commit the screen's groups: the rule, the index they were computed from, and each
    group's rows. Only the relation is stored; the uses it touches are derived on read."""
    record = {"screen": COPY_SCREEN, "index_sha256": index_sha256,
              "groups": [[list(s) for s in g] for g in groups]}
    _write_atomic(directory / PROBABLE_COPIES_NAME,
                  json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")


def read_probable_copies(path: Path) -> list[tuple[Source, ...]]:
    """The committed screen's groups, validated: this screen's rule, two or more distinct
    rows per group in order, and no row in two groups."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if set(record) != {"screen", "index_sha256", "groups"} or record["screen"] != COPY_SCREEN:
        raise ValueError(f"{path}: not the groups of the screen {COPY_SCREEN!r}")
    groups, seen = [], set()
    for raw in record["groups"]:
        group = tuple((int(shard), int(row)) for shard, row in raw)
        if len(group) < 2 or list(group) != sorted(set(group)) or seen & set(group):
            raise ValueError(f"{path}: a group needs two or more distinct, sorted rows")
        seen |= set(group)
        groups.append(group)
    return groups


@dataclass(frozen=True, order=True)
class Alias:
    """One row of a checksum-confirmed recording."""

    audio_sha256: str
    audio_filename: str
    shard: int
    row_index: int
    reciter_id: int

    @property
    def source(self) -> Source:
        return (self.shard, self.row_index)


def write_recording_aliases(rows: Iterable, directory: Path = EXPOSURE_DIR) -> None:
    """Record every checksum two or more rows carry, merged into the committed aliases,
    which are never dropped: evidence outlives the use that brought it (a pruned panel
    clip's alias still decides the next selection). ``rows`` are any records with
    ``audio_filename``, ``shard``, ``row_index``, ``reciter_id`` and ``audio_sha256``
    (staged clips, exposure rows); those without a checksum are skipped."""
    path = directory / RECORDING_ALIASES_NAME
    known = set(read_recording_aliases(path)) if path.exists() else set()
    known |= {Alias(r.audio_sha256, r.audio_filename, r.shard, r.row_index, r.reciter_id)
              for r in rows if r.audio_sha256 is not None}
    by_source: dict[Source, Alias] = {}
    for alias in known:
        if by_source.setdefault(alias.source, alias) != alias:
            raise ValueError(f"{alias.source} is recorded as {alias} and {by_source[alias.source]}")
    held = {sha: g for sha, g in _by_checksum(known).items() if len(g) > 1}
    _write_atomic(path, "".join(
        json.dumps(asdict(a), sort_keys=True) + "\n"
        for sha in sorted(held) for a in held[sha]))


def _by_checksum(aliases: Iterable[Alias]) -> dict[str, list[Alias]]:
    groups: dict[str, list[Alias]] = {}
    for alias in sorted(aliases):
        groups.setdefault(alias.audio_sha256, []).append(alias)
    return groups


def _alias_groups(aliases: Iterable[Alias]) -> list[tuple[Source, ...]]:
    return [tuple(a.source for a in g) for g in _by_checksum(aliases).values()]


def read_recording_aliases(path: Path) -> list[Alias]:
    """The committed aliases, validated: sorted, each checksum held by two or more rows."""
    with open(path, encoding="utf-8") as f:
        aliases = [Alias(**json.loads(raw)) for raw in f if raw.strip()]
    if aliases != sorted(set(aliases)) or any(
            len({a.source for a in g}) < 2 or not _SHA256.fullmatch(sha)
            for sha, g in _by_checksum(aliases).items()):
        raise ValueError(f"{path}: aliases must be sorted, unique, two or more per checksum")
    return aliases


def copies_across_uses(registry: ExposureRegistry) -> dict[tuple[str, str], int]:
    """For each pair of uses ``(a, b)``, how many recordings of ``a`` have an alias or a
    probable copy in ``b`` under another shard row: a row ``b`` lists, or any row of a
    shard use's shards."""
    uses_of: dict[tuple[int, int], set[str]] = {}
    for use, rows in registry.recordings.items():
        for row in rows:
            uses_of.setdefault(row.source, set()).add(use)
    shard_uses = {use: set(e.shards) for use, e in registry.shard_uses.items()}
    found: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for source, uses in uses_of.items():
        for copy in registry.same_recording(source) - {source}:
            theirs = uses_of.get(copy, set()) | {
                use for use, shards in shard_uses.items() if copy[0] in shards}
            for a in uses:
                for b in theirs - {a}:
                    found.setdefault((a, b), set()).add(source)
    return {key: len(sources) for key, sources in sorted(found.items())}


# --- exclusion: what a PRD use must not touch -------------------------------------------

#: The uses this PRD makes (acceptance rules §6, owner amendment): everything that must be
#: reciter-disjoint from the sealed panel.
PRD_USES = frozenset({
    *TRUTH_SITE_USES.values(), MINING_POOL, SYNTHETIC_EDIT_SOURCE, SYNTHETIC_EDIT_DONOR,
    PROBE_TRAINING, PROBE_KL_CONTROL, BIAS_TUNE, BIAS_SCORE, SHADDAH_PROBE_TUNING,
    SHIP_A_MINING, SHIP_A_TRAIN, SHIP_A_HELD_OUT, AUG_TRAINING, AUG_CONTROL,
    CONSONANT_TRAINING,
})
#: The PRD uses that train or tune; §6 also keeps the bias score half's reciters out of
#: them, edit donors included.
TRAINING_AND_TUNING_USES = frozenset({
    SYNTHETIC_EDIT_SOURCE, SYNTHETIC_EDIT_DONOR, PROBE_TRAINING, PROBE_KL_CONTROL, BIAS_TUNE,
    SHADDAH_PROBE_TUNING, SHIP_A_TRAIN, AUG_TRAINING, AUG_CONTROL, CONSONANT_TRAINING,
})


#: The uses an ``h448``-unseen population (Ship A's pool sites, mined clips and sampled
#: words, #126) may hold no recording of, not even as an alias or a probable copy: what
#: ``h448`` trained on, the frozen ``decode_evalset`` and the sealed panel.
UNSEEN_BARRED_USES = (H448_TRAINING, EVALSET_DEV, EVALSET_TEST, EVALSET_LEGACY, SEALED_PANEL)


def linked_sources(uses: Iterable[str], registry: ExposureRegistry) -> frozenset[Source]:
    """Every ``(shard, row)`` that is a recording of one of ``uses``, or that the registry
    ties to one (a checksum alias or a probable copy). A shard use's recordings are every row
    of its shards, so only its relatives are listed."""
    rows, shards = set(), set()
    for use in uses:
        registry._require(use)
        if use in registry.shard_uses:
            shards |= set(registry.shard_uses[use].shards)
        else:
            rows |= {row.source for row in registry.recordings[use]}
    linked = set(rows)
    for source in rows:
        linked |= registry.same_recording(source)
    linked |= {source for source in registry.related_sources()
               if any(shard in shards for shard, _ in registry.same_recording(source))}
    return frozenset(linked)


def unseen_violations(
    staged: Mapping[str, StagedClip], registry: ExposureRegistry | None = None
) -> frozenset[str]:
    """The staged clips an ``h448``-unseen population must leave out: each one whose
    recording is, or is tied to, a recording of :data:`UNSEEN_BARRED_USES`: a row a recording
    use lists, any row of a shard use's shards (``h448.training``), or a relative of either."""
    registry = registry or load_registry()
    linked = linked_sources(UNSEEN_BARRED_USES, registry)
    shards = {shard for use in UNSEEN_BARRED_USES if use in registry.shard_uses
              for shard in registry.shard_uses[use].shards}
    return frozenset(name for name, clip in staged.items()
                     if clip.shard in shards or (clip.shard, clip.row_index) in linked)


#: The uses a training or tuning stream may hold no recording of, not even as an alias or a
#: probable copy (#94): every evaluation use -- the frozen ``decode_evalset`` (both halves and
#: the legacy sample), the sealed panel, the truth sites, the human-label fixtures -- and both
#: of Ship A's verified halves. ``nonpro.evaluation_only`` is not here: it holds no Tadabur
#: recording, so the registry ties no shard row to it.
TRAINING_BARRED_USES = (
    EVALSET_DEV, EVALSET_TEST, EVALSET_LEGACY, SEALED_PANEL, *sorted(TRUTH_SITE_USES.values()),
    LABEL_P35_FIXTURE, LABEL_REJECT_REREAD, LABEL_REJECT_BLEED, LABEL_WAQF_EVENTS,
    LABEL_TASHKEEL_COUNTERFACTUAL, SHIP_A_TRAIN, SHIP_A_HELD_OUT,
)


def barred_recordings(
    uses: Iterable[str], registry: ExposureRegistry
) -> dict[Source, tuple[str, ...]]:
    """Every ``(shard, row)`` that is, or is tied to (an alias or a probable copy), a recording
    of one of ``uses``, with the uses it is tied to (:func:`linked_sources`, per use)."""
    barred: dict[Source, list[str]] = {}
    for use in uses:
        for source in linked_sources([use], registry):
            barred.setdefault(source, []).append(use)
    return {source: tuple(found) for source, found in sorted(barred.items())}


def excluded_reciters(for_use: str, registry: ExposureRegistry | None = None) -> frozenset[int]:
    """The canonical reciters a PRD use must not touch (§6): the sealed panel's always, and
    for a training or tuning use the bias score half's as well. Read from a complete
    registry, so a use that is missing evidence cannot pass as empty."""
    if _known(for_use) not in PRD_USES:
        raise ValueError(f"{for_use} is not a use this PRD makes; nothing to exclude for it")
    registry = registry or load_registry()
    barred = registry.reciters(SEALED_PANEL)
    if for_use in TRAINING_AND_TUNING_USES:
        barred |= registry.reciters(BIAS_SCORE)
    return frozenset(barred)


@dataclass
class RowExclusion:
    """Drops what a use must not touch from a stream of Tadabur rows **before** any audio is
    read, and counts it, so a full-shard consumer (training, mining) never reaches the sealed
    panel's audio and can report what it left out: the rows of excluded reciters
    (:func:`excluded_reciters`) and, for a training or tuning use, every row that is or is
    tied to a recording of :data:`TRAINING_BARRED_USES` (:func:`barred_recordings`). Each row
    is identified by its ``(shard, row_index)``, numbered before filtering, so the numbering
    is the parquet row's whatever is dropped."""

    use: str
    reciters: frozenset[int]
    recordings: Mapping[Source, tuple[str, ...]] = field(default_factory=dict)
    rows_seen: int = 0
    rows_excluded: int = 0
    #: Rows excluded per reason: ``reciter``, or each barred use a recording is tied to (a
    #: row tied to two uses counts under both, once in ``rows_excluded``).
    excluded_by: Counter = field(default_factory=Counter)

    @classmethod
    def for_use(cls, use: str, registry: ExposureRegistry | None = None) -> "RowExclusion":
        registry = registry or load_registry()
        recordings = (barred_recordings(TRAINING_BARRED_USES, registry)
                      if use in TRAINING_AND_TUNING_USES else {})
        return cls(use, excluded_reciters(use, registry), recordings)

    def keeps(self, source: Source, row: Mapping) -> bool:
        """Whether the row at ``source`` may be used; counts it either way."""
        self.rows_seen += 1
        if int(row["reciter_id"]) in self.reciters:
            reasons: tuple[str, ...] = ("reciter",)
        else:
            reasons = self.recordings.get(source, ())
        if not reasons:
            return True
        self.rows_excluded += 1
        self.excluded_by.update(reasons)
        return False

    def report(self) -> dict:
        return {"use": self.use, "excluded_reciters": len(self.reciters),
                "barred_recordings": len(self.recordings), "rows_seen": self.rows_seen,
                "rows_excluded": self.rows_excluded,
                "excluded_by": dict(sorted(self.excluded_by.items()))}


# --- building the uses that exist ------------------------------------------------------

LABEL_FILES = {
    LABEL_P35_FIXTURE: ("eval_fixtures/should_accept.jsonl", "eval_fixtures/should_reject.jsonl"),
    LABEL_REJECT_REREAD: ("eval_fixtures/reject_reread_verdicts.jsonl",),
    LABEL_REJECT_BLEED: ("eval_fixtures/reject_bleed_labels.jsonl",),
    LABEL_WAQF_EVENTS: (
        "waqf_event_fixtures/waqf_events.calibration.jsonl",
        "waqf_event_fixtures/waqf_events.test.jsonl",
    ),
    LABEL_TASHKEEL_COUNTERFACTUAL: ("tashkeel_counterfactual_fixtures/counterfactual_items.jsonl",),
}
#: The field each label file names its clip (or ``<clip>__seg<n>.wav`` segment) in.
_CLIP_FIELDS = ("audio_ref", "audio_filename")
_TADABUR_DIR = Path(__file__).parent


def label_file_clips(use: str) -> set[str]:
    """The whole Tadabur clips one human-label use's committed files name."""
    from .segment_score import parse_segment_id

    clips: set[str] = set()
    for relative in LABEL_FILES[use]:
        with open(_TADABUR_DIR / relative, encoding="utf-8") as f:
            for raw in filter(str.strip, f):
                row = json.loads(raw)
                (name,) = [row[k] for k in _CLIP_FIELDS if k in row]
                clips.add(parse_segment_id(name)[0] if "__seg" in name else name)
    return clips


def whole_recording(row: IndexRow | StagedClip, sha: str | None) -> Exposure:
    return Exposure(row.audio_filename, row.shard, row.row_index, row.reciter_id, None, None, sha)


def staged_exposures(staged: Mapping[str, StagedClip]) -> dict[str, list[Exposure]]:
    """The uses the staged-clip registry alone determines: each truth-site source's item
    spans (with the staged checksum) and the mining pool's whole clips. Ship A's mining use
    lists every recording its scan decoded, staged or not, so ``tadabur.ship_a_mining
    register`` writes it from the scan."""
    uses: dict[str, list[Exposure]] = {}
    for path in sorted(TRUTH_SITES_DIR.glob("*.jsonl")):
        if path.name.endswith(".relocation.jsonl"):
            continue
        for site in load_truth_sites(path):
            clip = staged[site.audio_filename]
            uses.setdefault(TRUTH_SITE_USES[site.source], []).append(Exposure(
                clip.audio_filename, clip.shard, clip.row_index, clip.reciter_id,
                site.start_sample, site.end_sample, clip.audio_sha256,
            ))
    uses[MINING_POOL] = [whole_recording(c, c.audio_sha256) for c in staged.values()
                         if MINING_POOL in c.uses]
    return uses


def indexed_exposures(
    index: Mapping[str, IndexRow],
    staged: Mapping[str, StagedClip],
    evalset_manifest: Path,
    muraja_clips: Iterable[str],
) -> dict[str, list[Exposure]]:
    """The uses only a shard index can place: the human-label files, the frozen
    ``decode_evalset`` (its manifest names rows, not filenames) and Muraja's corpus.
    A recording that was re-staged carries its staged checksum."""
    def exposure(name: str) -> Exposure:
        row = index[name]
        return whole_recording(row, staged[name].audio_sha256 if name in staged else None)

    uses = {use: [exposure(name) for name in label_file_clips(use)] for use in LABEL_FILES}
    uses[MURAJA_REREAD_CORPUS] = [exposure(name) for name in muraja_clips]
    by_source = {(row.shard, row.row_index): row.audio_filename for row in index.values()}
    for source, use in evalset_records(evalset_manifest):
        uses.setdefault(use, []).append(exposure(by_source[source]))
    return uses


def evalset_records(manifest_path: Path) -> list[tuple[tuple[int, int], str]]:
    """Every record of a ``decode_evalset`` manifest as ``((shard, row_index), use)``: its
    reciter half, or the legacy stratified sample for a record outside the population.

    The manifest names each clip ``tadabur_sh<shard>_i<row>_...`` (``decode_evalset``'s
    ``_clip_filename``); the shard is also checked against its own field.
    """
    pattern = re.compile(r"tadabur_sh(\d{3})_i(\d{5})_")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for clip in manifest["clips"]:
        match = pattern.match(clip["filename"])
        if match is None or int(match.group(1)) != clip["shard"]:
            raise ValueError(f"{manifest_path}: cannot read {clip['filename']!r}")
        use = (
            {"dev": EVALSET_DEV, "test": EVALSET_TEST}[clip["split"]]
            if clip.get("in_population", True) else EVALSET_LEGACY
        )
        records.append(((int(match.group(1)), int(match.group(2))), use))
    return records


def shard_exposure(
    index_path: Path, shards: Iterable[int], *, membership: str, shared_baseline: bool,
    note: str,
) -> ShardExposure:
    """Rows per canonical reciter in ``shards``, from a full shard index file."""
    wanted = set(shards)
    index = read_shard_index(index_path)
    covered = {row.shard for row in index.values()}
    if missing := wanted - covered:
        raise ValueError(f"{index_path} does not index shards {sorted(missing)[:10]}")
    counts = Counter(row.reciter_id for row in index.values() if row.shard in wanted)
    return ShardExposure(
        tuple(sorted(wanted)), dict(counts), hashlib.sha256(index_path.read_bytes()).hexdigest(),
        membership, shared_baseline, note,
    )


def training_stream_exposure(
    index_path: Path, shards: Iterable[int], use: str, *, membership: str, note: str,
    registry: ExposureRegistry | None = None,
) -> tuple[ShardExposure, dict]:
    """A streaming training use's shard use, and what its exclusion drops: rows per canonical
    reciter in ``shards`` counting only the rows :meth:`RowExclusion.for_use` admits (no
    excluded reciter, no row tied to an evaluation recording), from a full shard index. Not
    a shared-baseline exposure: it is the candidate's or the control's own."""
    wanted = set(shards)
    index = read_shard_index(index_path)
    if missing := wanted - {row.shard for row in index.values()}:
        raise ValueError(f"{index_path} does not index shards {sorted(missing)[:10]}")
    exclusion = RowExclusion.for_use(use, registry)
    counts = Counter(
        row.reciter_id
        for row in sorted(index.values(), key=lambda r: (r.shard, r.row_index))
        if row.shard in wanted
        and exclusion.keeps((row.shard, row.row_index), {"reciter_id": row.reciter_id})
    )
    exposure = ShardExposure(
        tuple(sorted(wanted)), dict(counts), hashlib.sha256(index_path.read_bytes()).hexdigest(),
        membership, False, note,
    )
    return exposure, exclusion.report()


def write_training_stream_use(
    use: str, exposure: ShardExposure, directory: Path = EXPOSURE_DIR
) -> None:
    """Write a streaming training use as a shard use, replacing the empty recording-level
    file that declared it unused (a use is recorded in one shape only)."""
    declared = _recording_path(use, directory)
    if declared.exists():
        if declared.read_text(encoding="utf-8").strip():
            raise ValueError(f"{declared} records recordings; refusing to replace it")
        declared.unlink()
    write_shard_use(use, exposure, directory)


def shard_use_drift(committed: ShardExposure, recomputed: ShardExposure) -> list[str]:
    """The fields on which a recomputed shard use disagrees with the committed one.

    ``note`` is prose about where the shards came from, not evidence, so an edited note is
    not drift; a changed shard list, admission count, index or membership is.
    """
    lines: list[str] = []
    if committed.shards != recomputed.shards:
        added = sorted(set(recomputed.shards) - set(committed.shards))
        removed = sorted(set(committed.shards) - set(recomputed.shards))
        lines.append(
            f"shards: committed {len(committed.shards)}, recomputed {len(recomputed.shards)}; "
            f"added {added[:10]}, removed {removed[:10]}"
        )
    if committed.rows_per_reciter != recomputed.rows_per_reciter:
        committed_rows = sum(committed.rows_per_reciter.values())
        recomputed_rows = sum(recomputed.rows_per_reciter.values())
        lines.append(
            f"rows_per_reciter: committed {committed_rows:,} rows over "
            f"{len(committed.rows_per_reciter)} reciters, recomputed {recomputed_rows:,} over "
            f"{len(recomputed.rows_per_reciter)}"
        )
    for name in ("index_sha256", "membership", "shared_baseline"):
        if getattr(committed, name) != getattr(recomputed, name):
            lines.append(
                f"{name}: committed {getattr(committed, name)!r}, "
                f"recomputed {getattr(recomputed, name)!r}"
            )
    return lines


def describe(registry: ExposureRegistry) -> str:
    """One line per use: recordings (or shards, or row-set rows), reciters."""
    lines = [f"{use:40s} {len(rows):>11d} QUD ayat {len({r.qud_reciter for r in rows}):>5d} "
             f"QUD reciters" for use, rows in sorted(registry.qud.items())]
    lines += [f"{use:40s} {len(rows):>11d} clips {len({r.dataset for r in rows}):>5d} datasets"
              for use, rows in sorted(registry.clips.items())]
    lines += [f"{use:40s} {row_set.count:>11d} rows of {row_set.overlay_commit[:8]} "
              f"{row_set.predicate['code']}"
              for use, row_set in sorted(registry.row_sets.items())]
    for use in registry.uses():
        if use in registry.shard_uses:
            size = f"{len(registry.shard_uses[use].shards)} shards"
        else:
            size = f"{len({r.source for r in registry.recordings[use]})} recordings"
        lines.append(f"{use:40s} {size:>18s} {len(registry.reciters(use)):>5d} reciters")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="write every use that exists before the panel")
    build.add_argument("--index", type=Path, required=True,
                       help="a shard index of all 385 shards (tadabur.staged_audio index)")
    commands.add_parser(
        "staged", help="rewrite the uses the staged-clip registry alone determines (each "
                       "truth-site source's items, the mining pool): after a new truth-site file")
    commands.add_parser("describe", help="summarize the committed registry")
    copies = commands.add_parser(
        "copies", help="rewrite the probable-copy screen and the checksum aliases")
    copies.add_argument("--index", type=Path, required=True,
                        help="a shard index of all 385 shards (tadabur.staged_audio index)")
    commands.add_parser("duplicates", help="report recordings held under two shard rows")
    stream = commands.add_parser(
        "training-stream", help="write a streaming training use (#94's arms) as a shard use: "
        "rows per reciter of the rows its exclusion admits")
    stream.add_argument("--use", required=True, choices=sorted(TRAINING_AND_TUNING_USES))
    stream.add_argument("--index", type=Path, required=True,
                        help="a shard index of all 385 shards (tadabur.staged_audio index)")
    stream.add_argument("--shards", required=True, help="the run's --stream-shards")
    stream.add_argument("--membership", choices=MEMBERSHIPS, required=True)
    stream.add_argument("--note", required=True)
    stream.add_argument(
        "--check",
        action="store_true",
        help="recompute and compare with the committed shard use instead of writing it; "
        "exits non-zero on drift, which is a run's launch-time data assertion",
    )
    args = parser.parse_args()

    if args.command == "training-stream":
        from .shard_reader import parse_shard_spec

        exposure, report = training_stream_exposure(
            args.index, parse_shard_spec(args.shards), args.use, membership=args.membership,
            note=args.note)
        if args.check:
            committed = load_registry().shard_uses.get(args.use)
            if committed is None:
                raise SystemExit(f"{args.use} has no committed shard use: write it first")
            drift = shard_use_drift(committed, exposure)
            if drift:
                raise SystemExit(
                    f"{args.use} has drifted from its committed shard use:\n"
                    + "\n".join(f"  {line}" for line in drift)
                )
            print(f"{args.use}: {len(exposure.shards)} shards, "
                  f"{sum(exposure.rows_per_reciter.values()):,} admitted rows -- "
                  f"matches the committed use")
            return
        write_training_stream_use(args.use, exposure)
        print(json.dumps(report, indent=1))
        return

    if args.command == "describe":
        print(describe(load_registry()))
        return
    if args.command == "staged":
        for use, rows in sorted(staged_exposures(load_staged_clips()).items()):
            write_use(use, rows)
        print(describe(load_registry()))
        return
    if args.command == "duplicates":
        _duplicates(load_registry())
        return
    if args.command == "copies":
        write_relations(args.index)
        _duplicates(load_registry())
        return
    build_registry(args.index)
    print(describe(load_registry()))


def _duplicates(registry: ExposureRegistry) -> None:
    groups = duplicate_recordings(registry)
    print(f"{len(groups)} checksums are held under more than one shard row:")
    for uses, count in sorted(Counter(tuple(g.values()) for g in groups.values()).items()):
        print(f"  {count:4d}  {' | '.join(', '.join(u) for u in uses)}")
    print(f"{len(registry.aliases)} rows have a checksum-confirmed alias; "
          f"{len(registry.copies)} indexed rows have a probable copy ({COPY_SCREEN}); "
          "recordings of one use with an alias or copy in another:")
    for (a, b), count in copies_across_uses(registry).items():
        print(f"  {count:4d}  {a} -> {b}")


def build_registry(index_path: Path, directory: Path = EXPOSURE_DIR) -> None:
    """Write every use this module derives, from the committed sources and a full shard
    index, and an empty file for each declared use no issue has written yet."""
    from .shard_reader import parse_shard_spec

    staged = load_staged_clips()
    index = read_shard_index(index_path)
    muraja = json.loads(MURAJA_CLIPS_PATH.read_text(encoding="utf-8"))
    uses = {
        **staged_exposures(staged),
        **indexed_exposures(index, staged, EVALSET_MANIFEST_PATH, muraja),
    }
    for use, rows in sorted(uses.items()):
        write_use(use, rows, directory)
    write_shard_use(H448_TRAINING, shard_exposure(
        index_path, parse_shard_spec(H448_STREAM_SHARDS), membership="exact",
        shared_baseline=True,
        note="h448's --stream-shards (runs/h448_stream/run_config.json on the GPU box)",
    ), directory)
    write_shard_use(H448_INIT_VALIDATION, shard_exposure(
        index_path, CLIPS_V2_SHARDS, membership="uncertain", shared_baseline=True,
        note="h448_init's calibration windows and h448's validation windows, from the "
        "clips_v2 corpus (the filter's passes over shards 0-19), lost with audit_run/: "
        "which rows is unknown, so every row of the shards counts",
    ), directory)
    for use, rows in synthetic_edit_exposures().items():
        write_use(use, rows, directory)
    for use in load_registry(directory).missing():
        write_empty(use, directory)
    write_relations(index_path, directory)


def write_relations(index_path: Path, directory: Path = EXPOSURE_DIR) -> dict[str, int]:
    """Rewrite the two recording records: the aliases from every checksum the staging
    registry and the uses hold (merged into those already committed), then the screen over
    a full shard index; return and print how the screen agrees with every known checksum."""
    registry = load_registry(directory)
    known = [*load_staged_clips().values(), *(r for rows in registry.recordings.values()
                                              for r in rows)]
    write_recording_aliases(known, directory)
    groups = copy_groups(read_shard_index(index_path).values())
    write_probable_copies(groups, hashlib.sha256(index_path.read_bytes()).hexdigest(), directory)
    aliases = read_recording_aliases(directory / RECORDING_ALIASES_NAME)
    checksums = {(r.shard, r.row_index): r.audio_sha256 for r in [*aliases, *known]
                 if r.audio_sha256 is not None}
    agreement = screen_agreement(groups, checksums)
    print(json.dumps({"screen": COPY_SCREEN, **agreement}, indent=1))
    return agreement


def synthetic_edit_exposures() -> dict[str, list[Exposure]]:
    """The synthetic-edit sources (whole clips) and donors (spans), from #88's committed
    ``synthetic_edits/edits.jsonl`` through its own :func:`tadabur.synthetic_edits.exposure_rows`."""
    from .synthetic_edits import exposure_rows, read_items

    return {use: [Exposure(**row) for row in rows]
            for use, rows in exposure_rows(read_items()).items()}


if __name__ == "__main__":
    main()
