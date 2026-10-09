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
(shard and row); :class:`ExposureOverlap` lists what they share.

The frozen inputs the indexed uses come from are committed in ``exposure/sources/``: the
``decode_evalset`` manifest and the list of clips shipped to Muraja.

Usage (from ``tools/``; ``--index`` is ``tadabur.staged_audio index --shards 0-384``)::

  python -m tadabur.exposure build --index stage/full_index.jsonl
  python -m tadabur.exposure describe
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
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .staged_audio import (
    MINING_POOL,
    SEALED_PANEL,
    IndexRow,
    StagedClip,
    load_staged_clips,
    read_shard_index,
)
from .truth_sites import NUM_SHARDS, SOURCES, TRUTH_SITES_DIR, load_truth_sites

EXPOSURE_DIR = Path(__file__).parent / "exposure"
#: The frozen inputs the indexed uses are derived from, committed beside them.
SOURCES_DIRNAME = "sources"
EVALSET_MANIFEST_PATH = EXPOSURE_DIR / SOURCES_DIRNAME / "decode_evalset.manifest.json"
MURAJA_CLIPS_PATH = EXPOSURE_DIR / SOURCES_DIRNAME / "muraja_clips.json"

# --- the vocabulary of uses ------------------------------------------------------------

#: A truth site of each source (``tadabur.truth_sites.SOURCES``) sits on its recording.
TRUTH_SITE_USES = {source: f"truth_site.{source}" for source in sorted(SOURCES)}
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
#: The uses acceptance rules §6 names that no issue has made yet. Each has an empty file
#: until its issue writes it, so "unused" is recorded rather than inferred from absence.
PROBE_TRAINING = "probe.training"
PROBE_KL_CONTROL = "probe.kl_control"
BIAS_TUNE = "bias.tune"
BIAS_SCORE = "bias.score"
SHADDAH_PROBE_TUNING = "shaddah_probe.tuning"

USES = frozenset({
    *TRUTH_SITE_USES.values(),
    LABEL_P35_FIXTURE, LABEL_REJECT_REREAD, LABEL_REJECT_BLEED, LABEL_WAQF_EVENTS,
    LABEL_TASHKEEL_COUNTERFACTUAL,
    MINING_POOL, EVALSET_DEV, EVALSET_TEST, EVALSET_LEGACY, H448_TRAINING, H448_INIT_VALIDATION,
    MURAJA_REREAD_CORPUS, SYNTHETIC_EDIT_SOURCE, SYNTHETIC_EDIT_DONOR, SEALED_PANEL,
    PROBE_TRAINING, PROBE_KL_CONTROL, BIAS_TUNE, BIAS_SCORE, SHADDAH_PROBE_TUNING,
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
    """Every use's exposures, recording-level and shard-level, as loaded and validated."""

    recordings: Mapping[str, tuple[Exposure, ...]]
    shard_uses: Mapping[str, ShardExposure]

    def missing(self) -> list[str]:
        """Declared uses (:data:`USES`) with no file: evidence that is absent, not empty."""
        return sorted(USES - set(self.recordings) - set(self.shard_uses))

    def shared_baseline_uses(self) -> list[str]:
        """The shard uses marked as exposures shared by the baseline and its candidates."""
        return sorted(u for u, e in self.shard_uses.items() if e.shared_baseline)

    def uses(self) -> list[str]:
        """Every use with a file, sorted."""
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
    one filename always the same shard, row, reciter and (where recorded) checksum."""
    recordings: dict[str, tuple[Exposure, ...]] = {}
    shard_uses: dict[str, ShardExposure] = {}
    for path in sorted(directory.iterdir()):
        if path.name in ("README.md", SOURCES_DIRNAME):
            continue
        if path.name.endswith(".shards.json"):
            use = _known(path.name.removesuffix(".shards.json"))
            shard_uses[use] = _parse_shard_use(json.loads(path.read_text(encoding="utf-8")), path)
        elif path.suffix == ".jsonl":
            recordings[_known(path.stem)] = _read_use(path)
        else:
            raise ValueError(f"{path}: not a use file")
    if both := set(recordings) & set(shard_uses):
        raise ValueError(f"{directory}: {sorted(both)} are recorded in both shapes")
    _check_consistent(recordings)
    return ExposureRegistry(recordings, shard_uses)


def _read_use(path: Path) -> tuple[Exposure, ...]:
    with open(path, encoding="utf-8") as f:
        rows = [
            parse_exposure(json.loads(raw), f"{path}:{lineno}")
            for lineno, raw in enumerate(f, 1) if raw.strip()
        ]
    if rows != sorted(set(rows)):
        raise ValueError(f"{path}: rows must be sorted and unique")
    return tuple(rows)


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

    ``by="reciter"`` compares canonical reciter ids; ``by="source"`` compares recordings
    (``(shard, row_index)``; a shard use covers every row of its shards). Only ``use`` is
    compared with each other use, never the others among themselves (two truth-site
    sources may share a clip). A named use with no file raises :class:`ExposureIncomplete`
    (absent evidence is not an empty use), and an unknown use name is an error. The
    registry is loaded from ``exposure/`` unless one is passed.
    """
    if by not in ("reciter", "source"):
        raise ValueError(f"by must be 'reciter' or 'source', not {by!r}")
    registry = registry or load_registry()
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
        return sorted(set(registry.shard_uses[a].shards) & set(registry.shard_uses[b].shards))
    if b in registry.shard_uses:
        a, b = b, a
    if a in registry.shard_uses:
        shards = set(registry.shard_uses[a].shards)
        return sorted({r.source for r in registry.recordings.get(b, ()) if r.shard in shards})
    sources_a = {r.source for r in registry.recordings.get(a, ())}
    return sorted(sources_a & {r.source for r in registry.recordings.get(b, ())})


# --- exclusion: what a PRD use must not touch -------------------------------------------

#: The uses this PRD makes (acceptance rules §6, owner amendment): everything that must be
#: reciter-disjoint from the sealed panel.
PRD_USES = frozenset({
    *TRUTH_SITE_USES.values(), MINING_POOL, SYNTHETIC_EDIT_SOURCE, SYNTHETIC_EDIT_DONOR,
    PROBE_TRAINING, PROBE_KL_CONTROL, BIAS_TUNE, BIAS_SCORE, SHADDAH_PROBE_TUNING,
})
#: The PRD uses that train or tune; §6 also keeps the bias score half's reciters out of
#: them, edit donors included.
TRAINING_AND_TUNING_USES = frozenset({
    SYNTHETIC_EDIT_SOURCE, SYNTHETIC_EDIT_DONOR, PROBE_TRAINING, PROBE_KL_CONTROL, BIAS_TUNE,
    SHADDAH_PROBE_TUNING,
})


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
    """Drops the rows of excluded reciters from a stream of Tadabur rows **before** any
    audio is read, and counts them, so a full-shard consumer (training, mining) never
    reaches the sealed panel's audio and can report what it left out. Rows keep their
    order; a consumer that numbers rows numbers them before filtering."""

    use: str
    reciters: frozenset[int]
    rows_seen: int = 0
    rows_excluded: int = 0

    @classmethod
    def for_use(cls, use: str, registry: ExposureRegistry | None = None) -> "RowExclusion":
        return cls(use, excluded_reciters(use, registry))

    def keeps(self, row: Mapping) -> bool:
        """Whether one row may be used; counts it either way."""
        self.rows_seen += 1
        if int(row["reciter_id"]) in self.reciters:
            self.rows_excluded += 1
            return False
        return True

    def filter(self, rows: Iterable[Mapping]) -> Iterable[Mapping]:
        """The rows :meth:`keeps` admits, in order."""
        return (row for row in rows if self.keeps(row))

    def report(self) -> dict:
        return {"use": self.use, "excluded_reciters": len(self.reciters),
                "rows_seen": self.rows_seen, "rows_excluded": self.rows_excluded}


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


def _whole(row: IndexRow | StagedClip, sha: str | None) -> Exposure:
    return Exposure(row.audio_filename, row.shard, row.row_index, row.reciter_id, None, None, sha)


def staged_exposures(staged: Mapping[str, StagedClip]) -> dict[str, list[Exposure]]:
    """The uses the staged-clip registry alone determines: each truth-site source's item
    spans (with the staged checksum) and the mining pool's whole clips."""
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
    uses[MINING_POOL] = [_whole(c, c.audio_sha256) for c in staged.values() if MINING_POOL in c.uses]
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
        return _whole(row, staged[name].audio_sha256 if name in staged else None)

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


def describe(registry: ExposureRegistry) -> str:
    """One line per use: recordings (or shards), reciters."""
    lines = []
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
    commands.add_parser("describe", help="summarize the committed registry")
    args = parser.parse_args()

    if args.command == "describe":
        print(describe(load_registry()))
        return
    build_registry(args.index)
    print(describe(load_registry()))


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
    for use in load_registry(directory).missing():
        write_use(use, [], directory)


if __name__ == "__main__":
    main()
