"""The reject sink — what the ``.balanced`` gate computes and today throws away.

``tadabur.filter.score_batch`` builds a full :class:`~tadabur.scorer.GateResult` for
every clip it decodes and keeps only ``if result.passed``. Every signal needed to
classify a *reject* — ``match_ratio``, ``max_insertion_run``, ``added_shadda``, the
edge trims, the decoded phonemes — is already there and is then discarded. This
module is where those rejects land instead.

The reason to keep them is that **re-reads are systematically absent from the passing
subset**. ADR-0001's ``scorer.MAX_INSERTION_RUN`` reject exists precisely because "a
repeated phrase barely dents ``match_ratio``", and a re-read *is* a repeated phrase,
so a clip with a substantial re-read cannot pass the gate. Muraja ADR-0016 mines that
reject pile for the natural re-reads its follow-along tuning corpus needs, on one
stated predicate (:func:`is_clean_re_read`).

Two pieces live here:

* :class:`RejectRecord` — one rejected clip, carrying the whole ``GateResult`` plus the
  clip identity, duration and decode, and the :func:`reject_causes` labels;
* :class:`RejectSink` — the append-only JSONL writer, with the same idempotency
  guarantee as :class:`~tadabur.manifest.FilterManifest` (a per-``audio_filename``
  seen-set, so a replayed batch appends no duplicates).

The sink deliberately has **no checkpoint of its own**. It is driven by
``FilterManifest.commit_batch``, which writes rejects and passers before advancing the
single ``clips_processed`` position, so one resume position governs both files and they
cannot drift apart. Nothing here touches the gate: :mod:`tadabur.scorer` is read, never
written, and a run without ``--rejects`` opens no sink at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from .normalization import normalize_phonemes
from .scorer import MAX_INSERTION_RUN, MIN_QUERY_PHONEMES, GateResult, Scorer

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# Reject-cause vocabulary. A clip can fail on more than one of the last three at once
# (a low-ratio decode may also carry a long insertion run), so :func:`reject_causes`
# returns every cause that applies rather than a single "primary" one — a breakdown
# that claimed to be a partition would be inventing an ordering the gate does not have.
# The first two are terminal: the gate returns early on them and never computes the
# rest, so they are reported alone.
CAUSE_MIN_QUERY = "min_query"
CAUSE_NO_ALIGNMENT = "no_alignment"
CAUSE_LOW_RATIO = "low_ratio"
CAUSE_INSERTION_RUN = "insertion_run"
CAUSE_ADDED_SHADDA = "added_shadda"
REJECT_CAUSES = (
    CAUSE_MIN_QUERY,
    CAUSE_NO_ALIGNMENT,
    CAUSE_LOW_RATIO,
    CAUSE_INSERTION_RUN,
    CAUSE_ADDED_SHADDA,
)

# The **clean re-read** predicate (Muraja ADR-0016 decision 2): a recitation that matches
# its reference everywhere except for a repeated span. The insertion-run floor is the
# gate's own reject bar — a clip clearing it would have passed — so it is read from
# :mod:`tadabur.scorer` rather than restated. The ratio floor is a separate *mining*
# policy: it keeps the repeat as the only substantial divergence, so that what is left
# after the repeat is recitation the oracle can grade word-for-word. It coincides
# numerically with ``scorer.STRICT.correct_threshold`` but is not derived from it — the
# strict gate is a training-data bar, this is a corpus-selection bar, and they are free
# to move apart. ``added_shadda`` is excluded because a clip carrying one is a
# mispronunciation as well as a repeat, which muddies a re-read oracle.
CLEAN_RE_READ_MIN_RATIO = 0.75


@dataclass(frozen=True)
class RejectRecord:
    """One clip the ``.balanced`` gate rejected, with the signals behind the verdict.

    ``audio_filename`` is Tadabur's stable per-clip reference and the sink's
    idempotency key; ``surah_ayah``, ``reciter_id`` and ``ayah_duration_s`` mirror
    :class:`~tadabur.manifest.ManifestRecord` so the two files are joinable and a
    reject is traceable back to its audio and reference ayah. The middle five fields
    are :class:`~tadabur.scorer.GateResult` verbatim — the gate's whole verdict, not a
    summary of it — and ``predicted_phonemes`` is the decode they were computed from,
    so any predicate over rejects (including :func:`is_clean_re_read`) can be re-derived
    offline without a GPU. ``causes`` labels *why* it failed (:data:`REJECT_CAUSES`).
    """

    audio_filename: str
    surah_ayah: str
    reciter_id: int
    ayah_duration_s: float
    match_ratio: float
    max_insertion_run: int
    leading_trim: int
    trailing_trim: int
    added_shadda: bool
    predicted_phonemes: str
    causes: tuple[str, ...] = ()

    @property
    def is_clean_re_read(self) -> bool:
        """Whether this reject matches the clean-re-read predicate (see module docs)."""
        return is_clean_re_read(
            match_ratio=self.match_ratio,
            max_insertion_run=self.max_insertion_run,
            added_shadda=self.added_shadda,
        )


def is_clean_re_read(
    match_ratio: float, max_insertion_run: int, added_shadda: bool
) -> bool:
    """Whether a *rejected* clip's gate signals mark it as a clean re-read.

    Callers pass signals from a clip that already failed the gate; the predicate does
    not re-check ``passed``, because the insertion-run floor
    (:data:`~tadabur.scorer.MAX_INSERTION_RUN`) is itself a gate reject — a clip
    clearing it and this ratio would have passed. See the module docstring for why
    each term is there.
    """
    return (
        max_insertion_run >= MAX_INSERTION_RUN
        and match_ratio >= CLEAN_RE_READ_MIN_RATIO
        and not added_shadda
    )


def reject_causes(predicted: str, result: GateResult, scorer: Scorer) -> tuple[str, ...]:
    """Label every gate condition ``predicted`` failed, in :data:`REJECT_CAUSES` order.

    Returns ``()`` for a passer. The two terminal causes are returned alone: the gate
    returns early on a too-short query and on a non-positive alignment, so the other
    signals on the :class:`~tadabur.scorer.GateResult` are defaults, not measurements,
    and reporting them as causes would be reading noise. Telling those two apart needs
    the query phoneme count, which the gate computes and does not keep, so it is
    recomputed here — a few microseconds of string work against a GPU decode, and the
    alternative (widening ``GateResult``) would move a module ADR-0001 pins.
    """
    if result.passed:
        return ()
    query_phoneme_count = sum(
        1 for ch in normalize_phonemes(predicted).normalized if ch != " "
    )
    if query_phoneme_count < MIN_QUERY_PHONEMES:
        return (CAUSE_MIN_QUERY,)
    if result.match_ratio <= 0.0:
        return (CAUSE_NO_ALIGNMENT,)

    causes: list[str] = []
    if result.match_ratio < scorer.params.correct_threshold:
        causes.append(CAUSE_LOW_RATIO)
    if result.max_insertion_run >= MAX_INSERTION_RUN:
        causes.append(CAUSE_INSERTION_RUN)
    if result.added_shadda:
        causes.append(CAUSE_ADDED_SHADDA)
    return tuple(causes)


def build_reject_record(
    audio_filename: str,
    surah_ayah: str,
    reciter_id: int,
    ayah_duration_s: float,
    predicted: str,
    result: GateResult,
    scorer: Scorer,
) -> RejectRecord:
    """Assemble a :class:`RejectRecord` from a clip and the gate verdict it earned."""
    return RejectRecord(
        audio_filename=audio_filename,
        surah_ayah=surah_ayah,
        reciter_id=reciter_id,
        ayah_duration_s=ayah_duration_s,
        match_ratio=result.match_ratio,
        max_insertion_run=result.max_insertion_run,
        leading_trim=result.leading_trim,
        trailing_trim=result.trailing_trim,
        added_shadda=result.added_shadda,
        predicted_phonemes=predicted,
        causes=reject_causes(predicted, result, scorer),
    )


class RejectSink:
    """Append-only JSONL sink for rejected clips, idempotent across resumes.

    Open with :meth:`open` (a context manager), which recovers the ``audio_filename``
    keys already on disk so a resumed — or crash-replayed — run appends no duplicate.
    It carries no progress checkpoint: :class:`~tadabur.manifest.FilterManifest` owns
    the single ``clips_processed`` position and commits this sink before advancing it.
    """

    def __init__(self, path: Path, file, seen: set[str]) -> None:  # type: ignore[no-untyped-def]
        self.path = path
        self._file = file
        self._seen = seen

    @classmethod
    def open(cls, path: Path) -> "RejectSink":
        """Open ``path`` for appending, recovering the keys already written to it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        seen = _read_seen_keys(path)
        return cls(path, open(path, "a", encoding="utf-8"), seen)

    def append_batch(self, records: list[RejectRecord]) -> None:
        """Append every not-yet-seen ``record`` as one JSONL line, then fsync.

        The fsync is what makes the caller's ordering meaningful: these lines are
        durable before the progress checkpoint that would let a resume skip past them.
        """
        for record in records:
            if record.audio_filename in self._seen:
                continue
            self._seen.add(record.audio_filename)
            self._file.write(
                json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n"
            )
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()

    @property
    def rejects_written(self) -> int:
        """Number of distinct rejected clips written to the sink so far."""
        return len(self._seen)

    def __enter__(self) -> "RejectSink":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_reject_records(path: Path) -> list[RejectRecord]:
    """Load every :class:`RejectRecord` from ``path`` in file order."""
    records: list[RejectRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(
                RejectRecord(
                    audio_filename=data["audio_filename"],
                    surah_ayah=data["surah_ayah"],
                    reciter_id=data["reciter_id"],
                    ayah_duration_s=data["ayah_duration_s"],
                    match_ratio=data["match_ratio"],
                    max_insertion_run=data["max_insertion_run"],
                    leading_trim=data["leading_trim"],
                    trailing_trim=data["trailing_trim"],
                    added_shadda=data["added_shadda"],
                    predicted_phonemes=data.get("predicted_phonemes", ""),
                    causes=tuple(data.get("causes", ())),
                )
            )
    return records


def write_clip_wav(directory: Path, audio_filename: str, waveform: "np.ndarray") -> Path:
    """Write ``waveform`` as a 16 kHz mono WAV named ``audio_filename`` under ``directory``.

    This is the staged audio the Muraja half of ADR-0016 replays, so it is written at
    :data:`~tadabur.audio.TARGET_SAMPLE_RATE` — the same 16 kHz mono the gate scored,
    not the source clip's rate — and under the clip's own stable name, which is the
    join key back to the reject manifest row. Rewriting the same clip after a crash
    replay reproduces identical bytes, so the staging directory is idempotent too.
    """
    # Imported here, not at module scope: reading the sink (:func:`read_reject_records`,
    # and everything :mod:`tadabur.reject_yield` and :mod:`tadabur.bleed_detect` do with
    # it) is pure JSON over signals the run already computed, and must not drag in the
    # audio stack. Only *writing* a clip's audio needs it.
    import soundfile as sf

    from .audio import TARGET_SAMPLE_RATE

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / audio_filename
    sf.write(path, waveform, TARGET_SAMPLE_RATE, subtype="FLOAT")
    return path


def _read_seen_keys(path: Path) -> set[str]:
    """Recover the ``audio_filename`` keys already written to ``path``."""
    if not path.exists():
        return set()
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["audio_filename"])
    return seen
