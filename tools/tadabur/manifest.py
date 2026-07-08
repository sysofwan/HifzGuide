"""The passing-subset manifest and its resumable, idempotent write sink.

The Tadabur filter (``tadabur.filter``) streams 365k+ clips through the model once
and keeps only those whose decoded phonemes clear the ``.balanced`` gate. This
module owns where those passers land and how a run resumes after a crash.

Two files sit side by side:

* the **manifest** — an append-only JSONL of one :class:`ManifestRecord` per
  passing clip (audio ref, ``surah:ayah``, ``match_ratio``, ``ayah_duration``,
  reciter);
* the **progress checkpoint** — a tiny JSON holding ``clips_processed``, the number
  of clips consumed from the (deterministically-ordered) stream so far.

Resumability rests on two guarantees. The checkpoint lets the filter ``skip`` the
clips it already scored — including the rejected ones that leave no manifest line —
so a resumed run does not re-infer them. And every commit is ordered
manifest-then-checkpoint with an fsync between, so the only crash window replays the
last (uncheckpointed) batch; a per-``audio_filename`` seen-set makes that replay
append no duplicate manifest lines. The manifest is therefore idempotent across any
number of resumes.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType


@dataclass(frozen=True)
class ManifestRecord:
    """One passing clip in the filtered training subset.

    ``audio_filename`` is Tadabur's stable per-clip audio reference (and the
    idempotency key). ``surah_ayah`` is ``"surah:ayah"``. ``match_ratio`` is the
    ``.balanced`` gate score and ``ayah_duration_s`` the duration of the 16 kHz
    waveform actually scored.
    """

    audio_filename: str
    surah_ayah: str
    match_ratio: float
    ayah_duration_s: float
    reciter_id: int


def _checkpoint_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(manifest_path.suffix + ".progress.json")


class FilterManifest:
    """Append-only manifest writer with a resumable progress checkpoint.

    Open with :meth:`open` (a context manager) so the resume state is read from any
    existing manifest and checkpoint on disk. :attr:`clips_processed` is where the
    filter should resume the stream; :meth:`commit_batch` records a scored batch's
    passers and advances that position atomically.
    """

    def __init__(
        self,
        manifest_path: Path,
        checkpoint_path: Path,
        file,  # type: ignore[no-untyped-def]  (an open text file handle)
        seen: set[str],
        clips_processed: int,
    ) -> None:
        self.manifest_path = manifest_path
        self.checkpoint_path = checkpoint_path
        self._file = file
        self._seen = seen
        self.clips_processed = clips_processed

    @classmethod
    def open(cls, manifest_path: Path) -> "FilterManifest":
        """Open ``manifest_path`` for appending, reading any prior resume state.

        Recovers the set of already-written ``audio_filename`` keys from an existing
        manifest and ``clips_processed`` from the sibling checkpoint, so a resumed
        run neither re-scores earlier clips nor rewrites their manifest lines.
        """
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = _checkpoint_path(manifest_path)

        seen = _read_seen_keys(manifest_path)
        clips_processed = _read_clips_processed(checkpoint_path)
        file = open(manifest_path, "a", encoding="utf-8")
        return cls(manifest_path, checkpoint_path, file, seen, clips_processed)

    def commit_batch(
        self, records: list[ManifestRecord], num_clips: int
    ) -> None:
        """Append a scored batch's ``records`` and advance the checkpoint by ``num_clips``.

        Writes each not-yet-seen record as one JSONL line, fsyncs the manifest, then
        atomically rewrites the checkpoint. Records whose ``audio_filename`` is
        already present are skipped, so replaying an uncheckpointed batch after a
        crash adds no duplicates.
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

        self.clips_processed += num_clips
        _write_clips_processed(self.checkpoint_path, self.clips_processed)

    def close(self) -> None:
        self._file.close()

    @property
    def passers_written(self) -> int:
        """Number of distinct passing clips written to the manifest so far."""
        return len(self._seen)

    def __enter__(self) -> "FilterManifest":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _read_seen_keys(manifest_path: Path) -> set[str]:
    """Recover the ``audio_filename`` keys already written to ``manifest_path``."""
    if not manifest_path.exists():
        return set()
    seen: set[str] = set()
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["audio_filename"])
    return seen


def _read_clips_processed(checkpoint_path: Path) -> int:
    """Read ``clips_processed`` from the checkpoint, or 0 if there is none."""
    if not checkpoint_path.exists():
        return 0
    with open(checkpoint_path, encoding="utf-8") as f:
        return int(json.load(f)["clips_processed"])


def _write_clips_processed(checkpoint_path: Path, clips_processed: int) -> None:
    """Atomically write the progress checkpoint (temp file + ``os.replace``)."""
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"clips_processed": clips_processed}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, checkpoint_path)
