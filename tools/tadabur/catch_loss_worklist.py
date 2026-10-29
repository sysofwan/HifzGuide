"""The round-2 catch-loss audit's worklist as the listening UI reads it (#225).

:func:`tadabur.catch_loss_audit.draw` writes ``worklist.jsonl``; the #61 UI serves it as its
whole-word question. This module is the file's shape and its two checks: the ayah text and the
word come from the Quran, and every clip is verified against the SHA-256 the worklist records
before the server starts. It imports nothing of the readout, so the UI can load it.

The page is told only what the row holds: an opaque id, the clip's provenance, the ayah, the
word index and the word's text. The stratum, either model's grade and the sampling weight
live in the audit's ``key.jsonl`` and never reach here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .truth_sites import audio_sha256
from .word_sample import WORD_MODE

#: What a worklist row holds, exactly.
WORKLIST_FIELDS = ("id", "split", "shard_file", "row", "audio_filename", "audio_sha256",
                   "start_sample", "end_sample", "surah_ayah", "word_index", "word")


@dataclass(frozen=True)
class CatchLossClip:
    """One worklist row's clip and its provenance, as the UI's word question needs it."""

    site_id: str
    surah_ayah: str
    reference: str
    audio_filename: str
    audio_sha256: str
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class CatchLossRow:
    """One worklist item as the #61 UI asks it: was the highlighted word recited correctly?

    ``site_id`` is the worklist's id, so the scorer joins the verdicts to the key. The
    reference is the whole ayah's words joined by a space, and the row's word is highlighted
    in it; the page is never told which stratum the item came from.
    """

    site: CatchLossClip
    word_index: int
    word_offsets: tuple[int, ...]

    @property
    def mode(self) -> str:
        return WORD_MODE

    @property
    def word_start(self) -> int:
        return 0  # the reference is the whole ayah

    @property
    def word_span(self) -> tuple[int, int]:
        return self.word_offsets[self.word_index], self.word_offsets[self.word_index + 1]


def _offsets(reference: str, words: Sequence[str]) -> tuple[int, ...]:
    """The word boundaries of ``reference`` (``" ".join(words)``): each start, then its end."""
    starts, position = [], 0
    for word in words:
        starts.append(position)
        position += len(word) + 1
    return tuple([*starts, len(reference)])


def read_catch_loss_worklist(
    path: Path, uthmani_words: Callable[[str], list[str]]
) -> list[CatchLossRow]:
    """The audit's worklist as the UI's rows, each row's ayah text and word checked."""
    rows: list[CatchLossRow] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        data = json.loads(raw)
        missing = [name for name in WORKLIST_FIELDS if name not in data]
        if missing:
            raise ValueError(f"{path}:{lineno}: no {missing} in a worklist row")
        words = uthmani_words(data["surah_ayah"])
        if not 0 <= data["word_index"] < len(words):
            raise ValueError(f"{path}:{lineno}: word {data['word_index']} is outside "
                             f"{data['surah_ayah']}")
        if data["word"] != words[data["word_index"]]:
            raise ValueError(f"{path}:{lineno}: the row's word is not "
                             f"{data['surah_ayah']}'s word {data['word_index']}")
        reference = " ".join(words)
        rows.append(CatchLossRow(
            CatchLossClip(data["id"], data["surah_ayah"], reference,
                          data["audio_filename"], data["audio_sha256"],
                          data["start_sample"], data["end_sample"]),
            data["word_index"], _offsets(reference, words)))
    if len({row.site.site_id for row in rows}) != len(rows):
        raise ValueError(f"{path}: two worklist rows share an id")
    return rows


def verify_clips(rows: Sequence[CatchLossRow], cache: Path) -> None:
    """Fail unless every row's clip is in ``cache`` under the SHA-256 the worklist records.

    The clips are the box's look cache (:func:`tadabur.tlog_eval_decode.clip_path`), held as
    their source bytes; the check runs before the server starts, so a damaged or substituted
    clip never reaches the owner's ear.
    """
    for digest in sorted({row.site.audio_sha256 for row in rows}):
        path = cache / f"{digest}.audio"
        if not path.is_file():
            raise ValueError(f"the look clip cache has no {path.name} for the worklist's "
                             f"{digest}")
        if audio_sha256(path) != digest:
            raise ValueError(f"{path}: its sha256 does not match the worklist's audio_sha256")
