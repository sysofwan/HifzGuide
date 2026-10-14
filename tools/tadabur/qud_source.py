"""Reading QUD (`QUD-Technologies/quranic-universal-ayahs`) at the pinned revision (#144).

QUD publishes one parquet config per mushaf, one row per ayah: the ayah's MP3 (``audio``),
its recited DigitalKhatt V2 text (``text_uthmani``, repeated words kept), waqf-aware
``segments`` (``[word_from, word_to, start_ms, end_ms]``, 1-based words of that text, ms of the
ayah clip), ``word_timestamps`` (``[word, start_ms, end_ms]``) and the source mapping. This
module is the one place the tools read it:

* :func:`parquet_table`: a Hub parquet file's chosen columns only (footer and those column
  chunks by HTTP range requests), for metadata;
* :func:`iter_mushaf_rows`: a mushaf's rows with their MP3 bytes, one shard downloaded at a
  time into a cache directory and deleted once read (the bulk audio is never kept);
* :func:`decode_ayah`: an ayah's MP3 decoded to the model's 16 kHz mono
  (:func:`tadabur.audio.decode_to_mono_16k`), and the segment spans in its samples.

An ayah clip is named :func:`audio_filename` and identified by the SHA-256 of its MP3 bytes
(:func:`mp3_sha256`), the bytes QUD publishes at :data:`QUD_REVISION`; a truth site's sample
span indexes the 16 kHz decode of those bytes.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import TARGET_SAMPLE_RATE, decode_to_mono_16k

QUD_DATASET = "QUD-Technologies/quranic-universal-ayahs"
#: The revision #144's selection was made at (``qud_eval/selection.json``); QUD republishes
#: often, so every read is pinned to it.
QUD_REVISION = "6f45ced88a179dfbb97d5b0f76ae69b4d4d1f175"
#: Every column of a mushaf config but the audio.
METADATA_COLUMNS = ["surah", "ayah", "duration_ms", "text_uthmani", "segments",
                    "word_timestamps", "source_url", "source_offset_ms"]
#: A mushaf config's shard: ``<riwayah dir>/<slug>-00000-of-00004.parquet``.
SHARD_NAME = re.compile(r"(.+)-\d{5}-of-\d{5}\.parquet")
#: Rows per Arrow batch when the audio is read (an ayah's MP3 is up to a few MB).
BATCH_ROWS = 16


def parquet_table(repo: str, revision: str, path: str, columns: list[str] | None = None):
    """A parquet file of a Hub dataset at a pinned revision, reading only ``columns``
    (the footer and those column chunks, by HTTP range requests)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"datasets/{repo}@{revision}/{path}", block_size=1 << 16) as f:
        return pq.ParquetFile(f).read(columns=columns)


def audio_filename(slug: str, surah: int, ayah: int) -> str:
    """The name an ayah clip of a mushaf goes by in truth sites and caches."""
    return f"qud_{slug}_{surah:03d}_{ayah:03d}.mp3"


def mp3_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class QudRow:
    """One ayah of one mushaf: its metadata and, when read with audio, its MP3 bytes."""

    slug: str
    surah: int
    ayah: int
    duration_ms: int
    text_uthmani: str
    segments: tuple[tuple[int, int, int, int], ...]
    word_timestamps: tuple[tuple[int, int, int], ...]
    source_url: str
    source_offset_ms: int
    mp3: bytes = b""

    @property
    def surah_ayah(self) -> str:
        return f"{self.surah}:{self.ayah}"

    @property
    def audio_filename(self) -> str:
        return audio_filename(self.slug, self.surah, self.ayah)


def row_of(slug: str, record: dict) -> QudRow:
    """A :class:`QudRow` from a parquet record (with or without its ``audio``)."""
    audio = record.get("audio") or {}
    return QudRow(
        slug=slug, surah=record["surah"], ayah=record["ayah"],
        duration_ms=record["duration_ms"], text_uthmani=record["text_uthmani"],
        segments=tuple(tuple(s) for s in record["segments"]),
        word_timestamps=tuple(tuple(w) for w in record["word_timestamps"]),
        source_url=record["source_url"], source_offset_ms=record["source_offset_ms"],
        mp3=audio.get("bytes") or b"",
    )


def mushaf_metadata(slug: str, shards: Sequence[str],
                    revision: str = QUD_REVISION) -> list[QudRow]:
    """Every row of a mushaf without audio, in shard order (column reads only)."""
    return [row_of(slug, record)
            for shard in shards
            for record in parquet_table(QUD_DATASET, revision, shard,
                                        METADATA_COLUMNS).to_pylist()]


def iter_mushaf_rows(slug: str, shards: Sequence[str], cache_dir: Path,
                     revision: str = QUD_REVISION) -> Iterator[QudRow]:
    """A mushaf's rows with their MP3 bytes, shard by shard: each shard is downloaded into
    ``cache_dir`` at ``revision`` and its cached blob deleted once its rows are read."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from .shard_reader import remove_shard_blob

    for shard in shards:
        path = hf_hub_download(QUD_DATASET, shard, repo_type="dataset", revision=revision,
                               cache_dir=str(cache_dir))
        try:
            for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=BATCH_ROWS, columns=["audio", *METADATA_COLUMNS]):
                yield from (row_of(slug, record) for record in batch.to_pylist())
        finally:
            remove_shard_blob(path)


def decode_ayah(row: QudRow) -> np.ndarray:
    """The ayah's MP3 as the model's 16 kHz mono float32 waveform."""
    if not row.mp3:
        raise ValueError(f"{row.slug} {row.surah_ayah} was read without its audio")
    return decode_to_mono_16k(row.mp3)


def segment_span(segment: Sequence[int], n_samples: int) -> tuple[int, int]:
    """A QUD segment's ``[start_ms, end_ms]`` as a sample span of the 16 kHz decode, clamped
    to the decode's length."""
    _, _, start_ms, end_ms = segment
    start = min(round(start_ms * TARGET_SAMPLE_RATE / 1000), n_samples)
    end = min(round(end_ms * TARGET_SAMPLE_RATE / 1000), n_samples)
    if end <= start:
        raise ValueError(f"segment {list(segment)} is empty in a {n_samples}-sample decode")
    return start, end


#: The Hub's dataset viewer serves one row at a time, audio included: what a few control
#: ayat cost, instead of a whole shard. It serves its own latest conversion of the dataset,
#: which may be a later revision than :data:`QUD_REVISION`.
ROWS_API = "https://datasets-server.huggingface.co/rows"
_ROWS_API_REVISION = re.compile(r"/--/([0-9a-f]{40})/--/")


@dataclass(frozen=True)
class ViewerRow:
    """One row as the dataset viewer serves it: the row, its MP3 and the revision served."""

    row: QudRow
    revision: str


def viewer_row(slug: str, row_index: int) -> ViewerRow:
    """Row ``row_index`` of a mushaf config with its MP3, from the dataset viewer."""
    import json
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode({"dataset": QUD_DATASET, "config": slug, "split": "train",
                                    "offset": row_index, "length": 1})
    with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=120) as response:
        (entry,) = json.loads(response.read())["rows"]
    record = entry["row"]
    (audio,) = record.pop("audio")
    with urllib.request.urlopen(audio["src"], timeout=120) as response:
        mp3 = response.read()
    return ViewerRow(row_of(slug, {**record, "audio": {"bytes": mp3}}),
                     _ROWS_API_REVISION.search(audio["src"]).group(1))
