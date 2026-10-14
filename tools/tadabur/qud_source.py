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


#: Waits (s) before each retry when the viewer is rate-limiting or briefly failing
#: (:data:`RETRIED`); a 501 means it cannot serve that mushaf at all and is not retried.
VIEWER_RETRY_WAITS = (5, 15, 45, 120, 300)
RETRIED = frozenset({429, 500, 502, 503, 504})


def _viewer_get(url: str) -> bytes:
    """``url``'s body, authenticated with the local Hub token when there is one, retried
    after :data:`VIEWER_RETRY_WAITS` while the viewer is rate-limiting or failing."""
    import time
    import urllib.error
    import urllib.request

    from huggingface_hub import get_token

    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token and "datasets-server" in url \
        and "cached-assets" not in url else {}
    for wait in (*VIEWER_RETRY_WAITS, None):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if wait is None or error.code not in RETRIED:
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")


def cached_row(cache_dir: Path, slug: str, row_index: int) -> ViewerRow | None:
    """A row kept by :func:`keep_row`, or ``None``."""
    import json

    meta = cache_dir / f"{slug}.{row_index}.json"
    if not meta.exists():
        return None
    record = json.loads(meta.read_text(encoding="utf-8"))
    mp3 = (cache_dir / f"{slug}.{row_index}.mp3").read_bytes()
    return ViewerRow(row_of(slug, {**record["row"], "audio": {"bytes": mp3}}), record["revision"])


def keep_row(cache_dir: Path, row_index: int, got: ViewerRow) -> ViewerRow:
    """Keep a fetched row (one JSON and one MP3) so a rerun asks again for nothing."""
    import json

    cache_dir.mkdir(parents=True, exist_ok=True)
    row = got.row
    (cache_dir / f"{row.slug}.{row_index}.mp3").write_bytes(row.mp3)
    record = {"surah": row.surah, "ayah": row.ayah, "duration_ms": row.duration_ms,
              "text_uthmani": row.text_uthmani, "segments": [list(x) for x in row.segments],
              "word_timestamps": [list(x) for x in row.word_timestamps],
              "source_url": row.source_url, "source_offset_ms": row.source_offset_ms}
    (cache_dir / f"{row.slug}.{row_index}.json").write_text(
        json.dumps({"row": record, "revision": got.revision}, ensure_ascii=False),
        encoding="utf-8")
    return got


def viewer_row(slug: str, row_index: int) -> ViewerRow:
    """Row ``row_index`` of a mushaf config with its MP3, from the dataset viewer."""
    import json
    import urllib.parse

    query = urllib.parse.urlencode({"dataset": QUD_DATASET, "config": slug, "split": "train",
                                    "offset": row_index, "length": 1})
    (entry,) = json.loads(_viewer_get(f"{ROWS_API}?{query}"))["rows"]
    record = entry["row"]
    (audio,) = record.pop("audio")
    mp3 = _viewer_get(audio["src"])
    revision = _ROWS_API_REVISION.search(audio["src"]).group(1)
    return ViewerRow(row_of(slug, {**record, "audio": {"bytes": mp3}}), revision)


def pinned_row(slug: str, shards: Sequence[str], row_index: int,
               revision: str = QUD_REVISION) -> ViewerRow:
    """Row ``row_index`` of a mushaf at the pinned revision, its MP3 included: only the row
    group that holds it is read (~150 MB). For the rows the dataset viewer cannot serve."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    for shard in shards:
        # Large blocks: a row group's audio is one contiguous range of ~150 MB.
        with fs.open(f"datasets/{QUD_DATASET}@{revision}/{shard}", block_size=32 << 20) as f:
            parquet = pq.ParquetFile(f)
            if row_index >= parquet.metadata.num_rows:
                row_index -= parquet.metadata.num_rows
                continue
            for group in range(parquet.metadata.num_row_groups):
                size = parquet.metadata.row_group(group).num_rows
                if row_index < size:
                    records = parquet.read_row_group(
                        group, columns=["audio", *METADATA_COLUMNS]).slice(row_index, 1)
                    return ViewerRow(row_of(slug, records.to_pylist()[0]), revision)
                row_index -= size
    raise IndexError(f"{slug} has no row {row_index}")
