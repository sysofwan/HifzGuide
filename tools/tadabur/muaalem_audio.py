"""Reading a little of Muaalem's training audio (``obadx/muaalem-annotated-v3``), for #144.

Muaalem's moshafs are parquet configs ``moshaf_<id>/train-*.parquet`` of 100-row groups,
each row a segment of a surah's recitation (16 kHz audio, its ``start_span`` / ``end_span``
ayah range). #144's overlap check needs only a few segments per moshaf, so this reads one row
group at a time by HTTP range requests (:func:`row_group_segments`), never a whole file.
"""

from __future__ import annotations

from dataclasses import dataclass

MUAALEM_DATASET = "obadx/muaalem-annotated-v3"
#: The revision #144's selection recorded (``qud_eval/sources.json``).
MUAALEM_REVISION = "8e3fe50ec5efcc4beab054acdbd0178bc690c246"
_COLUMNS = ["audio", "segment_index", "start_span", "end_span", "duration_seconds"]


@dataclass(frozen=True)
class MuaalemSegment:
    """One segment: its moshaf, index, ayah range (``None`` for a segment with no Quran, such
    as the isti'adha) and audio bytes."""

    moshaf: str
    segment_index: str
    first: tuple[int, int] | None
    last: tuple[int, int] | None
    audio: bytes


def first_shard(moshaf: str, revision: str = MUAALEM_REVISION) -> str:
    """The path of a moshaf config's first parquet file."""
    from huggingface_hub import HfApi

    files = sorted(f.path for f in HfApi().list_repo_tree(
        MUAALEM_DATASET, path_in_repo=f"moshaf_{moshaf}", repo_type="dataset",
        revision=revision) if f.path.endswith(".parquet"))
    return files[0]


def row_group_segments(moshaf: str, path: str, row_group: int,
                       revision: str = MUAALEM_REVISION) -> list[MuaalemSegment]:
    """One row group of a moshaf's parquet file, audio included."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    def span(value: dict | None) -> tuple[int, int] | None:
        return None if value is None else (value["sura_idx"], value["aya_idx"])

    with HfFileSystem().open(f"datasets/{MUAALEM_DATASET}@{revision}/{path}",
                             block_size=1 << 16) as f:
        records = pq.ParquetFile(f).read_row_group(row_group, columns=_COLUMNS).to_pylist()
    return [MuaalemSegment(moshaf, r["segment_index"], span(r["start_span"]),
                           span(r["end_span"]), r["audio"]["bytes"]) for r in records]
