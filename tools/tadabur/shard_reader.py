"""Read the full-config Tadabur parquet shards directly, bypassing ``datasets`` streaming.

The Tadabur filter needs the whole corpus, but the only config ``datasets`` streams
cleanly — ``preview`` — is a fixed **300-row** sample (verified), far too small to fill
the P3.5 poison-audit's per-contrast buckets (#6 wants ~30 admitted clips per soft pair).
The full ``default`` config is 385 shards of ~1000 rows each, but it embeds ~2.4 GB of WAV
bytes per shard, so every attempt to read a whole shard at once — ``datasets``
``streaming=True`` or a raw ``pyarrow`` ``read_row_group`` — trips

    pyarrow.lib.ArrowNotImplementedError:
        Nested data conversions not implemented for chunked array outputs

because the ``audio`` struct's binary child overflows pyarrow's 2 GB per-chunk limit and
pyarrow cannot materialize a struct over a *chunked* binary array. Reading each shard in
**small Arrow batches** (:data:`DEFAULT_BATCH_SIZE` rows) keeps every batch's audio binary
well under that limit, so the struct materializes and the full corpus becomes iterable.

Each yielded row is a plain ``dict`` shaped exactly like a ``datasets`` streaming row
(``audio`` = ``{"bytes", "path"}`` plus the top-level ``surah_id`` / ``ayah_id`` /
``reciter_id`` ints), so :func:`tadabur.filter.parse_clip` and
:func:`tadabur.dataset_source.resolve_audio_filename` consume it unchanged. This module is
torch-free; it pulls in ``pyarrow`` and ``huggingface_hub`` only inside the iterator so the
module imports cheaply (and tests can stub the download).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from .dataset_source import DATASET_ID

# The full ``default`` config ships 385 shards named ``data/train-{i:05d}.parquet``,
# each a single ~1000-row row group. Rows-per-shard is fixed, so a filter resuming a
# shard run can recover its position as ``clips_processed // ROWS_PER_SHARD`` (see
# :mod:`tadabur.filter`): whole shards are processed in order, so that division is exact.
SHARD_TEMPLATE = "data/train-{index:05d}.parquet"
NUM_SHARDS = 385
ROWS_PER_SHARD = 1000

# Small enough that a batch's embedded WAV bytes stay far below pyarrow's 2 GB chunk
# limit (the whole 1000-row shard is ~2.4 GB, so ~38 MB per 16-row batch).
DEFAULT_BATCH_SIZE = 16

# Only the columns the filter and the clip-staging pass actually need. Skipping the
# ``metadata`` / text columns keeps each batch lean; the audio struct is the heavy one.
NEEDED_COLUMNS = ["audio", "surah_id", "ayah_id", "reciter_id"]


def parse_shard_spec(spec: str) -> list[int]:
    """Parse a shard spec into a sorted, de-duplicated list of shard indices.

    Accepts comma-separated terms, each either a single index (``"7"``) or an inclusive
    range (``"0-9"``). ``"0-4,10,20-21"`` → ``[0, 1, 2, 3, 4, 10, 20, 21]``. Every index
    must be in ``[0, NUM_SHARDS)``; anything else (empty, malformed, reversed range,
    out of bounds) raises ``ValueError`` so a typo cannot silently narrow the corpus.
    """
    indices: set[int] = set()
    terms = [term.strip() for term in spec.split(",") if term.strip()]
    if not terms:
        raise ValueError(f"empty shard spec: {spec!r}")
    for term in terms:
        if "-" in term:
            lo_str, _, hi_str = term.partition("-")
            lo, hi = _shard_index(lo_str, spec), _shard_index(hi_str, spec)
            if hi < lo:
                raise ValueError(f"reversed shard range {term!r} in spec {spec!r}")
            indices.update(range(lo, hi + 1))
        else:
            indices.add(_shard_index(term, spec))
    return sorted(indices)


def _shard_index(text: str, spec: str) -> int:
    try:
        index = int(text)
    except ValueError as exc:
        raise ValueError(f"non-integer shard {text!r} in spec {spec!r}") from exc
    if not 0 <= index < NUM_SHARDS:
        raise ValueError(
            f"shard index {index} out of range [0, {NUM_SHARDS}) in spec {spec!r}"
        )
    return index


def iter_shard_rows(
    shard_indices: Iterable[int],
    *,
    dataset_id: str = DATASET_ID,
    cache_dir: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    columns: list[str] | None = None,
    delete_after: bool = False,
) -> Iterator[dict]:
    """Yield Tadabur rows from the given full-config parquet ``shard_indices``, in order.

    Each shard is fetched with ``hf_hub_download`` (reusing the HF cache) and iterated in
    ``batch_size``-row Arrow batches to dodge the 2 GB nested-chunk limit (see the module
    docstring). Rows are ``datasets``-shaped dicts, ready for
    :func:`tadabur.filter.parse_clip`. With ``delete_after`` the shard's cached blob is
    removed once consumed, so a long run over many 2.4 GB shards stays within a bounded
    disk budget; leave it off to keep shards cached across resumes.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for index in shard_indices:
        path = hf_hub_download(
            dataset_id,
            SHARD_TEMPLATE.format(index=index),
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        try:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(
                batch_size=batch_size, columns=columns or NEEDED_COLUMNS
            ):
                yield from batch.to_pylist()
        finally:
            if delete_after:
                _remove_shard_blob(path)


def _remove_shard_blob(path: str) -> None:
    """Best-effort delete of a downloaded shard, following the HF cache symlink to its blob.

    ``hf_hub_download`` returns a ``snapshots/`` symlink into a content-addressed
    ``blobs/`` file; unlinking only the symlink would leave the 2.4 GB blob on disk, so we
    remove the resolved target too. Failures are swallowed — freeing disk is an
    optimization, not a correctness requirement.
    """
    try:
        real = os.path.realpath(path)
        if os.path.islink(path):
            os.unlink(path)
        if os.path.exists(real):
            os.remove(real)
    except OSError:
        pass
