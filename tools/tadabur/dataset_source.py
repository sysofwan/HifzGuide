"""Torch-free identifiers and helpers for streaming the Tadabur source dataset.

The Tadabur constants, the ``audio_filename`` resolver and the ``surah:ayah`` key are needed by both the
GPU filtering path (``tadabur.filter``) and the no-model offline stages
(``tadabur.waqf_segments``, ``tadabur.audit_sampler``). Keeping them here — a
module that imports nothing heavier than the standard library — lets the offline
stages stream rows without pulling in ``tadabur.inference`` (torch/transformers)
merely to name a clip. ``tadabur.filter`` re-exports these so its public surface
is unchanged.
"""

from __future__ import annotations

from pathlib import Path

DATASET_ID = "FaisaI/tadabur"
AUDIO_COLUMN = "audio"


def resolve_audio_filename(row: dict) -> str:
    """The clip's stable ``audio_filename`` across Tadabur configs.

    The full ``default`` config carries ``audio_filename`` as a top-level column;
    the fast ``preview`` config does not, but its ``audio`` feature still exposes
    the same basename via ``path`` (e.g. ``tadabur_spk0106_S77_A30_...wav``). Fall
    back to that so ``--config-name preview`` — advertised as the fast streaming
    path — actually yields traceable, exportable clips. Fails loudly if neither is
    present, since a clip with no stable id cannot be matched back to its audio.
    """
    name = row.get("audio_filename")
    if name:
        return name
    audio = row.get(AUDIO_COLUMN) or {}
    path = audio.get("path")
    if path:
        return Path(path).name
    raise ValueError(f"Tadabur row has no audio_filename or audio.path: {row!r}")


def stream_rows(dataset_id: str, config_name: str, split: str):
    """Tadabur rows from the ``datasets`` stream, audio as raw bytes (``decode=False``),
    each through the sealed panel's row seal (:func:`tadabur.panel_seal.seal_row`). The
    one way the tools stream the dataset; the full-config shard reader is
    :func:`tadabur.shard_reader.iter_shard_rows`."""
    from datasets import Audio, load_dataset

    from .panel_seal import seal_rows

    dataset = load_dataset(dataset_id, name=config_name, split=split, streaming=True)
    return seal_rows(iter(dataset.cast_column(AUDIO_COLUMN, Audio(decode=False))))


def canonical_surah_ayah(surah_id: int, ayah_id: int) -> str:
    """Map a Tadabur ``(surah_id, ayah_id)`` to a canonical ``"surah:ayah"`` key.

    Tadabur numbers ``surah_id`` **0-indexed** (0–113, a surah *array index*, and
    the same 0-based number embedded in the audio filename, e.g. ``S77`` for
    Al-Naba, the 78th surah) while ``ayah_id`` is the natural **1-indexed** ayah
    number. Our reference cache is keyed by the canonical 1-indexed
    ``surah:ayah`` (``quran-transcript``), so we shift the surah by one here.
    Without this shift every clip gates against the wrong ayah and *nothing*
    passes the filter.
    """
    return f"{surah_id + 1}:{ayah_id}"
