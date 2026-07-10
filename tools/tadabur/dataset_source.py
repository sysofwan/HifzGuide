"""Torch-free identifiers and helpers for streaming the Tadabur source dataset.

The Tadabur constants and the ``audio_filename`` resolver are needed by both the
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
