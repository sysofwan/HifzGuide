"""Decode streamed Tadabur audio to 16 kHz mono — the model's expected input.

Tadabur clips arrive as raw WAV bytes when the ``datasets`` audio feature is read
with ``decode=False`` (which also avoids the ``torchcodec`` runtime dependency that
``datasets``' built-in decoder pulls in). This module decodes those bytes with
``soundfile`` and resamples/downmixes to 16 kHz mono with ``librosa`` so the
waveform matches what the ``SeamlessM4TFeatureExtractor`` — and thus the model —
expects. Keeping this as the single loader means the Phase 3 filter reuses the exact
same 16 kHz-mono preprocessing as this smoke test.

It is also where the sealed panel's seal sits for audio (:mod:`tadabur.panel_seal`): every
audio file the tools read goes through :func:`read_audio` or :func:`read_audio_bytes`,
and every byte buffer through :func:`decode_to_mono_16k`, each of which refuses panel
audio outside an authorized block.
"""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .panel_seal import refuse_sealed, refuse_sealed_bytes

# The Muaalem feature extractor / model operate at 16 kHz; resample everything here.
TARGET_SAMPLE_RATE = 16000


def read_audio(path: Path, **kwargs):
    """``soundfile.read`` of an audio file, after the panel seal's check. The tools read
    audio files only through here and :func:`read_audio_bytes`."""
    return sf.read(str(refuse_sealed(Path(path))), **kwargs)


def read_audio_bytes(path: Path) -> bytes:
    """An audio file's raw bytes, after the panel seal's check (for
    :func:`decode_to_mono_16k`)."""
    return refuse_sealed(Path(path)).read_bytes()


def decode_to_mono_16k(raw_audio: bytes) -> np.ndarray:
    """Decode WAV ``raw_audio`` to a 16 kHz mono float32 waveform.

    Downmixes multi-channel audio by averaging channels and resamples to
    ``TARGET_SAMPLE_RATE`` only when the source rate differs, so already-16 kHz mono
    clips pass through unresampled.
    """
    waveform, sample_rate = sf.read(io.BytesIO(refuse_sealed_bytes(raw_audio)), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
        )
    return np.ascontiguousarray(waveform, dtype=np.float32)
