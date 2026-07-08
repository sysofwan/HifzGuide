"""Decode streamed Tadabur audio to 16 kHz mono — the model's expected input.

Tadabur clips arrive as raw WAV bytes when the ``datasets`` audio feature is read
with ``decode=False`` (which also avoids the ``torchcodec`` runtime dependency that
``datasets``' built-in decoder pulls in). This module decodes those bytes with
``soundfile`` and resamples/downmixes to 16 kHz mono with ``librosa`` so the
waveform matches what the ``SeamlessM4TFeatureExtractor`` — and thus the model —
expects. Keeping this as the single loader means the Phase 3 filter reuses the exact
same 16 kHz-mono preprocessing as this smoke test.
"""

from __future__ import annotations

import io

import librosa
import numpy as np
import soundfile as sf

# The Muaalem feature extractor / model operate at 16 kHz; resample everything here.
TARGET_SAMPLE_RATE = 16000


def decode_to_mono_16k(raw_audio: bytes) -> np.ndarray:
    """Decode WAV ``raw_audio`` to a 16 kHz mono float32 waveform.

    Downmixes multi-channel audio by averaging channels and resamples to
    ``TARGET_SAMPLE_RATE`` only when the source rate differs, so already-16 kHz mono
    clips pass through unresampled.
    """
    waveform, sample_rate = sf.read(io.BytesIO(raw_audio), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
        )
    return np.ascontiguousarray(waveform, dtype=np.float32)
