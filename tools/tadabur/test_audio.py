"""Tests for the Tadabur audio loader (decode + 16 kHz-mono resampling)."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from tadabur.audio import TARGET_SAMPLE_RATE, decode_to_mono_16k


def _wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def test_resamples_to_16k_and_downmixes_to_mono():
    # 0.5 s stereo tone at 8 kHz -> expect mono, 16 kHz, ~0.5 s of samples.
    src_rate = 8000
    n = src_rate // 2
    t = np.linspace(0, 0.5, n, endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)

    waveform = decode_to_mono_16k(_wav_bytes(stereo, src_rate))

    assert waveform.ndim == 1
    assert waveform.dtype == np.float32
    assert abs(len(waveform) - TARGET_SAMPLE_RATE // 2) <= 2


def test_passes_through_already_16k_mono():
    n = TARGET_SAMPLE_RATE // 4
    tone = 0.1 * np.sin(np.linspace(0, 3.0, n, endpoint=False)).astype(np.float32)

    waveform = decode_to_mono_16k(_wav_bytes(tone, TARGET_SAMPLE_RATE))

    assert waveform.ndim == 1
    assert len(waveform) == n
    np.testing.assert_allclose(waveform, tone, atol=1e-6)
