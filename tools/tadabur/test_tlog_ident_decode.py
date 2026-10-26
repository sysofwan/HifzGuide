"""The S5b confirmation audio path (#204): Common Voice clips are 48 kHz, not 16 kHz.

The pre-registration pins ``tadabur.audio:decode_to_mono_16k_with_rate_and_decoder`` for the
confirmation rows, the loader the S6/S7 tlog rows use, because the frozen sets' 16 kHz-only
``read_clip_audio`` refuses the published clips.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from tadabur.audio import TARGET_SAMPLE_RATE
from tadabur.tlog_ident_decode import confirm_cv_samples

SECONDS = 0.5


def _write(path, sample_rate: int) -> int:
    times = np.arange(int(sample_rate * SECONDS), dtype=np.float32) / sample_rate
    waveform = (0.1 * np.sin(2 * np.pi * 440 * times)).astype(np.float32)
    sf.write(path, waveform, sample_rate, subtype="FLOAT")
    return len(waveform)


def test_a_48khz_clip_is_resampled_to_16khz(tmp_path):
    path = tmp_path / "common_voice_ar_00000000.wav"
    source = _write(path, 48_000)
    samples = confirm_cv_samples(path)
    assert samples.dtype == np.float32 and samples.ndim == 1
    assert abs(len(samples) - int(TARGET_SAMPLE_RATE * SECONDS)) <= 1
    assert len(samples) < source


def test_a_16khz_clip_passes_through_unresampled(tmp_path):
    path = tmp_path / "common_voice_ar_00000001.wav"
    source = _write(path, TARGET_SAMPLE_RATE)
    assert len(confirm_cv_samples(path)) == source
