"""Stretching one span of a waveform in place (torch-free, synthetic waveforms)."""

from __future__ import annotations

import numpy as np
import pytest

from tadabur.time_stretch import FRAME, added_samples, stretch_span

RATE = 16000


def _tone(seconds: float, hz: float, seed: int = 0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    noise = np.random.default_rng(seed).normal(scale=0.01, size=len(t))
    return (0.5 * np.sin(2 * np.pi * hz * t) + noise).astype(np.float32)


def _dominant_hz(x: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / RATE)[spectrum.argmax()])


@pytest.mark.parametrize("factor", [1.25, 1.5, 2.0])
def test_only_the_span_grows_and_the_rest_is_bit_identical(factor):
    x = np.random.default_rng(1).normal(size=RATE).astype(np.float32)
    start, end = 6000, 9200
    y = stretch_span(x, start, end, factor)

    added = added_samples(start, end, factor)
    assert len(y) == len(x) + added
    assert added == round((factor - 1) * (end - start))
    before, after = start - 2 * FRAME, end + 2 * FRAME
    np.testing.assert_allclose(y[:before], x[:before], atol=1e-6)
    np.testing.assert_allclose(y[after + added :], x[after:], atol=1e-6)


def test_factor_one_reproduces_the_input():
    x = _tone(0.5, 180)
    np.testing.assert_allclose(stretch_span(x, 2000, 5000, 1.0), x, atol=1e-6)


@pytest.mark.parametrize("hz", [110.0, 200.0, 330.0])
def test_a_stretched_voiced_hold_keeps_its_pitch_and_level(hz):
    x = _tone(1.0, hz)
    start, end = 4000, 8000  # a 250 ms hold
    y = stretch_span(x, start, end, 1.5)
    held = y[start + FRAME : end + added_samples(start, end, 1.5) - FRAME]

    assert _dominant_hz(held) == pytest.approx(hz, abs=4)
    rms = lambda a: float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))  # noqa: E731
    assert rms(held) == pytest.approx(rms(x[start:end]), rel=0.1)


def test_bad_spans_and_factors_are_refused():
    x = np.zeros(1000, dtype=np.float32)
    with pytest.raises(ValueError, match="inside"):
        stretch_span(x, 500, 1001, 1.5)
    with pytest.raises(ValueError, match="inside"):
        stretch_span(x, 500, 500, 1.5)
    with pytest.raises(ValueError, match="factor"):
        stretch_span(x, 100, 500, 0.8)
