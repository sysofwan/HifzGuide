"""Tests for training.perturbation: #159's grid, each condition's arithmetic, the noise pool.

Torch-free, on synthetic audio; the codec test needs ``ffmpeg`` with ``libopus``.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from training import perturbation as pt

SR = pt.TARGET_SAMPLE_RATE


def _tone(hz: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _power(x) -> float:
    return float(np.mean(np.square(np.asarray(x, dtype=np.float64))))


def _band_power(x: np.ndarray, hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(x))
    return float(spectrum[int(round(hz * len(x) / SR))])


def test_the_grid_names_each_condition_once_and_serialises():
    assert [c.name for c in pt.GRID] == [
        "clean", "noise_snr20", "noise_snr10", "noise_snr5", "reverb_rt0.3", "reverb_rt0.6",
        "reverb_rt1", "band_8k", "opus_12k_nb"]
    record = pt.grid_record()
    assert json.loads(json.dumps(record)) == record
    assert record["noise"]["revision"] == pt.NOISE.revision


def test_noise_is_added_at_the_snr_asked_for():
    clean = _tone(440)
    noise = np.random.default_rng(0).standard_normal(len(clean)).astype(np.float32)
    for snr in pt.SNRS_DB:
        noisy = pt.add_noise(clean, noise, snr)
        assert noisy.dtype == np.float32 and len(noisy) == len(clean)
        measured = 10 * np.log10(_power(clean) / _power(noisy.astype(np.float64) - clean))
        assert measured == pytest.approx(snr, abs=0.01)
    with pytest.raises(ValueError):
        pt.add_noise(clean, noise[:-1], 10.0)
    with pytest.raises(ValueError):
        pt.add_noise(np.zeros_like(clean), noise, 10.0)


def test_a_clip_draws_one_noise_segment_whatever_the_snr():
    draws = {pt.noise_draw(530, f"clip{i}") for i in range(200)}
    assert all(0 <= index < 530 and 0 <= offset < 1 for index, offset in draws)
    assert len({index for index, _ in draws}) > 100
    assert pt.noise_draw(530, "clip7") == pt.noise_draw(530, "clip7")
    with pytest.raises(ValueError):
        pt.noise_draw(0, "clip7")


def test_a_noise_segment_starts_at_its_offset_and_loops():
    noise = np.arange(10, dtype=np.float32)
    assert pt.noise_segment(noise, 0.5, 12).tolist() == [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6]


def test_the_impulse_response_has_its_direct_path_tail_energy_and_decay():
    rt60 = 0.6
    rir = pt.synthetic_rir(rt60, np.random.default_rng(1))
    onset = int(round(pt.RIR_TAIL_ONSET_S * SR))
    assert len(rir) == int(rt60 * SR)
    assert rir[0] == 1.0 and not rir[1:onset].any()
    assert 10 * np.log10(1.0 / np.sum(np.square(rir[1:]))) == pytest.approx(pt.DRR_DB)
    tenth = len(rir) // 10
    first, last = np.sum(np.square(rir[onset:tenth])), np.sum(np.square(rir[-tenth:]))
    assert 10 * np.log10(first / last) > 40  # ~54 dB decay between the first and last tenth


def test_reverb_keeps_the_clips_length_and_level():
    clean = _tone(300)
    wet = pt.reverberate(clean, pt.synthetic_rir(1.0, np.random.default_rng(2)))
    assert len(wet) == len(clean) and wet.dtype == np.float32
    assert _power(wet) == pytest.approx(_power(clean), rel=1e-4)
    assert not np.allclose(wet, clean)


def test_band_limit_keeps_the_low_band_and_removes_the_high():
    clean = _tone(1000) + _tone(6000)
    narrow = pt.band_limit(clean)
    assert len(narrow) == len(clean) and narrow.dtype == np.float32
    assert _band_power(narrow, 1000) == pytest.approx(_band_power(clean, 1000), rel=0.02)
    assert _band_power(narrow, 6000) < 1e-3 * _band_power(clean, 6000)


HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _impulse() -> np.ndarray:
    """48,001 samples, a unit impulse at sample 16,000."""
    out = np.zeros(48_001, dtype=np.float32)
    out[16_000] = 1.0
    return out


def _chirp() -> np.ndarray:
    """3 s sweeping the speech band, 300 Hz to 3.4 kHz."""
    from scipy.signal import chirp

    t = np.arange(3 * SR) / SR
    return (0.3 * chirp(t, 300, 3, 3400)).astype(np.float32)


def _pool() -> tuple:
    return tuple(pt.NoiseRecording(i, "CC0-1.0", "u", np.random.default_rng(i)
                                   .standard_normal(2 * SR).astype(np.float32)) for i in (3, 9))


@pytest.mark.parametrize("name", [c.name for c in pt.GRID])
def test_every_condition_keeps_the_clips_timing(name):
    if pt.CONDITIONS[name].kind == pt.CODEC and not HAS_FFMPEG:
        pytest.skip("ffmpeg is not installed")
    condition = pt.CONDITIONS[name]
    impulse, _ = pt.perturb(_impulse(), condition, "c", _pool())
    assert len(impulse) == 48_001 and int(np.argmax(np.abs(impulse))) == 16_000
    chirp = _chirp()
    swept, _ = pt.perturb(chirp, condition, "c", _pool())
    assert abs(pt.lag_samples(swept, chirp, 32)) < 0.25


def test_the_lag_is_measured_to_a_sub_sample_and_undone():
    chirp = _chirp()
    late = pt.advance(chirp, -1.5)  # 1.5 samples late
    assert pt.lag_samples(late, chirp, 16) == pytest.approx(1.5, abs=0.05)
    assert abs(pt.lag_samples(pt.advance(late, 1.5), chirp, 16)) < 0.01
    assert pt.lag_samples(chirp, np.zeros_like(chirp), 16) == 0.0
    with pytest.raises(ValueError):  # 40 samples late: no peak inside a 16-sample window
        pt.lag_samples(np.roll(_impulse(), 40), _impulse(), 16)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")
def test_the_opus_round_trip_keeps_length_and_reports_the_lag_it_removed():
    clean = _tone(500, 2.0)
    coded, lag = pt.opus_round_trip(clean)
    assert len(coded) == len(clean) and coded.dtype == np.float32
    assert 0 < abs(lag) < pt.MAX_CODEC_LAG  # the codec does delay; the lag is undone
    middle = slice(SR // 2, 3 * SR // 2)
    assert np.corrcoef(clean[middle], coded[middle])[0, 1] > 0.9
    again, lag_again = pt.opus_round_trip(clean)
    assert np.array_equal(coded, again) and lag == lag_again
    _, draw = pt.perturb(clean, pt.CONDITIONS["opus_12k_nb"], "c")
    assert draw == {"codec_lag": round(lag, 3)}


def test_perturb_applies_each_kind_and_reports_the_noise_drawn():
    clean = _tone(440)
    pool = _pool()
    assert pt.perturb(clean, pt.CONDITIONS["clean"], "c") == (clean, {})
    noisy, draw = pt.perturb(clean, pt.CONDITIONS["noise_snr10"], "c", pool)
    index, offset = pt.noise_draw(2, "c")
    assert draw == {"freesound_id": pool[index].freesound_id, "noise_offset": round(offset, 6)}
    assert pt.perturb(clean, pt.CONDITIONS["noise_snr5"], "c", pool)[1] == draw
    wet, _ = pt.perturb(clean, pt.CONDITIONS["reverb_rt0.3"], "c")
    assert np.array_equal(wet, pt.perturb(clean, pt.CONDITIONS["reverb_rt0.3"], "c")[0])
    assert not np.array_equal(wet, pt.perturb(clean, pt.CONDITIONS["reverb_rt0.3"], "d")[0])
    with pytest.raises(ValueError):
        pt.perturb(clean, pt.Condition("x", "unknown"), "c")


def test_the_pool_keeps_long_finite_audible_recordings():
    noise = np.random.default_rng(0).standard_normal(SR).astype(np.float32)
    assert pt.pool_eligible(noise)
    assert not pt.pool_eligible(noise[: SR // 2])
    assert not pt.pool_eligible(np.zeros(SR, dtype=np.float32))
    assert not pt.pool_eligible(np.where(np.arange(SR) == 5, np.nan, noise).astype(np.float32))


def test_a_saved_pool_loads_only_against_its_own_record(tmp_path):
    pool = tuple(pt.NoiseRecording(i, "CC-BY-4.0", f"u{i}", np.full(SR + i, 0.1 * i, np.float32))
                 for i in (2, 5))
    path = tmp_path / "pool.npz"
    pt.save_pool(pool, path)
    records = [r.as_dict() for r in pool]
    loaded = pt.load_pool(path, records)
    assert [r.freesound_id for r in loaded] == [2, 5]
    assert all(np.array_equal(a.samples, b.samples) for a, b in zip(loaded, pool))
    records[1]["samples_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        pt.load_pool(path, records)
