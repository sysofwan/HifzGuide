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
    """8 s sweeping most of the speech band, 250 Hz to 3.2 kHz: slow enough for the codec to
    code it faithfully, and not one of its calibration probes."""
    from scipy.signal import chirp

    t = np.arange(8 * SR) / SR
    return (0.3 * chirp(t, 250, 8, 3200)).astype(np.float32)


def _pool() -> tuple:
    return tuple(pt.NoiseRecording(i, "CC0-1.0", "u", np.random.default_rng(i)
                                   .standard_normal(2 * SR).astype(np.float32)) for i in (3, 9))


@pytest.mark.parametrize("probe", sorted(pt.verification_probes()))
@pytest.mark.parametrize("name", [c.name for c in pt.GRID])
def test_every_condition_keeps_the_clips_timing_on_every_probe(name, probe):
    condition = pt.CONDITIONS[name]
    if condition.kind == pt.CODEC and not HAS_FFMPEG:
        pytest.skip("ffmpeg is not installed")
    clean = pt.verification_probes()[probe]
    perturbed, _ = pt.perturb(clean, condition, "c", _pool())
    assert len(perturbed) == len(clean)
    assert abs(pt.residual_lag(perturbed, probe, clean)) <= pt.MAX_RESIDUAL_LAG


def test_the_lag_is_measured_to_a_sub_sample_and_undone():
    chirp = _chirp()
    late = pt.advance(chirp, -1.5)  # 1.5 samples late
    assert pt.lag_samples(late, chirp, 16) == pytest.approx(1.5, abs=0.05)
    assert abs(pt.lag_samples(pt.advance(late, 1.5), chirp, 16)) < 0.01
    with pytest.raises(ValueError):  # 40 samples late: no peak inside a 16-sample window
        pt.lag_samples(np.roll(_impulse(), 40), _impulse(), 16)


def test_a_lag_without_a_clear_peak_is_refused_never_guessed():
    noise = np.random.default_rng(7).standard_normal(48_000).astype(np.float32)
    with pytest.raises(ValueError):  # white noise 100 samples late: no peak in the window
        pt.lag_samples(np.roll(noise, 100), noise, 16)
    quiet = (1e-6 * noise).astype(np.float32)
    unrelated = np.random.default_rng(8).standard_normal(48_000).astype(np.float32)
    with pytest.raises(ValueError):  # near-silence against unrelated audio
        pt.lag_samples(unrelated, quiet, 16)
    with pytest.raises(ValueError):  # digital silence
        pt.lag_samples(noise, np.zeros_like(noise), 16)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")
def test_the_codec_lag_is_one_calibrated_constant_whatever_the_clip():
    from scipy.signal import butter, sosfilt

    lag = pt.codec_lag(pt.GRID_OPUS)
    assert 0 < abs(lag) < pt.MAX_CODEC_LAG  # the codec does delay
    # Above the narrowband cutoff, the codec leaves nothing to measure a lag on: refused, and
    # the clip is still shifted by the calibrated constant, never by its own content.
    sos = butter(8, 4500, "highpass", fs=SR, output="sos")
    high = sosfilt(sos, np.random.default_rng(0).standard_normal(48_000)).astype(np.float32)
    with pytest.raises(ValueError):
        pt.lag_samples(pt._opus(high, pt.GRID_OPUS), high, 16)
    assert pt.opus_round_trip(high)[1] == lag == pt.opus_round_trip(_tone(500))[1]


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")
def test_each_codec_configuration_gets_its_own_calibration():
    audio = pt.OpusConfig(16_000, "audio")
    voip = pt.OpusConfig(16_000)
    assert pt.codec_lag(audio) != pt.codec_lag(voip)
    assert {audio, voip} <= set(pt._CODEC_LAGS)
    assert pt.opus_round_trip(_tone(500), audio)[1] == pt.codec_lag(audio)


@pytest.fixture(scope="module")
def residual_table() -> dict:
    return pt.codec_residuals()


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")
def test_every_training_configuration_keeps_timing(residual_table):
    """Portable: holds on any ffmpeg/libopus build."""
    assert set(residual_table) == {c.label for c in pt.OPUS_CONFIGS}
    for config in residual_table.values():
        assert 0 < config["calibrated_lag"] < pt.MAX_CODEC_LAG
        assert all(abs(v) <= pt.MAX_RESIDUAL_LAG for v in config["residual"].values())


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")
def test_the_committed_table_holds_within_a_toolchains_drift(residual_table):
    """The committed table (ffmpeg 8.1.2, libopus 1.6.1) against this build. The contract is
    the portable test above (every residual within 2 samples); this one only guards against
    gross drift from the committed table. The codec is not pinned, and another build moves the
    calibrated lags by a few thousandths of a sample and a probe's lag by up to ~0.06 (ffmpeg
    7.1), so calibrated lags agree within 0.05 samples and raw and residual lags within 0.25."""
    import json

    from training.nonpro_gap import CODEC_RESIDUALS_PATH

    committed = json.loads(CODEC_RESIDUALS_PATH.read_text(encoding="utf-8"))["configs"]
    assert set(committed) == set(residual_table)
    for label, entry in committed.items():
        fresh = residual_table[label]
        assert fresh["calibrated_lag"] == pytest.approx(entry["calibrated_lag"], abs=0.05)
        for kind in ("raw", "residual"):
            assert fresh[kind].keys() == entry[kind].keys()
            for probe, value in entry[kind].items():
                assert fresh[kind][probe] == pytest.approx(value, abs=0.25), (label, kind, probe)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")
def test_the_opus_round_trip_keeps_length_and_reports_the_lag_it_removed():
    clean = _tone(500, 2.0)
    coded, lag = pt.opus_round_trip(clean)
    assert len(coded) == len(clean) and coded.dtype == np.float32
    middle = slice(SR // 2, 3 * SR // 2)
    assert np.corrcoef(clean[middle], coded[middle])[0, 1] > 0.9
    again, lag_again = pt.opus_round_trip(clean)
    assert np.array_equal(coded, again) and lag == lag_again == pt.codec_lag(pt.GRID_OPUS)
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
