"""Tests for training.augmentation: #94's mixture, its draws, its timing and its noise pool.

Torch-free, on synthetic audio; the codec cases need ``ffmpeg`` with ``libopus``.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter

import numpy as np
import pytest

from training import augmentation as aug
from training import perturbation as pt

SR = pt.TARGET_SAMPLE_RATE
HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _pool(*ids: int) -> tuple[pt.NoiseRecording, ...]:
    return tuple(pt.NoiseRecording(i, "CC0-1.0", "u", np.random.default_rng(i)
                                   .standard_normal(2 * SR).astype(np.float32))
                 for i in ids or (3, 9))


def _augmenter(monkeypatch, seed: int = 7, pool=None) -> aug.Augmenter:
    """An augmenter whose codec calibration is skipped (no codec draw is applied with it)."""
    monkeypatch.setattr(aug, "codec_lag", lambda config: 1.5)
    return aug.Augmenter(seed, pool or _pool())


# --- the draws ---------------------------------------------------------------------------


def test_the_mixture_is_the_recipes():
    assert dict(aug.MIXTURE) == {pt.CLEAN: 0.30, pt.NOISE_KIND: 0.35, pt.REVERB: 0.15,
                                 pt.BAND_LIMIT: 0.10, pt.CODEC: 0.10}
    assert aug.SNR_RANGE_DB == (3.0, 20.0) and aug.RT60_RANGE_S == (0.2, 1.0)
    assert pt.DRR_DB == 3.0 and pt.NARROWBAND_RATE == 8000
    assert [c.bitrate for c in pt.OPUS_CONFIGS] == [8_000, 12_000, 16_000]
    assert {(c.application, c.cutoff_hz) for c in pt.OPUS_CONFIGS} == {("voip", 4000)}


def test_draws_follow_the_mixture_and_stay_in_range():
    draws = [aug.draw(1, shard, row, 445) for shard in (21, 22) for row in range(10_000)]
    shares = Counter(d.kind for d in draws)
    for kind, share in aug.MIXTURE:
        assert shares[kind] / len(draws) == pytest.approx(share, abs=0.01)
    for d in draws:
        if d.kind == pt.NOISE_KIND:
            assert 3.0 <= d.level <= 20.0 and 0 <= d.noise_index < 445
            assert 0.0 <= d.noise_offset < 1.0
        elif d.kind == pt.REVERB:
            assert 0.2 <= d.level <= 1.0 and d.rir_seed is not None
        elif d.kind == pt.CODEC:
            assert d.level in (8_000.0, 12_000.0, 16_000.0)
        else:
            assert d == aug.Draw(d.kind)
    snrs = [d.level for d in draws if d.kind == pt.NOISE_KIND]
    assert min(snrs) < 3.5 and max(snrs) > 19.5  # uniform over the whole range


def test_a_draw_is_keyed_by_the_seed_and_the_row_alone():
    first = aug.draw(5, 30, 17, 445)
    random.seed(0)
    expected_next = random.random()
    random.seed(0)
    assert [aug.draw(5, 30, 17, 445) for _ in range(3)] == [first] * 3
    assert random.random() == expected_next  # the global random state is never touched
    rows = [aug.draw(5, 30, row, 445) for row in range(200)]
    assert rows[17] == first
    drawn = [d for d in rows if d.kind in (pt.NOISE_KIND, pt.REVERB)]  # continuous levels
    assert len(set(drawn)) == len(drawn) > 80
    assert [aug.draw(5, 31, row, 445) for row in range(200)] != rows  # another shard
    assert [aug.draw(6, 30, row, 445) for row in range(200)] != rows  # a new seed, new draws


def test_the_draws_use_seed_plus_two_not_the_shuffle_seeds():
    """``seed`` permutes the shards and ``seed + 1`` drives the shuffle buffer
    (training.distill_stream): the draws come from neither stream."""
    assert aug.DRAW_SEED_OFFSET == 2
    reference = random.Random("9:21:4")
    assert aug.draw(7, 21, 4, 445).kind == reference.choices(
        [k for k, _ in aug.MIXTURE], weights=[s for _, s in aug.MIXTURE])[0]


def test_an_empty_pool_is_refused():
    with pytest.raises(ValueError):
        aug.Augmenter(1, ())
    noise_row = next(row for row in range(100) if aug.draw(1, 21, row, 1).kind == pt.NOISE_KIND)
    with pytest.raises(ValueError):
        aug.draw(1, 21, noise_row, 0)


# --- the conditions ------------------------------------------------------------------------

#: One draw per condition at each end of its range.
CONDITION_DRAWS = {
    "clean": aug.Draw(pt.CLEAN),
    "noise_3db": aug.Draw(pt.NOISE_KIND, 3.0, 1, 0.25),
    "noise_20db": aug.Draw(pt.NOISE_KIND, 20.0, 0, 0.9),
    "reverb_0.2s": aug.Draw(pt.REVERB, 0.2, rir_seed=11),
    "reverb_1s": aug.Draw(pt.REVERB, 1.0, rir_seed=12),
    "band_8k": aug.Draw(pt.BAND_LIMIT),
    **{f"opus_{c.bitrate // 1000}k": aug.Draw(pt.CODEC, float(c.bitrate))
       for c in pt.OPUS_CONFIGS},
}


@pytest.mark.parametrize("probe", sorted(pt.verification_probes()))
@pytest.mark.parametrize("name", sorted(CONDITION_DRAWS))
def test_every_condition_keeps_the_clips_length_and_timing(name, probe):
    """Zero lag, to the perturbation contract's 2 samples, on every verification probe: the
    teacher's clean frames and the student's augmented frames cover the same audio."""
    chosen = CONDITION_DRAWS[name]
    if chosen.kind == pt.CODEC and not HAS_FFMPEG:
        pytest.skip("ffmpeg is not installed")
    clean = pt.verification_probes()[probe]
    augmented = aug.apply(clean, chosen, _pool())
    assert len(augmented) == len(clean) and augmented.dtype == np.float32
    assert abs(pt.residual_lag(augmented, probe, clean)) <= pt.MAX_RESIDUAL_LAG


def test_each_condition_changes_the_audio_as_drawn():
    t = np.arange(2 * SR) / SR
    clean = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 6000 * t)
             ).astype(np.float32)
    pool = _pool()
    assert aug.apply(clean, CONDITION_DRAWS["clean"], pool) is clean
    noisy = aug.apply(clean, CONDITION_DRAWS["noise_3db"], pool).astype(np.float64)
    snr = 10 * np.log10(np.mean(np.square(clean, dtype=np.float64))
                        / np.mean(np.square(noisy - clean)))
    assert snr == pytest.approx(3.0, abs=0.01)
    wet = aug.apply(clean, CONDITION_DRAWS["reverb_1s"], pool)
    assert np.sqrt(np.mean(np.square(wet, dtype=np.float64))) == pytest.approx(
        np.sqrt(np.mean(np.square(clean, dtype=np.float64))), rel=1e-4)
    assert not np.allclose(wet, clean, atol=1e-3)
    narrow = aug.apply(clean, CONDITION_DRAWS["band_8k"], pool)
    spectrum = np.abs(np.fft.rfft(narrow))
    assert spectrum[6000 * len(narrow) // SR] < 1e-3 * spectrum[440 * len(narrow) // SR]


def test_the_augmenter_applies_the_rows_draw_and_leaves_silence_clean(monkeypatch):
    augmenter = _augmenter(monkeypatch)
    clean = (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
    noise_row = next(r for r in range(100) if aug.draw(7, 21, r, 2).kind == pt.NOISE_KIND)
    expected = aug.apply(clean, aug.draw(7, 21, noise_row, 2), augmenter.pool)
    assert np.array_equal(augmenter(clean, 21, noise_row), expected)
    silent = np.zeros(SR, dtype=np.float32)
    assert augmenter(silent, 21, noise_row) is silent


def test_the_augmenter_calibrates_every_codec_configuration_up_front(monkeypatch):
    calibrated = []
    monkeypatch.setattr(aug, "codec_lag", lambda config: calibrated.append(config) or 1.5)
    augmenter = aug.Augmenter(1, _pool())
    assert calibrated == list(pt.OPUS_CONFIGS)
    assert set(augmenter.codec_lags) == {c.label for c in pt.OPUS_CONFIGS}


# --- the recipe a run's config carries -----------------------------------------------------


def test_the_recipe_record_names_every_constant_and_the_pools_checksum(tmp_path, monkeypatch):
    record = aug.recipe_record()
    assert json.loads(json.dumps(record)) == record
    assert record["version"] == aug.RECIPE_VERSION
    assert record["mixture"] == dict(aug.MIXTURE)
    other = tmp_path / "pool.json"
    other.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(aug, "POOL_PATH", other)
    assert aug.recipe_record()["noise_pool_sha256"] != record["noise_pool_sha256"]


# --- the noise pool ------------------------------------------------------------------------


def test_the_training_pool_reads_other_shards_of_d2s_collection():
    assert aug.TRAINING_NOISE.repo == pt.NOISE.repo
    assert aug.TRAINING_NOISE.revision == pt.NOISE.revision
    assert not set(aug.TRAINING_NOISE.shards) & set(pt.NOISE_SHARDS)


def test_the_committed_training_pool_is_disjoint_from_d2s():
    record = json.loads(aug.POOL_PATH.read_text(encoding="utf-8"))
    ids = [r["freesound_id"] for r in record["recordings"]]
    assert ids == sorted(set(ids)) and ids
    assert not set(ids) & aug.d2_pool_ids()
    assert record["dataset"]["shards"] == list(aug.TRAINING_NOISE.shards)
    assert record["dataset"]["revision"] == pt.NOISE.revision


def test_building_the_pool_drops_d2s_recordings(tmp_path, monkeypatch):
    d2 = sorted(aug.d2_pool_ids())[:2]
    built = _pool(d2[0], 5, d2[1], 8)
    monkeypatch.setattr(aug, "build_pool", lambda cache_dir, dataset: (
        built, {"undecodable": 0, "ineligible": 4}))
    kept, record = aug.build_training_pool(tmp_path)
    assert [r.freesound_id for r in kept] == [5, 8]
    assert record["left_out"] == {"undecodable": 0, "ineligible": 4, "in_d2_pool": 2}
    assert [r["freesound_id"] for r in record["recordings"]] == [5, 8]
    assert record["licences"] == {"CC0-1.0": 2}


def test_a_pool_holding_a_d2_recording_is_refused(tmp_path, monkeypatch):
    shared = _pool(sorted(aug.d2_pool_ids())[0], 5)
    path, record_path = tmp_path / "pool.npz", tmp_path / "pool.json"
    pt.save_pool(shared, path)
    record_path.write_text(json.dumps({"recordings": [r.as_dict() for r in shared]}),
                           encoding="utf-8")
    monkeypatch.setattr(aug, "POOL_PATH", record_path)
    with pytest.raises(ValueError, match="D2"):
        aug.load_training_pool(path)
    clean = _pool(5, 8)
    pt.save_pool(clean, path)
    record_path.write_text(json.dumps({"recordings": [r.as_dict() for r in clean]}),
                           encoding="utf-8")
    assert [r.freesound_id for r in aug.load_training_pool(path)] == [5, 8]
