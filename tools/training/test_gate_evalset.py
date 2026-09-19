"""Tests for the frozen gate-evaluation set: strata, sampling, statistics, provenance.

Everything here is torch-free on purpose. The numbers a ship decision turns on -- the
interval, the paired test, the population reweighting -- are arithmetic, and arithmetic that
only runs on a GPU box is arithmetic nobody checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.gate_evalset import (
    BOUNDARY_QUOTA_SHARE,
    GATE_EVAL_SHARD_START,
    NEAR_BAND_HALF_WIDTH,
    SCHEMA_VERSION,
    STRATA,
    EvalClip,
    EvalSet,
    _Reservoir,
    binomial_two_sided_p,
    boundary_quotas,
    check_provenance,
    directional_errors,
    gate_eval_shards,
    load_manifest,
    paired_comparison,
    population_weights,
    reciter_split,
    reweighted_agreement,
    stratum_for,
    training_shard_spec,
    wilson_interval,
)

THRESHOLD = 0.65


def test_strata_partition_the_ratio_line_at_the_shipped_bar():
    assert stratum_for(0.0, THRESHOLD) == "fail_clear"
    assert stratum_for(THRESHOLD - NEAR_BAND_HALF_WIDTH - 1e-9, THRESHOLD) == "fail_clear"
    assert stratum_for(THRESHOLD - NEAR_BAND_HALF_WIDTH, THRESHOLD) == "near_fail"
    assert stratum_for(THRESHOLD - 1e-9, THRESHOLD) == "near_fail"
    # The bar itself passes, so it belongs to the pass side -- getting this off by one
    # epsilon would put every exactly-at-bar clip in the wrong stratum.
    assert stratum_for(THRESHOLD, THRESHOLD) == "near_pass"
    assert stratum_for(THRESHOLD + NEAR_BAND_HALF_WIDTH, THRESHOLD) == "pass_clear"
    assert stratum_for(1.0, THRESHOLD) == "pass_clear"


def test_boundary_quotas_sum_to_the_target_and_favour_the_near_strata():
    for target in (100, 999, 1000, 1001):
        quotas = boundary_quotas(target)
        assert sum(quotas.values()) == target
        assert set(quotas) == set(STRATA)
        assert quotas["near_fail"] + quotas["near_pass"] > target // 2


def test_boundary_quota_shares_balance_the_trivial_baseline():
    """The pass strata must be ~half the boundary sample, or always-pass stays the headline."""
    pass_share = BOUNDARY_QUOTA_SHARE["near_pass"] + BOUNDARY_QUOTA_SHARE["pass_clear"]
    assert 0.45 <= pass_share <= 0.55


def test_reserved_shards_are_strided_and_outside_the_staged_block():
    shards = gate_eval_shards()
    assert len(shards) >= 10
    assert min(shards) >= GATE_EVAL_SHARD_START
    # Spread across the corpus, not a block at one end: shard order may group reciters.
    assert max(shards) - min(shards) > 300


def test_training_shard_spec_excludes_every_reserved_shard():
    from tadabur.shard_reader import parse_shard_spec

    training = set(parse_shard_spec(training_shard_spec()))
    reserved = set(gate_eval_shards())
    assert not (training & reserved)
    assert not any(index < GATE_EVAL_SHARD_START for index in training)
    # Nothing but the two reservations is dropped -- the spec must not quietly shrink the
    # corpus beyond what it claims to hold out.
    assert len(training) + len(reserved) == 385 - GATE_EVAL_SHARD_START


def test_reciter_split_is_stable_and_roughly_balanced():
    assert reciter_split(7) == reciter_split(7)
    splits = [reciter_split(i) for i in range(400)]
    assert set(splits) == {"dev", "test"}
    assert 0.4 < splits.count("test") / len(splits) < 0.6


def test_binomial_two_sided_p_matches_hand_computed_values():
    assert binomial_two_sided_p(0, 0) == 1.0
    # All ten discordant pairs on one side: 2 * (1/1024).
    assert binomial_two_sided_p(0, 10) == pytest.approx(2 / 1024)
    # A perfectly even split cannot be evidence of anything.
    assert binomial_two_sided_p(5, 10) == pytest.approx(1.0)
    assert binomial_two_sided_p(1, 10) == pytest.approx(22 / 1024)


def test_wilson_interval_brackets_the_estimate_and_narrows_with_n():
    low_small, high_small = wilson_interval(950, 1000)
    low_big, high_big = wilson_interval(1900, 2000)
    assert low_small < 0.95 < high_small
    assert (high_big - low_big) < (high_small - low_small)
    # The whole reason for n >= 1000: 95% observed at n=1000 does not exclude 95%.
    assert low_small < 0.95
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert wilson_interval(10, 10)[1] <= 1.0


def test_paired_comparison_counts_only_discordant_clips():
    student = [True, True, False, False, True]
    other = [True, False, True, False, False]
    result = paired_comparison(student, other, "student", "other")
    assert result.a_only_correct == 2
    assert result.b_only_correct == 1
    assert result.discordant == 3
    assert result.as_dict()["comparison"] == "student vs other"


def test_paired_comparison_refuses_unmatched_clip_sets():
    with pytest.raises(ValueError):
        paired_comparison([True], [True, False], "a", "b")


def test_directional_errors_separate_too_strict_from_too_lax():
    # (teacher_passed, student_passed)
    decisions = [(True, True), (True, False), (False, False), (False, True), (False, True)]
    errors = directional_errors(decisions)
    assert errors.teacher_passes == 2
    assert errors.teacher_fails == 3
    assert errors.false_fails == 1
    assert errors.false_passes == 2
    assert errors.false_fail_rate == pytest.approx(0.5)
    assert errors.false_pass_rate == pytest.approx(2 / 3)


def test_reweighted_agreement_recovers_the_population_number():
    scanned = {"fail_clear": 100, "near_fail": 100, "near_pass": 100, "pass_clear": 700}
    per_stratum = {
        "fail_clear": 1.0,
        "near_fail": 0.5,
        "near_pass": 0.5,
        "pass_clear": 1.0,
    }
    # 0.1 + 0.05 + 0.05 + 0.7 = 0.90, i.e. the enriched near-bar errors are diluted back.
    assert reweighted_agreement(per_stratum, scanned) == pytest.approx(0.90)


def test_reweighted_agreement_renormalises_over_covered_strata_only():
    scanned = {"fail_clear": 0, "near_fail": 50, "near_pass": 50, "pass_clear": 0}
    assert reweighted_agreement({"near_fail": 0.8, "near_pass": 0.6}, scanned) == pytest.approx(0.7)
    assert reweighted_agreement({"near_fail": 1.0}, {name: 0 for name in STRATA}) == 0.0


def test_population_weights_of_an_empty_scan_are_zero_not_a_crash():
    assert population_weights({"near_fail": 0}) == {"near_fail": 0.0}


def test_reservoir_caps_at_its_quota_and_keeps_every_item_reachable():
    reservoir = _Reservoir(3, seed=1)
    for index in range(50):
        reservoir.offer(index)
    assert len(reservoir.kept) == 3
    assert reservoir.seen == 50
    assert reservoir.is_full
    # A pure prefix would be {0, 1, 2}: the sample must be able to reach later items, or it
    # is sampling shard order (i.e. reciters) rather than the corpus.
    assert any(item >= 3 for item in reservoir.kept)


def test_reservoir_below_quota_keeps_everything():
    reservoir = _Reservoir(10, seed=1)
    for index in range(4):
        assert reservoir.offer(index)
    assert reservoir.kept == [0, 1, 2, 3]
    assert not reservoir.is_full


def _clip(name: str, ratio: float, passed: bool, **overrides) -> EvalClip:
    base = dict(
        filename=name,
        surah_ayah="78:1",
        reciter_id=3,
        shard=20,
        duration_s=4.0,
        stratum=stratum_for(ratio, THRESHOLD),
        split="dev",
        in_population=True,
        in_boundary=False,
        teacher_text="ab",
        teacher_ratio=ratio,
        teacher_passed=passed,
        teacher_insertion_run=0,
        teacher_added_shadda=False,
    )
    base.update(overrides)
    return EvalClip(**base)


def _evalset(clips, **provenance) -> EvalSet:
    base = {"teacher_model_id": "obadx/muaalem-model-v3_2", "correct_threshold": 0.65}
    base.update(provenance)
    return EvalSet(
        schema_version=SCHEMA_VERSION,
        clips=tuple(clips),
        scanned_by_stratum={name: 10 for name in STRATA},
        num_scanned=40,
        num_skipped=0,
        provenance=base,
    )


def test_subset_selects_by_view_and_split():
    clips = [
        _clip("a.wav", 0.9, True),
        _clip("b.wav", 0.5, False, in_population=False, in_boundary=True, split="test"),
    ]
    evalset = _evalset(clips)
    assert [c.filename for c in evalset.subset("population")] == ["a.wav"]
    assert [c.filename for c in evalset.subset("boundary")] == ["b.wav"]
    assert len(evalset.subset("all")) == 2
    assert [c.filename for c in evalset.subset("all", "test")] == ["b.wav"]
    with pytest.raises(ValueError):
        evalset.subset("nonsense")


def test_load_manifest_refuses_a_foreign_schema(tmp_path: Path):
    payload = _evalset([_clip("a.wav", 0.9, True)]).as_dict()
    payload["schema_version"] = "gate-evalset-v0"
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="gate-evalset-v0"):
        load_manifest(tmp_path)


def test_load_manifest_round_trips_a_written_set(tmp_path: Path):
    original = _evalset([_clip("a.wav", 0.9, True), _clip("b.wav", 0.42, False)])
    (tmp_path / "manifest.json").write_text(
        json.dumps(original.as_dict(), ensure_ascii=False), encoding="utf-8"
    )
    loaded = load_manifest(tmp_path)
    assert [c.filename for c in loaded.clips] == ["a.wav", "b.wav"]
    assert loaded.clips[1].stratum == "fail_clear"
    assert loaded.scanned_by_stratum == original.scanned_by_stratum


def test_check_provenance_refuses_a_different_bar_teacher_or_protocol():
    from training.distill_eval import PROTOCOL_VERSION
    from training.distill_loss import CONFIRM_TIMESTEPS

    good = _evalset(
        [_clip("a.wav", 0.9, True)],
        confirm_timesteps=CONFIRM_TIMESTEPS,
        protocol_version=PROTOCOL_VERSION,
    )
    check_provenance(good, "obadx/muaalem-model-v3_2", 0.65)

    with pytest.raises(SystemExit, match="correct_threshold"):
        check_provenance(good, "obadx/muaalem-model-v3_2", 0.75)
    with pytest.raises(SystemExit, match="teacher_model_id"):
        check_provenance(good, "some/other-teacher", 0.65)

    # The decode protocol is the field most likely to change without anyone thinking of the
    # cache, so it is checked in the same place as the rest rather than by a second caller.
    stale = _evalset(
        [_clip("a.wav", 0.9, True)],
        confirm_timesteps=CONFIRM_TIMESTEPS,
        protocol_version="confirmed-stream-v1",
    )
    with pytest.raises(SystemExit, match="protocol_version"):
        check_provenance(stale, "obadx/muaalem-model-v3_2", 0.65)


def test_the_exact_p_value_survives_more_than_1023_discordant_pairs():
    """``2.0 ** trials`` overflows at 1024 -- which is exactly when this gets called."""
    assert binomial_two_sided_p(0, 1024) < 1e-300
    assert binomial_two_sided_p(900, 2000) < 1e-4
    assert binomial_two_sided_p(1000, 2000) == pytest.approx(1.0)
    assert binomial_two_sided_p(750, 1500) == pytest.approx(1.0)


def test_lossless_audio_round_trips_bit_exactly(tmp_path):
    """The manifest caches a decode of the in-memory waveform; the student reads the file.

    Anything lossy between the two is a difference charged entirely to the student. Measured
    on real clips, a PCM_16 round trip changed the teacher's decoded string on 12 of 60 and
    moved match_ratio by up to 0.062 -- and Tadabur audio peaks above 1.0, so PCM_16 clips.
    """
    import numpy as np
    import soundfile as sf

    from training.gate_evalset import CLIPS_DIRNAME  # noqa: F401  (documents the layout)

    samples = (np.random.default_rng(0).standard_normal(4000) * 0.4).astype("float32")
    samples[0] = 1.037  # a real measured peak: PCM_16 would clip this
    path = tmp_path / "clip.wav"
    sf.write(str(path), samples, 16000, subtype="FLOAT")
    back, rate = sf.read(str(path), dtype="float32")
    assert rate == 16000
    assert np.array_equal(samples, back)


def test_the_streaming_dataset_refuses_every_reserved_shard():
    """"Remember not to train on those" is not a mechanism; the constructor is."""
    import pytest

    from training.distill_stream import StreamingWindowDataset, held_out_shards

    reserved = sorted(held_out_shards())
    assert set(gate_eval_shards()) <= set(reserved)
    assert set(range(0, GATE_EVAL_SHARD_START)) <= set(reserved)

    for shard in (0, 19, gate_eval_shards()[0], gate_eval_shards()[-1]):
        with pytest.raises(ValueError, match="held out"):
            StreamingWindowDataset([shard])


def test_the_default_training_spec_is_exactly_the_complement():
    from tadabur.shard_reader import parse_shard_spec

    from training.distill_stream import default_train_shards, held_out_shards

    trainable = set(parse_shard_spec(default_train_shards()))
    assert not (trainable & held_out_shards())
    assert len(trainable) + len(held_out_shards()) == 385
