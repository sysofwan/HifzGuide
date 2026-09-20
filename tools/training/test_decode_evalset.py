"""Tests for the frozen decode-evaluation set: shards, sampling, statistics, provenance.

Everything here is torch-free on purpose. The numbers a ship decision turns on -- the
clustered interval, the reciter split, the provenance guards -- are arithmetic, and
arithmetic that only runs on a GPU box is arithmetic nobody checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.decode_evalset import (
    GATE_EVAL_SHARD_START,
    SCHEMA_VERSION,
    EvalClip,
    EvalSet,
    _Reservoir,
    check_provenance,
    cluster_bootstrap,
    gate_eval_shards,
    load_manifest,
    reciter_split,
    training_shard_spec,
    wilson_interval,
)

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


def test_wilson_interval_brackets_the_estimate_and_narrows_with_n():
    low_small, high_small = wilson_interval(950, 1000)
    low_big, high_big = wilson_interval(1900, 2000)
    assert low_small < 0.95 < high_small
    assert (high_big - low_big) < (high_small - low_small)
    # The whole reason for n >= 1000: 95% observed at n=1000 does not exclude 95%.
    assert low_small < 0.95
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert wilson_interval(10, 10)[1] <= 1.0


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


def _clip(name: str, text: str = "abc", **overrides) -> EvalClip:
    base = dict(
        filename=name,
        surah_ayah="78:1",
        reciter_id=3,
        shard=20,
        duration_s=4.0,
        split="dev",
        teacher_text=text,
    )
    base.update(overrides)
    return EvalClip(**base)


def _evalset(clips, **provenance) -> EvalSet:
    base = {"teacher_model_id": "obadx/muaalem-model-v3_2"}
    base.update(provenance)
    return EvalSet(
        schema_version=SCHEMA_VERSION,
        clips=tuple(clips),
        num_scanned=40,
        num_skipped=0,
        provenance=base,
    )


def test_load_manifest_refuses_a_foreign_schema(tmp_path: Path):
    payload = _evalset([_clip("a.wav")]).as_dict()
    payload["schema_version"] = "gate-evalset-v0"
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="gate-evalset-v0"):
        load_manifest(tmp_path)


def test_load_manifest_round_trips_a_written_set(tmp_path: Path):
    original = _evalset([_clip("a.wav"), _clip("b.wav")])
    (tmp_path / "manifest.json").write_text(
        json.dumps(original.as_dict(), ensure_ascii=False), encoding="utf-8"
    )
    loaded = load_manifest(tmp_path)
    assert [c.filename for c in loaded.clips] == ["a.wav", "b.wav"]
    assert loaded.clips[0].teacher_text == "abc"


def test_check_provenance_refuses_a_different_bar_teacher_or_protocol():
    from training.distill_eval import PROTOCOL_VERSION
    from training.distill_loss import CONFIRM_TIMESTEPS

    good = _evalset(
        [_clip("a.wav")],
        confirm_timesteps=CONFIRM_TIMESTEPS,
        protocol_version=PROTOCOL_VERSION,
    )
    check_provenance(good, "obadx/muaalem-model-v3_2")

    with pytest.raises(SystemExit, match="teacher_model_id"):
        check_provenance(good, "some/other-teacher")

    # The decode protocol is the field most likely to change without anyone thinking of the
    # cache, so it is checked in the same place as the rest rather than by a second caller.
    stale = _evalset(
        [_clip("a.wav")],
        confirm_timesteps=CONFIRM_TIMESTEPS,
        protocol_version="confirmed-stream-v1",
    )
    with pytest.raises(SystemExit, match="protocol_version"):
        check_provenance(stale, "obadx/muaalem-model-v3_2")


def test_lossless_audio_round_trips_bit_exactly(tmp_path):
    """The manifest caches a decode of the in-memory waveform; the student reads the file.

    Anything lossy between the two is a difference charged entirely to the student. Measured
    on real clips, a PCM_16 round trip changed the teacher's decoded string on 12 of 60 and
    moved match_ratio by up to 0.062 -- and Tadabur audio peaks above 1.0, so PCM_16 clips.
    """
    import numpy as np
    import soundfile as sf

    from training.decode_evalset import CLIPS_DIRNAME  # noqa: F401  (documents the layout)

    samples = (np.random.default_rng(0).standard_normal(4000) * 0.4).astype("float32")
    samples[0] = 1.037  # a real measured peak: PCM_16 would clip this
    path = tmp_path / "clip.wav"
    sf.write(str(path), samples, 16000, subtype="FLOAT")
    back, rate = sf.read(str(path), dtype="float32")
    assert rate == 16000
    assert np.array_equal(samples, back)


def test_the_streaming_dataset_refuses_every_reserved_shard():
    """"Remember not to train on those" is not a mechanism; the constructor is."""
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


def test_the_fingerprint_changes_when_the_cached_truth_does():
    """Filenames repeat across rebuilds; the teacher's decodes underneath them may not."""
    base = _evalset([_clip("a.wav")], protocol_version="v2")
    assert base.fingerprint() == _evalset([_clip("a.wav")], protocol_version="v2").fingerprint()

    moved_truth = _evalset([_clip("a.wav", text="abd")], protocol_version="v2")
    assert moved_truth.fingerprint() != base.fingerprint()

    other_protocol = _evalset([_clip("a.wav")], protocol_version="v1")
    assert other_protocol.fingerprint() != base.fingerprint()


def test_the_training_spec_excludes_the_shards_a_set_was_actually_built_on():
    """`--shards` can override the canonical reservation; the guard must follow it."""
    from tadabur.shard_reader import parse_shard_spec

    spec = training_shard_spec(reserved_shards=[200, 201])
    trainable = set(parse_shard_spec(spec))
    assert not ({200, 201} & trainable)
    # And the canonical block is no longer reserved when it was not what was used.
    assert gate_eval_shards()[5] in trainable


# --- The one bootstrap, exercised as the shipping code calls it ---


def test_cluster_bootstrap_is_wider_when_the_outcome_clusters_by_reciter():
    """Same marginal proportion either way; only the clustering differs.

    The independent-sample interval cannot tell these apart, which is the whole problem.
    """
    clustered = [(reciter % 5 != 0, reciter) for reciter in range(40) for _ in range(10)]
    spread = [(i % 5 != 0, reciter) for reciter in range(40) for i in range(10)]
    assert sum(o for o, _ in clustered) == sum(o for o, _ in spread)

    mean = lambda rows: sum(o for o, _ in rows) / len(rows)  # noqa: E731
    wide = cluster_bootstrap(clustered, lambda r: r[1], mean, seed=1)
    narrow = cluster_bootstrap(spread, lambda r: r[1], mean, seed=1)
    naive = wilson_interval(sum(o for o, _ in clustered), len(clustered))

    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])
    assert (wide[1] - wide[0]) > (naive[1] - naive[0])


def test_cluster_bootstrap_brackets_the_estimate_and_is_deterministic():
    rows = [(i % 10 != 0, i // 5) for i in range(500)]
    mean = lambda r: sum(o for o, _ in r) / len(r)  # noqa: E731
    low, high = cluster_bootstrap(rows, lambda r: r[1], mean, seed=3)
    assert low <= 0.9 <= high and 0.0 <= low <= high <= 1.0
    assert (low, high) == cluster_bootstrap(rows, lambda r: r[1], mean, seed=3)
    assert cluster_bootstrap([], lambda r: r[1], mean, seed=3) == (0.0, 0.0)


def test_the_percentile_indices_track_the_iteration_count():
    """They were once literals (`draws[100], draws[3899]`) beside a literal `range(4000)`.

    Changing either produced a wrong interval with no error, so pin that they move together.
    """
    rows = [(i % 4 != 0, i) for i in range(200)]
    mean = lambda r: sum(o for o, _ in r) / len(r)  # noqa: E731
    for iterations in (200, 1000, 4000):
        low, high = cluster_bootstrap(
            rows, lambda r: r[1], mean, seed=5, iterations=iterations
        )
        assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_is_the_project_wide_one():
    """A second copy of this formula is a second chance to get the extremes wrong."""
    from training.counterfactual_eval import wilson_interval as canonical

    for successes, trials in ((0, 10), (10, 10), (950, 1000), (1, 3)):
        assert wilson_interval(successes, trials) == canonical(successes, trials)


def test_legacy_stratified_clips_are_dropped_rather_than_pooled(tmp_path: Path):
    """They are not a uniform draw, so pooling them would bias the headline toward the bar."""
    payload = _evalset([_clip("keep.wav")]).as_dict()
    payload["clips"].append({**payload["clips"][0], "filename": "drop.wav",
                             "in_population": False})
    payload["clips"][0]["in_population"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_manifest(tmp_path)
    assert [c.filename for c in loaded.clips] == ["keep.wav"]
    assert loaded.provenance["dropped_stratified_clips"] == 1


def test_a_manifest_without_the_legacy_flag_keeps_every_clip(tmp_path: Path):
    payload = _evalset([_clip("a.wav"), _clip("b.wav")]).as_dict()
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_manifest(tmp_path)
    assert len(loaded.clips) == 2
    assert loaded.provenance["dropped_stratified_clips"] == 0
