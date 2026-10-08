"""Tests for the exposure registry: its schema, its consistency checks, the disjointness
check, and the committed registry against the sources it was built from (#89)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from tadabur.exposure import (
    EVALSET_DEV,
    EVALSET_LEGACY,
    EVALSET_TEST,
    H448_STREAM_SHARDS,
    H448_TRAINING,
    LABEL_FILES,
    LABEL_P35_FIXTURE,
    LABEL_TASHKEEL_COUNTERFACTUAL,
    MINING_POOL,
    SYNTHETIC_EDIT_SOURCE,
    TRUTH_SITE_USES,
    USES,
    Exposure,
    ExposureOverlap,
    ShardExposure,
    check_disjoint,
    evalset_records,
    label_file_clips,
    load_registry,
    parse_exposure,
    shard_exposure,
    staged_exposures,
    write_shard_use,
    write_use,
)
from tadabur.staged_audio import load_staged_clips

SHA = "b" * 64
WAQF = TRUTH_SITE_USES["waqf_boundary"]


def _row(name="a.wav", shard=3, row=7, reciter=12, span=(None, None), sha=SHA) -> Exposure:
    return Exposure(name, shard, row, reciter, span[0], span[1], sha)


def _json(row: Exposure) -> dict:
    return dataclasses.asdict(row)


# --- the schema ------------------------------------------------------------------------


def test_parse_exposure_accepts_a_whole_clip_and_a_span():
    assert parse_exposure(_json(_row()), "t") == _row()
    assert parse_exposure(_json(_row(span=(0, 16000), sha=None)), "t").end_sample == 16000


@pytest.mark.parametrize("change", [
    {"reciter_id": True},                         # a bool is not an int
    {"start_sample": 0},                          # half a span
    {"start_sample": 5, "end_sample": 5},         # an empty span
    {"audio_sha256": "ABC"},
    {"audio_filename": "dir/a.wav"},
    {"shard": 385},
    {"extra": 1},
])
def test_parse_exposure_rejects_a_malformed_row(change):
    with pytest.raises(ValueError):
        parse_exposure({**_json(_row()), **change}, "t")


def test_write_and_load_round_trip_sorted_and_deduplicated(tmp_path):
    write_use(WAQF, [_row("b.wav", row=8), _row("a.wav"), _row("a.wav")], tmp_path)
    write_shard_use(H448_TRAINING, ShardExposure((21, 22), {12: 3, 40: 1}, SHA), tmp_path)
    registry = load_registry(tmp_path)
    assert [r.audio_filename for r in registry.recordings[WAQF]] == ["a.wav", "b.wav"]
    assert registry.reciters(H448_TRAINING) == {12, 40}
    assert registry.uses() == sorted([WAQF, H448_TRAINING])
    assert registry.uses_of(12) == sorted([WAQF, H448_TRAINING])
    assert registry.reciters(MINING_POOL) == frozenset()  # a use with no file is empty


def test_load_refuses_unknown_files_and_unsorted_rows(tmp_path):
    (tmp_path / "not_a_use.jsonl").write_text("")
    with pytest.raises(ValueError, match="unknown use"):
        load_registry(tmp_path)
    (tmp_path / "not_a_use.jsonl").unlink()
    rows = [_row("b.wav", row=8), _row("a.wav")]
    (tmp_path / f"{WAQF}.jsonl").write_text("".join(json.dumps(_json(r)) + "\n" for r in rows))
    with pytest.raises(ValueError, match="sorted"):
        load_registry(tmp_path)


@pytest.mark.parametrize("other", [
    _row("a.wav", reciter=99),             # one filename, two reciters
    _row("z.wav"),                         # two filenames, one shard row
    _row("a.wav", sha="c" * 64),           # one filename, two checksums
])
def test_a_recording_is_the_same_recording_in_every_use(tmp_path, other):
    write_use(WAQF, [_row("a.wav")], tmp_path)
    with pytest.raises(ValueError):
        write_use(MINING_POOL, [other], tmp_path)
    assert not (tmp_path / f"{MINING_POOL}.jsonl").exists()


def test_an_unknown_use_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown use"):
        write_use("panel_v2", [_row()], tmp_path)


# --- the check -------------------------------------------------------------------------


def _registry(tmp_path):
    write_use(WAQF, [_row("a.wav", reciter=1), _row("b.wav", row=8, reciter=2)], tmp_path)
    write_use(MINING_POOL, [_row("c.wav", shard=39, row=0, reciter=2)], tmp_path)
    write_use(SYNTHETIC_EDIT_SOURCE, [_row("d.wav", shard=21, row=0, reciter=3)], tmp_path)
    write_shard_use(H448_TRAINING, ShardExposure((21,), {3: 1, 7: 2}, SHA), tmp_path)
    return load_registry(tmp_path)


def test_check_disjoint_by_reciter_names_the_shared_reciters(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(ExposureOverlap, match=r"mining_pool and truth_site.waqf_boundary share 1 reciter"):
        check_disjoint(MINING_POOL, WAQF, registry=registry)
    check_disjoint(MINING_POOL, SYNTHETIC_EDIT_SOURCE, registry=registry)
    check_disjoint(MINING_POOL, WAQF, by="source", registry=registry)  # different recordings


def test_check_disjoint_by_source_counts_a_shard_use_as_all_its_rows(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(ExposureOverlap, match="share 1 sources"):
        check_disjoint(SYNTHETIC_EDIT_SOURCE, H448_TRAINING, by="source", registry=registry)
    check_disjoint(WAQF, H448_TRAINING, by="source", registry=registry)


def test_check_disjoint_compares_only_the_first_use_with_the_others(tmp_path):
    registry = _registry(tmp_path)
    # truth sites and the pool share reciter 2, but neither is compared with the other.
    check_disjoint(SYNTHETIC_EDIT_SOURCE, WAQF, MINING_POOL, registry=registry)


def test_check_disjoint_refuses_unknown_names_and_modes(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(ValueError, match="unknown use"):
        check_disjoint(WAQF, "the_panel", registry=registry)
    with pytest.raises(ValueError, match="by must be"):
        check_disjoint(WAQF, MINING_POOL, by="clip", registry=registry)


# --- the builders ----------------------------------------------------------------------


def test_shard_exposure_counts_rows_per_reciter_and_refuses_unindexed_shards(tmp_path):
    index = tmp_path / "index.jsonl"
    rows = [
        {"audio_filename": f"r{i}.wav", "shard": shard, "row_index": i, "reciter_id": reciter,
         "surah_id": 0, "ayah_id": 1, "ayah_duration_s": 3.0}
        for i, (shard, reciter) in enumerate([(21, 5), (21, 5), (22, 6), (39, 7)])
    ]
    index.write_text("".join(json.dumps(r) + "\n" for r in rows))
    found = shard_exposure(index, [21, 22])
    assert (found.shards, dict(found.rows_per_reciter)) == ((21, 22), {5: 2, 6: 1})
    with pytest.raises(ValueError, match="does not index"):
        shard_exposure(index, [21, 23])


def test_evalset_records_reads_each_split_and_the_legacy_sample(tmp_path):
    manifest = tmp_path / "manifest.json"
    clips = [
        {"filename": "tadabur_sh039_i00012_S1_A2_.wav", "shard": 39, "split": "dev"},
        {"filename": "tadabur_sh058_i00003_S1_A2_.wav", "shard": 58, "split": "test"},
        {"filename": "tadabur_sh058_i00004_S1_A2_.wav", "shard": 58, "split": "dev",
         "in_population": False},
    ]
    manifest.write_text(json.dumps({"clips": clips}))
    assert evalset_records(manifest) == [
        ((39, 12), EVALSET_DEV), ((58, 3), EVALSET_TEST), ((58, 4), EVALSET_LEGACY),
    ]
    clips[0]["shard"] = 40
    manifest.write_text(json.dumps({"clips": clips}))
    with pytest.raises(ValueError, match="cannot read"):
        evalset_records(manifest)


def test_label_files_name_whole_clips():
    fixtures = label_file_clips(LABEL_P35_FIXTURE)
    assert len(fixtures) == 206 and not any("__seg" in name for name in fixtures)
    assert all(name.endswith(".wav") for name in label_file_clips(LABEL_TASHKEEL_COUNTERFACTUAL))


# --- the committed registry ------------------------------------------------------------


def test_h448_stream_shards_are_the_training_spec_and_avoid_every_unseen_shard():
    from tadabur.mining_pool import HELD_OUT_BLOCK
    from tadabur.shard_reader import parse_shard_spec
    from training.decode_evalset import gate_eval_shards, training_shard_spec

    assert H448_STREAM_SHARDS == training_shard_spec()
    trained = set(parse_shard_spec(H448_STREAM_SHARDS))
    assert not trained & (set(HELD_OUT_BLOCK) | set(gate_eval_shards()))
    assert load_registry().shard_uses[H448_TRAINING].shards == tuple(sorted(trained))


def test_the_committed_registry_loads_and_uses_only_known_names():
    registry = load_registry()
    assert set(registry.uses()) <= USES
    assert {WAQF, MINING_POOL, EVALSET_DEV, EVALSET_TEST, H448_TRAINING} <= set(registry.uses())


def test_the_committed_staged_uses_match_the_staged_registry_and_truth_sites():
    registry = load_registry()
    for use, rows in staged_exposures(load_staged_clips()).items():
        assert registry.recordings[use] == tuple(sorted(set(rows))), use


def test_every_label_file_clip_is_in_its_committed_use():
    registry = load_registry()
    for use in LABEL_FILES:
        recorded = {row.audio_filename for row in registry.recordings[use]}
        assert recorded == label_file_clips(use), use
