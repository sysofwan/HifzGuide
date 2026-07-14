"""Unit tests for the reciter-disjoint waqf partitioner (``tadabur.waqf_partition``)."""

from __future__ import annotations

import json

import pytest

from tadabur import waqf_partition as wp


def _rows(*specs):
    # specs: (reciter_id, count) -> candidate rows for that reciter's clip.
    rows = []
    for reciter, count in specs:
        clip = f"tadabur_spk{reciter:04d}_S1_A1_x_000000.wav"
        for i in range(count):
            rows.append({"clip_id": clip, "boundary_index": i, "predicted": "wasl"})
    return rows


def test_reciter_of_parses_spk_field():
    assert wp.reciter_of("tadabur_spk0445_S24_A9_df81311e_000052.wav") == 445
    with pytest.raises(ValueError):
        wp.reciter_of("no_reciter_here.wav")


def test_partitions_are_reciter_disjoint():
    rows = _rows((1, 2), (2, 3), (3, 1), (4, 4))
    calibration, test, report = wp.partition(rows, excluded=set(), test_fraction=0.5, seed=0)
    cal_reciters = {wp.reciter_of(r["clip_id"]) for r in calibration}
    test_reciters = {wp.reciter_of(r["clip_id"]) for r in test}
    assert cal_reciters.isdisjoint(test_reciters)
    assert cal_reciters | test_reciters == {1, 2, 3, 4}
    assert len(calibration) + len(test) == len(rows)


def test_training_reciters_excluded_from_both():
    rows = _rows((1, 2), (2, 2), (3, 2))
    calibration, test, report = wp.partition(rows, excluded={2}, test_fraction=0.5, seed=0)
    present = {wp.reciter_of(r["clip_id"]) for r in calibration + test}
    assert 2 not in present
    assert report["dropped_train_reciters"] == [2]
    assert report["counts"]["dropped_train_rows"] == 2


def test_split_is_deterministic_in_seed():
    rows = _rows(*[(r, 1) for r in range(10)])
    a = wp.partition(rows, set(), 0.5, seed=7)[2]["test_reciters"]
    b = wp.partition(rows, set(), 0.5, seed=7)[2]["test_reciters"]
    c = wp.partition(rows, set(), 0.5, seed=8)[2]["test_reciters"]
    assert a == b
    assert a != c  # different seed => (very likely) different split


def test_test_fraction_controls_reciter_counts():
    rows = _rows(*[(r, 1) for r in range(10)])
    _, _, report = wp.partition(rows, set(), test_fraction=0.3, seed=0)
    assert report["counts"]["test_reciters"] == 3
    assert report["counts"]["calibration_reciters"] == 7


def test_invalid_test_fraction_rejected():
    with pytest.raises(ValueError):
        wp.partition(_rows((1, 1)), set(), test_fraction=1.5)


def test_train_reciters_from_manifest_and_ids(tmp_path):
    manifest = tmp_path / "passing.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(json.dumps({"reciter_id": 5, "audio_filename": "x"}) + "\n")
        f.write(json.dumps({"audio_filename": "tadabur_spk0006_S1_A1_x_0.wav"}) + "\n")
    reciters = wp.train_reciters([9], manifest)
    assert reciters == {5, 6, 9}
