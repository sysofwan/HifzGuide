"""Tests for training.qud_decode's bookkeeping (the decoding itself needs the GPU box)."""

from __future__ import annotations

from training import qud_decode as qd


def test_a_shard_resumes_from_its_complete_lines_only(tmp_path):
    path = qd.decodes_path(tmp_path, "s", 2, "h448")
    assert path.name == "s.2.h448.jsonl"
    assert qd.done_keys(path) == set()
    path.write_text('{"key": "x.mp3#0"}\n{"key": "x.mp3#1"}\n{"key": "x.mp3', encoding="utf-8")
    assert qd.done_keys(path) == {"x.mp3#0", "x.mp3#1"}


def test_an_item_is_its_ayah_clip_and_segment():
    assert qd.item_key("qud_s_001_002.mp3", 3) == "qud_s_001_002.mp3#3"


def test_the_pilot_mushafs_shards_come_from_the_selection():
    assert qd.mushaf_shards("anas_almiman_yt")[0].endswith("anas_almiman_yt-00000-of-00006.parquet")
