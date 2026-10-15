"""Tests for training.qud_decode's bookkeeping (the decoding itself needs the GPU box)."""

from __future__ import annotations

import pytest

from training import qud_decode as qd


def test_an_interrupted_shard_resumes_on_a_clean_line_and_merges(tmp_path, monkeypatch):
    from training import qud_pilot

    path = qd.decodes_path(tmp_path, "s", 2, "base")
    assert path.name == "s.2.base.jsonl"
    assert qd.resume_keys(path) == set()
    path.write_text('{"key": "x#0"}\n{"key": "x#1"}\n{"key": "x#', encoding="utf-8")  # killed
    assert qd.resume_keys(path) == {"x#0", "x#1"}
    with open(path, "a", encoding="utf-8") as f:  # what run_shard appends next
        f.write('{"key": "x#2"}\n')
    (tmp_path / "s.base.identity.json").write_text('{"model": "base"}')
    monkeypatch.setattr(qud_pilot, "DECODES_DIR", tmp_path / "merged")
    qud_pilot.merge_decodes(tmp_path, "s", "base", 3)
    assert list(qud_pilot.read_decodes("s", "base")) == ["x#0", "x#1", "x#2"]


def test_a_malformed_interior_record_is_refused(tmp_path):
    path = qd.decodes_path(tmp_path, "s", 0, "h448")
    path.write_text('{"key": "x#0"}\n{"key": "x#\n{"key": "x#2"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2 is not a decode record"):
        qd.resume_keys(path)


def test_an_item_is_its_ayah_clip_and_segment():
    assert qd.item_key("qud_s_001_002.mp3", 3) == "qud_s_001_002.mp3#3"


def test_the_pilot_mushafs_shards_come_from_the_selection():
    assert qd.mushaf_shards("anas_almiman_yt")[0].endswith("anas_almiman_yt-00000-of-00006.parquet")
