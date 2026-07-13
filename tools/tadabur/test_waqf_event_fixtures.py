"""Unit tests for the waqf event-fixture schema (``tadabur.waqf_event_fixtures``).

Covers the validating load/write round-trip, the three-class vocabulary guard on
both ``predicted`` and ``verdict``, unknown-field rejection, and the atomic
overwrite — all without touching the model or audio.
"""

from __future__ import annotations

import json

import pytest

from .waqf_event_fixtures import (
    MID_WORD_CLOSURE,
    WAQF,
    WASL,
    WaqfEventEntry,
    load_waqf_events,
    write_waqf_events,
)


def _entry(clip="c1", idx=0, predicted=WAQF, verdict=WAQF, note="") -> WaqfEventEntry:
    return WaqfEventEntry(
        clip_id=clip, audio_ref=clip, surah_ayah="2:5", boundary_index=idx,
        word_index=3, start_s=1.2, end_s=1.5, predicted=predicted, verdict=verdict, note=note,
    )


def test_write_then_load_roundtrip(tmp_path):
    path = tmp_path / "waqf_events.jsonl"
    entries = [
        _entry("a", 0, WAQF, WAQF, "clear stop"),
        _entry("a", 1, WASL, WAQF, "detector missed the stop"),
        _entry("b", 0, WAQF, MID_WORD_CLOSURE, "qalqala on ق, not a waqf"),
    ]
    write_waqf_events(entries, path)
    assert load_waqf_events(path) == entries


def test_missing_file_is_empty(tmp_path):
    assert load_waqf_events(tmp_path / "nope.jsonl") == []


def test_blank_and_comment_lines_ignored(tmp_path):
    path = tmp_path / "waqf_events.jsonl"
    line = json.dumps({
        "clip_id": "a", "audio_ref": "a", "surah_ayah": "2:5", "boundary_index": 0,
        "word_index": 3, "start_s": 1.0, "end_s": 1.2, "predicted": WAQF,
        "verdict": WASL, "note": "",
    })
    path.write_text(f"# a comment\n\n{line}\n", encoding="utf-8")
    entries = load_waqf_events(path)
    assert len(entries) == 1 and entries[0].verdict == WASL


def test_load_rejects_unknown_verdict_class(tmp_path):
    path = tmp_path / "waqf_events.jsonl"
    path.write_text(json.dumps({
        "clip_id": "a", "audio_ref": "a", "surah_ayah": "2:5", "boundary_index": 0,
        "word_index": 3, "start_s": 1.0, "end_s": 1.2, "predicted": WAQF,
        "verdict": "maybe", "note": "",
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_waqf_events(path)


def test_load_rejects_unknown_predicted_class(tmp_path):
    path = tmp_path / "waqf_events.jsonl"
    path.write_text(json.dumps({
        "clip_id": "a", "audio_ref": "a", "surah_ayah": "2:5", "boundary_index": 0,
        "word_index": 3, "start_s": 1.0, "end_s": 1.2, "predicted": "pause",
        "verdict": WAQF, "note": "",
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_waqf_events(path)


def test_load_rejects_unknown_field(tmp_path):
    path = tmp_path / "waqf_events.jsonl"
    path.write_text(json.dumps({
        "clip_id": "a", "audio_ref": "a", "surah_ayah": "2:5", "boundary_index": 0,
        "word_index": 3, "start_s": 1.0, "end_s": 1.2, "predicted": WAQF,
        "verdict": WAQF, "note": "", "extra": 1,
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_waqf_events(path)


def test_write_rejects_invalid_entry_before_touching_disk(tmp_path):
    path = tmp_path / "waqf_events.jsonl"
    write_waqf_events([_entry("a", 0)], path)
    bad = WaqfEventEntry("b", "b", "2:5", 0, 3, 1.0, 1.2, WAQF, "bogus", "")
    with pytest.raises(ValueError):
        write_waqf_events([bad], path)
    # The pre-existing valid file is untouched by the failed write.
    assert load_waqf_events(path) == [_entry("a", 0)]
