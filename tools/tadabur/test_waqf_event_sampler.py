"""Unit tests for the waqf candidate-boundary sampler (``tadabur.waqf_event_sampler``).

Covers the per-class stratified draw, its determinism / class-order-independence,
that a row is served under its clip's staged ``audio_filename`` (``audio_ref``), the
worklist JSONL round-trip, and the loud rejection of an unknown predicted class in
the candidate manifest.
"""

from __future__ import annotations

import json

import pytest

from .waqf_event_fixtures import MID_WORD_CLOSURE, WAQF, WASL
from .waqf_event_sampler import (
    WaqfCandidate,
    read_candidates,
    sample_clips,
    sample_worklist,
    write_clip_worklist,
    write_worklist,
)


def _cand(clip, idx, predicted) -> WaqfCandidate:
    return WaqfCandidate(
        clip_id=clip, audio_ref=f"{clip}.wav", surah_ayah="2:5", boundary_index=idx,
        word_index=idx + 1, start_s=float(idx), end_s=idx + 0.3, predicted=predicted,
    )


def _population():
    cands = []
    for i in range(10):
        cands.append(_cand(f"clip{i}", 0, WAQF))
        cands.append(_cand(f"clip{i}", 1, WASL))
    cands.append(_cand("clip0", 2, MID_WORD_CLOSURE))
    return cands


def test_sample_is_stratified_and_capped():
    items = sample_worklist(_population(), per_class=3, seed=0)
    by_class = {}
    for it in items:
        by_class.setdefault(it.predicted, []).append(it)
    assert len(by_class[WAQF]) == 3
    assert len(by_class[WASL]) == 3
    # Only one mid-word-closure candidate exists — take it all, don't pad.
    assert len(by_class[MID_WORD_CLOSURE]) == 1


def test_sample_serves_clip_under_its_staged_filename():
    # The whole clips are staged by tadabur.waqf_segments under their raw
    # audio_filename, so the worklist serves each row by that exact name (audio_ref),
    # not a hashed export name — the UI's --audio-dir is that staging directory.
    items = sample_worklist([_cand("clipA", 0, WAQF)], per_class=5, seed=0)
    assert items[0].local_audio_path == items[0].audio_ref == "clipA.wav"


def test_sample_is_deterministic_and_order_independent():
    pop = _population()
    a = sample_worklist(pop, per_class=4, seed=7)
    b = sample_worklist(list(reversed(pop)), per_class=4, seed=7)
    assert a == b
    # A different seed generally yields a different draw.
    c = sample_worklist(pop, per_class=4, seed=8)
    assert {(i.clip_id, i.boundary_index) for i in a} != {(i.clip_id, i.boundary_index) for i in c}


def test_worklist_jsonl_roundtrip(tmp_path):
    items = sample_worklist(_population(), per_class=2, seed=0)
    path = tmp_path / "worklist.jsonl"
    write_worklist(items, path)
    with open(path, encoding="utf-8") as f:
        loaded = [json.loads(line) for line in f]
    assert len(loaded) == len(items)
    assert loaded[0]["local_audio_path"] == items[0].local_audio_path


def test_sample_clips_is_distinct_and_covers_sampled_boundaries():
    # Two boundaries on clip0 (waqf + mid_word_closure) collapse to one clip entry.
    clips = sample_clips(_population(), per_class=2, seed=0)
    assert len(clips) == len(set(clips))  # distinct
    items = sample_worklist(_population(), per_class=2, seed=0)
    assert set(clips) == {i.clip_id for i in items}  # exactly the sampled clips


def test_sample_clips_is_deterministic():
    a = sample_clips(_population(), per_class=2, seed=0)
    b = sample_clips(_population(), per_class=2, seed=0)
    assert a == b


def test_clip_worklist_jsonl_roundtrip(tmp_path):
    clips = sample_clips(_population(), per_class=2, seed=0)
    path = tmp_path / "clips.jsonl"
    write_clip_worklist(clips, path)
    with open(path, encoding="utf-8") as f:
        loaded = [json.loads(line)["clip_id"] for line in f]
    assert loaded == clips


def test_read_candidates_rejects_unknown_class(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text(json.dumps({
        "clip_id": "a", "audio_ref": "a.wav", "surah_ayah": "2:5", "boundary_index": 0,
        "word_index": 1, "start_s": 0.0, "end_s": 0.3, "predicted": "pause",
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_candidates(path)


def test_read_candidates_roundtrip(tmp_path):
    path = tmp_path / "candidates.jsonl"
    cands = [_cand("a", 0, WAQF), _cand("a", 1, WASL)]
    with open(path, "w", encoding="utf-8") as f:
        from dataclasses import asdict
        for c in cands:
            f.write(json.dumps(asdict(c)) + "\n")
    assert read_candidates(path) == cands
