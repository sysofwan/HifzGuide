"""Unit tests for the waqf candidate-boundary producer (``tadabur.waqf_candidates``)."""

from __future__ import annotations

import json

from tadabur import waqf_candidates as wc
from tadabur.waqf_event_fixtures import MID_WORD_CLOSURE, WAQF, WASL


def _uniform_boundaries(_surah_ayah: str) -> None:
    # Force the uniform-by-word fallback so edge times are word-count proportional.
    return None


def _seg(clip, idx, start, end, words, surah_ayah="2:2"):
    return {
        "audio_filename": f"{clip}__seg{idx}.wav",
        "segment_index": idx,
        "start_s": start,
        "end_s": end,
        "uthmani": " ".join(f"w{i}" for i in range(words)),
        "surah_ayah": surah_ayah,
    }


def test_clip_base_strips_segment_suffix():
    assert wc.clip_base("tadabur_spk0000_S10_A38_x_000025__seg3.wav") == \
        "tadabur_spk0000_S10_A38_x_000025.wav"
    assert wc.clip_base("clip.wav") == "clip.wav"


def test_read_segments_assigns_contiguous_word_ranges(tmp_path):
    path = tmp_path / "segment_manifest.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        # Written out of order to check sort-by-index.
        f.write(json.dumps(_seg("clip_spk0001", 1, 10.6, 21.8, 9)) + "\n")
        f.write(json.dumps(_seg("clip_spk0001", 0, 0.0, 8.9, 10)) + "\n")
    segments = wc.read_segments(path)["clip_spk0001.wav"]
    assert [(s.segment_index, s.word_start, s.word_end) for s in segments] == \
        [(0, 0, 10), (1, 10, 19)]


def test_waqf_candidate_per_interior_segment_boundary():
    segs = [
        wc.Segment("c_spk0001.wav", "2:2", 0, 0.0, 8.9, 10, 0, 10),
        wc.Segment("c_spk0001.wav", "2:2", 1, 10.6, 21.8, 9, 10, 19),
    ]
    cands = wc.clip_candidates(segs, pauses=[], word_boundaries=_uniform_boundaries)
    waqfs = [c for c in cands if c.predicted == WAQF]
    assert len(waqfs) == 1
    w = waqfs[0]
    # Falls after the last word of segment 0 (word index 9), spanning the inter-seg gap.
    assert (w.word_index, w.start_s, w.end_s) == (9, 8.9, 10.6)


def test_wasl_candidate_per_interior_word_edge():
    segs = [wc.Segment("c_spk0001.wav", "2:2", 0, 0.0, 4.0, 4, 0, 4)]
    cands = wc.clip_candidates(segs, pauses=[], word_boundaries=_uniform_boundaries)
    wasls = [c for c in cands if c.predicted == WASL]
    # 4 words => interior edges after words 0, 1, 2 (segment end is not a wasl edge).
    assert [c.word_index for c in wasls] == [0, 1, 2]
    # Uniform interpolation places them at 1.0, 2.0, 3.0 s.
    assert [round(c.start_s, 3) for c in wasls] == [1.0, 2.0, 3.0]
    assert all(c.start_s == c.end_s for c in wasls)


def test_mid_word_closure_from_interior_pause_only():
    segs = [
        wc.Segment("c_spk0001.wav", "2:2", 0, 0.0, 8.9, 10, 0, 10),
        wc.Segment("c_spk0001.wav", "2:2", 1, 10.6, 21.8, 9, 10, 19),
    ]
    # One boundary pause (8.9->10.6, already a split) and one interior pause inside seg1.
    pauses = [(8.9, 10.6), (20.0, 20.5)]
    cands = wc.clip_candidates(segs, pauses, word_boundaries=_uniform_boundaries)
    mids = [c for c in cands if c.predicted == MID_WORD_CLOSURE]
    assert len(mids) == 1
    m = mids[0]
    assert (m.start_s, m.end_s) == (20.0, 20.5)
    # 20.0s is inside seg1 [10.6, 21.8]; word index lands in that segment's range.
    assert 10 <= m.word_index < 19


def test_pauses_outside_segment_spans_are_ignored():
    # A lead-in pause before the (re-cut) first-segment start must not become a candidate.
    segs = [wc.Segment("c_spk0001.wav", "2:2", 0, 2.0, 6.0, 3, 0, 3)]
    pauses = [(0.5, 1.0)]  # in the trimmed lead-in, before start_s=2.0
    cands = wc.clip_candidates(segs, pauses, word_boundaries=_uniform_boundaries)
    assert not [c for c in cands if c.predicted == MID_WORD_CLOSURE]


def test_boundary_index_is_contiguous_and_time_ordered():
    segs = [
        wc.Segment("c_spk0001.wav", "2:2", 0, 0.0, 3.0, 3, 0, 3),
        wc.Segment("c_spk0001.wav", "2:2", 1, 4.0, 7.0, 3, 3, 6),
    ]
    cands = wc.clip_candidates(segs, [(5.0, 5.3)], word_boundaries=_uniform_boundaries)
    assert [c.boundary_index for c in cands] == list(range(len(cands)))
    assert [c.start_s for c in cands] == sorted(c.start_s for c in cands)


def test_phoneme_proportional_edge_times():
    # boundaries: word0 = 4 phonemes, word1 = 1 phoneme => edge at 80% of the span.
    def boundaries(_sa):
        return [0, 4, 5]
    segs = [wc.Segment("c_spk0001.wav", "2:2", 0, 0.0, 10.0, 2, 0, 2)]
    cands = wc.clip_candidates(segs, [], word_boundaries=boundaries)
    wasls = [c for c in cands if c.predicted == WASL]
    assert len(wasls) == 1
    assert round(wasls[0].start_s, 3) == 8.0


def test_build_and_write_roundtrip(tmp_path):
    segs = {
        "c_spk0001.wav": [wc.Segment("c_spk0001.wav", "2:2", 0, 0.0, 3.0, 3, 0, 3)],
    }
    cands = wc.build_candidates(segs, {}, _uniform_boundaries)
    out = tmp_path / "candidates.jsonl"
    wc.write_candidates(cands, out)
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len(cands)
    assert set(lines[0]) == {
        "clip_id", "audio_ref", "surah_ayah", "boundary_index",
        "word_index", "start_s", "end_s", "predicted",
    }
