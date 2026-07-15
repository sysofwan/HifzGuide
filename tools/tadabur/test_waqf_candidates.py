"""Unit tests for the waqf candidate-boundary producer (``tadabur.waqf_candidates``)."""

from __future__ import annotations

import json

import pytest

from tadabur import waqf_candidates as wc
from tadabur.waqf_event_fixtures import MID_WORD_CLOSURE, WAQF, WASL


def _uniform_boundaries(_surah_ayah: str) -> None:
    # Force the uniform-by-word fallback so edge times are word-count proportional.
    return None


def _seg(clip, idx, start, end, words, surah_ayah="2:2", word_start=0):
    return {
        "audio_filename": f"{clip}__seg{idx}.wav",
        "segment_index": idx,
        "start_s": start,
        "end_s": end,
        "uthmani": " ".join(f"w{i}" for i in range(words)),
        "surah_ayah": surah_ayah,
        "word_start": word_start,
        "word_end": word_start + words,
    }


def test_clip_base_strips_segment_suffix():
    assert wc.clip_base("tadabur_spk0000_S10_A38_x_000025__seg3.wav") == \
        "tadabur_spk0000_S10_A38_x_000025.wav"
    assert wc.clip_base("clip.wav") == "clip.wav"


def test_read_segments_uses_explicit_word_ranges(tmp_path):
    path = tmp_path / "segment_manifest.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        # Written out of order to check sort-by-index.
        f.write(json.dumps(_seg("clip_spk0001", 1, 10.6, 21.8, 9, word_start=10)) + "\n")
        f.write(json.dumps(_seg("clip_spk0001", 0, 0.0, 8.9, 10, word_start=0)) + "\n")
    segments = wc.read_segments(path)["clip_spk0001.wav"]
    assert [(s.segment_index, s.word_start, s.word_end) for s in segments] == \
        [(0, 0, 10), (1, 10, 19)]


def test_read_segments_honours_overlapping_reread_ranges(tmp_path):
    # A re-read clip: segment 1 re-covers words segment 0 already read (overlap on word 1),
    # so the explicit ranges must be trusted verbatim, not re-derived contiguously.
    path = tmp_path / "segment_manifest.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_seg("clip_spk0001", 0, 0.0, 8.9, 2, word_start=0)) + "\n")
        f.write(json.dumps(_seg("clip_spk0001", 1, 9.2, 15.0, 2, word_start=1)) + "\n")
    segments = wc.read_segments(path)["clip_spk0001.wav"]
    assert [(s.segment_index, s.word_start, s.word_end) for s in segments] == \
        [(0, 0, 2), (1, 1, 3)]


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


def test_waqf_candidate_anchors_to_next_segment_on_reread_overlap():
    # Re-read: segment 1 re-covers word 3 that segment 0 already read (overlap). The
    # waqf marks the word right *before* the re-read resumes (nxt.word_start - 1 = 2),
    # not the inflated last word of segment 0 (prev.word_end - 1 = 3).
    segs = [
        wc.Segment("c_spk0001.wav", "2:2", 0, 0.0, 4.49, 4, 0, 4),
        wc.Segment("c_spk0001.wav", "2:2", 1, 5.23, 14.64, 8, 3, 11),
    ]
    cands = wc.clip_candidates(segs, pauses=[], word_boundaries=_uniform_boundaries)
    waqfs = [c for c in cands if c.predicted == WAQF]
    assert len(waqfs) == 1
    assert (waqfs[0].word_index, waqfs[0].start_s, waqfs[0].end_s) == (2, 4.49, 5.23)


def test_waqf_candidate_anchors_to_next_segment_on_coverage_gap():
    # Coverage gap: word 8 is claimed by neither segment (a dropped duplicate). The
    # waqf marks the last word before segment 1 resumes (nxt.word_start - 1 = 8),
    # not segment 0's short word_end (prev.word_end - 1 = 7).
    segs = [
        wc.Segment("c_spk0001.wav", "2:2", 0, 0.0, 8.13, 8, 0, 8),
        wc.Segment("c_spk0001.wav", "2:2", 1, 9.27, 12.36, 3, 9, 12),
    ]
    cands = wc.clip_candidates(segs, pauses=[], word_boundaries=_uniform_boundaries)
    waqfs = [c for c in cands if c.predicted == WAQF]
    assert len(waqfs) == 1
    assert (waqfs[0].word_index, waqfs[0].start_s, waqfs[0].end_s) == (8, 8.13, 9.27)


def test_waqf_candidate_full_restart_floors_at_prev_word_start():
    # A degenerate false start (segment 0) followed by a full re-read from word 0
    # (nxt.word_start == 0) must not produce a negative word_index; it floors at
    # prev.word_start.
    segs = [
        wc.Segment("c_spk0001.wav", "2:2", 0, 5.47, 6.25, 8, 0, 8),
        wc.Segment("c_spk0001.wav", "2:2", 1, 6.53, 19.71, 14, 0, 14),
    ]
    cands = wc.clip_candidates(segs, pauses=[], word_boundaries=_uniform_boundaries)
    waqfs = [c for c in cands if c.predicted == WAQF]
    assert len(waqfs) == 1
    assert waqfs[0].word_index == 0



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


def test_mid_word_closure_marks_last_completed_word_not_next():
    # 9 words uniformly across seg1 [10.0, 19.0] => 1.0s each; word k spans
    # [10+k, 11+k). A pause at 13.05 falls just after word 12 completes (edge 13.0),
    # so it must mark word 12 (the completed word), not word 13 (whose inflated span
    # the interpolation puts 13.05 inside).
    segs = [wc.Segment("c_spk0001.wav", "2:2", 1, 10.0, 19.0, 9, 10, 19)]
    cands = wc.clip_candidates(segs, [(13.05, 13.5)], word_boundaries=_uniform_boundaries)
    mids = [c for c in cands if c.predicted == MID_WORD_CLOSURE]
    assert len(mids) == 1
    assert mids[0].word_index == 12


def test_mid_word_closure_inside_opening_word_floors_at_word_start():
    # A pause inside the segment's first word (before any word edge) floors at
    # word_start rather than drifting to the last word.
    segs = [wc.Segment("c_spk0001.wav", "2:2", 1, 10.0, 19.0, 9, 10, 19)]
    cands = wc.clip_candidates(segs, [(10.3, 10.6)], word_boundaries=_uniform_boundaries)
    mids = [c for c in cands if c.predicted == MID_WORD_CLOSURE]
    assert len(mids) == 1
    assert mids[0].word_index == 10


def test_pauses_outside_segment_spans_are_ignored():
    # A lead-in pause before the (re-cut) first-segment start must not become a candidate.
    segs = [wc.Segment("c_spk0001.wav", "2:2", 0, 2.0, 6.0, 3, 0, 3)]
    pauses = [(0.5, 1.0)]  # in the trimmed lead-in, before start_s=2.0
    cands = wc.clip_candidates(segs, pauses, word_boundaries=_uniform_boundaries)
    assert not [c for c in cands if c.predicted == MID_WORD_CLOSURE]


def test_mid_word_closure_prefers_phoneme_aligned_attribution():
    # Time interpolation would mark word 12 (last completed at 13.05, see the test
    # above), but the phoneme-aligned sidecar says the decode had reached word 15 when
    # the pause fell. The sidecar word wins verbatim.
    segs = [wc.Segment("c_spk0001.wav", "2:2", 1, 10.0, 19.0, 9, 10, 19)]
    attributions = {wc._onset_key(13.05): 15}
    cands = wc.clip_candidates(
        segs, [(13.05, 13.5)], word_boundaries=_uniform_boundaries,
        pause_attributions=attributions,
    )
    mids = [c for c in cands if c.predicted == MID_WORD_CLOSURE]
    assert len(mids) == 1
    assert mids[0].word_index == 15


def test_mid_word_closure_falls_back_without_sidecar():
    # No sidecar supplied at all (None): pure legacy interpolation to the last completed
    # word (word 12), no raise.
    segs = [wc.Segment("c_spk0001.wav", "2:2", 1, 10.0, 19.0, 9, 10, 19)]
    cands = wc.clip_candidates(
        segs, [(13.05, 13.5)], word_boundaries=_uniform_boundaries,
        pause_attributions=None,
    )
    mids = [c for c in cands if c.predicted == MID_WORD_CLOSURE]
    assert len(mids) == 1
    assert mids[0].word_index == 12


def test_load_pause_attributions_indexes_all_pauses_including_null(tmp_path):
    path = tmp_path / "pause_attrib.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "clip_audio_filename": "c_spk0001.wav",
            "pauses": [
                {"start_s": 13.05, "end_s": 13.5, "kind": MID_WORD_CLOSURE,
                 "word_index": 15},
                {"start_s": 6.0, "end_s": 6.4, "kind": MID_WORD_CLOSURE,
                 "word_index": None},  # explicitly unplaceable -> kept as None
                {"start_s": 8.9, "end_s": 10.6, "kind": WAQF, "word_index": 9},
            ],
        }) + "\n")
        f.write("\n")  # blank line tolerated
    attributions = wc.load_pause_attributions(path)
    assert attributions == {"c_spk0001.wav": {
        wc._onset_key(13.05): 15,
        wc._onset_key(6.0): None,
        wc._onset_key(8.9): 9,
    }}


def test_mid_word_closure_raises_on_missing_attribution():
    # A sidecar is supplied for the clip but does not mention this pause's onset — a stale
    # or mismatched artifact. Rather than silently interpolating, that must raise.
    segs = [wc.Segment("c_spk0001.wav", "2:2", 1, 10.0, 19.0, 9, 10, 19)]
    attributions = {wc._onset_key(99.0): 3}  # unrelated onset
    with pytest.raises(ValueError, match="stale or"):
        wc.clip_candidates(
            segs, [(13.05, 13.5)], word_boundaries=_uniform_boundaries,
            pause_attributions=attributions,
        )


def test_build_candidates_raises_when_sidecar_missing_a_whole_clip():
    # A sidecar was loaded (phoneme-aligned mode) but this clip is wholly absent from it,
    # while it still has an interior mid-word pause. segment_clip always emits one
    # attribution per pause, so an absent clip with pauses means a stale / mismatched
    # artifact — it must raise, not silently interpolate.
    segs = {"c_spk0001.wav": [wc.Segment("c_spk0001.wav", "2:2", 1, 10.0, 19.0, 9, 10, 19)]}
    pauses = {"c_spk0001.wav": [(13.05, 13.5)]}
    other_clip_sidecar = {"c_spk9999.wav": {wc._onset_key(1.0): 3}}
    with pytest.raises(ValueError, match="stale or"):
        wc.build_candidates(
            segs, pauses, _uniform_boundaries,
            pause_attributions_by_clip=other_clip_sidecar,
        )


def test_build_candidates_pause_free_clip_absent_from_sidecar_is_fine():
    # A clip with no interior pause is legitimately absent from the sidecar (nothing to
    # attribute); phoneme-aligned mode must not raise for it.
    segs = {"c_spk0001.wav": [wc.Segment("c_spk0001.wav", "2:2", 0, 0.0, 3.0, 3, 0, 3)]}
    cands = wc.build_candidates(
        segs, {}, _uniform_boundaries,
        pause_attributions_by_clip={"c_spk9999.wav": {wc._onset_key(1.0): 3}},
    )
    assert not [c for c in cands if c.predicted == MID_WORD_CLOSURE]


def test_mid_word_closure_falls_back_on_explicit_null_attribution():
    # An explicitly unplaceable pause (None) falls back to the last-completed-word
    # interpolation (word 12), without raising.
    segs = [wc.Segment("c_spk0001.wav", "2:2", 1, 10.0, 19.0, 9, 10, 19)]
    attributions = {wc._onset_key(13.05): None}
    cands = wc.clip_candidates(
        segs, [(13.05, 13.5)], word_boundaries=_uniform_boundaries,
        pause_attributions=attributions,
    )
    mids = [c for c in cands if c.predicted == MID_WORD_CLOSURE]
    assert len(mids) == 1
    assert mids[0].word_index == 12


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
