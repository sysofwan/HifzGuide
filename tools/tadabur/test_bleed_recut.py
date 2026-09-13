"""Tests for the bleed re-cut — where the cut lands, and when it is thrown away.

``recut_span`` is pure arithmetic over phoneme onsets and VAD pauses, so every case
here is built by hand: a real clip would not let the pause-present and pause-absent
branches, or the two clamps, be exercised separately.
"""

from __future__ import annotations

from tadabur.bleed_detect import BleedVerdict, EdgeBleed
from tadabur.bleed_recut import (
    REASON_ACCEPTED,
    REASON_RATIO_NOT_IMPROVED,
    REASON_TRIM_GREW,
    RecutRecord,
    normalized_onsets,
    read_recut_records,
    recut_span,
    validate_recut,
    write_recut_records,
)
from tadabur.scorer import GateResult
from tadabur.waqf_detect import EDGE_RECUT_PAD_S

DURATION = 10.0
# Ten decoded phonemes, one per second. Phonemes 0-1 are a lead-in, 2-7 the ayah,
# 8-9 a trailing bleed.
ONSETS = [float(i) for i in range(10)]


def verdict(*, leading: bool = False, trailing: bool = False) -> BleedVerdict:
    return BleedVerdict(
        surah_ayah="2:2",
        leading=EdgeBleed(matched=5, span=5, query_start=0, query_end=2, detected=leading),
        trailing=EdgeBleed(matched=5, span=5, query_start=8, query_end=10, detected=trailing),
        query_length=10,
        query_covered_start=2,
        query_covered_end=8,
        ref_length=6,
        ref_covered_start=0,
        ref_covered_end=6,
    )


def test_a_clip_with_no_detected_bleed_is_left_whole():
    span = recut_span(verdict(), ONSETS, DURATION, [])

    assert (span.start_s, span.end_s) == (0.0, DURATION)
    assert not span.clipped


def test_a_leading_cut_without_a_pause_stops_a_pad_short_of_the_ayah():
    span = recut_span(verdict(leading=True), ONSETS, DURATION, [])

    assert span.start_s == ONSETS[2] - EDGE_RECUT_PAD_S
    assert span.end_s == DURATION
    assert not span.pause_anchored_leading
    assert span.clipped


def test_a_leading_cut_snaps_into_a_pause_between_the_bleed_and_the_ayah():
    span = recut_span(verdict(leading=True), ONSETS, DURATION, [(1.2, 1.8)])

    assert span.start_s == 1.5
    assert span.pause_anchored_leading


def test_a_pause_outside_the_window_is_not_snapped_to():
    """A silence inside the recitation is a waqf, not the ayah's lead-in boundary."""
    span = recut_span(verdict(leading=True), ONSETS, DURATION, [(4.2, 4.8)])

    assert span.start_s == ONSETS[2] - EDGE_RECUT_PAD_S
    assert not span.pause_anchored_leading


def test_a_trailing_cut_without_a_pause_keeps_a_pad_of_bleed():
    span = recut_span(verdict(trailing=True), ONSETS, DURATION, [])

    assert span.start_s == 0.0
    assert span.end_s == ONSETS[8] + EDGE_RECUT_PAD_S
    assert not span.pause_anchored_trailing


def test_a_trailing_cut_snaps_into_a_pause_before_the_bleed():
    span = recut_span(verdict(trailing=True), ONSETS, DURATION, [(7.3, 7.9)])

    assert span.end_s == 7.6
    assert span.pause_anchored_trailing


def test_both_edges_cut_at_once():
    span = recut_span(verdict(leading=True, trailing=True), ONSETS, DURATION, [])

    assert span.leading_clipped_s == span.start_s
    assert round(span.trailing_clipped_s, 6) == round(DURATION - span.end_s, 6)


def test_the_leading_cut_never_passes_the_ayah_s_first_phoneme():
    """The asymmetry the issue names: leaving bleed is recoverable, clipping is not."""
    tight = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    span = recut_span(verdict(leading=True), tight, DURATION, [])

    assert span.start_s == 0.0  # the pad would have gone negative; clamped, not applied


def test_the_trailing_cut_never_falls_before_the_ayah_s_last_phoneme():
    late = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.99, 9.99, 9.99]

    span = recut_span(verdict(trailing=True), late, DURATION, [])

    assert span.end_s >= late[7]


def test_a_degenerate_span_is_refused_rather_than_shipped_as_a_sliver():
    """Both edges claiming the same audio leaves nothing to trust; keep the clip whole."""
    collapsed = [0.0] * 10

    span = recut_span(verdict(leading=True, trailing=True), collapsed, DURATION, [])

    assert (span.start_s, span.end_s) == (0.0, DURATION)
    assert not span.clipped


def test_an_empty_decode_leaves_the_clip_whole():
    span = recut_span(verdict(leading=True), [], DURATION, [])

    assert (span.start_s, span.end_s) == (0.0, DURATION)


def test_normalized_onsets_take_the_first_raw_character_of_each_group():
    decode = "بَااا"  # one consonant cluster, then a madd run collapsing to one phoneme
    times = [0.0, 0.1, 0.2, 0.3, 0.4]

    onsets = normalized_onsets(decode, times)

    assert onsets == [0.0, 0.2]


def _gate(ratio: float, leading: int = 0, trailing: int = 0) -> GateResult:
    return GateResult(
        passed=False,
        match_ratio=ratio,
        max_insertion_run=0,
        leading_trim=leading,
        trailing_trim=trailing,
    )


def test_a_recut_is_kept_when_the_ratio_rises_and_no_trim_grows():
    validation = validate_recut(_gate(0.55, 13, 18), _gate(0.68, 0, 0))

    assert validation.accepted
    assert validation.reason == REASON_ACCEPTED


def test_a_recut_that_does_not_raise_the_ratio_is_discarded():
    validation = validate_recut(_gate(0.55), _gate(0.55))

    assert not validation.accepted
    assert validation.reason == REASON_RATIO_NOT_IMPROVED


def test_a_recut_that_grows_a_trim_is_discarded_even_if_the_ratio_rose():
    """A cut landing inside the recitation shows up as the aligner trimming a new edge."""
    validation = validate_recut(_gate(0.55, 0, 0), _gate(0.70, 4, 0))

    assert not validation.accepted
    assert validation.reason == REASON_TRIM_GREW


def test_recut_records_round_trip_sorted_by_clip(tmp_path):
    records = [
        RecutRecord(
            audio_filename=name,
            surah_ayah="2:2",
            reciter_id=1,
            duration_s=10.0,
            recitation_start_s=1.0,
            recitation_end_s=9.0,
            leading_bleed=True,
            trailing_bleed=True,
            accepted=True,
            reason=REASON_ACCEPTED,
            uncovered_head=0,
            uncovered_tail=0,
        )
        for name in ("b.wav", "a.wav")
    ]
    path = tmp_path / "recuts.jsonl"

    write_recut_records(path, records)
    loaded = read_recut_records(path)

    assert [r.audio_filename for r in loaded] == ["a.wav", "b.wav"]
    assert loaded[0].recitation_start_s == 1.0
