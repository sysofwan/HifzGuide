"""Unit tests for waqf-pause → word-boundary mapping (``tadabur.waqf_detect``).

The module is pure logic over a per-frame phoneme-id sequence, the injected VAD
``pauses``, and a spaceless phoneme ``reference`` + per-word ``boundaries``, so these
tests build synthetic ``class_ids`` (each phoneme a couple of frames; the pause a blank
gap) plus explicit pause intervals and a synthetic reference — no model, GPU, VAD, or
phonetizer is touched. Frame rate is fixed at 40ms/frame (``spf``) by choosing
``clip_duration_s = len(ids) * spf``.
"""

from __future__ import annotations

import pytest

from .phoneme_vocab import PHONEME_ID_TO_CHAR
from .waqf_detect import EDGE_RECUT_PAD_S, collapse_with_times, segment_clip

SPF = 0.04  # seconds per frame (matches the model's ~40ms logit frames)
PH = 2  # frames per synthetic phoneme
PAUSE = 12  # blank frames spanning the injected pause

# Three Uthmani "words" with distinct phoneme ids so the greedy decode never collapses
# adjacent duplicates: word0 -> بتثجح (5), word1 -> خدذرزسشصض (9), word2 -> طظع (3).
WORD0 = [2, 3, 4, 5, 6]
WORD1 = [7, 8, 9, 10, 11, 12, 13, 14, 15]
WORD2 = [16, 17, 18]


def _chars(ids: list[int]) -> str:
    return "".join(PHONEME_ID_TO_CHAR[i] for i in ids)


# The whole-ayah spaceless reference + per-word boundaries the phonetizer layer would
# hand segment_clip (see tadabur.waqf_segments.hafs_word_reference).
REFERENCE = _chars(WORD0 + WORD1 + WORD2)
BOUNDARIES = [0, 5, 14, 17]


def _clip(spec: list[tuple[int, int]]) -> tuple[list[int], float]:
    """Expand ``(id, count)`` runs into class_ids and the matching clip duration."""
    ids: list[int] = []
    for cid, count in spec:
        ids += [cid] * count
    return ids, len(ids) * SPF


def _phonemes(ids: list[int]) -> list[tuple[int, int]]:
    return [(cid, PH) for cid in ids]


def _pause_after(n_phonemes: int) -> tuple[float, float]:
    """The ``(start_s, end_s)`` of the blank gap following ``n_phonemes`` phonemes."""
    start_frame = n_phonemes * PH
    return (start_frame * SPF, (start_frame + PAUSE) * SPF)


# --- collapse_with_times ---------------------------------------------------------


def test_collapse_with_times_drops_blanks_and_collapses_repeats():
    ids = [2, 2, 0, 2, 3, 0, 0, 5]
    query, times = collapse_with_times(ids, SPF)
    assert query == _chars([2, 2, 3, 5])  # ب ب ت ج (blank splits the two ب)
    assert times == [0.0, 3 * SPF, 4 * SPF, 7 * SPF]


# --- segment_clip ----------------------------------------------------------------


def test_segment_clip_no_pauses_recuts_outer_edges_to_matched_span():
    # A clean single-segment clip (matched span == whole decode): the start clamps to 0.0
    # and the end is re-cut to the last aligned phoneme's onset + outward pad (< dur,
    # because the last decoded phoneme onset precedes the clip end).
    ids, dur = _clip(_phonemes(WORD0 + WORD1 + WORD2))
    _, times = collapse_with_times(ids, SPF)
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [])
    assert result.skip is None
    assert len(result.spans) == 1
    span = result.spans[0]
    assert (span.word_start, span.word_end) == (0, 3)
    assert span.start_s == 0.0
    assert span.end_s == pytest.approx(times[-1] + EDGE_RECUT_PAD_S)
    assert span.end_s < dur


def test_segment_clip_recuts_leading_bleed():
    # Prepend a prev-ayah tail (phonemes absent from this ayah's reference): the local
    # alignment trims it, so start_s re-cuts to the first matched phoneme's onset (minus
    # pad), not the clip start 0.0.
    lead = [19, 20, 21]  # غ ف ق — not in WORD0/WORD1/WORD2
    ids, dur = _clip(_phonemes(lead + WORD0 + WORD1 + WORD2))
    _, times = collapse_with_times(ids, SPF)
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [])
    assert result.skip is None
    span = result.spans[0]
    assert (span.word_start, span.word_end) == (0, 3)
    assert span.start_s == pytest.approx(times[len(lead)] - EDGE_RECUT_PAD_S)
    assert span.start_s > 0.0


def test_segment_clip_recuts_trailing_bleed():
    # Append a trailing next-word / takbir bleed: the local alignment trims it, so end_s
    # re-cuts to the last matched phoneme's onset (plus pad), not the clip end.
    trail = [19, 20, 21]  # غ ف ق — not in the reference
    matched = WORD0 + WORD1 + WORD2
    ids, dur = _clip(_phonemes(matched + trail))
    _, times = collapse_with_times(ids, SPF)
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [])
    assert result.skip is None
    span = result.spans[0]
    assert (span.word_start, span.word_end) == (0, 3)
    assert span.end_s == pytest.approx(times[len(matched) - 1] + EDGE_RECUT_PAD_S)
    assert span.end_s < dur


def test_segment_clip_recut_pads_outward_and_clamps_to_clip():
    # One frame per phoneme so the outward pad exceeds both clip edges: start clamps to
    # 0.0 and end clamps to the clip duration (never outside [0, dur]).
    ids = list(WORD0)  # 5 phonemes, 1 frame each
    dur = len(ids) * SPF
    ref = _chars(WORD0)
    result = segment_clip(ids, dur, ref, [0, 5], [])
    span = result.spans[0]
    assert span.start_s == 0.0
    assert span.end_s == dur


def test_segment_clip_recut_leaves_interior_boundaries_untouched():
    # Leading bleed + an interior waqf pause: the outer edges re-cut, but the interior
    # boundary stays exactly at the pause times.
    lead = [19, 20]
    ids, dur = _clip(
        _phonemes(lead + WORD0) + [(0, PAUSE)] + _phonemes(WORD1 + WORD2)
    )
    _, times = collapse_with_times(ids, SPF)
    pause = _pause_after(len(lead) + len(WORD0))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [pause])
    assert result.skip is None
    assert len(result.spans) == 2
    first, second = result.spans
    assert (first.word_start, first.word_end) == (0, 1)
    assert (second.word_start, second.word_end) == (1, 3)
    # Outer edges re-cut past the lead-in and before the clip end.
    assert first.start_s == pytest.approx(times[len(lead)] - EDGE_RECUT_PAD_S)
    assert first.start_s > 0.0
    assert second.end_s == pytest.approx(times[-1] + EDGE_RECUT_PAD_S)
    assert second.end_s < dur
    # Interior boundary untouched: it sits exactly at the pause.
    assert first.end_s == pytest.approx(pause[0])
    assert second.start_s == pytest.approx(pause[1])


def test_segment_clip_flags_repeated_even_without_pauses():
    # A repeated recitation must be flagged (not silently kept whole) even when the VAD
    # found no interior pause — the safeguards run before the no-pause short-circuit.
    ids, dur = _clip(_phonemes((WORD0 + WORD1 + WORD2) * 2))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [])
    assert result.skip == "repeated_recitation"
    assert result.spans == ()


def test_segment_clip_flags_low_alignment_even_without_pauses():
    ids, dur = _clip(_phonemes([19, 20, 21, 22, 23, 24, 25, 26]))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [])
    assert result.skip == "low_alignment"
    assert result.spans == ()


def test_segment_clip_splits_at_interior_word_boundary():
    ids, dur = _clip(
        _phonemes(WORD0) + [(0, PAUSE)] + _phonemes(WORD1 + WORD2)
    )
    pause = _pause_after(len(WORD0))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [pause])
    assert result.skip is None
    assert len(result.spans) == 2
    first, second = result.spans
    assert (first.word_start, first.word_end) == (0, 1)
    assert (second.word_start, second.word_end) == (1, 3)
    # The split cuts the clip time at the pause, not mid-word.
    assert round(first.end_s, 2) == round(pause[0], 2)
    assert round(second.start_s, 2) == round(pause[1], 2)


def test_segment_clip_rejects_mid_word_pause():
    # Pause after 4 phonemes into the 9-phoneme word1: far from any word edge.
    ids, dur = _clip(
        _phonemes(WORD0 + WORD1[:4]) + [(0, PAUSE)] + _phonemes(WORD1[4:] + WORD2)
    )
    pause = _pause_after(len(WORD0) + 4)
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [pause])
    assert result.skip is None
    assert len(result.spans) == 1
    assert (result.spans[0].word_start, result.spans[0].word_end) == (0, 3)


def test_segment_clip_skips_repeated_recitation():
    ids, dur = _clip(_phonemes((WORD0 + WORD1 + WORD2) * 2))  # decode ~2x reference
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [_pause_after(len(WORD0))])
    assert result.skip == "repeated_recitation"
    assert result.spans == ()


def test_segment_clip_skips_low_alignment():
    ids, dur = _clip(_phonemes([19, 20, 21, 22, 23, 24, 25, 26]))  # unrelated phonemes
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [_pause_after(2)])
    assert result.skip == "low_alignment"
    assert result.spans == ()


def test_segment_clip_empty_class_ids_returns_whole_span():
    result = segment_clip([], 1.0, REFERENCE, BOUNDARIES, [_pause_after(1)])
    assert result.skip is None
    assert len(result.spans) == 1
    assert (result.spans[0].word_start, result.spans[0].word_end) == (0, 3)


def test_segment_clip_empty_reference_returns_whole_span():
    ids, dur = _clip(_phonemes(WORD0))
    result = segment_clip(ids, dur, "", [0], [_pause_after(1)])
    assert result.skip is None
    assert len(result.spans) == 1
    assert (result.spans[0].word_start, result.spans[0].word_end) == (0, 0)
