"""Unit tests for model-driven waqf detection (``tadabur.waqf_detect``).

The module is pure logic over a per-frame phoneme-id sequence plus a spaceless
phoneme ``reference`` and per-word ``boundaries``, so these tests build synthetic
``class_ids`` (each phoneme a couple of frames; a pause a long blank run) and a
synthetic reference — no model, GPU, or phonetizer is touched. Frame rate is fixed at
40ms/frame (``spf``) by choosing ``clip_duration_s = len(ids) * spf``.
"""

from __future__ import annotations

from .phoneme_vocab import PHONEME_ID_TO_CHAR
from .waqf_detect import (
    collapse_with_times,
    find_blank_runs,
    segment_clip,
)

SPF = 0.04  # seconds per frame (matches the model's ~40ms logit frames)
PH = 2  # frames per synthetic phoneme
PAUSE = 12  # blank frames for a waqf (>= 0.35s / 0.04 = 8.75)

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


# --- find_blank_runs -------------------------------------------------------------


def test_find_blank_runs_keeps_only_runs_over_threshold():
    ids = [1, 1] + [0] * 3 + [1, 1] + [0] * 12 + [1, 1]  # short gap then a real pause
    runs = find_blank_runs(ids, SPF, min_pause_s=0.35)
    assert len(runs) == 1
    start, end = runs[0]
    assert round(start, 2) == round(7 * SPF, 2)
    assert round(end, 2) == round(19 * SPF, 2)


def test_find_blank_runs_closes_trailing_run():
    ids = [1, 1] + [0] * 12  # clip ends inside the pause
    runs = find_blank_runs(ids, SPF, min_pause_s=0.35)
    assert len(runs) == 1
    assert round(runs[0][1], 2) == round(14 * SPF, 2)


# --- collapse_with_times ---------------------------------------------------------


def test_collapse_with_times_drops_blanks_and_collapses_repeats():
    ids = [2, 2, 0, 2, 3, 0, 0, 5]
    query, times = collapse_with_times(ids, SPF)
    assert query == _chars([2, 2, 3, 5])  # ب ب ت ج (blank splits the two ب)
    assert times == [0.0, 3 * SPF, 4 * SPF, 7 * SPF]


# --- segment_clip ----------------------------------------------------------------


def test_segment_clip_no_pause_returns_whole_span():
    ids, dur = _clip(_phonemes(WORD0 + WORD1 + WORD2))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES)
    assert result.skip is None
    assert len(result.spans) == 1
    span = result.spans[0]
    assert (span.word_start, span.word_end) == (0, 3)
    assert (span.start_s, span.end_s) == (0.0, dur)


def test_segment_clip_splits_at_interior_word_boundary():
    ids, dur = _clip(
        _phonemes(WORD0) + [(0, PAUSE)] + _phonemes(WORD1 + WORD2)
    )
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES)
    assert result.skip is None
    assert len(result.spans) == 2
    first, second = result.spans
    assert (first.word_start, first.word_end) == (0, 1)
    assert (second.word_start, second.word_end) == (1, 3)
    # The split cuts the clip time at the pause, not mid-word.
    assert round(first.end_s, 2) == round(len(WORD0) * PH * SPF, 2)
    assert round(second.start_s, 2) == round((len(WORD0) * PH + PAUSE) * SPF, 2)


def test_segment_clip_rejects_mid_word_pause():
    # Pause after 4 phonemes into the 9-phoneme word1: far from any word edge.
    ids, dur = _clip(
        _phonemes(WORD0 + WORD1[:4]) + [(0, PAUSE)] + _phonemes(WORD1[4:] + WORD2)
    )
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES)
    assert result.skip is None
    assert len(result.spans) == 1
    assert (result.spans[0].word_start, result.spans[0].word_end) == (0, 3)


def test_segment_clip_skips_repeated_recitation():
    ids, dur = _clip(_phonemes((WORD0 + WORD1 + WORD2) * 2))  # decode ~2x reference
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES)
    assert result.skip == "repeated_recitation"
    assert result.spans == ()


def test_segment_clip_skips_low_alignment():
    ids, dur = _clip(_phonemes([19, 20, 21, 22, 23, 24, 25, 26]))  # unrelated phonemes
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES)
    assert result.skip == "low_alignment"
    assert result.spans == ()


def test_segment_clip_empty_class_ids_returns_whole_span():
    result = segment_clip([], 1.0, REFERENCE, BOUNDARIES)
    assert result.skip is None
    assert len(result.spans) == 1
    assert (result.spans[0].word_start, result.spans[0].word_end) == (0, 3)


def test_segment_clip_empty_reference_returns_whole_span():
    ids, dur = _clip(_phonemes(WORD0))
    result = segment_clip(ids, dur, "", [0])
    assert result.skip is None
    assert len(result.spans) == 1
    assert (result.spans[0].word_start, result.spans[0].word_end) == (0, 0)

