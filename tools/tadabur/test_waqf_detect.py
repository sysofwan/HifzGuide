"""Unit tests for waqf-pause → word-boundary mapping (``tadabur.waqf_detect``).

The module is pure logic over a per-frame phoneme-id sequence, the injected VAD
``pauses``, and a spaceless phoneme ``reference`` + per-word ``boundaries``, so these
tests build synthetic ``class_ids`` (each phoneme a couple of frames; the pause a blank
gap) plus explicit pause intervals and a synthetic reference — no model, GPU, VAD, or
phonetizer is touched. Frame rate is fixed at 40ms/frame (``spf``) by choosing
``clip_duration_s = len(ids) * spf``.
"""

from __future__ import annotations

from .phoneme_vocab import PHONEME_ID_TO_CHAR
from .waqf_detect import collapse_with_times, segment_clip

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


def test_segment_clip_no_pauses_returns_whole_span():
    ids, dur = _clip(_phonemes(WORD0 + WORD1 + WORD2))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [])
    assert result.skip is None
    assert len(result.spans) == 1
    span = result.spans[0]
    assert (span.word_start, span.word_end) == (0, 3)
    assert (span.start_s, span.end_s) == (0.0, dur)


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
