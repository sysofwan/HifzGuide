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
from .waqf_detect import (
    EDGE_RECUT_PAD_S,
    MID_WORD_CLOSURE,
    RE_READ,
    UNPLACED,
    WAQF,
    _ChunkAlign,
    _supported_end,
    _word_supported,
    collapse_with_times,
    segment_clip,
)

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


def test_segment_clip_recuts_pure_leadin_chunk_before_a_pause():
    # Regression (thermonuclear review, Finding 3): a pause can fall *before* this ayah's
    # first word, so the first VAD chunk is pure neighbour-ayah lead-in that aligns to
    # nothing. The outer re-cut must use the whole-clip alignment's matched span (not that
    # unreliable first chunk's own onset), so start_s still trims past the lead-in.
    lead = [19, 20, 21]  # غ ف ق — not in WORD0/WORD1/WORD2
    ids, dur = _clip(
        _phonemes(lead) + [(0, PAUSE)] + _phonemes(WORD0 + WORD1 + WORD2)
    )
    _, times = collapse_with_times(ids, SPF)
    pause_before = _pause_after(len(lead))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [pause_before])
    assert result.skip is None
    # The pure lead-in chunk is unreliable, so it does not anchor a split: one segment.
    assert len(result.spans) == 1
    (span,) = result.spans
    assert (span.word_start, span.word_end) == (0, 3)
    # start_s is re-cut to WORD0's onset (past the lead-in), not left at 0.0.
    assert span.start_s == pytest.approx(times[len(lead)] - EDGE_RECUT_PAD_S)
    assert span.start_s > 0.0


def test_segment_clip_all_blank_decode_flags_low_alignment():
    # Regression (thermonuclear review, Finding 1): a non-empty class_ids that decodes to
    # all blanks (empty query) must hit the low_alignment gate, not fall through as a
    # bogus whole-ayah span.
    ids, dur = _clip([(0, 200)])  # all-blank frames
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [])
    assert result.skip == "low_alignment"
    assert result.spans == ()


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


def test_segment_clip_splits_reread_into_two_overlapping_segments():
    # WORD0 WORD1 [waqf] WORD1 WORD2: the reciter stops after word1, re-reads it, then
    # carries on. The seam is a real waqf, so the clip is cut into two time-consecutive
    # segments whose word ranges overlap on word1 ("read until the waqf" / "re-read ...end").
    ids, dur = _clip(
        _phonemes(WORD0 + WORD1) + [(0, PAUSE)] + _phonemes(WORD1 + WORD2)
    )
    pause = _pause_after(len(WORD0 + WORD1))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [pause])
    assert result.skip is None
    assert result.re_reads == 1
    assert len(result.spans) == 2
    first, second = result.spans
    assert (first.word_start, first.word_end) == (0, 2)
    assert (second.word_start, second.word_end) == (1, 3)  # overlaps word1 (the re-read)
    # First segment ends at the waqf; the second resumes after it and runs to the clip end.
    assert first.start_s == 0.0
    assert first.end_s == pytest.approx(pause[0])
    assert second.start_s == pytest.approx(pause[1])
    assert second.end_s == pytest.approx(dur, abs=EDGE_RECUT_PAD_S + 1e-6)
    # The seam is a genuine re-read; the marker sits on the STOP word (word1, the last word
    # recited before the pause), not the resume word — this is the re-read marker fix.
    assert result.pauses[0].kind == RE_READ
    assert result.pauses[0].word_index == 1


def test_segment_clip_splits_whole_ayah_repeat_at_seam_into_two_clips():
    # The reciter recites the whole ayah, pauses, and recites it again. With a pause at the
    # seam this is two clean single-pass clips, not a skip: two full-ayah spans.
    ids, dur = _clip(
        _phonemes(WORD0 + WORD1 + WORD2) + [(0, PAUSE)] + _phonemes(WORD0 + WORD1 + WORD2)
    )
    pause = _pause_after(len(WORD0 + WORD1 + WORD2))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [pause])
    assert result.skip is None
    assert result.re_reads == 1
    assert len(result.spans) == 2
    assert all((s.word_start, s.word_end) == (0, 3) for s in result.spans)


def test_segment_clip_splits_reread_of_final_word_only():
    # A partial re-read: full ayah, waqf, then the last word repeated. Second segment is the
    # re-read tail (word2 only); it overlaps the first segment's final word.
    ids, dur = _clip(
        _phonemes(WORD0 + WORD1 + WORD2) + [(0, PAUSE)] + _phonemes(WORD2)
    )
    pause = _pause_after(len(WORD0 + WORD1 + WORD2))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [pause])
    assert result.skip is None
    assert result.re_reads == 1
    assert len(result.spans) == 2
    first, second = result.spans
    assert (first.word_start, first.word_end) == (0, 3)
    assert (second.word_start, second.word_end) == (2, 3)


def test_segment_clip_final_segment_uninflates_an_early_stop():
    # The reciter stops mid-ayah after word1; the reference still has word2. The final
    # segment's extent must bound to where the decode actually reached (words 0-1), not snap
    # to the whole ayah and invent a marker for the never-recited word2.
    ids, dur = _clip(_phonemes(WORD0 + WORD1))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [])
    assert result.skip is None
    assert len(result.spans) == 1
    assert (result.spans[0].word_start, result.spans[0].word_end) == (0, 2)


def test_segment_clip_final_segment_keeps_full_ayah_despite_trailing_fragment():
    # A COMPLETE recitation (words 0-2), a pause, then a tiny unreliable trailing fragment
    # (e.g. an elongated final-word tail or a post-ayah artifact). The final segment's extent
    # must come from the last RELIABLE chunk (the whole ayah), not the unreliable trailing
    # chunk, which would otherwise collapse the completed range.
    trail = [19, 20]  # غ ف — not in the reference; sub-word, so unreliable
    ids, dur = _clip(
        _phonemes(WORD0 + WORD1 + WORD2) + [(0, PAUSE)] + _phonemes(trail)
    )
    pause = _pause_after(len(WORD0 + WORD1 + WORD2))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [pause])
    assert result.skip is None
    assert len(result.spans) == 1
    assert (result.spans[0].word_start, result.spans[0].word_end) == (0, 3)


def test_segment_clip_emits_waqf_pause_attribution_on_word_edge():
    # An interior waqf after word0: the pause is attributed to the last completed word
    # (word 0), phoneme-aligned, with kind WAQF.
    ids, dur = _clip(_phonemes(WORD0) + [(0, PAUSE)] + _phonemes(WORD1 + WORD2))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [_pause_after(len(WORD0))])
    assert len(result.pauses) == 1
    attrib = result.pauses[0]
    assert attrib.kind == WAQF
    assert attrib.word_index == 0


def test_segment_clip_emits_mid_word_closure_attribution_at_completed_word():
    # A four-word ayah; the reciter pauses inside the last (long) word after words 0-2 are
    # complete and only ~45% into word 3. The pause is far from any word edge (mid-word),
    # so it is a mid_word_closure; the pre-pause chunk's matched span snaps to the nearest
    # word edge, which (being under halfway through word 3) is the edge *after* word 2, so
    # it is attributed to word 2 — the runtime phoneme-aligned answer, independent of timing.
    word3 = list(range(19, 30))  # 11 distinct phonemes
    reference = _chars(WORD0 + WORD1 + WORD2 + word3)
    boundaries = [0, 5, 14, 17, 28]
    ids, dur = _clip(
        _phonemes(WORD0 + WORD1 + WORD2 + word3[:5])
        + [(0, PAUSE)]
        + _phonemes(word3[5:])
    )
    pause = _pause_after(len(WORD0 + WORD1 + WORD2) + 5)
    result = segment_clip(ids, dur, reference, boundaries, [pause])
    assert len(result.pauses) == 1
    attrib = result.pauses[0]
    assert attrib.kind == MID_WORD_CLOSURE
    assert attrib.word_index == 2
    # Mid-word closure does not split the clip: one segment spanning the whole ayah.
    assert len(result.spans) == 1


def test_segment_clip_mid_word_closure_attributes_completed_word_when_tail_dropped():
    # Regression (5:73, 31:23): the reciter cleanly finishes a word and pauses, then resumes
    # at the *next* word, but the CTC decode drops that finished word's elongated / hamza
    # tail, so the pre-pause chunk's matched span ends several phonemes short of its edge
    # (> boundary_tol → classified mid-word, not a clean waqf). The strict "fully covered"
    # count under-attributes to word 0, but the clean forward resume at word 2 pins the stop
    # to word 1 — the pause is attributed to the completed word 1, not word 0.
    # Decode word0 fully, then only 5 of word1's 9 phonemes (tail dropped), pause, then
    # resume cleanly at word2 (a forward waqf, no re-read).
    ids, dur = _clip(
        _phonemes(WORD0 + WORD1[:5])
        + [(0, PAUSE)]
        + _phonemes(WORD2)
    )
    pause = _pause_after(len(WORD0) + 5)
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [pause])
    assert len(result.pauses) == 1
    attrib = result.pauses[0]
    assert attrib.kind == MID_WORD_CLOSURE
    assert attrib.word_index == 1


def test_segment_clip_reread_attributes_over_run_word_not_backward_resume():
    # A re-read (spk0518, 8:47): the reciter over-runs into word 2, pauses, then backs up
    # and resumes at word 1. Both chunk edges land on word boundaries with a backward jump,
    # so it is classified RE_READ. A re-read marker takes the decode-supported stop frontier
    # (``_supported_end`` = 2 here — words 0-1 fully recited, the lone phoneme of word 2 not
    # supported), so the pause stays on word 1, the last word cleanly finished before the
    # aborted over-run, ignoring the backward resume entirely.
    ids, dur = _clip(
        _phonemes(WORD0 + WORD1 + WORD2[:1])
        + [(0, PAUSE)]
        + _phonemes(WORD1 + WORD2)
    )
    pause = _pause_after(len(WORD0 + WORD1) + 1)
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [pause])
    assert len(result.pauses) == 1
    assert result.pauses[0].kind == RE_READ
    assert result.pauses[0].word_index == 1


def test_segment_clip_mid_word_breath_does_not_over_attribute_via_resume():
    # Regression (resume-guard): the reciter pauses for breath *inside* word1 and resumes
    # the *same* unfinished word. The post-pause chunk's matched span begins in the latter
    # half of word1, so it snaps within boundary_tol of word1's END edge (= word2's start),
    # which would make an unguarded ``resume = start_word - 1`` credit word1. But the resume
    # sits to the *left* of that edge (still inside word1), so it is rejected and the floor
    # ``reached`` (word0 completed) is used — the stop is attributed to word 0, not word 1.
    ids, dur = _clip(
        _phonemes(WORD0 + WORD1[:5])
        + [(0, PAUSE)]
        + _phonemes(WORD1[6:] + WORD2)
    )
    pause = _pause_after(len(WORD0) + 5)
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [pause])
    assert len(result.pauses) == 1
    attrib = result.pauses[0]
    assert attrib.kind == MID_WORD_CLOSURE
    assert attrib.word_index == 0


def test_segment_clip_low_alignment_emits_unplaced_attributions_per_pause():
    # A clip whose decode does not match this ayah is kept whole (skip=low_alignment) but
    # still has interior VAD pauses. segment_clip must emit one UNPLACED (word_index=None)
    # attribution per pause so the sidecar lists the clip: a clip absent from the sidecar
    # then unambiguously means a stale artifact, and these pauses fall back to interpolation.
    ids, dur = _clip(_phonemes(WORD0 + WORD1))
    # Reference the decode cannot match -> low_alignment.
    unrelated_ref = _chars([19, 20, 21, 22])
    pauses = [_pause_after(3), _pause_after(len(WORD0) + 2)]
    result = segment_clip(ids, dur, unrelated_ref, [0, 4], pauses)
    assert result.skip == "low_alignment"
    assert len(result.pauses) == len(pauses)
    assert all(p.kind == UNPLACED and p.word_index is None for p in result.pauses)


def test_segment_clip_whole_clip_path_emits_unplaced_attributions():
    # The degenerate whole-clip path (empty decode) with interior pauses also emits one
    # UNPLACED attribution per pause, upholding the one-per-pause sidecar invariant.
    result = segment_clip([], 2.0, REFERENCE, BOUNDARIES, [_pause_after(1), _pause_after(3)])
    assert result.skip is None
    assert len(result.pauses) == 2
    assert all(p.kind == UNPLACED and p.word_index is None for p in result.pauses)


def test_segment_clip_forward_waqf_is_not_a_reread():
    # A plain forward waqf must not be counted as a re-read.
    ids, dur = _clip(_phonemes(WORD0) + [(0, PAUSE)] + _phonemes(WORD1 + WORD2))
    result = segment_clip(ids, dur, REFERENCE, BOUNDARIES, [_pause_after(len(WORD0))])
    assert result.skip is None
    assert result.re_reads == 0
    assert len(result.spans) == 2


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

# --- decode-support discriminator (_word_supported / _supported_end) --------------
#
# These exercise the phantom-over-read vs genuine-re-read discriminator directly on
# crafted chunk alignments. A synthetic Smith-Waterman pass is monotonic and always
# prefers the nearest match, so it cannot manufacture a phantom over-read (an end that
# snaps forward past a duplicate word) the way messy real decodes do; the unit level is
# where that branch is pinned. ``BOUNDARIES = [0, 5, 14, 17]`` gives word lengths 5/9/3.


def _chunk(
    start_word: int,
    end_word: int,
    ref_start: int,
    ref_kinds: tuple[str, ...],
    *,
    reliable: bool = True,
) -> _ChunkAlign:
    """A minimal ``_ChunkAlign`` carrying just the fields the discriminator reads."""
    return _ChunkAlign(
        q_lo=0, q_hi=0, t_lo=0.0, t_hi=0.0,
        start_word=start_word, start_dist=0, end_word=end_word, end_dist=0,
        reliable=reliable, ref_start=ref_start,
        ref_end=ref_start + len(ref_kinds), ref_kinds=ref_kinds,
    )


def test_word_supported_requires_prefix_anchored_coverage():
    # Word 0 spans ref [0, 5): a chunk that matched all five phonemes covers it.
    chunk = _chunk(0, 1, 0, ("match",) * 5)
    assert _word_supported(chunk, BOUNDARIES, 0)


def test_word_supported_tolerates_a_dropped_terminal_phoneme():
    # Word 1 spans ref [5, 14) (len 9): 8 matched + a dropped tail (gap) is still 0.89
    # coverage, prefix-anchored — a CTC tail-drop must not read as "not recited".
    chunk = _chunk(1, 2, 5, ("match",) * 8 + ("gap",))
    assert _word_supported(chunk, BOUNDARIES, 1)


def test_word_supported_rejects_a_lone_over_snapped_phoneme():
    # Word 2 spans ref [14, 17) (len 3): a matched span that only clips its first phoneme
    # (a phantom over-read snapping one phoneme into the word) is < min(2, 3) diagonal and
    # 0.33 coverage — not recited.
    chunk = _chunk(2, 3, 14, ("match", "gap", "gap"))
    assert not _word_supported(chunk, BOUNDARIES, 2)


def test_supported_end_reports_true_coverage_for_a_clean_chunk():
    # A chunk that cleanly recited words 0-1 (snapped end 2) reports end 2.
    chunk = _chunk(0, 2, 0, ("match",) * 14)
    assert _supported_end(chunk, BOUNDARIES, n_words=3) == 2


def test_supported_end_truncates_a_phantom_over_read_at_the_support_gap():
    # A chunk whose matched span over-snapped to end_word 3 but only truly recited words 0-1
    # (word 2's ref [14, 17) is all gaps) reports end 2 — the word the reciter actually
    # reached — so the segment is un-inflated to a contiguous waqf, not a phantom re-read.
    chunk = _chunk(0, 3, 0, ("match",) * 14 + ("gap",) * 3)
    assert _supported_end(chunk, BOUNDARIES, n_words=3) == 2


def test_supported_end_repairs_a_tail_dropped_final_word():
    # A chunk that recited word 1 but tail-dropped its last phoneme snapped its end to 1;
    # the inclusive scan still credits word 1 (0.89 coverage), reporting end 2.
    chunk = _chunk(0, 1, 0, ("match",) * 5 + ("match",) * 8 + ("gap",))
    assert _supported_end(chunk, BOUNDARIES, n_words=3) == 2


def test_supported_end_trusts_the_snap_when_nothing_is_supported():
    # A fully unreliable decode (no diagonal support anywhere) must not collapse the segment
    # to zero words: the snapped end_word is trusted instead.
    chunk = _chunk(0, 2, 0, ("gap",) * 14)
    assert _supported_end(chunk, BOUNDARIES, n_words=3) == 2


def test_supported_end_bridges_a_single_word_mid_recitation_dropout():
    # earlier recited words 0-3 contiguously but the decode dropped word 1 (a single-word CTC
    # omission); a bare contiguity break would truncate at word 0 and lose the rest of a real
    # recitation. A one-word gap is bridged, so the end reaches word 3 (end 4).
    boundaries = [0, 5, 14, 17, 22]  # 4 words
    chunk = _chunk(0, 4, 0, ("match",) * 5 + ("gap",) * 9 + ("match",) * 3 + ("match",) * 5)
    assert _supported_end(chunk, boundaries, n_words=4) == 4


def test_supported_end_stops_at_a_multi_word_gap_before_an_isolated_duplicate():
    # earlier recited word 0, then a TWO-word gap (words 1-2 unsupported), then an isolated
    # duplicate at word 3 (the phantom over-read's far snap). A gap longer than one word is
    # not bridged, so the end stays at word 1 — the phantom does not drag coverage forward.
    boundaries = [0, 5, 14, 17, 22]  # 4 words
    chunk = _chunk(0, 4, 0, ("match",) * 5 + ("gap",) * 12 + ("match",) * 5)
    assert _supported_end(chunk, boundaries, n_words=4) == 1


# --- per-word onset times ------------------------------------------------------


def test_word_onset_times_places_each_word_at_its_first_aligned_phoneme():
    from tadabur.smith_waterman import smith_waterman
    from tadabur.waqf_detect import word_onset_times

    reference = "ءبتثجح"
    alignment = smith_waterman(reference, reference)
    decode_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    times = word_onset_times(alignment, decode_times, [0, 2, 4, 6], 0.0, 1.2)

    assert times == (0.0, 0.4, 0.8, 1.2)


def test_word_onset_times_are_monotone_and_inside_the_recitation():
    from tadabur.smith_waterman import smith_waterman
    from tadabur.waqf_detect import word_onset_times

    reference = "ءبتثجح"
    alignment = smith_waterman(reference, reference)
    decode_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    times = word_onset_times(alignment, decode_times, [0, 2, 4, 6], 0.5, 0.9)

    assert list(times) == sorted(times)
    assert all(0.5 <= t <= 0.9 for t in times)


def test_word_onset_times_empty_without_an_alignment():
    from tadabur.waqf_detect import word_onset_times

    class _Empty:
        ref_start = 0
        ref_end = 0
        ref_to_query: list[int] = []

    assert word_onset_times(_Empty(), [0.0], [0, 1], 0.0, 1.0) == ()
