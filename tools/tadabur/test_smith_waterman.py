"""Parity tests for Smith-Waterman alignment, ported from Muraja's
SmithWatermanTests.swift. Scores must match the Swift `.balanced` scorer."""

import pytest

from tadabur.smith_waterman import (
    GAP,
    MATCH,
    MISMATCH,
    RefMatchInfo,
    local_alignment_score,
    smith_waterman,
    tashkeel,
)


# MARK: - smith_waterman basic


def test_empty_query():
    result = smith_waterman(query="", reference="ابت")
    assert result.score == 0
    assert result.ref_matches == []


def test_empty_reference():
    result = smith_waterman(query="ابت", reference="")
    assert result.score == 0
    assert result.ref_matches == []


def test_both_empty():
    assert smith_waterman(query="", reference="").score == 0


def test_perfect_match():
    text = "ابتث"
    result = smith_waterman(query=text, reference=text)
    assert result.score == pytest.approx(len(text), abs=1e-3)
    assert all(info == MATCH for info in result.ref_matches)


def test_substring_alignment():
    result = smith_waterman(query="بت", reference="ابتثج")
    assert result.score == pytest.approx(2.0, abs=1e-3)
    assert result.ref_start == 1
    assert result.ref_end == 3


def test_mismatch_penalty():
    result = smith_waterman(query="ص", reference="ب")
    assert result.score >= 0


def test_gap_handling():
    result = smith_waterman(query="بث", reference="بتث")
    assert result.score > 0
    assert result.ref_end - result.ref_start >= 2


def test_ref_match_info_types():
    result = smith_waterman(query="بص", reference="بت")
    if result.ref_matches:
        assert result.ref_matches[0] == MATCH


def test_ref_to_query_mapping():
    result = smith_waterman(query="ابت", reference="ابت")
    assert len(result.ref_to_query) == result.ref_end - result.ref_start
    for i, q_idx in enumerate(result.ref_to_query):
        assert q_idx == result.query_start + i


def test_local_alignment():
    result = smith_waterman(query="جحخ", reference="ابتثجحخدذ")
    assert result.score == pytest.approx(3.0, abs=1e-3)
    assert result.ref_start == 4
    assert result.ref_end == 7


# MARK: - local_alignment_score


def test_score_only_empty():
    assert local_alignment_score(query="", reference="ابت") == 0
    assert local_alignment_score(query="ابت", reference="") == 0
    assert local_alignment_score(query="", reference="") == 0


def test_score_only_perfect_match():
    text = "ابتث"
    assert local_alignment_score(query=text, reference=text) == pytest.approx(len(text), abs=1e-3)


def test_score_only_substring():
    assert local_alignment_score(query="بت", reference="ابتثج") == pytest.approx(2.0, abs=1e-3)


def test_score_only_consistent_with_full():
    query, reference = "بتثج", "ابتثجحخدذ"
    full = smith_waterman(query=query, reference=reference).score
    score_only = local_alignment_score(query=query, reference=reference)
    assert full == pytest.approx(score_only, abs=1e-3)


def test_score_only_consistent_with_full_mismatch():
    query, reference = "بصث", "بتث"
    full = smith_waterman(query=query, reference=reference).score
    score_only = local_alignment_score(query=query, reference=reference)
    assert full == pytest.approx(score_only, abs=1e-3)


def test_score_only_consistent_with_gap():
    query, reference = "بث", "بتث"
    full = smith_waterman(query=query, reference=reference).score
    score_only = local_alignment_score(query=query, reference=reference)
    assert full == pytest.approx(score_only, abs=1e-3)


# MARK: - RefMatchInfo


def test_ref_match_info_equatable():
    assert MATCH == RefMatchInfo("match")
    assert GAP == RefMatchInfo("gap")
    assert MISMATCH == RefMatchInfo("mismatch")
    assert tashkeel("\u064E", "\u064F") == tashkeel("\u064E", "\u064F")
    assert MATCH != GAP
    assert MATCH != MISMATCH


def test_space_space_scores_zero():
    query, ref = "ب ت", "ب ت"
    assert local_alignment_score(query=query, reference=ref) == pytest.approx(2.0, abs=1e-2)
    assert smith_waterman(query=query, reference=ref).score == pytest.approx(2.0, abs=1e-2)


def test_space_only_input_scores_zero():
    assert local_alignment_score(query="   ", reference="   ") == pytest.approx(0.0, abs=1e-2)
