"""Parity tests for Smith-Waterman alignment, ported from Muraja's
SmithWatermanTests.swift. Scores must match the Swift `.balanced` scorer."""

import pytest

from tadabur.smith_waterman import (
    GAP,
    MATCH,
    MISMATCH,
    AlignedColumn,
    RefMatchInfo,
    local_alignment_score,
    longest_insertion_run,
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


# MARK: - aligned columns (issue #16)


def test_columns_perfect_match_are_all_pairs():
    from tadabur.smith_waterman import AlignedColumn

    result = smith_waterman(query="ابت", reference="ابت")
    assert result.columns == [
        AlignedColumn("ا", "ا"),
        AlignedColumn("ب", "ب"),
        AlignedColumn("ت", "ت"),
    ]


def test_columns_expose_reference_deletion_gap():
    from tadabur.smith_waterman import AlignedColumn

    # query drops the middle ت of the reference: a (None, ت) column.
    result = smith_waterman(query="بث", reference="بتث")
    assert AlignedColumn(None, "ت") in result.columns
    # reference chars, read in column order, reconstruct the aligned reference span.
    assert [c.ref_char for c in result.columns] == ["ب", "ت", "ث"]


def test_columns_expose_query_insertion():
    from tadabur.smith_waterman import AlignedColumn

    # query has an extra ت the reference lacks: a (ت, None) insertion column,
    # which the reference-indexed ref_matches cannot represent.
    result = smith_waterman(query="بتث", reference="بث")
    assert AlignedColumn("ت", None) in result.columns
    assert [c.query_char for c in result.columns] == ["ب", "ت", "ث"]


def test_empty_alignment_has_no_columns():
    assert smith_waterman(query="", reference="ابت").columns == []


# MARK: - longest_insertion_run (non-parity poison helper)


def _cols(spec: str) -> list[AlignedColumn]:
    """Build columns from a compact spec: 'i'=insertion, 'm'=match, ' '=space insert."""
    out = []
    for ch in spec:
        if ch == "i":
            out.append(AlignedColumn("ب", None))
        elif ch == "m":
            out.append(AlignedColumn("ب", "ب"))
        elif ch == " ":
            out.append(AlignedColumn(" ", None))
    return out


def test_longest_insertion_run_counts_consecutive_query_only_columns():
    assert longest_insertion_run(_cols("mmiiiimm")) == 4
    assert longest_insertion_run(_cols("iimmmiii")) == 3


def test_longest_insertion_run_zero_when_no_insertions():
    assert longest_insertion_run(_cols("mmmm")) == 0
    assert longest_insertion_run([]) == 0


def test_longest_insertion_run_broken_by_space_and_match():
    # A space column and a match column both reset the run.
    assert longest_insertion_run(_cols("ii mii")) == 2
    assert longest_insertion_run(_cols("iimii")) == 2
