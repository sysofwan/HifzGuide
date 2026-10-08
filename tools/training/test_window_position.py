"""Tests for the within-window position comparison.

The claim this tool could wrongly support is "committing a later block is better", which
would motivate a protocol change. The two ways it could be wrong by construction are a block
rule that does not match the deployed commit rule, and a self-consistency tally that pairs
the wrong windows. Both are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.decoding import confirmed_tokens
from training.distill_loss import CONFIRM_TIMESTEPS
from training import window_position as wp


def row(*spans) -> np.ndarray:
    """A 125-frame argmax row from ``(token, start, end_inclusive)`` spans; rest blank."""
    ids = np.zeros(125, dtype=np.int64)
    for token, start, end in spans:
        ids[start : end + 1] = token
    return ids


# --- the block rule ------------------------------------------------------------------


def test_block_zero_is_exactly_the_deployed_commit_rule():
    """If these ever diverge, every block comparison is against the wrong baseline."""
    for spans in (
        [(7, 0, 4)],
        [(7, 20, 30)],
        [(7, 0, 4), (9, 60, 64)],
        [(5, 24, 26)],
        [(3, 100, 124)],
    ):
        ids = row(*spans)
        assert wp.block_tokens(ids, 0) == confirmed_tokens(ids, CONFIRM_TIMESTEPS)


def test_every_segment_lands_in_exactly_one_block():
    ids = row((1, 2, 6), (2, 30, 34), (3, 55, 60), (4, 80, 84), (5, 110, 114))
    collected = [t for b in range(wp.NUM_BLOCKS) for t in wp.block_tokens(ids, b)]
    assert collected == [1, 2, 3, 4, 5]


def test_a_segment_is_owned_by_the_block_containing_its_midpoint():
    # Midpoint 24.5 -> block 0; midpoint 25.0 -> block 1.
    assert wp.block_tokens(row((7, 24, 25)), 0) == [7]
    assert wp.block_tokens(row((7, 24, 25)), 1) == []
    assert wp.block_tokens(row((7, 25, 25)), 1) == [7]
    assert wp.block_tokens(row((7, 25, 25)), 0) == []


def test_blank_only_rows_commit_nothing():
    for b in range(wp.NUM_BLOCKS):
        assert wp.block_tokens(np.zeros(125, dtype=np.int64), b) == []


def test_an_out_of_range_block_is_refused():
    for bad in (-1, wp.NUM_BLOCKS):
        with pytest.raises(ValueError, match="block"):
            wp.block_tokens(np.zeros(125, dtype=np.int64), bad)


def test_there_are_five_blocks_of_one_second():
    assert wp.NUM_BLOCKS == 5


# --- tallies -------------------------------------------------------------------------


def test_identical_models_agree_perfectly_at_every_position():
    rows = [row((1, 2, 6), (2, 30, 34)), row((3, 10, 14), (4, 60, 64))]
    agreement, _ = wp.tally_positions(rows, rows)
    assert all(t.accuracy == 1.0 for t in agreement if t.reference)


def test_a_substitution_is_charged_to_its_own_block():
    teacher = [row((1, 2, 6), (2, 30, 34))]
    student = [row((1, 2, 6), (9, 30, 34))]
    agreement, _ = wp.tally_positions(teacher, student)
    assert agreement[0].edits == 0
    assert agreement[1].edits == 1


def test_teacher_self_consistency_pairs_the_same_absolute_second():
    """Block b of window w must be compared with block 0 of window w+b, not w."""
    # Window 0 block 1 says token 5; window 1 block 0 says token 5 -- the same second.
    rows = [row((5, 30, 34)), row((5, 2, 6))]
    _, consistency = wp.tally_positions(rows, rows)
    assert consistency[1].edits == 0
    assert consistency[1].reference == 1

    # Now make window 1 block 0 disagree; the b=1 tally must notice.
    rows_bad = [row((5, 30, 34)), row((8, 2, 6))]
    _, consistency = wp.tally_positions(rows_bad, rows_bad)
    assert consistency[1].edits == 1


def test_self_consistency_at_block_zero_is_trivially_perfect():
    """b=0 compares a window's block 0 against itself, so it is a sanity anchor."""
    rows = [row((1, 2, 6)), row((2, 2, 6)), row((3, 2, 6))]
    _, consistency = wp.tally_positions(rows, rows)
    assert consistency[0].edits == 0
    assert consistency[0].blocks == len(rows)


def test_self_consistency_skips_blocks_with_no_later_window():
    """Block b of the last windows has no window w+b to compare against."""
    rows = [row((1, 2, 6)), row((2, 2, 6))]
    _, consistency = wp.tally_positions(rows, rows)
    assert consistency[0].blocks == 2   # windows 0 and 1
    assert consistency[1].blocks == 1   # only window 0 has a window 1
    assert consistency[4].blocks == 0   # no window 4 exists


def test_agreement_counts_every_window_at_every_block():
    rows = [row((1, 2, 6)), row((2, 2, 6)), row((3, 2, 6))]
    agreement, _ = wp.tally_positions(rows, rows)
    assert all(t.blocks == 3 for t in agreement)


def test_an_empty_tally_reports_zero_rather_than_dividing():
    assert wp.PositionTally().accuracy == 0.0


def test_accuracy_is_one_minus_edits_over_reference():
    t = wp.PositionTally()
    t.add([1, 2, 3, 4], [1, 9, 3, 4])
    assert t.accuracy == pytest.approx(0.75)
    assert t.as_dict()["reference_tokens"] == 4
