"""The allocation arithmetic the pre-registration fixes before any draw (#310, D2 and D6).

Committed before ``tadabur/followalong_audit/allocation.py`` exists and **locked**: later turns
may add tests but never weaken or rewrite these. No item is drawn here; this is the integer
arithmetic that says how many.

Contract this file pins
-----------------------
* ``primary_cause(causes)``: the first of ``target_pair, soft_pair, other_consonant, shaddah,
  tashkeel, skipped, unrecorded`` the word carries, whatever order it lists them in. No cause, or a
  cause outside the seven, raises ``ValueError``.
* ``allocate(sizes, total, *, floor=0)``: split ``total`` over the cells of ``sizes`` (cell ->
  pool size) in proportion to their size, by largest remainder, so that every cell gets at least
  ``floor`` (but never more than its pool) and the allocations sum to exactly ``total``. A cell
  whose floor binds is paid for by the cells above it. Ties go to the larger pool, then to the
  smaller key. ``total`` above the pools' sum, or a floor the total cannot pay, raises
  ``ValueError``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from tadabur.followalong_audit.allocation import allocate, primary_cause

POOLS = json.loads((Path(__file__).parent / "app_arm" / "pools.json").read_text(encoding="utf-8"))
ORDER = ("target_pair", "soft_pair", "other_consonant", "shaddah", "tashkeel", "skipped",
         "unrecorded")


@pytest.mark.parametrize("causes, expected", [
    (["tashkeel", "target_pair"], "target_pair"),
    (["tashkeel", "soft_pair"], "soft_pair"),
    (["tashkeel", "other_consonant"], "other_consonant"),
    (["skipped", "shaddah"], "shaddah"),
    (["unrecorded", "skipped"], "skipped"),
    (["unrecorded", "tashkeel"], "tashkeel"),
    (["unrecorded"], "unrecorded"),
])
def test_the_primary_cause_is_the_first_the_word_carries_in_the_pre_registered_order(
        causes, expected):
    assert primary_cause(causes) == expected


def test_the_primary_cause_does_not_depend_on_the_order_the_causes_are_listed_in():
    for order in itertools.permutations(ORDER[2:5]):
        assert primary_cause(list(order)) == "other_consonant"


def test_the_order_differs_from_word_flags_where_tashkeel_comes_first():
    from training.word_flags import CAUSES
    assert CAUSES[0] == "tashkeel" and ORDER[0] == "target_pair"
    assert primary_cause(["tashkeel", "soft_pair"]) == "soft_pair"


def test_a_word_with_no_cause_or_an_unknown_one_has_no_primary_cause():
    for causes in ([], ["haraka"], ["tashkeel", "haraka"]):
        with pytest.raises(ValueError):
            primary_cause(causes)


def test_the_pilot_split_is_the_largest_remainder_over_the_540_stratum_snips():
    sizes = {"S1": 200, "S3": 250, "S4": 60, "S5": 30}
    # 55.56 / 69.44 / 16.67 / 8.33: the floors sum to 148, S4 (.67) and S1 (.56) take the rest.
    assert allocate(sizes, 150) == {"S1": 56, "S3": 69, "S4": 17, "S5": 8}


def test_an_even_split_is_exact():
    assert allocate({"a": 50, "b": 50}, 10) == {"a": 5, "b": 5}
    assert allocate({"a": 1, "b": 1, "c": 1}, 3) == {"a": 1, "b": 1, "c": 1}


def test_the_allocation_keeps_the_cells_in_the_callers_order():
    sizes = {"z": 10, "a": 30}
    assert list(allocate(sizes, 4)) == ["z", "a"]


def test_a_floor_that_binds_is_paid_for_by_the_larger_cells():
    # 100 * 10 / 1010 = 0.99: proportional gives the small cell 1, the floor gives it 3.
    assert allocate({"big": 1000, "small": 10}, 100, floor=3) == {"big": 97, "small": 3}


def test_a_floor_that_does_not_bind_changes_nothing():
    sizes = {"S1": 200, "S3": 250, "S4": 60, "S5": 30}
    assert allocate(sizes, 150, floor=3) == allocate(sizes, 150)


def test_a_cell_never_gets_more_than_its_pool():
    assert allocate({"tiny": 2, "big": 1000}, 100, floor=3) == {"tiny": 2, "big": 98}
    assert allocate({"a": 5, "b": 5}, 10) == {"a": 5, "b": 5}


def test_ties_go_to_the_larger_pool_then_the_smaller_key():
    assert allocate({"b": 1, "a": 1}, 1) == {"b": 0, "a": 1}
    # 1 / 4 * 2 = .5 and 3 / 4 * 2 = 1.5: equal remainders, so the larger pool takes the last snip.
    assert allocate({"a": 1, "b": 3}, 2) == {"a": 0, "b": 2}


def test_the_allocation_does_not_depend_on_the_order_the_cells_are_listed_in():
    sizes = {"a": 7, "b": 13, "c": 29, "d": 3}
    first = allocate(sizes, 20, floor=1)
    for keys in itertools.permutations(sizes):
        assert allocate({key: sizes[key] for key in keys}, 20, floor=1) == first


def test_the_s1_cells_split_the_200_with_a_floor_of_three_whatever_their_exact_counts():
    cells = {(row["cause"], row["base_flags"]): row["words"]
             for row in POOLS["s1_by_cause_and_base"]}
    assert len(cells) == 14
    split = allocate(cells, 200, floor=3)
    assert sum(split.values()) == 200
    assert all(3 <= split[cell] <= cells[cell] for cell in cells)
    assert split[("target_pair", False)] == 3 and split[("target_pair", True)] == 3
    # A bigger pool never gets fewer snips than a smaller one.
    ordered = sorted(cells, key=cells.get)
    assert [split[cell] for cell in ordered] == sorted(split[cell] for cell in ordered)


@pytest.mark.parametrize("sizes, total, floor", [
    ({"a": 3, "b": 4}, 8, 0),
    ({"a": 100, "b": 100, "c": 100}, 8, 3),
    ({"a": 100}, -1, 0),
])
def test_an_allocation_the_pools_or_the_floors_cannot_pay_is_refused(sizes, total, floor):
    with pytest.raises(ValueError):
        allocate(sizes, total, floor=floor)
