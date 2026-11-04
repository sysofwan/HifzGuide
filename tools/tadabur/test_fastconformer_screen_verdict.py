"""Tests for #267's pre-registered screening rule: GO / PRIOR / STOP on boundary fixtures.

The rule (issue #267, owner-approved "approve #267", thresholds frozen):

* STOP on the correct side when Ship A correct sites flagged > teacher + 5 points, or decoys
  left unchanged < teacher - 5 points (a strict inequality each: exactly 5 points passes);
* GO when the correct side passes, mistakes flagged >= teacher - 5, edits heard >= teacher - 5
  and there are >= 3 unique catches;
* PRIOR otherwise (correct side passes; the catch side or the unique catches fail).

Gates (steps 1-2) are refusals, tested in ``test_fastconformer_screen_gates.py``.
"""

from __future__ import annotations

import dataclasses

import pytest

from tadabur.fastconformer_screen.verdict import (
    CATCH_MARGIN,
    CORRECT_MARGIN,
    GO,
    PRIOR,
    STOP,
    UNIQUE_MINIMUM,
    Rates,
    verdict,
)

#: The teacher's four rates, in percent (binary-exact so the boundaries are exact).
TEACHER = Rates(mistakes_flagged=50.0, correct_flagged=40.0, edits_heard=20.0, decoys_unchanged=60.0)


def fc(**changes) -> Rates:
    """FastConformer's rates: exactly the teacher's, with ``changes`` applied."""
    return dataclasses.replace(TEACHER, **changes)


def test_the_frozen_thresholds():
    assert (CORRECT_MARGIN, CATCH_MARGIN, UNIQUE_MINIMUM) == (5.0, 5.0, 3)
    assert (GO, PRIOR, STOP) == ("GO", "PRIOR", "STOP")


def test_rates_name_the_four_measures():
    assert [f.name for f in dataclasses.fields(Rates)] == [
        "mistakes_flagged", "correct_flagged", "edits_heard", "decoys_unchanged"]


def test_equal_to_the_teacher_with_three_unique_catches_is_go():
    assert verdict(TEACHER, fc(), unique_catches=3) == GO


# --- STOP: the correct side -------------------------------------------------------------------


def test_correct_sites_flagged_exactly_five_points_over_does_not_stop():
    assert verdict(TEACHER, fc(correct_flagged=45.0), unique_catches=3) == GO


def test_correct_sites_flagged_over_five_points_stops():
    assert verdict(TEACHER, fc(correct_flagged=45.5), unique_catches=3) == STOP


def test_decoys_unchanged_exactly_five_points_under_does_not_stop():
    assert verdict(TEACHER, fc(decoys_unchanged=55.0), unique_catches=3) == GO


def test_decoys_unchanged_under_five_points_below_stops():
    assert verdict(TEACHER, fc(decoys_unchanged=54.5), unique_catches=3) == STOP


def test_the_correct_side_stops_whatever_the_catch_side_says():
    best = fc(correct_flagged=60.0, mistakes_flagged=100.0, edits_heard=100.0)
    assert verdict(TEACHER, best, unique_catches=40) == STOP


# --- GO against PRIOR: the catch side -----------------------------------------------------------


def test_mistakes_flagged_exactly_five_points_under_still_goes():
    assert verdict(TEACHER, fc(mistakes_flagged=45.0), unique_catches=3) == GO


def test_mistakes_flagged_under_five_points_below_is_prior():
    assert verdict(TEACHER, fc(mistakes_flagged=44.5), unique_catches=3) == PRIOR


def test_edits_heard_exactly_five_points_under_still_goes():
    assert verdict(TEACHER, fc(edits_heard=15.0), unique_catches=3) == GO


def test_edits_heard_under_five_points_below_is_prior():
    assert verdict(TEACHER, fc(edits_heard=14.5), unique_catches=3) == PRIOR


@pytest.mark.parametrize(("unique", "expected"), [(0, PRIOR), (2, PRIOR), (3, GO), (4, GO)])
def test_three_unique_catches_are_the_bar(unique, expected):
    assert verdict(TEACHER, fc(mistakes_flagged=90.0, edits_heard=90.0), unique) == expected


def test_a_negative_unique_count_is_refused():
    with pytest.raises(ValueError):
        verdict(TEACHER, fc(), unique_catches=-1)


def test_the_verdict_never_pools_the_two_sides():
    # a great catch side cannot buy back a failing correct side, nor the reverse
    assert verdict(TEACHER, fc(correct_flagged=46.0, mistakes_flagged=100.0), unique_catches=9) == STOP
    assert verdict(TEACHER, fc(mistakes_flagged=0.0, edits_heard=0.0), unique_catches=9) == PRIOR
