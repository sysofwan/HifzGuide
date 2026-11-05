"""Tests for #267's site outcomes, the letter-level read of every column, and unique catches.

Every column (the teacher, C95, the round-2 EMA and FastConformer) is read the same way (the
architect's r4 decision): what does the whole-item decode carry at the site's letter (or its
shadda)? #267 counts "carries nothing" as flagged, and a *unique catch* is a decisive mistake
where FastConformer carries the heard letter and none of the three other columns does.
"""

from __future__ import annotations

import pytest

from tadabur.fastconformer_screen.outcomes import (
    COLUMNS,
    HEARD,
    NONE,
    OTHER,
    OUTCOMES,
    PRESCRIBED,
    breakdown,
    decoy_unchanged,
    edit_heard,
    is_flagged,
    outcome,
    student_carried,
    unique_catches,
)
from tadabur.fastconformer_screen.sets import ship_a_sites, split_sides

STUDENTS = ("base", "c95", "r2_candidate_ema")


def test_the_four_outcomes_and_four_columns():
    assert OUTCOMES == (HEARD, PRESCRIBED, OTHER, NONE) == ("heard", "prescribed", "other", "none")
    assert COLUMNS == ("fastconformer", "base", "c95", "r2_candidate_ema")


@pytest.mark.parametrize(("carried", "expected"), [
    ("ك", HEARD),        # the letter the human heard
    ("ق", PRESCRIBED),   # the text prior: the mushaf's letter
    ("ز", OTHER),
    (None, NONE),
])
def test_outcome_of_a_consonant_site(carried, expected):
    assert outcome(carried, prescribed="ق", heard="ك") == expected


def test_outcome_of_a_shaddah_site():
    assert outcome("not_held", prescribed="held", heard="not_held") == HEARD
    assert outcome("held", prescribed="held", heard="not_held") == PRESCRIBED
    assert outcome(None, prescribed="held", heard="not_held") == NONE


def test_on_correct_recitation_the_prescribed_letter_is_also_the_heard_one():
    assert outcome("ق", prescribed="ق", heard="ق") == HEARD


@pytest.mark.parametrize(("carried", "flagged"), [
    ("ق", False),
    ("ك", True),
    ("ز", True),
    (None, True),        # #267: a site that carries nothing is flagged
])
def test_a_site_is_flagged_when_it_does_not_carry_the_prescribed_letter(carried, flagged):
    assert is_flagged(carried, prescribed="ق") is flagged


def test_edit_heard_and_decoy_unchanged_read_the_carried_letter():
    assert edit_heard("ز", label="ز") is True
    assert edit_heard("ذ", label="ز") is False
    assert edit_heard(None, label="ز") is False
    assert decoy_unchanged("ذ", prescribed="ذ") is True
    assert decoy_unchanged("ز", prescribed="ذ") is False
    assert decoy_unchanged(None, prescribed="ذ") is False


def test_breakdown_counts_every_outcome_including_zeros():
    assert breakdown([HEARD, NONE, HEARD, PRESCRIBED]) == {
        "heard": 2, "prescribed": 1, "other": 0, "none": 1}


# --- unique catches ---------------------------------------------------------------------------


def row(fastconformer, base=PRESCRIBED, c95=PRESCRIBED, ema=PRESCRIBED):
    return dict(zip(COLUMNS, (fastconformer, base, c95, ema)))


def test_a_unique_catch_is_heard_by_fastconformer_and_by_none_of_the_others():
    outcomes = {
        "site-b": row(HEARD),                    # unique
        "site-a": row(HEARD, NONE, OTHER, NONE), # unique: the others carry nothing or another letter
        "site-c": row(HEARD, base=HEARD),        # the teacher heard it too
        "site-d": row(HEARD, c95=HEARD),
        "site-e": row(HEARD, ema=HEARD),
        "site-f": row(PRESCRIBED),               # FastConformer wrote the mushaf over it
        "site-g": row(OTHER),
    }
    assert unique_catches(outcomes) == ["site-a", "site-b"]    # by site, sorted


def test_unique_catches_refuse_a_site_missing_a_column():
    with pytest.raises(ValueError, match="site-x"):
        unique_catches({"site-x": {"fastconformer": HEARD, "base": NONE}})


def test_unique_catches_of_nothing_is_empty():
    assert unique_catches({}) == []


# --- the three students, read from their committed whole-item decodes --------------------------


@pytest.mark.parametrize("model", STUDENTS)
def test_student_carried_covers_every_ship_a_site_with_a_letter_or_nothing(model):
    sites = ship_a_sites()
    carried = student_carried(model, sites)
    assert set(carried) == {site.site_id for site in sites}
    for site in sites:
        value = carried[site.site_id]
        assert value is None or isinstance(value, str)


def test_the_students_are_read_from_committed_stream_b0_text_only():
    mistakes, _ = split_sides(ship_a_sites())
    carried = student_carried("base", mistakes)
    # the teacher mishears some real mistakes and writes the mushaf over others: both exist
    flagged = [is_flagged(carried[s.site_id], prescribed=s.prescribed) for s in mistakes]
    assert any(flagged) and not all(flagged)
