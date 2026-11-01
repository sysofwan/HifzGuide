"""Tests for the card-reproduction gate and character-level PER (#245, self-check r1)."""

from __future__ import annotations

import pytest

from training.zipformer_gate import (
    CARD_PER,
    GATE_LIMIT_POINTS,
    SCALE_GAP_POINTS,
    GateStop,
    character_per,
    pin_scale,
    require_card_reproduction,
    source_per,
)


def _measured(**per: float) -> dict[str, dict]:
    return {source: {"per": value, "clips": 1, "misses": 0} for source, value in per.items()}


def test_the_cards_per_values_are_pinned():
    assert CARD_PER == {"studio": 1.43, "phone": 3.65, "unseen": 9.10}


def test_matching_the_card_passes():
    require_card_reproduction(_measured(**CARD_PER))


def test_exactly_the_limit_is_within_it():
    require_card_reproduction(_measured(studio=1.93, phone=3.65, unseen=9.10))


@pytest.mark.parametrize(("source", "value"), [("studio", 1.94), ("phone", 4.16),
                                               ("unseen", 9.61)])
def test_over_the_limit_on_any_source_stops(source, value):
    per = _measured(**CARD_PER)
    per[source]["per"] = value
    with pytest.raises(GateStop):
        require_card_reproduction(per)


def test_a_half_point_from_the_per_ratio_is_within_it():
    # source_per returns 100 * 83 / 2000 = 4.15; binary subtraction gives 0.5000000000000004.
    measured = 100 * 83 / 2000
    assert abs(measured - CARD_PER["phone"]) > GATE_LIMIT_POINTS
    per = _measured(studio=1.43, unseen=9.10)
    per["phone"] = {"per": measured, "clips": 1, "misses": 0}
    require_card_reproduction(per)


def test_over_the_limit_from_the_per_ratio_stops():
    per = _measured(studio=1.43, unseen=9.10)
    per["phone"] = {"per": 100 * 84 / 2000, "clips": 1, "misses": 0}
    with pytest.raises(GateStop):
        require_card_reproduction(per)


def _rates(edits: dict[str, int], units: int = 2000) -> dict[str, float]:
    """PERs as :func:`source_per` computes them, from each source's edit count."""
    return {source: 100 * count / units for source, count in edits.items()}


#: The review's scale-rule example: a two-point phone gap with scale 1 better everywhere.
_TWO_POINT_LOW = _rates({"studio": 28, "phone": 73, "unseen": 182})
_TWO_POINT_HIGH = _rates({"studio": 100, "phone": 113, "unseen": 400})


def test_an_exact_two_point_scale_gap_from_the_per_ratio_stops_the_rule():
    assert _TWO_POINT_HIGH["phone"] - _TWO_POINT_LOW["phone"] > SCALE_GAP_POINTS
    with pytest.raises(GateStop):
        pin_scale({1.0: _TWO_POINT_LOW, 32_768.0: _TWO_POINT_HIGH})


def test_just_over_two_points_from_the_per_ratio_is_decisive():
    high = _rates({"studio": 100, "phone": 114, "unseen": 400})
    assert pin_scale({1.0: _TWO_POINT_LOW, 32_768.0: high}) == 1.0


def test_a_card_source_the_run_did_not_score_stops():
    with pytest.raises(GateStop):
        require_card_reproduction(_measured(studio=1.43, phone=3.65))


def test_a_source_with_no_reference_units_stops():
    per = _measured(**CARD_PER)
    per["phone"]["per"] = None
    with pytest.raises(GateStop):
        require_card_reproduction(per)


def test_character_per_pools_characters_and_counts_misses():
    rows = [
        {"source": "studio", "gold": ["اا", "بَ"], "pred": ["اا", "بَ"]},
        {"source": "studio", "gold": ["بَ"], "pred": ["ب"]},
        {"source": "studio", "gold": None, "pred": ["س"]},
    ]
    assert character_per(rows) == {
        "studio": {"per": pytest.approx(100 * 1 / 6), "clips": 2, "misses": 1}}
    # the same rows at the unit level: one edit over three reference units
    assert source_per(rows)["studio"]["per"] == pytest.approx(100 * 1 / 3)
