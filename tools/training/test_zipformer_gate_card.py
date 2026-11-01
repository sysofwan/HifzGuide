"""Tests for the card-reproduction gate and character-level PER (#245, self-check r1)."""

from __future__ import annotations

import pytest

from training.zipformer_gate import (
    CARD_PER,
    GateStop,
    character_per,
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
