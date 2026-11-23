"""P2, the CoreML-vs-PyTorch bridge (#309): per-word strict outcome agreement, bar 97%.

``agreement(app, pytorch)`` compares two sets of ``word_grade_rows`` lines word by word on the
strict outcome — ``flagged``, ``passed`` or ``ungraded`` — and lists every disagreement. The sets
must name the same words: a bridge that silently drops the words one side lacks would read
better than it is. ``bridge_verdict`` is ``met`` at or above :data:`P2_BAR`, ``below`` under it;
below the bar the run proceeds, so the verdict is a report and never an exception.
"""

from __future__ import annotations

import pytest

from tadabur.followalong_audit.bridge import P2_BAR, agreement, bridge_verdict


def cell(graded: bool = True, flagged: bool = False) -> dict:
    return {"graded": graded, "flagged": flagged, "causes": ["tashkeel"] if flagged else []}


def row(item: str, word: int, strict: dict, balanced: dict | None = None) -> dict:
    return {"item": item, "word": word, "strict": strict, "shipped_default": balanced or strict}


def words(cells: list[dict], item: str = "a#0") -> list[dict]:
    return [row(item, index + 1, c) for index, c in enumerate(cells)]


def test_the_bar_is_97_percent():
    assert P2_BAR == 0.97


def test_identical_outcomes_agree_completely():
    rows = words([cell(), cell(flagged=True), cell(graded=False)])
    result = agreement(rows, rows)
    assert (result["words"], result["agree"], result["rate"]) == (3, 3, 1.0)
    assert result["disagreements"] == []


def test_a_flag_one_side_does_not_raise_is_a_listed_disagreement():
    app = words([cell(), cell(flagged=True)])
    torch = words([cell(), cell()])
    result = agreement(app, torch)
    assert (result["agree"], result["rate"]) == (1, 0.5)
    assert result["disagreements"] == [
        {"item": "a#0", "word": 2, "app": "flagged", "pytorch": "passed"}]


def test_ungraded_against_a_pass_is_a_disagreement():
    result = agreement(words([cell(graded=False)]), words([cell()]))
    assert result["disagreements"] == [
        {"item": "a#0", "word": 1, "app": "ungraded", "pytorch": "passed"}]


def test_two_flags_with_different_causes_still_agree_on_the_outcome():
    app = [row("a#0", 1, {"graded": True, "flagged": True, "causes": ["tashkeel"]})]
    torch = [row("a#0", 1, {"graded": True, "flagged": True, "causes": ["other_consonant"]})]
    assert agreement(app, torch)["rate"] == 1.0


def test_only_the_strict_outcome_is_compared():
    app = [row("a#0", 1, cell(), balanced=cell(flagged=True))]
    torch = [row("a#0", 1, cell(), balanced=cell())]
    assert agreement(app, torch)["rate"] == 1.0


def test_the_two_sides_must_name_the_same_words():
    with pytest.raises(ValueError):
        agreement(words([cell(), cell()]), words([cell()]))


def test_the_verdict_is_met_at_the_bar_and_below_under_it_without_raising():
    assert bridge_verdict(0.97) == "met"
    assert bridge_verdict(1.0) == "met"
    assert bridge_verdict(0.9699) == "below"
