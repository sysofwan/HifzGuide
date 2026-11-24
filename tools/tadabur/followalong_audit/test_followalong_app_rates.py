"""Flags per 1,000 graded words, by cause, with a clip-bootstrap interval (#309).

``flag_rates(rows, scoring, *, seed, iterations)`` reads ``word_grade_rows`` lines: the
denominator is the **graded** words (an ungraded word is S5, not a pass), a word counts once per
cause it carries (so cause rows can sum past the total), and the interval resamples whole clips
(``item``), since a clip's words are not independent.
"""

from __future__ import annotations

import pytest

from tadabur.followalong_audit.rates import flag_rates
from training.word_flags import CAUSES

NOT_FLAGGED = {"graded": True, "flagged": False, "causes": []}
UNGRADED = {"graded": False, "flagged": False, "causes": []}


def flagged(*causes: str) -> dict:
    return {"graded": True, "flagged": True, "causes": list(causes)}


def clip_rows(item: str, cells: list[dict], balanced: list[dict] | None = None) -> list[dict]:
    balanced = balanced or [NOT_FLAGGED if c["graded"] else UNGRADED for c in cells]
    return [{"item": item, "word": index + 1, "strict": strict, "shipped_default": shipped}
            for index, (strict, shipped) in enumerate(zip(cells, balanced))]


ROWS = (
    clip_rows("a#0", [flagged("tashkeel"), flagged("tashkeel", "soft_pair")] + [NOT_FLAGGED] * 8)
    + clip_rows("b#0", [NOT_FLAGGED] * 2 + [flagged("other_consonant")] + [NOT_FLAGGED] * 7)
    + clip_rows("c#0", [NOT_FLAGGED] * 5 + [UNGRADED] * 5)
)


def rates(rows=ROWS, scoring="strict", seed=1, iterations=200) -> dict:
    return flag_rates(rows, scoring, seed=seed, iterations=iterations)


def test_the_rate_is_flagged_words_per_1000_graded_words():
    total = rates()
    assert (total["graded"], total["flagged"]) == (25, 3)
    assert total["per_1000"] == pytest.approx(120.0)


def test_ungraded_words_are_not_in_the_denominator():
    assert rates()["graded"] == 25  # 30 words, 5 ungraded


def test_every_cause_has_a_row_and_a_word_counts_once_per_cause_it_carries():
    causes = rates()["causes"]
    assert set(causes) == set(CAUSES)
    assert causes["tashkeel"]["flagged"] == 2 and causes["tashkeel"]["per_1000"] == pytest.approx(80.0)
    assert causes["soft_pair"]["flagged"] == 1 and causes["soft_pair"]["per_1000"] == pytest.approx(40.0)
    assert causes["other_consonant"]["per_1000"] == pytest.approx(40.0)
    assert causes["skipped"]["flagged"] == 0 and causes["skipped"]["per_1000"] == 0.0


def test_each_scoring_is_read_from_its_own_cell():
    rows = clip_rows("a#0", [flagged("tashkeel"), NOT_FLAGGED],
                     [NOT_FLAGGED, flagged("other_consonant")])
    assert rates(rows, "strict")["causes"]["tashkeel"]["flagged"] == 1
    assert rates(rows, "shipped_default")["causes"]["tashkeel"]["flagged"] == 0
    assert rates(rows, "shipped_default")["causes"]["other_consonant"]["flagged"] == 1


def test_the_interval_brackets_the_rate_and_a_rerun_with_the_seed_repeats_it():
    first = rates(seed=5)
    low, high = first["interval"]
    assert 0 <= low <= first["per_1000"] <= high
    assert rates(seed=5) == first
    assert all(c["interval"][0] <= c["per_1000"] <= c["interval"][1] for c in first["causes"].values())


def test_clips_that_all_look_alike_leave_no_uncertainty():
    rows = [row for item in ("a#0", "b#0", "c#0", "d#0")
            for row in clip_rows(item, [flagged("tashkeel")] + [NOT_FLAGGED] * 9)]
    assert list(rates(rows)["interval"]) == [100.0, 100.0]


def test_the_interval_resamples_clips_so_one_heavy_clip_makes_it_wide():
    rows = (clip_rows("a#0", [flagged("tashkeel")] * 10)
            + [row for item in ("b#0", "c#0", "d#0", "e#0")
               for row in clip_rows(item, [NOT_FLAGGED] * 10)])
    low, high = rates(rows, iterations=500)["interval"]
    assert low == 0.0 and high >= 200.0
