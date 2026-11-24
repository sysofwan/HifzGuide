"""The pilot is the 540 design's prefix, not an extra 150 (#310, D6, R1.1).

D6 fixes one 540-snip design drawn first as a seeded permutation per cell; the 150-snip pilot is
each cell's prefix. The document must say the 200/250/60/30 allocations are **cumulative** totals,
give the initial remaining main allocation (144/181/43/22, 390), and charge all served time to the
3 h budget. The locked size tests keep their assertions; this file pins the consistency.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREREG = ROOT / "docs" / "followalong-audit-preregistration.md"
TOTALS = {"S1": 200, "S3": 250, "S4": 60, "S5": 30}
PILOT = {"S1": 56, "S3": 69, "S4": 17, "S5": 8}


def _text() -> str:
    return PREREG.read_text(encoding="utf-8")


def _section(number: int) -> str:
    parts = re.split(r"^## (\d+)\. .*$", _text(), flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))[str(number)]


def _cells(line: str) -> list[str]:
    return [cell.strip().strip("`*") for cell in line.strip().strip("|").split("|")]


def _repeats_row() -> list[str]:
    return next(row for row in map(_cells, _section(3).splitlines())
                if row and row[0].startswith("repeats"))


def test_the_design_is_the_cumulative_540_not_a_pilot_plus_540():
    text = _section(3)
    assert re.search(r"540.*cumulative pilot and main", text)
    assert re.search(r"pilot is the \*\*first 150 of those 540\*\*", text)
    assert not re.search(r"main sitting is \*\*540\*\*", text)
    assert not re.search(r"pilot precedes it", text)


def test_the_initial_remaining_main_allocation_is_the_design_minus_the_pilot():
    text = _section(3)
    remaining = {name: TOTALS[name] - PILOT[name] for name in TOTALS}
    assert remaining == {"S1": 144, "S3": 181, "S4": 43, "S5": 22}
    assert sum(remaining.values()) == 390
    for value in remaining.values():
        assert str(value) in text
    assert "**390**" in text


def test_the_repeats_row_pools_over_the_same_540_not_an_extra_150():
    row = _repeats_row()
    assert row[1] == "540" and row[2] == "25" and row[3] == "7"
    assert "150 + 540" not in _text()


def test_the_resize_and_budget_read_cumulative_totals():
    text = _section(8)
    assert re.search(r"first 150", text) and re.search(r"remaining 390", text)
    assert re.search(r"floors for the \*\*final\*\* totals", text)
    assert re.search(r"charged against the \*\*3 h\*\*", text)
    for floor in ("S1 ≥ 150", "S4 ≥ 40", "S5 ≥ 20", "repeats ≥ 15"):
        assert floor in text
