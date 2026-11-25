"""The S5 engine-ungraded disclosure is the committed join, and its direction (#310, A1).

Self-check r2 found the first A1 wording said "at most 2,580" when the covered-and-engine-ungraded
count is a *minimum*, and that 1,207 had no committed input. The join is now part of
``app_arm/cells.json`` (``coverage.s5_engine_graded``) and the document must quote the difference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "tools" / "tadabur" / "followalong_audit"
PREREG = ROOT / "docs" / "followalong-audit-preregistration.md"
CELLS = json.loads((AUDIT / "app_arm" / "cells.json").read_text(encoding="utf-8"))
S5 = ("S5-leading", "S5-trailing", "S5-interior")


def _section(number: int) -> str:
    parts = re.split(r"^## (\d+)\. .*$", PREREG.read_text(encoding="utf-8"), flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))[str(number)]


def test_the_committed_join_covers_only_the_covered_s5_words_and_never_exceeds_them():
    coverage = CELLS["coverage"]
    assert set(coverage["s5_engine_graded"]) == set(S5)
    for name in S5:
        assert 0 <= coverage["s5_engine_graded"][name] <= coverage["frame"][name], name
    assert sum(coverage["s5_engine_graded"].values()) < sum(
        coverage["frame"][name] for name in S5)


def test_section_3_quotes_the_engine_ungraded_difference_the_record_holds():
    coverage = CELLS["coverage"]
    graded = sum(coverage["s5_engine_graded"].values())
    ungraded = sum(coverage["frame"][name] for name in S5) - graded
    text = _section(3)
    assert f"left ungraded {ungraded:,} words the transcript covers" in text
    assert f"graded only {graded:,}" in text
    assert "at most 2,580" not in text
