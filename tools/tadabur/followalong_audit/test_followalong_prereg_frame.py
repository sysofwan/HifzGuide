"""The estimates' frozen frame and formulas (#310, R1.2).

The headline frame is the sampled subpopulation — every stratum with positive inclusion
probability — because inverse-inclusion weighting cannot recover a zero-probability pool. This
pins the frame's size from the committed pools, its exclusion of the residual and
oracle-explained pools, S4's role in each denominator, the two formulas, and the document's
conditional label.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "tools" / "tadabur" / "followalong_audit"
PREREG = ROOT / "docs" / "followalong-audit-preregistration.md"
POOLS = json.loads((AUDIT / "app_arm" / "pools.json").read_text(encoding="utf-8"))["pools"]["strata"]
CELLS = json.loads((AUDIT / "app_arm" / "cells.json").read_text(encoding="utf-8"))
SAMPLED = ("S1", "S3", "S4", "S5")
UNSAMPLED = ("residual", "oracle-explained")


def _section(number: int) -> str:
    parts = re.split(r"^## (\d+)\. .*$", PREREG.read_text(encoding="utf-8"), flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))[str(number)]


def test_the_frame_is_every_covered_word_of_the_strata_with_positive_inclusion_probability():
    # Amended by A1 (2026-11-24): the candidate pools hold 54,555 words; the frame is the 45,398
    # the Teacher decode graded, and the other 9,157 are a reported pool with pi = 0.
    candidate = POOLS["S1"] + POOLS["S1-pending"] + sum(POOLS[name] for name in ("S3", "S4", "S5"))
    assert candidate == 54555
    assert CELLS["coverage"]["frame_words"] + CELLS["coverage"]["not_covered_words"] == candidate
    text = _section(7)
    assert "sampled subpopulation" in text and "**45,398 words**" in text
    assert "π = 0" in text
    for name in UNSAMPLED:
        assert name in text and f"{POOLS[name]:,}" in text


def test_the_document_never_presents_the_estimates_as_whole_population_rates():
    text = _section(7)
    assert "conditional on this frame" in text
    assert re.search(r"not a rate for all 57,935", text)
    assert re.search(r"nor for all placeable phone\s+audio", text)
    assert "acceptance-rules.md" in text
    assert "never weighted into a rate" in text
    assert "separately labelled bound" in text


def test_both_formulas_are_frozen_with_the_cell_weights():
    text = _section(7)
    assert "Σ_{w∈frame, verdict correct} π_w⁻¹ F_w / Σ_{w∈frame, verdict correct} π_w⁻¹" in text
    # Amended by A4 (2026-11-24): the headline miss rate is on H = S1 ∪ S3 ∪ S5-interior.
    assert re.search(r"H\s*=\s*S1\s*∪\s*S3\s*∪\s*S5-interior", text)
    assert ("Σ_{w∈H, verdict mistake, App not flagged} π_w⁻¹ / "
            "Σ_{w∈H, verdict mistake} π_w⁻¹") in text
    assert "cell** inclusion probability π" in text
    assert "1,000" in text and "100 ·" in text


def test_s4_contributes_to_both_denominators_and_its_own_bound_is_not_a_headline():
    text = _section(7)
    # Amended by A4: S4 is in the false-flag denominator, and out of the headline miss rate.
    assert re.search(r"S4[^.\n]*false-flag denominator", text)
    assert "cannot enter the false-flag numerator" in text
    assert re.search(r"S4[^.\n]*(not|never)[^.\n]*headline miss", text)
    assert re.search(r"S4[^.\n]*bound, not a headline", text)


def test_section_3_names_the_same_frame():
    text = _section(3)
    assert "45,398 words" in text
    assert "π = 0" in text
