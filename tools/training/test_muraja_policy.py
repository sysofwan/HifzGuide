"""Today's Muraja configuration and every allowance's ON/OFF state table (§9), at site level."""

from __future__ import annotations

from dataclasses import replace

import pytest

from training.muraja_policy import (
    ALLOWANCES,
    CORRECT,
    MURAJA_REVISION,
    NOT_GRADED,
    TODAY,
    WRONG,
    at_pause,
    grade,
)
from training.site_outcomes import DHAL_ZAH, SiteOutcome, item_outcomes
from training.test_site_outcomes import (
    DAL,
    DAMMA,
    DHAKARA,
    DHAL,
    FATHA,
    KATABA,
    MADA,
    MADDA,
    QULHU,
    TAA,
    ZAH,
    ZAI,
    make,
)

WAW, YA, HAMZA = "و", "ي", "ء"
# "وَكَتَبَلَ": the fatha on و is a dropped-haraka-exempt carrier.
WAKATABA = WAW + FATHA + KATABA

SOFT = "ذ↔ز"


def committed(mark: str | None, *marks: str) -> SiteOutcome:
    return SiteOutcome(True, mark, marks or ((mark,) if mark else ()))


def test_today_is_muraja_balanced_defaults():
    assert TODAY.revision == MURAJA_REVISION
    assert TODAY.mode == "balanced"
    assert TODAY.tashkeel_errors and TODAY.shaddah_suppression and TODAY.end_word
    assert not TODAY.suppress_haraka_drop and not TODAY.empty_slot_not_graded
    assert TODAY.dropped_haraka_letters == frozenset("واءي")
    assert len(TODAY.soft_pairs) == 6 and DHAL_ZAH not in TODAY.soft_pairs


# --- tashkeel ---------------------------------------------------------------------------


def test_haraka_state_table_today():
    site = make(KATABA, 2, "fatha", "fatha", "fatha")
    assert grade(site, committed("fatha")) == CORRECT
    assert grade(site, committed("kasra")) == WRONG
    assert grade(site, committed(None)) == WRONG  # a dropped haraka on an ordinary letter
    assert grade(site, committed("multiple", "kasra", "fatha")) == CORRECT  # trailing haraka
    assert grade(site, committed("multiple", "fatha", "kasra")) == WRONG
    assert grade(site, SiteOutcome(False, None)) == NOT_GRADED  # no slot in the decode


def test_sukun_state_table_today():
    site = make(QULHU, 2, "sukun", "sukun", "sukun")
    assert grade(site, committed(None)) == CORRECT
    assert grade(site, committed("damma")) == WRONG


@pytest.mark.parametrize("letter", [WAW, YA, HAMZA])
def test_dropped_haraka_exemption_on_and_off(letter):
    site = make(letter + FATHA + KATABA, 0, "fatha", "fatha", "fatha")
    exemption = next(a for a in ALLOWANCES if a.name == "dropped_haraka_exemption")
    assert exemption.affects(site)
    assert grade(site, committed(None)) == NOT_GRADED
    assert grade(site, committed(None), exemption.switch_off(TODAY)) == WRONG
    assert grade(site, committed("kasra")) == WRONG  # a wrong haraka is never exempt


def test_suppress_haraka_drop_on_and_off():
    site = make(KATABA, 2, "fatha", "fatha", "fatha")
    allowance = next(a for a in ALLOWANCES if a.name == "suppress_haraka_drop")
    assert allowance.affects(site)
    assert grade(site, committed(None), replace(TODAY, suppress_haraka_drop=True)) == NOT_GRADED
    assert allowance.switch_off(TODAY) == TODAY  # already off today


def test_end_word_exempts_the_last_consonant():
    site = make(KATABA, 6, "fatha", "fatha", "fatha")
    assert at_pause(site)
    assert grade(site, committed("kasra")) == NOT_GRADED
    assert grade(site, committed("kasra"), replace(TODAY, end_word=False)) == WRONG
    pause = make(QULHU, 2, "sukun", "sukun", "sukun", stratum="waqf_boundary:waqf",
                 source="waqf_boundary")
    assert at_pause(pause) and grade(pause, committed("damma")) == NOT_GRADED


def test_tashkeel_toggle_off_grades_nothing():
    site = make(KATABA, 2, "fatha", "fatha", "fatha")
    assert grade(site, committed("kasra"), replace(TODAY, tashkeel_errors=False)) == NOT_GRADED


def test_empty_slot_not_graded_rule():
    candidate = replace(TODAY, empty_slot_not_graded=True)
    site = make(WAKATABA, 4, "fatha", "fatha", "fatha")
    assert grade(site, committed(None), candidate) == NOT_GRADED
    assert grade(site, committed("fatha"), candidate) == CORRECT
    assert grade(site, committed("kasra"), candidate) == WRONG
    sukun = make(QULHU, 2, "sukun", "sukun", "sukun")
    assert grade(sukun, committed("sukun"), candidate) == CORRECT
    assert grade(sukun, committed("damma"), candidate) == WRONG
    assert grade(sukun, committed(None), candidate) == NOT_GRADED


# --- shaddah ----------------------------------------------------------------------------


def test_shaddah_suppression_on_and_off():
    site = make(MADDA, 2, "shaddah", "held", "held")
    allowance = next(a for a in ALLOWANCES if a.name == "shaddah_suppression")
    assert allowance.affects(site)
    assert grade(site, committed("held")) == CORRECT
    assert grade(site, committed("not_held")) == NOT_GRADED
    assert grade(site, committed("not_held"), allowance.switch_off(TODAY)) == WRONG


def test_an_added_gemination_is_never_graded():
    site = make(MADA, 2, "shaddah", "not_held", "not_held")
    assert not next(a for a in ALLOWANCES if a.name == "shaddah_suppression").affects(site)
    for config in (TODAY, replace(TODAY, shaddah_suppression=False)):
        assert grade(site, committed("held"), config) == NOT_GRADED
        assert grade(site, committed("not_held"), config) == CORRECT


# --- pairs ------------------------------------------------------------------------------


def test_soft_pair_on_and_off_per_pair():
    site = make(DHAKARA, 2, SOFT, DHAL, DHAL)
    allowance = next(a for a in ALLOWANCES if a.name == f"soft_pair {SOFT}")
    assert allowance.affects(site)
    assert grade(site, committed(DHAL)) == CORRECT
    assert grade(site, committed(ZAI)) == NOT_GRADED
    assert grade(site, committed(ZAI), allowance.switch_off(TODAY)) == WRONG
    other = next(a for a in ALLOWANCES if a.name.startswith("soft_pair") and a is not allowance)
    assert not other.affects(site)
    assert grade(site, committed(ZAI), other.switch_off(TODAY)) == NOT_GRADED


def test_a_deletion_or_a_non_soft_swap_is_wrong():
    site = make(DHAKARA, 2, SOFT, DHAL, DHAL)
    assert grade(site, SiteOutcome(False, None)) == WRONG
    assert grade(site, committed(TAA)) == WRONG
    dhal_zah = replace(site, mark=DHAL_ZAH)
    assert grade(dhal_zah, committed(ZAH)) == WRONG  # ذ↔ظ is not a Muraja soft pair


def test_allowance_names_are_unique_and_cover_every_soft_pair():
    names = [a.name for a in ALLOWANCES]
    assert len(names) == len(set(names))
    assert sum(name.startswith("soft_pair") for name in names) == 6


def test_affected_populations_are_decided_by_the_truth_record():
    haraka = make(KATABA, 2, "fatha", "fatha", "fatha")
    sukun = make(QULHU, 2, "sukun", "sukun", "sukun")
    damma_on_waw = make(WAW + DAMMA + KATABA, 0, "damma", "damma", "fatha")
    affected = {a.name for a in ALLOWANCES if a.affects(haraka)}
    assert affected == {"suppress_haraka_drop"}
    assert not any(a.affects(sukun) for a in ALLOWANCES)
    assert {a.name for a in ALLOWANCES if a.affects(damma_on_waw)} == {"dropped_haraka_exemption"}


def test_a_collapsed_geminate_discards_its_tashkeel_in_every_mode():
    site = make(MADDA, 3, "fatha", "fatha", "fatha")  # fatha on the second د of مَددَرَ
    collapsed = SiteOutcome(True, "kasra", ("kasra",), geminate_collapsed=True)
    for suppression in (True, False):
        for empty_rule in (True, False):
            config = replace(TODAY, shaddah_suppression=suppression, empty_slot_not_graded=empty_rule)
            assert grade(site, collapsed, config) == NOT_GRADED
    assert grade(site, SiteOutcome(True, "kasra", ("kasra",))) == WRONG


def test_a_collapsed_geminate_read_from_a_real_decode():
    site = make(MADDA, 3, "fatha", "fatha", "fatha")
    decode = MADA.replace(DAL + FATHA, DAL + "ِ", 1)  # مَدِرَ: one د, a wrong haraka
    outcome = item_outcomes([site], decode)[site.site_id]
    assert outcome.geminate_collapsed
    assert grade(site, outcome) == NOT_GRADED
    assert grade(site, outcome, replace(TODAY, shaddah_suppression=False)) == NOT_GRADED
