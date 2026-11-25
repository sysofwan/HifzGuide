"""Amendments A2-A5 as the pre-registration states them (#310, owner-approved 2026-11-24).

Committed before the document changes and **locked**: later turns may add tests but never weaken
or rewrite these. A1's document facts are pinned beside its code in
``test_followalong_prereg_coverage.py``; the attribution behaviour of A2 and A3 in
``test_followalong_prereg_nograde.py``. This file pins what the document says about them:

* A2 -- section 5 lists **M-nograde** with its definition, its side rule (follow-along when the
  Teacher arm graded the word, unknown when it did not) and the precedence
  ``M1r → M1 → M1u → M-nograde → M2 → M3``;
* A3 -- rule 5 is read on the Teacher-graded words only, the Teacher-ungraded count stays a reported
  bound, both extremes are reported, and the owner decides when they disagree;
* A4 -- section 7's headline miss rate is on S1 + S3 + S5-interior, with S4 and S5-edge as separate
  labelled bounds and the full-frame π-weighted estimate as a sensitivity with its interval; rule
  2's share is stated as the unweighted fraction of listened words, with the π-weighted share
  without S4 as a sensitivity;
* A5 -- the round-3 catch-loss check result, the edge-excluded sensitivity defined by the clip's
  ayah range, and section 5's M1 sentence read the right way round.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREREG = ROOT / "docs" / "followalong-audit-preregistration.md"


def _text() -> str:
    return PREREG.read_text(encoding="utf-8")


def _section(number: int) -> str:
    parts = re.split(r"^## (\d+)\. .*$", _text(), flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))[str(number)]


def _expect(text: str, *patterns: str) -> None:
    for pattern in patterns:
        assert re.search(pattern, text, flags=re.I | re.S), f"missing /{pattern}/"


def _bullet(text: str, label: str) -> str:
    match = re.search(rf"^- \*\*{re.escape(label)}\*\*.*?(?=^- \*\*|^\S|\Z)", text,
                      flags=re.M | re.S)
    assert match, f"no bullet for {label}"
    return match.group(0)


# --- A2 ------------------------------------------------------------------------------------------
def test_section_5_defines_m_nograde_and_its_side_rule():
    bullet = _bullet(_section(5), "M-nograde")
    _expect(bullet, r"ungraded", r"no (App )?status", r"any check",
            r"follow-along[^.\n]*Teacher[^.\n]*graded", r"unknown")


def test_section_5_gives_the_new_miss_precedence():
    _expect(_section(5), r"M1r\s*→\s*M1\s*→\s*M1u\s*→\s*M-nograde\s*→\s*M2\s*→\s*M3")


def test_section_5_says_an_unknown_no_grade_word_is_outside_rule_two_s_share():
    _expect(_section(5) + _section(6), r"unknown[^.\n]*(excluded|outside|neither|reported)")


# --- A3 ------------------------------------------------------------------------------------------
def test_rule_five_is_read_on_teacher_graded_words_with_both_extremes_reported():
    rule = _section(6)
    _expect(rule, r"Teacher[- ]graded", r"all-F2", r"all-F3", r"extremes?",
            r"(owner|Fable)[^.\n]*decide", r"disagree")


def test_the_teacher_ungraded_count_stays_a_reported_bound_and_on_the_model_side_for_rule_two():
    text = _section(5) + _section(6)
    _expect(text, r"Teacher-ungraded[^.\n]*(reported|bound)",
            r"model side[^.\n]*rule 2|rule 2[^.\n]*model side")


# --- A4 ------------------------------------------------------------------------------------------
def test_section_7_headline_miss_rate_is_on_s1_s3_and_the_interior_ungraded_words():
    text = _section(7)
    _expect(text, r"headline miss rate[^.]*S1[^.]*S3[^.]*S5-interior",
            r"S4[^.\n]*S5-edge[^.\n]*(separate|bound)|S5-edge[^.\n]*S4[^.\n]*(separate|bound)",
            r"full-frame[^.\n]*(π-weighted|weighted)[^.\n]*sensitivity",
            r"sensitivity[^.]*interval")


def test_section_7_does_not_put_s4_or_the_edge_words_in_the_headline():
    text = _section(7)
    assert not re.search(r"S4 contributes to \*\*both denominators\*\*", text)
    _expect(text, r"S4[^.\n]*(not|never)[^.\n]*headline",
            r"S5-edge[^.\n]*(not|never)[^.\n]*headline|edge[^.\n]*separate[^.\n]*bound")


def test_rule_two_states_that_its_share_is_the_unweighted_fraction_of_listened_words():
    text = _section(5) + _section(6)
    _expect(text, r"unweighted fraction[^.\n]*listened",
            r"π-weighted[^.\n]*without S4[^.\n]*sensitivity")


# --- A5 ------------------------------------------------------------------------------------------
def test_section_3_records_the_round_three_catch_loss_check_result():
    _expect(_section(3), r"round[- ]3[^\n]*(0 of (the )?150|matched 0)")
    assert not re.search(r"round 3's worklist is checked the same way", _section(3))


def test_the_edge_excluded_sensitivity_is_defined_by_the_clip_s_ayah_range():
    for number in (3, 7):
        text = _section(number)
        assert not re.search(r"first and last engine word", text)
    _expect(_section(7), r"without the first and last word of each clip's ayah range")


def test_section_5_describes_m1_as_the_engine_forgiving_a_deviation_the_check_decoded():
    bullet = _bullet(_section(5), "M1")
    assert not re.search(r"decoded the deviation away", bullet)
    _expect(bullet, r"forgiv", r"setting check", r"errors", r"score below 1")


def test_the_changelog_records_the_five_amendments_and_the_owner_s_approval():
    text = _text().split("## Changelog", 1)[1]
    for amendment in ("A1", "A2", "A3", "A4", "A5"):
        assert amendment in text
    _expect(text, r"2026-11-24", r"cross-check", r"owner")
