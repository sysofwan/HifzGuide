"""Acceptance tests for the follow-along audit's draw amendment (#311, doc-only slice).

The amendment fills in the details the frozen pre-registration (#310) leaves open before S5's
draw, and changes no frozen number or rule. These tests pin the facts the amendment states, not
its prose: the frozen document's SHA-256, the owner's approval, the seed's formula and shuffle
idiom, and the pilot arithmetic it quotes (recomputed with the repo's own ``allocate`` over the
committed main allocations).

Contract this file pins
-----------------------
* ``docs/followalong-audit-preregistration.md`` is byte-identical to the frozen text (SHA-256
  ``33d3ba26…``): an amendment, never an edit.
* ``docs/followalong-audit-draw-amendment.md`` is dated 2026-11-25, frozen before any draw, cites
  the frozen document by SHA-256, and carries the owner's issue-311 approval comment.
* It states ten numbered items, tagged **addition** (1 and 10) or **clarification** (2–9).
* The seed is the 64-bit big-endian prefix of ``sha256("followalong-audit-draw:" + M_hex)`` over
  the first ``main`` commit that adds the file, with decimal-seed sub-seeds; the shuffle idiom
  and ``draw_seed.json``'s ancestor-or-reproduce rule are spelled out in the document itself.
* The pilot numbers it quotes (56 / 69 / 17 / 8, S5 3 / 3 / 2, and S1's 56 split) are what
  ``allocate`` produces over the committed main allocations.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path

from tadabur.followalong_audit.allocation import allocate

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
AUDIT = ROOT / "tools" / "tadabur" / "followalong_audit"
PREREG = DOCS / "followalong-audit-preregistration.md"
AMENDMENT = DOCS / "followalong-audit-draw-amendment.md"

PREREG_SHA256 = "33d3ba26dbe6df3a72b0a03637934274a1c2526ac9e0dc306afd1545ea28ed39"
DATE = "2026-11-25"
APPROVAL = "https://github.com/sysofwan/HifzGuide/issues/311#issuecomment-7168352666"
KINDS = {1: "addition", 2: "clarification", 3: "clarification", 4: "clarification",
         5: "clarification", 6: "clarification", 7: "clarification", 8: "clarification",
         9: "clarification", 10: "addition"}
STRATUM_SIZES = {"S1": 200, "S3": 250, "S4": 60, "S5": 30}
S1_CAUSES = ("other_consonant", "tashkeel", "soft_pair", "skipped", "unrecorded", "shaddah",
             "target_pair")


def _text() -> str:
    assert AMENDMENT.is_file(), "docs/followalong-audit-draw-amendment.md does not exist"
    return AMENDMENT.read_text(encoding="utf-8")


def _titles() -> dict[int, str]:
    return {int(number): title for number, title in re.findall(r"^## (\d+)\. (.*)$", _text(),
                                                               flags=re.M)}


def _item(number: int) -> str:
    parts = re.split(r"^## (\d+)\. .*$", _text(), flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))[str(number)]


def _s1_cells() -> list[dict]:
    return json.loads((AUDIT / "app_arm" / "cells.json").read_text(encoding="utf-8"))[
        "s1_primary_cells"]


def _expect(text: str, *patterns: str) -> None:
    for pattern in patterns:
        assert re.search(pattern, text, flags=re.I | re.S), f"missing /{pattern}/"


# --- the frozen document stays untouched; the amendment is new, dated and approved ----------------
def test_the_frozen_preregistration_stays_byte_identical():
    assert sha256(PREREG.read_bytes()).hexdigest() == PREREG_SHA256


def test_the_amendment_is_dated_frozen_before_any_draw_and_cites_the_frozen_document():
    text = _text()
    assert text.splitlines()[0].endswith(f"({DATE})"), "the title is not dated"
    status = re.search(r"\*\*Status:\*\*(.*?)(?=\n\n)", text, flags=re.S).group(0)
    _expect(status, r"frozen", r"before any draw")
    _expect(text, r"followalong-audit-preregistration\.md", PREREG_SHA256, r"byte-identical")


def test_the_owner_approval_line_carries_the_issue_311_comment():
    text = _text()
    assert f"Owner approval: {APPROVAL}" in text
    cross_check = re.search(r"^Principal cross-check: (.+)$", text, flags=re.M).group(1)
    assert DATE in cross_check
    assert "pending" not in cross_check.lower()


def test_the_amendment_states_ten_items_each_tagged_addition_or_clarification():
    titles = _titles()
    assert sorted(titles) == list(range(1, 11))
    for number, kind in KINDS.items():
        assert kind in titles[number].lower(), f"item {number} is not tagged {kind}"


# --- the seed and the shuffle are pinned here, not in code written after the seed is known --------
def test_the_seed_is_the_first_adding_commit_hashed_to_a_64_bit_integer():
    text = _item(1)
    _expect(text, r"\*\*first\*\* commit on `main`", r"--diff-filter=A", r"revert and re-merge")
    for literal in ('sha256("followalong-audit-draw:" + M_hex)',
                    "first 8 bytes, big-endian", "64-bit unsigned integer",
                    "40-character lowercase", 'sha256(str(seed) + ":" + purpose)',
                    "`ids`", "`bootstrap`"):
        assert literal in text, literal


def test_the_shuffle_and_draw_seed_record_are_spelled_out_in_the_document():
    text = _item(1)
    for literal in ("ascending `(key, position)`", "random.Random(subseed).shuffle",
                    "draw_seed.json", "D must be an ancestor of M", "clean checkout",
                    "re-derivation test"):
        assert literal in text, literal


# --- no frozen number or rule moves ---------------------------------------------------------------
def test_the_amendment_quotes_the_frozen_sizes_and_the_regenerated_flip_report():
    text = _text()
    _expect(text, r"changes \*\*no frozen number and no rule\*\*")
    for literal in ("49,000", "540; pilot 150; S1 200 / S3 250 / S4 60 / S5 30; 25 repeats",
                    "157", "positions 1–130", "at least 20", "60 confirmed misses",
                    "any** stratum", "S1 ∪ S3 ∪ S5-interior", "4,000 resamples", "≥ 50%",
                    "≥ 3 instances in ≥ 3 clips", "S1 cells"):
        assert literal in text, literal
    _expect(text, r"`word_grades`\s+table")
    item_two = _item(2)
    _expect(item_two, r"did not reproduce", r"π = 0", r"simple random sample")


# --- the pilot arithmetic the amendment quotes, recomputed ----------------------------------------
def test_the_pilot_numbers_it_quotes_recompute_from_the_committed_main_allocations():
    pilot = allocate(STRATUM_SIZES, 150)
    assert pilot == {"S1": 56, "S3": 69, "S4": 17, "S5": 8}
    s5 = allocate({"interior": 10, "leading": 10, "trailing": 10}, 8)
    assert s5 == {"interior": 3, "leading": 3, "trailing": 2}

    cells = _s1_cells()
    split = allocate({f"{cell['cause']}|{cell['base_flags']}": cell["n"] for cell in cells}, 56)

    quoted = _item(3)
    assert f"{pilot['S1']} / {pilot['S3']} / {pilot['S4']} / {pilot['S5']}" in quoted
    assert (f"interior {s5['interior']}, leading {s5['leading']}, "
            f"trailing {s5['trailing']}") in quoted
    for cause in S1_CAUSES:
        assert f"{cause} {split[f'{cause}|False']} / {split[f'{cause}|True']}" in quoted, cause
