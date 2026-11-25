"""Amendment A1 to the pre-registration: the per-word audio-coverage gate (#310, owner-approved).

Committed before the change exists and **locked**: later turns may add tests but never weaken or
rewrite these. The principal's cross-check found that most of S5 is words the recording does not
hold, so the draw frame must be the *covered* words, with the rest a reported pool.

Contract this file pins
-----------------------
* ``strata.covered(word)``: a word is **covered** when the Teacher's whole-clip decode graded it
  -- the **base** detector's ``detector_graded["base"]``. (The inventory's all-four-detectors row
  is a stricter proxy that would leave 138 S5 words, 7 of them interior, so it is not the rule.)
* ``strata_run.draw_frame(stratified)``: the words the draw may pick from, sorted by
  ``(key, position)``: the covered words of S1 (S1-pending included), S3, S4 and S5. Each is
  ``{"key", "position", "stratum", "edge", "whole_played"}`` with ``stratum`` in
  ``S1 S3 S4 S5``; for S5 ``edge`` is ``leading`` (before the clip's first App-graded word, and
  every word of a wholly ungraded clip), ``trailing`` (after its last) or ``interior``, decided
  over *all* the clip's words, covered or not; otherwise ``None``. ``whole_played`` is true for the
  S5 leading and trailing words -- served as whole clips -- and false for every other word.
  Residual and oracle-explained words, and uncovered words, are not in the frame.
* ``strata_run.coverage_record(stratified)``: ``{"frame": {class: covered words},
  "not_covered": {class: uncovered words}, "frame_words", "not_covered_words"}`` over the six
  classes ``S1 S3 S4 S5-leading S5-trailing S5-interior`` (the unsampled pools are not counted).
* ``primary_cells`` counts only covered S1 words; ``cells_record`` carries ``"coverage"``.
* The committed ``app_arm/cells.json`` holds that record, recomputed at the pinned harness, and the
  pre-registration quotes it (sections 1, 3 and 7).
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

from tadabur.followalong_audit.allocation import PRIMARY_CAUSES, allocate
from tadabur.followalong_audit.strata import covered, stratify
from tadabur.followalong_audit.strata_run import (
    cells_record,
    coverage_record,
    draw_frame,
    primary_cells,
)

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "tools" / "tadabur" / "followalong_audit"
PREREG = ROOT / "docs" / "followalong-audit-preregistration.md"
CELLS = json.loads((AUDIT / "app_arm" / "cells.json").read_text(encoding="utf-8"))
POOLS = json.loads((AUDIT / "app_arm" / "pools.json").read_text(encoding="utf-8"))["pools"]["strata"]

DETECTORS = ("base", "c95", "r2_candidate_ema", "r3_candidate_ema")
CLASSES = ("S1", "S3", "S4", "S5-leading", "S5-trailing", "S5-interior")

#: The committed record, as the strata run at 028bc51 computed it.
EXPECTED_FRAME = {"S1": 4697, "S3": 1447, "S4": 38047, "S5-leading": 510, "S5-trailing": 549,
                  "S5-interior": 148}
EXPECTED_NOT_COVERED = {"S1": 459, "S3": 12, "S4": 551, "S5-leading": 3423,
                        "S5-trailing": 4554, "S5-interior": 158}


def word(key: str, position: int, app: str, *, flagging=(), oracle: str | None = "passed",
         base_graded: bool = True, causes=()) -> dict:
    return {"key": key, "position": position, "app": app, "oracle": oracle,
            "detectors": {name: name in flagging for name in DETECTORS},
            "detector_graded": {name: name != "base" or base_graded for name in DETECTORS},
            "causes": list(causes)}


def clips() -> list[dict]:
    return stratify([
        # a: leading 0, interior 2, trailing 6 and 7 (7 is not covered); S1, S1-pending, S3, S4.
        word("a#0", 0, "ungraded"),
        word("a#0", 1, "passed"),
        word("a#0", 2, "ungraded"),
        word("a#0", 3, "flagged", causes=["tashkeel"]),
        word("a#0", 4, "passed", flagging=["base"]),
        word("a#0", 5, "flagged", oracle=None, causes=["skipped"]),
        word("a#0", 6, "ungraded"),
        word("a#0", 7, "ungraded", base_graded=False),
        # b: the unsampled pools and one uncovered word of each sampled stratum.
        word("b#0", 0, "passed", flagging=["c95"]),
        word("b#0", 1, "flagged", oracle="flagged", causes=["tashkeel"]),
        word("b#0", 2, "passed", base_graded=False),
        word("b#0", 3, "flagged", causes=["tashkeel"], base_graded=False),
        word("b#0", 4, "passed", flagging=["c95", "r3_candidate_ema"], base_graded=False),
        # c: wholly ungraded, so every word is leading; d: coverage does not move the edges.
        word("c#0", 0, "ungraded"),
        word("c#0", 1, "ungraded"),
        word("c#0", 2, "ungraded", base_graded=False),
        word("d#0", 0, "passed", base_graded=False),
        word("d#0", 1, "ungraded"),
        word("d#0", 2, "passed"),
    ])


# --- the rule -----------------------------------------------------------------------------------
def test_a_word_is_covered_when_the_teacher_decode_graded_it():
    assert covered(word("a#0", 0, "passed")) is True
    assert covered(word("a#0", 0, "passed", base_graded=False)) is False


def test_only_the_base_detector_decides_coverage():
    only_others = word("a#0", 0, "passed", base_graded=False)
    assert all(only_others["detector_graded"][name] for name in DETECTORS[1:])
    assert covered(only_others) is False
    stricter = word("a#0", 0, "passed")
    stricter["detector_graded"]["c95"] = False
    assert covered(stricter) is True


# --- the draw frame -----------------------------------------------------------------------------
def test_the_draw_frame_holds_the_covered_words_of_the_four_sampled_strata():
    frame = {(row["key"], row["position"]): (row["stratum"], row["edge"], row["whole_played"])
             for row in draw_frame(clips())}
    assert frame == {
        ("a#0", 0): ("S5", "leading", True),
        ("a#0", 1): ("S4", None, False),
        ("a#0", 2): ("S5", "interior", False),
        ("a#0", 3): ("S1", None, False),
        ("a#0", 4): ("S3", None, False),
        ("a#0", 5): ("S1", None, False),
        ("a#0", 6): ("S5", "trailing", True),
        ("c#0", 0): ("S5", "leading", True),
        ("c#0", 1): ("S5", "leading", True),
        ("d#0", 1): ("S5", "interior", False),
        ("d#0", 2): ("S4", None, False),
    }


def test_the_draw_frame_leaves_out_uncovered_words_and_the_unsampled_pools():
    keys = {(row["key"], row["position"]) for row in draw_frame(clips())}
    assert ("a#0", 7) not in keys and ("c#0", 2) not in keys and ("d#0", 0) not in keys
    assert not keys & {("b#0", position) for position in range(5)}


def test_the_draw_frame_is_sorted_by_clip_then_position():
    rows = draw_frame(list(reversed(clips())))
    assert [(row["key"], row["position"]) for row in rows] == sorted(
        (row["key"], row["position"]) for row in rows)


def test_only_s5_edge_words_are_served_as_whole_clips():
    rows = draw_frame(clips())
    assert {(row["stratum"], row["edge"]) for row in rows if row["whole_played"]} == {
        ("S5", "leading"), ("S5", "trailing")}
    assert all(row["edge"] is None for row in rows if row["stratum"] != "S5")


def test_an_edge_is_decided_over_every_word_of_the_clip_covered_or_not():
    assert {row["edge"] for row in draw_frame(clips()) if row["key"] == "d#0"} == {None, "interior"}


# --- the coverage record ------------------------------------------------------------------------
def test_the_coverage_record_counts_the_frame_and_the_uncovered_pool_by_class():
    record = coverage_record(clips())
    assert record["frame"] == {"S1": 2, "S3": 1, "S4": 2, "S5-leading": 3, "S5-trailing": 1,
                               "S5-interior": 2}
    assert record["not_covered"] == {"S1": 1, "S3": 1, "S4": 2, "S5-leading": 1,
                                     "S5-trailing": 1, "S5-interior": 0}
    assert (record["frame_words"], record["not_covered_words"]) == (11, 6)


def test_the_coverage_record_agrees_with_the_draw_frame():
    rows = clips()
    frame = draw_frame(rows)
    assert coverage_record(rows)["frame_words"] == len(frame)


# --- the cells ----------------------------------------------------------------------------------
def s1_rows(uncovered: int) -> list[dict]:
    serial = itertools.count()
    words = []
    for cause in PRIMARY_CAUSES:
        for base in (False, True):
            for covered_word in [True] * 20 + [False] * uncovered:
                # An uncovered word was never graded by the base detector, so it cannot flag.
                words.append(word(f"s{next(serial)}#0", 0, "flagged", causes=[cause],
                                  flagging=["base"] if base and covered_word else (),
                                  base_graded=covered_word))
    return stratify(words)


def test_the_s1_cells_count_covered_words_only():
    cells = primary_cells(s1_rows(uncovered=5))
    assert len(cells) == 14
    assert all(cell["words"] == 20 for cell in cells)
    assert sum(cell["n"] for cell in cells) == 200


def test_the_cells_record_carries_the_coverage_record():
    rows = s1_rows(uncovered=3)
    record = cells_record(rows, detector_ungraded=0)
    assert record["coverage"] == coverage_record(rows)
    assert record["s1_primary_cells"] == primary_cells(rows)


# --- the committed record -----------------------------------------------------------------------
def test_the_committed_record_is_the_one_the_strata_run_computed():
    coverage = CELLS["coverage"]
    assert coverage["frame"] == EXPECTED_FRAME
    assert coverage["not_covered"] == EXPECTED_NOT_COVERED
    assert coverage["frame_words"] == sum(EXPECTED_FRAME.values()) == 45398
    assert coverage["not_covered_words"] == sum(EXPECTED_NOT_COVERED.values()) == 9157


def test_every_candidate_pool_splits_into_covered_and_not_covered_exactly():
    coverage = CELLS["coverage"]
    both = {name: coverage["frame"][name] + coverage["not_covered"][name] for name in CLASSES}
    assert both["S1"] == POOLS["S1"] + POOLS["S1-pending"]
    assert both["S3"] == POOLS["S3"] and both["S4"] == POOLS["S4"]
    assert both["S5-leading"] + both["S5-trailing"] + both["S5-interior"] == POOLS["S5"]
    assert coverage["frame_words"] + coverage["not_covered_words"] == 54555


def test_the_s4_split_still_describes_the_whole_candidate_s4_pool():
    assert CELLS["s4"]["words"] == CELLS["coverage"]["frame"]["S4"] + CELLS["coverage"][
        "not_covered"]["S4"]


def test_the_committed_cells_are_the_covered_s1_words_with_the_floor_and_largest_remainder_split():
    counts = {(row["cause"], row["base_flags"]): row["words"] for row in CELLS["s1_primary_cells"]}
    assert sum(counts.values()) == CELLS["coverage"]["frame"]["S1"] == 4697
    quoted = {(row["cause"], row["base_flags"]): row["n"] for row in CELLS["s1_primary_cells"]}
    assert quoted == allocate(counts, 200, floor=3)


def test_the_draw_can_pay_its_s5_buckets_from_the_covered_words():
    for bucket in ("leading", "trailing", "interior"):
        assert CELLS["coverage"]["frame"][f"S5-{bucket}"] >= 10


# --- the document -------------------------------------------------------------------------------
def _section(number: int) -> str:
    parts = re.split(r"^## (\d+)\. .*$", PREREG.read_text(encoding="utf-8"), flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))[str(number)]


def _rows(section: str) -> list[list[str]]:
    return [[cell.strip().strip("`*") for cell in line.strip().strip("|").split("|")]
            for line in section.splitlines() if line.strip().startswith("|")]


def _int(cell: str) -> int:
    return int(cell.replace(",", ""))


def _numeric(row: list[str]) -> bool:
    return len(row) >= 3 and all(re.fullmatch(r"[\d,]+", cell) for cell in row[1:3])


def test_section_1_defines_the_coverage_gate_and_the_shrunk_population():
    text = _section(1)
    for pattern in (r"cover(age|ed)", r"Teacher[^.\n]*(whole-clip )?decode[^.\n]*grad",
                    r"not covered by audio", r"π = 0", r"reported", r"before the draw"):
        assert re.search(pattern, text, flags=re.I), pattern
    assert f"{CELLS['coverage']['frame_words']:,}" in text
    assert f"{CELLS['coverage']['not_covered_words']:,}" in text


def test_section_3_quotes_the_covered_and_the_not_covered_count_of_every_class():
    rows = {row[0]: row for row in _rows(_section(3))
            if _numeric(row) and row[0] in CLASSES}
    assert set(rows) == set(CLASSES)
    for name in CLASSES:
        assert (_int(rows[name][1]), _int(rows[name][2])) == (EXPECTED_FRAME[name],
                                                               EXPECTED_NOT_COVERED[name])


def test_section_3_draws_s5_from_covered_words_and_serves_its_edge_snips_whole():
    text = _section(3)
    for pattern in (r"S5[^.\n]*(leading|trailing)[^.\n]*cover", r"whole clips?",
                    r"whole_played\s*=\s*true"):
        assert re.search(pattern, text, flags=re.I), pattern


def test_section_3_sizes_every_stratum_by_its_covered_pool():
    rows = {row[0].split(":")[0]: row for row in _rows(_section(3))
            if len(row) >= 5 and re.match(r"S[1345]:", row[0])}
    sizes = {"S1": 200, "S3": 250, "S4": 60, "S5": 30}
    frame = {"S1": EXPECTED_FRAME["S1"], "S3": EXPECTED_FRAME["S3"], "S4": EXPECTED_FRAME["S4"],
             "S5": sum(EXPECTED_FRAME[f"S5-{bucket}"]
                       for bucket in ("leading", "trailing", "interior"))}
    for name, size in sizes.items():
        assert _int(rows[name][1]) == frame[name]
        assert rows[name][4] == f"{size / frame[name]:.3g}"


def test_section_3_s5_buckets_use_the_covered_words_for_their_inclusion_probability():
    rows = {row[0]: row for row in _rows(_section(3))
            if len(row) >= 4 and row[0] in ("leading", "trailing", "interior")}
    for bucket in ("leading", "trailing", "interior"):
        words = EXPECTED_FRAME[f"S5-{bucket}"]
        assert _int(rows[bucket][1]) == words
        assert rows[bucket][3] == f"{10 / words:.3g}"


def test_section_7_names_the_covered_frame_and_the_reported_pool():
    text = _section(7)
    assert f"**{CELLS['coverage']['frame_words']:,} words**" in text
    for pattern in (r"not covered by audio", r"π = 0", r"sampled subpopulation"):
        assert re.search(pattern, text), pattern
    assert f"{CELLS['coverage']['not_covered_words']:,}" in text
