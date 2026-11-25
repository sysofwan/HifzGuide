"""Amendment A1 to the pre-registration: the audio-coverage gate on S5 (#310, owner-approved).

Committed before the change exists and **locked**: later turns may add tests but never weaken or
rewrite these. The principal's cross-check found that most of S5 is words the recording does not
hold, so S5's draw frame is the *covered* words, with the rest a reported pool. This file was
re-locked after principal-r3 ruled that coverage comes from the Teacher's transcript, never from
an engine grade, and applies to App-ungraded words only (it replaces the first lock's "base
detector graded" rule, which was circular: it read the same engine that left the word ungraded).

Contract this file pins
-----------------------
* ``coverage.teacher_text(row)``: the Teacher's whole-clip decode, the concatenated ``hop`` of the
  row's ``checks`` (an overlap is never part of it).
* ``coverage.covered_words(text, reference, spans)``: for each ``(start, end)`` span of the
  reference (code points, spaces counted, ``end`` exclusive), whether some **anchored** character
  of ``text`` aligns into it, with ``training.muraja_policy``'s engine-free alignment (spaces
  ignored, a run of at least three identical characters anchors). Spans and alignment share one
  coordinate system: the reference as the ayah stores it, spaces included.
* ``coverage.ayah_coverage(teacher_row, text, surah, ayah, words)``: ``{word: covered}`` for the
  Muraja phoneme ``words``, where ``text`` offers ``phonemes(surah, ayah)`` and
  ``word_groups(surah, ayah, word)`` (as ``MurajaText`` does) and a word's span is the minimum
  start to the maximum end of its groups.
* ``strata_run.with_transcript_coverage(words, coverage)``: each word gets
  ``transcript_covered`` -- ``coverage[(key, word)]`` for an App-ungraded word (``KeyError`` when
  missing) and ``None`` for any other, which the gate does not read. The input is not mutated.
* ``strata.covered(word)``: true for every App-graded word, whatever any detector graded;
  for an App-ungraded word its ``transcript_covered``.
* ``strata_run.draw_frame(stratified)``: the words the draw may pick from, sorted by
  ``(key, position)``: every word of S1 (S1-pending included), S3 and S4, and the covered words of
  S5. Each is ``{"key", "position", "stratum", "edge", "whole_played"}`` with ``stratum`` in
  ``S1 S3 S4 S5``; for S5 ``edge`` is ``leading`` (before the clip's first App-graded word, and
  every word of a wholly ungraded clip), ``trailing`` (after its last) or ``interior``, decided
  over *all* the clip's words, covered or not; otherwise ``None``. ``whole_played`` is true for the
  S5 leading and trailing words, served as whole clips. Residual and oracle-explained words and
  uncovered S5 words are not in the frame.
* ``strata_run.coverage_record(stratified)``: ``{"frame": {class: words}, "not_covered": {class:
  words}, "frame_words", "not_covered_words"}`` over ``S1 S3 S4 S5-leading S5-trailing
  S5-interior``; the three non-S5 classes are never uncovered.
* ``primary_cells`` counts every S1 word (the gate does not touch S1); ``cells_record`` carries
  ``"coverage"``.
* The committed ``app_arm/cells.json`` holds that record, recomputed at the pinned harness, and the
  pre-registration quotes it (sections 1, 3 and 7).
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import pytest

from tadabur.followalong_audit.allocation import PRIMARY_CAUSES
from tadabur.followalong_audit.coverage import ayah_coverage, covered_words, teacher_text
from tadabur.followalong_audit.strata import covered, stratify
from tadabur.followalong_audit.strata_run import (
    cells_record,
    coverage_record,
    draw_frame,
    primary_cells,
    with_transcript_coverage,
)

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "tools" / "tadabur" / "followalong_audit"
PREREG = ROOT / "docs" / "followalong-audit-preregistration.md"
CELLS = json.loads((AUDIT / "app_arm" / "cells.json").read_text(encoding="utf-8"))
POOLS = json.loads((AUDIT / "app_arm" / "pools.json").read_text(encoding="utf-8"))["pools"]["strata"]

DETECTORS = ("base", "c95", "r2_candidate_ema", "r3_candidate_ema")
CLASSES = ("S1", "S3", "S4", "S5-leading", "S5-trailing", "S5-interior")
S5_BUCKETS = ("S5-leading", "S5-trailing", "S5-interior")
#: S5's candidate composition (committed pools and word tables): leading / trailing / interior.
S5_CANDIDATES = {"S5-leading": 3933, "S5-trailing": 5103, "S5-interior": 306}
#: What the rule produces, with anchored indices and spans in one coordinate system (the ayah's
#: reference, spaces counted), from the committed gating decode and Muraja's reference at 028bc51.
#: Principal-r3's 1,961 / 1,305 / 210 came from counting the alignment's indices with spaces
#: stripped against spans that keep them; the orchestrator ruled that the rule stands.
EXPECTED_S5_FRAME = {"S5-leading": 2038, "S5-trailing": 1553, "S5-interior": 196}
EXPECTED_S5_NOT_COVERED = {"S5-leading": 1895, "S5-trailing": 3550, "S5-interior": 110}

REFERENCE = "abcde fghij klmno"
SPANS = [(0, 5), (6, 11), (12, 17)]


def word(key: str, position: int, app: str, *, flagging=(), oracle: str | None = "passed",
         covered_by_audio: bool = True, causes=(), detector_graded: bool = True) -> dict:
    return {"key": key, "position": position, "word": position + 1, "app": app, "oracle": oracle,
            "detectors": {name: name in flagging for name in DETECTORS},
            "detector_graded": dict.fromkeys(DETECTORS, detector_graded),
            "causes": list(causes),
            "transcript_covered": covered_by_audio if app == "ungraded" else None}


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
        word("a#0", 7, "ungraded", covered_by_audio=False),
        # b: the unsampled pools, and words no detector graded: the gate does not read them.
        word("b#0", 0, "passed", flagging=["c95"]),
        word("b#0", 1, "flagged", oracle="flagged", causes=["tashkeel"]),
        word("b#0", 2, "passed", detector_graded=False),
        word("b#0", 3, "flagged", causes=["tashkeel"], detector_graded=False),
        word("b#0", 4, "passed", flagging=["c95", "r3_candidate_ema"], detector_graded=False),
        # c: wholly ungraded, so every word is leading.
        word("c#0", 0, "ungraded"),
        word("c#0", 1, "ungraded"),
        word("c#0", 2, "ungraded", covered_by_audio=False),
        # d: S4 words around an interior ungraded word.
        word("d#0", 0, "passed"),
        word("d#0", 1, "ungraded"),
        word("d#0", 2, "passed"),
        # e: an uncovered leading word does not move the other words' edges.
        word("e#0", 0, "ungraded", covered_by_audio=False),
        word("e#0", 1, "passed"),
        word("e#0", 2, "ungraded"),
    ])


# --- the transcript rule ------------------------------------------------------------------------
def test_the_teacher_text_is_the_concatenated_hops_and_never_the_overlaps():
    row = {"checks": [{"hop": "ab", "overlap": "zz", "flush": False},
                      {"hop": "cd", "overlap": "", "flush": True}]}
    assert teacher_text(row) == "abcd"
    assert teacher_text({"checks": []}) == ""


@pytest.mark.parametrize("text, expected", [
    ("abcde fghij", [True, True, False]),
    ("abcde fghij klmno", [True, True, True]),
    ("klmno", [False, False, True]),  # spans and alignment share the ayah's coordinates
    ("xxfghxx", [False, True, False]),
    ("defgh", [True, True, False]),  # one anchored run across a word boundary
    ("fg", [False, False, False]),  # a run of two does not anchor
    ("ab xx de", [False, False, False]),  # two runs of two, neither anchors
    ("", [False, False, False]),
    ("zzzzz", [False, False, False]),
])
def test_a_word_is_covered_when_an_anchored_run_of_the_transcript_aligns_into_its_span(
        text, expected):
    assert covered_words(text, REFERENCE, SPANS) == expected


def test_spaces_in_the_transcript_do_not_change_what_is_covered():
    assert covered_words("abc de fg hij", REFERENCE, SPANS) == covered_words(
        "abcdefghij", REFERENCE, SPANS) == [True, True, False]


class FakeText:
    """What ``MurajaText`` offers: an ayah's phonemes and a phoneme word's group spans."""

    groups = {1: [(0, 2), (2, 5)], 2: [(6, 8), (8, 11)], 3: [(12, 14), (14, 17)]}

    def phonemes(self, surah: int, ayah: int) -> str:
        assert (surah, ayah) == (1, 2)
        return REFERENCE

    def word_groups(self, surah: int, ayah: int, word: int) -> list[tuple[int, int]]:
        assert (surah, ayah) == (1, 2)
        return self.groups[word]


def test_an_ayah_s_coverage_aligns_the_teacher_decode_to_each_word_s_group_span():
    row = {"checks": [{"hop": "abcde", "overlap": "zzz", "flush": False},
                      {"hop": " fghij", "overlap": "", "flush": True}]}
    assert ayah_coverage(row, FakeText(), 1, 2, [1, 2, 3]) == {1: True, 2: True, 3: False}


class GappedText(FakeText):
    groups = {**FakeText.groups, 3: [(12, 13), (16, 17)]}


def test_a_word_s_span_runs_from_its_first_group_start_to_its_last_group_end():
    # "lmn" aligns to reference indices 13-15, in the gap between the word's two groups.
    row = {"checks": [{"hop": "lmn", "overlap": "", "flush": False}]}
    assert ayah_coverage(row, GappedText(), 1, 2, [3]) == {3: True}
    assert ayah_coverage(row, FakeText(), 1, 2, [2]) == {2: False}


# --- covered, and where it applies -----------------------------------------------------------------
def test_an_app_ungraded_word_is_covered_when_its_transcript_alignment_says_so():
    assert covered(word("a#0", 0, "ungraded", covered_by_audio=True)) is True
    assert covered(word("a#0", 0, "ungraded", covered_by_audio=False)) is False


@pytest.mark.parametrize("app", ["passed", "flagged"])
def test_an_app_graded_word_is_always_covered_whatever_any_detector_graded(app):
    assert covered(word("a#0", 0, app, detector_graded=False)) is True


def test_no_detector_grade_decides_coverage():
    ungraded = word("a#0", 0, "ungraded", covered_by_audio=False)
    ungraded["detector_graded"] = dict.fromkeys(DETECTORS, True)
    assert covered(ungraded) is False
    ungraded = word("a#0", 0, "ungraded", covered_by_audio=True)
    ungraded["detector_graded"] = dict.fromkeys(DETECTORS, False)
    assert covered(ungraded) is True


def test_an_ungraded_word_without_a_transcript_coverage_is_refused():
    bare = word("a#0", 0, "ungraded")
    del bare["transcript_covered"]
    with pytest.raises(KeyError):
        covered(bare)


def test_the_coverage_is_joined_on_the_clip_and_the_muraja_word_for_ungraded_words_only():
    words = [{"key": "a#0", "word": 1, "app": "ungraded"},
             {"key": "a#0", "word": 2, "app": "passed"},
             {"key": "b#0", "word": 1, "app": "ungraded"}]
    joined = with_transcript_coverage(words, {("a#0", 1): True, ("b#0", 1): False})
    assert [row["transcript_covered"] for row in joined] == [True, None, False]
    assert all("transcript_covered" not in row for row in words)


def test_an_ungraded_word_the_transcript_never_assessed_is_refused():
    with pytest.raises(KeyError):
        with_transcript_coverage([{"key": "a#0", "word": 1, "app": "ungraded"}], {})


# --- the draw frame -----------------------------------------------------------------------------
def test_the_draw_frame_holds_every_graded_sampled_word_and_the_covered_ungraded_ones():
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
        ("b#0", 2): ("S4", None, False),
        ("b#0", 3): ("S1", None, False),
        ("b#0", 4): ("S3", None, False),
        ("c#0", 0): ("S5", "leading", True),
        ("c#0", 1): ("S5", "leading", True),
        ("d#0", 0): ("S4", None, False),
        ("d#0", 1): ("S5", "interior", False),
        ("d#0", 2): ("S4", None, False),
        ("e#0", 1): ("S4", None, False),
        ("e#0", 2): ("S5", "trailing", True),
    }


def test_the_draw_frame_leaves_out_uncovered_s5_words_and_the_unsampled_pools():
    keys = {(row["key"], row["position"]) for row in draw_frame(clips())}
    assert {("a#0", 7), ("c#0", 2), ("e#0", 0), ("b#0", 0), ("b#0", 1)}.isdisjoint(keys)


def test_a_graded_word_no_detector_graded_is_still_in_the_frame():
    keys = {(row["key"], row["position"]) for row in draw_frame(clips())}
    assert {("b#0", 2), ("b#0", 3), ("b#0", 4)} <= keys


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
    edges = {row["position"]: row["edge"] for row in draw_frame(clips()) if row["key"] == "e#0"}
    assert edges == {1: None, 2: "trailing"}


# --- the coverage record ------------------------------------------------------------------------
def test_the_coverage_record_counts_the_frame_and_the_uncovered_pool_by_class():
    record = coverage_record(clips())
    assert record["frame"] == {"S1": 3, "S3": 2, "S4": 5, "S5-leading": 3, "S5-trailing": 2,
                               "S5-interior": 2}
    assert record["not_covered"] == {"S1": 0, "S3": 0, "S4": 0, "S5-leading": 2,
                                     "S5-trailing": 1, "S5-interior": 0}
    assert (record["frame_words"], record["not_covered_words"]) == (17, 3)


def test_the_coverage_record_agrees_with_the_draw_frame():
    rows = clips()
    assert coverage_record(rows)["frame_words"] == len(draw_frame(rows))


# --- the cells ----------------------------------------------------------------------------------
def s1_rows(*, half_ungraded_by_detectors: bool) -> list[dict]:
    serial = itertools.count()
    words = []
    for cause in PRIMARY_CAUSES:
        for base in (False, True):
            for index in range(20):
                graded = not (half_ungraded_by_detectors and index % 2)
                words.append(word(f"s{next(serial)}#0", 0, "flagged", causes=[cause],
                                  flagging=["base"] if base else (), detector_graded=graded))
    return stratify(words)


def test_the_s1_cells_count_every_s1_word_the_gate_does_not_touch_s1():
    assert primary_cells(s1_rows(half_ungraded_by_detectors=True)) == primary_cells(
        s1_rows(half_ungraded_by_detectors=False))
    cells = primary_cells(s1_rows(half_ungraded_by_detectors=True))
    assert len(cells) == 14 and all(cell["words"] == 20 for cell in cells)
    assert sum(cell["n"] for cell in cells) == 200


def test_the_cells_record_carries_the_coverage_record():
    rows = s1_rows(half_ungraded_by_detectors=True)
    record = cells_record(rows, detector_ungraded=0)
    assert record["coverage"] == coverage_record(rows)
    assert record["s1_primary_cells"] == primary_cells(rows)


# --- the committed record -----------------------------------------------------------------------
def test_the_committed_gate_leaves_s1_s3_and_s4_whole():
    coverage = CELLS["coverage"]
    assert coverage["frame"]["S1"] == POOLS["S1"] + POOLS["S1-pending"] == 5156
    assert coverage["frame"]["S3"] == POOLS["S3"] == 1459
    assert coverage["frame"]["S4"] == POOLS["S4"] == 38598
    assert all(coverage["not_covered"][name] == 0 for name in ("S1", "S3", "S4"))


def test_every_s5_edge_class_splits_into_covered_and_not_covered_exactly():
    coverage = CELLS["coverage"]
    for name, candidates in S5_CANDIDATES.items():
        assert coverage["frame"][name] + coverage["not_covered"][name] == candidates
    assert sum(S5_CANDIDATES.values()) == POOLS["S5"]


def test_the_committed_frame_totals_are_the_sums_of_its_classes():
    coverage = CELLS["coverage"]
    assert set(coverage["frame"]) == set(coverage["not_covered"]) == set(CLASSES)
    assert coverage["frame_words"] == sum(coverage["frame"].values())
    assert coverage["not_covered_words"] == sum(coverage["not_covered"].values())
    assert coverage["frame_words"] + coverage["not_covered_words"] == 54555


def test_the_draw_can_pay_its_s5_buckets_from_the_covered_words():
    for name in S5_BUCKETS:
        assert CELLS["coverage"]["frame"][name] >= 10


def test_the_s1_cells_are_the_approved_table_the_gate_does_not_change():
    counts = {(row["cause"], row["base_flags"]): (row["words"], row["n"])
              for row in CELLS["s1_primary_cells"]}
    assert sum(words for words, _ in counts.values()) == 5156
    assert counts[("other_consonant", True)] == (1607, 61)
    assert counts[("target_pair", False)] == (24, 3)


def test_the_s4_split_still_describes_the_whole_candidate_s4_pool():
    assert CELLS["s4"]["words"] == CELLS["coverage"]["frame"]["S4"]


def test_the_committed_s5_frame_is_what_the_transcript_rule_produces():
    coverage = CELLS["coverage"]
    assert {name: coverage["frame"][name] for name in S5_BUCKETS} == EXPECTED_S5_FRAME
    assert {name: coverage["not_covered"][name] for name in S5_BUCKETS} == EXPECTED_S5_NOT_COVERED
    assert sum(EXPECTED_S5_FRAME.values()) == 3787 and sum(EXPECTED_S5_NOT_COVERED.values()) == 5555
    assert coverage["frame_words"] == 5156 + 1459 + 38598 + 3787 == 49000
    assert coverage["not_covered_words"] == 5555


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


def test_section_1_defines_the_coverage_gate_by_the_transcript_and_never_an_engine_grade():
    text = _section(1)
    for pattern in (r"cover(age|ed)", r"Teacher[^.\n]*(whole-clip )?(decode|transcript)",
                    r"align", r"no engine grade", r"App-ungraded|S5 only|only[^.\n]*S5",
                    r"not covered by audio", r"π = 0", r"reported", r"before the draw"):
        assert re.search(pattern, text, flags=re.I), pattern
    assert f"{CELLS['coverage']['frame_words']:,}" in text
    assert f"{CELLS['coverage']['not_covered_words']:,}" in text


def test_section_3_quotes_the_covered_and_the_not_covered_count_of_every_class():
    rows = {row[0]: row for row in _rows(_section(3)) if _numeric(row) and row[0] in CLASSES}
    assert set(rows) == set(CLASSES)
    for name in CLASSES:
        assert (_int(rows[name][1]), _int(rows[name][2])) == (
            CELLS["coverage"]["frame"][name], CELLS["coverage"]["not_covered"][name])


def test_section_3_draws_s5_from_covered_words_and_serves_its_edge_snips_whole():
    text = _section(3)
    for pattern in (r"S5[^.\n]*(leading|trailing)[^.\n]*cover", r"whole clips?",
                    r"whole_played\s*=\s*true"):
        assert re.search(pattern, text, flags=re.I), pattern


def test_section_3_discloses_the_s5_words_the_engine_left_ungraded_but_the_transcript_covers():
    assert re.search(r"engine left[^.\n]*ungraded[^.\n]*transcript[^.\n]*\d", _section(3),
                     flags=re.I | re.S)


def test_section_3_sizes_every_stratum_by_its_frame():
    rows = {row[0].split(":")[0]: row for row in _rows(_section(3))
            if len(row) >= 5 and re.match(r"S[1345]:", row[0])}
    sizes = {"S1": 200, "S3": 250, "S4": 60, "S5": 30}
    frame = {"S1": 5156, "S3": 1459, "S4": 38598,
             "S5": sum(CELLS["coverage"]["frame"][name] for name in S5_BUCKETS)}
    for name, size in sizes.items():
        assert _int(rows[name][1]) == frame[name]
        assert rows[name][4] == f"{size / frame[name]:.3g}"


def test_section_3_s5_buckets_use_the_covered_words_for_their_inclusion_probability():
    rows = {row[0]: row for row in _rows(_section(3))
            if len(row) >= 4 and row[0] in ("leading", "trailing", "interior")}
    for bucket in ("leading", "trailing", "interior"):
        words = CELLS["coverage"]["frame"][f"S5-{bucket}"]
        assert _int(rows[bucket][1]) == words
        assert rows[bucket][3] == f"{10 / words:.3g}"


def test_section_7_names_the_covered_frame_and_the_reported_pool():
    text = _section(7)
    assert f"**{CELLS['coverage']['frame_words']:,} words**" in text
    for pattern in (r"not covered by audio", r"π = 0", r"sampled subpopulation"):
        assert re.search(pattern, text), pattern
    assert f"{CELLS['coverage']['not_covered_words']:,}" in text
