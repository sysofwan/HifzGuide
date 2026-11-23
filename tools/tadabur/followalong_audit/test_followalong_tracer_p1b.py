"""Gate P1b (#307): the 30-character overlap clip of Muraja's ``.log`` against the full overlap.

``TranscribeLog.swift:43-46`` cuts every overlap to ``String.prefix(30)`` Characters. The gate
replays the same clips from clipped and from full overlaps and asks whether any reported word's
kept quality differs, per mode.
"""

from __future__ import annotations

import copy

from tadabur.followalong_audit.p1b import changed_decisions, clip_overlaps
from training.decoding import clusters

LONG = "بَتِ" * 20  # 40 Swift Characters, 80 scalars
CHECKS = [
    {"hop": "", "overlap": LONG, "flush": False},
    {"hop": LONG, "overlap": LONG, "flush": False},
    {"hop": LONG, "overlap": "", "flush": True},
    {"hop": LONG, "overlap": "", "flush": False, "stop": True},
]


def test_an_overlap_is_cut_to_thirty_characters_not_thirty_scalars():
    clipped = clip_overlaps(CHECKS)
    assert clipped[0]["overlap"] == "بَتِ" * 15
    assert len(clusters(clipped[0]["overlap"])) == 30
    assert len(clipped[0]["overlap"]) == 60


def test_the_cut_never_separates_a_haraka_from_its_consonant():
    for limit in range(1, 41):
        overlap = clip_overlaps(CHECKS, limit=limit)[1]["overlap"]
        assert len(clusters(overlap)) == limit
        assert LONG.startswith(overlap)


def test_only_the_overlap_is_cut():
    clipped = clip_overlaps(CHECKS)
    assert [(c["hop"], c["flush"], c.get("stop")) for c in clipped] == \
        [(c["hop"], c["flush"], c.get("stop")) for c in CHECKS]
    assert clipped[2]["overlap"] == "" and clipped[3]["overlap"] == ""


def test_an_overlap_within_the_limit_is_untouched():
    short = [{"hop": "", "overlap": "بَتِ" * 15, "flush": False}]
    assert clip_overlaps(short) == short


def test_the_full_checks_are_not_modified_in_place():
    before = copy.deepcopy(CHECKS)
    clip_overlaps(CHECKS)
    assert CHECKS == before


def row(item: str, scoring: str, word: int, quality: str) -> dict:
    return {"item": item, "scoring": scoring, "word": word, "quality": quality, "score": 0.9}


def test_a_decision_that_differs_is_reported_with_both_qualities():
    full = [row("c1", "strict", 1, "correct"), row("c1", "strict", 2, "correct"),
            row("c1", "shipped_default", 2, "minor")]
    clipped = [row("c1", "strict", 1, "correct"), row("c1", "strict", 2, "tashkeelError"),
               row("c1", "shipped_default", 2, "minor")]
    assert changed_decisions(full, clipped) == [
        {"item": "c1", "scoring": "strict", "word": 2, "full": "correct", "clipped": "tashkeelError"}]


def test_identical_decisions_report_nothing_whatever_the_row_order():
    full = [row("c1", "strict", 1, "correct"), row("c2", "strict", 1, "minor")]
    assert changed_decisions(full, list(reversed(full))) == []


def test_a_word_the_clipped_run_leaves_ungraded_is_a_difference_not_an_error():
    full = [row("c1", "strict", 1, "correct"), row("c1", "strict", 2, "minor")]
    clipped = [row("c1", "strict", 1, "correct")]
    assert changed_decisions(full, clipped) == [
        {"item": "c1", "scoring": "strict", "word": 2, "full": "minor", "clipped": None}]


def test_a_word_only_the_clipped_run_grades_is_a_difference_not_an_error():
    full = [row("c1", "strict", 1, "correct")]
    clipped = [row("c1", "strict", 1, "correct"), row("c1", "strict", 2, "minor")]
    assert changed_decisions(full, clipped) == [
        {"item": "c1", "scoring": "strict", "word": 2, "full": None, "clipped": "minor"}]


def test_differences_come_in_item_scoring_word_order_whichever_run_lacks_the_word():
    full = [row("c2", "strict", 1, "minor"), row("c1", "shipped_default", 3, "correct")]
    clipped = [row("c1", "strict", 2, "correct"), row("c2", "strict", 1, "correct")]
    assert [(d["item"], d["scoring"], d["word"]) for d in changed_decisions(full, clipped)] == [
        ("c1", "shipped_default", 3), ("c1", "strict", 2), ("c2", "strict", 1)]
