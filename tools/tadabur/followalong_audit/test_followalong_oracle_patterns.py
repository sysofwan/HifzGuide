"""Pattern grouping (#308): findings grouped by cause, a candidate Muraja issue at 10 words.

A finding is one word the engine graded wrongly (a false flag or a miss). Grouping counts the
distinct ayah words a pattern covers — ``(surah, ayah, word)`` — not findings: the same word
flagged in strict and balanced, or in five clips of one ayah, is one word of the pattern.
"""

from __future__ import annotations

import pytest

from tadabur.followalong_audit.patterns import (
    CANDIDATE_WORDS,
    EXAMPLES,
    group_findings,
    issue_body,
)

DIMENSIONS = ("form", "error_kind", "position", "boundary")


def finding(surah: int = 2, ayah: int = 2, word: int = 1, *, mode: str = "strict",
            form: str = "لَاا", error_kind: str = "substitution", position: str = "middle",
            boundary: bool = False, item: str | None = None) -> dict:
    return {
        "item": item or f"{surah}_{ayah}_{word}.flac#0", "surah": surah, "ayah": ayah, "word": word,
        "mode": mode, "form": form, "error_kind": error_kind, "position": position,
        "boundary": boundary, "quality": "wrong", "statuses": [None],
    }


def twelve_words(**fields) -> list[dict]:
    return [finding(2, 10 + n, 3, **fields) for n in range(12)]


def test_the_candidate_threshold_is_ten_words():
    assert CANDIDATE_WORDS == 10


def test_ten_distinct_words_make_a_candidate_and_nine_do_not():
    ten = [finding(2, n + 1, 1) for n in range(10)]
    (pattern,) = group_findings(ten, DIMENSIONS)
    assert (pattern["words"], pattern["candidate"]) == (10, True)
    (pattern,) = group_findings(ten[:9], DIMENSIONS)
    assert (pattern["words"], pattern["candidate"]) == (9, False)


def test_one_word_flagged_in_both_modes_and_in_many_clips_counts_once():
    repeats = [finding(2, 2, 1, mode=mode, item=f"clip{n}")
               for mode in ("strict", "shipped_default") for n in range(6)]
    (pattern,) = group_findings(repeats, DIMENSIONS)
    assert pattern["words"] == 1
    assert pattern["findings"] == 12


def test_findings_that_differ_on_a_dimension_fall_in_separate_patterns():
    mixed = twelve_words(position="first") + twelve_words(position="last", form="رَببِ")
    patterns = group_findings(mixed, DIMENSIONS)
    assert len(patterns) == 2
    assert {pattern["key"]["position"] for pattern in patterns} == {"first", "last"}
    assert all(pattern["words"] == 12 for pattern in patterns)


def test_the_key_names_each_requested_dimension_with_its_value():
    (pattern,) = group_findings(twelve_words(boundary=True), DIMENSIONS)
    assert pattern["key"] == {"form": "لَاا", "error_kind": "substitution",
                              "position": "middle", "boundary": True}


def test_grouping_by_one_dimension_pools_the_others():
    mixed = twelve_words(position="first", form="a") + [
        finding(5, n + 1, 1, position="first", form="b") for n in range(12)]
    (pattern,) = group_findings(mixed, ("position",))
    assert pattern["key"] == {"position": "first"}
    assert (pattern["words"], pattern["findings"]) == (24, 24)


def test_pooled_patterns_count_the_distinct_ayah_words_not_the_findings():
    # Two forms on the same twelve ayah words: pooled by position they are twelve words.
    same = twelve_words(position="first", form="a") + twelve_words(position="first", form="b")
    (pattern,) = group_findings(same, ("position",))
    assert pattern["words"] == 12
    assert pattern["findings"] == 24


def test_patterns_are_ordered_by_words_then_key_and_ignore_input_order():
    big = twelve_words(form="big")
    small = [finding(3, n + 1, 1, form="small") for n in range(4)]
    mid = [finding(4, n + 1, 1, form="mid") for n in range(7)]
    forward = group_findings(big + small + mid, DIMENSIONS)
    assert [pattern["words"] for pattern in forward] == [12, 7, 4]
    assert group_findings(list(reversed(small + mid + big)), DIMENSIONS) == forward


def test_examples_are_the_first_findings_in_ayah_order_and_capped():
    (pattern,) = group_findings(list(reversed(twelve_words())), DIMENSIONS)
    assert EXAMPLES == 3
    assert [(e["surah"], e["ayah"], e["word"]) for e in pattern["examples"]] == \
        [(2, 10, 3), (2, 11, 3), (2, 12, 3)]
    assert pattern["examples"][0]["statuses"] == [None]  # the replay evidence rides along


def test_a_finding_without_a_requested_dimension_is_refused():
    broken = finding()
    del broken["position"]
    with pytest.raises(KeyError):
        group_findings([broken], DIMENSIONS)


def test_no_findings_make_no_patterns():
    assert group_findings([], DIMENSIONS) == []


def test_a_candidate_gets_a_ready_to_file_issue_body():
    (pattern,) = group_findings(twelve_words(position="last", boundary=True), DIMENSIONS)
    body = issue_body(pattern)
    assert "12 words" in body
    assert "2:10" in body and "2:11" in body and "2:12" in body  # the examples, by ayah
    assert "substitution" in body and "last" in body
    assert "لَاا" in body


def test_a_pattern_below_ten_words_has_no_issue_body():
    (pattern,) = group_findings([finding(2, n + 1, 1) for n in range(9)], DIMENSIONS)
    with pytest.raises(ValueError):
        issue_body(pattern)
