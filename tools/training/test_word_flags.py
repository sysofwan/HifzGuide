"""Why the app flags a sampled word (#90): the cause of each kept-grade error, and the word
sample's per-scoring view. Torch-free, on synthetic grades."""

from __future__ import annotations

import pytest

from tadabur.exclusions import Exclusions
from tadabur.word_sample import SampledWord, VerifiedWord, WordSite
from training import word_flags as w
from training.muraja_policy import (
    FLAGGED_ELSEWHERE,
    MINOR,
    NOT_FLAGGED,
    NOT_GRADED,
    TASHKEEL_ERROR,
    SiteGrade,
)

FATHA = "َ"


def mismatch(expected: str, heard: str, group: int = 0) -> dict:
    return {"kind": {"type": "mismatch", "expected": expected, "heard": heard}, "group_index": group}


def deletion(expected: str, group: int = 0) -> dict:
    return {"kind": {"type": "deletion", "expected": expected}, "group_index": group}


def tashkeel(expected: str | None, heard: str | None, group: int = 0) -> dict:
    return {"kind": {"type": "tashkeel", "expected": expected, "heard": heard}, "group_index": group}


@pytest.mark.parametrize("error, cause", [
    (mismatch("ذ", "ز"), w.SOFT_PAIR), (mismatch("ك", "ق"), w.SOFT_PAIR),
    (mismatch("ط", "ق"), w.TARGET_PAIR), (mismatch("ش", "س"), w.TARGET_PAIR),
    (mismatch("ظ", "ذ"), w.TARGET_PAIR), (mismatch("ب", "م"), w.OTHER_CONSONANT),
    (deletion("ل"), w.SHADDAH_CAUSE), (deletion("ب"), w.OTHER_CONSONANT),
    (tashkeel(FATHA, None), None),
])
def test_each_consonant_error_has_one_cause(error, cause):
    assert w.consonant_cause(error, "ءَللَذِۦۦنَ") == cause  # ل doubled, ب absent


def test_a_tashkeel_error_word_is_flagged_for_its_marks():
    grade = SiteGrade(FLAGGED_ELSEWHERE, TASHKEEL_ERROR,
                      (tashkeel(FATHA, None), tashkeel(None, FATHA, 1), mismatch("ذ", "ز")))
    flag = w.flag(grade, "x")
    assert flag.causes == (w.TASHKEEL,)
    assert flag.marks == ("fatha dropped", "sukun other")
    assert flag.errors == ("g0:tash", "g1:tash", "g0:ذ→ز")


def test_a_minor_word_is_flagged_for_its_consonants_never_its_tashkeel():
    grade = SiteGrade(FLAGGED_ELSEWHERE, MINOR,
                      (tashkeel(FATHA, None), mismatch("ط", "ق"), deletion("ل", 2)))
    flag = w.flag(grade, "ءَللَذِۦۦنَ")
    assert flag.causes == (w.TARGET_PAIR, w.SHADDAH_CAUSE) and flag.marks == ()


def test_a_flag_with_no_error_of_its_kind_is_unrecorded_and_a_skip_is_its_own_cause():
    assert w.flag(SiteGrade(FLAGGED_ELSEWHERE, MINOR, (tashkeel(FATHA, None),)), "x").causes == (
        w.UNRECORDED,)
    assert w.flag(SiteGrade(FLAGGED_ELSEWHERE, w.SKIPPED_QUALITY), "x").causes == (w.SKIPPED,)


def _word(n: int, heard: str, draw: float = 0.5) -> VerifiedWord:
    site = SampledWord(f"word_sample:c{n}.wav#0@0", f"c{n}.wav", 1, "0" * 64, n, "1:1", 0, 0, 10,
                       "ab", 0, 0, f"word_sample:reciter:{n}", 2)
    return VerifiedWord(WordSite(site, 1.0, draw, draw, 0, (0, 2), 0, 10), heard)


def test_the_view_counts_false_flags_and_causes_on_correct_words_only():
    words = [_word(1, "correct"), _word(2, "correct", 0.25), _word(3, "correct"),
             _word(4, "mistake"), _word(5, "unclear")]
    grades = {
        words[0].site.site_id: SiteGrade(FLAGGED_ELSEWHERE, MINOR, (mismatch("ط", "ق"),)),
        words[1].site.site_id: SiteGrade(NOT_FLAGGED, "correct"),
        words[2].site.site_id: SiteGrade(NOT_GRADED),
        words[3].site.site_id: SiteGrade(NOT_FLAGGED, "correct"),
        words[4].site.site_id: SiteGrade(FLAGGED_ELSEWHERE, MINOR, (mismatch("ب", "م"),)),
    }
    view = w.scoring_view(words, grades, {k: "x" for k in grades})
    assert view["correct_words"] == 3 and view["not_graded"] == 1
    assert view["false_flags"]["words"] == 1
    assert view["false_flags"]["point"] == pytest.approx(2 / (2 + 4 + 2))  # weights 2, 4, 2
    assert view["causes"][w.TARGET_PAIR]["words"] == 1
    assert view["causes"][w.OTHER_CONSONANT]["words"] == 0  # the unclear word is no false flag
    assert view["mistakes"] == {words[3].site.site_id: NOT_FLAGGED}


def test_the_report_leaves_out_excluded_words():
    words = [_word(1, "correct"), _word(2, "correct")]
    grades = {"s": {"a/checks": {x.site.site_id: SiteGrade(NOT_FLAGGED, "correct") for x in words}}}
    report = w.word_report(words, grades, {}, Exclusions.of(not_hafs={words[0].site.site_id}))
    assert report["not_hafs"] == [words[0].site.site_id]
    assert report["arms"]["a/checks"]["s"]["correct_words"] == 1
