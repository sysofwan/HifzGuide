"""A word's flag follows the scoring's displayed quality (#309, S1.16).

``word_grade_rows`` used to call a word flagged on ``quality != correct`` and hand ``flag`` a
grade whose quality was the raw one. Both now read ``training.muraja_policy.displayed``, the
engine's own display rule, so a scoring that hides tashkeel errors (or runs lenient) grades and
attributes causes by what it actually showed.
"""

from __future__ import annotations

from types import SimpleNamespace

from tadabur.followalong_audit.app_arm import causes_of
from tadabur.followalong_audit.app_run import word_grade_rows

ITEM = "1_2_3.flac#0"


def trace(word: int, quality: str, *, scoring: str = "strict", errors=()) -> dict:
    return {"item": ITEM, "scoring": scoring, "word": word, "quality": quality,
            "errors": list(errors)}


def causes_from_trace(row: dict) -> tuple[str, ...]:
    return ("tashkeel",)


def cell_of(row: dict) -> dict:
    return word_grade_rows(ITEM, 2, [row], causes_from_trace)[0]["strict"]


def test_a_shown_tashkeel_error_is_flagged():
    assert cell_of(trace(1, "tashkeelError")) == {
        "graded": True, "flagged": True, "causes": ["tashkeel"]}


def test_a_quality_the_scoring_does_not_show_is_not_flagged(monkeypatch):
    monkeypatch.setattr("tadabur.followalong_audit.app_run.displayed",
                        lambda quality, scoring: "correct")
    assert cell_of(trace(1, "minor")) == {
        "graded": True, "flagged": False, "causes": []}


TASHKEEL_ERROR = {"kind": {"type": "tashkeel", "expected": None, "heard": None},
                  "group_index": 0}


def test_attribution_reads_the_displayed_quality_too(monkeypatch):
    text = SimpleNamespace(phonemes=lambda surah, ayah: "بَتُ")
    row = trace(1, "tashkeelError", errors=[TASHKEEL_ERROR])
    assert "tashkeel" in causes_of(text, 1, 2)(row)
    monkeypatch.setattr("tadabur.followalong_audit.app_arm.displayed",
                        lambda quality, scoring: "correct")
    assert "tashkeel" not in causes_of(text, 1, 2)(row)
