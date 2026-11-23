"""The parity gate's explanation rules on synthetic grade lines (#306).

A difference no named 1.0.28 change explains must raise, so each rule is checked for what it
accepts and for the near-misses it must refuse. Harness-free: the reference reads `explain`
makes come from fixed values, not Muraja's database.
"""

from __future__ import annotations

import pytest

from training import followalong_parity as fp

WORD = "1:1:1"


def line(*, gap: int = 0, cov: str = "1.00") -> str:
    return (f"Grade end word {WORD}: cov={cov} acc=1.00 score=0.60 mm=0 gap={gap} "
            "last=false → minor")


class Text:
    """The three reference reads :func:`fp.explain` makes, from fixed values per word."""

    def __init__(self, phonemes: list[str], rows: list[list[str]], printed: list[str]):
        self._phonemes, self._rows, self._printed = phonemes, rows, printed

    def phonemes(self, surah: int, ayah: int) -> str:
        return " ".join(self._phonemes)

    def word_rows(self, surah: int, ayah: int, word: int) -> list[str]:
        return self._rows[word - 1]

    def printed(self, surah: int, ayah: int, word: int) -> str:
        return self._printed[word - 1]


def explain(text, pin, current, pin_lines, current_lines) -> str:
    return fp.explain("s", WORD, pin, current, pin_lines, current_lines, text)


def grade(quality: str, errors=()) -> dict:
    return {"quality": quality, "score": 0.0, "errors": list(errors)}


def deletion(letter: str, group: int) -> dict:
    return {"kind": {"type": "deletion", "expected": letter}, "group_index": group}


# --- madd alif --------------------------------------------------------------------------------------
#: ``بَاااا``: the four alifs a gap drop may be bounded by.
MADD = Text(["بَاااا"], [["بَ", "ا", "ا", "ا", "ا"]], ["بَاااا"])


def test_madd_alif_explains_a_gap_drop_within_the_word_s_alifs():
    assert explain(MADD, grade("minor"), grade("correct"), [line(gap=4)], [line(gap=0)]) == "madd_alif"


def test_madd_alif_raises_when_the_dropped_gaps_exceed_the_word_s_alifs():
    bare = Text(["بَ"], [["بَ"]], ["بَ"])
    with pytest.raises(SystemExit):
        explain(bare, grade("minor"), grade("correct"), [line(gap=2)], [line(gap=0)])


def test_madd_alif_raises_when_the_pin_recorded_a_non_alif_deletion():
    with pytest.raises(SystemExit):
        explain(MADD, grade("minor", [deletion("ل", 0)]), grade("correct"),
                [line(gap=2)], [line(gap=0)])


# --- tanween at waqf --------------------------------------------------------------------------------
def test_tanween_at_waqf_explains_a_trailing_nun_at_the_ayah_end():
    text = Text(["خَيۡرٌۚ"], [["خَ", "يۡ", "رُ", "ن"]], ["خَيۡرٌۚ"])
    assert explain(text, grade("minor"), grade("correct"),
                   [line(cov="0.50")], [line(cov="1.00")]) == "tanween_at_waqf"


def test_tanween_at_waqf_raises_mid_ayah_without_a_waqf_sign():
    text = Text(["خَيۡرٌ", "بِ"], [["خَ", "يۡ", "رُ", "ن"], ["بِ"]], ["خَيۡرٌ", "بِ"])
    with pytest.raises(SystemExit):
        explain(text, grade("minor"), grade("correct"),
                [line(cov="0.50")], [line(cov="1.00")])


# --- error indexing ---------------------------------------------------------------------------------
def test_error_indexing_explains_a_duplicate_collapsed_to_the_row():
    text = Text(["للِ"], [["للِ"]], ["للِ"])
    pin = grade("minor", [deletion("ل", 1)])
    current = grade("minor", [deletion("ل", 0)])
    assert explain(text, pin, current, [line()], [line()]) == "error_indexing"


# --- stop pass --------------------------------------------------------------------------------------
def test_stop_pass_explains_a_moved_status_with_identical_decisions():
    text = Text(["بَ"], [["بَ"]], ["بَ"])
    assert explain(text, grade("minor"), grade("correct"), [line()], [line()]) == "stop_pass"


def test_stop_pass_raises_with_no_grade_line():
    text = Text(["بَ"], [["بَ"]], ["بَ"])
    with pytest.raises(SystemExit):
        explain(text, grade("skipped"), grade("correct"), [], [])


def test_stop_pass_raises_when_the_errors_moved_unexplained():
    text = Text(["بَ"], [["بَ"]], ["بَ"])
    with pytest.raises(SystemExit):
        explain(text, grade("minor", [deletion("ل", 0)]), grade("minor", [deletion("ن", 0)]),
                [line()], [line()])
