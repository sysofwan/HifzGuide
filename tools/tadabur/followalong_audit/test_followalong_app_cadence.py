"""P3, cadence (#309): preview:window ratios of device logs, and the words a latency moves.

``log_preview_window_ratio(text)`` is the previews per window among a device log's
``Update: type=...`` lines (``None`` when it ran no window). ``cadence_report(base, other,
scoring)`` compares two App-arm runs of the same clips (0 ms against 130 ms) word by word on the
outcome ``flagged`` / ``passed`` / ``ungraded``; more than 2% of words changing is a follow-along
finding (exactly 2% is not).
"""

from __future__ import annotations

import pytest

from tadabur.followalong_audit.cadence import (
    CADENCE_THRESHOLD,
    cadence_report,
    log_preview_window_ratio,
)


def update(kind: str) -> str:
    return (f"[2026-11-01 10:00:00] [Transcribe] Update: type={kind}, hop=ab, "
            f"overlap=cd, didFlush=false")


def test_the_ratio_is_previews_per_window():
    log = "\n".join([update("window")] * 2 + [update("preview")] * 5 + ["Page: 3"])
    assert log_preview_window_ratio(log) == 2.5


def test_a_log_with_no_window_has_no_ratio():
    assert log_preview_window_ratio(update("preview")) is None


def test_lines_that_are_not_updates_do_not_count():
    assert log_preview_window_ratio("Preview 41ms — text\n" + update("window")) == 0.0


def cell(flagged: bool = False, graded: bool = True) -> dict:
    return {"graded": graded, "flagged": flagged, "causes": []}


def run(flags: list[bool]) -> list[dict]:
    return [{"item": "a#0", "word": index + 1, "strict": cell(flag), "shipped_default": cell()}
            for index, flag in enumerate(flags)]


def test_the_threshold_is_2_percent():
    assert CADENCE_THRESHOLD == 0.02


def test_no_change_is_no_finding():
    report = cadence_report(run([False] * 100), run([False] * 100), "strict")
    assert (report["words"], report["changed"], report["share"], report["finding"]) == (
        100, 0, 0.0, False)


def test_exactly_two_percent_is_not_a_finding_and_more_is():
    base = run([False] * 100)
    two = cadence_report(base, run([True] * 2 + [False] * 98), "strict")
    three = cadence_report(base, run([True] * 3 + [False] * 97), "strict")
    assert (two["changed"], two["finding"]) == (2, False)
    assert (three["changed"], three["share"], three["finding"]) == (3, 0.03, True)


def test_a_word_gaining_or_losing_its_grade_is_a_change():
    other = run([False] * 100)
    other[0]["strict"] = cell(graded=False)
    assert cadence_report(run([False] * 100), other, "strict")["changed"] == 1


def test_each_scoring_is_reported_from_its_own_cell():
    other = run([False] * 100)
    other[0]["shipped_default"] = cell(flagged=True)
    assert cadence_report(run([False] * 100), other, "strict")["changed"] == 0
    assert cadence_report(run([False] * 100), other, "shipped_default")["changed"] == 1


def test_the_two_runs_must_hold_the_same_words():
    with pytest.raises(ValueError):
        cadence_report(run([False] * 3), run([False] * 2), "strict")
