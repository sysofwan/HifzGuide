"""The device-log (#151) and owner-session (#153) reports regenerated for the re-pin (#306).

The reports keep the pin's numbers frozen (their word rows, rates and tables are replayed at
99c326f by ``test_device_logs.py`` and ``test_device_exports.py``). ``muraja.main`` is now the
audit's commit, 028bc51, and each report gains a ``current`` section: strict and balanced
(``shipped_default``) at 028bc51, and every word whose kept grade changed between the two pins::

    report["current"] = {"commit": <028bc51>,
                         "scorings": ["strict", "shipped_default"],
                         "groups": {<group>: {<scoring>: {<view>: ...}}},   # device logs: report["groups"]'s shape
                         "rates": {<scoring>: {<view>: ...}},               # owner sessions: report["rates"]'s shape
                         "changed_words": [{"log"|"session", "word", "scoring", "pin", "current"}]}

A changed word's ``pin`` and ``current`` are the qualities the engine kept at each commit
(``pending``-to-settled and the like included); the rows are sorted by log or session, word
position, then scoring.
"""

from __future__ import annotations

import json

import pytest

from training import device_exports as x
from training import device_logs as dl
from training import muraja_policy as mp
from training.device_exports_doc import render as render_exports
from training.device_logs_doc import render as render_logs

LOGS = json.loads(dl.REPORT_PATH.read_text(encoding="utf-8"))
OWNER = json.loads(x.REPORT_PATH.read_text(encoding="utf-8"))
CURRENT = "028bc513eeb233caa7240a5c37c423da59f3d9d7"
SCORINGS = ["strict", "shipped_default"]


def test_muraja_main_is_the_audits_commit_in_both_reports_and_the_code():
    assert dl.MURAJA_MAIN == mp.CURRENT_COMMIT == CURRENT
    assert LOGS["muraja"]["pin"] == OWNER["muraja"]["pin"] == mp.MURAJA_COMMIT
    assert LOGS["muraja"]["main"] == OWNER["muraja"]["main"] == CURRENT


@pytest.mark.parametrize("report", [LOGS, OWNER], ids=["device_logs", "owner_sessions"])
def test_the_current_section_shows_strict_and_balanced_at_the_audits_commit(report):
    current = report["current"]
    assert current["commit"] == CURRENT and current["scorings"] == SCORINGS
    # each build record names the commit the section was graded at
    assert report["muraja"]["builds"]["main"]["muraja_commit"] == CURRENT


def test_the_device_log_rates_at_the_current_commit_cover_every_group_and_both_scorings():
    for group in dl.GROUPS:
        for scoring in SCORINGS:
            for view in (dl.ALL_WORDS, dl.UNEXPLAINED):
                cell = LOGS["current"]["groups"][group][scoring][view]["false_flags"]
                assert {"flags", "words", "per_1000"} <= set(cell)


def test_the_owner_session_rates_at_the_current_commit_cover_both_scorings():
    for scoring in SCORINGS:
        assert OWNER["current"]["rates"][scoring][dl.UNEXPLAINED]["false_flags"]["words"] > 0


def changed_rows(report: dict, key: str) -> list[dict]:
    rows = report["current"]["changed_words"]
    assert all(set(r) == {key, "word", "scoring", "pin", "current"} for r in rows)
    assert all(r["scoring"] in SCORINGS and r["pin"] != r["current"] for r in rows)
    order = [(r[key], dl.position(r["word"]), SCORINGS.index(r["scoring"])) for r in rows]
    assert order == sorted(order)
    return rows


def test_every_changed_word_of_the_device_logs_is_listed_in_a_log_that_differs_between_the_pins():
    rows = changed_rows(LOGS, "log")
    assert set(r["log"] for r in rows) <= set(LOGS["muraja"]["main_differs_from_pin"])


def test_every_changed_word_of_the_owner_sessions_is_in_a_target_session():
    rows = changed_rows(OWNER, "session")
    assert {r["session"] for r in rows} <= {s["session"] for s in OWNER["sessions"] if s["target"]}


def test_the_docs_are_regenerated_from_the_reports_and_list_strict_and_every_changed_word():
    for path, render, report, key in ((dl.DOC_PATH, render_logs, LOGS, "log"),
                                      (x.DOC_PATH, render_exports, OWNER, "session")):
        doc = path.read_text(encoding="utf-8")
        assert doc == render(report)
        assert CURRENT in doc and "strict" in doc
        for row in report["current"]["changed_words"]:
            assert row[key] in doc and row["word"] in doc
