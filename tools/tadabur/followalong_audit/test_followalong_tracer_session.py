"""The App and Oracle arms' engine request (#307): the session, and the per-clip cadence."""

from __future__ import annotations

from tadabur.followalong_audit.session import preview_window_ratio, session_request
from training.muraja_policy import AUDIT_SCORINGS, engine_runs

CHECKS = [{"hop": "", "overlap": "بَ", "flush": False},
          {"hop": "بَ", "overlap": "تِ", "flush": False, "stop": True}]


def test_the_session_starts_on_the_clips_own_ayah_at_its_first_word():
    request = session_request("clip#1", 100, 3, 4, CHECKS)
    assert (request["surah"], request["ayah"], request["start_word"]) == (100, 3, 1)


def test_every_word_of_the_ayah_is_reported():
    assert session_request("clip#1", 100, 3, 4, CHECKS)["report_words"] == [1, 2, 3, 4]


def test_the_request_replays_strict_and_balanced_over_the_checks_unchanged():
    request = session_request("clip#1", 100, 3, 4, CHECKS)
    assert request["item"] == "clip#1"
    assert request["checks"] == CHECKS
    assert request["scorings"] == engine_runs(AUDIT_SCORINGS)
    assert {s["mode"] for s in request["scorings"]} == {"strict", "balanced"}


def test_the_ratio_counts_previews_per_window_and_leaves_flushes_and_the_stop_out():
    previews = [{"hop": "", "overlap": "بَ", "flush": False}] * 6
    windows = [{"hop": "بَ", "overlap": "تِ", "flush": False}] * 2
    flush = [{"hop": "بَ", "overlap": "", "flush": True}]
    stop = [{"hop": "بَ", "overlap": "", "flush": False, "stop": True}]
    assert preview_window_ratio(previews + windows + flush + stop) == 3.0


def test_a_clip_with_no_window_has_no_ratio():
    assert preview_window_ratio([{"hop": "", "overlap": "بَ", "flush": False}]) is None
