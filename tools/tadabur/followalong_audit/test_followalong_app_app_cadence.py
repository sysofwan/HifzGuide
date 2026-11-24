"""The App arm's own previews-per-window, read from its committed subset traces (#309, R1.3).

P3 compares the h448-era device logs' cadence with the audit's. The device side is read from the
logs; the App side must be measured, not assumed, so :func:`app_ratios` aggregates the same facet
from the subset traces each latency run committed. The aggregation is pooled
(``sum(previews) / sum(windows)``), flushes and the closing stop are neither kind, and a clip with
no window or no trace is counted apart rather than dropped.
"""

from __future__ import annotations

import gzip
import json

from tadabur.followalong_audit.cadence import app_ratios


def check(*, flush: bool = False, stop: bool = False, hop: str = "") -> dict:
    return {"flush": flush, "hop": hop, "overlap": "ءَفَلَ", **({"stop": True} if stop else {})}


def write_traces(directory, name: str, checks: list[dict] | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    body = "" if checks is None else json.dumps(
        {"item": f"{name}#0", "word": 1, "checks": checks}, sort_keys=True) + "\n"
    path = directory / f"{name}.jsonl.gz"
    path.write_bytes(gzip.compress(body.encode("utf-8"), mtime=0))


def test_the_ratio_is_pooled_over_the_clips_that_ran_a_window(tmp_path):
    # A: 6 previews, 2 windows (3.0). B: 4 previews, 4 windows (1.0). Pooled 10/6 = 1.67, not
    # the mean of the two per-clip ratios (2.0).
    write_traces(tmp_path, "a", [check()] * 6 + [check(hop="ء")] * 2 + [check(stop=True)])
    write_traces(tmp_path, "b", [check()] * 4 + [check(hop="ب")] * 4)
    report = app_ratios(tmp_path, latency_ms=0)
    assert (report["clips"], report["previews"], report["windows"]) == (2, 10, 6)
    assert report["ratio"] == 10 / 6 and report["traces"] == tmp_path.name


def test_flushes_are_neither_a_preview_nor_a_window(tmp_path):
    write_traces(tmp_path, "a", [check(flush=True)] * 9 + [check()] + [check(hop="ء")])
    assert app_ratios(tmp_path, latency_ms=130)["ratio"] == 1.0


def test_a_clip_without_a_window_or_without_rows_is_counted_apart(tmp_path):
    write_traces(tmp_path, "windowed", [check()] * 2 + [check(hop="ء")])
    write_traces(tmp_path, "no_window", [check()] * 3)
    write_traces(tmp_path, "no_rows", None)
    report = app_ratios(tmp_path, latency_ms=0)
    assert (report["clips"], report["clips_without_window"],
            report["clips_without_rows"]) == (2, 1, 1)
    # The window-less clip's previews still count, as the device logs count theirs: 5 previews
    # over the subset's single window.
    assert (report["previews"], report["windows"], report["ratio"]) == (5, 1, 5.0)


def test_no_window_at_all_reports_no_ratio(tmp_path):
    write_traces(tmp_path, "no_window", [check()] * 3)
    report = app_ratios(tmp_path, latency_ms=0)
    assert report["ratio"] is None and report["windows"] == 0


def test_the_committed_p3_traces_have_a_ratio_at_both_latencies():
    """The committed subsets are the evidence P3 reports; both must carry a measured ratio."""
    from pathlib import Path

    from tadabur.followalong_audit.app_doc import APP_ARM_DIR
    from tadabur.followalong_audit.cadence import P3_LATENCIES

    for latency in P3_LATENCIES:
        directory = Path(APP_ARM_DIR) / "traces" / f"p3_{latency}"
        assert directory.is_dir(), f"{directory} is not committed"
        report = app_ratios(directory, latency_ms=latency)
        assert report["clips"] + report["clips_without_rows"] \
            == len(list(directory.glob("*.jsonl.gz")))
        assert report["clips"] > 0 and report["ratio"] is not None
