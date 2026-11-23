"""The draw's population and #248's exclusion (#307).

The draw may only name clips whose Oracle arm can exist. That set is #248's graded population:
the phase A gating rows less the clips of the ayat the phonetizer cannot read, which
``tadabur/consonant2_readout/report.json`` froze. Pinning both the report's bytes and the
resulting counts makes a word-times regeneration or a reworded reason trip a test instead of
silently changing what the tracer selects.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tadabur.followalong_audit.clips import (
    gating_rows,
    oracle_eligible,
    selection_record,
)
from tadabur.followalong_audit.tracer import load_word_times
from tadabur.tlog_index import sha256_file
from training.nonpro_decode import item_key

REPORT = Path(__file__).resolve().parents[1] / "consonant2_readout" / "report.json"

#: The #248 artifact's SHA-256, as the tracer's docstring cites it. Re-pin deliberately when the
#: report is regenerated, after re-checking the filter still matches it.
REPORT_SHA256 = "2149ea0f2f5dc4ba125f718f89f6a942fd2c30a39850f1d259e2823e338b1117"


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return gating_rows()


@pytest.fixture(scope="module")
def excluded() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))["coverage"]["graded"]["excluded_clips"]


@pytest.fixture(scope="module")
def eligible(rows) -> list[dict]:
    return oracle_eligible(rows, load_word_times())


def test_the_pinned_248_report_is_the_one_the_filter_was_measured_against():
    assert sha256_file(REPORT) == REPORT_SHA256


def test_the_filter_keeps_exactly_the_population_248_graded(rows, excluded, eligible):
    assert len(rows) == 5_762
    assert len(eligible) == len(rows) - excluded["count"] == 5_742
    dropped = {item_key(row["audio_path"]) for row in rows} - {
        item_key(row["audio_path"]) for row in eligible}
    assert dropped == set(excluded["keys"])
    assert sorted({row["filename_label"] for row in rows
                   if item_key(row["audio_path"]) in dropped}) == sorted(excluded["ayat"])


def test_a_clip_whose_alignment_failed_but_has_a_reference_stays_eligible(rows, eligible):
    word_times = load_word_times()
    assert all(word_times[item_key(row["audio_path"])].get("reason") is None
               or not str(word_times[item_key(row["audio_path"])]["reason"]).startswith(
                   "the phonetizer cannot handle")
               for row in eligible)


def test_the_fallback_row_is_kept_and_the_unreadable_one_dropped():
    rows = [{"audio_path": f"{name}.flac", "split": "clean", "shard_file": "s", "row": index}
            for index, name in enumerate(("aligned", "unreadable", "short", "missing"))]
    word_times = {
        "aligned.flac#0": {"reason": None},
        "unreadable.flac#0": {"reason": "the phonetizer cannot handle 106:2: KeyError('ف')"},
        "short.flac#0": {"reason": "76 frames cannot hold the 63-label transcript"},
    }
    assert [row["audio_path"] for row in oracle_eligible(rows, word_times)] == [
        "aligned.flac", "short.flac"]


def test_the_selection_record_states_the_population_the_seed_chose_from(rows, excluded,
                                                                       eligible):
    record = selection_record(rows, eligible, size=20, seed=7)
    assert (record["seed"], record["size"]) == (7, 20)
    assert (record["phase_a_rows"], record["eligible_rows"]) == (5_762, 5_742)
    assert (record["excluded_rows"], record["excluded_ayat"]) == (
        excluded["count"], sorted(excluded["ayat"]))
    assert record["excluded_items"] == sorted(excluded["keys"])
    assert "#248" in record["exclusion"]
