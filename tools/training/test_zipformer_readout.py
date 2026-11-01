"""Tests for the zipformer screen readout's rules (#245): the gap closure and the verdict.

The heavy sections (the harness, the benchmark references, the decodes) are proven by the
committed record they wrote; these tests pin the two rules a reader must not have to re-derive:
what "closes half the gap" means for a metric that is lower-is-better, and how the three
preregistered outcomes follow from the six numbers. Torch-free, no decodes, no harness.
"""

from __future__ import annotations

import json

import pytest

from training import zipformer_readout as readout
from training.zipformer_comparison_doc import render


# --- closing the gap -----------------------------------------------------------------------


def test_closing_half_the_gap_on_a_lower_is_better_metric():
    # teacher 10, candidate 20: the gap is 10, so zipformer must reach 15
    row = readout.closure(teacher=10.0, candidate=20.0, arm=15.0)
    assert row["gap"] == 10.0 and row["target"] == 15.0 and row["closed"] is True
    assert readout.closure(10.0, 20.0, 15.01)["closed"] is False
    assert readout.closure(10.0, 20.0, 14.0)["closed"] is True


def test_a_gap_the_candidate_already_closed_is_closed_by_matching_the_candidate():
    # the candidate is below the teacher (gap -2): anything at or below the candidate closes it
    assert readout.closure(10.0, 8.0, 8.0)["closed"] is True
    assert readout.closure(10.0, 8.0, 9.5)["closed"] is False
    assert readout.closure(10.0, 10.0, 10.0)["closed"] is True


# --- the verdict ---------------------------------------------------------------------------


def _record(*, qdat=20.0, consonant=2.0, haraka=2.0, heard=35.0, decoy=96.0, ship=35.0) -> dict:
    """A record shaped like the readout's, with only the numbers the verdict reads."""
    def errors(value):
        return {"consonant": {"error": value}, "haraka": {"error": value}}

    return {
        "qdat": {"models": {"base": {"point": 30.0}, "r2_candidate_ema": {"point": 37.0},
                            "zipformer-checks-w1": {"point": qdat}}},
        "benchmark": {"sources": {"tlog_holdout": {
            "base": errors(11.0), "r2_candidate_ema": errors(10.0),
            "zipformer": {"consonant": {"error": consonant},
                          "haraka": {"error": haraka}}}}},
        "edits": {"arms": {
            "edit": {"zipformer": {"point": heard}, "r2_candidate_ema": {"point": 35.0},
                     "base": {"point": 47.0}, "c95": {"point": 37.0}},
            "decoy": {"zipformer": {"point": decoy}, "r2_candidate_ema": {"point": 67.0},
                      "base": {"point": 99.0}, "c95": {"point": 71.0}}}},
        "ship_a": {"mistakes_flagged": {"models": {"zipformer-checks-w1": {"point": ship},
                                                   "r2_candidate_ema": {"point": 35.0},
                                                   "base": {"point": 33.0}}}},
    }


def test_a_model_that_closes_the_gap_and_keeps_the_catch_side_is_go():
    verdict = readout.verdict(_record())
    assert verdict["correct_passed"] and verdict["catch_passed"]
    assert verdict["verdict"] == "GO"


def test_a_model_that_passes_the_correct_side_but_loses_the_catch_side_is_prior():
    verdict = readout.verdict(_record(heard=17.0))
    assert verdict["correct_passed"] and not verdict["catch_passed"]
    assert verdict["verdict"] == "PRIOR"


def test_decoys_under_the_floor_are_a_catch_failure():
    verdict = readout.verdict(_record(decoy=94.7))
    assert verdict["catch_side"]["edit_decoys_unchanged"]["passed"] is False
    assert verdict["verdict"] == "PRIOR"


def test_a_model_that_misses_the_correct_side_is_stop():
    verdict = readout.verdict(_record(qdat=40.0))
    assert verdict["correct_passed"] is False
    assert verdict["verdict"] == "STOP"


def test_the_ship_a_bar_is_the_candidate_minus_five_points():
    verdict = readout.verdict(_record(ship=30.5))
    bar = verdict["catch_side"]["ship_a_mistakes_flagged"]
    assert bar["bar"] == pytest.approx(30.0)
    assert bar["passed"] is True
    assert readout.verdict(_record(ship=29.0))["verdict"] == "PRIOR"


# --- the committed page --------------------------------------------------------------------


def test_the_committed_doc_is_the_render_of_the_committed_record():
    record = json.loads(readout.REPORT_PATH.read_text(encoding="utf-8"))
    assert readout.DOC_PATH.read_text(encoding="utf-8") == render(record)
