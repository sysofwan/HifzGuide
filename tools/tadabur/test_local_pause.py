"""The local pause check (#134): its score on synthetic VAD frames and its calibration rule."""

from __future__ import annotations

import numpy as np
import pytest

from tadabur import local_pause as lp


def test_the_longest_silence_counts_only_frames_inside_the_window():
    probabilities = np.array([0.9] * 5 + [0.0] * 10 + [0.6] * 8 + [0.0] * 5 + [0.9] * 20)
    assert lp.longest_silence(probabilities, 0.0, 0.2, 0.5) == pytest.approx(0.1)
    assert lp.longest_silence(probabilities, 0.3, 0.5, 0.5) == pytest.approx(0.16)
    assert lp.longest_silence(probabilities, 0.3, 0.5, 0.7) == 0.0
    assert lp.longest_silence(probabilities, 0.56, 1.0, 0.5) == pytest.approx(0.4)
    assert lp.longest_silence(probabilities, 0.56, 0.6, 0.5) == pytest.approx(0.04)


def test_a_boundary_is_read_between_the_two_words_onsets():
    assert lp.boundary_window([0.0, 1.2, 2.5, 3.0], 2) == [1.2, 2.5 + lp.WINDOW_PAD_S]


def _boundary(split, waqf, segmenter, silence):
    """A fixture boundary whose longest silence is ``silence`` seconds at every threshold."""
    return {"split": split, "waqf": waqf, "segmenter": segmenter,
            "scores": {str(t): silence for t in lp.THRESHOLDS}}


def test_the_rule_takes_the_highest_recall_at_the_precision_floor():
    """Ten pauses the segmenter missed (silences 0.30 to 1.20 s) and ten wasl edges with a
    short silence each: a duration at or below 0.08 s flags a wasl edge too (precision 10/20
    < 0.9); 0.10 s keeps every pause and no wasl edge, and among those ties the longest such
    duration (0.30 s), then the highest threshold (0.5), wins."""
    pauses = [_boundary(lp.CALIBRATION, True, False, 0.3 + 0.1 * k) for k in range(10)]
    edges = [_boundary(lp.CALIBRATION, False, False, 0.08) for _ in range(10)]
    report = lp.calibrate(pauses + edges)
    assert report["operating_point"] == {"threshold": 0.5, "duration": 0.3}
    calibration = report["splits"][lp.CALIBRATION]
    assert calibration["segmenter"]["recall"] == 0.0
    assert calibration["local_check"] == {"tp": 10, "fp": 0, "positives": 10,
                                          "precision": 1.0, "recall": 1.0}
    assert len(report["calibration_curve"]) == len(lp.THRESHOLDS) * len(lp.DURATIONS)


def test_the_point_is_chosen_where_the_segmenter_did_not_cut():
    """A long silence the segmenter cut at a wasl edge would hold every candidate point under
    the precision floor; the guard never reads such a boundary, so the rule does not either.
    The curve over every boundary is kept for reference."""
    missed = [_boundary(lp.CALIBRATION, True, False, 0.3 + 0.1 * k) for k in range(3)]
    edges = [_boundary(lp.CALIBRATION, False, False, 0.08) for _ in range(10)]
    cut = [_boundary(lp.CALIBRATION, False, True, 2.0), _boundary(lp.CALIBRATION, True, True, 2.0)]
    report = lp.calibrate(missed + edges + cut)
    assert report["operating_point"] == {"threshold": 0.5, "duration": 0.3}
    residual = report["splits"][lp.CALIBRATION]["local_check_where_segmenter_did_not_cut"]
    assert (residual["tp"], residual["fp"], residual["positives"]) == (3, 0, 3)
    every = {(p["threshold"], p["duration"]): p for p in report["calibration_curve_every_boundary"]}
    assert every[(0.5, 0.3)]["precision"] == 0.8  # 4 of 5 flagged, the cut wasl edge among them


def test_the_test_split_never_moves_the_operating_point():
    calibration = [_boundary(lp.CALIBRATION, True, False, 0.5),
                   _boundary(lp.CALIBRATION, False, False, 0.0)]
    quiet = lp.calibrate(calibration + [_boundary(lp.TEST, False, False, 0.9)] * 5)
    loud = lp.calibrate(calibration + [_boundary(lp.TEST, True, True, 0.0)] * 5)
    assert quiet["operating_point"] == loud["operating_point"]
    assert quiet["splits"][lp.TEST]["local_check"]["fp"] == 5
    assert loud["splits"][lp.TEST]["local_check"]["recall"] == 0.0


def test_no_operating_point_without_a_point_at_the_floor():
    """Every pause looks like every wasl edge: precision stays at 1/2."""
    rows = [_boundary(lp.CALIBRATION, True, False, 0.5),
            _boundary(lp.CALIBRATION, False, False, 0.5)]
    assert lp.calibrate(rows)["operating_point"] is None
