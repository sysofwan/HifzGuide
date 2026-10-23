"""The consonant round's pre-registered bars and verdict (#175, bars from #174).

Every condition is checked at its boundary and as "cannot certify". The fixture is a synthetic
report shaped as the round's `confirm_report` writes it.
"""

from __future__ import annotations

import pytest

from training import aug_readout_metrics as am
from training import aug_readout_spec as asp
from training import aug_readout_verdict as av
from training import consonant_readout as cr
from training.acceptance_stats import CANNOT_CERTIFY, FAIL, NO_BOUND, PASS

VS = "candidate−c95"


def _interval(lower, upper, method="bootstrap", reason=""):
    return {"lower": lower, "upper": upper, "method": method, "reason": reason}


def _view(points: dict, difference: tuple | None = None, scale=am.PER_1000,
          interval: dict | None = None, veto: dict | None = None) -> dict:
    comparisons = {}
    if difference is not None:
        comparisons[VS] = {"point": difference[0],
                           "interval": interval or _interval(difference[1], difference[2])}
        if veto is not None:
            comparisons[VS]["veto_interval"] = veto
    return {"scale": scale, "observations": 100, "clusters": 20,
            "models": {m: {"point": p, "num": 0, "den": 1, "interval": _interval(p, p)}
                       for m, p in points.items()},
            "comparisons": comparisons}


def _report() -> dict:
    agreement = _view({"candidate": 95.0, "c95": 94.0}, (1.0, 0.5, 1.5), am.POINTS)
    departs = _view({"candidate": 92.0, "c95": 91.6}, (0.4, -0.1, 0.9), am.POINTS)
    bench = _view({"candidate": 90.0, "c95": 90.0}, (0.0, -1.5, 1.5), am.POINTS)
    substitution = _view({"candidate": 1.0, "c95": 1.2}, (-0.2, -0.5, 0.1),
                         veto=_interval(-0.6, 0.2))
    return {
        "main": {"all_cause": _view({"candidate": 40.0, "c95": 47.41, "h448": 57.0,
                                     "base": 21.0}, (-7.41, -10.0, -5.0)),
                 "consonant_cause": _view({"candidate": 10.0, "c95": 13.0},
                                          (-3.0, -5.0, -1.0)),
                 "tashkeel_cause": _view({"candidate": 5.2, "c95": 5.0}, (0.2, -0.3, 0.7))},
        "mechanism": {"tarteel_v1_confirm": agreement, "qdat": agreement,
                      "tarteel_v1_confirm_departs": departs, "qdat_departs": departs},
        "consistency": {"qdat": _view({"candidate": 60.0, "c95": 64.0}, (-4.0, -8.0, -1.0))},
        "guards": {"qud_other": _view({"candidate": 4.0, "c95": 3.6}, (0.4, -0.2, 0.9)),
                   "tadabur_dev_unseen": _view({"candidate": 94.5, "c95": 94.6},
                                               (-0.1, -0.6, 0.4), am.POINTS),
                   "qdat_bench": {k: bench for k in ("consonant", "haraka", "shaddah")}},
        "substitutions": {"س→ش": substitution},
        "catch": {"ship_a_verified": {"sites": 8,
                                      "models": {"candidate": {"graded": 8, "flagged": 4},
                                                 "c95": {"graded": 8, "flagged": 5}},
                                      "difference": -1}},
    }


def _condition(result: dict, name: str) -> dict:
    return next(c for c in result["conditions"] if c["name"] == name)


def _verdict(report: dict | None = None) -> dict:
    return av.verdict(report or _report(), spec=asp.CONSONANT1)


def _comparison(report: dict, *path: str) -> dict:
    view = report
    for key in path:
        view = view[key]
    return view["comparisons"][VS]


def test_every_pre_registered_condition_is_read_from_the_report():
    result = _verdict()
    assert [c["name"] for c in result["conditions"]] == [
        "bar1.all_cause", "bar2.reduction", "bar3.consonant_cause",
        "bar4a.tarteel_v1_confirm", "bar4a.qdat",
        "bar4b.tarteel_v1_confirm", "bar4b.qdat", "bar5.qdat",
        "g1.qud_other", "g2.tadabur_dev_unseen", "g3.qdat_bench.consonant",
        "g3.qdat_bench.haraka", "g3.qdat_bench.shaddah", "g4.tashkeel_cause",
        "g5.س→ش", "g6.ship_a_verified"]
    assert [c["verdict"] for c in result["conditions"]] == [PASS] * 16
    assert result["verdict"] == PASS


@pytest.mark.parametrize("edit, name", [
    (lambda r: _comparison(r, "main", "all_cause").update(
        interval=_interval(-1.0, 0.0)), "bar1.all_cause"),  # < 0 is strict
    (lambda r: r["main"]["all_cause"]["models"]["candidate"].update(
        point=42.72), "bar2.reduction"),  # a reduction of 4.69, under 4.7
    (lambda r: _comparison(r, "main", "consonant_cause").update(
        interval=_interval(-1.0, 0.0)), "bar3.consonant_cause"),
    (lambda r: _comparison(r, "mechanism", "tarteel_v1_confirm").update(
        interval=_interval(0.0, 1.0)), "bar4a.tarteel_v1_confirm"),  # > 0 is strict
    (lambda r: _comparison(r, "mechanism", "qdat").update(
        interval=_interval(-1.0, 0.0)), "bar4a.qdat"),
    (lambda r: _comparison(r, "mechanism", "tarteel_v1_confirm_departs").update(
        interval=_interval(-0.51, 0.2)), "bar4b.tarteel_v1_confirm"),
    (lambda r: _comparison(r, "mechanism", "qdat_departs").update(
        interval=_interval(-1.0, -0.51)), "bar4b.qdat"),
    (lambda r: r["consistency"]["qdat"]["models"]["candidate"].update(
        point=64.0), "bar5.qdat"),  # not lower
    (lambda r: _comparison(r, "guards", "qud_other").update(
        interval=_interval(-1.0, 1.01)), "g1.qud_other"),
    (lambda r: _comparison(r, "guards", "tadabur_dev_unseen").update(
        interval=_interval(-1.01, 0.5)), "g2.tadabur_dev_unseen"),
    (lambda r: _comparison(r, "guards", "qdat_bench", "consonant").update(
        interval=_interval(-2.01, 0.0)), "g3.qdat_bench.consonant"),
    (lambda r: _comparison(r, "guards", "qdat_bench", "haraka").update(
        interval=_interval(-2.01, 0.0)), "g3.qdat_bench.haraka"),
    (lambda r: _comparison(r, "guards", "qdat_bench", "shaddah").update(
        interval=_interval(-2.01, 0.0)), "g3.qdat_bench.shaddah"),
    (lambda r: _comparison(r, "main", "tashkeel_cause").update(
        interval=_interval(-1.0, 1.01)), "g4.tashkeel_cause"),
    (lambda r: _comparison(r, "substitutions", "س→ش").update(
        veto_interval=_interval(0.001, 0.2)), "g5.س→ش"),
    (lambda r: r["catch"]["ship_a_verified"].update(difference=-3), "g6.ship_a_verified"),
])
def test_each_bar_fails_just_past_its_boundary(edit, name):
    report = _report()
    edit(report)
    result = _verdict(report)
    assert _condition(result, name)["verdict"] == FAIL
    assert result["verdict"] == FAIL


def test_the_inclusive_bars_hold_at_their_boundary():
    report = _report()
    _comparison(report, "mechanism", "tarteel_v1_confirm_departs")["interval"] = _interval(
        -0.5, 0.2)
    _comparison(report, "mechanism", "qdat_departs")["interval"] = _interval(-0.5, 0.2)
    _comparison(report, "guards", "qud_other")["interval"] = _interval(-1.0, 1.0)
    _comparison(report, "guards", "tadabur_dev_unseen")["interval"] = _interval(-1.0, 0.5)
    for kind in ("consonant", "haraka", "shaddah"):
        _comparison(report, "guards", "qdat_bench", kind)["interval"] = _interval(-2.0, 0.0)
    _comparison(report, "main", "tashkeel_cause")["interval"] = _interval(-1.0, 1.0)
    _comparison(report, "substitutions", "س→ش")["veto_interval"] = _interval(-0.6, 0.0)
    report["catch"]["ship_a_verified"]["difference"] = -2
    assert _verdict(report)["verdict"] == PASS


def test_the_strict_bars_pass_just_inside_their_boundary_and_the_reduction_at_4_7():
    report = _report()
    _comparison(report, "main", "all_cause")["interval"] = _interval(-1.0, -0.001)
    _comparison(report, "main", "consonant_cause")["interval"] = _interval(-1.0, -0.001)
    _comparison(report, "mechanism", "tarteel_v1_confirm")["interval"] = _interval(0.001, 1.0)
    _comparison(report, "mechanism", "qdat")["interval"] = _interval(0.001, 1.0)
    report["main"]["all_cause"]["models"]["candidate"]["point"] = 42.71  # 4.7 exactly
    assert _verdict(report)["verdict"] == PASS


@pytest.mark.parametrize("edit, name", [
    (lambda r: _comparison(r, "main", "all_cause").update(
        interval=_interval(None, None, NO_BOUND, "degenerate")), "bar1.all_cause"),
    (lambda r: r["main"]["all_cause"]["models"]["c95"].update(point=None), "bar2.reduction"),
    (lambda r: _comparison(r, "main", "consonant_cause").update(
        interval=_interval(None, None, NO_BOUND, "degenerate")), "bar3.consonant_cause"),
    (lambda r: _comparison(r, "mechanism", "tarteel_v1_confirm").update(
        interval=_interval(None, None, NO_BOUND, "degenerate")), "bar4a.tarteel_v1_confirm"),
    (lambda r: _comparison(r, "mechanism", "qdat_departs").update(
        interval=_interval(None, None, NO_BOUND, "degenerate")), "bar4b.qdat"),
    (lambda r: r["consistency"]["qdat"]["models"]["c95"].update(point=None), "bar5.qdat"),
    (lambda r: _comparison(r, "guards", "qud_other").update(
        interval=_interval(None, None, NO_BOUND, "degenerate")), "g1.qud_other"),
    (lambda r: _comparison(r, "guards", "tadabur_dev_unseen").update(
        interval=_interval(None, None, NO_BOUND, "degenerate")), "g2.tadabur_dev_unseen"),
    (lambda r: _comparison(r, "guards", "qdat_bench", "haraka").update(
        interval=_interval(None, None, NO_BOUND, "degenerate")), "g3.qdat_bench.haraka"),
    (lambda r: _comparison(r, "main", "tashkeel_cause").update(
        interval=_interval(None, None, NO_BOUND, "degenerate")), "g4.tashkeel_cause"),
    (lambda r: _comparison(r, "substitutions", "س→ش").pop("veto_interval"), "g5.س→ش"),
    (lambda r: r["catch"].update(ship_a_verified=None), "g6.ship_a_verified"),
])
def test_a_missing_input_or_undefined_interval_cannot_certify(edit, name):
    report = _report()
    edit(report)
    result = _verdict(report)
    assert _condition(result, name)["verdict"] == CANNOT_CERTIFY
    assert result["verdict"] == CANNOT_CERTIFY


def test_a_fail_outranks_cannot_certify():
    report = _report()
    _comparison(report, "guards", "qud_other")["interval"] = _interval(
        None, None, NO_BOUND, "degenerate")
    _comparison(report, "main", "consonant_cause")["interval"] = _interval(-1.0, 0.0)
    result = _verdict(report)
    assert _condition(result, "g1.qud_other")["verdict"] == CANNOT_CERTIFY
    assert result["verdict"] == FAIL


# --- the C95-against-C95 tracer ---------------------------------------------------------------


def test_the_tracer_reads_zero_no_bound_and_cannot_certify():
    nothing = _interval(None, None, NO_BOUND, "sparse or degenerate")
    zero = _view({"candidate": 0.0, "c95": 0.0}, (0.0, None, None), interval=nothing)
    text = cr.tracer_text({"main.all_cause": zero})
    assert "point difference 0.00" in text
    assert "no bound" in text
    assert "cannot certify" in text
