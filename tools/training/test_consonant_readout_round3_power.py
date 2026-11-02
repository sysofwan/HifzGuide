"""The round-3 power rule and the known-exclusion rule (#251, #248 Part 1 items 2-3).

The interface these tests pin:

``training.consonant3_readout`` (new module):

* ``pass_probability(margin, se)``: ``Φ(|margin| / se − 1.96)``, the chance a candidate with no
  true change passes a 95%-bound guard (symmetric in the margin's sign);
* ``catch_pass_probability(d)``: G6's exact ``P(Bin(d, ½) ≥ ⌈(d − 2)/2⌉)`` for ``d`` discordant
  sites;
* ``power_table(spec, *, g4_se, guard_ses, discordant)``: the rows of the pre-registration's
  table, in this order: G1, G2, G3.consonant, G3.haraka, one G4 row per registered margin in
  ``spec.registered_margins`` (not effective), the effective G4 row, G6. A row is a dict with
  ``guard``, ``set``, ``margin``, ``se_source``, ``se`` (``None`` for G6), ``p_pass``,
  ``effective`` (bool), ``result`` (``"pass"`` at ``spec.bars["power_at_least"]`` or above,
  else ``"under"``) and ``note``; G6's row also holds ``discordant``. ``guard_ses`` maps ``g1``,
  ``g2``, ``g3_consonant``, ``g3_haraka`` to their SEs; ``g4_se`` is the already-ρ-scaled SE_G4;
* ``require_power_rule(rows)``: ``SystemExit`` naming the guard of any **effective** row that is
  under; a rejected registered row alone never refuses;
* ``rho()``: round 2's realised pooled G4 SE over its freeze stand-in SE, read from the committed
  ``ROUND2_REPORT_PATH`` and ``ROUND2_FREEZE_PATH`` (module constants, read at call time);
* ``discordant_sites(spec)``: ``d`` from ``training.consonant_readout.catch_arm`` over
  ``("r2_candidate_ema", "c95")`` (keywords ``spec=`` and ``identities=``, the r2 identity read
  from round 2's ``models.json``), the sites exactly one of the two models flags. ``catch_arm``
  and ``g6_scope`` are resolved through ``training.consonant_readout`` at call time; a missing
  cache (``catch_arm`` returns ``None``) refuses.

``training.tlog_eval``: ``KNOWN_UNPHONETIZABLE`` and ``require_known_exclusions(excluded)``.

Harness-free: ``catch_arm`` is replaced where it would grade through Muraja.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

import generate_phonemes
from training import aug_readout_metrics as am
from training import aug_readout_spec as asp
from training import consonant_readout as cr
from training import tlog_eval

REFUSED = (SystemExit, ValueError)
GUARDS = ["G1", "G2", "G3.consonant", "G3.haraka", "G4", "G4", "G6"]
SES = {"g1": 0.19, "g2": 0.05, "g3_consonant": 0.45, "g3_haraka": 0.49}


@pytest.fixture
def c3():
    from training import consonant3_readout

    return consonant3_readout


# --- Φ -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("margin, se, expected", [
    (2.5, 0.76, 0.908), (1.0, 0.76, 0.26), (2.5, 1.07, 0.646), (3.5, 1.07, 0.905)])
def test_the_no_change_pass_probability_is_phi_of_margin_over_se_less_1_96(c3, margin, se,
                                                                           expected):
    assert c3.pass_probability(margin, se) == pytest.approx(expected, abs=1e-3)


def test_a_lower_bound_guard_s_margin_counts_by_its_size(c3):
    assert c3.pass_probability(-1.0, 0.5) == c3.pass_probability(1.0, 0.5)


def test_a_wider_se_never_raises_the_pass_probability(c3):
    probabilities = [c3.pass_probability(3.5, se) for se in (0.2, 0.6, 1.0, 1.4, 2.0)]
    assert probabilities == sorted(probabilities, reverse=True)


# --- the sign test -------------------------------------------------------------------------------


@pytest.mark.parametrize("d, expected", [
    (0, 1.0), (1, 1.0), (2, 1.0), (3, 7 / 8), (4, 15 / 16), (5, 13 / 16), (6, 57 / 64),
    (7, 99 / 128), (8, 219 / 256), (9, 191 / 256), (10, 848 / 1024)])
def test_g6_is_the_exact_sign_test_for_its_discordant_sites(c3, d, expected):
    assert c3.catch_pass_probability(d) == pytest.approx(expected, abs=1e-9)


def test_the_pre_registered_rounded_values_hold(c3):
    stated = {1: 1.00, 5: 0.8125, 7: 0.7734, 8: 0.855, 10: 0.828}
    for d, value in stated.items():
        assert c3.catch_pass_probability(d) == pytest.approx(value, abs=5e-4)


def test_g6_is_not_monotone_in_d_so_it_is_never_thresholded_on_d(c3):
    assert c3.catch_pass_probability(8) > c3.catch_pass_probability(7)
    assert c3.catch_pass_probability(5) < c3.catch_pass_probability(4)
    assert c3.catch_pass_probability(8) >= 0.85 > c3.catch_pass_probability(7)


# --- the table -----------------------------------------------------------------------------------


def _table(c3, *, spec=None, g4_se=1.07, guard_ses=None, discordant=1):
    return c3.power_table(spec or asp.CONSONANT3, g4_se=g4_se, guard_ses=guard_ses or SES,
                          discordant=discordant)


def test_the_table_has_a_row_per_guard_and_g4_at_both_margins(c3):
    rows = _table(c3)
    assert [row["guard"] for row in rows] == GUARDS
    assert [row["margin"] for row in rows] == [1.0, -1.0, -2.0, -2.0, 2.5, 3.5, -2.0]
    assert [row["effective"] for row in rows] == [True] * 4 + [False, True, True]


def test_each_row_names_its_set_and_where_its_se_comes_from(c3):
    for row in _table(c3):
        assert row["set"] and isinstance(row["set"], str)
        assert row["se_source"] and isinstance(row["se_source"], str)


def test_the_margins_are_the_spec_s_bars(c3):
    bars = asp.CONSONANT3.bars
    margins = {row["guard"]: row["margin"] for row in _table(c3) if row["effective"]}
    assert margins == {"G1": bars["qud_upper_at_most"], "G2": bars["tadabur_lower_at_least"],
                       "G3.consonant": bars["bench_lower_at_least"],
                       "G3.haraka": bars["bench_lower_at_least"],
                       "G4": bars["tashkeel_upper_at_most"], "G6": bars["catch_at_least"]}


def test_g4_prints_the_rejected_registered_row_beside_the_effective_one(c3):
    registered, effective = [row for row in _table(c3) if row["guard"] == "G4"]
    assert registered["p_pass"] == pytest.approx(0.646, abs=1e-3)
    assert registered["result"] == "under"
    assert "registered, rejected by the rule (A1)" in registered["note"]
    assert effective["p_pass"] == pytest.approx(0.905, abs=1e-3)
    assert effective["result"] == "pass"
    assert registered["se"] == effective["se"] == 1.07


def test_a_spec_with_no_registered_margin_prints_no_registered_row(c3):
    spec = dataclasses.replace(asp.CONSONANT3, registered_margins={})
    assert [row["guard"] for row in _table(c3, spec=spec)] == [
        "G1", "G2", "G3.consonant", "G3.haraka", "G4", "G6"]


def test_the_sparse_guards_use_the_pass_probability_of_their_own_se(c3):
    rows = {row["guard"]: row for row in _table(c3)}
    for guard, key in (("G1", "g1"), ("G2", "g2"), ("G3.consonant", "g3_consonant"),
                       ("G3.haraka", "g3_haraka")):
        assert rows[guard]["se"] == SES[key]
        assert rows[guard]["p_pass"] == c3.pass_probability(rows[guard]["margin"], SES[key])
        assert rows[guard]["result"] == "pass"


def test_g6_is_the_sign_test_of_its_discordant_count_and_has_no_se(c3):
    g6 = _table(c3, discordant=5)[-1]
    assert g6["guard"] == "G6" and g6["discordant"] == 5 and g6["se"] is None
    assert g6["p_pass"] == pytest.approx(0.8125)
    assert g6["result"] == "under"
    assert _table(c3, discordant=8)[-1]["result"] == "pass"


def test_the_threshold_is_the_spec_s_power_bar(c3):
    strict = dataclasses.replace(
        asp.CONSONANT3, bars={**asp.CONSONANT3.bars, "power_at_least": 0.95})
    effective = [row for row in _table(c3, spec=strict) if row["guard"] == "G4"][-1]
    assert effective["p_pass"] == pytest.approx(0.905, abs=1e-3)
    assert effective["result"] == "under"


# --- the rule ------------------------------------------------------------------------------------


def test_a_table_with_every_effective_row_at_or_above_the_bar_passes(c3):
    c3.require_power_rule(_table(c3))  # the rejected registered row alone refuses nothing


@pytest.mark.parametrize("kwargs, guard", [
    ({"g4_se": 1.3}, "G4"),
    ({"discordant": 5}, "G6"),
    ({"guard_ses": {**SES, "g1": 4.0}}, "G1"),
    ({"guard_ses": {**SES, "g2": 2.0}}, "G2"),
    ({"guard_ses": {**SES, "g3_consonant": 1.4}}, "G3.consonant"),
    ({"guard_ses": {**SES, "g3_haraka": 1.4}}, "G3.haraka")])
def test_an_effective_row_under_the_bar_refuses_by_naming_its_guard(c3, kwargs, guard):
    with pytest.raises(SystemExit, match=guard):
        c3.require_power_rule(_table(c3, **kwargs))


def test_the_effective_g4_margin_decides_not_the_registered_one(c3):
    """+2.5 is under at SE 1.07 and the rule still passes: A1's +3.5 is the effective margin."""
    rows = _table(c3)
    assert [r for r in rows if r["guard"] == "G4"][0]["result"] == "under"
    c3.require_power_rule(rows)


# --- ρ, from round 2's committed files -------------------------------------------------------------


def test_rho_is_round_2_s_realised_pooled_se_over_its_freeze_stand_in():
    from training import consonant3_readout as c3

    report = json.loads(c3.ROUND2_REPORT_PATH.read_text(encoding="utf-8"))
    interval = report["main"]["tashkeel_cause"]["comparisons"][
        am.comparison_name(am.CANDIDATE, am.C95)]["interval"]
    freeze = json.loads(c3.ROUND2_FREEZE_PATH.read_text(encoding="utf-8"))
    expected = ((interval["upper"] - interval["lower"]) / 3.92
                / freeze["power_check"]["tashkeel"]["se"])
    assert c3.rho() == pytest.approx(expected, rel=1e-12)
    assert c3.rho() == pytest.approx(0.998, abs=2e-3)  # 0.762 / 0.764


def test_the_round_2_paths_are_the_committed_artifacts():
    from training import consonant3_readout as c3

    assert c3.ROUND2_FREEZE_PATH == asp.CONSONANT2.root / "freeze.json"
    assert c3.ROUND2_REPORT_PATH == asp.CONSONANT2.root / "report.json"
    assert c3.ROUND2_FREEZE_PATH.exists() and c3.ROUND2_REPORT_PATH.exists()


# --- G6's d, through catch_arm ---------------------------------------------------------------------


def test_the_discordant_count_is_the_sites_one_model_flags_and_the_other_does_not(
        c3, monkeypatch):
    from training.muraja_policy import WORD_FLAGGED

    flagged = sorted(WORD_FLAGGED)[0]
    calls = []

    def catch_arm(sites, models, **kwargs):
        calls.append((tuple(models), kwargs))
        return {"models": {
            "r2_candidate_ema": {"outcomes": {"a": flagged, "b": flagged, "c": "not_flagged",
                                              "d": "not_flagged"}},
            "c95": {"outcomes": {"a": flagged, "b": "not_flagged", "c": flagged,
                                 "d": "not_flagged"}}}}

    monkeypatch.setattr(cr, "catch_arm", catch_arm)
    monkeypatch.setattr(cr, "g6_scope", lambda: {"sites": [], "candidates": 0, "excluded": {}})
    assert c3.discordant_sites(asp.CONSONANT3) == 2  # b and c
    (models, kwargs), = calls
    assert models == ("r2_candidate_ema", "c95")
    models_json = json.loads((asp.CONSONANT2.root / "models.json").read_text(encoding="utf-8"))
    assert kwargs["identities"] == {
        "r2_candidate_ema": models_json["tlog.look/r2_candidate_ema"]["identity"]}
    assert kwargs["spec"] is asp.CONSONANT3


def test_a_missing_cache_cannot_be_read_as_zero_discordant_sites(c3, monkeypatch):
    monkeypatch.setattr(cr, "catch_arm", lambda *args, **kwargs: None)
    monkeypatch.setattr(cr, "g6_scope", lambda: {"sites": [], "candidates": 0, "excluded": {}})
    with pytest.raises(REFUSED):
        c3.discordant_sites(asp.CONSONANT3)


# --- the known exclusions (#248 Part 1 items 2-3) ------------------------------------------------


def test_the_known_unphonetizable_ayat_are_the_phonetizer_s_fallback_set():
    assert tlog_eval.KNOWN_UNPHONETIZABLE == frozenset(generate_phonemes.FALLBACK_PHONEMES)
    assert isinstance(tlog_eval.KNOWN_UNPHONETIZABLE, frozenset)
    assert {"55:17", "90:8", "106:1", "106:4"} <= tlog_eval.KNOWN_UNPHONETIZABLE


def test_a_known_ayah_passes():
    tlog_eval.require_known_exclusions({"106_1_x.flac#0": "106:1", "55_17_y.flac#0": "55:17"})


def test_an_ayah_outside_the_set_refuses_by_naming_it():
    with pytest.raises(REFUSED, match="2:255"):
        tlog_eval.require_known_exclusions({"106_1_x.flac#0": "106:1", "2_255_y.flac#0": "2:255"})


def test_a_row_that_names_no_ayah_passes():
    tlog_eval.require_known_exclusions({"90_10_undefined_z.flac#0": None})
    tlog_eval.require_known_exclusions({})


def test_round_2_s_twelve_look_exclusions_name_only_known_ayat():
    freeze = json.loads((asp.CONSONANT2.root / "freeze.json").read_text(encoding="utf-8"))
    assert freeze["excluded"]["count"] == 12
    assert set(freeze["excluded"]["ayat"]) <= tlog_eval.KNOWN_UNPHONETIZABLE


def test_round_2_s_excluded_record_keeps_its_four_fields():
    record = tlog_eval.excluded_record({"b": "55:17", "a": None})
    assert set(record) == {"keys", "count", "ayat", "reason"}
    assert record["keys"] == ["a", "b"] and record["count"] == 2 and record["ayat"] == ["55:17"]
