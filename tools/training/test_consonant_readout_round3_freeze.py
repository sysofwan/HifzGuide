"""The round-3 freeze phase (#251): tlog look gating only, the power rule, the known exclusions.

``python -m training.consonant_readout freeze --spec consonant3 --half look`` grades C95, round
2's EMA (``r2``) and the teacher on tlog look gating, and writes ``freeze.json`` and its doc only
if every guard's effective-margin pass probability is at the spec's ``power_at_least``. Every
source here is synthetic; round 2's committed freeze and report are replaced by fixtures through
the two module constants below.

The interface these tests pin, beyond ``test_consonant_readout_round3_power``:

``training.consonant3_readout``
  * ``ROUND2_FREEZE_PATH`` / ``ROUND2_REPORT_PATH``: round 2's artifacts, read at call time (the
    G5 list under ``substitutions``, the 12 look exclusions under ``excluded.keys``, ρ's inputs);
  * ``guard_ses(sources, spec)``: G1/G2/G3's paired bootstrap SEs, keyed ``g1``, ``g2``,
    ``g3_consonant``, ``g3_haraka``; ``discordant_sites(spec)``: G6's ``d``. ``freeze_record``
    calls both through the module, so a test replaces them;
  * ``freeze_record(sources, spec)`` and ``render(record)``.

``consonant_readout.write_freeze(sources, "look", path, doc_path, spec=CONSONANT3)`` and
``run_freeze(spec, sources, "look")`` dispatch to it. The record holds:

  * ``round`` ``"consonant3"``, ``half``, ``items``, ``pins.prereg_sha256``, ``inputs``;
  * ``models`` for ``c95``, ``r2``, ``base`` (as round 2's) and ``substitutions`` (G5's list);
  * ``excluded``: round 2's ``keys``/``count``/``ayat``/``reason`` plus ``by_reason``
    ``{"unphonetizable": n, "names_no_ayah": m}``;
  * ``power_table`` (the rows of ``power_table``), ``power_reported`` (two report-only lines,
    each ``label`` and ``se``), ``g5_se`` (per frozen substitution: ``substitution``, ``se``,
    ``events`` per model, ``zero_events``) and ``power_rule`` ``{"satisfied": True,
    "amendments": ["A1"]}``.

A dry run returns the same record without writing; every refusal comes before any file is written.
A refusal's message names ``G5`` for the list, ``round 2`` for a look-exclusion mismatch, the
ayah for an unknown one, and the guard (``G4``, ``G6``, ...) for a power shortfall.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys

import pytest

from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import aug_readout_metrics as am
from training import aug_readout_spec as asp
from training import consonant_readout as cr
from training import round2_fixtures as fx
from training.test_consonant_readout_round2_freeze import (
    DAL_TA, TALLY, TASHKEEL_WORDS, TEACHER_FLAGS, TOP_TEN, UNGRADED, _plan)
from training.truth_baseline import CHECK_STORE

SHIPPED = fx.SHIPPED
SE_GUARDS = {"g1": 0.19, "g2": 0.05, "g3_consonant": 0.45, "g3_haraka": 0.49}
#: Round 2's look exclusions, as the fixture's tranche leaves them out: two known ayat and a row
#: naming no ayah.
EXCLUDED = {"106_1_a.flac#0": "106:1", "55_17_b.flac#0": "55:17",
            "90_10_undefined_c.flac#0": None}
EXTRA_TASHKEEL = 8
DROPPED_ZAY_SIN = 3
TALLY_NO_DAL = tuple(row for row in TALLY if row[0] != DAL_TA)


def _r2_plan(c95: list) -> list:
    """Round 2's EMA: C95's flags less three ز→س words, plus eight tashkeel flags."""
    plan = list(c95)
    free = [w for w, flag in enumerate(c95) if flag is None][:-UNGRADED]
    for w in range(DROPPED_ZAY_SIN):
        plan[w] = None
    for w in free[::10][:EXTRA_TASHKEEL]:
        plan[w] = fx.TASHKEEL_FLAG
    return plan


class Round3Sources(fx.TlogSources):
    """Tlog look gating, synthetic, with the rows the item builder left out."""

    def __init__(self, *args, excluded, **kwargs):
        super().__init__(*args, **kwargs)
        self.excluded = excluded
        self.models_read: tuple = ()

    def tlog(self, role, models):
        self.models_read = tuple(models)
        return dataclasses.replace(super().tlog(role, models), excluded=self.excluded)


class Run:
    """One round-3 freeze over synthetic look gating, in a repository holding the
    pre-registration, against fixtures standing in for round 2's committed artifacts."""

    def __init__(self, tmp_path, monkeypatch, *, tally=TALLY, committed=True, pinned=True,
                 rho=0.01, excluded=EXCLUDED, round2_excluded=None, round2_list=None,
                 guard_ses=SE_GUARDS, discordant=1):
        from training import consonant3_readout as c3

        self.repo = fx.init_repo(tmp_path / "repo")
        write = fx.commit_prereg if committed else fx.write_prereg
        self.prereg, self.sha = write(self.repo)
        self.spec = dataclasses.replace(
            asp.CONSONANT3, prereg_path=self.prereg,
            pins=dataclasses.replace(asp.CONSONANT3.pins,
                                     prereg_sha256=self.sha if pinned else "0" * 64))
        plan = _plan(tally)
        self.plan = {"c95": plan, "r2": _r2_plan(plan),
                     "base": [fx.consonant_flag("ز→س") if w < TEACHER_FLAGS else None
                              for w in range(len(plan))]}
        self.items = [fx.tlog_item(n, 1 + n % 14) for n in range(len(plan) // 2)]
        words = len(plan)
        self.sources = Round3Sources(
            tmp_path / "files", self.items, lambda model, i, p: self.plan[model][2 * i + p],
            graded=lambda i, p: 2 * i + p < words - UNGRADED, spec=self.spec,
            excluded=excluded)
        self.record_path = tmp_path / "freeze.json"
        self.doc_path = tmp_path / "freeze.md"
        self.c3 = c3
        self.rho = rho
        self._round2_files(tmp_path, monkeypatch, excluded if round2_excluded is None
                           else round2_excluded, round2_list)
        monkeypatch.setattr(c3, "guard_ses", lambda sources, spec: dict(guard_ses))
        monkeypatch.setattr(c3, "discordant_sites", lambda spec: discordant)

    def _round2_files(self, tmp_path, monkeypatch, excluded, listed):
        """Round 2's freeze and report, written so that ρ is ``self.rho``."""
        listed = [*TOP_TEN, DAL_TA] if listed is None else listed
        stand_in = 1.0
        half_width = self.rho * stand_in * 3.92 / 2
        comparison = am.comparison_name(am.CANDIDATE, am.C95)
        freeze = {"substitutions": [{"substitution": name} for name in listed],
                  "excluded": {"keys": sorted(excluded)},
                  "power_check": {"tashkeel": {"se": stand_in}}}
        report = {"main": {"tashkeel_cause": {"comparisons": {comparison: {"interval": {
            "lower": -half_width, "upper": half_width}}}}}}
        freeze_path, report_path = tmp_path / "round2_freeze.json", tmp_path / "round2_report.json"
        freeze_path.write_text(json.dumps(freeze, ensure_ascii=False), encoding="utf-8")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        monkeypatch.setattr(self.c3, "ROUND2_FREEZE_PATH", freeze_path)
        monkeypatch.setattr(self.c3, "ROUND2_REPORT_PATH", report_path)

    def se_look(self) -> float:
        """SE_look(r2 EMA − C95): tashkeel false flags per 1,000 on the look gating words."""
        grades = self.sources.grade(self.items, {}, ("r2", "c95"))
        terms = am.false_flag_terms(grades, ("r2", "c95"), SHIPPED, cause=am.TASHKEEL_CAUSE)
        return am.paired_se(terms, "r2", "c95", am.PER_1000)

    def write(self) -> dict:
        return cr.write_freeze(self.sources, "look", path=self.record_path,
                               doc_path=self.doc_path, spec=self.spec)

    def refuses(self, match: str):
        with pytest.raises(SystemExit, match=match):
            self.write()
        assert not self.record_path.exists() and not self.doc_path.exists()


@pytest.fixture
def run(tmp_path, monkeypatch):
    return Run(tmp_path, monkeypatch)


def _row(record: dict, guard: str, *, effective: bool = True) -> dict:
    rows = [r for r in record["power_table"]
            if r["guard"] == guard and r["effective"] is effective]
    assert len(rows) == 1
    return rows[0]


# --- what it writes --------------------------------------------------------------------------------


def test_the_freeze_writes_its_record_and_its_doc_and_returns_the_record(run):
    record = run.write()
    assert json.loads(run.record_path.read_text(encoding="utf-8")) == record
    assert run.doc_path.exists()
    assert record["round"] == "consonant3" and record["half"] == "look"
    assert record["items"] == len(run.items)
    assert record["pins"]["prereg_sha256"] == run.sha
    assert run.sources.read == ["look"]  # tlog look gating, once; no other set, no confirm row


def test_the_freeze_needs_no_round_3_candidate_identity(run):
    assert set(run.spec.models) == {"c95", "r2"}
    run.write()
    assert run.sources.models_read == ("c95", "r2", "base")


def test_the_cause_split_is_c95_beside_the_round_2_ema_and_the_teacher(run):
    record = run.write()
    assert set(record["models"]) == {"c95", "r2", "base"}
    flagged = sum(words for _, _, words in TALLY) + TASHKEEL_WORDS
    shipped = {m: record["models"][m]["scorings"]["shipped_default"] for m in record["models"]}
    assert shipped["c95"]["flagged"] == flagged
    assert shipped["r2"]["flagged"] == flagged - DROPPED_ZAY_SIN + EXTRA_TASHKEEL
    assert shipped["r2"]["causes"]["tashkeel"]["words"] == TASHKEEL_WORDS + EXTRA_TASHKEEL
    assert shipped["base"]["flagged"] == TEACHER_FLAGS


def test_the_inputs_digest_the_decodes_and_the_g6_caches_and_identity(run):
    record = run.write()
    decodes = ag.digests(run.sources.decode_path("tlog.look", m) for m in ("c95", "r2", "base"))
    caches = [CHECK_STORE.directory / f"{m}.json" for m in ("r2_candidate_ema", "c95")]
    models_json = asp.CONSONANT2.root / "models.json"
    assert decodes.items() <= record["inputs"].items()
    assert ag.digests([*caches, models_json]).items() <= record["inputs"].items()


# --- G5 --------------------------------------------------------------------------------------------


def test_g5_is_round_2_s_list_reproduced_exactly(run):
    record = run.write()
    assert [row["substitution"] for row in record["substitutions"]] == [*TOP_TEN, DAL_TA]


@pytest.mark.parametrize("listed", [
    [*TOP_TEN[:-1], DAL_TA],  # one fewer
    [*reversed(TOP_TEN), DAL_TA],  # the same set, another order
    [*TOP_TEN[:-1], "ع→ء", DAL_TA],  # one substituted
    TOP_TEN])  # without the د→ت
def test_a_g5_list_that_differs_from_round_2_s_refuses(tmp_path, monkeypatch, listed):
    Run(tmp_path, monkeypatch, round2_list=listed).refuses("G5")


def test_g5_reports_a_se_per_substitution_and_events_per_model(run):
    record = run.write()
    rows = {row["substitution"]: row for row in record["g5_se"]}
    assert list(rows) == [*TOP_TEN, DAL_TA]
    zay_sin = rows["ز→س"]
    assert zay_sin["events"] == {"c95": 12, "r2": 12 - DROPPED_ZAY_SIN}
    assert zay_sin["se"] > 0 and math.isfinite(zay_sin["se"])
    assert not zay_sin["zero_events"]
    assert not any(row["zero_events"] for row in rows.values())


def test_g5_flags_a_substitution_with_no_event_in_either_model(tmp_path, monkeypatch):
    record = Run(tmp_path, monkeypatch, tally=TALLY_NO_DAL).write()
    rows = {row["substitution"]: row for row in record["g5_se"]}
    assert rows[DAL_TA]["events"] == {"c95": 0, "r2": 0}
    assert rows[DAL_TA]["zero_events"] is True and rows[DAL_TA]["se"] == 0.0
    assert [name for name, row in rows.items() if row["zero_events"]] == [DAL_TA]


def test_g5_has_no_pass_probability_it_is_reported_not_tested(run):
    record = run.write()
    assert "G5" not in {row["guard"] for row in record["power_table"]}
    assert all("p_pass" not in row for row in record["g5_se"])


# --- the look exclusions ---------------------------------------------------------------------------


def test_the_look_exclusions_equal_round_2_s_and_are_split_by_reason(run):
    excluded = run.write()["excluded"]
    assert excluded["keys"] == sorted(EXCLUDED) and excluded["count"] == 3
    assert excluded["ayat"] == ["106:1", "55:17"]
    assert excluded["by_reason"] == {"unphonetizable": 2, "names_no_ayah": 1}
    assert excluded["reason"]


def test_an_exclusion_round_2_did_not_have_refuses(tmp_path, monkeypatch):
    Run(tmp_path, monkeypatch, round2_excluded={"106_1_a.flac#0": "106:1",
                                                "90_10_undefined_c.flac#0": None}
        ).refuses("round 2")


def test_an_exclusion_round_2_had_and_this_freeze_lacks_refuses(tmp_path, monkeypatch):
    Run(tmp_path, monkeypatch, round2_excluded={**EXCLUDED, "90_8_d.flac#0": "90:8"}
        ).refuses("round 2")


def test_an_excluded_row_naming_an_unknown_ayah_refuses_even_when_round_2_had_it(
        tmp_path, monkeypatch):
    both = {**EXCLUDED, "2_255_z.flac#0": "2:255"}
    Run(tmp_path, monkeypatch, excluded=both, round2_excluded=both).refuses("2:255")


# --- the power-rule table --------------------------------------------------------------------------


def test_g4_is_the_unscaled_look_se_times_rho_at_both_margins(tmp_path, monkeypatch):
    run = Run(tmp_path, monkeypatch, rho=0.01)
    expected = run.se_look() * 0.01
    assert run.se_look() > 0
    record = run.write()
    registered, effective = _row(record, "G4", effective=False), _row(record, "G4")
    assert registered["margin"] == 2.5 and effective["margin"] == 3.5
    assert registered["se"] == pytest.approx(expected, rel=1e-9)
    assert effective["se"] == pytest.approx(expected, rel=1e-9)
    assert effective["p_pass"] == run.c3.pass_probability(3.5, effective["se"])
    assert "registered, rejected by the rule (A1)" in registered["note"]


def test_rho_scales_g4_s_se(tmp_path, monkeypatch):
    run = Run(tmp_path, monkeypatch, rho=0.02)
    assert _row(run.write(), "G4")["se"] == pytest.approx(run.se_look() * 0.02, rel=1e-9)


def test_the_two_reported_lines_are_labelled_and_never_feed_the_rule(tmp_path, monkeypatch):
    run = Run(tmp_path, monkeypatch, rho=0.01)
    record = run.write()
    scaled, realised = record["power_reported"]
    for line in (scaled, realised):
        assert "assumes no further divergence from C95" in line["label"]
    assert scaled["se"] == pytest.approx(run.se_look() * math.sqrt(2853 / 5670), rel=1e-9)
    assert realised["se"] == pytest.approx(0.01, rel=1e-9)  # (UB − LB) / 3.92 of round 2
    for row in (r for r in record["power_table"] if r["guard"] == "G4"):
        assert row["se"] != pytest.approx(scaled["se"], rel=1e-3)
        assert row["se"] != pytest.approx(realised["se"], rel=1e-3) or run.se_look() == 1.0


def test_the_sparse_guards_and_g6_come_from_their_own_readings(tmp_path, monkeypatch):
    record = Run(tmp_path, monkeypatch, discordant=1).write()
    for guard, key in (("G1", "g1"), ("G2", "g2"), ("G3.consonant", "g3_consonant"),
                       ("G3.haraka", "g3_haraka")):
        assert _row(record, guard)["se"] == SE_GUARDS[key]
    assert _row(record, "G6")["discordant"] == 1
    assert _row(record, "G6")["p_pass"] == 1.0
    assert all(r["result"] == "pass" for r in record["power_table"] if r["effective"])


def test_the_record_carries_the_satisfied_rule_and_its_amendment(run):
    assert run.write()["power_rule"] == {"satisfied": True, "amendments": ["A1"]}


# --- the refusal -----------------------------------------------------------------------------------


def test_an_effective_g4_under_the_bar_refuses_the_whole_freeze(tmp_path, monkeypatch):
    run = Run(tmp_path, monkeypatch, rho=100.0)  # SE_G4 ≈ 100 × SE_look: far under 0.85
    run.refuses("G4")


@pytest.mark.parametrize("kwargs, guard", [
    ({"discordant": 5}, "G6"),
    ({"guard_ses": {**SE_GUARDS, "g1": 4.0}}, "G1"),
    ({"guard_ses": {**SE_GUARDS, "g3_haraka": 1.4}}, "G3.haraka")])
def test_any_effective_guard_under_the_bar_refuses_the_whole_freeze(tmp_path, monkeypatch,
                                                                    kwargs, guard):
    Run(tmp_path, monkeypatch, **kwargs).refuses(guard)


def test_g6_is_judged_by_its_sign_test_not_by_a_threshold_on_d(tmp_path, monkeypatch):
    Run(tmp_path, monkeypatch, discordant=8).write()  # 0.855: passes
    Run(tmp_path / "again", monkeypatch, discordant=7).refuses("G6")  # 0.773: refuses


def test_a_freeze_that_moves_nothing_still_never_moves_an_existing_freeze(run):
    run.record_path.write_text('{"kept": 1}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="exists"):
        run.write()
    assert run.record_path.read_text(encoding="utf-8") == '{"kept": 1}\n'
    assert not run.doc_path.exists()


# --- the pre-registration comes first --------------------------------------------------------------


def test_the_freeze_refuses_without_a_committed_pre_registration_and_grades_nothing(
        tmp_path, monkeypatch):
    run = Run(tmp_path, monkeypatch, committed=False)
    run.refuses("committed")
    assert run.sources.read == [] and run.sources.grade_calls == 0


def test_the_freeze_refuses_an_unpinned_pre_registration_before_grading(tmp_path, monkeypatch):
    run = Run(tmp_path, monkeypatch, pinned=False)
    run.refuses("pin")
    assert run.sources.grade_calls == 0


def test_the_freeze_reads_look_gating_only(run):
    with pytest.raises(SystemExit, match="confirm"):
        cr.write_freeze(run.sources, "confirm", path=run.record_path, doc_path=run.doc_path,
                        spec=run.spec)
    assert run.sources.read == [] and run.sources.grade_calls == 0
    assert not run.record_path.exists()


def test_the_real_round_3_spec_pins_the_pre_registration_committed_in_head():
    """#253's merge state: the document is committed and its pin holds, so the freeze may
    grade. The scratch ``committed=False`` run above keeps the pre-merge refusal covered."""
    from training.aug_readout_gate import require_preregistration_text

    assert require_preregistration_text(asp.CONSONANT3) == asp.CONSONANT3.pins.prereg_sha256


# --- the doc and the command -----------------------------------------------------------------------


def test_the_doc_prints_both_g4_rows_and_the_reported_lines(run):
    record = run.write()
    text = run.doc_path.read_text(encoding="utf-8")
    assert "registered, rejected by the rule (A1)" in text
    assert "assumes no further divergence from C95" in text
    assert "3.5" in text and "2.5" in text
    assert cr.freeze_doc_path(run.spec) == run.spec.freeze_doc_path
    assert run.c3.render(record) == text


def test_the_round_s_own_freeze_paths():
    assert cr.freeze_path(asp.CONSONANT3) == asp.CONSONANT3.root / "freeze.json"
    assert cr.freeze_doc_path(asp.CONSONANT3).name == "consonant-round3-freeze.md"


def test_the_dry_run_prints_the_table_and_writes_nothing(run, monkeypatch, capsys):
    monkeypatch.setitem(asp.SPECS, "consonant3", run.spec)
    monkeypatch.setattr(ar, "Sources", lambda spec: run.sources)
    monkeypatch.setattr(sys, "argv", ["consonant_readout", "freeze", "--spec", "consonant3",
                                      "--half", "look", "--dry-run"])
    cr.main()
    printed = json.loads(capsys.readouterr().out)
    assert [row["guard"] for row in printed["power_table"]] == [
        "G1", "G2", "G3.consonant", "G3.haraka", "G4", "G4", "G6"]
    assert printed["power_rule"] == {"satisfied": True, "amendments": ["A1"]}
    assert run.sources.read == ["look"]
    assert not run.record_path.exists() and not run.doc_path.exists()


def test_the_dry_run_grades_nothing_before_the_pre_registration_is_committed(
        tmp_path, monkeypatch):
    run = Run(tmp_path, monkeypatch, committed=False)
    with pytest.raises(SystemExit, match="committed"):
        cr.run_freeze(run.spec, run.sources, "look")
    assert run.sources.read == [] and run.sources.grade_calls == 0


def test_round_2_s_freeze_command_still_takes_its_own_dry_run(monkeypatch, tmp_path, capsys):
    """``--spec consonant2`` still summarises C95 alone: round 3's dispatch leaves it be."""
    from training.test_consonant_readout_round2_freeze import Run as Round2Run

    run = Round2Run(tmp_path, _plan())
    monkeypatch.setitem(asp.SPECS, "consonant2", run.spec)
    monkeypatch.setattr(ar, "Sources", lambda spec: run.sources)
    monkeypatch.setattr(sys, "argv", ["consonant_readout", "freeze", "--spec", "consonant2",
                                      "--half", "look", "--dry-run"])
    cr.main()
    assert json.loads(capsys.readouterr().out)["model"] == "c95"


# --- the inputs the decision reads ---------------------------------------------------------------


def test_the_inputs_fingerprint_every_source_the_power_rule_reads(run):
    """R1.1: ρ's artifacts, the guard sets' decodes and G6's decode caches decide the freeze, so
    each of them is in the record's ``inputs``."""
    record = run.write()
    expected = [run.c3.ROUND2_REPORT_PATH, run.c3.ROUND2_FREEZE_PATH, run.c3.ROUND2_MODELS_PATH,
                *(run.sources.decode_path(name, model)
                  for name in run.c3.GUARD_SETS for model in run.c3.POWER_MODELS),
                *(CHECK_STORE.directory / f"{m}.json" for m in ("r2_candidate_ema", "c95"))]
    assert ag.digests(expected).items() <= record["inputs"].items()


def test_changing_a_power_rule_source_moves_its_recorded_digest(run):
    before = ag.digests(run.c3.freeze_inputs(run.sources))
    run.c3.ROUND2_REPORT_PATH.write_text('{"main": {}}\n', encoding="utf-8")
    after = ag.digests(run.c3.freeze_inputs(run.sources))
    name = ag.input_name(run.c3.ROUND2_REPORT_PATH)
    assert before[name] != after[name]
