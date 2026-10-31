"""Round 2's pre-claim phase (#230): the gates read before any tranche is claimed.

``training.consonant2_verdict`` (new) holds the interface these tests pin:

* ``PRECLAIM_GATES``: the pre-claim gates by name, in order. ``catch_loss`` is the audit; the rest
  are round 1's condition names: ``g3.qdat_bench.consonant``, ``g3.qdat_bench.haraka``,
  ``g6.ship_a_verified``, ``bar4a.qdat``, ``bar4b.qdat``, ``bar5.qdat``, ``g1.qud_other``,
  ``g2.tadabur_dev_unseen``. G3's shaddah part is reported, never a gate.
* ``preclaim_conditions(report, spec)``: the eight non-audit gates as condition dicts
  (``name``, ``verdict``, ...) from a report shaped as round 1's confirm report: ``mechanism``
  (``qdat``, ``qdat_departs``), ``consistency.qdat``, ``guards`` (``qud_other``,
  ``tadabur_dev_unseen``, ``qdat_bench.{consonant,haraka,shaddah}``), ``catch.ship_a_verified``.
  Each reads the frozen number in ``spec.bars``; a missing table cannot certify.
* ``audit_condition(result)``: the ``catch_loss`` condition from ``catch_loss/result.json`` as
  ``tadabur.catch_loss_audit.score`` writes it: its ``verdict`` is read, not re-derived, and the
  condition carries the result's ``removed`` counts.
* ``preclaim_report(sources, spec, choice, workers=1)``: reads the non-tlog sets once, for the
  chosen candidate and control (``choice`` is ``committed_choice(...).choice``), and returns the
  report above plus ``control``, the control's non-tlog columns. The tests replace it.
* ``run_preclaim(sources, spec, layout=None, workers=1)``: refuses without a committed
  ``choice.json`` (``SystemExit``, "not committed") and without a committed ``catch_loss/
  result.json`` (naming ``result.json``), before reading anything; reads once; writes
  ``preclaim.json`` (``Layout.preclaim``) with ``verdict`` (``pass`` / ``fail`` /
  ``cannot_certify``), ``gates`` (name -> condition), ``reported`` (the G3 shaddah line),
  ``control`` and the choice's commit. Anything but a pass also writes ``verdict.json``
  (``phase`` ``preclaim``, the same verdict) and the round's readout doc at ``spec.doc_path``;
  a pass writes neither. A second call refuses: the sitting reads once.

Harness-free: the non-tlog tables come from round 1's synthetic report fixture.
"""

from __future__ import annotations

import json

import pytest

from training import aug_readout_gate as ag
from training import round2_verdict_fixtures as rv
from training.acceptance_stats import CANNOT_CERTIFY, FAIL, PASS
from training.test_consonant_readout_verdict import _comparison, _interval

REFUSED = (SystemExit, ValueError)
GATES = ["catch_loss", "g3.qdat_bench.consonant", "g3.qdat_bench.haraka",
         "g6.ship_a_verified", "bar4a.qdat", "bar4b.qdat", "bar5.qdat", "g1.qud_other",
         "g2.tadabur_dev_unseen"]
CONTROL_COLUMNS = rv.CONTROL_COLUMNS


@pytest.fixture
def cv():
    from training import consonant2_verdict

    return consonant2_verdict


class Reads:
    """A ``preclaim_report`` replacement: records its calls and returns ``report``."""

    def __init__(self, report: dict | None = None):
        self.report = report or rv.clean_report()
        self.calls: list[tuple] = []

    def __call__(self, sources, spec, choice, workers=1):
        self.calls.append((sources, spec, choice["models"], workers))
        return self.report


class Sources:
    """The sitting's inputs, as the phase asks for them (nothing here is graded)."""

    def input_files(self, phase, weights):
        return []

    def harness_identity(self):
        return {"binary_sha256": "a" * 64}


def _round(tmp_path, real=10, correct=90, unclear=0):
    round_ = rv.committed_round(tmp_path)
    rv.plant_audit(round_, real, correct, unclear)
    return round_


def _run(cv, round_, monkeypatch, report: dict | None = None) -> tuple[dict, Reads]:
    reads = Reads(report)
    monkeypatch.setattr(cv, "preclaim_report", reads)
    return cv.run_preclaim(Sources(), round_.spec, round_.layout), reads


def _gate(preclaim: dict, name: str) -> dict:
    return preclaim["gates"][name]


# --- the gates, one by one ---------------------------------------------------------------------


def test_the_gates_are_the_pre_registered_list_in_order(cv):
    assert list(cv.PRECLAIM_GATES) == GATES


def _conditions(cv, report=None) -> dict[str, dict]:
    import training.aug_readout_spec as asp

    report = report or rv.clean_report()
    return {c["name"]: c for c in cv.preclaim_conditions(report, asp.CONSONANT2)}


def test_a_clean_report_passes_every_non_audit_gate(cv):
    conditions = _conditions(cv)
    assert list(conditions) == GATES[1:]
    assert {c["verdict"] for c in conditions.values()} == {PASS}


@pytest.mark.parametrize("edit, name", [
    (lambda r: _comparison(r, "guards", "qdat_bench", "consonant").update(
        interval=_interval(-2.01, 1.0)), "g3.qdat_bench.consonant"),  # ≥ −2.0 pt
    (lambda r: _comparison(r, "guards", "qdat_bench", "haraka").update(
        interval=_interval(-2.01, 1.0)), "g3.qdat_bench.haraka"),
    (lambda r: r["catch"]["ship_a_verified"].update(difference=-3), "g6.ship_a_verified"),
    (lambda r: _comparison(r, "mechanism", "qdat").update(
        interval=_interval(0.0, 1.0)), "bar4a.qdat"),  # > 0 is strict
    (lambda r: _comparison(r, "mechanism", "qdat_departs").update(
        interval=_interval(-0.51, 0.2)), "bar4b.qdat"),  # ≥ −0.5 pt
    (lambda r: r["consistency"]["qdat"]["models"]["candidate"].update(point=64.0),
     "bar5.qdat"),  # not below C95's
    (lambda r: _comparison(r, "guards", "qud_other").update(
        interval=_interval(-0.2, 1.01)), "g1.qud_other"),  # ≤ +1.0
    (lambda r: _comparison(r, "guards", "tadabur_dev_unseen").update(
        interval=_interval(-1.01, 0.4)), "g2.tadabur_dev_unseen"),  # ≥ −1.0 pt
])
def test_each_gate_fails_just_past_its_frozen_number(cv, edit, name):
    report = rv.clean_report()
    edit(report)
    conditions = _conditions(cv, report)
    assert conditions[name]["verdict"] == FAIL
    assert [n for n, c in conditions.items() if c["verdict"] != PASS] == [name]


@pytest.mark.parametrize("path, name", [
    (("mechanism", "qdat"), "bar4a.qdat"),
    (("mechanism", "qdat_departs"), "bar4b.qdat"),
    (("consistency", "qdat"), "bar5.qdat"),
    (("guards", "qud_other"), "g1.qud_other"),
    (("guards", "tadabur_dev_unseen"), "g2.tadabur_dev_unseen"),
    (("guards", "qdat_bench", "consonant"), "g3.qdat_bench.consonant"),
    (("guards", "qdat_bench", "haraka"), "g3.qdat_bench.haraka"),
])
def test_a_table_that_was_not_read_cannot_certify_its_gate(cv, path, name):
    report = rv.clean_report()
    holder = report
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = None
    assert _conditions(cv, report)[name]["verdict"] == CANNOT_CERTIFY


def test_no_g6_read_cannot_certify_the_catch_gate(cv):
    report = rv.clean_report()
    report["catch"]["ship_a_verified"] = None
    assert _conditions(cv, report)["g6.ship_a_verified"]["verdict"] == CANNOT_CERTIFY


def test_a_g3_shaddah_only_failure_leaves_the_pre_claim_gates_passing(cv):
    report = rv.clean_report()
    _comparison(report, "guards", "qdat_bench", "shaddah").update(
        interval=_interval(-9.0, -5.0))
    assert "g3.qdat_bench.shaddah" not in _conditions(cv, report)
    assert {c["verdict"] for c in _conditions(cv, report).values()} == {PASS}


# --- the audit -------------------------------------------------------------------------------------


@pytest.mark.parametrize("real, correct, unclear, verdict", [
    (20, 80, 0, PASS), (21, 79, 0, FAIL), (16, 64, 20, PASS),
    (0, 79, 21, CANNOT_CERTIFY), (79, 0, 21, CANNOT_CERTIFY)])
def test_the_audit_gate_is_the_scorer_s_verdict(cv, tmp_path, real, correct, unclear, verdict):
    round_ = rv.committed_round(tmp_path)
    result = rv.plant_audit(round_, real, correct, unclear)
    condition = cv.audit_condition(result)
    assert condition["name"] == "catch_loss" and condition["verdict"] == verdict
    assert condition["removed"] == result["removed"]


# --- the phase: what it refuses ----------------------------------------------------------------


def test_the_phase_refuses_without_a_committed_choice_and_reads_nothing(cv, tmp_path,
                                                                        monkeypatch):
    round_ = _round(tmp_path)
    elsewhere = ag.Layout(round_.repo / "no-choice-here")
    reads = Reads()
    monkeypatch.setattr(cv, "preclaim_report", reads)
    with pytest.raises(SystemExit, match="not committed"):
        cv.run_preclaim(Sources(), round_.spec, elsewhere)
    assert reads.calls == [] and not round_.preclaim.exists()


def test_the_phase_refuses_without_the_audit_result_and_reads_nothing(cv, tmp_path, monkeypatch):
    round_ = rv.committed_round(tmp_path)
    reads = Reads()
    monkeypatch.setattr(cv, "preclaim_report", reads)
    with pytest.raises(REFUSED, match="result.json"):
        cv.run_preclaim(Sources(), round_.spec, round_.layout)
    assert reads.calls == [] and not round_.preclaim.exists()


def test_an_audit_result_that_is_not_committed_counts_for_nothing(cv, tmp_path, monkeypatch):
    round_ = rv.committed_round(tmp_path)
    rv.plant_audit(round_, 10, 90, commit=False)
    reads = Reads()
    monkeypatch.setattr(cv, "preclaim_report", reads)
    with pytest.raises(REFUSED, match="result.json"):
        cv.run_preclaim(Sources(), round_.spec, round_.layout)
    assert reads.calls == []


def test_an_audit_result_edited_since_its_commit_counts_for_nothing(cv, tmp_path, monkeypatch):
    round_ = _round(tmp_path)
    path = round_.catch_loss / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["verdict"] = "pass"
    result["removed"]["real"] = 0
    path.write_text(json.dumps(result), encoding="utf-8")
    reads = Reads()
    monkeypatch.setattr(cv, "preclaim_report", reads)
    with pytest.raises(REFUSED, match="result.json"):
        cv.run_preclaim(Sources(), round_.spec, round_.layout)
    assert reads.calls == []


def test_the_sitting_reads_once(cv, tmp_path, monkeypatch):
    round_ = _round(tmp_path)
    _run(cv, round_, monkeypatch)
    with pytest.raises(SystemExit):
        cv.run_preclaim(Sources(), round_.spec, round_.layout)


# --- the phase: outcomes -------------------------------------------------------------------------


def test_every_gate_passing_writes_a_pass_and_no_verdict_or_doc(cv, tmp_path, monkeypatch):
    round_ = _round(tmp_path)
    preclaim, reads = _run(cv, round_, monkeypatch)

    assert preclaim["verdict"] == PASS
    assert list(preclaim["gates"]) == GATES
    assert {gate["verdict"] for gate in preclaim["gates"].values()} == {PASS}
    assert json.loads(round_.preclaim.read_text(encoding="utf-8")) == preclaim
    assert not round_.layout.verdict.exists() and not round_.spec.doc_path.exists()
    ((_, spec, models, _),) = reads.calls  # the non-tlog sets, read once, for the chosen arms
    assert spec is round_.spec
    assert models == {"candidate": "r2_candidate_ema", "control": "r2_control_ema"}


def test_the_preclaim_records_the_choice_the_control_columns_and_the_shaddah_line(
        cv, tmp_path, monkeypatch):
    round_ = _round(tmp_path)
    report = rv.clean_report()
    _comparison(report, "guards", "qdat_bench", "shaddah").update(
        interval=_interval(-9.0, -5.0))
    preclaim, _ = _run(cv, round_, monkeypatch, report)

    assert preclaim["verdict"] == PASS  # shaddah is reported only (owner, #173)
    assert preclaim["control"] == CONTROL_COLUMNS
    assert preclaim["reported"]["g3.qdat_bench.shaddah"]["verdict"] == FAIL
    committed = ag.committed_choice(round_.layout, spec=round_.spec)
    assert preclaim["choice_commit"] == committed.commit
    assert preclaim["choice_sha256"] == committed.sha256


@pytest.mark.parametrize("real, correct, unclear, outcome", [
    (21, 79, 0, FAIL), (0, 79, 21, CANNOT_CERTIFY)])
def test_an_audit_that_does_not_pass_ends_the_round(cv, tmp_path, monkeypatch, real, correct,
                                                    unclear, outcome):
    round_ = _round(tmp_path, real, correct, unclear)
    preclaim, _ = _run(cv, round_, monkeypatch)

    assert preclaim["verdict"] == outcome
    assert _gate(preclaim, "catch_loss")["verdict"] == outcome
    assert _gate(preclaim, "g1.qud_other")["verdict"] == PASS
    final = json.loads(round_.layout.verdict.read_text(encoding="utf-8"))
    assert (final["verdict"], final["phase"]) == (outcome, "preclaim")
    doc = round_.spec.doc_path.read_text(encoding="utf-8")
    assert "clip-level split, same users possible on both sides" in doc
    assert "catch_loss" in doc


def test_a_failed_non_tlog_gate_ends_the_round_with_the_control_beside_it(cv, tmp_path,
                                                                          monkeypatch):
    round_ = _round(tmp_path)
    report = rv.clean_report()
    _comparison(report, "guards", "qud_other").update(interval=_interval(-0.2, 1.5))
    preclaim, _ = _run(cv, round_, monkeypatch, report)

    assert preclaim["verdict"] == FAIL
    assert _gate(preclaim, "g1.qud_other")["verdict"] == FAIL
    assert preclaim["control"] == CONTROL_COLUMNS
    assert json.loads(round_.layout.verdict.read_text(encoding="utf-8"))["verdict"] == FAIL
    assert round_.spec.doc_path.is_file()


def test_a_fail_outranks_a_cannot_certify(cv, tmp_path, monkeypatch):
    round_ = _round(tmp_path, 0, 79, 21)  # the audit cannot certify
    report = rv.clean_report()
    _comparison(report, "guards", "qud_other").update(interval=_interval(-0.2, 1.5))
    preclaim, _ = _run(cv, round_, monkeypatch, report)
    assert preclaim["verdict"] == FAIL


def test_a_missing_input_cannot_certify_the_round(cv, tmp_path, monkeypatch):
    round_ = _round(tmp_path)
    report = rv.clean_report()
    report["guards"]["tadabur_dev_unseen"] = None
    preclaim, _ = _run(cv, round_, monkeypatch, report)
    assert preclaim["verdict"] == CANNOT_CERTIFY
    assert _gate(preclaim, "g2.tadabur_dev_unseen")["verdict"] == CANNOT_CERTIFY
    assert json.loads(round_.layout.verdict.read_text(encoding="utf-8"))["phase"] == "preclaim"


# --- the command ---------------------------------------------------------------------------------


def test_the_preclaim_command_takes_the_round_2_spec_only(cv, monkeypatch):
    import sys

    from training import aug_readout_spec as asp
    from training import consonant_readout as cr

    seen = {}

    def run(sources, spec, layout=None, workers=1):
        seen.update(spec=spec, sources_spec=sources.spec, root=layout.root if layout else None)
        return {"verdict": PASS}

    monkeypatch.setattr(cv, "run_preclaim", run)
    monkeypatch.setattr(sys, "argv", ["consonant_readout", "preclaim", "--spec", "consonant2"])
    cr.main()
    assert seen["spec"] is asp.CONSONANT2 and seen["sources_spec"] is asp.CONSONANT2
    assert seen["root"] in (None, asp.CONSONANT2.root)

    monkeypatch.setattr(sys, "argv", ["consonant_readout", "preclaim", "--spec", "consonant1"])
    with pytest.raises(SystemExit) as stopped:
        cr.main()
    assert stopped.value.code == 2
