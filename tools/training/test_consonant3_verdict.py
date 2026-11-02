"""Round 3's readout (#255): the pre-claim and pooled verdict over ``confirm.3`` + ``confirm.4``,
with no control arm, through the code round 2 uses.

``training.consonant2_verdict`` reads the round's tranches and arms from its spec:

* **Tranches.** ``run_verdict`` requires the committed claims of ``spec.confirm_tranches`` (round 3:
  ``confirm.3`` and ``confirm.4``, both, before it reads either) and pools exactly those into one
  set clustered by surah; ``report["tranches"]`` is keyed by them. ``render``'s text names them,
  never ``confirm.1``/``confirm.2``. The tranche-reading ``Sources`` methods
  (``tlog_views``) accept the spec's tranches only, and the confirm sitting declares those
  tranches' manifests as inputs.
* **No control** (``CONTROL not in spec.look_roles``). ``preclaim_report`` reads the candidate, C95
  and the teacher without a ``control`` key; ``preclaim.json``, the ended round's ``verdict.json``,
  ``report.json`` and ``verdict.json`` carry no ``control`` key and no ``candidate_vs_control``; the
  docs have no control section. A round with a control (round 2) is unchanged: the SHA-256 pins
  below are ``render`` over round 2's committed ``preclaim.json`` and ``report.json`` as of #249.
* **G4** is read from ``spec.bars``: round 3's +3.5 (amendment A1), its bar text says so, and the
  verdict's doc quotes the pre-registration's "required disclosure" lines verbatim.
* **Known exclusions.** The two tranches' left-out rows go through
  ``tlog_eval.require_known_exclusions`` (``SystemExit`` naming the ayah) before ``sources.grade``
  is ever called, and are reported by reason:
  ``report["coverage"]["graded"]["excluded_clips"]["by_reason"]`` is
  ``tlog_eval.exclusion_reasons`` of the union.
* **The r2 model is never a confirm decode target** (the sitting's roster holds the candidate at
  its chosen weights, C95 and the teacher).

Synthetic grades; nothing runs a harness. ``test_consonant2_verdict`` pins the shared mechanics.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re

import pytest

from tadabur.tarteel_split import CONFIRM
from training import aug_readout as ar
from training import aug_readout_spec as asp
from training import consonant_readout as cr
from training import round2_verdict_fixtures as rv
from training import round3_verdict_fixtures as r3
from training import tlog_eval
from training.acceptance_stats import FAIL, PASS
from training.consonant2_readout import input_files, sitting_roster
from training.test_consonant2_verdict import (
    ALL, SUBSTITUTION, ConfirmSources, _flagged, _item, _pooled)
from training.test_consonant_readout_verdict import _comparison, _interval

REFUSED = (SystemExit, ValueError)
ROUND3 = asp.CONSONANT3
STATEMENT = "clip-level split, same users possible on both sides"
TRANCHES = ("confirm.3", "confirm.4")
TRANCHE_ITEMS = {"confirm.3": range(0, 20), "confirm.4": range(20, 40)}
BARS = ["bar1.all_cause", "bar2.reduction", "bar3.consonant_cause", "bar4a.tlog_confirm",
        "bar4b.tlog_confirm", "g4.tashkeel_cause", f"g5.{SUBSTITUTION}"]
#: ``render`` over round 2's committed records at HEAD before #255 (the hand-edited readout doc
#: is not an oracle: it carries a plain-language section the renderer does not write).
ROUND2_FULL_RENDER_SHA256 = "a72ed24f9c92a4361d0aee89161378b9fe5a5c81e8a36c1757f311f87091dadd"
ROUND2_PRECLAIM_RENDER_SHA256 = "0dc33b1a5f5f397e444af578ab3cf9bd526b6235dca5e3c332945472061e1fb8"
#: The pre-registration's G4 disclosure, one rendered line each (docs/consonant-round3-
#: preregistration.md, "What G4 at +3.5 does and does not protect (required disclosure)").
DISCLOSURE = (
    "What G4 at +3.5 does and does not protect (required disclosure).",
    "- a candidate with **no true change passes about 91%** of the time;",
    "- a candidate with a **true +1.0 regression passes about two thirds of the time** (0.65);",
    "- a candidate with a true +2.0 regression passes about 29% of the time.",
)


@pytest.fixture
def cv():
    from training import consonant2_verdict

    return consonant2_verdict


@pytest.fixture(autouse=True)
def reproduction_check_passes(monkeypatch):
    """The verdict runs #95's reproduction check first (#249); its own tests cover it."""
    monkeypatch.setattr(ar, "reproduction", lambda *args, **kwargs: ar.REPRODUCTION)
    monkeypatch.setattr(ar, "reproduction_inputs", lambda *args, **kwargs: [], raising=False)


class PreclaimSources(ar.Sources):
    def input_files(self, phase, weights):
        return []

    def harness_identity(self):
        return {"muraja_commit": asp.MURAJA_COMMIT, "binary_sha256": asp.HARNESS_SHA256}


class Round3Sources(ConfirmSources):
    """``test_consonant2_verdict.ConfirmSources`` over round 3's tranches and decode names; counts
    the grades it is asked for, and leaves out the rows ``excluded`` names per tranche."""

    def __init__(self, spec, excluded=None):
        super().__init__(spec)
        self.tranches = {role: [_item(n) for n in numbers]
                         for role, numbers in TRANCHE_ITEMS.items()}
        self.excluded = excluded or {}
        self.grade_calls = 0

    @staticmethod
    def text(model, n):
        return ConfirmSources.text("candidate" if model == "r3_candidate_ema" else model, n)

    def tlog_views(self, role, models):
        data = self.tlog(role, models)
        data = dataclasses.replace(data, excluded=dict(self.excluded.get(role, {})))
        return data, data

    def grade(self, items, decodes, models):
        self.grade_calls += 1
        return super().grade(items, decodes, models)


def _preclaimed(tmp_path, monkeypatch, cv, *, real=10, report=None) -> rv.Round:
    """A round-3 repository whose pre-claim ran (and is committed when it passed)."""
    round_ = r3.committed_round(tmp_path)
    rv.plant_audit(round_, real, 100 - real)
    monkeypatch.setattr(cv, "preclaim_report",
                        lambda sources, spec, choice, workers=1: report or r3.clean_report())
    cv.run_preclaim(PreclaimSources(), round_.spec, round_.layout)
    return round_


def _opened(tmp_path, monkeypatch, cv, *, claims=TRANCHES, committed=TRANCHES) -> rv.Round:
    """A round whose pre-claim passed in HEAD, whose claims are written (and committed when
    named in ``committed``) and whose freeze names one substitution."""
    round_ = _preclaimed(tmp_path, monkeypatch, cv)
    round_.commit(round_.preclaim, message="the pre-claim")
    for role in claims:
        r3.plant_claim(round_, role, commit=role in committed)
    freeze = round_.layout.root / "freeze.json"
    freeze.write_text(json.dumps({"substitutions": [
        {"cause": "other_consonant", "substitution": SUBSTITUTION, "words": 9}]}),
        encoding="utf-8")
    round_.commit(freeze, message="the freeze")
    return round_


def _verdict(cv, round_, sources=None) -> dict:
    sources = sources or Round3Sources(round_.spec)
    return cv.run_verdict(sources, round_.spec, round_.layout, tranches_dir=round_.tranches)


def _doc(round_) -> str:
    return round_.spec.doc_path.read_text(encoding="utf-8")


# --- the bars read round 3's own numbers ----------------------------------------------------------


def test_g4_reads_the_round_3_margin_from_the_spec(cv):
    margin = ROUND3.bars["tashkeel_upper_at_most"]
    assert margin == 3.5

    def g4(upper):
        report = _pooled()
        _comparison(report, "main", "tashkeel_cause").update(interval=_interval(-0.2, upper))
        return {c["name"]: c for c in cv.pooled_conditions(report, ROUND3)}["g4.tashkeel_cause"]

    assert (g4(3.5)["verdict"], g4(3.5)["threshold"]) == (PASS, margin)
    assert g4(3.51)["verdict"] == FAIL


def test_the_round_3_bar_texts_name_its_tranches_and_its_g4_margin(cv):
    conditions = {c["name"]: c for c in cv.pooled_conditions(_pooled(), ROUND3)}
    for name in BARS[:6]:
        assert "confirm.1" not in conditions[name]["bar"], name
    for name in ("bar1.all_cause", "bar3.consonant_cause", "g4.tashkeel_cause"):
        assert "confirm.3+4" in conditions[name]["bar"], name
    g4 = conditions["g4.tashkeel_cause"]["bar"]
    assert "+3.5" in g4 and "+1.0" not in g4


def test_round_2_s_bar_texts_are_unchanged(cv):
    committed = json.loads((asp.CONSONANT2.root / "report.json").read_text(encoding="utf-8"))
    stored = {c["name"]: c["bar"] for c in committed["verdict"]["conditions"]}
    live = {c["name"]: c["bar"] for c in cv.pooled_conditions(_pooled(), asp.CONSONANT2)}
    assert {name: live[name] for name in BARS[:6]} == {name: stored[name] for name in BARS[:6]}


# --- round 2's render is byte-identical -------------------------------------------------------------


def test_round_2_s_committed_records_render_byte_identically(cv):
    root = asp.CONSONANT2.root
    preclaim = json.loads((root / "preclaim.json").read_text(encoding="utf-8"))
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))

    assert hashlib.sha256(cv.render(preclaim, report).encode("utf-8")).hexdigest() == \
        ROUND2_FULL_RENDER_SHA256
    assert hashlib.sha256(cv.render(preclaim, None).encode("utf-8")).hexdigest() == \
        ROUND2_PRECLAIM_RENDER_SHA256


# --- the no-control pre-claim -------------------------------------------------------------------------


def test_the_preclaim_report_of_a_round_without_a_control_has_no_control_key(monkeypatch):
    from training import consonant2_verdict

    monkeypatch.setattr(ar, "absent_sets", lambda *args: (
        ar.QDAT_SET, ar.QUD_OTHER_SET, ar.TADABUR_SET))  # every non-tlog set: not committed
    monkeypatch.setattr(cr, "g6_scope", lambda: {"sites": []})
    monkeypatch.setattr(cr, "catch_arm", lambda *args, **kwargs: None)

    report = consonant2_verdict.preclaim_report(ar.Sources(ROUND3), ROUND3, {"weights": "ema"})

    assert "control" not in report
    assert report["guards"]["qud_other"] is None  # nothing was read, and nothing raised


def test_a_passing_preclaim_of_a_round_without_a_control_carries_no_control_key(
        cv, tmp_path, monkeypatch):
    round_ = _preclaimed(tmp_path, monkeypatch, cv)
    record = json.loads(round_.preclaim.read_text(encoding="utf-8"))

    assert record["verdict"] == PASS
    assert "control" not in record
    assert set(record["gates"]) == set(cv.PRECLAIM_GATES)  # the file is written sorted


def test_an_ended_round_without_a_control_writes_no_control_anywhere(cv, tmp_path, monkeypatch):
    report = r3.clean_report()
    _comparison(report, "guards", "qud_other").update(interval=_interval(-0.2, 1.5))
    round_ = _preclaimed(tmp_path, monkeypatch, cv, report=report)

    assert json.loads(round_.preclaim.read_text(encoding="utf-8"))["verdict"] == FAIL
    final = json.loads(round_.layout.verdict.read_text(encoding="utf-8"))
    assert final["verdict"] == FAIL and "control" not in final
    assert "control" not in json.loads(round_.preclaim.read_text(encoding="utf-8"))
    doc = _doc(round_)
    assert "control" not in doc.lower()
    assert "confirm.3" in doc and "confirm.4" in doc
    assert "confirm.1" not in doc and "confirm.2" not in doc


# --- the verdict over confirm.3 + confirm.4 -----------------------------------------------------------


def test_a_candidate_that_clears_every_bar_passes_round_3_without_a_control(
        cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    sources = Round3Sources(round_.spec)
    report = _verdict(cv, round_, sources)

    conditions = {c["name"]: c for c in report["verdict"]["conditions"]}
    assert list(conditions) == BARS
    assert {c["verdict"] for c in conditions.values()} == {PASS}, conditions
    assert report["verdict"]["verdict"] == PASS and report["verdict"]["bars"] == dict(
        ROUND3.bars)
    assert conditions["g4.tashkeel_cause"]["threshold"] == 3.5
    assert sources.read == list(TRANCHES)  # in claim order, and nothing else
    written = json.loads(round_.layout.verdict.read_text(encoding="utf-8"))
    assert written["verdict"] == PASS and written["statement"] == STATEMENT
    for record in (report, written, json.loads(round_.layout.report.read_text(encoding="utf-8"))):
        assert "control" not in record and "candidate_vs_control" not in record


def test_the_two_round_3_tranches_are_one_pooled_set_clustered_by_surah(
        cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_)

    pooled = report["main"]["all_cause"]
    assert (pooled["observations"], pooled["clusters"]) == (80, 15)
    assert set(pooled["models"]) == {"candidate", "c95", "base"}
    assert pooled["models"]["c95"]["num"] == _flagged("c95", ALL)
    assert set(report["tranches"]) == set(TRANCHES)
    for role, numbers in TRANCHE_ITEMS.items():
        table = report["tranches"][role]["all_cause"]
        assert (table["observations"], table["clusters"]) == (40, 10)
        assert table["models"]["c95"]["num"] == _flagged("c95", numbers)
    assert set(report["sizes"]) == {"pooled", *TRANCHES}
    assert set(report["coverage"]["tranches"]) == set(TRANCHES)


def test_the_round_3_doc_names_its_tranches_g4_and_quotes_the_disclosure(
        cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    _verdict(cv, round_)
    doc = _doc(round_)

    assert STATEMENT in doc
    assert "confirm.3" in doc and "confirm.4" in doc
    assert "confirm.1" not in doc and "confirm.2" not in doc
    assert "control" not in doc.lower()
    assert re.search(r"g4\.tashkeel_cause: pass .*\+3\.5", doc)
    assert "+1.0" not in doc.split("g4.tashkeel_cause", 1)[1].splitlines()[0]
    prereg = r3.REAL_PREREG.read_text(encoding="utf-8")
    for line in DISCLOSURE:
        assert line in prereg  # the pins below are the pre-registration's own words
        assert line in doc
    assert "A1" in doc


# --- the claims gate the sitting ---------------------------------------------------------------------


def _refused(cv, round_, sources):
    with pytest.raises(REFUSED):
        _verdict(cv, round_, sources)
    assert sources.read == []
    assert not round_.layout.verdict.exists() and not round_.layout.report.exists()


@pytest.mark.parametrize("claims", [(), ("confirm.3",), ("confirm.4",)])
def test_round_3_reads_neither_tranche_until_both_claims_are_committed(
        cv, tmp_path, monkeypatch, claims):
    round_ = _opened(tmp_path, monkeypatch, cv, claims=claims, committed=claims)
    _refused(cv, round_, Round3Sources(round_.spec))


def test_round_3_refuses_a_claim_written_but_not_committed(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv, committed=("confirm.3",))
    _refused(cv, round_, Round3Sources(round_.spec))


def test_round_3_reads_its_tranches_once(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    _verdict(cv, round_)
    with pytest.raises(SystemExit):
        _verdict(cv, round_)


# --- known exclusions ---------------------------------------------------------------------------------


@pytest.mark.parametrize("role", TRANCHES)
def test_an_unknown_ayah_left_out_of_a_tranche_refuses_before_any_grade(
        cv, tmp_path, monkeypatch, role):
    round_ = _opened(tmp_path, monkeypatch, cv)
    sources = Round3Sources(round_.spec, excluded={role: {"2_255_x.flac#0": "2:255"}})

    with pytest.raises(SystemExit, match="2:255"):
        _verdict(cv, round_, sources)
    assert sources.grade_calls == 0
    assert not round_.layout.report.exists() and not round_.layout.verdict.exists()


def test_the_known_exclusions_pass_and_are_reported_by_reason(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    excluded = {"confirm.3": {"106_1_a.flac#0": "106:1", "55_17_b.flac#0": "55:17"},
                "confirm.4": {"90_10_c.flac#0": "90:10", "90_5_undefined_d.flac#0": None}}
    report = _verdict(cv, round_, Round3Sources(round_.spec, excluded=excluded))

    graded = report["coverage"]["graded"]
    assert graded["excluded"] == 4
    assert graded["excluded_clips"]["by_reason"] == {"unphonetizable": 3, "names_no_ayah": 1}
    assert graded["excluded_clips"]["by_reason"] == tlog_eval.exclusion_reasons(
        {**excluded["confirm.3"], **excluded["confirm.4"]})


# --- the tranche-reading Sources and the sitting's declared inputs ------------------------------------


def test_the_sources_read_the_spec_s_tranches_only(tmp_path):
    round3, round2 = ar.Sources(ROUND3), ar.Sources(asp.CONSONANT2)
    for sources in (round3, round2):
        sources.tranches_dir = tmp_path  # no manifest and no claim: nothing can be read

    for role in ("confirm.1", "confirm.2"):  # the round-2 tranches are no part of round 3
        with pytest.raises(SystemExit, match=role):
            round3.tlog_views(role, ("base",))
    for role in TRANCHES:
        with pytest.raises(SystemExit, match=role):
            round2.tlog_views(role, ("base",))
    for role in TRANCHES:  # past the guard, the unclaimed tranche is refused by its claim
        with pytest.raises(ValueError):
            round3.tlog_views(role, ("base",))


def test_the_confirm_sitting_declares_the_spec_s_tranche_manifests(tmp_path):
    sources = ar.Sources(ROUND3)
    sources.tranches_dir = tmp_path
    files = input_files(sources, CONFIRM, "ema")

    assert tmp_path / "confirm.3.jsonl" in files and tmp_path / "confirm.4.jsonl" in files
    assert tmp_path / "confirm.1.jsonl" not in files and tmp_path / "confirm.2.jsonl" not in files


def test_the_confirm_roster_holds_the_candidate_c95_and_the_teacher_never_round_2_s_model():
    roster = sitting_roster(ROUND3, CONFIRM, "ema")

    assert set(roster["tlog.confirm"]) == {"r3_candidate_ema", asp.C95, "base"}
    for models in roster.values():
        assert asp.R2 not in models
        assert not any(model.startswith(asp.R2_ARM_PREFIX) for model in models)
        assert "control" not in models and "r3_candidate_live" not in models
