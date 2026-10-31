"""Round 2's pooled verdict (#230): bars 1, 2, 3, 4a.tlog, 4b.tlog, G4 and G5 on confirm.1+2.

``training.consonant2_verdict`` (new; the pre-claim and claim halves are pinned beside this file)
holds the interface these tests pin:

* ``POOLED_BARS``: the gating conditions, in order: ``bar1.all_cause``, ``bar2.reduction``,
  ``bar3.consonant_cause``, ``bar4a.tlog_confirm``, ``bar4b.tlog_confirm``,
  ``g4.tashkeel_cause`` and one ``g5.<substitution>`` per frozen substitution.
* ``pooled_conditions(report, spec)``: those conditions from a report shaped as round 1's
  (``main.{all_cause,consonant_cause,tashkeel_cause}``, ``mechanism.{tlog_confirm,
  tlog_confirm_departs}``, ``substitutions``, ``g5_frozen``). Each reads the frozen number in
  ``spec.bars`` (never a literal) and the 99.5% interval for G5; a table that was not read, an
  interval with no bound or an unobserved frozen substitution cannot certify.
* ``final_verdict(preclaim, conditions)``: the decision rule over ``preclaim["verdict"]`` and the
  pooled conditions: any fail fails, else any cannot-certify cannot certify, else pass.
* ``run_verdict(sources, spec, layout=None, workers=1, tranches_dir=TRANCHES_DIR)``: refuses
  (``SystemExit`` or ``ValueError``) before reading any tranche unless ``preclaim.json`` is
  committed and unchanged in HEAD with an overall pass and confirm.1's and confirm.2's claims are
  committed, and refuses once ``verdict.json`` exists (one read). It reads ``sources.tlog``
  for ``confirm.1`` and ``confirm.2`` (gating items), pools them before resampling
  (``tlog_eval.confirm_views``: the cluster is the surah), grades with the roles ``candidate``,
  ``control``, ``c95`` and ``base`` and writes ``report.json`` and ``verdict.json``; it returns
  the report. The report carries ``statement`` (:data:`STATEMENT`), ``main`` (pooled tables),
  ``tranches[confirm.k]`` (the same tables for each tranche alone), ``bar2_companions``
  (``ten_percent_of_c95``, ``gap_closed``, ``teacher_shared_flags``; an undefined rate is
  reported as ``None``, never a crash), ``candidate_vs_control``
  (``table`` with the ``candidate−control`` comparison, and a ``caveat`` stating "n = 1"),
  ``strata`` (``gating.id_floor_ok`` as its own tables), ``coverage.reported_strata`` (status
  "not decoded": P3 decodes gating rows only) and ``verdict`` (``bars``, ``conditions``,
  ``preclaim``, ``verdict``). ``verdict.json`` carries the statement and the verdict.
* ``render(preclaim, report=None)``: the round's readout doc as text. ``report=None`` is a round
  the pre-claim ended.

Everything below uses synthetic grades; nothing runs a harness.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Mapping

import pytest

from training import aug_readout_spec as asp
from training import consonant_readout as cr
from training import round2_fixtures as fx
from training import round2_verdict_fixtures as rv
from training import tlog_eval
from training import aug_readout as ar
from training.acceptance_stats import CANNOT_CERTIFY, FAIL, NO_BOUND, PASS
from training.test_consonant_readout_verdict import VS, _comparison, _interval, _report, _view

REFUSED = (SystemExit, ValueError)
STATEMENT = "clip-level split, same users possible on both sides"
CONTROL_VS = "candidate−control"
SUBSTITUTION = "س→ش"
CANDIDATE, CONTROL = "r2_candidate_ema", "r2_control_ema"
BARS = ["bar1.all_cause", "bar2.reduction", "bar3.consonant_cause", "bar4a.tlog_confirm",
        "bar4b.tlog_confirm", "g4.tashkeel_cause", f"g5.{SUBSTITUTION}"]


@pytest.fixture
def cv():
    from training import consonant2_verdict

    return consonant2_verdict


# --- the bars, on hand-built tables ---------------------------------------------------------------


def _pooled() -> dict:
    """A report shaped as the verdict builds it, every bar clear of its number."""
    report = rv.clean_report()
    return {
        "main": report["main"],
        "mechanism": {"tlog_confirm": report["mechanism"]["tarteel_v1_confirm"],
                      "tlog_confirm_departs": report["mechanism"]["tarteel_v1_confirm_departs"]},
        "substitutions": report["substitutions"],
        "g5_frozen": [SUBSTITUTION]}


def _conditions(cv, report=None, spec=asp.CONSONANT2) -> dict[str, dict]:
    return {c["name"]: c for c in cv.pooled_conditions(report or _pooled(), spec)}


def test_the_pooled_bars_are_the_pre_registered_list_in_order(cv):
    assert list(cv.POOLED_BARS)[:6] == BARS[:6]
    conditions = _conditions(cv)
    assert list(conditions) == BARS
    assert {c["verdict"] for c in conditions.values()} == {PASS}


@pytest.mark.parametrize("edit, name", [
    (lambda r: _comparison(r, "main", "all_cause").update(
        interval=_interval(-1.0, 0.0)), "bar1.all_cause"),  # < 0 is strict
    (lambda r: r["main"]["all_cause"]["models"]["candidate"].update(point=42.72),
     "bar2.reduction"),  # a reduction of 4.69, under 4.7
    (lambda r: _comparison(r, "main", "consonant_cause").update(
        interval=_interval(-1.0, 0.0)), "bar3.consonant_cause"),
    (lambda r: _comparison(r, "mechanism", "tlog_confirm").update(
        interval=_interval(0.0, 1.0)), "bar4a.tlog_confirm"),  # > 0 is strict
    (lambda r: _comparison(r, "mechanism", "tlog_confirm_departs").update(
        interval=_interval(-0.51, 0.2)), "bar4b.tlog_confirm"),  # ≥ −0.5 pt
    (lambda r: _comparison(r, "main", "tashkeel_cause").update(
        interval=_interval(-0.2, 1.01)), "g4.tashkeel_cause"),  # ≤ +1.0
    (lambda r: r["substitutions"][SUBSTITUTION]["comparisons"][VS].update(
        veto_interval=_interval(0.01, 0.5)), f"g5.{SUBSTITUTION}"),  # 99.5% lower bound > 0
])
def test_each_bar_fails_just_past_its_frozen_number(cv, edit, name):
    report = _pooled()
    edit(report)
    conditions = _conditions(cv, report)
    assert conditions[name]["verdict"] == FAIL
    assert [n for n, c in conditions.items() if c["verdict"] != PASS] == [name]


@pytest.mark.parametrize("edit, name", [
    (lambda r: _comparison(r, "main", "all_cause").update(interval=_interval(-1.0, -0.001)),
     "bar1.all_cause"),
    (lambda r: r["main"]["all_cause"]["models"]["candidate"].update(point=42.71),
     "bar2.reduction"),  # exactly 4.7 passes, in floating point too
    (lambda r: _comparison(r, "mechanism", "tlog_confirm_departs").update(
        interval=_interval(-0.5, 0.2)), "bar4b.tlog_confirm"),
    (lambda r: _comparison(r, "main", "tashkeel_cause").update(interval=_interval(-0.2, 1.0)),
     "g4.tashkeel_cause"),
    (lambda r: r["substitutions"][SUBSTITUTION]["comparisons"][VS].update(
        veto_interval=_interval(0.0, 0.5)), f"g5.{SUBSTITUTION}"),
])
def test_each_bar_passes_exactly_at_its_number_where_the_bar_is_not_strict(cv, edit, name):
    report = _pooled()
    edit(report)
    assert _conditions(cv, report)[name]["verdict"] == PASS


def test_the_bars_read_the_spec_s_numbers_not_literals(cv):
    report = _pooled()
    report["main"]["all_cause"]["models"]["candidate"]["point"] = 42.71  # a reduction of 4.7
    assert _conditions(cv, report)["bar2.reduction"]["verdict"] == PASS
    stricter = dataclasses.replace(
        asp.CONSONANT2, bars={**asp.CONSONANT2.bars, "reduction_at_least": 10.0,
                              "tashkeel_upper_at_most": 0.1})
    conditions = _conditions(cv, report, stricter)
    assert conditions["bar2.reduction"]["verdict"] == FAIL
    assert conditions["g4.tashkeel_cause"]["verdict"] == FAIL  # UB 0.7 > 0.1


def test_g5_gates_every_frozen_substitution_and_only_those(cv):
    report = _pooled()
    other = _view({"candidate": 3.0, "c95": 1.0}, (2.0, 1.0, 3.0), veto=_interval(0.5, 3.5))
    report["substitutions"]["د→ت"] = other  # observed, and a clear rise: but not frozen
    assert _conditions(cv, report)[f"g5.{SUBSTITUTION}"]["verdict"] == PASS
    assert "g5.د→ت" not in _conditions(cv, report)
    report["g5_frozen"] = [SUBSTITUTION, "د→ت"]
    conditions = _conditions(cv, report)
    assert conditions["g5.د→ت"]["verdict"] == FAIL
    assert conditions[f"g5.{SUBSTITUTION}"]["verdict"] == PASS


@pytest.mark.parametrize("break_it, name", [
    (lambda r: r["main"].update(all_cause=None), "bar1.all_cause"),
    (lambda r: r["main"].update(all_cause=None), "bar2.reduction"),
    (lambda r: r["main"].update(consonant_cause=None), "bar3.consonant_cause"),
    (lambda r: r["mechanism"].update(tlog_confirm=None), "bar4a.tlog_confirm"),
    (lambda r: r["mechanism"].update(tlog_confirm_departs=None), "bar4b.tlog_confirm"),
    (lambda r: r["main"].update(tashkeel_cause=None), "g4.tashkeel_cause"),
    (lambda r: r["substitutions"].clear(), f"g5.{SUBSTITUTION}"),  # a frozen one never observed
    (lambda r: r.update(g5_frozen=[]), "g5"),  # no frozen list: nothing to gate
])
def test_an_empty_cell_cannot_certify_its_bar(cv, break_it, name):
    report = _pooled()
    break_it(report)
    assert _conditions(cv, report)[name]["verdict"] == CANNOT_CERTIFY


@pytest.mark.parametrize("path, name", [
    (("main", "all_cause"), "bar1.all_cause"),
    (("main", "consonant_cause"), "bar3.consonant_cause"),
    (("mechanism", "tlog_confirm"), "bar4a.tlog_confirm"),
    (("mechanism", "tlog_confirm_departs"), "bar4b.tlog_confirm"),
    (("main", "tashkeel_cause"), "g4.tashkeel_cause"),
])
def test_an_interval_with_no_bound_cannot_certify_its_bar(cv, path, name):
    report = _pooled()
    _comparison(report, *path).update(
        interval=_interval(None, None, method=NO_BOUND, reason="an empty cell"))
    assert _conditions(cv, report)[name]["verdict"] == CANNOT_CERTIFY


def test_an_undefined_rate_cannot_certify_bar_2(cv):
    report = _pooled()
    report["main"]["all_cause"]["models"]["c95"]["point"] = None
    assert _conditions(cv, report)["bar2.reduction"]["verdict"] == CANNOT_CERTIFY


@pytest.mark.parametrize("preclaim, pooled, expected", [
    (PASS, [PASS, PASS], PASS),
    (PASS, [PASS, FAIL, CANNOT_CERTIFY], FAIL),
    (PASS, [PASS, CANNOT_CERTIFY], CANNOT_CERTIFY),
    (FAIL, [PASS, PASS], FAIL),
    (CANNOT_CERTIFY, [PASS, PASS], CANNOT_CERTIFY),
    (CANNOT_CERTIFY, [FAIL], FAIL),
])
def test_the_final_verdict_is_the_decision_rule_over_both_halves(cv, preclaim, pooled, expected):
    conditions = [{"name": f"c{n}", "verdict": v} for n, v in enumerate(pooled)]
    assert cv.final_verdict({"verdict": preclaim}, conditions) == expected


# --- the phase: end to end on synthetic tlog grades ------------------------------------------------

#: Tranche 1 holds surahs 1-10 and tranche 2 surahs 6-15: 15 clusters pooled, 10 each alone (the
#: interval rules need at least 10 clusters and replicates that differ, so the flags vary by clip).
TRANCHE_ITEMS = {"confirm.1": range(0, 20), "confirm.2": range(20, 40)}


def _item(n: int):
    surah = 1 + n % 10 if n < 20 else 6 + n % 10
    item = fx.tlog_item(n, surah)
    return dataclasses.replace(item, labels={"assigned_label": "2:1", "id_floor_ok": n % 2 == 0})


ROLE = {CANDIDATE: "candidate", CONTROL: "control"}


def _best(model, n, position):
    """C95 flags a consonant at every clip but each third and a tashkeel on each seventh; the
    control a consonant at the even ones of those; the teacher one word; the candidate none.
    ``model`` is a decode name or, once the verdict has bound the roles, the role."""
    role = ROLE.get(model, model)
    if role == "c95" and position == 1 and n % 3:
        return rv.FLAG
    if role == "c95" and position == 0 and n % 7 == 0:
        return fx.TASHKEEL_FLAG
    if role == "control" and position == 1 and n % 3 and n % 2 == 0:
        return rv.FLAG
    if role == "base" and position == 1 and n == 1:
        return rv.FLAG
    return None


def _worse(model, n, position):
    """The candidate flags a consonant at every clip but each third; the others none."""
    return rv.FLAG if position == 1 and n % 3 and ROLE.get(model, model) == "candidate" else None


def _flagged(role: str, numbers, flags=_best, cause=None) -> int:
    """How many words ``role`` flags over the clips ``numbers`` (optionally of one cause)."""
    flags_ = [flags(role, n, p) for n in numbers for p in (0, 1)]
    return sum(1 for flag in flags_
               if flag is not None and (cause is None or cause in flag.causes))


ALL = range(40)


def _number(item) -> int:
    return int(item.audio_filename.split("_")[1])  # ``<surah>_<n>_<n>.flac``


class ConfirmSources(ar.Sources):
    """The confirm tranches, synthetic: only ``tlog('confirm.1'|'confirm.2')`` (and the Tarteel v1
    confirm half, reported) can be read; every pre-claim set fails the test."""

    harness_record = fx.TlogSources.harness_record

    @staticmethod
    def text(model: str, n: int) -> str:
        """What a model hears: the teacher and the candidate the swapped reading, the control the
        clean one, C95 the swapped one on each fourth clip."""
        role = ROLE.get(model, model)
        return fx.SWAPPED if role in ("base", "candidate") or (role == "c95" and n % 4 == 0) \
            else fx.CLEAN

    def __init__(self, spec, flags=_best, graded=lambda n, p: True):
        self.spec, self.flags, self.graded = spec, flags, graded
        self.tranches = {role: [_item(n) for n in numbers]
                         for role, numbers in TRANCHE_ITEMS.items()}
        self.read: list[str] = []

    def input_files(self, phase, weights):
        return []

    def harness_identity(self):
        return dict(self.harness_record)

    def tlog(self, role, models):
        assert role in self.tranches, f"{role} was read"
        self.read.append(role)
        return ar.SetData(self.tranches[role], {
            m: {i.key: fx.decode_line(i, self.text(m, _number(i))) for i in self.tranches[role]}
            for m in models})

    def tlog_views(self, role, models):
        # The fixture's decode lines name no span, so the trimmed read is the whole ayah.
        data = self.tlog(role, models)
        return data, data

    def tlog_span_trimmed(self, role, models):
        return self.tlog(role, models)

    def tadabur(self, models):
        from training.nonpro_gap import DevClip

        clips = [DevClip(f"dev_{n:03d}.flac", 30 + n % 3, (n, 0), 1.0,
                         fx.CLEAN if n % 2 else fx.SWAPPED) for n in range(6)]
        decodes = {m: {c.filename: self.text(m, 900 + n) for n, c in enumerate(clips)}
                   for m in models}
        blanks = {m: {c.filename: (1, 100) for c in clips} for m in models}
        return ar.TadaburData(clips, decodes, blanks, {})

    def tarteel(self, which_half, models):
        items = [fx.tlog_item(100 + n, 20 + n) for n in range(4)]
        return ar.SetData(items, {m: {i.key: fx.decode_line(i, self.text(m, _number(i)))
                                      for i in items} for m in models})

    def grade(self, items, decodes, models):
        grades = []
        for model in models:
            for item in items:
                n = _number(item)
                for position in (0, 1):
                    graded = self.graded(n, position)
                    flag = self.flags(model, n, position) if graded else None
                    grades.append(fx.word_grade(model, item, position, flag, graded))
        return grades

    def _refused(self, *args, **kwargs):
        raise AssertionError("a pre-claim set was read in the verdict")

    qdat = qud_other = qud = retasy = bench = _refused


class Sources(ar.Sources):
    """Pre-claim inputs, as ``run_preclaim`` asks for them."""

    def input_files(self, phase, weights):
        return []

    def harness_identity(self):
        return {"muraja_commit": asp.MURAJA_COMMIT, "binary_sha256": asp.HARNESS_SHA256}


def _claim(round_: rv.Round, role: str, *, commit: bool = True):
    path = round_.tranches / f"{role}.claim.json"
    path.write_text(json.dumps({"role": role, "issue": 227, "commit": rv.CLAIM_COMMIT,
                                "date": "2026-11-02"}), encoding="utf-8")
    if commit:
        round_.commit(path, message=f"claim {role}")


def _opened(tmp_path, monkeypatch, cv, *, real=10, claims=("confirm.1", "confirm.2"),
            commit_preclaim=True) -> rv.Round:
    """A round whose pre-claim passed and was committed, whose claims are committed and whose
    freeze names one substitution."""
    round_ = rv.committed_round(tmp_path)
    rv.plant_audit(round_, real, 100 - real)
    monkeypatch.setattr(cv, "preclaim_report",
                        lambda sources, spec, choice, workers=1: rv.clean_report())
    cv.run_preclaim(Sources(), round_.spec, round_.layout)
    if commit_preclaim:
        round_.commit(round_.preclaim, message="the pre-claim")
    for role in claims:
        _claim(round_, role)
    (round_.layout.root / "freeze.json").write_text(json.dumps(
        {"substitutions": [{"cause": "other_consonant", "substitution": SUBSTITUTION,
                            "words": 9}]}), encoding="utf-8")
    round_.commit(round_.layout.root / "freeze.json", message="the freeze")
    return round_


def _verdict(cv, round_, sources=None, **kwargs) -> dict:
    sources = sources or ConfirmSources(round_.spec)
    return cv.run_verdict(sources, round_.spec, round_.layout, tranches_dir=round_.tranches,
                          **kwargs)


def _by_name(report: dict) -> dict[str, dict]:
    return {c["name"]: c for c in report["verdict"]["conditions"]}


def test_a_candidate_that_clears_every_bar_passes_the_round(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_)

    conditions = _by_name(report)
    assert list(conditions) == BARS
    assert {c["verdict"] for c in conditions.values()} == {PASS}, conditions
    assert report["verdict"]["preclaim"] == PASS and report["verdict"]["verdict"] == PASS
    assert report["verdict"]["bars"] == dict(round_.spec.bars)
    assert json.loads(round_.layout.report.read_text(encoding="utf-8"))["verdict"] == \
        report["verdict"]
    written = json.loads(round_.layout.verdict.read_text(encoding="utf-8"))
    assert written["verdict"] == PASS and written["statement"] == STATEMENT


def test_the_two_tranches_are_one_pooled_set_clustered_by_surah(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    sources = ConfirmSources(round_.spec)
    report = _verdict(cv, round_, sources)

    pooled = report["main"]["all_cause"]
    assert (pooled["observations"], pooled["clusters"]) == (80, 15)  # surahs 6-10 are in both
    assert pooled["models"]["c95"]["num"] == _flagged("c95", ALL)
    assert pooled["models"]["c95"]["den"] == 80
    assert pooled["models"]["candidate"]["num"] == 0
    alone = report["tranches"]
    assert set(alone) == {"confirm.1", "confirm.2"}
    for role, numbers in TRANCHE_ITEMS.items():
        table = alone[role]["all_cause"]
        assert (table["observations"], table["clusters"]) == (40, 10)
        assert table["models"]["c95"]["num"] == _flagged("c95", numbers)
    assert sorted(sources.read) == ["confirm.1", "confirm.2"]


def test_the_other_arm_and_the_teacher_sit_beside_the_candidate(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_)

    models = report["main"]["all_cause"]["models"]
    assert set(models) >= {"candidate", "control", "c95", "base"}
    assert models["control"]["num"] == _flagged("control", ALL)
    assert models["base"]["num"] == _flagged("base", ALL)
    assert VS in report["main"]["all_cause"]["comparisons"]  # the bars read candidate − c95 only
    versus = report["candidate_vs_control"]
    assert CONTROL_VS in versus["table"]["comparisons"]
    assert "n = 1" in versus["caveat"]
    assert CONTROL_VS not in {c.get("comparison") for c in report["verdict"]["conditions"]}


def test_the_bar_2_companions_are_reported_not_gated(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_)

    rate = {role: 1000 * _flagged(role, ALL) / 80 for role in ("c95", "base", "candidate")}
    companions = report["bar2_companions"]
    assert companions["ten_percent_of_c95"] == pytest.approx(0.1 * rate["c95"])
    assert companions["gap_closed"] == pytest.approx(
        (rate["c95"] - rate["candidate"]) / (rate["c95"] - rate["base"]))
    assert companions["teacher_shared_flags"] == {"candidate": 0, "c95": 1}
    assert not [name for name in _by_name(report) if "companion" in name]


def test_the_id_floor_set_is_reported_alone_and_the_strata_are_not_decoded(cv, tmp_path,
                                                                           monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_)

    id_floor = report["strata"][tlog_eval.ID_FLOOR_OK]["all_cause"]
    assert id_floor["observations"] == 40  # the 20 even-numbered clips, two words each
    assert report["coverage"]["reported_strata"]["status"] == "not decoded"
    assert report["main"]["all_cause"]["observations"] == 80  # never pooled into gating
    assert "span-trimmed" not in report["coverage"]["reported_strata"]["reason"]


def test_the_reported_lines_are_built_beside_the_bars(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_)

    tarteel = report["tarteel_confirm"]
    assert set(tarteel) == {"tables", "agreement", "departs", "equals"}
    assert set(tarteel["tables"]) == set(report["main"])
    assert tarteel["tables"]["all_cause"]["observations"] == 8
    # Round 1's bar-4 readings: one departure and one equal word per item, and the candidate
    # hears the teacher's own reading, so it agrees on the whole reference (8 words of 8 words
    # for the agreement reading, 4 + 4 across the two subsets) while the control agrees on the
    # equal word only.
    words = {name: tarteel[name]["models"] for name in ("agreement", "departs", "equals")}
    assert words["agreement"]["candidate"]["den"] == 8
    assert (words["departs"]["candidate"]["den"], words["equals"]["candidate"]["den"]) == (4, 4)
    assert words["agreement"]["candidate"]["num"] == 8
    assert words["departs"]["candidate"]["num"] == 4
    assert words["departs"]["c95"]["num"] == 1  # C95 hears the teacher's reading on one clip
    assert words["equals"]["c95"]["num"] == 4
    assert report["sensitivity"]["span_trimmed"]["all_cause"]["observations"] == 80
    assert set(report["coverage"]["tranches"]) == {"confirm.1", "confirm.2"}
    tranche = report["coverage"]["tranches"]["confirm.1"]
    assert (tranche["clips"], tranche["partial_span"], tranche["excluded"]) == (20, 0, 0)
    assert tranche["empty_slots"]["observations"] == 20
    assert {row["reciter_id"] for row in report["dose_response"]["reciters"]} == {30, 31, 32}
    assert report["dose_response"]["reciters"][0]["clips"] == 2
    assert "shaddah_cause" in report["main"]
    assert "shaddah_cause" in report["tranches"]["confirm.1"]
    assert "tlog_confirm_shaddah" in report["mechanism"]
    gated = {c["name"] for c in report["verdict"]["conditions"]}
    assert not [name for name in gated if "shaddah" in name or "tarteel" in name
                or "dose" in name]


def test_every_verdict_carries_the_clip_level_split_statement(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_)
    assert report["statement"] == STATEMENT == cv.STATEMENT
    assert json.loads(round_.layout.verdict.read_text(encoding="utf-8"))["statement"] == STATEMENT


def test_a_candidate_that_flags_more_than_c95_fails_the_round(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_, ConfirmSources(round_.spec, flags=_worse))

    conditions = _by_name(report)
    assert conditions["bar1.all_cause"]["verdict"] == FAIL
    assert conditions["bar2.reduction"]["verdict"] == FAIL
    assert report["verdict"]["verdict"] == FAIL
    assert json.loads(round_.layout.verdict.read_text(encoding="utf-8"))["verdict"] == FAIL


def test_a_round_with_no_graded_word_cannot_certify(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_, ConfirmSources(round_.spec, graded=lambda n, p: False))

    conditions = _by_name(report)
    for name in ("bar1.all_cause", "bar2.reduction", "bar3.consonant_cause"):
        assert conditions[name]["verdict"] == CANNOT_CERTIFY
    assert report["verdict"]["verdict"] == CANNOT_CERTIFY


# --- what the phase refuses ------------------------------------------------------------------------


def _refused(cv, round_, sources):
    with pytest.raises(REFUSED):
        _verdict(cv, round_, sources)
    assert sources.read == []
    assert not round_.layout.verdict.exists() and not round_.layout.report.exists()


def test_the_verdict_refuses_without_a_committed_preclaim(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv, commit_preclaim=False)
    _refused(cv, round_, ConfirmSources(round_.spec))


def test_the_verdict_refuses_a_preclaim_that_did_not_pass(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    record = json.loads(round_.preclaim.read_text(encoding="utf-8"))
    round_.write_commit(round_.preclaim, json.dumps({**record, "verdict": FAIL}), "a fail")
    _refused(cv, round_, ConfirmSources(round_.spec))


@pytest.mark.parametrize("claims", [(), ("confirm.1",)])
def test_the_verdict_refuses_while_a_claim_is_not_committed(cv, tmp_path, monkeypatch, claims):
    round_ = _opened(tmp_path, monkeypatch, cv, claims=claims)
    _refused(cv, round_, ConfirmSources(round_.spec))


def test_the_verdict_refuses_a_claim_that_is_written_but_not_committed(cv, tmp_path,
                                                                       monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv, claims=("confirm.1",))
    _claim(round_, "confirm.2", commit=False)
    _refused(cv, round_, ConfirmSources(round_.spec))


def test_the_tranches_are_read_once(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    _verdict(cv, round_)
    with pytest.raises(SystemExit):
        _verdict(cv, round_, ConfirmSources(round_.spec))


def test_the_verdict_never_reads_a_pre_claim_set_or_a_third_tranche(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    sources = ConfirmSources(round_.spec)
    _verdict(cv, round_, sources)  # qdat, QUD, Tadabur and bench reads would raise in the fake
    assert set(sources.read) == {"confirm.1", "confirm.2"}


# --- the doc and the command ------------------------------------------------------------------------


def test_an_unchosen_selector_or_checkpoint_refuses_the_pooled_read(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    records = json.loads(round_.layout.models.read_text(encoding="utf-8"))
    records["tlog.confirm/r2_candidate_ema"] = fx.model_record(round_.spec,
                                                               "r2_candidate_live")
    round_.layout.models.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(REFUSED):
        _verdict(cv, round_)

    record = fx.model_record(round_.spec, "r2_candidate_ema")
    record["identity"] = {**record["identity"], "checkpoint_sha256": "0" * 64}
    records["tlog.confirm/r2_candidate_ema"] = record
    round_.layout.models.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(REFUSED):
        _verdict(cv, round_)
    assert not round_.layout.verdict.exists()


def test_a_freeze_edited_after_its_commit_refuses_the_pooled_read(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    (round_.layout.root / "freeze.json").write_text(
        json.dumps({"substitutions": []}), encoding="utf-8")
    with pytest.raises(REFUSED, match="not committed"):
        _verdict(cv, round_)
    assert not round_.layout.verdict.exists()


def _preserved_confirm_teacher(round_, identities: Mapping) -> dict:
    """The pooled merge's preserved identities beside the confirm decodes (``meta.confirm.json``
    as ``tadabur.tlog_eval_decode.merge_confirm_decodes`` writes them)."""
    path = round_.spec.root / "decodes" / "meta.confirm.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"identities": identities}), encoding="utf-8")
    return identities


def test_the_preserved_confirm_teacher_is_the_identity_the_sitting_validates(
        cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    teacher = fx.confirm_manifest(round_.spec)["tlog.confirm/base"]
    _preserved_confirm_teacher(round_, {"tlog.confirm/base": teacher})

    report = _verdict(cv, round_, ConfirmSources(round_.spec))
    assert report["verdict"]["verdict"] == PASS
    sitting = json.loads((round_.layout.root / "confirm.claim.json").read_text(encoding="utf-8"))
    assert sitting["identities"]["base"] == teacher["identity"]


@pytest.mark.parametrize("edit", [
    lambda record: record.update(fingerprints={"not": "the frozen protocol"}),
    lambda record: record["identity"].update(checkpoint_sha256="0" * 64),
])
def test_a_confirm_teacher_from_other_weights_refuses_the_pooled_read(
        cv, tmp_path, monkeypatch, edit):
    """The teacher's pooled confirm decodes define bars 4a/4b and the departure population, so
    a preserved identity that is not the frozen baseline under the frozen protocol refuses
    before any tranche is read."""
    round_ = _opened(tmp_path, monkeypatch, cv)
    teacher = fx.confirm_manifest(round_.spec)["tlog.confirm/base"]
    edit(teacher)
    _preserved_confirm_teacher(round_, {"tlog.confirm/base": teacher})

    sources = ConfirmSources(round_.spec)
    with pytest.raises(REFUSED):
        _verdict(cv, round_, sources)
    assert sources.read == []
    assert not round_.layout.verdict.exists()


class MissingInput(ConfirmSources):
    """A round whose declared confirm decode is absent: the sitting must not read it."""

    def input_files(self, phase, weights):
        return [self.decode_path("tlog.confirm", CANDIDATE)]


class MissingTeacherDecode(MissingInput):
    """A round whose declared **teacher** decode of the tranches is absent: the reader always
    reads it, and ``needs`` names only the students."""

    def input_files(self, phase, weights):
        return [self.decode_path("tlog.confirm", "base")]


class MissingCore(MissingInput):
    """A round whose declared core metadata (the Tadabur dev grid) is absent."""

    def input_files(self, phase, weights):
        from training.nonpro_gap import GRID_PATH

        return [self.spec.root / GRID_PATH.name]


class EmptyTranches(ConfirmSources):
    """Both claimable tranches hold no gating item."""

    def __init__(self, spec, **kwargs):
        super().__init__(spec, **kwargs)
        self.tranches = {"confirm.1": [], "confirm.2": []}


class NoFloorItems(ConfirmSources):
    """Every gating item is outside the id-floor reported set, which stays empty."""

    def __init__(self, spec, **kwargs):
        super().__init__(spec, **kwargs)
        self.tranches = {
            role: [dataclasses.replace(item, labels={"assigned_label": "2:1",
                                                     "id_floor_ok": False})
                   for item in items]
            for role, items in self.tranches.items()}


def test_a_declared_input_that_is_absent_cannot_certify_without_reading(cv, tmp_path,
                                                                        monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    sources = MissingInput(round_.spec)
    report = _verdict(cv, round_, sources)

    assert report["verdict"]["verdict"] == CANNOT_CERTIFY
    assert "tlog.confirm" in report["unreadable"]["reason"]
    assert sources.read == []
    assert json.loads(round_.layout.verdict.read_text(encoding="utf-8"))["verdict"] == \
        CANNOT_CERTIFY
    assert "Nothing was read" in round_.spec.doc_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("sources, named", [
    (MissingTeacherDecode, "tlog.confirm.base"),
    (MissingCore, "grid.json"),
])
def test_a_missing_teacher_or_core_input_cannot_certify_without_reading(
        cv, tmp_path, monkeypatch, sources, named):
    round_ = _opened(tmp_path, monkeypatch, cv)
    declaring = sources(round_.spec)
    report = _verdict(cv, round_, declaring)

    assert report["verdict"]["verdict"] == CANNOT_CERTIFY
    assert named in report["unreadable"]["reason"]
    assert declaring.read == []
    written = json.loads(round_.layout.verdict.read_text(encoding="utf-8"))
    assert written["verdict"] == CANNOT_CERTIFY
    assert "Nothing was read" in round_.spec.doc_path.read_text(encoding="utf-8")


def test_empty_gating_tranches_cannot_certify_without_grading(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_, EmptyTranches(round_.spec))

    assert report["verdict"]["verdict"] == CANNOT_CERTIFY
    assert "no item" in report["unreadable"]["reason"]
    assert json.loads(round_.layout.report.read_text(encoding="utf-8"))["unreadable"]


def test_an_empty_reported_stratum_does_not_block_the_gating_verdict(cv, tmp_path,
                                                                     monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_, NoFloorItems(round_.spec))

    assert report["strata"][tlog_eval.ID_FLOOR_OK] == cv.EMPTY_CELL
    assert report["verdict"]["verdict"] == PASS
    assert "reported empty" in round_.spec.doc_path.read_text(encoding="utf-8")


def test_the_doc_states_the_verdict_every_bar_and_the_statement(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_)
    preclaim = json.loads(round_.preclaim.read_text(encoding="utf-8"))

    text = cv.render(preclaim, report)
    assert STATEMENT in text
    for name in BARS:
        assert name in text
    assert "not decoded" in text
    assert "| all_cause |" in text and "[" in text  # the tables carry their intervals
    assert f"{report['main']['all_cause']['models']['c95']['point']:.3f}" in text
    assert f"{report['candidate_vs_control']['table']['models']['candidate']['point']:.3f}" \
        in text
    assert "n = 1" in text
    assert "Tarteel v1 confirm" in text and "shaddah agreement" in text
    # Tarteel v1's confirm half as round 1 reads it: bar 4's agreement, its departure subset and
    # the teacher-equals companion, each with its measured point.
    tarteel = report["tarteel_confirm"]
    assert "Tarteel v1 confirm, word agreement with the teacher" in text
    assert "Tarteel v1 confirm, agreement on the teacher-departure words" in text
    assert "Tarteel v1 confirm, agreement where the teacher equals the reference" in text
    for reading in ("agreement", "departs", "equals"):
        point = tarteel[reading]["models"]["c95"]["point"]
        assert f"{point:.3f}" in text
    assert len({tarteel[reading]["models"]["c95"]["point"]
                for reading in ("agreement", "departs", "equals")}) == 3
    assert "| 30 | 2 |" in text  # the first reciter's clips and its two points
    assert text == cv.render(preclaim, report)  # generated, reproducible


def test_the_doc_of_a_round_the_preclaim_ended_names_the_gate_and_the_unclaimed_tranches(
        cv, tmp_path, monkeypatch):
    round_ = rv.committed_round(tmp_path)
    rv.plant_audit(round_, 21, 79)
    monkeypatch.setattr(cv, "preclaim_report",
                        lambda sources, spec, choice, workers=1: rv.clean_report())
    preclaim = cv.run_preclaim(Sources(), round_.spec, round_.layout)

    text = cv.render(preclaim, None)
    assert STATEMENT in text and "catch_loss" in text
    assert "confirm.1" in text and "confirm.2" in text
    assert text == round_.spec.doc_path.read_text(encoding="utf-8")


def test_the_verdict_command_takes_the_round_2_spec_only(cv, monkeypatch):
    seen = {}

    def run(sources, spec, layout=None, workers=1, **kwargs):
        seen.update(spec=spec, sources_spec=sources.spec)
        return {"verdict": {"verdict": PASS}}

    monkeypatch.setattr(cv, "run_verdict", run)
    monkeypatch.setattr(sys, "argv", ["consonant_readout", "verdict", "--spec", "consonant2"])
    cr.main()
    assert seen["spec"] is asp.CONSONANT2 and seen["sources_spec"] is asp.CONSONANT2

    monkeypatch.setattr(sys, "argv", ["consonant_readout", "verdict", "--spec", "consonant1"])
    with pytest.raises(SystemExit) as stopped:
        cr.main()
    assert stopped.value.code == 2


def test_the_round_s_confirm_command_still_refuses_the_pooled_read_through_95_s_tables(
        monkeypatch):
    """The pooled read is ``consonant_readout verdict``; #95's ``confirm`` stays refused."""
    monkeypatch.setattr(sys, "argv", ["aug_readout", "confirm", "--spec", "consonant2"])
    with pytest.raises(SystemExit):
        ar.main()
