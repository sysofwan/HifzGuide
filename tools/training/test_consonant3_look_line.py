"""Round 3's report-only look line (#255, item 4): the candidate against round 2's EMA.

The pre-claim sitting (``training.consonant2_verdict.run_preclaim``) of a round that pins round 2's
candidate as ``asp.R2`` reads the **tlog look gating rows only** — never a confirm row — and
reports, per 1,000 graded words, the chosen candidate minus round 2's EMA for the all-cause,
consonant-caused and tashkeel-caused false flags. The interface these tests pin:

* the lines are ``preclaim["reported"]["look.vs_r2.all_cause"]``, ``…consonant_cause`` and
  ``…tashkeel_cause``: mappings with ``point`` (candidate − r2) and **no** ``verdict`` — there is no
  bar, so nothing here can move a gate or the pre-claim's verdict;
* the sitting reads them through ``sources.tlog("look", models)`` — ``models`` holds the candidate
  at its **chosen** weights and ``R2`` — and ``sources.grade``; the computation sits in
  ``run_preclaim`` itself, outside ``preclaim_report`` (which the tests replace, as round 2's do);
* they are read only when the candidate's, round 2's and the teacher's look decode files
  (``sources.decode_path("tlog.look", model)``) exist. Absent ones are recorded as
  ``{"status": "not read", "reason": <text>}`` for each of the three lines, nothing is read, and the
  pre-claim is written all the same: a report-only line is never a reason to refuse or to cannot-certify;
* the files are declared inputs of round 3's pre-claim sitting (``consonant2_readout.input_files``):
  the three decodes and the look manifest ``<tranches_dir>/look.jsonl``;
* ``render`` prints each line (its name and the point to three decimals), also on the doc of a round
  the pre-claim ended;
* the confirm sitting and ``run_verdict`` never read round 2's EMA or the look: no ``tlog_views``
  call names ``R2``, no confirm input is a look file, and ``report.json`` and ``verdict.json`` carry
  no such line.

Round 2 (no ``R2`` in its spec) is unchanged: no line, no look read, no look input.
"""

from __future__ import annotations

import json

import pytest

from tadabur.tarteel_split import CONFIRM
from training import aug_readout as ar
from training import aug_readout_spec as asp
from training import round2_fixtures as fx
from training import round2_verdict_fixtures as rv
from training import round3_verdict_fixtures as r3
from training.acceptance_stats import FAIL, PASS
from training.aug_readout_spec import R2, TLOG_LOOK
from training.consonant2_readout import PRECLAIM, input_files
from training.test_consonant3_verdict import PreclaimSources, Round3Sources, _opened, _verdict
from training.test_consonant_readout_verdict import _comparison, _interval

NAMES = ("look.vs_r2.all_cause", "look.vs_r2.consonant_cause", "look.vs_r2.tashkeel_cause")
ITEMS = 12
WORDS = 2 * ITEMS


@pytest.fixture
def cv():
    from training import consonant2_verdict

    return consonant2_verdict


@pytest.fixture(autouse=True)
def reproduction_check_passes(monkeypatch):
    """The verdict runs #95's reproduction check first (#249); its own tests cover it."""
    monkeypatch.setattr(ar, "reproduction", lambda *args, **kwargs: ar.REPRODUCTION)
    monkeypatch.setattr(ar, "reproduction_inputs", lambda *args, **kwargs: [], raising=False)


def _flag_counts(consonant: int, tashkeel: int):
    """Flags by clip index: a consonant flag on the second word of the first ``consonant`` clips,
    a tashkeel flag on the first word of the first ``tashkeel`` clips."""
    def flags(n: int, position: int):
        if position == 1 and n < consonant:
            return rv.FLAG
        if position == 0 and n < tashkeel:
            return fx.TASHKEEL_FLAG
        return None
    return flags


class LookSources(PreclaimSources):
    """The tlog look, synthetic: the candidate (under any of its names) and round 2's EMA flag as
    ``candidate``/``r2`` say, C95 and the teacher nothing. Records what is read."""

    def __init__(self, files, candidate, r2):
        self.files = files
        files.mkdir(parents=True, exist_ok=True)
        self.items = [fx.tlog_item(n, 1 + n % 5) for n in range(ITEMS)]
        self.flags = {"candidate": candidate, "r2": r2}
        self.reads: list[tuple[str, tuple[str, ...]]] = []

    def decode_path(self, set_name, model):
        path = self.files / f"{set_name}.{model}.jsonl.gz"
        path.write_text(f"{set_name} {model}", encoding="utf-8")
        return path

    def tlog(self, role, models):
        self.reads.append((role, tuple(models)))
        assert role == "look", f"{role} was read"
        return ar.SetData(self.items, {m: {i.key: fx.decode_line(i, fx.CLEAN) for i in self.items}
                                       for m in models})

    @staticmethod
    def _arm(model):
        if model == R2:
            return "r2"
        return "candidate" if model.startswith("r3_candidate") or model == "candidate" else None

    def grade(self, items, decodes, models):
        grades = []
        for model in models:
            flags = self.flags.get(self._arm(model))
            for item in items:
                n = int(item.audio_filename.split("_")[1])
                for position in (0, 1):
                    flag = flags(n, position) if flags else None
                    grades.append(fx.word_grade(model, item, position, flag))
        return grades


class NoLookRead(PreclaimSources):
    """A pre-claim whose sources fail the test if the look is read."""

    def tlog(self, role, models):
        raise AssertionError(f"{role} was read")


def _preclaim(tmp_path, monkeypatch, cv, sources, *, real=10, report=None, round_=None):
    round_ = round_ or r3.committed_round(tmp_path)
    rv.plant_audit(round_, real, 100 - real)
    monkeypatch.setattr(cv, "preclaim_report",
                        lambda srcs, spec, choice, workers=1: report or r3.clean_report())
    record = cv.run_preclaim(sources, round_.spec, round_.layout)
    return round_, record


def _look_sources(tmp_path, candidate=(4, 2), r2=(8, 1)):
    return LookSources(tmp_path / "look", _flag_counts(*candidate), _flag_counts(*r2))


# --- the three lines ------------------------------------------------------------------------------


def test_the_preclaim_reports_the_candidate_minus_round_2_on_the_look_gating_rows(
        cv, tmp_path, monkeypatch):
    sources = _look_sources(tmp_path)
    _, record = _preclaim(tmp_path, monkeypatch, cv, sources)

    reported = record["reported"]
    assert all(name in reported for name in NAMES)
    # candidate: 4 consonant + 2 tashkeel flags of 24 words; round 2's EMA: 8 + 1.
    expected = {"look.vs_r2.all_cause": (6 - 9) / WORDS * 1000,
                "look.vs_r2.consonant_cause": (4 - 8) / WORDS * 1000,
                "look.vs_r2.tashkeel_cause": (2 - 1) / WORDS * 1000}
    for name, point in expected.items():
        assert reported[name]["point"] == pytest.approx(point), name
        assert "verdict" not in reported[name], name


def test_the_look_is_read_once_for_the_chosen_candidate_and_round_2_only(
        cv, tmp_path, monkeypatch):
    sources = _look_sources(tmp_path)
    _preclaim(tmp_path, monkeypatch, cv, sources)

    ((role, models),) = sources.reads
    assert role == "look"
    assert {"r3_candidate_ema", R2} <= set(models)  # the chosen weights, never the other selector
    assert "r3_candidate_live" not in models
    assert not any(model.startswith(asp.R2_ARM_PREFIX) for model in models)


def test_the_look_lines_sit_beside_the_gates_and_never_move_them(cv, tmp_path, monkeypatch):
    # The candidate is far worse than round 2's EMA on the look rows: nothing gates on it.
    sources = _look_sources(tmp_path, candidate=(12, 12), r2=(0, 0))
    _, record = _preclaim(tmp_path, monkeypatch, cv, sources)

    assert record["verdict"] == PASS
    assert set(record["gates"]) == set(cv.PRECLAIM_GATES)
    assert record["reported"]["look.vs_r2.all_cause"]["point"] > 0
    assert "g3.qdat_bench.shaddah" in record["reported"]  # the existing reported line stays
    assert not any(name in record["gates"] for name in NAMES)


def test_the_doc_of_a_round_the_preclaim_ended_prints_each_line(cv, tmp_path, monkeypatch):
    sources = _look_sources(tmp_path)
    round_, record = _preclaim(tmp_path, monkeypatch, cv, sources, real=21)

    assert record["verdict"] == FAIL
    lines = round_.spec.doc_path.read_text(encoding="utf-8").splitlines()
    for name, point in zip(NAMES, (-125.0, -166.667, 41.667)):
        (line,) = [line for line in lines if name in line]
        assert f"{point:.3f}" in line


# --- the look decodes are not there ----------------------------------------------------------------


def test_absent_look_decodes_are_recorded_as_not_read_without_blocking_the_preclaim(
        cv, tmp_path, monkeypatch):
    # ``NoLookRead`` has the real ``decode_path``: round 3's candidate has no look decode here.
    _, record = _preclaim(tmp_path, monkeypatch, cv, NoLookRead())

    assert record["verdict"] == PASS
    for name in NAMES:
        assert record["reported"][name]["status"] == "not read", name
        assert record["reported"][name]["reason"], name
        assert "point" not in record["reported"][name], name


# --- the sitting declares the look files -------------------------------------------------------------


def test_round_3_s_preclaim_declares_the_three_look_decodes_and_the_look_manifest(tmp_path):
    sources = ar.Sources(asp.CONSONANT3)
    sources.tranches_dir = tmp_path
    files = input_files(sources, PRECLAIM, "ema")

    for model in (R2, "r3_candidate_ema", "base"):
        assert sources.decode_path(TLOG_LOOK, model) in files, model
    assert tmp_path / "look.jsonl" in files
    assert sources.decode_path(TLOG_LOOK, "r3_candidate_live") not in files


def test_round_2_declares_no_look_file_in_its_preclaim(tmp_path):
    sources = ar.Sources(asp.CONSONANT2)
    sources.tranches_dir = tmp_path
    files = input_files(sources, PRECLAIM, "ema")

    assert tmp_path / "look.jsonl" not in files
    assert not any(TLOG_LOOK in path.name for path in files)


def test_round_2_s_preclaim_reads_no_look_and_reports_no_look_line(cv, tmp_path, monkeypatch):
    round_ = rv.committed_round(tmp_path)
    rv.plant_audit(round_, 10, 90)
    monkeypatch.setattr(cv, "preclaim_report",
                        lambda sources, spec, choice, workers=1: rv.clean_report())
    record = cv.run_preclaim(NoLookRead(), round_.spec, round_.layout)

    assert set(record["reported"]) == {"g3.qdat_bench.shaddah"}
    assert "vs_r2" not in round_.preclaim.read_text(encoding="utf-8")


# --- never on confirm --------------------------------------------------------------------------------


class SpiedSources(Round3Sources):
    def __init__(self, spec):
        super().__init__(spec)
        self.asked: list[tuple[str, tuple[str, ...]]] = []

    def tlog_views(self, role, models):
        self.asked.append((role, tuple(models)))
        return super().tlog_views(role, models)

    def tlog(self, role, models):
        self.asked.append((role, tuple(models)))
        return super().tlog(role, models)


def test_the_verdict_reads_neither_round_2_s_ema_nor_the_look(cv, tmp_path, monkeypatch):
    round_ = _opened(tmp_path, monkeypatch, cv)
    sources = SpiedSources(round_.spec)
    report = _verdict(cv, round_, sources)

    assert sources.asked and {role for role, _ in sources.asked} == {"confirm.3", "confirm.4"}
    for _, models in sources.asked:
        assert R2 not in models
        assert not any(model.startswith(asp.R2_ARM_PREFIX) for model in models)
    for text in (round_.layout.report.read_text(encoding="utf-8"),
                 round_.layout.verdict.read_text(encoding="utf-8")):
        assert "vs_r2" not in text
    assert "look_vs_r2" not in report and "look" not in report.get("sizes", {})


def test_the_confirm_sitting_declares_no_look_file_and_no_round_2_decode(tmp_path):
    sources = ar.Sources(asp.CONSONANT3)
    sources.tranches_dir = tmp_path
    files = input_files(sources, CONFIRM, "ema")

    assert tmp_path / "look.jsonl" not in files
    for path in files:
        assert TLOG_LOOK not in path.name, path
        assert "r2_candidate" not in path.name, path
        assert path != sources.decode_path(TLOG_LOOK, R2)


def test_a_look_line_in_a_report_or_verdict_is_not_what_the_confirm_read_writes(
        cv, tmp_path, monkeypatch):
    # The pre-claim holds the lines; the verdict's doc shows them beside the gates via the
    # pre-claim record, and the pooled report itself has no place for them.
    sources = _look_sources(tmp_path)
    round_, record = _preclaim(tmp_path, monkeypatch, cv, sources)
    assert record["verdict"] == PASS
    round_.commit(round_.preclaim, message="the pre-claim")
    for role in ("confirm.3", "confirm.4"):
        r3.plant_claim(round_, role)
    freeze = round_.layout.root / "freeze.json"
    freeze.write_text(json.dumps({"substitutions": [
        {"cause": "other_consonant", "substitution": "س→ش", "words": 9}]}), encoding="utf-8")
    round_.commit(freeze, message="the freeze")
    sources = SpiedSources(round_.spec)
    report = _verdict(cv, round_, sources)

    doc = round_.spec.doc_path.read_text(encoding="utf-8")
    assert all(name in doc for name in NAMES)  # from the pre-claim record
    assert "vs_r2" not in json.dumps(report)
