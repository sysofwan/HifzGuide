"""The round-2 freeze phase (#227): tlog look gating only, after the pre-registration is committed.

``python -m training.consonant_readout freeze --spec consonant2 --half look`` grades C95, #178
(``c178``) and the teacher on tlog look gating through ``Sources.tlog("look", models)`` and
``Sources.grade``, and writes ``freeze.json`` and its doc. The real freeze is the freeze issue's;
every source here is synthetic.

The interface these tests pin: ``consonant_readout.write_freeze(sources, half, path, doc_path,
spec=...)`` and ``run_freeze(spec, sources, half, model)`` with ``spec`` a round-2 spec; the
``--spec`` option of the ``freeze`` command; ``freeze_path(spec)``/``freeze_doc_path(spec)``
(round 2's are ``<root>/freeze.json`` and ``docs/consonant-round2-freeze.md``; round 1's the
existing constants); and these record fields: ``round``, ``half``,
``items``, ``pins.prereg_sha256``, ``inputs`` (digests of the decode files read), ``models``
(per model, as round 1's), ``substitutions`` (the G5 list), ``dal_ta_reading`` (flagged words per
model), ``power_check`` (``label``, and the paired clustered SE of c178 − c95 per 1,000 for
``all_cause`` and ``tashkeel``) and ``coverage`` (per model, ``words``/``graded``/
``graded_share``).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import aug_readout_metrics as am
from training import consonant_readout as cr
from training import round2_fixtures as fx
from training.acceptance_stats import Resample

DAL_TA = "د→ت"
TOP_TEN = ["ز→س", "ب→ت", "ج→ز", "س→ش", "ص→س", "ح→ه", "خ→ح", "ر→ل", "ط→ت", "ق→ك"]
#: (substitution, cause, flagged C95 words); the madd alif and a shaddah row are never G5's.
TALLY = (
    ("ز→س", "other_consonant", 12), ("ب→ت", "other_consonant", 11),
    ("ج→ز", "other_consonant", 10), ("س→ش", "target_pair", 9), ("ص→س", "soft_pair", 8),
    ("ح→ه", "soft_pair", 7), ("خ→ح", "other_consonant", 6), ("ر→ل", "other_consonant", 5),
    ("ط→ت", "other_consonant", 4), ("ق→ك", "other_consonant", 3),
    ("ع→ء", "other_consonant", 2), (DAL_TA, "other_consonant", 1),
    ("ا→∅", "other_consonant", 50), ("ن→∅", "shaddah", 40))
TASHKEEL_WORDS = 5
UNGRADED = 7
TEACHER_FLAGS = 3  # the teacher flags a few words too, so its cause split is not empty
PADDING = 120


def _plan(tally=TALLY) -> list:
    """C95's flag for every word, in order: the tally, the tashkeel words, then unflagged ones."""
    plan = [fx.consonant_flag(sub, cause) for sub, cause, words in tally for _ in range(words)]
    plan += [fx.TASHKEEL_FLAG] * TASHKEEL_WORDS
    plan += [None] * PADDING
    return plan + [None] * (len(plan) % 2)


def _c178(plan: list) -> list:
    """#178: C95's flags, plus three more د→ت words and 40 of a substitution C95 never has."""
    edited = list(plan)
    free = [w for w, flag in enumerate(edited) if flag is None][:-UNGRADED]
    for w in free[:3]:
        edited[w] = fx.consonant_flag(DAL_TA)
    for w in free[3:43]:
        edited[w] = fx.consonant_flag("غ→ك")
    return edited


class Run:
    """One freeze over synthetic look gating, in a repository holding the pre-registration
    (committed or merely written, pinned to its SHA-256 or to another)."""

    def __init__(self, tmp_path, plan, c178=None, *, committed=True, pinned=True):
        self.repo = fx.init_repo(tmp_path / "repo")
        write = fx.commit_prereg if committed else fx.write_prereg
        self.prereg, self.sha = write(self.repo)
        self.spec = fx.filled_spec(prereg=(self.prereg, self.sha))
        if not pinned:
            self.spec = fx.filled_spec(prereg=(self.prereg, "0" * 64))
        teacher = [fx.consonant_flag("ز→س") if w < TEACHER_FLAGS else None
                   for w in range(len(plan))]
        self.plan = {"c95": plan, "c178": c178 or plan, "base": teacher}
        self.items = [fx.tlog_item(n, 1 + n % 14) for n in range(len(plan) // 2)]
        words = len(plan)
        self.sources = fx.TlogSources(
            tmp_path / "files", self.items,
            lambda model, i, p: self.plan[model][2 * i + p],
            graded=lambda i, p: 2 * i + p < words - UNGRADED, spec=self.spec)
        self.record_path = tmp_path / "freeze.json"
        self.doc_path = tmp_path / "freeze.md"

    def write(self) -> dict:
        return cr.write_freeze(self.sources, "look", path=self.record_path,
                               doc_path=self.doc_path, spec=self.spec)


def _record(tmp_path, plan=None, *, c178_edits=True) -> tuple[Run, dict]:
    """The freeze of ``plan`` (C95's flags), #178 carrying its extra flags unless it equals C95."""
    plan = plan or _plan()
    run = Run(tmp_path, plan, _c178(plan) if c178_edits else None)
    return run, run.write()


# --- what it writes --------------------------------------------------------------------------------


def test_the_freeze_writes_its_record_and_its_doc_and_returns_the_record(tmp_path):
    run, record = _record(tmp_path)
    assert json.loads(run.record_path.read_text(encoding="utf-8")) == record
    assert record["round"] == "consonant2" and record["half"] == "look"
    assert record["items"] == len(run.items)
    assert run.sources.read == ["look"]  # tlog look gating, once; no other set, no confirm row
    assert record["pins"]["prereg_sha256"] == run.sha
    digests = ag.digests(run.sources.decode_path("tlog.look", m)
                         for m in ("c95", "c178", "base"))
    assert digests.items() <= record["inputs"].items()


def test_the_cause_split_is_c95_s_beside_c178_and_the_teacher(tmp_path):
    run, record = _record(tmp_path)
    assert set(record["models"]) == {"c95", "c178", "base"}
    c95 = record["models"]["c95"]["scorings"]["shipped_default"]
    flagged = sum(words for _, _, words in TALLY) + TASHKEEL_WORDS
    assert c95["flagged"] == flagged
    assert c95["graded"] == len(_plan()) - UNGRADED
    assert c95["causes"]["tashkeel"]["words"] == TASHKEEL_WORDS
    base = record["models"]["base"]["scorings"]["shipped_default"]
    assert base["flagged"] == TEACHER_FLAGS


def test_g5_is_c95_s_top_ten_consonant_substitutions_plus_the_dal_ta(tmp_path):
    run, record = _record(tmp_path)
    rows = record["substitutions"]
    assert [row["substitution"] for row in rows[:10]] == TOP_TEN
    assert [row["substitution"] for row in rows] == [*TOP_TEN, DAL_TA]
    assert rows[-1] == {"cause": "other_consonant", "substitution": DAL_TA, "words": 1}
    # The madd alif, a shaddah deletion and #178's own substitution never enter it.
    assert not {"ا→∅", "ن→∅", "غ→ك"} & {row["substitution"] for row in rows}


def test_the_dal_ta_is_not_added_twice_when_it_is_already_in_the_top_ten(tmp_path):
    tally = tuple((s, c, 20 if s == DAL_TA else w) for s, c, w in TALLY)
    _, record = _record(tmp_path, _plan(tally))
    subs = [row["substitution"] for row in record["substitutions"]]
    assert len(subs) == 10 and subs.count(DAL_TA) == 1 and subs[0] == DAL_TA
    assert "ق→ك" not in subs


def test_the_dal_ta_reading_is_c95_against_c178_and_changes_nothing(tmp_path):
    run, record = _record(tmp_path)
    assert record["dal_ta_reading"]["c95"]["words"] == 1
    assert record["dal_ta_reading"]["c178"]["words"] == 4
    # #178's extra flags moved neither the list nor C95's tally.
    assert [row["substitution"] for row in record["substitutions"]] == [*TOP_TEN, DAL_TA]


def test_the_power_check_is_the_paired_clustered_se_and_decides_nothing(tmp_path):
    run, record = _record(tmp_path)
    power = record["power_check"]
    assert "decides nothing" in power["label"]
    grades = run.sources.grade(run.items, {}, ("c178", "c95"))
    for key, cause in (("all_cause", None), ("tashkeel", am.TASHKEEL_CAUSE)):
        terms = am.false_flag_terms(grades, ("c178", "c95"), fx.SHIPPED, cause=cause)
        draw = Resample.by_cluster(terms.clusters)
        gap = draw.ratios(*terms.by_model["c178"]) - draw.ratios(*terms.by_model["c95"])
        assert power[key]["se"] == pytest.approx(float(np.std(gap)) * am.PER_1000, rel=1e-3)
    assert power["all_cause"]["se"] > 0


def test_the_power_check_of_a_model_equal_to_c95_is_zero(tmp_path):
    _, record = _record(tmp_path, c178_edits=False)
    assert record["power_check"]["all_cause"]["se"] == 0
    assert record["power_check"]["tashkeel"]["se"] == 0


def test_graded_word_coverage_is_measured_not_projected(tmp_path):
    run, record = _record(tmp_path)
    words = len(_plan())
    coverage = record["coverage"]["c95"]
    assert (coverage["words"], coverage["graded"]) == (words, words - UNGRADED)
    assert coverage["graded_share"] == pytest.approx((words - UNGRADED) / words)


def test_the_doc_states_the_readings_and_cites_the_pre_registration(tmp_path):
    run, record = _record(tmp_path)
    doc = run.doc_path.read_text(encoding="utf-8")
    assert run.sha in doc and "decides nothing" in doc and DAL_TA in doc
    for row in record["substitutions"]:
        assert row["substitution"] in doc


# --- what it refuses -------------------------------------------------------------------------------


def test_the_freeze_refuses_any_half_but_look_before_reading_anything(tmp_path):
    run = Run(tmp_path, _plan())
    with pytest.raises(SystemExit, match="confirm"):
        cr.write_freeze(run.sources, "confirm", path=run.record_path, doc_path=run.doc_path,
                        spec=run.spec)
    assert run.sources.read == [] and run.sources.grade_calls == 0
    assert not run.record_path.exists() and not run.doc_path.exists()


def test_the_freeze_refuses_without_a_committed_pre_registration_and_grades_nothing(tmp_path):
    run = Run(tmp_path, _plan(), committed=False)  # written, not committed
    with pytest.raises(SystemExit, match="committed"):
        run.write()
    assert run.sources.read == [] and run.sources.grade_calls == 0
    assert not run.record_path.exists() and not run.doc_path.exists()


def test_the_freeze_refuses_a_pre_registration_edited_since_its_commit(tmp_path):
    run = Run(tmp_path, _plan())
    run.prereg.write_text("edited\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="committed"):
        run.write()
    assert run.sources.grade_calls == 0


def test_the_freeze_refuses_an_unpinned_pre_registration_before_grading(tmp_path):
    run = Run(tmp_path, _plan(), pinned=False)  # committed, but the spec pins another SHA
    with pytest.raises(SystemExit, match="pin"):
        run.write()
    assert run.sources.grade_calls == 0
    assert not run.record_path.exists()


def test_the_freeze_never_moves_an_existing_freeze(tmp_path):
    run = Run(tmp_path, _plan())
    run.record_path.write_text('{"kept": 1}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="exists"):
        run.write()
    assert run.record_path.read_text(encoding="utf-8") == '{"kept": 1}\n'
    assert run.sources.grade_calls == 0 and not run.doc_path.exists()


def test_the_round_s_own_freeze_path_is_the_round_root_s():
    from training import aug_readout_spec as asp

    assert cr.freeze_path(asp.CONSONANT2) == asp.CONSONANT2.root / "freeze.json"
    assert cr.freeze_path(asp.CONSONANT1) == cr.FREEZE_PATH
    assert cr.freeze_doc_path(asp.CONSONANT2).name == "consonant-round2-freeze.md"
    assert cr.freeze_doc_path(asp.CONSONANT1) == cr.FREEZE_DOC_PATH


# --- the dry run -----------------------------------------------------------------------------------


def test_the_dry_run_summarises_c95_on_tlog_look_and_writes_nothing(tmp_path, monkeypatch,
                                                                    capsys):
    run = Run(tmp_path, _plan())
    from training import aug_readout_spec as asp

    monkeypatch.setitem(asp.SPECS, "consonant2", run.spec)
    monkeypatch.setattr(ar, "Sources", lambda spec: run.sources)
    monkeypatch.setattr(sys, "argv", ["consonant_readout", "freeze", "--spec", "consonant2",
                                      "--half", "look", "--dry-run"])
    cr.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["model"] == "c95"
    assert summary["causes"]["tashkeel"] == TASHKEEL_WORDS
    assert [row["substitution"] for row in summary["substitutions"]][:3] == TOP_TEN[:3]
    assert run.sources.read == ["look"]
    assert not run.record_path.exists() and not run.doc_path.exists()


def test_the_dry_run_grades_nothing_before_the_pre_registration_is_committed(tmp_path):
    run = Run(tmp_path, _plan(), committed=False)
    with pytest.raises(SystemExit, match="committed"):
        cr.run_freeze(run.spec, run.sources, "look")
    assert run.sources.read == [] and run.sources.grade_calls == 0
    with pytest.raises(SystemExit, match="confirm"):
        cr.run_freeze(run.spec, run.sources, "confirm")


def test_round_1_s_freeze_command_is_unchanged(monkeypatch, tmp_path, capsys):
    """No ``--spec`` still means the consonant round: its dry run reads Tarteel v1's look half."""
    from training.test_consonant_readout_pipeline import LookHalf, _freeze_grades, _item

    items = [_item(i) for i in range(3)]
    sources = LookHalf(items, _freeze_grades(items))
    monkeypatch.setattr(ar, "Sources", lambda spec: sources)
    monkeypatch.setattr(sys, "argv", ["consonant_readout", "freeze", "--half", "look",
                                      "--dry-run"])
    cr.main()
    assert json.loads(capsys.readouterr().out)["model"] == "c95"
    assert sources.read == ["look"]
