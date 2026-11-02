"""The round-3 readout's guards (#255): a report-only line that cannot gate, and the pins.

* The look line's files are declared (``consonant2_readout.input_files``) but **not required**
  (``Sources.reported_inputs`` / ``missing_inputs``): an absent one leaves the sitting present,
  is digested when it exists, and ``_look_lines`` proves the tranche readable (decodes present
  and the manifest committed unchanged) before reading it — otherwise every line says "not
  read" and the pre-claim is written all the same.
* Every line of the pre-registration's required disclosure the readout quotes is the committed
  file's own words; only the labelled readout note is this repo's.
* The no-control confirm roster is pinned to the candidate, the baseline and the teacher, so a
  context column planted there is refused.
* ``render`` reads the round from the record's issue, never defaulting to round 2.

Synthetic grades and git repositories; nothing runs a harness.
"""

from __future__ import annotations

import json

import pytest

from training import aug_readout as ar
from training import aug_readout_spec as asp
from training import consonant2_readout as cr
from training import round2_verdict_fixtures as rv
from training import round3_verdict_fixtures as r3
from training.acceptance_stats import PASS
from training.aug_readout_gate import input_name
from training.aug_readout_metrics import BASE
from training.consonant2_readout import PRECLAIM
from training.test_consonant3_verdict import PreclaimSources


@pytest.fixture
def cv():
    from training import consonant2_verdict

    return consonant2_verdict


class LookDeclared(PreclaimSources):
    """Round 3's pre-claim declaring the look-line files and nothing else."""

    def input_files(self, phase, weights):
        return cr.look_line_inputs(self, self.spec, weights)


class LookDecodes(PreclaimSources):
    """The look decodes exist; the look manifest never does. Reading the look fails the test."""

    def __init__(self):
        self.written = []

    def decode_path(self, set_name, model):
        path = self.files / f"{set_name}.{model}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
        self.written.append(path)
        return path

    def tlog(self, role, models):
        raise AssertionError(f"{role} was read without a committed manifest")


# --- the look line's files are declared, never required -------------------------------------------


def test_the_look_line_s_files_are_declared_but_never_required(tmp_path):
    from training.consonant2_verdict import read_inputs

    round_ = r3.committed_round(tmp_path)
    sources = LookDeclared()
    sources.spec = round_.spec
    choice = json.loads(round_.layout.choice.read_text(encoding="utf-8"))
    declared = read_inputs(sources, round_.spec, round_.layout, choice, "ema", PRECLAIM)

    assert declared.present and declared.missing_inputs == []
    files = cr.look_line_inputs(sources, round_.spec, "ema")
    assert files == cr.reported_inputs(sources, PRECLAIM, "ema")
    # Only the ones that exist are digested; none of them is required.
    assert set(declared.inputs) == {input_name(path) for path in files if path.exists()}

    files[1].parent.mkdir(parents=True, exist_ok=True)
    files[1].write_text("x", encoding="utf-8")
    declared = read_inputs(sources, round_.spec, round_.layout, choice, "ema", PRECLAIM)
    assert declared.present and input_name(files[1]) in declared.inputs


def test_round_3_s_real_preclaim_keeps_its_look_files_out_of_the_required_missing():
    sources = ar.Sources(asp.CONSONANT3)
    declared = sources.input_files(PRECLAIM, "ema")
    optional = sources.reported_inputs(PRECLAIM, "ema")

    assert optional and set(optional) <= set(declared)
    assert not set(optional) & set(sources.missing_inputs(PRECLAIM, "ema"))


@pytest.mark.parametrize("manifest", [None, "uncommitted"])
def test_a_look_tranche_that_is_not_readable_records_not_read_and_still_writes_the_preclaim(
        cv, tmp_path, monkeypatch, manifest):
    sources = LookDecodes()
    round_ = r3.committed_round(tmp_path)
    sources.files = tmp_path / "decodes"
    if manifest:
        (round_.tranches / "look.jsonl").write_text("{}\n", encoding="utf-8")  # untracked
    rv.plant_audit(round_, 10, 90)
    monkeypatch.setattr(cv, "preclaim_report", lambda *args, **kwargs: r3.clean_report())
    sources.spec, sources.tranches_dir = round_.spec, round_.tranches
    record = cv.run_preclaim(sources, round_.spec, round_.layout)

    assert record["verdict"] == PASS and round_.preclaim.exists()
    for name in cr.LOOK_LINE_NAMES.values():
        assert record["reported"][name]["status"] == "not read", name
        assert record["reported"][name]["reason"], name


# --- the disclosure is quoted, the readout note is labelled --------------------------------------


def test_every_quoted_disclosure_line_is_the_pre_registration_s_own():
    prereg = " ".join(asp.CONSONANT3_PREREG_PATH.read_text(encoding="utf-8").split())
    quoted = [line for line in asp.CONSONANT3_DISCLOSURE
              if line and not line.startswith("###")
              and not line.startswith("**Readout note")]

    assert len(quoted) == 5  # the intro, the three pass-rate bullets, the vetoes sentence
    for line in quoted:
        assert " ".join(line.split()) in prereg, line


# --- the no-control confirm roster is pinned ------------------------------------------------------


def test_a_context_column_planted_on_the_confirm_tranche_is_refused(cv):
    roster = {"tlog.confirm": ("r3_candidate_ema", asp.R2, asp.C95, BASE)}
    with pytest.raises(SystemExit, match=r"\br2\b"):
        cv.require_own_confirm_arms(roster, asp.CONSONANT3, "ema")

    own = {"tlog.confirm": ("r3_candidate_ema", asp.C95, BASE)}
    cv.require_own_confirm_arms(own, asp.CONSONANT3, "ema")
    # Round 2 decodes its control on the same tranches: the rule is the no-control round's.
    cv.require_own_confirm_arms({"tlog.confirm": ("r2_candidate_ema", "r2_control_ema", "c95")},
                                asp.CONSONANT2, "ema")


# --- render reads the round from the record -------------------------------------------------------


def test_render_reads_the_round_from_the_record_issue(cv):
    record = {"issue": asp.CONSONANT3.issue, "verdict": PASS, "gates": {}}
    text = cv.render(record, None)

    assert "confirm.3 and confirm.4" in text and "confirm.1" not in text
    with pytest.raises(SystemExit):
        cv.render({"issue": 9999, "verdict": PASS, "gates": {}}, None)
