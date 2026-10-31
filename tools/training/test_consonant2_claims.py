"""Round 2's sequential tranche claims (#230).

``training.consonant2_verdict.claim(spec, role, *, tranches_dir, date, layout=None)`` (new) opens
one confirm tranche for this round and returns the claim file's path:

* it writes ``<tranches_dir>/<role>.claim.json`` through ``tadabur.tlog_tranches.claim_tranche``
  (exactly its ``CLAIM_FIELDS``: ``issue`` is the spec's, ``commit`` the commit that holds the
  committed ``preclaim.json``, ``date`` the one given) and the operator commits it; commit 1 is
  confirm.1's, commit 2 confirm.2's;
* it refuses (``SystemExit`` or ``ValueError``, no file written) unless ``preclaim.json`` is
  committed and unchanged in HEAD with an overall ``pass``;
* confirm.2 is refused until confirm.1's claim is committed in HEAD (``require_claim_order``);
* confirm.3 and confirm.4 are refused for this round under any input, as is any role that is not
  a confirm tranche, and a tranche already claimed.

Every git state is a real repository: ``committed in HEAD`` is the gate.
"""

from __future__ import annotations

import json
import sys

import pytest

from tadabur import tlog_tranches as tranches
from training import aug_readout_spec as asp
from training import consonant_readout as cr
from training import round2_verdict_fixtures as rv
from training.acceptance_stats import FAIL, PASS

REFUSED = (SystemExit, ValueError)
DATE = "2026-11-02"


@pytest.fixture
def cv():
    from training import consonant2_verdict

    return consonant2_verdict


class Sources:
    def input_files(self, phase, weights):
        return []

    def harness_identity(self):
        return {"binary_sha256": "a" * 64}


def _round(tmp_path, monkeypatch, cv, *, real=10, correct=90, commit=True) -> rv.Round:
    """A committed round whose pre-claim phase ran (``preclaim.json`` written, and committed
    unless ``commit`` is false); the audit passes for ``real`` of 100 mistakes at most 20."""
    round_ = rv.committed_round(tmp_path)
    rv.plant_audit(round_, real, correct)
    monkeypatch.setattr(cv, "preclaim_report",
                        lambda sources, spec, choice, workers=1: rv.clean_report())
    cv.run_preclaim(Sources(), round_.spec, round_.layout)
    if commit:
        round_.commit(round_.preclaim, message="the pre-claim")
    return round_


def _claim(cv, round_, role, **kwargs):
    return cv.claim(round_.spec, role, tranches_dir=round_.tranches, date=DATE,
                    layout=round_.layout, **kwargs)


def _claim_file(round_, role):
    return round_.tranches / f"{role}.claim.json"


def _assert_refused(cv, round_, role):
    with pytest.raises(REFUSED):
        _claim(cv, round_, role)
    assert not _claim_file(round_, role).exists()


def test_a_passing_preclaim_in_head_opens_confirm_1_through_the_tranche_claim(
        cv, tmp_path, monkeypatch):
    round_ = _round(tmp_path, monkeypatch, cv)
    held_in = round_.head()  # the commit that holds preclaim.json

    path = _claim(cv, round_, "confirm.1")

    assert path == _claim_file(round_, "confirm.1")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert set(record) == set(tranches.CLAIM_FIELDS)
    assert record == {"role": "confirm.1", "issue": round_.spec.issue, "commit": held_in,
                      "date": DATE}


def test_confirm_2_follows_a_committed_claim_1_and_names_the_same_input_commit(
        cv, tmp_path, monkeypatch):
    round_ = _round(tmp_path, monkeypatch, cv)
    held_in = round_.head()
    round_.commit(_claim(cv, round_, "confirm.1"), message="claim confirm.1")

    path = _claim(cv, round_, "confirm.2")

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["role"] == "confirm.2" and record["commit"] == held_in
    assert round_.head() != held_in  # claim 1's own commit sits on top, and is not named


def test_confirm_2_is_refused_until_claim_1_is_committed_in_head(cv, tmp_path, monkeypatch):
    round_ = _round(tmp_path, monkeypatch, cv)
    _assert_refused(cv, round_, "confirm.2")  # no claim 1 at all
    _claim(cv, round_, "confirm.1")  # written, not committed
    _assert_refused(cv, round_, "confirm.2")
    round_.commit(_claim_file(round_, "confirm.1"), message="claim confirm.1")
    _claim(cv, round_, "confirm.2")


@pytest.mark.parametrize("role", ["confirm.3", "confirm.4", "confirm.5", "look", "train",
                                  "confirm", ""])
def test_no_other_tranche_can_be_claimed_whatever_is_in_head(cv, tmp_path, monkeypatch, role):
    round_ = _round(tmp_path, monkeypatch, cv)
    round_.commit(_claim(cv, round_, "confirm.1"), message="claim confirm.1")
    round_.commit(_claim(cv, round_, "confirm.2"), message="claim confirm.2")
    _assert_refused(cv, round_, role)


def test_confirm_3_is_refused_even_though_the_tranche_module_would_allow_it(
        cv, tmp_path, monkeypatch):
    round_ = _round(tmp_path, monkeypatch, cv)
    round_.commit(_claim(cv, round_, "confirm.1"), message="claim confirm.1")
    round_.commit(_claim(cv, round_, "confirm.2"), message="claim confirm.2")
    tranches.require_claim_order("confirm.3", round_.tranches)  # S7 itself would open it
    _assert_refused(cv, round_, "confirm.3")


def test_a_tranche_is_claimed_once(cv, tmp_path, monkeypatch):
    round_ = _round(tmp_path, monkeypatch, cv)
    round_.commit(_claim(cv, round_, "confirm.1"), message="claim confirm.1")
    before = _claim_file(round_, "confirm.1").read_text(encoding="utf-8")
    with pytest.raises(REFUSED):
        _claim(cv, round_, "confirm.1")
    assert _claim_file(round_, "confirm.1").read_text(encoding="utf-8") == before


# --- no claim without a passing pre-claim in HEAD -------------------------------------------------


def test_no_preclaim_no_claim(cv, tmp_path):
    round_ = rv.committed_round(tmp_path)
    _assert_refused(cv, round_, "confirm.1")


def test_an_uncommitted_preclaim_opens_nothing(cv, tmp_path, monkeypatch):
    round_ = _round(tmp_path, monkeypatch, cv, commit=False)
    assert json.loads(round_.preclaim.read_text(encoding="utf-8"))["verdict"] == PASS
    _assert_refused(cv, round_, "confirm.1")


def test_a_preclaim_edited_after_its_commit_opens_nothing(cv, tmp_path, monkeypatch):
    round_ = _round(tmp_path, monkeypatch, cv, real=21, correct=79)
    assert json.loads(round_.preclaim.read_text(encoding="utf-8"))["verdict"] == FAIL
    record = json.loads(round_.preclaim.read_text(encoding="utf-8"))
    record["verdict"] = PASS
    round_.preclaim.write_text(json.dumps(record), encoding="utf-8")
    _assert_refused(cv, round_, "confirm.1")  # the edit is not in HEAD


@pytest.mark.parametrize("real, correct, outcome", [(21, 79, "fail"), (0, 79, "cannot_certify")])
def test_a_committed_preclaim_that_is_not_a_pass_opens_nothing(
        cv, tmp_path, monkeypatch, real, correct, outcome):
    round_ = _round(tmp_path, monkeypatch, cv, real=real, correct=correct)
    assert json.loads(round_.preclaim.read_text(encoding="utf-8"))["verdict"] == outcome
    _assert_refused(cv, round_, "confirm.1")
    _assert_refused(cv, round_, "confirm.2")


# --- the command ----------------------------------------------------------------------------------


def test_the_claim_command_takes_the_round_2_spec_a_role_and_a_date(cv, monkeypatch):
    seen = {}

    def claim(spec, role, **kwargs):
        seen.update(spec=spec, role=role, **kwargs)
        return "written"

    monkeypatch.setattr(cv, "claim", claim)
    monkeypatch.setattr(sys, "argv", ["consonant_readout", "claim", "--spec", "consonant2",
                                      "--role", "confirm.1", "--date", DATE])
    cr.main()
    assert seen["spec"] is asp.CONSONANT2 and seen["role"] == "confirm.1"
    assert seen["date"] == DATE

    monkeypatch.setattr(sys, "argv", ["consonant_readout", "claim", "--spec", "consonant1",
                                      "--role", "confirm.1", "--date", DATE])
    with pytest.raises(SystemExit) as stopped:
        cr.main()
    assert stopped.value.code == 2
