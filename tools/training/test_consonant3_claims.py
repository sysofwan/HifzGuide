"""Round 3's sequential tranche claims (#255): ``confirm.3`` then ``confirm.4``.

``training.consonant2_verdict.claim(spec, role, *, tranches_dir, date, layout=None)`` is
spec-driven: the roles it may open are ``spec.confirm_tranches``.

* ``CONSONANT3`` opens ``confirm.3`` and then ``confirm.4`` (the claim names the spec's issue and
  the commit holding the passing ``preclaim.json``); it refuses ``confirm.1`` and ``confirm.2``
  with a message naming the tranches the round reads, and writes no file;
* ``confirm.4`` is refused until ``confirm.3``'s claim is committed in HEAD
  (``tadabur.tlog_tranches.require_claim_order``), which itself needs ``confirm.2``'s committed
  claim: HEAD has it;
* ``CONSONANT2`` still refuses ``confirm.3`` and ``confirm.4``, naming its own two tranches, even
  when the order check would let them through.

Every git state is a real repository.
"""

from __future__ import annotations

import json

import pytest

from training import aug_readout as ar
from training import aug_readout_spec as asp
from training import round2_verdict_fixtures as rv
from training import round3_verdict_fixtures as r3
from training.acceptance_stats import PASS

REFUSED = (SystemExit, ValueError)
DATE = "2026-11-02"
ROUND3_ROLES = r"confirm\.3.*confirm\.4"
ROUND2_ROLES = r"confirm\.1.*confirm\.2"


@pytest.fixture
def cv():
    from training import consonant2_verdict

    return consonant2_verdict


class Sources(ar.Sources):
    def input_files(self, phase, weights):
        return []

    def harness_identity(self):
        return {"muraja_commit": asp.MURAJA_COMMIT, "binary_sha256": asp.HARNESS_SHA256}


def _passed_round(tmp_path, monkeypatch, cv, *, round3=True) -> rv.Round:
    """A committed round whose pre-claim passed and is committed in HEAD."""
    if round3:
        round_, report = r3.committed_round(tmp_path), r3.clean_report()
    else:
        round_, report = rv.committed_round(tmp_path), rv.clean_report()
    rv.plant_audit(round_, 10, 90)
    monkeypatch.setattr(cv, "preclaim_report", lambda sources, spec, choice, workers=1: report)
    cv.run_preclaim(Sources(), round_.spec, round_.layout)
    round_.commit(round_.preclaim, message="the pre-claim")
    return round_


def _claim(cv, round_, role):
    return cv.claim(round_.spec, role, tranches_dir=round_.tranches, date=DATE,
                    layout=round_.layout)


def _claim_file(round_, role):
    return round_.tranches / f"{role}.claim.json"


def test_round_3_claims_confirm_3_and_then_confirm_4(cv, tmp_path, monkeypatch):
    round_ = _passed_round(tmp_path, monkeypatch, cv)
    preclaim = json.loads(round_.preclaim.read_text(encoding="utf-8"))
    assert preclaim["verdict"] == PASS
    preclaim_commit = round_.head()  # the commit that holds preclaim.json

    path = _claim(cv, round_, "confirm.3")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert path == _claim_file(round_, "confirm.3")
    assert (record["role"], record["issue"], record["date"]) == (
        "confirm.3", asp.CONSONANT3.issue, DATE)
    assert record["commit"] == preclaim_commit
    round_.commit(path, message="claim confirm.3")

    path = _claim(cv, round_, "confirm.4")
    assert json.loads(path.read_text(encoding="utf-8"))["role"] == "confirm.4"


def test_confirm_4_without_confirm_3_s_claim_is_refused_and_writes_nothing(
        cv, tmp_path, monkeypatch):
    round_ = _passed_round(tmp_path, monkeypatch, cv)
    with pytest.raises(REFUSED, match=r"confirm\.3"):  # the order check names the missing claim
        _claim(cv, round_, "confirm.4")
    assert not _claim_file(round_, "confirm.4").exists()


def test_confirm_4_with_confirm_3_s_claim_written_but_not_committed_is_refused(
        cv, tmp_path, monkeypatch):
    round_ = _passed_round(tmp_path, monkeypatch, cv)
    _claim(cv, round_, "confirm.3")  # written, not committed
    with pytest.raises(REFUSED, match=r"confirm\.3"):
        _claim(cv, round_, "confirm.4")
    assert not _claim_file(round_, "confirm.4").exists()


@pytest.mark.parametrize("role", ["confirm.1", "confirm.2"])
def test_round_3_refuses_round_2_s_tranches_naming_its_own(cv, tmp_path, monkeypatch, role):
    round_ = _passed_round(tmp_path, monkeypatch, cv)
    before = _claim_file(round_, role).read_text(encoding="utf-8")
    with pytest.raises(REFUSED, match=ROUND3_ROLES):
        _claim(cv, round_, role)
    assert _claim_file(round_, role).read_text(encoding="utf-8") == before


@pytest.mark.parametrize("role", ["look", "confirm", "confirm.5", ""])
def test_round_3_refuses_a_role_that_is_not_a_confirm_tranche(cv, tmp_path, monkeypatch, role):
    round_ = _passed_round(tmp_path, monkeypatch, cv)
    with pytest.raises(REFUSED):
        _claim(cv, round_, role)
    assert sorted(path.name for path in round_.tranches.glob("*.claim.json")) == [
        "confirm.1.claim.json", "confirm.2.claim.json"]


def test_round_3_opens_nothing_without_a_committed_passing_preclaim(cv, tmp_path, monkeypatch):
    round_ = r3.committed_round(tmp_path)  # the pre-claim phase has not run
    with pytest.raises(REFUSED):
        _claim(cv, round_, "confirm.3")
    assert not _claim_file(round_, "confirm.3").exists()


@pytest.mark.parametrize("role", ["confirm.3", "confirm.4"])
def test_round_2_still_refuses_round_3_s_tranches_naming_its_own(cv, tmp_path, monkeypatch,
                                                                 role):
    round_ = _passed_round(tmp_path, monkeypatch, cv, round3=False)
    for earlier in ("confirm.1", "confirm.2"):  # the order check would let confirm.3 through
        r3.plant_claim(round_, earlier)
    with pytest.raises(REFUSED, match=ROUND2_ROLES):
        _claim(cv, round_, role)
    assert not _claim_file(round_, role).exists()
