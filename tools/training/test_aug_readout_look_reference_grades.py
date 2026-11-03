"""The look reference grades (#257): ``training.aug_readout.write_look_reference_grades``.

Round 3's catch-loss draw compares the round's ``r2`` reference, which the look sitting never
grades. The command grades it from its committed decodes on the look export's words at the
shipped default, writes ``look_grades_r2.jsonl.gz`` beside the export — the reference's rows
plus a word-for-word copy of the export's C95 rows — and records what produced it in
``look_grades_r2.provenance.json``.

The sitting is the round-3 draw test's throwaway repository
(:class:`tadabur.test_catch_loss_audit_three_model.Round3Audit`): its export is committed, its
models are the real ``CONSONANT3`` pins, and its reference source is the synthetic one the draw
tests use, with ``r2`` flagging block A/E/F/G's first word.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

from tadabur.test_catch_loss_audit_draw import _Sources
from tadabur.test_catch_loss_audit_three_model import (
    EXPORT, R2_FLAG, REFERENCE, Round3Audit, _read)
from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import aug_readout_spec as asp

REFUSED = (SystemExit, ValueError)
R2_BLOCKS = ("A", "E", "F", "G")


def _reference_source(look: Round3Audit, items=None) -> _Sources:
    """The look set with ``r2`` flagging each item's first word in :data:`R2_BLOCKS`."""

    def flags(model, n, position):
        if position != 0 or not any(n in look.blocks[name] for name in R2_BLOCKS):
            return None
        return R2_FLAG

    return _Sources(look.repo / "files", items or look.items, flags, spec=look.spec)


def _provenance(look: Round3Audit) -> dict:
    return json.loads(ar.look_reference_provenance_path(look.layout.root, REFERENCE)
                      .read_text(encoding="utf-8"))


# --- the file ------------------------------------------------------------------------------------


def test_it_writes_the_reference_rows_after_the_exports_c95_rows(tmp_path):
    look = Round3Audit(tmp_path)

    path = ar.write_look_reference_grades(_reference_source(look), look.layout, spec=look.spec)

    assert path == ar.look_reference_path(look.layout.root, REFERENCE)
    rows = _read(path)
    export = [row for row in _read(look.export) if row["model"] == asp.C95]
    assert rows[:len(export)] == export  # word for word, in the export's order
    reference = rows[len(export):]
    assert {row["model"] for row in reference} == {REFERENCE}
    assert {(row["key"], row["position"]) for row in reference} == {
        (row["key"], row["position"]) for row in export}
    assert sum(row["flagged"] for row in reference) == look.r2_flags


def test_the_grades_are_reproducible_bytes(tmp_path):
    look = Round3Audit(tmp_path)
    sources = _reference_source(look)

    first = ar.write_look_reference_grades(sources, look.layout, spec=look.spec).read_bytes()
    again = ar.write_look_reference_grades(sources, look.layout, spec=look.spec).read_bytes()

    assert again == first


def test_the_provenance_record_names_the_decode_the_harness_and_the_counts(tmp_path):
    look = Round3Audit(tmp_path)
    sources = _reference_source(look)

    path = ar.write_look_reference_grades(sources, look.layout, spec=look.spec)

    record = _provenance(look)
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["rows"] == len(_read(path))
    assert (record["reference"], record["scoring"]) == (REFERENCE, "shipped_default")
    assert record["counts"] == {"c95": look.c95_flags, "reference": look.r2_flags,
                                "both": look.both_flags}
    assert record["harness"] == sources.harness_identity()
    assert record["identity"] == ag.identity_record(look.spec.models[REFERENCE])
    assert record["export"]["sha256"] == ag.sha256_file(look.export)
    decode = sources.decode_path(look.spec.look_set, REFERENCE)
    assert record["decode"]["sha256"] == ag.sha256_file(decode)
    assert record["disclosed"] is None  # the fixture's spec declares none


# --- the refusals --------------------------------------------------------------------------------


def _assert_refused(look: Round3Audit, sources, spec, message: str) -> None:
    """The command refuses and leaves the sitting's files untouched (the fixture's own, here)."""
    path = ar.look_reference_path(look.layout.root, REFERENCE)
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(REFUSED, match=message):
        ar.write_look_reference_grades(sources, look.layout, spec=spec)
    assert (path.read_bytes() if path.exists() else None) == before
    assert not ar.look_reference_provenance_path(look.layout.root, REFERENCE).exists()


def test_it_refuses_a_round_that_names_no_reference(tmp_path):
    look = Round3Audit(tmp_path)
    spec = dataclasses.replace(look.spec, catch_loss_reference=None)

    _assert_refused(look, _reference_source(look), spec, "reference")


def test_it_refuses_an_export_that_is_not_committed(tmp_path):
    look = Round3Audit(tmp_path, uncommitted=(EXPORT,))

    _assert_refused(look, _reference_source(look), look.spec, "look_grades")


def test_it_refuses_a_reference_that_graded_other_words_than_the_export(tmp_path):
    look = Round3Audit(tmp_path)

    _assert_refused(look, _reference_source(look, look.items[:-1]), look.spec, "words")


def test_it_refuses_a_harness_other_than_the_pinned_one(tmp_path):
    look = Round3Audit(tmp_path)
    sources = _reference_source(look)
    sources.harness_record = {**sources.harness_record, "binary_sha256": "0" * 64}

    _assert_refused(look, sources, look.spec, "harness")


def test_it_refuses_counts_the_pre_registration_did_not_disclose(tmp_path):
    probe = Round3Audit(tmp_path / "probe")
    wrong = {"c95": probe.c95_flags, "reference": probe.r2_flags + 1,
             "both": probe.both_flags}
    look = Round3Audit(tmp_path / "look", disclosure=wrong)

    _assert_refused(look, _reference_source(look), look.spec, "disclosed")


# --- the CLI and the committed file --------------------------------------------------------------


def test_the_cli_grades_the_round_the_spec_names(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ar, "write_look_reference_grades",
        lambda sources, layout, spec: calls.append(
            (sources.spec.name, layout.root, spec.name)) or Path("grades"))
    monkeypatch.setattr(sys, "argv", ["aug_readout", "reference-grades", "--spec", "consonant3"])

    ar.main()

    assert calls == [("consonant3", asp.CONSONANT3.root, "consonant3")]


def test_the_committed_reference_grades_carry_their_provenance():
    root = asp.CONSONANT3.root
    record = json.loads(ar.look_reference_provenance_path(root, REFERENCE)
                        .read_text(encoding="utf-8"))

    assert record["sha256"] == ag.sha256_file(root / ar.LOOK_REFERENCE_GRADES
                                              .format(reference=REFERENCE))
    assert record["counts"] == {"c95": 3505, "reference": 2623, "both": 1624}
    assert (record["harness"]["muraja_commit"], record["harness"]["binary_sha256"]) == (
        asp.MURAJA_COMMIT, asp.HARNESS_SHA256)
    assert record["identity"] == ag.identity_record(asp.CONSONANT3.models[REFERENCE])
    assert record["decode"]["commit"]
