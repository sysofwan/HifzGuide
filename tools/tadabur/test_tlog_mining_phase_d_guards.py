"""Phase D's merge guards (#263): a part that is not this row's, and parts with no producer.

Review round 1 (findings R1.1, R1.2) found both failures invisible to the completeness checks: a
resume knows a part only by its ``key``, and a standalone merge — or a resume whose parts are
already complete — never loads the teacher. Either way the merge would publish a success-shaped
artifact: timings aligned to another ayah or another clip's audio, or no producer at all.

This file pins the two refusals that close them, as published behaviour of

* ``tadabur.tlog_mining.merge.merge_mined_word_times``: every record must repeat its selected
  row's ``surah_ayah`` (the row's ``assigned_label``) and audio provenance (``audio_sha256``,
  ``audio_path``, ``split``, ``shard_file``, ``row``); a mismatch names the key and the field and
  nothing is written; and
* ``tadabur.tlog_mining.__main__.merge --phase D``: the parts must hold ``teacher.json`` naming
  the pinned base teacher (``tadabur.tlog_ident.BASE_MODEL``/``BASE_REVISION``) with its
  whole-spans decode fingerprint; a missing record, another model, a missing or wrong revision or
  another mode refuses before any output file is written.

The pinned producer identity (``BASE_MODEL``/``BASE_REVISION``, ``training.decoding.SPANS``) is
accepted and nothing else is, so the guard refuses unpinned inputs, never a real run.
"""

from __future__ import annotations

import json

import pytest

from tadabur.test_tlog_mining_phase_b import SHARD_OF, _merge_inputs
from tadabur.test_tlog_mining_phase_d import (PROVENANCE, _merge, _merge_args, _names, _record,
                                              _write_parts)
from tadabur.tlog_ident import BASE_MODEL, BASE_REVISION
from tadabur.tlog_mining import __main__ as cli
from tadabur.tlog_mining import merge
from training.decoding import SPANS


def _mutated_first_record(tmp_path, **changes):
    """Both tranches' parts, confirm.1's first record replaced by a record with ``changes``."""
    rows, listed = _merge_inputs(tmp_path)
    row = rows["confirm.1"][0]
    _write_parts(tmp_path / "parts", rows, skip={("confirm.1", row["row"])},
                 extra=[(SHARD_OF["confirm.1"], {**_record(row), **changes})])
    return rows, listed


def test_a_part_record_of_another_ayah_is_refused_under_a_valid_key(tmp_path):
    """The key is complete and unique; only the reference is wrong. That is the R1.1 failure."""
    rows, listed = _mutated_first_record(tmp_path, surah_ayah="9:9")
    with pytest.raises(ValueError, match=r"surah_ayah.*9:9"):
        _merge(tmp_path, rows, listed)
    assert _names(tmp_path) == []


def test_a_part_record_of_another_clip_is_refused_under_a_valid_key(tmp_path):
    """The key names this row but the audio checksum is another clip's."""
    rows, listed = _mutated_first_record(tmp_path, audio_sha256="0" * 64)
    with pytest.raises(ValueError, match="audio_sha256"):
        _merge(tmp_path, rows, listed)
    assert _names(tmp_path) == []


def test_the_guard_is_the_selected_rows_not_the_records(tmp_path):
    """The records' own ``assigned_label`` is the truth; a merge of them needs no rewrite."""
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows)
    merge.merge_mined_word_times(tmp_path / "parts", tmp_path / "out", rows=rows,
                                 unphonetizable=listed, provenance=PROVENANCE)
    assert _names(tmp_path) == ["meta.mined.json", "mined.confirm.1.jsonl.gz",
                                "mined.confirm.2.jsonl.gz"]


# --- the producer record ----------------------------------------------------------------------


TEACHER = {"identity": {"hub_revision": BASE_REVISION, "model_ref": BASE_MODEL},
           "fingerprint": {"mode": SPANS, "weights_dtype": "bf16"}}


def _cli_merge(tmp_path, monkeypatch, rows, listed):
    """``merge --phase D`` over ``rows``, with the selection stubbed to them."""
    monkeypatch.setattr(cli, "phase_d_rows",
                        lambda tranche, **kwargs: (rows[tranche], listed[tranche]))
    return cli.merge(_merge_args(tmp_path))


def test_the_pinned_teacher_identity_and_fingerprint_merge_cleanly(tmp_path, monkeypatch):
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows)
    (tmp_path / "parts" / cli.TEACHER_RECORD).write_text(json.dumps(TEACHER), encoding="utf-8")
    assert _cli_merge(tmp_path, monkeypatch, rows, listed) == 0
    meta = json.loads((tmp_path / "out" / "meta.mined.json").read_text(encoding="utf-8"))
    assert meta["teacher"] == TEACHER


def test_the_merge_refuses_parts_with_no_producer_record(tmp_path, monkeypatch):
    """Complete part files restored without ``teacher.json``: no identity, so no artifact."""
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows)
    with pytest.raises(ValueError, match="teacher.json"):
        _cli_merge(tmp_path, monkeypatch, rows, listed)
    assert _names(tmp_path) == []


@pytest.mark.parametrize("record", [
    {"fingerprint": {"mode": SPANS}},
    {"identity": {"hub_revision": BASE_REVISION, "model_ref": BASE_MODEL}},
    {"identity": {"model_ref": BASE_MODEL}, "fingerprint": {}},
    {"identity": {"hub_revision": BASE_REVISION, "model_ref": "unpinned-student"},
     "fingerprint": {"mode": SPANS}},
    {"identity": {"model_ref": BASE_MODEL}, "fingerprint": {"mode": SPANS}},
    {"identity": {"hub_revision": "wrong-revision", "model_ref": BASE_MODEL},
     "fingerprint": {"mode": SPANS}},
    {"identity": {"hub_revision": BASE_REVISION, "model_ref": BASE_MODEL},
     "fingerprint": {"mode": "spans"}},
], ids=["no-identity", "no-fingerprint", "no-mode", "wrong-model", "no-revision",
        "wrong-revision", "not-whole-spans"])
def test_the_merge_refuses_a_producer_record_that_is_not_the_pinned_teacher(
        tmp_path, monkeypatch, record):
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows)
    (tmp_path / "parts" / cli.TEACHER_RECORD).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="teacher"):
        _cli_merge(tmp_path, monkeypatch, rows, listed)
    assert _names(tmp_path) == []
