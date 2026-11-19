"""Phase C of #263 (the round-3 chosen weights over the mined pool): pins, rows, merge, driver.

Phase C decodes the **round-3 chosen weights** (``r3_candidate_ema``, and only them: base, C95
and round 2's EMA are phase B's and are not decoded again) in the ``checks`` and ``spans``
stores over two row sets of ``confirm.1`` and ``confirm.2``: the gating rows (the rows round 2's
committed ``tlog.confirm.*`` decodes hold) and phase B's labelled reported rows. Everything here
is torch-free; the launcher is ``test_tlog_mining_phase_c_launcher.py``.

Contract this file pins
-----------------------
``tadabur.tlog_mining.selection``

* ``require_phase(phase, *, amendment_path=MINING_AMENDMENT_PATH, round3_dir=ROUND3_DIR)``:
  ``"C"`` passes only when **all** of these hold, each failure a ``ValueError`` that names what
  failed (``amendment`` and ``SHA-256``; ``choice`` and ``r3_candidate_ema``; ``choice`` and
  ``checkpoint``; ``verdict`` and ``pass``): the amendment is the pinned one (as for ``"B"``);
  ``<round3_dir>/choice.json`` names ``models.candidate == "r3_candidate_ema"`` and
  ``identities.r3_candidate_ema.checkpoint_sha256 == training.aug_readout_spec.R3_CANDIDATE_SHA256``
  (the one place the pin lives); ``<round3_dir>/verdict.json`` has ``verdict == "pass"``.
  ``ROUND3_DIR`` is ``tools/tadabur/consonant3_readout``. Whether the files are *committed* is the
  launcher's check (``test_tlog_mining_phase_c_launcher.py``). Phases ``A``/``B`` are unchanged.
* ``PHASE_C == "C"``, ``PHASE_C_TRANCHES == ("confirm.1", "confirm.2")``.
* ``phase_c_rows(tranche, *, tranches_dir, split_dir, layout, spec, phonetizable=None)
  -> (decodable, unphonetizable)``: ``unphonetizable`` is exactly ``phase_b_rows``' listing;
  ``decodable`` is ``phase_a_rows``' gating rows (never filtered by the phonetizer: round 2's
  gating decodes hold every one) plus ``phase_b_rows``' decodable rows, each row once, in any
  order. Any tranche but ``confirm.1``/``confirm.2`` is a ``ValueError`` before a file opens.

``tadabur.tlog_mining.mined_decode.PHASE_C_MODELS == ("r3_candidate_ema",)``; ``MINED_MODELS`` is
phase B's, unchanged.

``tadabur.tlog_mining.merge.merge_phase_c_decodes(parts_dir, out_dir, *, rows, unphonetizable,
provenance, models)``: ``rows`` maps a tranche to ``phase_c_rows``' decodable list and
``unphonetizable`` to its listing. Each model must hold every decodable row exactly once
(missing, repeated, foreign or listed rows are a ``ValueError`` before anything is written).
Writes, in ``out_dir``, **never** ``meta.json`` (phase B's committed name):

* ``tlog.mined.<model>.jsonl.gz``: the mined (non-gating) rows, as phase B writes them (sorted by
  key; ``tranche`` and the manifest's ``label_source`` added);
* ``tlog.gating.<model>.jsonl.gz``: the gating rows, sorted by key, each line gaining ``tranche``
  (round 2's ``tlog.confirm.<model>`` record shape, plus ``spans``);
* ``meta.<model>.json`` for the single model: ``code_sha`` (from ``provenance``), ``models``,
  ``label_sources`` (a count over the **mined** rows), the ``unphonetizable`` listing, and
  ``identities`` keyed ``tlog.mined/<model>``.

``tadabur.tlog_mining.__main__``: ``run`` and ``decode`` with ``phase="C"`` read their rows
through ``phase_c_rows`` only, run **one** worker whatever the card offers, and decode with the
committed round-3 spec: ``ted.require_confirm_models(["r3_candidate_ema"], spec=CONSONANT3)`` and
``ted.load_models(parts_dir, models=["r3_candidate_ema"], spec=CONSONANT3)``. Phase B's calls stay
exactly as they are (no ``spec`` argument).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tadabur import tlog_eval_decode as ted
from tadabur.test_tlog_mining_phase_b import (AMENDMENT, MINED, SHARD_OF, _batch_rows, _claimed, _decode_args, _driver_args,
                                              _listing, _no_stream, _phonetizable, _quiet_box,
                                              _tranche)
from tadabur.test_tlog_mining_selection import _split_tree
from tadabur.tlog_mining import __main__ as cli
from tadabur.tlog_mining import merge, mined_decode, selection
from tadabur.tlog_mining.batch import BatchResult
from tadabur.tlog_mining.selection import phase_a_rows, phase_b_rows, require_phase
from tadabur.tlog_pass1 import read_jsonl_gz
from training.aug_readout_spec import CONSONANT3, R3_CANDIDATE_SHA256
from training.nonpro_decode import item_key

REPO = Path(__file__).resolve().parents[2]
ROUND3 = REPO / "tools" / "tadabur" / "consonant3_readout"
ROUND2_GATING = (REPO / "tools" / "tadabur" / "consonant2_readout" / "decodes"
                 / "tlog.confirm.r2_candidate_ema.jsonl.gz")
MINING_DECODES = REPO / "tools" / "tadabur" / "tlog_mining" / "decodes"
MODEL = "r3_candidate_ema"
MODELS = (MODEL,)


def _round3(directory: Path, *, choice=None, verdict="pass") -> Path:
    """A round-3 readout directory: the real choice (``choice`` edits it) and a verdict file."""
    directory.mkdir(parents=True, exist_ok=True)
    document = json.loads((ROUND3 / "choice.json").read_text(encoding="utf-8"))
    if choice is not None:
        choice(document)
    (directory / "choice.json").write_text(json.dumps(document), encoding="utf-8")
    (directory / "verdict.json").write_text(json.dumps({"verdict": verdict}), encoding="utf-8")
    return directory


def _require(directory: Path, **kwargs):
    return require_phase("C", amendment_path=AMENDMENT, round3_dir=directory, **kwargs)


# --- the pins ---------------------------------------------------------------------------------


def test_phase_c_runs_under_the_committed_round_3_choice_verdict_and_amendment():
    assert selection.PHASE_C == "C" and selection.PHASE_C_TRANCHES == MINED
    require_phase("C")
    assert json.loads((ROUND3 / "verdict.json").read_text(encoding="utf-8"))["verdict"] == "pass"


def test_phase_c_accepts_a_directory_holding_the_pinned_choice_and_a_passing_verdict(tmp_path):
    _require(_round3(tmp_path / "round3"))


def test_the_pinned_checkpoint_is_the_round_3_spec_pin_and_the_committed_choice_holds_it():
    choice = json.loads((ROUND3 / "choice.json").read_text(encoding="utf-8"))
    assert choice["identities"][MODEL]["checkpoint_sha256"] == R3_CANDIDATE_SHA256
    assert CONSONANT3.models[MODEL].checkpoint_sha256 == R3_CANDIDATE_SHA256


@pytest.mark.parametrize("edit, pattern", [
    (lambda choice: choice["models"].update(candidate="r3_candidate_live"),
     r"(?s)choice.*r3_candidate_ema"),
    (lambda choice: choice["identities"][MODEL].update(checkpoint_sha256="0" * 64),
     r"(?is)choice.*checkpoint"),
], ids=["other-candidate", "other-checkpoint"])
def test_phase_c_is_refused_for_a_choice_that_is_not_the_pinned_round_3_identity(
        tmp_path, edit, pattern):
    with pytest.raises(ValueError, match=pattern):
        _require(_round3(tmp_path / "round3", choice=edit))


@pytest.mark.parametrize("verdict", ["fail", "pending", None])
def test_phase_c_is_refused_unless_round_3s_verdict_is_pass(tmp_path, verdict):
    directory = _round3(tmp_path / "round3")
    if verdict is None:
        (directory / "verdict.json").unlink()
    else:
        (directory / "verdict.json").write_text(json.dumps({"verdict": verdict}),
                                                encoding="utf-8")
    with pytest.raises(ValueError, match=r"(?is)verdict.*pass|pass.*verdict"):
        _require(directory)


def test_phase_c_is_refused_without_the_choice_file(tmp_path):
    directory = _round3(tmp_path / "round3")
    (directory / "choice.json").unlink()
    with pytest.raises(ValueError, match="choice"):
        _require(directory)


def test_phase_c_is_refused_for_a_changed_or_missing_amendment(tmp_path):
    changed = tmp_path / "amendment.md"
    changed.write_bytes(AMENDMENT.read_bytes() + b"\nA late edit.\n")
    directory = _round3(tmp_path / "round3")
    for path in (changed, tmp_path / "absent.md"):
        with pytest.raises(ValueError, match="amendment") as raised:
            require_phase("C", amendment_path=path, round3_dir=directory)
        assert "SHA-256" in str(raised.value)


# --- the rows ---------------------------------------------------------------------------------


@pytest.fixture
def pool(tmp_path):
    """Both claims, the committed choice and manifests with every stratum, plus the split tree."""
    return _claimed(tmp_path, {role: _tranche(role) for role in MINED})


def _select(fixture, tranche, rows=None, **kwargs):
    round_, split = fixture
    kwargs.setdefault("phonetizable", _phonetizable)
    rows = rows or selection.phase_c_rows
    return rows(tranche, tranches_dir=round_.tranches, split_dir=split, layout=round_.layout,
                spec=round_.spec, **kwargs)


def _keys(rows) -> list[str]:
    return sorted(item_key(row["audio_path"]) for row in rows)


@pytest.mark.parametrize("tranche", MINED)
def test_phase_c_rows_are_the_gating_rows_plus_phase_bs_rows_and_phase_bs_listing(
        pool, tranche):
    decodable, listed = _select(pool, tranche)
    b_decodable, b_listed = _select(pool, tranche, phase_b_rows)
    round_, split = pool
    gating = phase_a_rows(tranche, tranches_dir=round_.tranches, split_dir=split,
                          layout=round_.layout, spec=round_.spec)
    assert _keys(listed) == _keys(b_listed) and _keys(b_listed)
    assert _keys(decodable) == _keys([*gating, *b_decodable])
    assert len(set(_keys(decodable))) == len(decodable)
    assert sorted(row["row"] for row in decodable) == [1, 2, 3, 4, 5, 6]
    assert {row["stratum"] for row in decodable if row["row"] in (1, 2)} == {"gating"}


def test_a_gating_row_is_decoded_whatever_the_phonetizer_says(pool):
    decodable, listed = _select(pool, "confirm.1", phonetizable=lambda label: False)
    assert sorted(row["row"] for row in decodable) == [1, 2]
    assert sorted(row["row"] for row in listed) == [3, 4, 5, 6, 8]


@pytest.mark.parametrize("tranche", ["look", "confirm.3", "confirm.4", "train", "confirm"])
def test_no_other_tranche_is_selected(pool, tranche):
    with pytest.raises(ValueError, match="confirm.1"):
        _select(pool, tranche)


def test_a_row_the_split_tree_gives_to_train_is_refused(tmp_path):
    round_, _ = _claimed(tmp_path, {role: _tranche(role) for role in MINED})
    row = _tranche("confirm.1")[0]
    split = _split_tree(tmp_path / "train-split",
                        {role: _tranche(role) for role in MINED},
                        override={(row["split"], row["shard_file"], row["row"]): "train"})
    with pytest.raises(ValueError, match="train"):
        _select((round_, split), "confirm.1")


def test_the_real_trees_give_round_twos_gating_keys_and_phase_bs_committed_rows():
    """The committed manifests: 5,762 gating rows (round 2's confirm decodes), 10,642 mined
    rows with the 26 phase B listed, every row once."""
    gating_keys = set(line["key"] for line in read_jsonl_gz(ROUND2_GATING))
    mined_keys = set(line["key"] for line in read_jsonl_gz(
        MINING_DECODES / "tlog.mined.base.jsonl.gz"))
    meta = json.loads((MINING_DECODES / "meta.json").read_text(encoding="utf-8"))
    decodable_keys: list[str] = []
    listed_keys: list[str] = []
    for tranche in MINED:
        decodable, listed = selection.phase_c_rows(tranche, phonetizable=_phonetizable)
        decodable_keys += _keys(decodable)
        listed_keys += _keys(listed)
    assert len(decodable_keys) == len(set(decodable_keys))
    assert set(decodable_keys) == gating_keys | mined_keys
    assert len(gating_keys) == 5762 and len(mined_keys) == 10_642
    assert not gating_keys & mined_keys
    assert sorted(listed_keys) == sorted(entry["key"] for entry in meta["unphonetizable"])


# --- the models -------------------------------------------------------------------------------


def test_phase_c_decodes_the_round_3_weights_only_and_phase_bs_set_is_unchanged():
    assert mined_decode.PHASE_C_MODELS == MODELS
    assert mined_decode.MINED_MODELS == ("base", "c95", "r2_candidate_ema")


# --- the merge --------------------------------------------------------------------------------


def _inputs(role_rows=None):
    """Per tranche the decodable rows (two gating, four mined) and the listed row."""
    rows = {role: [{**row, "shard_file": SHARD_OF[role]} for row in _tranche(role)
                   if row["row"] in (1, 2, 3, 4, 5, 6)] for role in MINED}
    listed = {role: [{**row, "shard_file": SHARD_OF[role]}
                     for row in _tranche(role) if row["row"] == 8] for role in MINED}
    return rows, listed


def _line(row: dict, model: str) -> dict:
    return {"key": item_key(row["audio_path"]), "audio_filename": row["audio_path"],
            "assigned_label": row["assigned_label"], "stratum": row["stratum"],
            "checks": [{"hop": model}], "spans": f"{model}:spans"}


def _write_parts(parts: Path, rows, *, models=MODELS, skip=(), extra=None) -> None:
    extra = extra or {}
    for tranche, tranche_rows in rows.items():
        for model in models:
            lines = [_line(row, model) for row in [*tranche_rows, *extra.get(tranche, ())]
                     if (tranche, model, row["row"]) not in skip]
            path = ted.decodes_path(parts, SHARD_OF[tranche], model, role=tranche)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(line) + "\n" for line in lines),
                            encoding="utf-8")
    for model in models:
        (parts / f"{model}.identity.json").write_text(
            json.dumps({"identity": {"model_ref": model}, "fingerprint": {"store": "checks"}}),
            encoding="utf-8")


def _merge(tmp_path, rows, listed):
    return merge.merge_phase_c_decodes(tmp_path / "parts", tmp_path / "out", rows=rows,
                                 unphonetizable=listed, provenance={"code_sha": "abc"},
                                 models=MODELS)


def _names(tmp_path) -> list[str]:
    out = tmp_path / "out"
    return sorted(path.name for path in out.iterdir()) if out.exists() else []


def test_the_merge_writes_mined_and_gating_files_and_its_own_meta_never_phase_bs(tmp_path):
    rows, listed = _inputs()
    _write_parts(tmp_path / "parts", rows)
    _merge(tmp_path, rows, listed)
    assert _names(tmp_path) == sorted([f"tlog.mined.{MODEL}.jsonl.gz",
                                       f"tlog.gating.{MODEL}.jsonl.gz", f"meta.{MODEL}.json"])
    by_key = {item_key(row["audio_path"]): (tranche, row)
              for tranche in MINED for row in rows[tranche]}
    for kind, wanted in (("mined", lambda row: row["stratum"] != "gating"),
                         ("gating", lambda row: row["stratum"] == "gating")):
        lines = read_jsonl_gz(tmp_path / "out" / f"tlog.{kind}.{MODEL}.jsonl.gz")
        assert [line["key"] for line in lines] == sorted(line["key"] for line in lines)
        expected = {key for key, (_, row) in by_key.items() if wanted(row)}
        assert {line["key"] for line in lines} == expected and len(lines) == len(expected) == (
            8 if kind == "mined" else 4)
        for line in lines:
            tranche, row = by_key[line["key"]]
            assert line["tranche"] == tranche and line["stratum"] == row["stratum"]
            assert line["assigned_label"] == row["assigned_label"]
            assert line["spans"] == f"{MODEL}:spans" and line["checks"] == [{"hop": MODEL}]
            if kind == "mined":
                assert line["label_source"] == row["label_source"]


def test_the_meta_reports_label_sources_of_the_mined_rows_the_listing_and_the_identity(tmp_path):
    rows, listed = _inputs()
    _write_parts(tmp_path / "parts", rows)
    _merge(tmp_path, rows, listed)
    meta = json.loads((tmp_path / "out" / f"meta.{MODEL}.json").read_text(encoding="utf-8"))
    assert meta["code_sha"] == "abc" and meta["models"] == [MODEL]
    assert meta["label_sources"] == dict(Counter(
        row["label_source"] for tranche in MINED for row in rows[tranche]
        if row["stratum"] != "gating"))
    assert sorted((entry["tranche"], entry["key"], entry["assigned_label"])
                  for entry in meta["unphonetizable"]) == sorted(
        (tranche, item_key(row["audio_path"]), "55:17") for tranche in MINED
        for row in listed[tranche])
    assert set(meta["identities"]) == {f"tlog.mined/{MODEL}"}
    assert meta["identities"][f"tlog.mined/{MODEL}"]["identity"] == {"model_ref": MODEL}


@pytest.mark.parametrize("problem", ["missing gating", "missing mined", "repeated", "listed",
                                     "absent model"])
def test_a_partial_or_foreign_part_is_refused_and_nothing_is_written(tmp_path, problem):
    rows, listed = _inputs()
    skip = {"missing gating": {("confirm.2", MODEL, 1)},
            "missing mined": {("confirm.2", MODEL, 5)}}.get(problem, ())
    extra = {"listed": {"confirm.1": listed["confirm.1"]},
             "repeated": {"confirm.1": rows["confirm.1"][:1]}}.get(problem)
    _write_parts(tmp_path / "parts", rows, models=() if problem == "absent model" else MODELS,
                 skip=skip, extra=extra)
    with pytest.raises(ValueError):
        _merge(tmp_path, rows, listed)
    assert _names(tmp_path) == []


def test_a_clip_in_both_tranches_is_refused(tmp_path):
    rows, listed = _inputs()
    rows["confirm.2"][2] = {**rows["confirm.1"][2], "shard_file": SHARD_OF["confirm.2"]}
    _write_parts(tmp_path / "parts", rows)
    with pytest.raises(ValueError, match="both"):
        _merge(tmp_path, rows, listed)


# --- the driver -------------------------------------------------------------------------------


def test_the_phase_c_driver_reads_both_tranches_through_phase_c_selection_only(
        tmp_path, monkeypatch):
    _quiet_box(monkeypatch)
    asked = []

    def rows(tranche, **kwargs):
        asked.append(tranche)
        return [], []

    def never(*args, **kwargs):
        raise AssertionError("another phase's selection is not phase C's")

    monkeypatch.setattr(cli, "phase_c_rows", rows)
    monkeypatch.setattr(cli, "phase_a_rows", never)
    monkeypatch.setattr(cli, "phase_b_rows", never)
    assert cli.run(_driver_args(tmp_path, "C")) == 0
    assert asked == list(MINED)


def test_the_driver_runs_one_phase_c_worker_whatever_the_card_offers(tmp_path, monkeypatch):
    _quiet_box(monkeypatch)
    monkeypatch.setattr(cli, "phase_c_rows", lambda role, **kwargs: ([], []))
    args = _driver_args(tmp_path, "C")
    assert cli.run(args) == 0
    assert args.parallel == 1
    assert cli._worker_count(4, 40_000, "C") == 1


def test_the_phase_c_driver_refuses_existing_parts_without_resume(tmp_path, monkeypatch):
    """A part already in the parts directory is a run to resume, never one to overwrite."""
    _quiet_box(monkeypatch)
    monkeypatch.setattr(cli, "phase_c_rows", lambda role, **kwargs: ([], []))
    (tmp_path / "tlog.confirm.1.clean-00001-of-00411.r3_candidate_ema.jsonl").write_text(
        "", encoding="utf-8")
    args = _driver_args(tmp_path, "C")
    args.resume = False
    with pytest.raises(SystemExit, match="--resume"):
        cli.run(args)


# --- the decode worker ------------------------------------------------------------------------


def test_the_phase_c_worker_binds_the_round_3_spec_and_decodes_every_row_of_the_phase(
        tmp_path, monkeypatch):
    rows, _ = _batch_rows(tmp_path)
    _listing(tmp_path)
    args = _decode_args(tmp_path)
    args.phase = "C"
    calls, fetched, batches = [], [], []

    def confirm(models, spec=None):
        calls.append(("confirm", list(models), spec))

    def load(parts_dir, models=None, spec=None):
        calls.append(("load", list(models), spec))
        return {}

    def batch(shard_rows, parts, *, role, **kwargs):
        batches.append([row["row"] for row in shard_rows])
        for model in MODELS:
            path = ted.decodes_path(parts, shard_rows[0]["shard_file"], model, role=role)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(
                json.dumps({"key": item_key(row["audio_path"])}) + "\n"
                for row in shard_rows), encoding="utf-8")
        return BatchResult(done=True, written=len(shard_rows))

    def never(*args, **kwargs):
        raise AssertionError("another phase's selection is not phase C's")

    monkeypatch.setattr(cli, "phase_c_rows", lambda role, **kwargs: (rows, []))
    monkeypatch.setattr(cli, "phase_a_rows", never)
    monkeypatch.setattr(cli, "phase_b_rows", never)
    monkeypatch.setattr(cli.ted, "require_confirm_models", confirm)
    monkeypatch.setattr(cli.ted, "load_models", load)
    monkeypatch.setattr(cli.ted, "ayah_word_counts", lambda: (lambda label: 3))
    monkeypatch.setattr(cli.ted, "shard_stream", lambda cache: _no_stream)
    monkeypatch.setattr(cli, "_fetch_shard",
                        lambda shard, shard_rows, a: fetched.append(shard))
    monkeypatch.setattr(cli, "decode_batch", batch)

    assert cli.decode(args) == 0
    assert [(name, models) for name, models, _ in calls] == [
        ("confirm", [MODEL]), ("load", [MODEL])]
    assert all(spec is CONSONANT3 for _, _, spec in calls)
    assert fetched == ["clean-00101-of-00411", "clean-00102-of-00411"]
    assert batches == [[1, 2], [3, 4]]
    # Held by the round-3 model alone: a second batch fetches and decodes nothing.
    assert cli.decode(args) == 0
    assert len(fetched) == 2 and len(batches) == 2


def test_the_phase_b_worker_still_calls_the_decoder_without_a_spec(tmp_path, monkeypatch):
    rows, _ = _batch_rows(tmp_path)
    _listing(tmp_path)
    args = _decode_args(tmp_path, budget=0.0)
    calls = []
    monkeypatch.setattr(cli, "phase_b_rows", lambda role, **kwargs: (rows, []))
    monkeypatch.setattr(cli.ted, "require_confirm_models",
                        lambda *a, **k: calls.append(("confirm", a, k)))
    monkeypatch.setattr(cli.ted, "load_models", lambda *a, **k: calls.append(("load", a, k)) or {})
    monkeypatch.setattr(cli.ted, "ayah_word_counts", lambda: (lambda label: 3))
    monkeypatch.setattr(cli.ted, "shard_stream", lambda cache: _no_stream)
    assert cli.decode(args) == 0
    assert calls[0] == ("confirm", (list(mined_decode.MINED_MODELS),), {})
    assert calls[1][2] == {"models": list(mined_decode.MINED_MODELS)}


def test_the_align_worker_still_refuses_phase_c(tmp_path):
    """``align`` is phase A's worker whatever the pins say."""
    args = SimpleNamespace(phase="C", role="look", shard_index=0, shard_count=1,
                           parts_dir=tmp_path, clip_dir=tmp_path, shard_cache=tmp_path,
                           tranches_dir=tmp_path, split_dir=tmp_path, look_clip_cache=None,
                           worklist=None, budget_seconds=1.0)
    with pytest.raises(ValueError, match="phase C"):
        cli.align(args)
