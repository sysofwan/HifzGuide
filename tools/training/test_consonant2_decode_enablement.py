"""Round 2's non-tlog decodes can be taken, merged and read (#230, the decode-enablement slice).

The GPU decodes of the chosen candidate and control on qdat, QUD other half, Tadabur dev and
Tarteel v1 confirm run through the existing ``training.aug_readout_decode --spec consonant2``.
Two things stood between those shards and a readout, and these tests pin both:

* ``aug_readout.merge(decodes_dir, layout, spec=<round 2>)`` merges the non-tlog sets' shards
  for the round's chosen arms (it used to refuse every tlog round). The tlog sets are still
  merged only by ``tadabur.tlog_eval_decode``; a tlog identity file here is refused, as is a
  weight selector the committed choice did not pick, an unchosen role's reused file, and any
  merge before the choice is committed. It keeps the identities ``models.json`` already holds
  (the look's) and inherits C95's and the baselines' from the rounds that committed them.
* ``training.consonant_baseline`` produces G6's held-out caches for the round's arms:
  ``held_out_identity(model, spec=...)``, ``update_held_out(audio_dir, model, spec=...)`` (refused
  before the choice is committed and for weights it did not pick; the control's chosen weights
  are allowed, reported beside the candidate) and ``assert_held_out_caches(items, model=...,
  spec=...)``; the ``held-out`` command takes ``--spec`` and offers a round's own models only:
  C95 and the
  round's look models (``aug_readout_gate.look_models(spec)``: ``candidate_{live,ema}`` in round 1,
  ``r2_{candidate,control}_{live,ema}`` in round 2), whether or not the spec pins them yet.

Harness-free and GPU-free; the merge's file concatenation is stubbed as round 1's tests stub it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import aug_readout_spec as asp
from training import consonant_baseline as cb
from training import round2_fixtures as fx
from training import round2_verdict_fixtures as rv
from training import truth_baseline as tb
from training.test_consonant_baseline import _cache

REFUSED = (SystemExit, ValueError)
#: The sets a round-2 sitting decodes for its arms: the guards and the reported Tarteel confirm.
NON_TLOG_SETS = ("tarteel_v1.confirm", "qdat", "qud_other", "tadabur_dev_unseen")
CANDIDATE, CONTROL = "r2_candidate_ema", "r2_control_ema"


def _gpu(directory: Path, round_: rv.Round, models=(CANDIDATE, CONTROL),
         sets=NON_TLOG_SETS) -> Path:
    """A decodes directory as the shard runs leave it: one identity record per set and model."""
    directory.mkdir(exist_ok=True)
    for set_name in sets:
        for model in models:
            (directory / f"{set_name}.{model}.identity.json").write_text(
                json.dumps(fx.model_record(round_.spec, model)), encoding="utf-8")
    return directory


@pytest.fixture
def merged_calls(monkeypatch):
    """The file merge, recorded: ``(parts, identity, destination, expected)`` per call."""
    calls = []
    monkeypatch.setattr("training.external_corpus.merge_decode_files",
                        lambda *args: calls.append(args))
    return calls


# --- the merge -------------------------------------------------------------------------------------


def test_the_merge_writes_each_chosen_arm_s_file_in_the_rounds_own_root(tmp_path, merged_calls):
    round_ = rv.committed_round(tmp_path)
    ar.merge(_gpu(tmp_path / "gpu", round_), round_.layout, spec=round_.spec)

    counts = ar.expected_counts(round_.spec)
    wanted = {(set_name, model) for set_name in NON_TLOG_SETS for model in (CANDIDATE, CONTROL)}
    written = {Path(destination): expected for _, _, destination, expected in merged_calls}
    assert written == {round_.spec.root / "decodes" / f"{set_name}.{model}.jsonl.gz":
                       counts[set_name] for set_name, model in wanted}


def test_the_merge_keeps_the_look_s_identities_and_adds_the_sets_it_merged(tmp_path,
                                                                           merged_calls):
    round_ = rv.committed_round(tmp_path)
    before = json.loads(round_.layout.models.read_text(encoding="utf-8"))
    ar.merge(_gpu(tmp_path / "gpu", round_), round_.layout, spec=round_.spec)

    manifest = json.loads(round_.layout.models.read_text(encoding="utf-8"))
    assert before.items() <= manifest.items()
    for set_name in NON_TLOG_SETS:
        for model in (CANDIDATE, CONTROL):
            assert manifest[f"{set_name}/{model}"]["identity"] == ag.identity_record(
                round_.spec.models[model])


def test_the_merge_inherits_c95_and_the_baselines_so_the_manifest_is_complete(tmp_path,
                                                                              merged_calls):
    round_ = rv.committed_round(tmp_path)
    ar.merge(_gpu(tmp_path / "gpu", round_), round_.layout, spec=round_.spec)

    manifest = json.loads(round_.layout.models.read_text(encoding="utf-8"))
    choice = ag.committed_choice(round_.layout, spec=round_.spec).choice
    required = {name: models for name, models in ag.needs(
        "confirm", choice["weights"], round_.spec).items() if not name.startswith("tlog.")}
    assert set(required) == set(NON_TLOG_SETS)
    identities = ag.require_manifest(manifest, required, choice, spec=round_.spec)
    assert {"c95", "h448", "base", CANDIDATE, CONTROL} <= set(identities)
    assert {"tadabur_dev_unseen/h448", "tadabur_dev_unseen/base"} <= set(manifest)


def test_a_candidate_only_merge_is_allowed(tmp_path, merged_calls):
    round_ = rv.committed_round(tmp_path)
    ar.merge(_gpu(tmp_path / "gpu", round_, models=(CANDIDATE,)), round_.layout,
             spec=round_.spec)
    assert len(merged_calls) == len(NON_TLOG_SETS)


@pytest.mark.parametrize("set_name", ["tlog.look", "tlog.confirm"])
def test_a_tlog_set_is_still_merged_only_by_the_tlog_decoder(tmp_path, merged_calls, set_name):
    round_ = rv.committed_round(tmp_path)
    gpu = _gpu(tmp_path / "gpu", round_, sets=(set_name,))
    with pytest.raises(SystemExit, match="tlog_eval_decode"):
        ar.merge(gpu, round_.layout, spec=round_.spec)
    assert merged_calls == []


def test_a_merge_refuses_weights_the_committed_choice_did_not_pick(tmp_path, merged_calls):
    round_ = rv.committed_round(tmp_path)  # both arms chose EMA
    gpu = _gpu(tmp_path / "gpu", round_, models=("r2_candidate_live",), sets=("qdat",))
    with pytest.raises(REFUSED, match="choice"):
        ar.merge(gpu, round_.layout, spec=round_.spec)
    assert merged_calls == []


def test_a_merge_refuses_another_rounds_reused_file(tmp_path, merged_calls):
    round_ = rv.committed_round(tmp_path)
    gpu = _gpu(tmp_path / "gpu", round_, models=("c95",), sets=("qdat",))
    with pytest.raises(SystemExit, match="another round"):
        ar.merge(gpu, round_.layout, spec=round_.spec)
    assert merged_calls == []


def test_a_merge_before_the_choice_is_committed_reads_nothing(tmp_path, merged_calls):
    round_ = rv.committed_round(tmp_path)
    elsewhere = ag.Layout(round_.repo / "no-choice-here")
    elsewhere.root.mkdir()
    with pytest.raises(SystemExit, match="not committed"):
        ar.merge(_gpu(tmp_path / "gpu", round_), elsewhere, spec=round_.spec)
    assert merged_calls == []
    assert not elsewhere.models.exists()


def test_a_merge_with_a_foreign_identity_record_writes_no_manifest(tmp_path, merged_calls):
    round_ = rv.committed_round(tmp_path)
    gpu = _gpu(tmp_path / "gpu", round_, sets=("qdat",))
    record = fx.model_record(round_.spec, CANDIDATE)
    record["identity"] = {**record["identity"], "checkpoint_sha256": "0" * 64}
    (gpu / f"qdat.{CANDIDATE}.identity.json").write_text(json.dumps(record), encoding="utf-8")
    before = round_.layout.models.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="pinned checkpoint"):
        ar.merge(gpu, round_.layout, spec=round_.spec)
    assert round_.layout.models.read_text(encoding="utf-8") == before


# --- G6's held-out caches for the round's arms ---------------------------------------------------


def test_the_held_out_identity_is_the_rounds_pin_for_an_arm(tmp_path):
    round_ = rv.committed_round(tmp_path, checkpoints=True)
    for model in (CANDIDATE, CONTROL):
        assert cb.held_out_identity(model, spec=round_.spec) == ag.identity_record(
            round_.spec.models[model])
    assert cb.held_out_identity() == ag.identity_record(asp.CONSONANT1.models["c95"])


def test_an_arm_is_not_a_round_1_model(tmp_path):
    with pytest.raises(KeyError):
        cb.held_out_identity(CANDIDATE)  # round 1's pins hold no r2_* arm


def _caches_spy(monkeypatch):
    calls = []
    monkeypatch.setattr(cb, "model_caches", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(cb, "held_out_items", lambda registry=None: {})
    monkeypatch.setattr(cb, "load_staged_clips", lambda path: {})
    monkeypatch.setattr(cb, "assert_held_out_caches", lambda *args, **kwargs: None)
    return calls


def test_the_chosen_arms_are_decoded_at_their_chosen_weights_only(tmp_path, monkeypatch):
    round_ = rv.committed_round(tmp_path, rv.MIXED_FLAGS, checkpoints=True)
    calls = _caches_spy(monkeypatch)
    # The look chose the candidate's live and the control's EMA weights.
    cb.update_held_out(tmp_path, "r2_candidate_live", spec=round_.spec)
    cb.update_held_out(tmp_path, "r2_control_ema", spec=round_.spec)

    (live_args, live_kwargs), (ema_args, ema_kwargs) = calls
    assert (live_args[0], live_args[1], live_kwargs["use_ema"]) == (
        "r2_candidate_live", round_.spec.models["r2_candidate_live"].ref, False)
    assert (ema_args[0], ema_args[1], ema_kwargs["use_ema"]) == (
        "r2_control_ema", round_.spec.models["r2_control_ema"].ref, True)
    for unchosen in ("r2_candidate_ema", "r2_control_live"):
        with pytest.raises(SystemExit, match="committed choice reads"):
            cb.update_held_out(tmp_path, unchosen, spec=round_.spec)
    assert len(calls) == 2


def test_no_held_out_clip_is_decoded_before_the_choice_is_committed(tmp_path, monkeypatch):
    round_ = rv.committed_round(tmp_path, checkpoints=True)
    calls = _caches_spy(monkeypatch)
    import dataclasses

    elsewhere = dataclasses.replace(round_.spec, root=round_.repo / "no-choice-here")
    with pytest.raises(SystemExit, match="not committed"):
        cb.update_held_out(tmp_path, CANDIDATE, spec=elsewhere)
    assert calls == []


def test_a_checkpoint_that_is_not_the_pin_is_refused_before_decoding(tmp_path, monkeypatch):
    round_ = rv.committed_round(tmp_path, checkpoints=True)
    calls = _caches_spy(monkeypatch)
    Path(round_.spec.models[CANDIDATE].ref).write_bytes(b"another checkpoint")
    with pytest.raises(SystemExit, match="pin"):
        cb.update_held_out(tmp_path, CANDIDATE, spec=round_.spec)
    assert calls == []


def test_round_1_s_held_out_decode_is_unchanged(tmp_path, monkeypatch):
    calls = _caches_spy(monkeypatch)
    monkeypatch.setattr(cb, "file_sha256", lambda ref: asp.C95_SHA256)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    cb.update_held_out(tmp_path)  # C95, round 1's spec by default
    assert calls[0][0][0] == "c95" and calls[0][0][1] == asp.C95_REF


def test_the_held_out_caches_must_be_the_rounds_pin_for_the_arm(tmp_path):
    round_ = rv.committed_round(tmp_path, checkpoints=True)
    identity = ag.identity_record(round_.spec.models[CANDIDATE])
    ref = identity["model_ref"]
    check, decoded = tmp_path / "checks", tmp_path / "decodes"
    check.mkdir()
    decoded.mkdir()
    items = {"one": object()}
    _cache(decoded / f"{CANDIDATE}.json", CANDIDATE, ref, identity,
           {"one": {"spans": "", "stream_b0": ""}}, tb.PROTOCOLS)
    _cache(check / f"{CANDIDATE}.json", CANDIDATE, ref, identity,
           {"one": {"checks": []}}, tb.CHECK_PROTOCOLS)
    assert cb.assert_held_out_caches(items, model=CANDIDATE, check_dir=check,
                                     decode_dir=decoded, spec=round_.spec) == identity
    other = {**identity, "weights": tb.LIVE}  # the same checkpoint, the other selector
    _cache(check / f"{CANDIDATE}.json", CANDIDATE, ref, other,
           {"one": {"checks": []}}, tb.CHECK_PROTOCOLS)
    with pytest.raises(ValueError, match="pin"):
        cb.assert_held_out_caches(items, model=CANDIDATE, check_dir=check, decode_dir=decoded,
                                  spec=round_.spec)


def _held_out(monkeypatch, *argv):
    seen = {}

    def update(audio_dir, model=cb.BASELINE, spec=asp.CONSONANT1):
        seen.update(audio_dir=audio_dir, model=model, spec=spec)
        return {}

    monkeypatch.setattr(cb, "update_held_out", update)
    monkeypatch.setattr(sys, "argv", ["consonant_baseline", "held-out", *argv])
    cb.main()
    return seen


def test_the_held_out_command_takes_the_round_spec_and_its_arms(tmp_path, monkeypatch, capsys):
    seen = _held_out(monkeypatch, "--spec", "consonant2", "--audio-dir", str(tmp_path),
                     "--model", CANDIDATE)
    assert (seen["spec"], seen["model"], seen["audio_dir"]) == (
        asp.CONSONANT2, CANDIDATE, tmp_path)


def test_the_held_out_command_defaults_to_round_1_and_refuses_a_foreign_model(
        tmp_path, monkeypatch, capsys):
    seen = _held_out(monkeypatch, "--audio-dir", str(tmp_path))
    assert (seen["spec"], seen["model"]) == (asp.CONSONANT1, "c95")
    for argv in (["--model", CANDIDATE, "--audio-dir", str(tmp_path)],
                 ["--spec", "consonant2", "--model", "candidate_ema",
                  "--audio-dir", str(tmp_path)]):
        with pytest.raises(SystemExit) as stopped:
            _held_out(monkeypatch, *argv)
        assert stopped.value.code == 2
