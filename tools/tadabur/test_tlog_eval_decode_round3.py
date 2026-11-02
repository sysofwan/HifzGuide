"""The spec-driven tlog decoder (#256): ``tadabur.tlog_eval_decode`` decodes any tlog round's models.

Written before the implementation and **locked**: later turns may add tests but never weaken or
rewrite these. Round 2's decode tests (``test_tlog_eval_decode``, ``test_tlog_eval_confirm``,
``test_tlog_eval_decode_shared``, ``test_consonant_readout_round2_spec``) stay as they are, so every
new ``spec`` parameter defaults to ``CONSONANT2`` and round 2's calls and outputs do not change.

The interface these tests pin:

* ``RoundSpec.confirm_tranches``: the claimable confirm tranches a round pools, in claim order.
  The default is round 2's ``("confirm.1", "confirm.2")``; ``CONSONANT3`` is
  ``("confirm.3", "confirm.4")``. ``training.tlog_eval.CONFIRM_TRANCHES`` stays round 2's.
* ``round_model_specs(spec)``: the decode specs of the round's look arms (``look_models(spec)``:
  ``r3_candidate_live`` and ``r3_candidate_ema`` for round 3), each from the identity the spec
  pins; an arm the spec does not pin is a ``ValueError`` naming it. ``round2_model_specs`` stays.
* ``require_pinned_identities(identities, models, spec=CONSONANT2)``, ``load_models(parts_dir,
  models, spec=CONSONANT2)``, ``require_confirm_models(models, spec=CONSONANT2)``,
  ``read_confirm_manifest(role, ..., spec=CONSONANT2)`` and ``selection_rows(role, spec=...)`` read
  the spec's pins, tranches and committed choice. A model that is another round's arm
  (``r2_*`` under ``CONSONANT3``) is refused by name.
* ``decode_shards(..., spec=CONSONANT2)`` passes the spec to the checks above. The default round's
  call keeps the two-argument form ``load_models(parts_dir, models)`` (round 2's tests stub it so);
  any other spec is passed as ``load_models(parts_dir, models, spec=spec)``.
* ``run_provenance(..., spec=CONSONANT2)`` and ``cache_selection(..., spec=CONSONANT2)`` take it too,
  and the CLI's ``cache``/``coverage`` call ``selection_rows(role, spec=spec)``.
* ``merge_confirm_decodes(..., spec=CONSONANT2)`` pools exactly ``spec.confirm_tranches`` and
  writes ``meta.confirm.json`` naming them.
* The CLI takes ``--spec`` (default ``consonant2``; only tlog rounds are choices) on ``decode``,
  ``cache``, ``coverage``, ``merge``, ``merge-confirm``, ``parity`` and ``gate``; ``merge`` and
  ``merge-confirm`` write to ``<spec.root>/decodes`` unless ``--out-dir`` says otherwise.

No test decodes anything or opens a GPU; every git state is real.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

from tadabur import tlog_eval_decode as ted
from tadabur.test_tlog_eval_confirm import SHARDS, _claim, _lines, _manifests, _rows
from tadabur.test_tlog_eval_decode import BASE_REVISION, PINNED_SHA256, _write_jsonl
from training import aug_readout_gate as ag
from training import aug_readout_spec as asp
from training import round2_fixtures as fx
from training.truth_baseline import DEFAULT_MODELS

ROUND3 = asp.CONSONANT3
ROUND2 = asp.CONSONANT2
R3_ARMS = ("r3_candidate_live", "r3_candidate_ema")
TRANCHES = ("confirm.3", "confirm.4")
CHOSEN = ("base", "c95", "r3_candidate_ema")
R3_SHA = "3" * 64


def _r3_identities() -> dict[str, asp.ModelIdentity]:
    """What the run issue will pin: one checkpoint read at its live and its averaged weights."""
    return {f"r3_candidate_{weights}": asp.ModelIdentity(
        "/root/scratch/issue-252/runs/consonant3_candidate/checkpoint.pt", weights, 10_000,
        fx.RECIPE, R3_SHA) for weights in ("live", "ema")}


def _filled() -> asp.RoundSpec:
    return dataclasses.replace(ROUND3, models={**ROUND3.models, **_r3_identities()})


class _Choice:
    def __init__(self, weights) -> None:
        self.choice = {"weights": weights}


@pytest.fixture
def chosen(monkeypatch) -> list[dict]:
    """``committed_choice`` stubbed to choose the EMA weights; records each call's arguments."""
    calls: list[dict] = []

    def committed_choice(layout=None, spec=None):
        calls.append({"layout": layout, "spec": spec})
        return _Choice({"candidate": "ema"})

    monkeypatch.setattr("training.aug_readout_gate.committed_choice", committed_choice)
    return calls


# --- the spec's tranches --------------------------------------------------------------------------


def test_each_round_names_the_confirm_tranches_it_pools():
    assert ROUND2.confirm_tranches == ("confirm.1", "confirm.2")
    assert ROUND3.confirm_tranches == TRANCHES
    assert asp.CONSONANT1.confirm_tranches == ("confirm.1", "confirm.2")


# --- the models and their pins --------------------------------------------------------------------


def test_round_3_s_arms_resolve_from_a_spec_that_pins_them():
    specs = ted.round_model_specs(_filled())
    assert tuple(specs) == R3_ARMS
    for name, identity in _r3_identities().items():
        assert (specs[name].ref, specs[name].use_ema, specs[name].step, specs[name].recipe) == (
            identity.ref, identity.weights == "ema", 10_000, fx.RECIPE)


def test_round_3_s_arms_are_refused_by_name_until_the_spec_pins_them():
    with pytest.raises(ValueError, match="r3_"):
        ted.round_model_specs(ROUND3)


def test_round_2_s_arms_still_resolve_through_the_generic_helper():
    assert tuple(ted.round_model_specs(fx.filled_spec())) == ted.ROUND2_MODELS


def _identities(**arms: str) -> dict:
    return {"base": {"model_ref": DEFAULT_MODELS["base"], "hub_revision": BASE_REVISION},
            "c95": {"model_ref": "/ckpt/c95.pt", "checkpoint_sha256": PINNED_SHA256["c95"]},
            **{name: {"model_ref": "/ckpt/r3.pt", "checkpoint_sha256": sha}
               for name, sha in arms.items()}}


def test_the_round_3_identities_pass_at_their_pin_and_each_wrong_one_is_refused():
    spec = _filled()
    models = ("base", "c95", *R3_ARMS)
    ted.require_pinned_identities(_identities(**dict.fromkeys(R3_ARMS, R3_SHA)), models,
                                  spec=spec)
    for name in R3_ARMS:
        wrong = _identities(**{**dict.fromkeys(R3_ARMS, R3_SHA), name: "0" * 64})
        with pytest.raises(ValueError, match=name):
            ted.require_pinned_identities(wrong, models, spec=spec)


def test_an_unpinned_round_3_arm_is_refused_by_the_identity_check():
    with pytest.raises(ValueError, match="r3_candidate_ema"):
        ted.require_pinned_identities(_identities(r3_candidate_ema=R3_SHA),
                                      ("base", "c95", "r3_candidate_ema"), spec=ROUND3)


def test_round_2_s_identity_check_is_unchanged_without_a_spec():
    identities = _identities()
    identities["candidate_ema"] = {"checkpoint_sha256": PINNED_SHA256["candidate_ema"]}
    ted.require_pinned_identities(identities, ("base", "c95", "candidate_ema"))


def _fake_loader(monkeypatch, seen: dict):
    """``load_decoders`` stubbed: writes each model's identity at its pin and returns markers."""
    import json

    def load_decoders(identity_path, specs):
        seen.update(specs)
        for name, spec in specs.items():
            if name == "base":
                identity = {"model_ref": DEFAULT_MODELS["base"], "hub_revision": BASE_REVISION}
            else:
                identity = {"model_ref": spec.ref, "checkpoint_sha256": (
                    PINNED_SHA256["c95"] if name == "c95" else R3_SHA)}
            identity_path(name).write_text(json.dumps({"identity": identity}), encoding="utf-8")
        return {name: object() for name in specs}

    monkeypatch.setattr(ted, "load_decoders", load_decoders)


def test_load_models_resolves_the_teacher_c95_and_the_round_3_arm(tmp_path, monkeypatch):
    seen: dict = {}
    _fake_loader(monkeypatch, seen)
    decoders = ted.load_models(tmp_path, CHOSEN, spec=_filled())
    assert set(decoders) == set(CHOSEN)
    assert seen["r3_candidate_ema"].ref == _r3_identities()["r3_candidate_ema"].ref
    assert seen["r3_candidate_ema"].use_ema


@pytest.mark.parametrize("name", ["r2_candidate_ema", "r2_control_live", "r2", "c178"])
def test_load_models_refuses_another_round_s_model_before_loading_anything(
        tmp_path, monkeypatch, name):
    seen: dict = {}
    _fake_loader(monkeypatch, seen)
    with pytest.raises(ValueError, match=name):
        ted.load_models(tmp_path, ("base", name), spec=_filled())
    assert seen == {}


def test_load_models_refuses_round_3_s_arms_under_round_2(tmp_path, monkeypatch):
    seen: dict = {}
    _fake_loader(monkeypatch, seen)
    with pytest.raises(ValueError, match="r3_candidate_ema"):
        ted.load_models(tmp_path, ("r3_candidate_ema",), spec=ROUND2)
    assert seen == {}


def test_load_models_refuses_an_arm_the_round_has_not_pinned_yet(tmp_path, monkeypatch):
    seen: dict = {}
    _fake_loader(monkeypatch, seen)
    with pytest.raises(ValueError, match="r3_"):
        ted.load_models(tmp_path, ("r3_candidate_ema",), spec=ROUND3)
    assert seen == {}


# --- the confirm decode's model gate --------------------------------------------------------------


@pytest.mark.parametrize("weights", [{"candidate": "ema"}, "ema"])
def test_a_round_3_confirm_accepts_the_teacher_c95_and_the_chosen_weights(
        monkeypatch, weights):
    monkeypatch.setattr("training.aug_readout_gate.committed_choice",
                        lambda *args, **kwargs: _Choice(weights))
    ted.require_confirm_models(CHOSEN, spec=ROUND3)
    ted.require_confirm_models(("r3_candidate_ema",), spec=ROUND3)


def test_a_round_3_confirm_reads_the_round_s_own_committed_choice(chosen):
    ted.require_confirm_models(CHOSEN, spec=ROUND3)
    assert [call["spec"] for call in chosen] == [ROUND3]
    assert chosen[0]["layout"].root == ROUND3.root


@pytest.mark.parametrize("model", ["r2_candidate_ema", "r2_candidate_live", "r2_control_ema",
                                   "r2", "c178", "candidate_ema", "h448",
                                   "r3_candidate_live", "r3_control_ema"])
def test_a_round_3_confirm_refuses_every_other_model_by_name(chosen, model):
    with pytest.raises(SystemExit, match=model):
        ted.require_confirm_models(("base", "c95", model), spec=ROUND3)


def test_round_2_s_confirm_gate_still_refuses_round_3_s_arms(monkeypatch):
    monkeypatch.setattr("training.aug_readout_gate.committed_choice",
                        lambda *args, **kwargs: _Choice({"candidate": "ema", "control": "ema"}))
    ted.require_confirm_models(("base", "c95", "r2_candidate_ema", "r2_control_ema"))
    with pytest.raises(SystemExit, match="r3_candidate_ema"):
        ted.require_confirm_models(("r3_candidate_ema",))


def test_a_confirm_decode_refuses_an_unchosen_selector_before_it_loads_a_model(
        tmp_path, chosen, monkeypatch):
    loaded: list = []
    monkeypatch.setattr(ted, "load_models", lambda *args, **kwargs: loaded.append(args))
    with pytest.raises(SystemExit, match="r3_candidate_live"):
        ted.decode_shards(parts_dir=tmp_path, cache_dir=tmp_path,
                          models=("r3_candidate_live",), role="confirm.3", spec=ROUND3)
    with pytest.raises(SystemExit, match="r2_candidate_ema"):
        ted.decode_shards(parts_dir=tmp_path, cache_dir=tmp_path,
                          models=("r2_candidate_ema",), role="confirm.4", spec=ROUND3)
    assert loaded == []


# --- the tranches: claims, order, planted rows ----------------------------------------------------


def _round(tmp_path: Path, *, claims: tuple[str, ...] = ("confirm.1", "confirm.2",
                                                          "confirm.3", "confirm.4"),
           manifests: dict | None = None):
    """A repository with every manifest committed and ``claims`` committed, round 3's spec."""
    from training import round2_verdict_fixtures as rv

    repo = fx.init_repo(tmp_path / "repo")
    tranches = repo / "tranches"
    tranches.mkdir()
    round_ = rv.Round(repo, ROUND3, ag.Layout(repo / "readout"), tranches)
    _manifests(round_, manifests)
    for role in claims:
        _claim(round_, role)
    return round_


def _read(round_, role: str, layout=None):
    return ted.read_confirm_manifest(role, tranches_dir=round_.tranches, layout=layout,
                                     spec=ROUND3)


def test_both_claimed_round_3_tranches_are_read_whole(tmp_path, chosen):
    round_ = _round(tmp_path)
    assert _read(round_, "confirm.3") == _rows("confirm.3")
    assert _read(round_, "confirm.4") == _rows("confirm.4")
    assert chosen[0]["spec"] == ROUND3 and chosen[0]["layout"].root == ROUND3.root


@pytest.mark.parametrize("role", ["confirm.1", "confirm.2", "look", "train", "confirm.5", ""])
def test_no_other_role_is_read_for_round_3(tmp_path, chosen, role):
    with pytest.raises(ValueError):
        _read(_round(tmp_path), role)


def test_a_round_3_role_is_refused_before_any_file_is_opened(tmp_path, chosen):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        ted.read_confirm_manifest("confirm.1", tranches_dir=empty, spec=ROUND3)


def test_round_2_still_refuses_round_3_s_tranches(tmp_path, chosen):
    round_ = _round(tmp_path)
    for role in TRANCHES:
        with pytest.raises(ValueError):
            ted.read_confirm_manifest(role, tranches_dir=round_.tranches, spec=ROUND2)


@pytest.mark.parametrize("claims", [
    ("confirm.1", "confirm.2"),
    ("confirm.1", "confirm.2", "confirm.4"),
])
def test_an_unclaimed_round_3_tranche_is_refused(tmp_path, chosen, claims):
    round_ = _round(tmp_path, claims=claims)
    with pytest.raises(ValueError, match="claim"):
        _read(round_, "confirm.3")


def test_confirm_4_is_refused_while_claim_3_is_not_committed(tmp_path, chosen):
    round_ = _round(tmp_path, claims=("confirm.1", "confirm.2"))
    _claim(round_, "confirm.3", commit=False)
    _claim(round_, "confirm.4")
    with pytest.raises(ValueError, match="confirm.3"):
        _read(round_, "confirm.4")


def test_a_round_3_claim_that_is_written_but_not_committed_opens_nothing(tmp_path, chosen):
    round_ = _round(tmp_path, claims=("confirm.1", "confirm.2"))
    _claim(round_, "confirm.3", commit=False)
    with pytest.raises(ValueError, match="claim"):
        _read(round_, "confirm.3")


def test_a_committed_round_3_claim_is_not_enough_without_the_committed_choice(tmp_path):
    round_ = _round(tmp_path)
    with pytest.raises(SystemExit, match="not committed"):
        _read(round_, "confirm.3", layout=ag.Layout(tmp_path / "no-readout"))


@pytest.mark.parametrize("other", ["look", "confirm.1", "confirm.2", "confirm.4"])
def test_a_row_planted_from_any_other_tranche_is_refused_for_confirm_3(
        tmp_path, chosen, other):
    planted = dict(_rows(other)[0], group_key="other")
    round_ = _round(tmp_path, manifests={"confirm.3": [*_rows("confirm.3")[:3], planted]})
    with pytest.raises(ValueError, match=rf"also in {other.replace('.', r'[.]')}\b"):
        _read(round_, "confirm.3")


def test_selection_rows_reads_a_confirm_tranche_through_the_spec_and_never_the_look(
        monkeypatch):
    seen = {}

    def read_confirm_manifest(role, **kwargs):
        seen.update(role=role, **kwargs)
        return _rows(role)

    monkeypatch.setattr(ted, "read_confirm_manifest", read_confirm_manifest)
    monkeypatch.setattr(
        ted, "read_manifest",
        lambda *args, **kwargs: pytest.fail("a confirm role must never read the look manifest"))
    assert ted.selection_rows("confirm.4", spec=ROUND3) == _rows("confirm.4")
    assert seen["role"] == "confirm.4" and seen["spec"] == ROUND3


# --- the decode entry point -----------------------------------------------------------------------


def test_the_look_decode_loads_round_3_s_arms_through_the_spec(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ted, "read_manifest", lambda role, **kwargs: [])
    monkeypatch.setattr(
        ted, "load_models", lambda parts_dir, models, **kwargs: seen.update(
            models=tuple(models), **kwargs) or {})
    monkeypatch.setattr(ted, "ayah_word_counts", lambda: None)
    ted.decode_shards(parts_dir=tmp_path, cache_dir=tmp_path, models=R3_ARMS, role="look",
                      spec=_filled())
    assert seen["models"] == R3_ARMS
    assert seen["spec"] == _filled()


def test_a_confirm_decode_reads_only_the_spec_s_tranches(tmp_path, chosen, monkeypatch):
    monkeypatch.setattr(ted, "load_models", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="confirm.3"):
        ted.decode_shards(parts_dir=tmp_path, cache_dir=tmp_path, models=CHOSEN,
                          role="confirm.1", spec=ROUND3)


# --- the pooled merge -----------------------------------------------------------------------------


MODELS = ("base", "c95")


def _r3_manifests() -> dict[str, list[dict]]:
    return {role: [dict(row) for row in _rows(role) if row["stratum"] == "gating"]
            for role in TRANCHES}


def _r3_parts(tmp_path: Path, manifests: dict, faults: dict | None = None) -> Path:
    import json

    faults = faults or {}
    parts = tmp_path / "parts"
    for role, rows in manifests.items():
        for model in MODELS:
            _write_jsonl(ted.decodes_path(parts, SHARDS[role], model, role=role),
                         _lines(rows, model, **faults.get((role, model), {})))
    for model in MODELS:
        (parts / f"{model}.identity.json").write_text(
            json.dumps({"model": model, "identity": {"checkpoint_sha256": "a" * 64},
                        "fingerprints": {}}), encoding="utf-8")
    return parts


def test_the_round_3_merge_pools_confirm_3_and_confirm_4(tmp_path):
    import gzip
    import json

    manifests = _r3_manifests()
    out = tmp_path / "decodes"
    counts = ted.merge_confirm_decodes(_r3_parts(tmp_path, manifests), out, manifests=manifests,
                                       provenance={"code_sha": "c" * 40}, models=MODELS,
                                       spec=ROUND3)
    assert counts == {"base": 6, "c95": 6}
    for model in MODELS:
        with gzip.open(out / f"tlog.confirm.{model}.jsonl.gz", "rt", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle]
        assert {line["tranche"] for line in lines} == set(TRANCHES)
        assert [line["key"] for line in lines] == sorted(line["key"] for line in lines)
    meta = json.loads((out / "meta.confirm.json").read_text(encoding="utf-8"))
    assert meta["tranches"] == list(TRANCHES)
    assert not (out / "meta.json").exists()


@pytest.mark.parametrize("roles", [("confirm.1", "confirm.2"), ("confirm.3",),
                                   ("confirm.3", "confirm.4", "confirm.2")])
def test_the_round_3_merge_needs_exactly_its_two_tranches(tmp_path, roles):
    manifests = {role: _rows(role) for role in roles}
    with pytest.raises(ValueError, match="confirm"):
        ted.merge_confirm_decodes(tmp_path / "parts", tmp_path / "decodes",
                                  manifests=manifests, provenance={}, models=MODELS,
                                  spec=ROUND3)
    assert not (tmp_path / "decodes").exists()


@pytest.mark.parametrize("role, fault", [
    ("confirm.3", {"drop": (1,)}), ("confirm.3", {"repeat": True}),
    ("confirm.4", {"extra": True}), ("confirm.4", {"drop": (3,)})])
def test_the_round_3_merge_refuses_a_model_that_misses_repeats_or_adds_a_row(
        tmp_path, role, fault):
    manifests = _r3_manifests()
    parts = _r3_parts(tmp_path, manifests, {(role, "c95"): fault})
    with pytest.raises(ValueError, match="c95"):
        ted.merge_confirm_decodes(parts, tmp_path / "decodes", manifests=manifests,
                                  provenance={}, models=MODELS, spec=ROUND3)


def test_round_2_s_merge_still_refuses_round_3_s_tranches(tmp_path):
    manifests = _r3_manifests()
    with pytest.raises(ValueError, match="confirm"):
        ted.merge_confirm_decodes(_r3_parts(tmp_path, manifests), tmp_path / "decodes",
                                  manifests=manifests, provenance={}, models=MODELS)


# --- the command line -----------------------------------------------------------------------------


def _cli(monkeypatch, *argv: str, **stubs) -> dict:
    """``main`` on ``argv`` with ``stubs`` in place of the named functions; each stub records
    its keyword arguments (and the positional ones as ``args``) and returns ``{}``."""
    seen: dict[str, dict] = {}
    for name, result in stubs.items():
        def stub(*args, _name=name, _result=result, **kwargs):
            seen[_name] = {"args": args, **kwargs}
            return _result
        monkeypatch.setattr(ted, name, stub)
    monkeypatch.setattr(sys, "argv", ["tlog_eval_decode", *argv])
    ted.main()
    return seen


def _decode_argv(tmp_path: Path, *extra: str) -> list[str]:
    return ["decode", "--parts-dir", str(tmp_path), "--cache-dir", str(tmp_path), *extra]


def test_decode_takes_round_3_s_spec(tmp_path, monkeypatch):
    seen = _cli(monkeypatch, *_decode_argv(tmp_path, "--spec", "consonant3", "--role",
                                           "confirm.3", "--models", *CHOSEN),
                decode_shards={})["decode_shards"]
    assert seen["spec"] == ROUND3
    assert seen["role"] == "confirm.3" and tuple(seen["models"]) == CHOSEN


def test_decode_defaults_to_round_2(tmp_path, monkeypatch):
    seen = _cli(monkeypatch, *_decode_argv(tmp_path), decode_shards={})["decode_shards"]
    assert seen["spec"] == ROUND2


@pytest.mark.parametrize("name", ["aug95", "consonant1", "round4", ""])
def test_a_spec_that_is_not_a_tlog_round_is_refused_by_the_command_line(
        tmp_path, monkeypatch, name):
    with pytest.raises(SystemExit) as refused:
        _cli(monkeypatch, *_decode_argv(tmp_path, "--spec", name), decode_shards={})
    assert refused.value.code == 2


@pytest.fixture
def merge_stubs(monkeypatch) -> dict:
    """What ``merge-confirm`` reads, stubbed: each tranche's manifest and claim, by role."""
    seen = {"manifests": [], "claims": []}

    def read_confirm_manifest(role, **kwargs):
        seen["manifests"].append((role, kwargs.get("spec")))
        return [role]

    def claim_record(role, *args, **kwargs):
        seen["claims"].append(role)
        return {"role": role}

    monkeypatch.setattr(ted, "read_confirm_manifest", read_confirm_manifest)
    monkeypatch.setattr("tadabur.tlog_tranches.claim_record", claim_record)
    return seen


def _merge_confirm(monkeypatch, tmp_path: Path, *argv: str) -> dict:
    return _cli(monkeypatch, "merge-confirm", "--parts-dir", str(tmp_path), *argv,
                merge_confirm_decodes={}, head_sha="h", provenance={})["merge_confirm_decodes"]


def test_merge_confirm_pools_the_spec_s_tranches_into_the_spec_s_decodes(
        tmp_path, monkeypatch, merge_stubs):
    merged = _merge_confirm(monkeypatch, tmp_path, "--models", *CHOSEN, "--spec", "consonant3")
    assert merged["spec"] == ROUND3
    assert merged["args"][1] == ROUND3.root / "decodes"
    assert tuple(merged["models"]) == CHOSEN
    assert set(merged["manifests"]) == set(merged["claims"]) == set(TRANCHES)
    assert merge_stubs["manifests"] == [(role, ROUND3) for role in TRANCHES]
    assert merge_stubs["claims"] == list(TRANCHES)


def test_merge_confirm_defaults_to_round_2_s_tranches_and_decodes(
        tmp_path, monkeypatch, merge_stubs):
    merged = _merge_confirm(monkeypatch, tmp_path, "--models", "base", "c95")
    assert merged["spec"] == ROUND2
    assert merged["args"][1] == ROUND2.root / "decodes"
    assert set(merged["manifests"]) == {"confirm.1", "confirm.2"}


def test_merge_confirm_out_dir_overrides_the_spec_s_decodes(tmp_path, monkeypatch, merge_stubs):
    out = tmp_path / "elsewhere"
    merged = _merge_confirm(monkeypatch, tmp_path, "--models", "base", "--spec", "consonant3",
                            "--out-dir", str(out))
    assert merged["args"][1] == out


def test_the_look_merge_writes_into_the_spec_s_decodes_for_the_spec_s_arms(
        tmp_path, monkeypatch):
    seen = _cli(
        monkeypatch, "merge", "--parts-dir", str(tmp_path), "--models", *R3_ARMS,
        "--spec", "consonant3", merge_decodes={}, run_provenance={})
    assert seen["merge_decodes"]["args"][1] == ROUND3.root / "decodes"
    assert tuple(seen["merge_decodes"]["models"]) == R3_ARMS
    assert seen["run_provenance"]["spec"] == ROUND3


def test_the_look_merge_still_defaults_to_round_2(tmp_path, monkeypatch):
    seen = _cli(monkeypatch, "merge", "--parts-dir", str(tmp_path),
                merge_decodes={}, run_provenance={})
    assert seen["merge_decodes"]["args"][1] == ROUND2.root / "decodes"
    assert seen["run_provenance"]["spec"] == ROUND2


def test_cache_takes_the_spec_for_a_confirm_tranche(tmp_path, monkeypatch):
    seen = _cli(monkeypatch, "cache", "--clip-cache", str(tmp_path), "--shard-cache",
                str(tmp_path), "--role", "confirm.4", "--spec", "consonant3",
                cache_selection={})["cache_selection"]
    assert seen["spec"] == ROUND3 and seen["role"] == "confirm.4"


def test_coverage_takes_the_spec_for_a_confirm_tranche(tmp_path, monkeypatch):
    seen = _cli(monkeypatch, "coverage", "--clip-cache", str(tmp_path), "--role", "confirm.3",
                "--spec", "consonant3", selection_rows=[], cache_coverage={})
    assert seen["selection_rows"]["args"] == ("confirm.3",)
    assert seen["selection_rows"]["spec"] == ROUND3
