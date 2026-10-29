"""The round-2 spec over the round-1 pipeline (#227): its basics, pins, refusals and bars.

The new interface these tests pin (names the implementation must provide):

* ``aug_readout_spec.CONSONANT2`` (registered as ``"consonant2"``) and a ``RoundSpec.prereg_path``
  field (round 2's is ``docs/consonant-round2-preregistration.md``);
* round-2 decode roles ``r2_candidate_{live,ema}`` and ``r2_control_{live,ema}``, which the spec
  refuses until its ``models`` hold their identities, never resolving them to #94's checkpoints;
* ``aug_readout_gate.TLOG_PROVENANCE`` and the sets ``tlog.look`` (read before the choice) and
  ``tlog.confirm`` (gated);
* ``tadabur.tlog_eval_decode.ROUND2_MODELS`` and ``round2_model_specs(spec)``.

Harness-free and GPU-free.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from tadabur import tlog_eval_decode as ted
from tadabur.tarteel_split import LOOK
from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import aug_readout_metrics as am
from training import aug_readout_spec as asp
from training import aug_readout_verdict as av
from training import round2_fixtures as fx
from training.acceptance_stats import FAIL, PASS
from training.test_consonant_readout_verdict import _report, _view
from training.truth_baseline import CHECKS

TOOLS = Path(__file__).resolve().parents[1]
ROOT2 = TOOLS / "tadabur" / "consonant2_readout"
HARNESS = {"muraja_commit": asp.MURAJA_COMMIT, "binary_sha256": asp.HARNESS_SHA256}
REFUSED = (SystemExit, ValueError)


# --- the spec ------------------------------------------------------------------------------------


def test_the_round_2_spec_is_registered_with_its_basics():
    spec = asp.spec("consonant2")
    assert spec is asp.CONSONANT2
    assert spec.name == "consonant2"
    assert spec.baseline == asp.C95 == "c95"
    assert spec.root == ROOT2
    assert spec.doc_path == TOOLS.parent / "docs" / "consonant-round2-readout.md"
    assert spec.prereg_path == TOOLS.parent / "docs" / "consonant-round2-preregistration.md"
    assert (am.CANDIDATE, am.C95) in spec.comparisons
    assert {"tlog.look", "tlog.confirm", "qdat", "qdat_bench", "qud_other",
            "tadabur_dev_unseen", "ship_a_verified", "tarteel_v1.look",
            "tarteel_v1.confirm"} <= set(spec.sets)
    assert {"consonant1": asp.CONSONANT1, "aug95": asp.AUG95}.items() <= asp.SPECS.items()


def test_the_round_2_spec_pins_muraja_and_the_harness():
    assert asp.CONSONANT2.pins.muraja_commit == asp.MURAJA_COMMIT == (
        "99c326f3fb7c6c53ff525568976c7e2442ccde3c")
    assert asp.CONSONANT2.pins.harness_sha256 == asp.HARNESS_SHA256 == (
        "cfcb99ba583fd2324f7ede0e5ab896379c54fa52783460d583e44d103eb68d8b")


def test_the_pre_registration_pin_is_the_committed_document_s_sha256():
    """#229 fills it: the pin is the committed doc's bytes, so an edited doc refuses."""
    pinned = asp.CONSONANT2.pins.prereg_sha256
    assert pinned == asp.CONSONANT2_PREREG_SHA256
    assert pinned == ag.sha256_file(asp.CONSONANT2.prereg_path)
    asp.check_pins(asp.CONSONANT2, HARNESS, pinned)  # every pin matches: no refusal
    with pytest.raises(SystemExit, match="pre-registration pin"):
        asp.check_pins(asp.CONSONANT2, HARNESS, "b" * 64)


def test_a_filled_pin_refuses_a_moved_pre_registration_and_a_moved_harness():
    pins = dataclasses.replace(asp.CONSONANT2.pins, prereg_sha256="a" * 64)
    spec = dataclasses.replace(asp.CONSONANT2, pins=pins)
    asp.check_pins(spec, HARNESS, "a" * 64)
    with pytest.raises(SystemExit, match="pre-registration pin"):
        asp.check_pins(spec, HARNESS, "b" * 64)
    for moved in ({**HARNESS, "muraja_commit": "0" * 40}, {**HARNESS, "binary_sha256": "0" * 64}):
        with pytest.raises(SystemExit, match="pin"):
            asp.check_pins(spec, moved, "a" * 64)


def test_round_1_pins_are_untouched():
    assert asp.CONSONANT1.pins.prereg_sha256 == asp.PREREG_SHA256
    assert asp.CONSONANT1.prereg_path == asp.PREREG_PATH


# --- the models: C95 and #178 are pinned, the arms are refused until filled ------------------------


def test_c95_and_the_178_context_column_are_the_round_1_identities():
    models = asp.CONSONANT2.models
    assert models[asp.C95] == asp.CONSONANT1.models[asp.C95]
    assert models["c178"] == asp.CONSONANT1.models["candidate_ema"]
    spec = ag.model_spec("c178", spec=asp.CONSONANT2)
    assert (spec.ref, spec.use_ema, spec.step, spec.recipe) == (
        asp.CANDIDATE_REF, True, 10_000, asp.CANDIDATE_RECIPE)
    assert ag.model_spec("c95", spec=asp.CONSONANT2).ref == asp.C95_REF


@pytest.mark.parametrize("model", fx.R2_MODELS)
def test_an_unfilled_round_2_role_is_refused_by_name(model):
    with pytest.raises(REFUSED, match=model):
        ag.model_spec(model, spec=asp.CONSONANT2)
    record = {"model": model, "fingerprints": {}, "identity": {
        "model_ref": "/x/checkpoint.pt", "weights": model.rsplit("_", 1)[1], "step": 10_000,
        "recipe": asp.CANDIDATE_RECIPE, "checkpoint_sha256": "c" * 64}}
    with pytest.raises(REFUSED):
        ag.validate_identity("tlog.look", model, record, spec=asp.CONSONANT2)


@pytest.mark.parametrize("model", ["candidate_live", "candidate_ema", "control_live",
                                   "control_ema"])
def test_the_round_never_falls_through_to_94_s_checkpoints(model):
    try:
        resolved = ag.model_spec(model, spec=asp.CONSONANT2)
    except REFUSED:
        return
    assert resolved.ref not in ag.ARMS.values()


def test_a_filled_arm_resolves_to_its_own_checkpoint_and_weights():
    spec = fx.filled_spec()
    for model, identity in fx.r2_identities().items():
        resolved = ag.model_spec(model, spec=spec)
        assert (resolved.ref, resolved.use_ema, resolved.step, resolved.recipe) == (
            identity.ref, identity.weights == "ema", 10_000, fx.RECIPE)
        record = fx.model_record(spec, model)
        assert ag.validate_identity("tlog.look", model, record, spec=spec) == record["identity"]


# --- where the round's decodes are read ---------------------------------------------------------


def test_the_tlog_look_decodes_are_212_s_committed_files_in_the_round_root():
    sources = ar.Sources(asp.CONSONANT2)
    decodes = ROOT2 / "decodes"
    assert sources.decode_path("tlog.look", "base") == decodes / "tlog.look.base.jsonl.gz"
    assert sources.decode_path("tlog.look", "c95") == decodes / "tlog.look.c95.jsonl.gz"
    # #178 keeps its committed name and meaning: c178 is the file called candidate_ema.
    c178 = sources.decode_path("tlog.look", "c178")
    assert c178 == decodes / "tlog.look.candidate_ema.jsonl.gz"
    for path in (sources.decode_path("tlog.look", m) for m in ("base", "c95", "c178")):
        assert path.is_file()


def test_the_arms_decodes_never_share_a_file_with_212_s():
    sources = ar.Sources(fx.filled_spec())
    paths = {m: sources.decode_path("tlog.look", m) for m in fx.R2_MODELS}
    assert paths["r2_candidate_ema"] == ROOT2 / "decodes" / "tlog.look.r2_candidate_ema.jsonl.gz"
    assert len(set(paths.values())) == 4
    assert sources.decode_path("tlog.look", "c178") not in paths.values()


@pytest.mark.parametrize("set_name", ["tarteel_v1.look", "tarteel_v1.confirm", "qdat",
                                      "qud_other", "tadabur_dev_unseen"])
def test_the_non_tlog_sets_read_the_decodes_round_1_maps_for_c95_and_178(set_name):
    one, two = ar.Sources(asp.CONSONANT1), ar.Sources(asp.CONSONANT2)
    assert two.decode_path(set_name, "c95") == one.decode_path(set_name, "c95")
    assert two.decode_path(set_name, "c178") == one.decode_path(set_name, "candidate_ema")
    for model in ("h448", "base") if set_name in asp.REUSED_SETS else ():
        assert two.decode_path(set_name, model) == one.decode_path(set_name, model)


# --- the bars ------------------------------------------------------------------------------------


def test_the_bars_are_round_1_s_plus_the_catch_loss_audit():
    bars = dict(asp.CONSONANT2.bars)
    assert bars.pop("catch_loss_real_at_most") == 0.20
    assert bars.pop("catch_loss_decisive_at_least") == 80
    assert bars == dict(asp.CONSONANT1.bars)
    assert asp.CONSONANT2.bars["reduction_at_least"] == 4.7


def test_the_verdict_records_the_catch_loss_bars():
    result = av.verdict(_report(), spec=asp.CONSONANT2)
    assert result["bars"]["catch_loss_real_at_most"] == 0.20
    assert result["bars"]["catch_loss_decisive_at_least"] == 80


def _shaddah_fails(report: dict) -> dict:
    bad = _view({"candidate": 85.0, "c95": 90.0}, (-5.0, -7.0, -3.0), am.POINTS)
    report["guards"]["qdat_bench"]["shaddah"] = bad
    return report


def test_a_shaddah_only_g3_failure_leaves_round_2_s_verdict_unchanged():
    clean = av.verdict(_report(), spec=asp.CONSONANT2)
    failed = av.verdict(_shaddah_fails(_report()), spec=asp.CONSONANT2)
    assert clean["verdict"] == failed["verdict"] == PASS
    assert [c["verdict"] for c in failed["conditions"]] == [
        c["verdict"] for c in clean["conditions"]]


@pytest.mark.parametrize("kind", ["consonant", "haraka"])
def test_g3_still_gates_consonant_and_haraka_in_round_2(kind):
    report = _report()
    report["guards"]["qdat_bench"][kind] = _view({"candidate": 85.0, "c95": 90.0},
                                                 (-5.0, -7.0, -3.0), am.POINTS)
    result = av.verdict(report, spec=asp.CONSONANT2)
    assert result["verdict"] == FAIL
    assert next(c for c in result["conditions"]
                if c["name"] == f"g3.qdat_bench.{kind}")["verdict"] == FAIL


def test_round_1_still_gates_shaddah():
    assert av.verdict(_shaddah_fails(_report()), spec=asp.CONSONANT1)["verdict"] == FAIL


# --- round 1's reproduction targets, pinned in the report that holds them ------------------------


def test_the_three_reproduction_counts_are_the_committed_ones():
    assert ar.REPRODUCTION == {"tarteel_v1_confirm": {"flagged": 1971, "graded": 41572},
                               "qdat": {"flagged": 670, "graded": 11502}}
    report = json.loads((TOOLS / "tadabur" / "consonant_readout" / "report.json")
                        .read_text(encoding="utf-8"))
    assert report["reproduction"]["counts"] == ar.REPRODUCTION
    models = report["main"]["all_cause"]["models"]
    assert (models["c95"]["num"], models["c95"]["den"]) == (1971, 41572)
    assert (models["candidate"]["num"], models["candidate"]["den"]) == (1837, 41652)


# --- the tlog sets in the gate -------------------------------------------------------------------


def test_the_tlog_sets_are_declared_with_their_provenance():
    provenance = ("split", "shard_file", "row", "audio_sha256", "start_sample", "end_sample")
    assert ag.TLOG_PROVENANCE == provenance
    assert ag.SETS["tlog.look"] == ag.ReadoutSet("tlog.look", (CHECKS,), provenance,
                                                 gated=False)
    assert ag.SETS["tlog.confirm"] == ag.ReadoutSet("tlog.confirm", (CHECKS,), provenance)
    assert "tlog.confirm" in ag.GATED and "tlog.look" not in ag.GATED


def test_round_1_reads_the_same_look_set_and_models_as_before():
    assert ag.needs(LOOK, None, asp.CONSONANT1) == {"tarteel_v1.look": ag.LOOK_MODELS}
    assert ag.needs(LOOK, None) == {"tarteel_v1.look": ag.LOOK_MODELS}


def test_the_look_sitting_reads_tlog_look_for_the_four_arm_weights_only():
    need = ag.needs(LOOK, None, fx.filled_spec())
    assert {name: set(models) for name, models in need.items()} == {
        "tlog.look": set(fx.R2_MODELS)}


def test_a_gated_set_refuses_before_a_choice_is_committed():
    spec = fx.filled_spec()
    for set_name in ("tlog.confirm", "qdat", "qud_other"):
        with pytest.raises(SystemExit, match="choice"):
            ag.allowed_models(set_name, None, spec)


def test_after_the_choice_only_each_arm_s_chosen_weights_are_read():
    choice = {"weights": {"candidate": "ema", "control": "live"}}
    allowed = set(ag.allowed_models("tlog.confirm", choice, fx.filled_spec()))
    assert {"r2_candidate_ema", "r2_control_live", "c95"} <= allowed
    assert not allowed & {"r2_candidate_live", "r2_control_ema"}


def test_a_tlog_decode_must_carry_its_whole_provenance(tmp_path):
    spec = dataclasses.replace(fx.filled_spec(), root=tmp_path)
    items = [fx.tlog_item(n, 1 + n) for n in range(3)]
    base = {i.key: fx.decode_line(i, fx.CLEAN) for i in items}
    path = ar.Sources(spec).decode_path("tlog.look", "r2_candidate_ema")
    path.parent.mkdir(parents=True)

    def write(lines):
        import gzip

        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines))

    keys = set(base)
    write(base.values())
    assert set(ar.read_readout("tlog.look", "r2_candidate_ema", base, keys, spec=spec)) == keys
    for edit, match in ((lambda line: line.update(row=7), "not the audio"),
                        (lambda line: line.update(shard_file="clean-9"), "not the audio"),
                        (lambda line: line.pop("split"), "to check its audio by")):
        moved = [dict(line) for line in base.values()]
        edit(moved[0])
        write(moved)
        with pytest.raises(ValueError, match=match):
            ar.read_readout("tlog.look", "r2_candidate_ema", base, keys, spec=spec)


# --- the confirm tranches stay sealed -------------------------------------------------------------


@pytest.mark.parametrize("role", ["confirm", "confirm.1", "confirm.2", "confirm.3", "confirm.4"])
def test_no_confirm_row_is_read_without_the_committed_claims(role):
    sources = ar.Sources(asp.CONSONANT2)
    with pytest.raises(REFUSED, match="claim|confirm"):
        sources.tlog(role, ("base", "c95", "c178"))


# --- the decoder -----------------------------------------------------------------------------------


def test_the_decoder_gains_the_round_2_models_and_keeps_212_s_unchanged():
    assert ted.ROUND2_MODELS == fx.R2_MODELS
    assert ted.MODELS == ("base", "c95", "candidate_ema", "h448")
    assert ted.RUN_MODELS == ("base", "c95", "candidate_ema")


def test_the_decoder_resolves_the_round_2_models_from_a_filled_spec_only():
    specs = ted.round2_model_specs(fx.filled_spec())
    assert tuple(specs) == fx.R2_MODELS
    for model, identity in fx.r2_identities().items():
        assert (specs[model].ref, specs[model].use_ema, specs[model].step,
                specs[model].recipe) == (identity.ref, identity.weights == "ema", 10_000,
                                         fx.RECIPE)
    with pytest.raises(REFUSED, match="r2_"):
        ted.round2_model_specs(asp.CONSONANT2)


# --- the command line ---------------------------------------------------------------------------


def test_the_look_command_takes_the_round_2_spec_and_its_root(monkeypatch, capsys):
    seen = {}

    def run_look(sources, layout, workers, spec):
        seen.update(spec=spec, root=layout.root, sources_spec=sources.spec)
        return {"weights": {"candidate": "ema", "control": "live"}}

    monkeypatch.setattr(ar, "run_look", run_look)
    monkeypatch.setattr(sys, "argv", ["aug_readout", "look", "--spec", "consonant2"])
    ar.main()
    assert seen == {"spec": asp.CONSONANT2, "root": ROOT2, "sources_spec": asp.CONSONANT2}
