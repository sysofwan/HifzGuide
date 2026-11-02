"""The round-3 spec over the round-2 pipeline (#251): its basics, pins, models and decode sources.

The interface these tests pin (names the implementation must provide):

* ``aug_readout_spec.CONSONANT3`` (registered as ``"consonant3"``) and a ``RoundSpec`` field
  ``registered_margins: Mapping[str, tuple[float, str]]`` (empty for every other round);
* the round-3 candidate's decode roles ``r3_candidate_{live,ema}`` (``arm_prefix="r3_"``), which
  the spec refuses by name until the run issue fills its ``models``;
* ``models["r2"]``: round 2's candidate at its averaged weights, the round's context column,
  whose look decodes are read from round 2's committed files, never copied.

Harness-free and GPU-free.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import aug_readout_metrics as am
from training import aug_readout_spec as asp

TOOLS = Path(__file__).resolve().parents[1]
ROOT2 = TOOLS / "tadabur" / "consonant2_readout"
HARNESS = {"muraja_commit": asp.MURAJA_COMMIT, "binary_sha256": asp.HARNESS_SHA256}
REFUSED = (SystemExit, ValueError)
R3_MODELS = ("r3_candidate_live", "r3_candidate_ema")
NON_TLOG_SETS = ("qdat", "qud_other", "tadabur_dev_unseen")


def test_the_round_3_spec_is_registered_with_its_basics():
    spec = asp.spec("consonant3")
    assert spec is asp.CONSONANT3
    assert spec.name == "consonant3"
    assert spec.baseline == asp.C95 == "c95"
    assert spec.root == TOOLS / "tadabur" / "consonant3_readout"
    docs = TOOLS.parent / "docs"
    assert spec.prereg_path == docs / "consonant-round3-preregistration.md"
    assert spec.doc_path == docs / "consonant-round3-readout.md"
    assert spec.freeze_doc_path == docs / "consonant-round3-freeze.md"
    assert spec.look_set == asp.TLOG_LOOK
    assert (am.CANDIDATE, am.C95) in spec.comparisons
    assert asp.tlog_round(spec)
    assert {"consonant1": asp.CONSONANT1, "consonant2": asp.CONSONANT2,
            "aug95": asp.AUG95}.items() <= asp.SPECS.items()


def test_the_round_reads_round_2_s_sets_and_has_no_control_arm():
    spec = asp.CONSONANT3
    assert spec.sets == asp.CONSONANT2.sets
    assert spec.look_roles == (am.CANDIDATE,)
    assert am.CONTROL not in spec.students
    assert (spec.g3_kinds, spec.g3_reported_kinds) == (
        asp.CONSONANT2.g3_kinds, asp.CONSONANT2.g3_reported_kinds)


def test_the_round_pins_round_2_s_muraja_and_harness_and_its_pre_registration():
    pins = asp.CONSONANT3.pins
    assert (pins.muraja_commit, pins.harness_sha256) == (asp.MURAJA_COMMIT, asp.HARNESS_SHA256)
    assert pins.prereg_sha256 == asp.CONSONANT3_PREREG_SHA256


def test_the_pre_registration_pin_is_the_committed_document_s_sha256():
    """#253 fills it: the pin is the committed doc's bytes, so an edited doc refuses."""
    pinned = asp.CONSONANT3.pins.prereg_sha256
    assert pinned == ag.sha256_file(asp.CONSONANT3.prereg_path)
    asp.check_pins(asp.CONSONANT3, HARNESS, pinned)  # every pin matches: no refusal
    with pytest.raises(SystemExit, match="pre-registration pin"):
        asp.check_pins(asp.CONSONANT3, HARNESS, "b" * 64)


# --- the bars and the amendment --------------------------------------------------------------------


def test_g4_is_registered_at_2_5_and_effective_at_3_5_under_amendment_a1():
    spec = asp.CONSONANT3
    assert spec.bars["tashkeel_upper_at_most"] == 3.5
    assert spec.bars["power_at_least"] == 0.85
    assert spec.registered_margins == {"tashkeel_upper_at_most": (2.5, "A1")}


def test_every_other_bar_is_round_2_s_unchanged():
    changed = {"tashkeel_upper_at_most", "power_at_least"}
    carried = {name: bar for name, bar in asp.CONSONANT3.bars.items() if name not in changed}
    assert carried == {name: bar for name, bar in asp.CONSONANT2.bars.items()
                       if name not in changed}
    assert set(asp.CONSONANT3.bars) == set(asp.CONSONANT2.bars) | {"power_at_least"}


def test_round_2_and_round_1_are_untouched_by_the_amendment():
    for spec in (asp.AUG95, asp.CONSONANT1, asp.CONSONANT2):
        assert spec.registered_margins == {}
        assert "power_at_least" not in spec.bars
    assert asp.CONSONANT2.bars["tashkeel_upper_at_most"] == 1.0


# --- the models ------------------------------------------------------------------------------------


def test_c95_and_the_round_2_ema_are_the_pinned_models():
    models = asp.CONSONANT3.models
    assert set(models) == {asp.C95, "r2"}
    assert models[asp.C95] == asp.CONSONANT1.models[asp.C95]
    assert models["r2"] == asp.ModelIdentity(
        asp.R2_CANDIDATE_REF, asp.EMA, 10_000, asp.CANDIDATE_RECIPE, asp.R2_CANDIDATE_SHA256)
    assert asp.R2_CANDIDATE_SHA256.startswith("8935")
    assert asp.R2_CANDIDATE_SHA256.endswith("ea58")
    resolved = ag.model_spec("r2", spec=asp.CONSONANT3)
    assert (resolved.ref, resolved.use_ema, resolved.step, resolved.recipe) == (
        asp.R2_CANDIDATE_REF, True, 10_000, asp.CANDIDATE_RECIPE)


def test_the_round_s_candidate_decodes_at_both_weights_under_its_own_prefix():
    assert ag.look_models(asp.CONSONANT3) == R3_MODELS
    assert ar.look_graded_models(asp.CONSONANT3) == (*R3_MODELS, asp.C95)


@pytest.mark.parametrize("model", R3_MODELS)
def test_an_unfilled_round_3_role_is_refused_by_name(model):
    """The run issue fills the candidate; until then no manifest names it and nothing resolves
    it to round 2's or #94's checkpoint."""
    assert model not in asp.CONSONANT3.models
    with pytest.raises(REFUSED, match=model):
        ag.model_spec(model, spec=asp.CONSONANT3)


# --- where the decodes are read from ---------------------------------------------------------------


def test_the_look_gating_decodes_are_round_2_s_committed_files():
    spec = asp.CONSONANT3
    decodes = ROOT2 / "decodes"
    assert ag.round_decodes_file(spec, asp.TLOG_LOOK, asp.C95) == decodes / "tlog.look.c95.jsonl.gz"
    assert ag.round_decodes_file(spec, asp.TLOG_LOOK, am.BASE) == (
        decodes / "tlog.look.base.jsonl.gz")
    assert ag.round_decodes_file(spec, asp.TLOG_LOOK, "r2") == (
        decodes / "tlog.look.r2_candidate_ema.jsonl.gz")
    for model in (asp.C95, am.BASE, "r2"):
        assert ag.round_decodes_file(spec, asp.TLOG_LOOK, model).exists()


@pytest.mark.parametrize("set_name", NON_TLOG_SETS)
def test_the_non_tlog_decodes_are_where_round_2_read_them(set_name):
    spec = asp.CONSONANT3
    for model in (asp.C95, am.BASE):
        assert ag.round_decodes_file(spec, set_name, model) == (
            ag.round_decodes_file(asp.CONSONANT2, set_name, model))
    r2 = ag.round_decodes_file(spec, set_name, "r2")
    assert r2 == ROOT2 / "decodes" / f"{set_name}.r2_candidate_ema.jsonl.gz"
    assert r2.exists()


def test_round_2_s_own_decode_sources_are_unchanged():
    assert ag.round_decodes_file(asp.CONSONANT2, asp.TLOG_LOOK, asp.C178) == (
        ROOT2 / "decodes" / "tlog.look.candidate_ema.jsonl.gz")
    assert (asp.TLOG_LOOK, "r2") not in asp.CONSONANT2.decode_sources
