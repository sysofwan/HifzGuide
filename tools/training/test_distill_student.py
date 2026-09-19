"""Tests for the thin-student sizing ladder.

The analytic path is torch-free and always runs. The instantiation tests need torch and
the vendored Muaalem package, so they are skipped when those are unavailable -- the CI
box for this repo is the CUDA machine, but the sizing arithmetic must stay checkable
anywhere.
"""

from __future__ import annotations

import pytest

from training import distill_student as ds


# --- The shape contract is what makes a student a drop-in at all ---------------------


def test_presets_all_keep_teacher_depth():
    """Depth is the axis we do not cut -- see the module docstring and section 6."""
    for spec in ds.PRESETS.values():
        assert spec.num_hidden_layers == ds.TEACHER_NUM_LAYERS


def test_presets_hold_the_teacher_ffn_ratio():
    """One axis varies, so the agreement-vs-size curve is interpretable."""
    teacher_ratio = ds.TEACHER_INTERMEDIATE_SIZE / ds.TEACHER_HIDDEN_SIZE
    for spec in ds.PRESETS.values():
        assert spec.intermediate_size / spec.hidden_size == teacher_ratio


def test_head_divides_width():
    for spec in ds.PRESETS.values():
        assert spec.hidden_size % spec.num_attention_heads == 0


def test_rejects_indivisible_width():
    with pytest.raises(ValueError, match="not divisible"):
        ds.StudentSpec("bad", hidden_size=384, intermediate_size=1536, num_attention_heads=7)


# --- The size model is calibrated against a measurement, so pin it to that ----------


def test_six_bit_model_reproduces_the_measured_teacher():
    """504 MB measured at 6-bit (section 3.3). The size model must land near it.

    Checked at 6%: the analytic parameter count runs ~3.5% under the instantiated count,
    and the teacher pays per-package metadata six times over.
    """
    teacher = ds.teacher_sizing()
    assert teacher.size_6bit_mb == pytest.approx(504.0, rel=0.06)


def test_six_bit_model_reproduces_the_measured_h384_exports():
    """Both measured exports, which differ only in position-embedding type.

    rotary measured 61.7 MB over 85.55M graph values; relative_key measured 130.3 MB over
    181.30M. Predicting each from its own graph-constant count pins the bytes/value.
    """
    assert ds.palettized_6bit_mb(85_545_963) == pytest.approx(61.7, rel=0.02)
    assert ds.palettized_6bit_mb(181_300_000) == pytest.approx(130.3, rel=0.02)
    assert ds.PRESETS["h384"].position_embeddings_type == "rotary"


def test_relative_key_adds_position_constants_and_rotary_does_not():
    """The finding that decides the chunk count: ~96M values, independent of width."""
    rotary = ds.PRESETS["h384"]
    relative = ds.replace(rotary, position_embeddings_type=ds.TEACHER_POSITION_EMBEDDINGS)

    assert ds.estimate_position_constant_params(rotary) == 0
    assert ds.estimate_position_constant_params(relative) == pytest.approx(
        96_000_000, rel=0.01
    )

    # Independent of hidden_size -- the whole reason it dominates a thin student.
    narrow = ds.replace(
        relative, name="n", hidden_size=256, intermediate_size=1024, num_attention_heads=4
    )
    assert ds.estimate_position_constant_params(narrow) == (
        ds.estimate_position_constant_params(relative)
    )


def test_position_constants_flip_h384_across_the_chunk_budget():
    """h384 is one chunk with rotary and two with relative_key. Measured both ways."""
    rotary = ds.size_student(ds.PRESETS["h384"])
    relative = ds.size_student(
        ds.replace(
            ds.PRESETS["h384"], position_embeddings_type=ds.TEACHER_POSITION_EMBEDDINGS
        )
    )
    assert rotary.chunks == 1
    assert relative.chunks == 2


def test_six_bit_hits_its_nominal_ratio():
    """Measured 0.753 and 0.756 bytes/value across two exports -- nominal 6/8."""
    assert ds.PALETTIZED_6BIT_RATIO == pytest.approx(0.75)
    assert ds.BYTES_PER_GRAPH_VALUE_6BIT == pytest.approx(0.75)


def test_chunking_follows_the_demonstrated_ceiling():
    """A student at or under the proven 99 MB chunk is one chunk; past it, more."""
    one_chunk = int(90 * 1024 * 1024 / ds.BYTES_PER_GRAPH_VALUE_6BIT)
    assert ds.estimated_chunks(one_chunk) == 1

    two_chunk = int(120 * 1024 * 1024 / ds.BYTES_PER_GRAPH_VALUE_6BIT)
    assert ds.estimated_chunks(two_chunk) == 2


def test_h384_is_a_single_chunk_and_h512_is_not():
    """The load-bearing sizing claim: h384 fits one ANE chunk, h512 needs two."""
    h384 = ds.size_student(ds.PRESETS["h384"])
    h512 = ds.size_student(ds.PRESETS["h512"])
    assert h384.chunks == 1
    assert h384.size_6bit_mb < ds.PROVEN_MAX_CHUNK_MB
    assert h512.chunks == 2


def test_h448_is_the_widest_single_chunk_preset():
    """The band a 256/384/512 ladder skips.

    h384 uses only ~63% of the demonstrated 99 MB chunk budget, so there is real width to
    be had for free. h448 must stay one chunk while carrying meaningfully more capacity
    than h384; if it ever crosses the ceiling the ladder needs rethinking, not silence.
    """
    h384 = ds.size_student(ds.PRESETS["h384"])
    h448 = ds.size_student(ds.PRESETS["h448"])

    assert h448.chunks == 1
    assert h448.size_6bit_mb < ds.PROVEN_MAX_CHUNK_MB
    assert h448.num_params > h384.num_params * 1.3
    # ...and it is genuinely the widest that fits: h512 does not.
    assert ds.size_student(ds.PRESETS["h512"]).chunks == 2


def test_flop_ratio_is_quadratic_in_width_at_equal_depth():
    h512 = ds.PRESETS["h512"]
    assert h512.flop_ratio == pytest.approx(0.25)     # (512/1024)^2
    assert ds.PRESETS["h256"].flop_ratio == pytest.approx(0.0625)


def test_flop_ratio_accounts_for_depth():
    half_depth = ds.replace(ds.PRESETS["h512"], num_hidden_layers=12)
    assert half_depth.flop_ratio == pytest.approx(0.125)


# --- Instantiation: the analytic table must track the real model --------------------

# NOT `pytest.importorskip` at module level: that raises Skipped during *import*, so pytest
# skips the whole file -- including the torch-free sizing tests above, which are the ones
# guarding the load-bearing "h384 is one chunk" claim and are supposed to run anywhere.
# Verified: a mid-module importorskip in a 2-test file reports "1 skipped", not "1 passed,
# 1 skipped".
try:
    import torch
    import tadabur.muaalem  # noqa: F401

    HAS_TORCH = True
except Exception:  # pragma: no cover - depends on the box
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="instantiation tests need torch + the vendored Muaalem package"
)


@needs_torch
@pytest.mark.parametrize("name", sorted(ds.PRESETS))
def test_student_honours_the_deployed_shape_contract(name):
    """(1, 250, 160) -> (1, 125, 43), or it cannot replace the teacher."""
    model = ds.build_student(ds.PRESETS[name])
    frames, classes = ds.verify_shape_contract(model)
    assert (frames, classes) == (ds.DEPLOYED_LOGIT_FRAMES, ds.NUM_PHONEME_CLASSES)


@needs_torch
@pytest.mark.parametrize("name", sorted(ds.PRESETS))
def test_analytic_estimate_tracks_the_real_parameter_count(name):
    """The table is a planning tool; it may under-count but must not drift far.

    It runs consistently ~3.5% low. The cause is the layer-norm/bias approximation in
    `_conformer_layer_params`, NOT relative-position embeddings -- every preset is rotary
    and has none. Anything past 6% means the architecture changed under us.
    """
    spec = ds.PRESETS[name]
    model = ds.build_student(spec)
    measured = ds.count_parameters(model)
    assert ds.estimate_params(spec) == pytest.approx(measured, rel=0.06)


@needs_torch
@pytest.mark.parametrize("name", sorted(ds.PRESETS))
def test_student_carries_only_the_phoneme_head(name):
    """The 10 sifat heads must cost no parameters and take no gradient."""
    model = ds.build_student(ds.PRESETS[name])
    assert set(model.level_to_lm_head.keys()) == {ds.PHONEME_LEVEL}


# --- The student must be deterministic in train() mode ------------------------------
#
# This is the single most expensive bug found in this work, so it gets the most direct
# possible test. Distillation asks the student to reproduce a teacher that runs in eval()
# on clean input. Any train-time stochasticity therefore makes the target unlearnable at
# the perturbed positions and the input different every step: the student could not overfit
# even 32 fixed windows, and the full run plateaued for 4000 steps looking exactly like a
# model that had converged.


def test_no_stochastic_regularisation_is_configured():
    """Every dropout and masking knob must be off, whatever the class defaults are.

    Two of these were missed on the first pass because they do not look like the others:
    `conformer_conv_dropout` and `final_dropout` default to 0.1 and are absent from the
    obvious list of four. Enumerating the config rather than naming fields catches the next
    one too.
    """
    config = ds.build_student_config(ds.PRESETS["h384"])
    live = {
        name: value
        for name, value in vars(config).items()
        if ("dropout" in name or name == "layerdrop") and value
    }
    assert not live, f"stochastic regularisation still enabled: {live}"


def test_spec_augment_is_disabled_by_its_own_flag():
    """Zeroing the probabilities is NOT enough to disable SpecAugment.

    transformers computes `num_masked_span = max(num_masked_span, min_masks)`, and the
    teacher config carries `mask_time_min_masks=2`, so `mask_time_prob=0.0` alone still
    masks two spans of 10 frames per sequence in train() mode.
    """
    config = ds.build_student_config(ds.PRESETS["h384"])
    assert config.apply_spec_augment is False
    assert config.mask_time_min_masks == 0
    assert config.mask_feature_min_masks == 0


@needs_torch
@pytest.mark.parametrize("name", sorted(ds.PRESETS))
def test_train_mode_forward_is_deterministic(name):
    """The property all of the above exists to guarantee, asserted end to end."""
    model = ds.build_student(ds.PRESETS[name])
    model.train()
    features = torch.zeros(2, ds.DEPLOYED_FEATURE_FRAMES, ds.FEATURE_INPUT_DIM)
    with torch.no_grad():
        first = model(features, return_dict=True)["logits"][ds.PHONEME_LEVEL]
        second = model(features, return_dict=True)["logits"][ds.PHONEME_LEVEL]
    assert torch.equal(first, second)


@needs_torch
def test_train_and_eval_modes_agree():
    """A train/eval gap is the signature of exactly this class of bug."""
    model = ds.build_student(ds.PRESETS["h256"])
    features = torch.zeros(2, ds.DEPLOYED_FEATURE_FRAMES, ds.FEATURE_INPUT_DIM)
    model.train()
    with torch.no_grad():
        train_out = model(features, return_dict=True)["logits"][ds.PHONEME_LEVEL]
    model.eval()
    with torch.no_grad():
        eval_out = model(features, return_dict=True)["logits"][ds.PHONEME_LEVEL]
    assert torch.equal(train_out, eval_out)
