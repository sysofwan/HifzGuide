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


def test_flop_ratio_is_quadratic_in_width_at_equal_depth():
    h512 = ds.PRESETS["h512"]
    assert h512.flop_ratio == pytest.approx(0.25)     # (512/1024)^2
    assert ds.PRESETS["h256"].flop_ratio == pytest.approx(0.0625)


def test_flop_ratio_accounts_for_depth():
    half_depth = ds.replace(ds.PRESETS["h512"], num_hidden_layers=12)
    assert half_depth.flop_ratio == pytest.approx(0.125)


# --- Instantiation: the analytic table must track the real model --------------------

torch = pytest.importorskip("torch", reason="instantiation tests need torch")
pytest.importorskip("tadabur.muaalem", reason="needs the vendored Muaalem package")


@pytest.mark.parametrize("name", sorted(ds.PRESETS))
def test_student_honours_the_deployed_shape_contract(name):
    """(1, 250, 160) -> (1, 125, 43), or it cannot replace the teacher."""
    model = ds.build_student(ds.PRESETS[name])
    frames, classes = ds.verify_shape_contract(model)
    assert (frames, classes) == (ds.DEPLOYED_LOGIT_FRAMES, ds.NUM_PHONEME_CLASSES)


@pytest.mark.parametrize("name", sorted(ds.PRESETS))
def test_analytic_estimate_tracks_the_real_parameter_count(name):
    """The table is a planning tool; it may under-count but must not drift far.

    The known gap is the relative-position embeddings, which the estimate omits; it runs
    consistently ~3.5% low. Anything past 6% means the architecture changed under us.
    """
    spec = ds.PRESETS[name]
    model = ds.build_student(spec)
    measured = ds.count_parameters(model)
    assert ds.estimate_params(spec) == pytest.approx(measured, rel=0.06)


@pytest.mark.parametrize("name", sorted(ds.PRESETS))
def test_student_carries_only_the_phoneme_head(name):
    """The 10 sifat heads must cost no parameters and take no gradient."""
    model = ds.build_student(ds.PRESETS[name])
    assert set(model.level_to_lm_head.keys()) == {ds.PHONEME_LEVEL}
