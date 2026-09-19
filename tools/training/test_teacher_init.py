"""Tests for the teacher->student transplant.

The valuable tests here build a *real* pair of Wav2Vec2-BERT models at toy widths and run the
transplant over them on CPU. Every rule in :func:`training.teacher_init.transplant` is a slice
of a specific tensor, and the failure mode is not a crash -- it is a plausible model that
computes something the teacher does not. Three classes of that were live during development
and each has a test below: a LayerNorm whose module is named ``layer_norm`` rather than
``*_layer_norm`` and so matched no rule; a GLU pointwise convolution whose value and gate
halves were selected independently, pairing each kept channel with a stranger's gate; and a
norm copied coordinate-wise without correcting for normalising over 384 dimensions instead of
1024.
"""

from __future__ import annotations

import pytest
import torch

from training.distill_student import StudentSpec
from training.teacher_init import (
    CalibrationStats,
    Selection,
    _glu_rows,
    choose,
    collect_importance,
    collect_layernorm_moments,
    layernorm_correction,
    select_heads,
    select_top,
    transplant,
)

HEAD_DIM = 8
TEACHER_HIDDEN = 32
TEACHER_HEADS = 4
STUDENT_HIDDEN = 16
STUDENT_HEADS = 2
INTERMEDIATE_TEACHER = 64
INTERMEDIATE_STUDENT = 32
FEATURE_DIM = 10
FRAMES = 12


def _model(hidden, heads, intermediate, position_embeddings_type):
    from tadabur.muaalem import (
        Wav2Vec2BertForMultilevelCTC,
        Wav2Vec2BertForMultilevelCTCConfig,
    )

    config = Wav2Vec2BertForMultilevelCTCConfig(
        level_to_vocab_size={"phonemes": 5},
        level_to_loss_weight={"phonemes": 1.0},
        hidden_size=hidden,
        num_hidden_layers=2,
        num_attention_heads=heads,
        intermediate_size=intermediate,
        feature_projection_input_dim=FEATURE_DIM,
        position_embeddings_type=position_embeddings_type,
        add_adapter=True,
        num_adapter_layers=1,
        adapter_kernel_size=3,
        adapter_stride=2,
        output_hidden_size=hidden,
        conv_depthwise_kernel_size=7,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        final_dropout=0.0,
        conformer_conv_dropout=0.0,
        apply_spec_augment=False,
        mask_time_prob=0.0,
        mask_time_min_masks=0,
        mask_feature_prob=0.0,
        mask_feature_min_masks=0,
    )
    model = Wav2Vec2BertForMultilevelCTC(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def pair():
    torch.manual_seed(0)
    teacher = _model(TEACHER_HIDDEN, TEACHER_HEADS, INTERMEDIATE_TEACHER, "relative_key")
    student = _model(STUDENT_HIDDEN, STUDENT_HEADS, INTERMEDIATE_STUDENT, "rotary")
    features = [torch.randn(2, FRAMES, FEATURE_DIM) for _ in range(2)]
    return teacher, student, features


@pytest.fixture(scope="module")
def spec():
    return StudentSpec(
        "toy",
        hidden_size=STUDENT_HIDDEN,
        intermediate_size=INTERMEDIATE_STUDENT,
        num_attention_heads=STUDENT_HEADS,
        num_hidden_layers=2,
        position_embeddings_type="rotary",
    )


def test_select_top_returns_indices_in_ascending_order():
    scores = torch.tensor([0.1, 5.0, 0.2, 4.0, 3.0])
    assert select_top(scores, 3).tolist() == [1, 3, 4]
    with pytest.raises(ValueError):
        select_top(scores, 9)


def test_select_heads_keeps_each_head_whole_and_contiguous():
    scores = torch.tensor([0.0, 9.0, 0.0, 7.0])
    indices = select_heads(scores, 2, head_dim=8).tolist()
    # Heads 1 and 3 -> channels 8..15 and 24..31, whole and in order.
    assert indices == list(range(8, 16)) + list(range(24, 32))


def test_glu_rows_pair_each_value_channel_with_its_own_gate():
    residual = torch.tensor([0, 3, 5])
    assert _glu_rows(residual, 8).tolist() == [0, 3, 5, 8, 11, 13]


def test_layernorm_correction_reproduces_the_teacher_on_kept_coordinates():
    torch.manual_seed(1)
    x = torch.randn(64, TEACHER_HIDDEN) * 2.0 + 0.5
    gamma, beta = torch.randn(TEACHER_HIDDEN), torch.randn(TEACHER_HIDDEN)
    selected = select_top(x.abs().mean(dim=0), STUDENT_HIDDEN)

    teacher_out = torch.nn.functional.layer_norm(x, (TEACHER_HIDDEN,)) * gamma + beta
    moments = (
        float(x.mean(dim=-1).mean()),
        float(x.var(dim=-1, unbiased=False).sqrt().mean()),
        float(x[:, selected].mean(dim=-1).mean()),
        float(x[:, selected].var(dim=-1, unbiased=False).sqrt().mean()),
    )
    new_gamma, new_beta = layernorm_correction(gamma, beta, selected, moments)
    student_out = (
        torch.nn.functional.layer_norm(x[:, selected], (STUDENT_HIDDEN,)) * new_gamma
        + new_beta
    )

    corrected_error = (student_out - teacher_out[:, selected]).abs().mean()
    naive_out = (
        torch.nn.functional.layer_norm(x[:, selected], (STUDENT_HIDDEN,)) * gamma[selected]
        + beta[selected]
    )
    naive_error = (naive_out - teacher_out[:, selected]).abs().mean()
    # The correction must actually buy something -- an uncorrected copy is mis-scaled by the
    # ratio of the two standard deviations, which is exactly what this measures.
    assert corrected_error < naive_error


def test_transplant_copies_every_student_tensor_it_has_a_rule_for(pair, spec):
    teacher, student, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    moments = collect_layernorm_moments(teacher, features, selection.residual)
    report = transplant(student, teacher, selection, moments, qk_mode="copy")

    # Nothing may be left behind silently: the only tensors not copied under --qk copy are
    # ones with no teacher counterpart at all.
    unexplained = [
        name for name in report["left_random_names"] if "no teacher counterpart" not in name
    ]
    assert unexplained == [], unexplained
    assert report["copied"] > 40


def test_every_layernorm_site_is_corrected_not_merely_sliced(pair, spec):
    """The conv module's norm is named ``layer_norm``, not ``*_layer_norm``.

    A suffix match on ``_layer_norm`` misses it in every encoder layer, which leaves those
    norms at random init inside an otherwise transplanted network -- and nothing about the
    resulting model looks wrong until it trains badly.
    """
    teacher, _, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    moments = collect_layernorm_moments(teacher, features, selection.residual)

    hidden_norms = {
        name
        for name, module in teacher.named_modules()
        if isinstance(module, torch.nn.LayerNorm)
        and tuple(module.normalized_shape) == (TEACHER_HIDDEN,)
    }
    assert hidden_norms <= set(moments)
    assert any("conv_module.layer_norm" in name for name in moments)


def test_random_query_key_mode_leaves_exactly_those_tensors_alone(pair, spec):
    teacher, student, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    moments = collect_layernorm_moments(teacher, features, selection.residual)
    report = transplant(student, teacher, selection, moments, qk_mode="random")

    left = [name for name in report["left_random_names"] if "--qk random" in name]
    assert left, "expected encoder q/k to be reported as left at init"
    assert all(".linear_q." in name or ".linear_k." in name for name in left)
    # The adapter's attention carries no positional embedding at all, so it always copies.
    assert not any(".adapter." in name for name in left)


def test_damped_query_key_is_the_copy_scaled(pair, spec):
    teacher, student, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    moments = collect_layernorm_moments(teacher, features, selection.residual)

    transplant(student, teacher, selection, moments, qk_mode="copy")
    copied = student.state_dict()[
        "wav2vec2_bert.encoder.layers.0.self_attn.linear_q.weight"
    ].clone()
    transplant(student, teacher, selection, moments, qk_mode="damp", qk_scale=0.25)
    damped = student.state_dict()[
        "wav2vec2_bert.encoder.layers.0.self_attn.linear_q.weight"
    ]
    assert torch.allclose(damped, copied * 0.25, atol=1e-6)


def test_the_phoneme_head_is_copied_exactly_on_kept_channels(pair, spec):
    teacher, student, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    moments = collect_layernorm_moments(teacher, features, selection.residual)
    transplant(student, teacher, selection, moments, qk_mode="copy")

    teacher_head = teacher.state_dict()["level_to_lm_head.phonemes.weight"]
    student_head = student.state_dict()["level_to_lm_head.phonemes.weight"]
    assert torch.allclose(
        student_head, teacher_head.index_select(1, selection.residual), atol=1e-6
    )
    assert torch.allclose(
        student.state_dict()["level_to_lm_head.phonemes.bias"],
        teacher.state_dict()["level_to_lm_head.phonemes.bias"],
        atol=1e-6,
    )


def test_transplanted_student_still_honours_its_shape_contract(pair, spec):
    teacher, student, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    moments = collect_layernorm_moments(teacher, features, selection.residual)
    transplant(student, teacher, selection, moments, qk_mode="copy")

    with torch.no_grad():
        out = student(features[0], return_dict=True)["logits"]["phonemes"]
    assert out.shape == (2, FRAMES // 2, 5)
    assert torch.isfinite(out).all()


def test_choose_refuses_a_student_whose_head_dim_does_not_match(spec):
    stats = CalibrationStats(residual_importance=torch.rand(TEACHER_HIDDEN))
    mismatched = StudentSpec(
        "bad",
        hidden_size=STUDENT_HIDDEN,
        intermediate_size=INTERMEDIATE_STUDENT,
        num_attention_heads=4,  # head_dim 4, teacher's is 8
        num_hidden_layers=2,
    )
    with pytest.raises(ValueError, match="head selection"):
        choose(mismatched, stats, HEAD_DIM)


def test_selection_is_reported_as_a_concentration_number():
    stats = CalibrationStats(residual_importance=torch.tensor([10.0, 1.0, 1.0, 1.0]))
    assert stats.captured_variance_share(torch.tensor([0])) == pytest.approx(10 / 13)
