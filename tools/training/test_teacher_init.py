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
def teacher_and_features():
    """Expensive and read-only, so shared."""
    torch.manual_seed(0)
    teacher = _model(TEACHER_HIDDEN, TEACHER_HEADS, INTERMEDIATE_TEACHER, "relative_key")
    return teacher, [torch.randn(2, FRAMES, FEATURE_DIM) for _ in range(2)]


@pytest.fixture
def pair(teacher_and_features):
    """A **fresh** student per test.

    It used to be module-scoped, so every `transplant` left the model in whatever state the
    previous test produced, in whatever order pytest happened to run them. That cost a real
    assertion: the `--qk random` test could only check the report, because by the time it ran
    the query/key weights were already the copied teacher weights from an earlier arm.
    """
    torch.manual_seed(1)
    teacher, features = teacher_and_features
    return teacher, _model(STUDENT_HIDDEN, STUDENT_HEADS, INTERMEDIATE_STUDENT, "rotary"), features


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
    moments = collect_layernorm_moments(teacher, features, selection)
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
    moments = collect_layernorm_moments(teacher, features, selection)

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
    moments = collect_layernorm_moments(teacher, features, selection)

    # Snapshot before the transplant. Only possible because the student is now a fresh
    # per-test fixture; under the old module-scoped one it already held a previous arm's
    # copied weights and this assertion could not be written at all.
    before = {k: v.clone() for k, v in student.state_dict().items()}
    report = transplant(student, teacher, selection, moments, qk_mode="random")
    after = student.state_dict()

    left = [name for name in report["left_random_names"] if "--qk random" in name]
    assert left, "expected encoder q/k to be reported as left at init"
    assert all(".linear_q." in name or ".linear_k." in name for name in left)
    # The adapter's attention carries no positional embedding at all, so it always copies.
    assert not any(".adapter." in name for name in left)

    for entry in left:
        name = entry.split(" (")[0]
        assert torch.equal(after[name], before[name]), f"{name} was modified"
    # ... and value, which sits in the same attention block, WAS copied. That contrast is
    # the point: --qk random is selective, not a no-op on the whole module.
    value = "wav2vec2_bert.encoder.layers.0.self_attn.linear_v.weight"
    assert not torch.equal(after[value], before[value])


def test_damped_query_key_is_the_copy_scaled(pair, spec):
    teacher, student, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    moments = collect_layernorm_moments(teacher, features, selection)

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
    moments = collect_layernorm_moments(teacher, features, selection)
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
    moments = collect_layernorm_moments(teacher, features, selection)
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


def test_transplant_preserves_full_precision_from_an_fp32_source(pair, spec):
    """The copy must not be taken from the bf16 teacher the training loop runs.

    ``distill_train.load_teacher`` casts to bf16, which keeps 8 mantissa bits. A transplant
    read from it starts the student on rounded weights for no reason at all, and nothing
    about the resulting checkpoint says so.
    """
    teacher, student, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    moments = collect_layernorm_moments(teacher, features, selection)
    transplant(student, teacher, selection, moments, qk_mode="copy")

    exact = teacher.state_dict()["level_to_lm_head.phonemes.weight"].index_select(
        1, selection.residual
    )
    got = student.state_dict()["level_to_lm_head.phonemes.weight"]
    # Exact equality, not allclose: a straight index_select of an fp32 source is bit-identical,
    # and a bf16 round trip would not be.
    assert torch.equal(got, exact)


def test_no_calibration_forward_escapes_the_autocast_helper(pair, spec, monkeypatch):
    """A bf16 teacher on fp32 features raises inside the first Linear without autocast.

    Asserts the invariant by counting calls rather than by matching source text, so an
    ordinary refactor does not fail it and a new raw `teacher(...)` call does.
    """
    from training import teacher_init

    teacher, _, features = pair
    calls = []
    real = teacher_init._teacher_forward
    monkeypatch.setattr(
        teacher_init, "_teacher_forward",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    stats = teacher_init.collect_importance(teacher, features)
    assert len(calls) == len(features)

    selection = choose(spec, stats, HEAD_DIM)
    calls.clear()
    collect_layernorm_moments(teacher, features, selection)
    assert len(calls) == len(features)


def test_conv_internal_channels_are_selected_in_their_own_space(pair, spec):
    """The conv module's channels are not the residual stream, only the same width.

    A single global selection would keep an arbitrary subset of them, and the shapes would
    all still line up -- which is exactly why this needs a test rather than an assertion.
    """
    teacher, _, features = pair
    stats = collect_importance(teacher, features)
    assert stats.conv_importance, "no conv module was scored"
    selection = choose(spec, stats, HEAD_DIM)

    site = next(iter(selection.conv))
    assert site.endswith("conv_module.pointwise_conv2")
    assert len(selection.conv[site]) == STUDENT_HIDDEN
    # It would be a coincidence, not a design, for the two rankings to agree.
    assert not torch.equal(selection.conv[site], selection.residual)


def test_the_depthwise_norm_is_measured_over_the_conv_space_not_the_residual(pair, spec):
    teacher, _, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)

    site = "wav2vec2_bert.encoder.layers.0.conv_module.depthwise_layer_norm"
    consumer = "wav2vec2_bert.encoder.layers.0.conv_module.pointwise_conv2"
    assert torch.equal(selection.index_for_layernorm(site), selection.conv[consumer])
    other = "wav2vec2_bert.encoder.layers.0.conv_module.layer_norm"
    assert torch.equal(selection.index_for_layernorm(other), selection.residual)


def test_the_glu_halves_of_the_conv_are_taken_from_the_conv_selection(pair, spec):
    teacher, student, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    moments = collect_layernorm_moments(teacher, features, selection)
    transplant(student, teacher, selection, moments, qk_mode="copy")

    name = "wav2vec2_bert.encoder.layers.0.conv_module.pointwise_conv1.weight"
    internal = selection.conv[
        "wav2vec2_bert.encoder.layers.0.conv_module.pointwise_conv2"
    ]
    expected = (
        teacher.state_dict()[name]
        .index_select(0, _glu_rows(internal, TEACHER_HIDDEN))
        .index_select(1, selection.residual)
    )
    assert torch.equal(student.state_dict()[name], expected)


def test_the_adapter_convolutions_still_index_by_the_residual_stream(pair, spec):
    """Their GLU output IS the residual -- one branch is added straight back to it."""
    teacher, student, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    moments = collect_layernorm_moments(teacher, features, selection)
    transplant(student, teacher, selection, moments, qk_mode="copy")

    name = "wav2vec2_bert.adapter.layers.0.residual_conv.weight"
    expected = (
        teacher.state_dict()[name]
        .index_select(0, _glu_rows(selection.residual, TEACHER_HIDDEN))
        .index_select(1, selection.residual)
    )
    assert torch.equal(student.state_dict()[name], expected)


def test_branch_gains_are_fitted_per_branch_and_stay_sane(pair, spec):
    """One gain per branch, and none of them a rescaling that would amplify noise.

    With random weights the dropped units contribute orthogonally to the kept ones, so the
    least-squares optimum is ~1.0 -- NOT the 16/6 the "the branch is quieter now" story
    predicts. That is the point of fitting it rather than assuming it.
    """
    from training.teacher_init import collect_branch_gains

    teacher, _, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    gains = collect_branch_gains(teacher, features, selection)

    assert gains, "no branch gain was fitted"
    # One per attention output, FFN output and conv output, across both layers and the adapter.
    assert set(gains) <= set(selection.heads) | set(selection.ffn) | set(selection.conv)
    assert all(0.25 < value < 4.0 for value in gains.values()), gains


def test_the_branch_gain_is_the_least_squares_optimum(pair, spec):
    """Recompute one site's gain directly and check the hook agrees.

    The closed form is easy to get subtly wrong -- restricted to the wrong channels, fitted
    with the bias included, or accumulated per batch instead of pooled -- and every version
    of it produces a plausible scalar.
    """
    from training.teacher_init import collect_branch_gains

    teacher, _, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    gains = collect_branch_gains(teacher, features, selection)

    site = "wav2vec2_bert.encoder.layers.0.self_attn.linear_out"
    captured = []
    module = dict(teacher.named_modules())[site]
    handle = module.register_forward_pre_hook(lambda m, a: captured.append(a[0].detach()))
    try:
        with torch.no_grad():
            for batch in features:
                teacher(batch, return_dict=True)
    finally:
        handle.remove()

    weight = module.weight.float()
    heads = selection.heads[site]
    numerator = denominator = 0.0
    for activation in captured:
        flat = activation.reshape(-1, activation.shape[-1]).float()
        kept = (flat.index_select(-1, heads) @ weight.index_select(1, heads).T).index_select(
            -1, selection.residual
        )
        full = (flat @ weight.T).index_select(-1, selection.residual)
        numerator += float((kept * full).sum())
        denominator += float((kept * kept).sum())

    assert gains[site] == pytest.approx(numerator / denominator, rel=1e-4)


def test_the_branch_gain_is_folded_into_the_copied_output_weight(pair, spec):
    from training.teacher_init import collect_branch_gains

    teacher, student, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    moments = collect_layernorm_moments(teacher, features, selection)
    gains = collect_branch_gains(teacher, features, selection)

    transplant(student, teacher, selection, moments, qk_mode="copy", branch_gains=None)
    plain = student.state_dict()[
        "wav2vec2_bert.encoder.layers.0.self_attn.linear_out.weight"
    ].clone()
    transplant(student, teacher, selection, moments, qk_mode="copy", branch_gains=gains)
    scaled = student.state_dict()[
        "wav2vec2_bert.encoder.layers.0.self_attn.linear_out.weight"
    ]
    gain = gains["wav2vec2_bert.encoder.layers.0.self_attn.linear_out"]
    assert torch.allclose(scaled, plain * gain, atol=1e-6)


def test_the_bias_is_not_rescaled_with_the_branch(pair, spec):
    """The bias is added once, not summed over units, so it is not what went missing."""
    from training.teacher_init import collect_branch_gains

    teacher, student, features = pair
    stats = collect_importance(teacher, features)
    selection = choose(spec, stats, HEAD_DIM)
    moments = collect_layernorm_moments(teacher, features, selection)
    gains = collect_branch_gains(teacher, features, selection)
    transplant(student, teacher, selection, moments, qk_mode="copy", branch_gains=gains)

    name = "wav2vec2_bert.encoder.layers.0.self_attn.linear_out.bias"
    expected = teacher.state_dict()[name].index_select(0, selection.residual)
    assert torch.equal(student.state_dict()[name], expected)


def test_the_two_aggregations_rank_channels_differently(pair, spec):
    """Otherwise the flag is decoration and the Minitron ablation cannot be reproduced."""
    teacher, _, features = pair
    l2 = collect_importance(teacher, features, "l2_over_examples").residual_importance
    mean = collect_importance(teacher, features, "mean_over_examples").residual_importance

    assert l2.shape == mean.shape
    assert not torch.allclose(l2, mean)
    # L2 over examples is >= the mean over examples, elementwise, by Cauchy-Schwarz once the
    # per-example scores are non-negative -- which they are, being means of absolute values.
    assert (l2 >= 0).all() and (mean >= 0).all()


def test_an_unknown_aggregation_is_refused(pair):
    teacher, _, features = pair
    with pytest.raises(ValueError, match="unknown aggregation"):
        collect_importance(teacher, features, "whatever_scores_best")


def test_the_default_aggregation_is_the_one_with_evidence():
    from training.teacher_init import DEFAULT_AGGREGATION

    assert DEFAULT_AGGREGATION == "l2_over_examples"
