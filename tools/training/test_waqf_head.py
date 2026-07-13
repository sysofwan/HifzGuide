"""Tests for the waqf head + detached joint loss (ADR-0004, issue #28).

The load-bearing claim is **isolation**: training the waqf head must produce zero
backbone gradient, so the joint run stays comparable to a phoneme-only whole-clip run
(ADR-0004's ablation ladder). :func:`test_waqf_loss_produces_zero_backbone_gradient`
proves the stop-gradient through a real (tiny) Muaalem backbone. The rest pin the
pause-weighting, the KL numerics, and the no-pause-collapse diagnostic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tadabur.muaalem.configuration_multi_level_ctc import (
    Wav2Vec2BertForMultilevelCTCConfig,
)
from tadabur.muaalem.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from training.waqf_head import (
    JointLoss,
    WaqfHead,
    WaqfJointModel,
    frame_mask_from_lengths,
    joint_loss,
    pause_collapse_diagnostic,
    pause_frame_weights,
    phoneme_ctc_loss,
    waqf_distillation_loss,
)


def _tiny_model(levels=None):
    torch.manual_seed(0)
    config = Wav2Vec2BertForMultilevelCTCConfig(
        level_to_vocab_size=levels or {"phonemes": 43, "hams_or_jahr": 3},
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        feature_projection_input_dim=160,
        add_adapter=True,
        output_hidden_size=16,
        num_adapter_layers=1,
        adapter_kernel_size=3,
        adapter_stride=2,
    )
    return Wav2Vec2BertForMultilevelCTC(config)


# --- WaqfHead: stop-gradient at the head boundary ----------------------------


def test_head_detaches_its_input():
    head = WaqfHead(feature_dim=8)
    features = torch.randn(2, 5, 8, requires_grad=True)
    head(features).sum().backward()
    # The head trains, but nothing upstream of it receives a gradient.
    assert features.grad is None
    assert head.classifier.weight.grad is not None


def test_head_emits_one_logit_per_frame():
    head = WaqfHead(feature_dim=8)
    assert head(torch.randn(3, 7, 8)).shape == (3, 7)


def test_mlp_fallback_head_shape():
    head = WaqfHead(feature_dim=8, hidden_dim=16)
    assert head(torch.randn(2, 4, 8)).shape == (2, 4)


# --- WaqfJointModel: one forward, phoneme + waqf, sifat dropped ---------------


def test_joint_model_emits_aligned_phoneme_and_waqf_lattices():
    model = WaqfJointModel(_tiny_model())
    out = model(torch.randn(2, 20, 160))
    assert out.phoneme_logits.shape == (2, 10, 43)  # 20 feature frames → 10 student
    assert out.silence_logits.shape == (2, 10)
    assert out.student_lengths.tolist() == [10, 10]


def test_sifat_heads_are_dropped_from_the_graph():
    # The joint forward + full loss must never touch a sifat head — it takes no gradient.
    model = WaqfJointModel(_tiny_model())
    out = model(torch.randn(1, 20, 160))
    labels = torch.randint(0, 43, (1, 4))
    target = torch.zeros(1, 10)
    joint_loss(out, labels, target, model.muaalem.config).total.backward()

    sifat_head = model.muaalem.level_to_lm_head["hams_or_jahr"]
    assert sifat_head.weight.grad is None


# --- The acceptance criterion: zero backbone gradient from the waqf loss ------


def _backbone_params(model):
    return {
        name: p
        for name, p in model.muaalem.named_parameters()
        if p.requires_grad
    }


def test_waqf_loss_produces_zero_backbone_gradient():
    # The isolation guarantee: with only the waqf KL in the loss, every backbone (and
    # phoneme-head) parameter must end with no gradient, while the waqf head learns.
    model = WaqfJointModel(_tiny_model())
    out = model(torch.randn(2, 20, 160))
    target = torch.rand(2, 10)

    waqf_distillation_loss(out.silence_logits, target).backward()

    for name, param in _backbone_params(model).items():
        assert param.grad is None or torch.count_nonzero(param.grad) == 0, name
    assert any(p.grad is not None for p in model.waqf_head.parameters())


def test_phoneme_ctc_alone_does_reach_the_backbone():
    # Control: the phoneme objective DOES flow to the backbone, so the zero-gradient
    # result above is the stop-gradient, not a dead graph.
    model = WaqfJointModel(_tiny_model())
    out = model(torch.randn(2, 20, 160))
    labels = torch.randint(0, 43, (2, 4))
    phoneme_ctc_loss(
        out.phoneme_logits, labels, out.student_lengths, model.muaalem.config
    ).backward()
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in _backbone_params(model).values()
    )


def test_joint_loss_backbone_gradient_equals_phoneme_only():
    # ADR-0004's (2)→(3) go/no-go, in miniature: adding the detached waqf term must not
    # change a single backbone gradient versus the phoneme-only loss on the same forward.
    model = WaqfJointModel(_tiny_model()).eval()  # eval → no dropout/spec-augment RNG
    features = torch.randn(2, 20, 160)
    labels = torch.randint(0, 43, (2, 4))
    target = torch.rand(2, 10)
    config = model.muaalem.config

    out = model(features)
    phoneme_ctc_loss(out.phoneme_logits, labels, out.student_lengths, config).backward()
    # Some backbone params (e.g. the spec-augment mask embedding in eval) never take a
    # gradient; compare only the ones the phoneme objective actually touches.
    phoneme_only = {
        n: p.grad.clone()
        for n, p in _backbone_params(model).items()
        if p.grad is not None
    }

    model.zero_grad(set_to_none=True)
    out = model(features)
    joint_loss(out, labels, target, config).total.backward()
    joint = {n: p.grad for n, p in _backbone_params(model).items()}

    assert phoneme_only  # the objective really did reach the backbone
    for name, grad in phoneme_only.items():
        torch.testing.assert_close(joint[name], grad)


# --- pause_frame_weights: silence up-weighted, boundary interpolated ----------


def test_pause_weights_interpolate_from_speech_to_silence():
    target = torch.tensor([0.0, 0.5, 1.0])
    weights = pause_frame_weights(target, pause_weight=5.0)
    torch.testing.assert_close(weights, torch.tensor([1.0, 3.0, 5.0]))


def test_pause_weight_below_one_is_rejected():
    with pytest.raises(ValueError, match="pause_weight"):
        pause_frame_weights(torch.zeros(3), pause_weight=0.5)


# --- waqf_distillation_loss: proper KL, masked, weight-normalised -------------


def test_kl_is_zero_when_student_matches_teacher():
    target = torch.tensor([[0.0, 1.0, 0.3, 0.8]])
    logits = torch.logit(target.clamp(1e-6, 1 - 1e-6))
    loss = waqf_distillation_loss(logits, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_kl_is_positive_on_mismatch():
    target = torch.tensor([[1.0, 1.0, 1.0]])
    logits = torch.tensor([[-5.0, -5.0, -5.0]])  # predicts speech, teacher is silent
    assert waqf_distillation_loss(logits, target).item() > 0


def test_masked_frames_do_not_contribute():
    target = torch.tensor([[0.0, 1.0, 0.5]])
    logits = torch.logit(target.clamp(1e-6, 1 - 1e-6)).clone()
    logits[0, 2] = -10.0  # a large error, but on a padded frame
    mask = torch.tensor([[True, True, False]])
    assert waqf_distillation_loss(logits, target, mask).item() == pytest.approx(0.0, abs=1e-5)


def test_all_frames_masked_out_is_zero_not_nan():
    logits = torch.randn(1, 3)
    target = torch.rand(1, 3)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    loss = waqf_distillation_loss(logits, target, mask)
    assert loss.item() == 0.0


def test_pause_weighting_amplifies_silence_frame_error():
    # A batch dominated by well-fit speech frames with one mis-fit silence frame: pause
    # weighting stops the rare silence error from being diluted away by the majority.
    target = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    logits = torch.tensor([[-8.0, -8.0, -8.0, 0.0]])  # speech fit; silence frame missed
    weighted = waqf_distillation_loss(logits, target, pause_weight=5.0)
    unweighted = waqf_distillation_loss(logits, target, pause_weight=1.0)
    assert weighted.item() > unweighted.item()


# --- pause_collapse_diagnostic ------------------------------------------------


def test_collapse_flagged_when_head_predicts_all_speech():
    target = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])  # 60% silence
    logits = torch.full((1, 5), -10.0)  # head predicts speech everywhere
    diag = pause_collapse_diagnostic(logits, target)
    assert diag.collapsed
    assert diag.predicted_silence_rate == 0.0
    assert diag.target_silence_rate == pytest.approx(0.6)


def test_no_collapse_when_head_tracks_teacher():
    target = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    logits = torch.tensor([[5.0, 5.0, -5.0, -5.0]])
    diag = pause_collapse_diagnostic(logits, target)
    assert not diag.collapsed
    assert diag.predicted_silence_rate == pytest.approx(0.5)


def test_no_collapse_flag_when_teacher_has_no_silence():
    # An all-speech window can't collapse: there is nothing to miss.
    target = torch.zeros(1, 5)
    logits = torch.full((1, 5), -10.0)
    assert not pause_collapse_diagnostic(logits, target).collapsed


def test_collapse_diagnostic_respects_frame_mask():
    # The silence all lives in padded frames, so the valid frames are pure speech and
    # there is nothing to collapse.
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    logits = torch.full((1, 4), -10.0)
    mask = torch.tensor([[True, True, False, False]])
    diag = pause_collapse_diagnostic(logits, target, mask)
    assert diag.target_silence_rate == 0.0
    assert not diag.collapsed


# --- joint_loss end-to-end ---------------------------------------------------


def test_joint_loss_composes_ctc_and_kl():
    model = WaqfJointModel(_tiny_model())
    out = model(torch.randn(2, 20, 160))
    labels = torch.randint(0, 43, (2, 4))
    target = torch.rand(2, 10)
    result = joint_loss(out, labels, target, model.muaalem.config, waqf_loss_weight=2.0)
    assert isinstance(result, JointLoss)
    torch.testing.assert_close(
        result.total, result.phoneme_ctc + 2.0 * result.waqf_kl
    )


def test_joint_loss_masks_padded_student_frames():
    # A short second example (fewer real frames) must not have its padded silence frames
    # scored: student_lengths drives both the CTC input length and the KL mask.
    model = WaqfJointModel(_tiny_model())
    out = model(torch.randn(2, 20, 160))
    out.student_lengths[1] = 4  # pretend example 1 only has 4 valid frames
    labels = torch.randint(0, 43, (2, 3))
    target = torch.rand(2, 10)
    result = joint_loss(out, labels, target, model.muaalem.config)
    assert torch.isfinite(result.total)


# --- frame_mask_from_lengths -------------------------------------------------


def test_frame_mask_from_lengths():
    mask = frame_mask_from_lengths(torch.tensor([2, 4]), num_frames=4)
    assert mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
