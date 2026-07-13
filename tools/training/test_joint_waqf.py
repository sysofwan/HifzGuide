"""Tests for the joint detached-waqf run (ADR-0004 rung (3), issue #31).

The load-bearing claims: the waqf KL leaves the backbone gradient untouched (the asserted
rung (2)↔(3) isolation), the held-out distillation floor logic is exact, and the
capacity-fallback ladder is a governed, ordered policy (with re-ablation on any unfreeze).
The heavy CUDA/model paths run on a tiny CPU model; the VRAM preflight is guarded.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tadabur.muaalem.configuration_multi_level_ctc import (
    Wav2Vec2BertForMultilevelCTCConfig,
)
from tadabur.muaalem.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from training.waqf_head import WaqfJointModel
from training.windowed_batch import JointWindowedBatch, WindowedCtcBatch
from training.joint_waqf import (
    FALLBACK_LADDER,
    JointTrainConfig,
    SilenceCounts,
    assert_backbone_isolation,
    backbone_isolation_report,
    build_joint_model,
    distillation_verdict,
    next_fallback,
    stage_by_name,
    train,
)


def _tiny_model():
    torch.manual_seed(0)
    config = Wav2Vec2BertForMultilevelCTCConfig(
        level_to_vocab_size={"phonemes": 43, "hams_or_jahr": 3},
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
        layerdrop=0.0,
    )
    return Wav2Vec2BertForMultilevelCTC(config)


def _joint_batch(num_windows=2, feature_frames=20):
    features = torch.randn(num_windows, feature_frames, 160)
    mask = torch.ones(num_windows, feature_frames, dtype=torch.long)
    lattice = (feature_frames + 1) // 2
    labels = torch.randint(1, 43, (num_windows, lattice // 2))
    target_silence = torch.rand(num_windows, lattice)
    return JointWindowedBatch(
        phoneme=WindowedCtcBatch(features, mask, labels, [("c", i) for i in range(num_windows)]),
        target_silence=target_silence,
    )


def _joint_forward_inputs(num_windows=2, feature_frames=20):
    b = _joint_batch(num_windows, feature_frames)
    return b.phoneme.input_features, b.phoneme.attention_mask


# --- gradient isolation: the go/no-go (2)→(3) check ---------------------------


def test_waqf_kl_leaves_backbone_gradient_untouched():
    joint = WaqfJointModel(_tiny_model())
    report = backbone_isolation_report(joint, _joint_batch(), pause_weight=5.0)
    assert report.isolated
    assert report.max_backbone_grad == 0.0
    assert report.num_backbone_params_with_grad == 0


def test_waqf_head_still_learns_under_the_detach():
    # Isolation must not mean the head is dead: the KL still reaches the waqf head itself.
    joint = WaqfJointModel(_tiny_model())
    batch = _joint_batch()
    output = joint(batch.phoneme.input_features, batch.phoneme.attention_mask)
    from training.waqf_head import frame_mask_from_lengths, waqf_distillation_loss

    frame_mask = frame_mask_from_lengths(
        output.student_lengths, output.silence_logits.shape[1]
    )
    waqf_distillation_loss(
        output.silence_logits, batch.target_silence, frame_mask, 5.0
    ).backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0
               for p in joint.waqf_head.parameters())


def test_broken_detach_fails_the_isolation_assertion():
    joint = WaqfJointModel(_tiny_model())
    # Break the stop-gradient: classify the *attached* states so the KL reaches the backbone.
    joint.waqf_head.forward = joint.waqf_head.classify  # type: ignore[method-assign]
    report = backbone_isolation_report(joint, _joint_batch(), pause_weight=5.0)
    assert not report.isolated
    assert report.num_backbone_params_with_grad > 0
    with pytest.raises(AssertionError, match="detach is broken"):
        assert_backbone_isolation(joint, _joint_batch(), pause_weight=5.0)


def test_build_joint_model_selects_head_width():
    linear = build_joint_model(_wrap_identity(_tiny_model()), stage_by_name("linear"))
    assert isinstance(linear.waqf_head.classifier, torch.nn.Linear)
    mlp = build_joint_model(_wrap_identity(_tiny_model()), stage_by_name("mlp_head"))
    assert isinstance(mlp.waqf_head.classifier, torch.nn.Sequential)


def test_build_joint_model_forces_adapter_layerdrop_off():
    # A checkpoint that ships nonzero layerdrop would let the adapter stochastically skip
    # subsampling in train mode, breaking the frozen windowing contract. build_joint_model
    # must neutralize it so the lattice length stays deterministic.
    model = _tiny_model()
    model.config.layerdrop = 0.1
    model.wav2vec2_bert.adapter.layerdrop = 0.1
    joint = build_joint_model(_wrap_identity(model), stage_by_name("linear"))
    assert joint.muaalem.config.layerdrop == 0.0
    assert joint.muaalem.wav2vec2_bert.adapter.layerdrop == 0.0

    # Determinism under the contract: repeated train-mode forwards keep the lattice length.
    joint.train()
    lengths = set()
    for _ in range(8):
        out = joint(*_joint_forward_inputs())
        lengths.add(int(out.silence_logits.shape[1]))
    assert lengths == {10}


class _IdentityPeft:
    """Stand-in for a PEFT model: get_base_model returns the wrapped module unchanged."""

    def __init__(self, model):
        self._model = model

    def get_base_model(self):
        return self._model


def _wrap_identity(model):
    return _IdentityPeft(model)


# --- held-out distillation floor (pure verdict logic) ------------------------


def test_perfect_prediction_meets_floor():
    counts = SilenceCounts(
        true_positive=40, predicted_silent=40, target_silent=40,
        valid_frames=100, kl_sum=0.0, kl_weight=100,
    )
    report = distillation_verdict(counts, min_silence_f1=0.5)
    assert report.silence_f1 == 1.0
    assert not report.collapsed
    assert report.meets_floor


def test_low_f1_misses_floor():
    counts = SilenceCounts(
        true_positive=5, predicted_silent=40, target_silent=40,
        valid_frames=100, kl_sum=1.0, kl_weight=100,
    )
    report = distillation_verdict(counts, min_silence_f1=0.5)
    assert report.silence_f1 < 0.5
    assert not report.meets_floor


def test_pause_collapse_misses_floor_even_at_zero_predictions():
    # Head predicts no silence while the teacher has plenty → collapsed, floor missed.
    counts = SilenceCounts(
        true_positive=0, predicted_silent=0, target_silent=30,
        valid_frames=100, kl_sum=2.0, kl_weight=100,
    )
    report = distillation_verdict(counts, min_silence_f1=0.5)
    assert report.collapsed
    assert not report.meets_floor


def test_no_silence_anywhere_is_a_trivial_pass():
    # Neither head nor teacher fire: F1 is 1.0 and there is nothing to collapse.
    counts = SilenceCounts(
        true_positive=0, predicted_silent=0, target_silent=0,
        valid_frames=100, kl_sum=0.0, kl_weight=100,
    )
    report = distillation_verdict(counts)
    assert report.silence_f1 == 1.0
    assert not report.collapsed
    assert report.meets_floor


# --- capacity-fallback ladder: the governed gate -----------------------------


def test_ladder_is_ordered_and_terminal():
    names = [s.name for s in FALLBACK_LADDER]
    assert names == [
        "linear", "mlp_head", "pause_weight_retune", "partial_unfreeze", "blank_run",
    ]
    assert next_fallback(stage_by_name("blank_run")) is None


def test_next_fallback_walks_the_ladder():
    assert next_fallback(stage_by_name("linear")).name == "mlp_head"
    assert next_fallback(stage_by_name("mlp_head")).name == "pause_weight_retune"
    assert next_fallback(stage_by_name("pause_weight_retune")).name == "partial_unfreeze"


def test_only_partial_unfreeze_requires_reablation_and_breaks_isolation():
    for stage in FALLBACK_LADDER:
        if stage.name == "partial_unfreeze":
            assert stage.breaks_isolation and stage.requires_reablation
        else:
            assert not stage.breaks_isolation and not stage.requires_reablation


def test_head_only_stages_are_exactly_the_detached_ones():
    head_only = {s.name for s in FALLBACK_LADDER if s.is_head_only}
    assert head_only == {"linear", "mlp_head", "pause_weight_retune"}


def test_stage_by_name_rejects_unknown():
    with pytest.raises(ValueError, match="unknown fallback stage"):
        stage_by_name("full_finetune")


def test_pause_weight_retune_raises_the_pause_weight():
    assert stage_by_name("pause_weight_retune").pause_weight > stage_by_name("linear").pause_weight


# --- train refuses to silently run an isolation-breaking stage ---------------


def test_train_refuses_non_head_only_stage(tmp_path):
    config = JointTrainConfig(stage_name="partial_unfreeze")
    with pytest.raises(ValueError, match="not head-only"):
        train(tmp_path / "labels.jsonl", tmp_path / "audio", tmp_path / "soft",
              tmp_path / "out", config, torch.device("cpu"))
