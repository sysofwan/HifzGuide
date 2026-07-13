"""Tests for the whole-clip phoneme-only LoRA run (ADR-0004 rung (2), issue #29).

The load-bearing claims: LoRA leaves the backbone base frozen and the sifat heads
untrained (phoneme-only isolation), the phoneme forward is bit-identical to the joint
model's (so rung (2)↔(3) differ only by the waqf head), and the memory preflight measures
a real worst-case batch against the 16 GB budget. The heavy CUDA paths are guarded.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tadabur.muaalem.configuration_multi_level_ctc import (
    Wav2Vec2BertForMultilevelCTCConfig,
)
from tadabur.muaalem.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from training.waqf_head import WaqfJointModel, phoneme_forward
from training.whole_clip_phoneme import (
    LoRASettings,
    attach_phoneme_lora,
    base_of,
    preflight_batch_memory,
    set_seed,
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
    )
    return Wav2Vec2BertForMultilevelCTC(config)


# --- LoRA attach: backbone frozen, phoneme head trains, sifat frozen ---------


def test_lora_freezes_backbone_and_trains_phoneme_head_only():
    peft_model = attach_phoneme_lora(_tiny_model(), LoRASettings(rank=4))
    trainable = {
        name for name, p in peft_model.named_parameters() if p.requires_grad
    }
    assert trainable, "expected some trainable parameters"
    # Every trainable parameter is either a LoRA adapter or the phoneme head — never the
    # sifat head, and never a frozen backbone base weight.
    assert all("lora_" in n or "phonemes" in n for n in trainable)
    assert not any("hams_or_jahr" in n for n in trainable)


def test_lora_backbone_base_weight_is_frozen():
    peft_model = attach_phoneme_lora(_tiny_model(), LoRASettings(rank=4))
    base_attn = dict(peft_model.named_parameters())
    frozen = [
        p for n, p in base_attn.items()
        if "self_attn.linear_q.base_layer.weight" in n
    ]
    assert frozen and all(not p.requires_grad for p in frozen)


# --- phoneme forward is identical to the joint model's -----------------------


def test_phoneme_forward_matches_joint_model_logits():
    # rung (2) phoneme-only and rung (3) joint must share the phoneme path exactly.
    model = _tiny_model()
    features = torch.randn(2, 20, 160)
    joint = WaqfJointModel(model)
    joint.eval()
    model.eval()
    with torch.no_grad():
        joint_out = joint(features)
        solo = phoneme_forward(model, "phonemes", features)
    assert torch.equal(joint_out.phoneme_logits, solo.phoneme_logits)
    assert torch.equal(joint_out.student_lengths, solo.student_lengths)


def test_phoneme_forward_shapes():
    model = _tiny_model()
    out = phoneme_forward(model, "phonemes", torch.randn(3, 20, 160))
    assert out.phoneme_logits.shape == (3, 10, 43)
    assert out.student_lengths.tolist() == [10, 10, 10]


# --- determinism -------------------------------------------------------------


def test_set_seed_is_reproducible():
    set_seed(123)
    a = torch.randn(4)
    set_seed(123)
    b = torch.randn(4)
    assert torch.equal(a, b)


# --- memory preflight (CUDA-only) --------------------------------------------


class _StubFeatureExtractor:
    sampling_rate = 16000

    def __call__(self, waveforms, sampling_rate, return_tensors, padding):
        frames = [len(w) // 320 for w in waveforms]
        max_frames = max(frames)
        features = torch.zeros(len(waveforms), max_frames, 160)
        mask = torch.zeros(len(waveforms), max_frames, dtype=torch.long)
        for i, n in enumerate(frames):
            mask[i, :n] = 1
        return type("F", (), {"input_features": features, "attention_mask": mask})()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="preflight measures CUDA memory")
def test_preflight_measures_and_fits_under_budget():
    device = torch.device("cuda")
    peft_model = attach_phoneme_lora(_tiny_model(), LoRASettings(rank=4)).to(
        device, torch.bfloat16
    )
    report = preflight_batch_memory(
        peft_model,
        _StubFeatureExtractor(),
        np.zeros(320 * 250, dtype=np.float32),
        num_windows=4,
        device=device,
        window_feature_frames=250,
        budget_gib=15.0,
    )
    assert report.num_windows == 4
    assert report.peak_reserved_gib > 0
    assert report.fits  # a tiny model is nowhere near 15 GiB


def test_preflight_rejects_cpu():
    peft_model = attach_phoneme_lora(_tiny_model(), LoRASettings(rank=4))
    with pytest.raises(RuntimeError, match="CUDA"):
        preflight_batch_memory(
            peft_model,
            _StubFeatureExtractor(),
            np.zeros(320 * 10, dtype=np.float32),
            num_windows=2,
            device=torch.device("cpu"),
        )
