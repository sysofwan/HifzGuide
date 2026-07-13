"""Tests for the ADR-0004 ablation ladder (issue #33, P7.E).

Three load-bearing claims: rung (2) and rung (3) share the phoneme path bit-for-bit (the
deterministic (2)→(3) identity go/no-go), the should-accept / should-reject deltas across the
three rungs are computed exactly, and a whole-clip-move should-reject regression fires the
ordered LoRA-native lever (rank/alpha → L2-SP), never sifat. The heavy model path runs on a
tiny CPU model.
"""

from __future__ import annotations

import pytest
import torch

from tadabur.muaalem.configuration_multi_level_ctc import (
    Wav2Vec2BertForMultilevelCTCConfig,
)
from tadabur.muaalem.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from training import ablation_ladder as al
from training.ablation_ladder import (
    AblationLadder,
    LORA_LEVER_LADDER,
    LoRALever,
    PhonemeIdentityReport,
    assert_phoneme_identity,
    lever_by_name,
    next_lora_lever,
    phoneme_identity_report,
    recommend_lora_lever,
    whole_clip_move_regressed,
)
from training.whole_clip_phoneme import (
    LoRASettings,
    lora_anchor_snapshot,
    lora_l2sp_penalty,
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


def _batch(num_windows=2, feature_frames=20):
    torch.manual_seed(1)
    features = torch.randn(num_windows, feature_frames, 160)
    mask = torch.ones(num_windows, feature_frames, dtype=torch.long)
    lattice = (feature_frames + 1) // 2
    labels = torch.randint(1, 43, (num_windows, lattice // 2))
    return features, mask, labels


def _eval_json(recall, discrimination, accept_total=10, reject_total=10):
    accepted = None if recall is None else round(recall * accept_total)
    rejected = None if discrimination is None else round(discrimination * reject_total)
    return {
        "should_accept": {
            "total": accept_total,
            "accepted": accepted,
            "rejected": None if accepted is None else accept_total - accepted,
            "recall": recall,
        },
        "should_reject": {
            "total": reject_total,
            "accepted": None if rejected is None else reject_total - rejected,
            "rejected": rejected,
            "discrimination": discrimination,
        },
    }


# --- (2)↔(3) phoneme identity check ------------------------------------------


def test_rung2_and_rung3_share_phoneme_path_bit_for_bit():
    features, mask, labels = _batch()
    report = phoneme_identity_report(_tiny_model(), features, mask, labels)
    assert report.identical
    assert report.logits_match and report.max_logit_diff == 0.0
    assert report.grads_match and report.max_grad_diff == 0.0


def test_assert_phoneme_identity_returns_the_report():
    features, mask, labels = _batch()
    report = assert_phoneme_identity(_tiny_model(), features, mask, labels)
    assert isinstance(report, PhonemeIdentityReport)
    assert report.identical


def test_assert_phoneme_identity_raises_when_the_path_diverges(monkeypatch):
    diverged = PhonemeIdentityReport(
        logits_match=False, max_logit_diff=1e-2, grads_match=True, max_grad_diff=0.0
    )
    monkeypatch.setattr(al, "phoneme_identity_report", lambda *a, **k: diverged)
    features, mask, labels = _batch()
    with pytest.raises(AssertionError, match="phoneme path"):
        assert_phoneme_identity(_tiny_model(), features, mask, labels)


def test_identity_check_is_deterministic_across_runs():
    features, mask, labels = _batch()
    first = phoneme_identity_report(_tiny_model(), features, mask, labels)
    second = phoneme_identity_report(_tiny_model(), features, mask, labels)
    assert first == second


# --- three-rung ladder deltas ------------------------------------------------


def test_ladder_reports_deltas_across_all_three_rungs():
    ladder = AblationLadder.from_reports(
        _eval_json(recall=0.60, discrimination=0.90),
        _eval_json(recall=0.75, discrimination=0.88),
        _eval_json(recall=0.75, discrimination=0.88),
    )
    move = ladder.whole_clip_move
    assert move.from_rung == "segmented_phoneme_only"
    assert move.to_rung == "whole_clip_phoneme_only"
    assert move.recall_delta == pytest.approx(0.15)
    assert move.discrimination_delta == pytest.approx(-0.02)

    addition = ladder.waqf_head_addition
    assert addition.recall_delta == pytest.approx(0.0)
    assert addition.discrimination_delta == pytest.approx(0.0)


def test_ladder_json_round_trips_the_three_rungs():
    ladder = AblationLadder.from_reports(
        _eval_json(0.6, 0.9), _eval_json(0.75, 0.9), _eval_json(0.75, 0.9)
    )
    payload = ladder.to_json_dict()
    assert set(payload["rungs"]) == {
        "segmented_phoneme_only",
        "whole_clip_phoneme_only",
        "whole_clip_joint_waqf",
    }
    assert set(payload["transitions"]) == {"whole_clip_move", "waqf_head_addition"}


def test_missing_side_yields_none_delta_not_a_crash():
    ladder = AblationLadder.from_reports(
        _eval_json(recall=None, discrimination=0.9),
        _eval_json(recall=0.7, discrimination=0.9),
        _eval_json(recall=0.7, discrimination=0.9),
    )
    assert ladder.whole_clip_move.recall_delta is None
    assert ladder.whole_clip_move.discrimination_delta == pytest.approx(0.0)


# --- whole-clip-move regression → LoRA-native lever --------------------------


def test_regression_fires_the_first_lora_native_lever():
    ladder = AblationLadder.from_reports(
        _eval_json(0.6, 0.90), _eval_json(0.75, 0.80), _eval_json(0.75, 0.80)
    )
    assert whole_clip_move_regressed(ladder)
    lever = recommend_lora_lever(ladder)
    assert lever is not None
    assert lever.name == "lower_rank_alpha"
    assert lever.lora.rank == 8 and lever.lora.alpha == 16
    assert lever.l2_sp == 0.0


def test_second_lever_is_l2_sp_after_lower_rank_alpha():
    ladder = AblationLadder.from_reports(
        _eval_json(0.6, 0.90), _eval_json(0.75, 0.80), _eval_json(0.75, 0.80)
    )
    lever = recommend_lora_lever(ladder, current=lever_by_name("lower_rank_alpha"))
    assert lever.name == "lower_rank_alpha_l2_sp"
    assert lever.l2_sp > 0.0


def test_no_regression_recommends_no_lever():
    ladder = AblationLadder.from_reports(
        _eval_json(0.6, 0.90), _eval_json(0.75, 0.90), _eval_json(0.75, 0.90)
    )
    assert not whole_clip_move_regressed(ladder)
    assert recommend_lora_lever(ladder) is None


def test_improved_discrimination_is_not_a_regression():
    ladder = AblationLadder.from_reports(
        _eval_json(0.6, 0.85), _eval_json(0.75, 0.92), _eval_json(0.75, 0.92)
    )
    assert not whole_clip_move_regressed(ladder)


def test_tolerance_can_absorb_small_drops():
    ladder = AblationLadder.from_reports(
        _eval_json(0.6, 0.90), _eval_json(0.75, 0.88), _eval_json(0.75, 0.88)
    )
    assert whole_clip_move_regressed(ladder, tolerance=0.0)
    assert not whole_clip_move_regressed(ladder, tolerance=0.05)


def test_lever_ladder_is_ordered_and_terminates():
    assert LORA_LEVER_LADDER[0].name == "baseline"
    names = [l.name for l in LORA_LEVER_LADDER]
    assert names == ["baseline", "lower_rank_alpha", "lower_rank_alpha_l2_sp"]
    assert next_lora_lever(LORA_LEVER_LADDER[-1]) is None


def test_sifat_is_not_on_the_lora_lever_ladder():
    for lever in LORA_LEVER_LADDER:
        assert "sifat" not in lever.name
        assert "sifat" not in lever.description.lower()


def test_lever_by_name_rejects_unknown():
    with pytest.raises(ValueError, match="unknown LoRA lever"):
        lever_by_name("reattach_sifat")


# --- L2-SP lever mechanics (consumed by rung (2)) ----------------------------


def test_l2sp_penalty_is_zero_at_the_anchor_and_grows_off_it():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    peft = get_peft_model(
        _tiny_model(),
        LoraConfig(r=4, lora_alpha=8, target_modules=["linear_q", "linear_v"], bias="none"),
    )
    anchors = lora_anchor_snapshot(peft)
    assert anchors  # adapters were selected
    assert float(lora_l2sp_penalty(peft, anchors).detach()) == 0.0

    with torch.no_grad():
        for name, param in peft.named_parameters():
            if param.requires_grad and "lora_" in name:
                param.add_(1.0)
    assert float(lora_l2sp_penalty(peft, anchors).detach()) > 0.0


def test_lora_lever_config_feeds_train_config():
    from training.whole_clip_phoneme import TrainConfig

    lever = lever_by_name("lower_rank_alpha_l2_sp")
    config = TrainConfig(lora=lever.lora, l2_sp=lever.l2_sp)
    assert config.lora.rank == 8 and config.l2_sp == lever.l2_sp
