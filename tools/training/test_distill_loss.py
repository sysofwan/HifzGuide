"""Tests for the distillation objective.

The load-bearing claims are that the two weightings actually bias the loss the way the
module docstring says, that the KL is a real KL (zero iff the distributions match), and
that the blank-collapse diagnostic fires on the failure mode it exists to catch.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training import distill_loss as dl


def _logits(batch=2, frames=125, classes=43, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, frames, classes, generator=generator)


def _blank_dominated_logits(batch=2, frames=125, classes=43, nonblank_frames=()):
    """Logits whose argmax is blank everywhere except the given frame indices."""
    logits = torch.full((batch, frames, classes), -5.0)
    logits[:, :, dl.BLANK_ID] = 5.0
    for frame in nonblank_frames:
        logits[:, frame, dl.BLANK_ID] = -5.0
        logits[:, frame, 7] = 5.0
    return logits


# --- Frame weighting ----------------------------------------------------------------


def test_nonblank_frames_are_upweighted():
    teacher = _blank_dominated_logits(nonblank_frames=(60, 61))
    weights = dl.frame_weights(teacher, nonblank_weight=3.0, confirm_weight=1.0)
    assert weights[0, 60].item() == pytest.approx(3.0)
    assert weights[0, 0].item() == pytest.approx(1.0)


def test_confirmed_region_is_upweighted():
    teacher = _blank_dominated_logits()
    weights = dl.frame_weights(teacher, nonblank_weight=1.0, confirm_weight=2.0)
    assert weights[0, 0].item() == pytest.approx(2.0)
    assert weights[0, dl.CONFIRM_TIMESTEPS - 1].item() == pytest.approx(2.0)
    assert weights[0, dl.CONFIRM_TIMESTEPS].item() == pytest.approx(1.0)


def test_the_two_weightings_multiply():
    """A non-blank frame inside the confirmed region carries both boosts."""
    teacher = _blank_dominated_logits(nonblank_frames=(3,))
    weights = dl.frame_weights(teacher, nonblank_weight=3.0, confirm_weight=2.0)
    assert weights[0, 3].item() == pytest.approx(6.0)


def test_weights_come_from_the_teacher_not_the_student():
    """The weighting must be a property of the target, so the student cannot game it."""
    teacher = _blank_dominated_logits(nonblank_frames=(10,))
    first = dl.frame_weights(teacher)
    second = dl.frame_weights(teacher.clone())
    assert torch.equal(first, second)


def test_frame_weights_rejects_wrong_rank():
    with pytest.raises(ValueError, match="expected"):
        dl.frame_weights(torch.randn(125, 43))


# --- The KL term --------------------------------------------------------------------


def test_kl_is_zero_when_the_student_matches_the_teacher():
    logits = _logits()
    loss = dl.weighted_kl(logits.clone(), logits, weights=None, temperature=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_kl_is_positive_when_they_differ():
    assert dl.weighted_kl(_logits(seed=1), _logits(seed=2)).item() > 0.0


def test_kl_ignores_shifts_that_do_not_change_the_distribution():
    """Softmax is shift-invariant, so a constant offset must not move the loss."""
    teacher = _logits(seed=3)
    shifted = teacher + 4.2
    assert dl.weighted_kl(shifted, teacher, temperature=1.0).item() == pytest.approx(
        0.0, abs=1e-5
    )


def test_weighting_moves_the_loss_toward_the_weighted_frames():
    """A student wrong only on non-blank frames is penalised more once they are boosted."""
    teacher = _blank_dominated_logits(nonblank_frames=(60,))
    student = teacher.clone()
    student[:, 60, 7] = -5.0     # wrong exactly where the teacher says non-blank
    student[:, 60, 11] = 5.0

    flat = dl.weighted_kl(student, teacher, weights=None, temperature=1.0)
    boosted = dl.weighted_kl(
        student,
        teacher,
        dl.frame_weights(teacher, nonblank_weight=5.0, confirm_weight=1.0),
        temperature=1.0,
    )
    assert boosted.item() > flat.item()


def test_kl_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        dl.weighted_kl(_logits(batch=2), _logits(batch=3))


def test_kl_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        dl.weighted_kl(_logits(), _logits(), temperature=0.0)


# --- Feature matching ---------------------------------------------------------------


def test_feature_loss_falls_to_zero_for_a_perfect_projection():
    """An identity-projectable student should reach ~0 loss and cosine 1."""
    student_hidden = tuple(torch.randn(2, 250, 8) for _ in range(5))
    projector = dl.FeatureProjector(student_dim=8, teacher_dim=8, num_taps=1)
    with torch.no_grad():
        projector.projections[0].weight.copy_(torch.eye(8))
        projector.projections[0].bias.zero_()

    loss, cosine = dl.feature_matching_loss(
        student_hidden, student_hidden, projector, tap_layers=(4,)
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert cosine == pytest.approx(1.0, abs=1e-5)


def test_feature_loss_rejects_out_of_range_taps():
    hidden = tuple(torch.randn(1, 10, 4) for _ in range(3))
    projector = dl.FeatureProjector(4, 4, num_taps=1)
    with pytest.raises(IndexError, match="out of range"):
        dl.feature_matching_loss(hidden, hidden, projector, tap_layers=(9,))


def test_projector_has_one_map_per_tap():
    projector = dl.FeatureProjector(384, 1024, num_taps=len(dl.DEFAULT_TAP_LAYERS))
    assert len(projector.projections) == len(dl.DEFAULT_TAP_LAYERS)


# --- Agreement and the collapse diagnostic ------------------------------------------


def test_perfect_agreement_when_logits_are_identical():
    logits = _logits()
    stats = dl.agreement_stats(logits.clone(), logits)
    assert stats.overall_agreement == pytest.approx(1.0)
    assert stats.confirmed_agreement == pytest.approx(1.0)
    assert stats.nonblank_agreement == pytest.approx(1.0)


def test_blank_collapse_is_detected():
    """The failure the weighting exists to prevent: student predicts blank everywhere."""
    teacher = _blank_dominated_logits(nonblank_frames=tuple(range(40, 90)))
    student = _blank_dominated_logits()          # all blank

    stats = dl.agreement_stats(student, teacher)
    assert stats.is_collapsing()
    assert stats.blank_collapse_margin > 0.15
    assert stats.nonblank_agreement == pytest.approx(0.0)


def test_healthy_student_is_not_flagged_as_collapsing():
    teacher = _blank_dominated_logits(nonblank_frames=tuple(range(40, 90)))
    stats = dl.agreement_stats(teacher.clone(), teacher)
    assert not stats.is_collapsing()
    assert stats.blank_collapse_margin == pytest.approx(0.0)


def test_confirmed_agreement_tracks_only_the_confirmed_region():
    """A student wrong only outside [0, 25) keeps confirmed agreement at 1.0."""
    teacher = _blank_dominated_logits()
    student = teacher.clone()
    student[:, dl.CONFIRM_TIMESTEPS :, dl.BLANK_ID] = -5.0
    student[:, dl.CONFIRM_TIMESTEPS :, 9] = 5.0

    stats = dl.agreement_stats(student, teacher)
    assert stats.confirmed_agreement == pytest.approx(1.0)
    assert stats.overall_agreement < 0.3


# --- Distance from breakout ---------------------------------------------------------


def test_breakout_reports_agreement_as_rank_one():
    """A student that already matches the teacher is rank 1 with no margin to close."""
    teacher = _blank_dominated_logits(nonblank_frames=tuple(range(40, 90)))
    stats = dl.breakout_stats(teacher.clone(), teacher)
    assert stats.target_rank == pytest.approx(1.0)
    assert stats.prob_margin < 0
    assert stats.top5_agreement == pytest.approx(1.0)


def test_breakout_separates_nearly_escaped_from_hopeless():
    """The distinction argmax agreement cannot make: both score nonblank_agreement 0."""
    teacher = _blank_dominated_logits(nonblank_frames=tuple(range(40, 90)))

    # Nearly escaped: blank narrowly ahead of the correct class.
    close = torch.full_like(teacher, -5.0)
    close[:, :, dl.BLANK_ID] = 1.0
    close[:, 40:90, 7] = 0.9

    # Hopeless: blank overwhelming, and a crowd of distractors above the correct class,
    # which is where it actually sits when the student has learned nothing useful.
    far = torch.full_like(teacher, -5.0)
    far[:, :, dl.BLANK_ID] = 10.0
    far[:, 40:90, 11:20] = 2.0   # nine distractors outrank the target
    far[:, 40:90, 7] = -5.0

    close_stats = dl.breakout_stats(close, teacher)
    far_stats = dl.breakout_stats(far, teacher)

    # Argmax cannot tell them apart...
    assert dl.agreement_stats(close, teacher).nonblank_agreement == pytest.approx(0.0)
    assert dl.agreement_stats(far, teacher).nonblank_agreement == pytest.approx(0.0)

    # ...but the continuous view can.
    assert close_stats.target_rank < far_stats.target_rank
    assert close_stats.target_prob > far_stats.target_prob
    assert close_stats.prob_margin < far_stats.prob_margin


def test_breakout_rank_counts_classes_scoring_higher():
    teacher = _blank_dominated_logits(nonblank_frames=(10,))
    student = torch.full_like(teacher, -5.0)
    student[:, 10, dl.BLANK_ID] = 3.0   # blank highest
    student[:, 10, 11] = 2.0            # a distractor second
    student[:, 10, 7] = 1.0             # teacher's class third
    stats = dl.breakout_stats(student, teacher)
    assert stats.target_rank == pytest.approx(3.0)


def test_breakout_ignores_blank_frames():
    """Measured only where the student is failing -- teacher-non-blank frames."""
    teacher = _blank_dominated_logits(nonblank_frames=(10,))
    student = _blank_dominated_logits(nonblank_frames=(10,))
    stats = dl.breakout_stats(student, teacher)
    # Only frame 10 is scored; the student matches there, so rank is 1.
    assert stats.target_rank == pytest.approx(1.0)


def test_breakout_handles_an_all_blank_teacher():
    """No non-blank frames to score -- must not divide by zero."""
    teacher = _blank_dominated_logits()
    stats = dl.breakout_stats(teacher.clone(), teacher)
    assert stats.target_prob == 0.0
    assert stats.target_rank == 0.0


# --- The assembled objective --------------------------------------------------------


def test_distillation_loss_assembles_and_backprops():
    student_logits = _logits(seed=5).requires_grad_(True)
    teacher_logits = _logits(seed=6)
    student_hidden = tuple(torch.randn(2, 250, 16, requires_grad=True) for _ in range(25))
    teacher_hidden = tuple(torch.randn(2, 250, 32) for _ in range(25))
    projector = dl.FeatureProjector(16, 32, num_taps=len(dl.DEFAULT_TAP_LAYERS))

    output = dl.distillation_loss(
        student_logits, teacher_logits, student_hidden, teacher_hidden, projector
    )
    output.total.backward()

    assert torch.isfinite(output.total)
    assert student_logits.grad is not None
    assert set(output.as_dict()) >= {"total", "logit_loss", "confirmed_agreement"}


def test_loss_weights_are_respected():
    """Zeroing the feature weight must remove that term from the total."""
    student_logits = _logits(seed=7)
    teacher_logits = _logits(seed=8)
    student_hidden = tuple(torch.randn(2, 250, 16) for _ in range(25))
    teacher_hidden = tuple(torch.randn(2, 250, 32) for _ in range(25))
    projector = dl.FeatureProjector(16, 32, num_taps=len(dl.DEFAULT_TAP_LAYERS))

    config = dl.DistillLossConfig(feature_weight=0.0, ctc_weight=0.0)
    output = dl.distillation_loss(
        student_logits, teacher_logits, student_hidden, teacher_hidden, projector, config
    )
    assert float(output.total.detach()) == pytest.approx(output.logit_loss, rel=1e-5)


# --- The CTC anchor: the term that makes all-blank unstable --------------------------


def test_teacher_targets_collapse_repeats_and_drop_blanks():
    """Standard CTC collapse -- the same one scanCTC does on device."""
    logits = torch.full((1, 8, 43), -5.0)
    # sequence: blank, 7, 7, blank, 7, 9, 9, blank  ->  [7, 7, 9]
    for step, token in enumerate([0, 7, 7, 0, 7, 9, 9, 0]):
        logits[0, step, token] = 5.0
    targets, lengths = dl.teacher_target_sequences(logits)
    assert lengths.tolist() == [3]
    assert targets.tolist() == [7, 7, 9]


def test_all_blank_teacher_yields_an_empty_target():
    logits = _blank_dominated_logits(batch=1)
    targets, lengths = dl.teacher_target_sequences(logits)
    assert lengths.tolist() == [0]
    assert targets.numel() == 0


def test_ctc_anchor_punishes_the_all_blank_student():
    """The load-bearing property: all-blank has probability zero under any alignment.

    This is precisely what the frame-wise KL cannot express -- an all-blank student scores
    a *better* per-frame KL than a wrong-token one, which is why the basin is stable
    without this term.
    """
    teacher = _blank_dominated_logits(nonblank_frames=tuple(range(40, 60)))
    all_blank = _blank_dominated_logits()
    matching = teacher.clone()

    assert dl.ctc_anchor_loss(all_blank, teacher) > dl.ctc_anchor_loss(matching, teacher)


def test_ctc_anchor_is_finite_and_small_for_a_matching_student():
    teacher = _blank_dominated_logits(nonblank_frames=tuple(range(40, 60)))
    loss = dl.ctc_anchor_loss(teacher.clone(), teacher)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_ctc_anchor_is_zero_when_the_teacher_decodes_to_nothing():
    """No target means no sequence constraint -- must not produce NaN."""
    teacher = _blank_dominated_logits()
    loss = dl.ctc_anchor_loss(_logits(), teacher)
    assert loss.item() == pytest.approx(0.0)


def test_ctc_anchor_backprops():
    teacher = _blank_dominated_logits(nonblank_frames=tuple(range(40, 60)))
    student = _blank_dominated_logits().requires_grad_(True)
    dl.ctc_anchor_loss(student, teacher).backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_a_zero_weight_removes_a_term_from_the_total():
    """Weights must actually gate terms, since the ablations are run by setting them to 0.

    Feature matching is enabled explicitly here: it is off by default now, so a test that
    wants KL+feature has to ask for it rather than inherit it.
    """
    student_logits = _logits(seed=11)
    teacher_logits = _logits(seed=12)
    student_hidden = tuple(torch.randn(2, 250, 16) for _ in range(25))
    teacher_hidden = tuple(torch.randn(2, 250, 32) for _ in range(25))
    projector = dl.FeatureProjector(16, 32, num_taps=len(dl.DEFAULT_TAP_LAYERS))

    off = dl.distillation_loss(
        student_logits, teacher_logits, student_hidden, teacher_hidden, projector,
        dl.DistillLossConfig(ctc_weight=0.0, feature_weight=1.0),
    )
    # None, not 0.0: a skipped term must be distinguishable from a perfectly-satisfied one.
    assert off.ctc_loss is None
    assert float(off.total.detach()) == pytest.approx(
        off.logit_loss + off.feature_loss, rel=1e-5
    )


# --- The shipped objective must match the one the ADR says shipped ------------------


def test_shipped_defaults_are_pure_weighted_kl():
    """The objective the code runs by default must be the one ADR-0010 documents.

    This test exists because that was once false and nobody noticed. The ADR recorded that
    the CTC and feature terms were removed -- and quoted the 84.58% -> 89.37% gain from
    removing them -- while every default in the code still enabled both. The winning run
    only got the documented objective because the flags were passed by hand; anyone running
    the documented command would have reproduced the broken recipe and burned ~12 GPU-hours
    landing back at 84.58%.
    """
    config = dl.DistillLossConfig()
    assert config.logit_weight == 1.0
    assert config.ctc_weight == 0.0, "CTC destabilises training; ADR-0010 removed it"
    assert config.feature_weight == 0.0, "0.5% gradient share; ADR-0010 removed it"
    assert config.hard_weight == 0.0


def test_default_config_computes_only_the_kl_term():
    """End-to-end: with defaults, the other three terms must contribute nothing."""
    student = _logits(seed=21)
    teacher = _logits(seed=22)
    hidden_s = tuple(torch.randn(2, 250, 16) for _ in range(25))
    hidden_t = tuple(torch.randn(2, 250, 32) for _ in range(25))
    projector = dl.FeatureProjector(16, 32, num_taps=len(dl.DEFAULT_TAP_LAYERS))

    out = dl.distillation_loss(student, teacher, hidden_s, hidden_t, projector)
    # None rather than 0.0 -- for MSE and CTC zero IS the optimum, so a disabled term
    # logged as 0.0000 reads as a perfect match in the metrics.
    assert out.ctc_loss is None
    assert out.feature_loss is None
    assert out.hard_loss is None
    assert out.feature_cosine is None
    assert float(out.total.detach()) == pytest.approx(out.logit_loss, rel=1e-5)


def test_disabled_terms_serialise_as_null_not_zero():
    """metrics.jsonl must distinguish "not computed" from "achieved the optimum"."""
    student, teacher = _logits(seed=31), _logits(seed=32)
    hidden = tuple(torch.randn(2, 250, 16) for _ in range(25))
    hidden_t = tuple(torch.randn(2, 250, 32) for _ in range(25))
    projector = dl.FeatureProjector(16, 32, num_taps=len(dl.DEFAULT_TAP_LAYERS))

    record = dl.distillation_loss(student, teacher, hidden, hidden_t, projector).as_dict()
    assert record["ctc_loss"] is None
    assert record["feature_loss"] is None
    assert record["logit_loss"] is not None


def test_an_all_zero_objective_is_refused_at_config_time():
    """Otherwise it surfaces minutes later as an autograd error inside backward().

    This is not hypothetical: an ablation script passed --logit-weight 0 --ctc-weight 0
    and relied on feature_weight still defaulting to 1.0. It does not any more, so all
    four weights were zero, `total` was a constant, and the run died in backward() after
    loading the teacher -- with a message that says nothing about loss weights.
    """
    with pytest.raises(ValueError, match="nothing to optimise"):
        dl.DistillLossConfig(logit_weight=0.0, feature_weight=0.0,
                             hard_weight=0.0, ctc_weight=0.0)


def test_a_single_enabled_term_is_a_valid_objective():
    """The guard must not reject the ablation arms themselves."""
    assert dl.DistillLossConfig(logit_weight=0.0, feature_weight=1.0).feature_weight == 1.0
    assert dl.DistillLossConfig(logit_weight=0.0, ctc_weight=1.0).ctc_weight == 1.0


# --- log-line formatting ------------------------------------------------------------
#
# These exist because the whole suite passed while both training loops were broken.
# Reporting disabled terms as None made every `:.3f` consumer raise TypeError, and under
# the default KL-only objective four of the five components are None on every step -- so
# `distill_train` would have died at its first log line. Nothing tested the print path.


def test_fmt_term_renders_a_disabled_term_without_raising():
    assert dl.fmt_term(None) == "  off"
    assert dl.fmt_term(None, 4) == "   off"


def test_fmt_term_keeps_columns_aligned():
    """A disabled term must occupy the same width as the number it replaces."""
    for digits in (3, 4):
        assert len(dl.fmt_term(None, digits)) == len(dl.fmt_term(1.5, digits))


def test_fmt_term_still_formats_numbers():
    assert dl.fmt_term(0.1234, 3) == "0.123"
    assert dl.fmt_term(0.1234, 4) == "0.1234"


def test_the_default_objective_can_render_its_own_log_line():
    """The exact failure: KL-only leaves four of five components None."""
    output = dl.distillation_loss(
        _logits(seed=1), _logits(seed=2), None, None, None, dl.DistillLossConfig(),
    )
    record = output.as_dict()
    assert record["feature_loss"] is None and record["ctc_loss"] is None
    line = (
        f"kl {dl.fmt_term(record['logit_loss'], 4)} "
        f"feat {dl.fmt_term(record['feature_loss'], 4)} "
        f"ctc {dl.fmt_term(record['ctc_loss'], 4)} "
        f"hard {dl.fmt_term(record['hard_loss'], 4)} "
        f"cos {dl.fmt_term(record['feature_cosine'])}"
    )
    assert "off" in line
