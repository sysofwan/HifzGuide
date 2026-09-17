"""Tests for the run-level machinery: the LR schedule and the resume guard.

Both are things that only misbehave hours into a run, which is exactly when nobody is
watching, so they are pinned here rather than discovered on a 12-hour job.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training import distill_train as dt


def _config(**overrides) -> dt.TrainConfig:
    base = dict(preset="h384", audio_root="/audio", out_dir="/runs/x")
    base.update(overrides)
    return dt.TrainConfig(**base)


# --- LR schedule --------------------------------------------------------------------


def test_warmup_ramps_from_near_zero_to_one():
    assert dt.lr_lambda(0, 2000, 40000) == pytest.approx(1 / 2000)
    assert dt.lr_lambda(1999, 2000, 40000) == pytest.approx(1.0)


def test_cosine_decays_to_the_floor_not_to_zero():
    """Decaying to exactly 0 stops learning entirely before the run ends."""
    end = dt.lr_lambda(40000, 2000, 40000)
    assert end == pytest.approx(0.01, abs=1e-6)


def test_schedule_is_monotonic_after_warmup():
    values = [dt.lr_lambda(s, 2000, 40000) for s in range(2000, 40000, 1000)]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))


def test_schedule_is_clamped_past_the_end():
    """A resume that overshoots must not produce a negative or rising LR."""
    assert dt.lr_lambda(50000, 2000, 40000) == pytest.approx(0.01, abs=1e-6)


def test_zero_warmup_does_not_divide_by_zero():
    assert dt.lr_lambda(0, 0, 1000) > 0


# --- Resume guard -------------------------------------------------------------------


def test_identical_config_resumes():
    config = _config()
    dt.check_resume_compatible(dt.asdict(config), config)   # must not raise


def test_resume_rejects_a_changed_loss_weight():
    """The case that motivated this: resuming a KL-only run with the CTC anchor on."""
    saved = dt.asdict(_config(ctc_weight=0.0))
    with pytest.raises(SystemExit, match="ctc_weight"):
        dt.check_resume_compatible(saved, _config(ctc_weight=1.0))


def test_resume_rejects_a_changed_learning_rate():
    saved = dt.asdict(_config(learning_rate=3e-4))
    with pytest.raises(SystemExit, match="learning_rate"):
        dt.check_resume_compatible(saved, _config(learning_rate=1e-4))


def test_resume_rejects_a_changed_preset():
    saved = dt.asdict(_config(preset="h384"))
    with pytest.raises(SystemExit, match="preset"):
        dt.check_resume_compatible(saved, _config(preset="h256"))


def test_resume_rejects_a_changed_step_budget():
    """steps feeds the cosine schedule, so changing it reshapes the restored LR curve."""
    saved = dt.asdict(_config(steps=40000))
    with pytest.raises(SystemExit, match="steps"):
        dt.check_resume_compatible(saved, _config(steps=60000))


def test_resume_allows_operational_changes():
    """Worker count and logging cadence do not change the experiment."""
    saved = dt.asdict(_config(num_workers=8, log_every=50, eval_every=2000))
    current = _config(num_workers=4, log_every=100, eval_every=1000, eval_batches=5)
    dt.check_resume_compatible(saved, current)   # must not raise


def test_resume_reports_every_mismatch_at_once():
    """One run, one fix -- not a guessing game one field at a time."""
    saved = dt.asdict(_config(learning_rate=3e-4, ctc_weight=0.0, seed=1))
    with pytest.raises(SystemExit) as excinfo:
        dt.check_resume_compatible(saved, _config(learning_rate=1e-4, ctc_weight=1.0, seed=2))
    message = str(excinfo.value)
    assert "learning_rate" in message
    assert "ctc_weight" in message
    assert "seed" in message


def test_resume_tolerates_a_checkpoint_missing_a_field():
    """An older checkpoint predating a config field must still resume."""
    saved = dt.asdict(_config())
    saved.pop("ctc_weight")
    dt.check_resume_compatible(saved, _config())   # must not raise


def test_every_loss_weight_is_guarded():
    """A new loss term added without guarding it would resume silently divergent."""
    loss_fields = {
        "logit_weight",
        "feature_weight",
        "ctc_weight",
        "temperature",
        "nonblank_weight",
        "confirm_weight",
    }
    assert loss_fields <= set(dt.RESUME_CRITICAL_FIELDS)
