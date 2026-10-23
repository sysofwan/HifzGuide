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


# --- WSD schedule -------------------------------------------------------------------


def test_wsd_holds_the_peak_through_the_stable_phase():
    """The point of the probe: the rate must not drift while the plateau is measured."""
    held = [dt.lr_lambda(s, 500, 10000, 2000) for s in range(500, 8000, 250)]
    assert held == [pytest.approx(1.0)] * len(held)


def test_wsd_warms_up_before_holding():
    assert dt.lr_lambda(0, 500, 10000, 2000) == pytest.approx(1 / 500)
    assert dt.lr_lambda(499, 500, 10000, 2000) == pytest.approx(1.0)


def test_wsd_cooldown_starts_exactly_cooldown_steps_from_the_end():
    assert dt.lr_lambda(7999, 500, 10000, 2000) == pytest.approx(1.0)
    assert dt.lr_lambda(8000, 500, 10000, 2000) == pytest.approx(1.0)
    assert dt.lr_lambda(9000, 500, 10000, 2000) == pytest.approx(0.505, abs=1e-6)


def test_wsd_lands_on_the_same_floor_as_cosine():
    """Common terminal rate is what makes a WSD arm comparable to a cosine one."""
    assert dt.lr_lambda(10000, 500, 10000, 2000) == pytest.approx(dt.LR_FLOOR, abs=1e-9)


def test_wsd_is_monotonic_and_clamped_past_the_end():
    values = [dt.lr_lambda(s, 500, 10000, 2000) for s in range(500, 12000, 100)]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))
    assert dt.lr_lambda(20000, 500, 10000, 2000) == pytest.approx(dt.LR_FLOOR, abs=1e-9)


def test_a_cooldown_longer_than_the_run_still_warms_up_first():
    """Clamping the cooldown start protects the warmup rather than overrunning it."""
    assert dt.lr_lambda(499, 500, 10000, 50000) == pytest.approx(1.0)
    assert dt.lr_lambda(500, 500, 10000, 50000) == pytest.approx(1.0)
    assert dt.lr_lambda(10000, 500, 10000, 50000) == pytest.approx(dt.LR_FLOOR, abs=1e-9)


def test_zero_cooldown_is_the_cosine_schedule():
    """The default must be bit-identical to the schedule every prior run was trained on."""
    for step in (0, 1999, 5000, 20000, 39999, 40000, 50000):
        assert dt.lr_lambda(step, 2000, 40000, 0) == dt.lr_lambda(step, 2000, 40000)


def test_cooldown_steps_is_a_resume_critical_field():
    """Restoring a WSD optimiser state under a cosine lambda silently changes the run."""
    saved = dt.asdict(_config(cooldown_steps=2000))
    with pytest.raises(SystemExit, match="cooldown_steps"):
        dt.check_resume_compatible(saved, _config(cooldown_steps=0))


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
    """A new loss term added without guarding it would resume silently divergent.

    Derived from TrainConfig rather than hand-listed. The hand-written version had already
    gone stale -- it omitted `hard_weight` -- so it would have passed for a newly added
    `*_weight` that nobody added to RESUME_CRITICAL_FIELDS, which is the exact regression
    the docstring claims it catches.
    """
    from dataclasses import fields

    weights = {f.name for f in fields(dt.TrainConfig) if f.name.endswith("_weight")}
    assert weights, "no *_weight fields found -- has TrainConfig been renamed?"
    assert weights <= set(dt.RESUME_CRITICAL_FIELDS), (
        f"unguarded loss weights: {sorted(weights - set(dt.RESUME_CRITICAL_FIELDS))}"
    )


# --- Augmentation (#94) -------------------------------------------------------------


def test_resume_refuses_a_changed_augmentation_recipe():
    """The control arm resumed as the augmented one (or under another recipe or noise pool)
    would splice two experiments onto one optimiser state."""
    recipe = {"version": "aug94-mixture-v1", "noise_pool_sha256": "a" * 64}
    for saved, current in (({}, recipe), (recipe, {}),
                           (recipe, {**recipe, "noise_pool_sha256": "b" * 64})):
        with pytest.raises(SystemExit, match="augment"):
            dt.check_resume_compatible(dt.asdict(_config(augment=saved)),
                                       _config(augment=current))
    dt.check_resume_compatible(dt.asdict(_config(augment=recipe)), _config(augment=dict(recipe)))


def test_the_data_order_and_exclusion_are_resume_critical_and_the_pool_path_is_not():
    for field in ("augment", "shuffle_buffer", "exposure_use", "stream_shards", "seed"):
        assert field in dt.RESUME_CRITICAL_FIELDS
    assert "noise_pool" not in dt.RESUME_CRITICAL_FIELDS  # its content is in the recipe's hash
    with pytest.raises(SystemExit, match="shuffle_buffer"):
        dt.check_resume_compatible(dt.asdict(_config(shuffle_buffer=1024)),
                                   _config(shuffle_buffer=512))
    with pytest.raises(SystemExit, match="exposure_use"):
        dt.check_resume_compatible(dt.asdict(_config(exposure_use="aug.control")),
                                   _config(exposure_use="aug.training"))


def test_a_legacy_checkpoint_resumes_only_as_the_clean_run_it_was():
    """A checkpoint from before #94 has none of the new fields: it ran clean, with no
    exposure use and a 1024-window buffer, and resuming it as anything else is refused."""
    legacy = dt.asdict(_config())
    for field in dt.LEGACY_FIELD_VALUES:
        legacy.pop(field)
    dt.check_resume_compatible(legacy, _config(shuffle_buffer=1024))  # as it ran
    for changed, match in (
        (_config(shuffle_buffer=1024, augment={"version": "aug94-mixture-v1"}), "augment"),
        (_config(shuffle_buffer=1024, exposure_use="aug.training"), "exposure_use"),
        (_config(), "shuffle_buffer"),  # today's default, 512: a different data order
    ):
        with pytest.raises(SystemExit, match=match):
            dt.check_resume_compatible(legacy, changed)


def test_a_new_checkpoint_is_not_read_through_the_legacy_values():
    recipe = {"version": "aug94-mixture-v1"}
    saved = dt.asdict(_config(augment=recipe, exposure_use="aug.training"))
    dt.check_resume_compatible(saved, _config(augment=recipe, exposure_use="aug.training"))
    with pytest.raises(SystemExit, match="augment"):
        dt.check_resume_compatible(saved, _config(exposure_use="aug.training"))


def test_a_warm_start_reads_weights_and_never_the_source_runs_config(tmp_path):
    """--init-from starts a new run from h448's weights (a legacy checkpoint), so it must
    not be held to that run's config the way a resume is."""
    student, projector = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    legacy = dt.asdict(_config())
    for field in dt.LEGACY_FIELD_VALUES:
        legacy.pop(field)
    path = tmp_path / "checkpoint.pt"
    torch.save({"step": 40_000, "student": student.state_dict(),
                "projector": projector.state_dict(), "config": legacy}, path)
    fresh, fresh_projector = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    assert dt.load_student_weights(path, fresh, fresh_projector, "cpu") == 40_000
    assert torch.equal(fresh.weight, student.weight)


def test_a_streamed_pair_splits_into_teacher_and_student_inputs():
    clean, augmented = torch.zeros(2, 250, 160), torch.ones(2, 250, 160)
    assert dt.teacher_student_inputs([clean, augmented]) == (clean, augmented)
    teacher, student = dt.teacher_student_inputs(clean)  # the staged corpus: one tensor
    assert teacher is clean and student is clean


def test_the_teacher_reads_clean_input_and_the_student_augmented(monkeypatch):
    seen = {}

    class Model(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, features, output_hidden_states=False, return_dict=True):
            seen[self.name] = features
            return {"logits": {"phonemes": features.sum(-1, keepdim=True)}}

    monkeypatch.setattr(dt, "distillation_loss", lambda **kwargs: kwargs)
    clean, augmented = torch.zeros(1, 250, 160), torch.ones(1, 250, 160)
    out = dt.forward_step(clean, augmented, Model("teacher"), Model("student"), None,
                          _config().loss_config())
    assert seen["teacher"] is clean and seen["student"] is augmented
    assert torch.equal(out["teacher_logits"], clean.sum(-1, keepdim=True))
    assert torch.equal(out["student_logits"], augmented.sum(-1, keepdim=True))


# --- Weight averaging ---


def test_weight_average_tracks_a_constant_exactly():
    import torch

    from training.distill_train import WeightAverage

    model = torch.nn.Linear(4, 3)
    averager = WeightAverage(model, 0.9)
    for _ in range(200):
        averager.update(model)
    for name, value in model.state_dict().items():
        assert torch.allclose(averager.shadow[name], value.float(), atol=1e-5)


def test_weight_average_ramps_in_so_it_is_not_pinned_to_the_random_init():
    """A fixed 0.999 decay leaves the average ~63% initialisation after 1000 steps.

    That reads as "averaging made the model worse" for the first several thousand steps of
    every run, for a reason that has nothing to do with averaging.
    """
    import torch

    from training.distill_train import WeightAverage

    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    averager = WeightAverage(model, 0.999)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
    for _ in range(20):
        averager.update(model)
    # With a fixed 0.999 this would still be ~0.02; the ramp gets it most of the way there.
    assert float(averager.shadow["weight"].mean()) > 0.5


def test_weight_average_refuses_a_decay_outside_the_open_unit_interval():
    import pytest
    import torch

    from training.distill_train import WeightAverage

    for decay in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            WeightAverage(torch.nn.Linear(2, 2), decay)


def test_apply_to_substitutes_only_the_floating_point_tensors():
    import torch

    from training.distill_train import WeightAverage

    model = torch.nn.Linear(2, 2)
    averager = WeightAverage(model, 0.5)
    with torch.no_grad():
        model.weight.fill_(5.0)
    averager.update(model)
    substituted = averager.apply_to(model)
    assert set(substituted) == set(model.state_dict())
    assert not torch.allclose(substituted["weight"], model.weight)


def test_ema_decay_is_a_resume_critical_field():
    """Resuming at a different decay splices two different averages into one tensor."""
    from training.distill_train import RESUME_CRITICAL_FIELDS

    assert "ema_decay" in RESUME_CRITICAL_FIELDS


def test_the_checkpoint_stores_the_average_once(tmp_path):
    """Two copies of 85.5M values is 342 MB a save on a box that runs at 93% full."""
    import torch

    from training.distill_train import TrainConfig, WeightAverage, save_checkpoint

    model = torch.nn.Linear(4, 3)
    averager = WeightAverage(model, 0.9)
    averager.update(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, 10, model, model, optimizer, scheduler,
        TrainConfig(preset="h384", audio_root="", out_dir=str(tmp_path), ema_decay=0.9),
        averager,
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert "ema_state" in state
    assert "student_ema" not in state


def test_the_averaged_weights_round_trip_through_the_loader(tmp_path):
    """Exercise `load_student_from_checkpoint`, not a copy of its expression.

    The previous version of this test pasted the reconstruction out of the loader into the
    test and asserted that copy matched `apply_to`. Changing the loader's `if name in shadow`
    to `if name in live` left it passing.
    """
    import torch

    from training.decoding import load_student_from_checkpoint
    from training.distill_student import PRESETS, build_student
    from training.distill_train import TrainConfig, WeightAverage, save_checkpoint

    spec = PRESETS["h256"]
    student = build_student(spec)
    averager = WeightAverage(student, 0.5)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.mul_(0.0).add_(0.25)
    averager.update(student)

    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, 10, student, student, optimizer, scheduler,
        TrainConfig(preset="h256", audio_root="", out_dir=str(tmp_path), ema_decay=0.5),
        averager,
    )

    device = torch.device("cpu")
    loaded, _, step = load_student_from_checkpoint(path, device, use_ema=True)
    assert step == 10
    shadow = averager.state_dict()["shadow"]
    for name, value in loaded.state_dict().items():
        if name in shadow:
            assert torch.allclose(value, shadow[name].to(value.dtype)), name


def test_asking_for_averaged_weights_a_run_never_kept_is_an_error(tmp_path):
    """Silently falling back to the live weights would report the wrong model's number."""
    import pytest
    import torch

    from training.decoding import load_student_from_checkpoint
    from training.distill_student import PRESETS, build_student
    from training.distill_train import TrainConfig, save_checkpoint

    student = build_student(PRESETS["h256"])
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, 5, student, student, optimizer, scheduler,
        TrainConfig(preset="h256", audio_root="", out_dir=str(tmp_path)), None,
    )
    with pytest.raises(SystemExit, match="no averaged weights"):
        load_student_from_checkpoint(path, torch.device("cpu"), use_ema=True)


# --- the warm start (#178) ----------------------------------------------------------

# #174 fixes C95 as the *EMA* weights, but `--init-from` loaded only the live ones: the two
# sit in one checkpoint, and a run that silently warm-started from the wrong one would be a
# different experiment under the same name. These pin the choice and its failure mode.


class _Shaped(torch.nn.Module):
    """A student-shaped module: a float weight, a float buffer and an int buffer.

    The buffers are the point. ``WeightAverage.apply_to`` substitutes the averaged shadow's
    floating tensors and keeps everything else from the live state, so an averaged warm
    start has to move ``running`` and leave ``count`` alone; a loader that rebuilt the
    module from the shadow alone would silently drop the latter.
    """

    def __init__(self, value: float, count: int = 1) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        torch.nn.init.constant_(self.lin.weight, value)
        torch.nn.init.constant_(self.lin.bias, value)
        self.register_buffer("running", torch.full((2,), value))
        self.register_buffer("count", torch.tensor(count))


def _warm_start(tmp_path, live: float, averaged: float | None, *, legacy: bool = False):
    """A checkpoint of a run that reached step 10,000, optionally with an average."""
    student, projector = _Shaped(live, count=7), torch.nn.Linear(2, 2)
    state = {
        "step": 10_000,
        "student": student.state_dict(),
        "projector": projector.state_dict(),
        "config": {},
    }
    if legacy:
        averaged_state = {
            name: (torch.full_like(value, averaged) if value.is_floating_point() else value)
            for name, value in student.state_dict().items()
        }
        state["student_ema"] = averaged_state
    elif averaged is not None:
        state["ema_state"] = {
            "decay": 0.999,
            "step": 10_000,
            "shadow": {
                name: torch.full_like(value, averaged)
                for name, value in student.state_dict().items()
                if value.is_floating_point()
            },
        }
    path = tmp_path / "checkpoint.pt"
    torch.save(state, path)
    return path


def test_a_warm_start_defaults_to_the_live_weights(tmp_path):
    """The default is live even when the checkpoint carries an average: #94 unchanged."""
    path = _warm_start(tmp_path, live=0.5, averaged=0.1)
    student, projector = _Shaped(9.0), torch.nn.Linear(2, 2)
    assert dt.load_student_weights(path, student, projector, "cpu") == 10_000
    assert torch.allclose(student.lin.weight, torch.full((2, 2), 0.5))
    assert torch.allclose(student.running, torch.full((2,), 0.5))


def test_asking_for_the_averaged_weights_loads_the_averaged_shadow(tmp_path):
    path = _warm_start(tmp_path, live=0.5, averaged=0.1)
    student, projector = _Shaped(9.0), torch.nn.Linear(2, 2)
    dt.load_student_weights(path, student, projector, "cpu", use_ema=True)
    assert torch.allclose(student.lin.weight, torch.full((2, 2), 0.1))
    assert torch.allclose(student.lin.bias, torch.full((2,), 0.1))
    assert torch.allclose(student.running, torch.full((2,), 0.1))  # a float buffer averages
    assert student.count.item() == 7  # an integer buffer does not: it comes from the live state


def test_a_legacy_materialised_average_is_read_verbatim(tmp_path):
    path = _warm_start(tmp_path, live=0.5, averaged=0.1, legacy=True)
    student, projector = _Shaped(9.0), torch.nn.Linear(2, 2)
    dt.load_student_weights(path, student, projector, "cpu", use_ema=True)
    assert torch.allclose(student.lin.weight, torch.full((2, 2), 0.1))
    assert torch.allclose(student.running, torch.full((2,), 0.1))


def test_an_averaged_warm_start_without_an_average_is_refused(tmp_path):
    """Falling back to the live weights would warm-start a different model than #174 names."""
    path = _warm_start(tmp_path, live=0.5, averaged=None)
    student, projector = _Shaped(9.0), torch.nn.Linear(2, 2)
    with pytest.raises(SystemExit, match="averaged"):
        dt.load_student_weights(path, student, projector, "cpu", use_ema=True)


def test_the_init_weights_flag_defaults_to_live_and_accepts_ema():
    parser = dt.build_parser()
    assert parser.parse_args(["--audio-root", "audio"]).init_weights == "live"
    assert parser.parse_args(
        ["--audio-root", "audio", "--init-weights", "ema"]
    ).init_weights == "ema"
    with pytest.raises(SystemExit):
        parser.parse_args(["--audio-root", "audio", "--init-weights", "shadow"])


def test_the_init_weights_choice_is_not_resume_critical():
    """It initialises a fresh run; it is not part of what a resume must reproduce."""
    import dataclasses

    assert "init_weights" not in dt.RESUME_CRITICAL_FIELDS
    assert "init_weights" not in {field.name for field in dataclasses.fields(dt.TrainConfig)}
