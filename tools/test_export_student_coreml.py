"""The release ships the EMA weights -- the ones #95 actually scored -- not the live ones.

`export_student_coreml` read `state["student"]` unconditionally. Exporting the #95 candidate
that way would ship a model nothing scored, and nothing would fail: the package is valid, the
size is right and the traces agree with the eager forward of the *wrong* weights. The choice
lives at the one seam the exporter owns -- the state dict it hands to the tracer -- so it is
pinned here rather than left to the export log.

The fixtures use an `h256` checkpoint because the point is weight selection, not architecture:
live weights are 0.25 and the averaged ones are a decayed average from 0.0, so the two cannot
be mistaken for each other.
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

import export_student_coreml
import verify_student_export
from training.distill_student import PRESETS, build_student
from training.distill_train import TrainConfig, WeightAverage, save_checkpoint


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    """A checkpoint carrying both weight sets, as a run with ``--ema-decay`` writes one."""
    path = tmp_path_factory.mktemp("checkpoint") / "checkpoint.pt"

    student = build_student(PRESETS["h256"])
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.zero_()
    averager = WeightAverage(student, 0.5)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.fill_(0.25)
    averager.update(student)

    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    save_checkpoint(
        path, 10, student, student, optimizer, scheduler,
        TrainConfig(preset="h256", audio_root="", out_dir=str(path.parent), ema_decay=0.5),
        averager,
    )

    state = torch.load(path, map_location="cpu", weights_only=False)
    return path, state["student"], state["ema_state"]["shadow"]


def _assert_shadow_weights(loaded: dict, live: dict, shadow: dict) -> None:
    for name in shadow:
        assert torch.allclose(loaded[name], shadow[name].to(loaded[name].dtype)), name
    # A loader that returned the live weights would satisfy the shape contract and the size
    # budget, so assert the two sets actually differ and the export followed the asked-for one.
    assert any(not torch.allclose(loaded[name], live[name]) for name in shadow)


def test_the_exported_weights_are_the_ema_shadow_not_the_live_weights(candidate):
    path, live, shadow = candidate
    model, preset, step = export_student_coreml.load_student(path, None, use_ema=True)

    assert (preset, step) == ("h256", 10)
    _assert_shadow_weights(model.state_dict(), live, shadow)


def test_exporting_a_checkpoint_without_averaged_weights_fails_loudly(candidate, tmp_path):
    """Silently shipping the live weights under an ``--ema`` flag is the failure this guards."""
    path, _, _ = candidate
    state = torch.load(path, map_location="cpu", weights_only=False)
    del state["ema_state"]
    stripped = tmp_path / "live_only.pt"
    torch.save(state, stripped)

    with pytest.raises(SystemExit, match="no averaged weights"):
        export_student_coreml.load_student(stripped, None, use_ema=True)


class _StopBeforeTracing(Exception):
    """Halts the export CLI after it has chosen and built the student."""


def test_the_export_cli_threads_ema_into_the_model_it_would_trace(candidate, tmp_path, monkeypatch):
    """The flag, not just the loader, has to reach the weights the tracer receives."""
    path, live, shadow = candidate
    handed = {}

    def capture(model, traced_path):
        handed["model"] = model
        raise _StopBeforeTracing

    monkeypatch.setattr(export_student_coreml, "trace_student", capture)
    monkeypatch.setattr(
        sys, "argv",
        [
            "export_student_coreml.py", "--checkpoint", str(path), "--ema",
            "--output-dir", str(tmp_path / "export"),
        ],
    )
    with pytest.raises(_StopBeforeTracing):
        export_student_coreml.main()

    _assert_shadow_weights(handed["model"].state_dict(), live, shadow)


@pytest.mark.parametrize(
    "module",
    [export_student_coreml, verify_student_export],
    ids=["export_student_coreml", "verify_student_export"],
)
def test_both_clis_expose_the_ema_flag(module, monkeypatch, capsys):
    """The verifier must be able to load the same weights the release exports."""
    monkeypatch.setattr(sys, "argv", [f"{module.__name__}.py", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        module.main()

    assert exit_info.value.code == 0
    assert "--ema" in capsys.readouterr().out
