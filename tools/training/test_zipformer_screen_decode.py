"""Tests for the screen batch's projection and its stop gate (#245 architect r2, AC 6).

The batch decodes every set from its own source, but times the run first: 20 qdat items through
the verdict store, projected over each job's decoded audio at the measured rate. A projection
past 12 h stops the batch before any screen decode — the stop is an exit code and a recorded
projection, so the launcher's one exit file says the batch never started.

The arithmetic here is the compute table's too: each job's decoded audio is
:func:`training.zipformer_decode.arms_decoded_samples` over the set's own items, divided by the
probe's throughput. Torch-free: no ONNX file, no audio, no download.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training import zipformer_screen_decode as screen


# --- the projection ----------------------------------------------------------------------


def test_the_projection_divides_each_job_s_decoded_audio_by_the_probe_s_rate():
    timing = {"decoded_audio_seconds": 3600.0, "wall_seconds": 60.0}  # 60 s decoded per second
    record = screen.project(timing, {"qdat/w1": 7200, "edits/decode": 1800})
    assert record["decoded_audio_per_second"] == pytest.approx(60.0)
    assert record["jobs"] == {
        "qdat/w1": {"decoded_audio_seconds": 7200, "projected_seconds": 120.0},
        "edits/decode": {"decoded_audio_seconds": 1800, "projected_seconds": 30.0},
    }
    assert record["projected_seconds"] == pytest.approx(150.0)
    assert record["projected_hours"] == pytest.approx(0.04, abs=0.01)
    assert record["stop"] is False


def test_the_horizon_is_twelve_hours_and_only_an_excess_stops_the_batch():
    timing = {"decoded_audio_seconds": 1.0, "wall_seconds": 1.0}
    assert screen.HORIZON_HOURS == 12.0
    assert screen.project(timing, {"a": 12 * 3600})["stop"] is False
    assert screen.project(timing, {"b": 12 * 3600 + 1})["stop"] is True


def test_a_probe_that_measured_no_decode_projects_nothing():
    with pytest.raises(ValueError):
        screen.project({"decoded_audio_seconds": 0.0, "wall_seconds": 1.0}, {"a": 1})
    with pytest.raises(ValueError):
        screen.project({"decoded_audio_seconds": 1.0, "wall_seconds": 0.0}, {"a": 1})


def test_decoded_audio_is_reported_in_seconds_never_in_samples():
    # 16 kHz: one 60 s item is 960,000 samples, and its one stream is 60 s of decode
    assert screen.decode_seconds("qdat", "decode", [960_000]) == 60.0
    # the verdict store's two arms re-decode 38.44x that item's audio, in seconds not samples
    assert screen.decode_seconds("qdat", "w1", [960_000]) == pytest.approx(36_902_400 / 16_000)


def test_the_probe_s_rate_is_audio_seconds_per_wall_second():
    # 600 s of audio decoded in 60 s is 10x realtime, so a 6,000 s job projects 600 s
    record = screen.project({"decoded_audio_seconds": 600.0, "wall_seconds": 60.0},
                            {"a": 6000.0})
    assert record["decoded_audio_per_second"] == pytest.approx(10.0)
    assert record["jobs"]["a"] == {"decoded_audio_seconds": 6000.0, "projected_seconds": 600.0}


# --- the batch: the probe gate, then every job -------------------------------------------


def test_the_batch_stops_before_any_decode_when_the_projection_is_past_the_horizon(
        tmp_path, monkeypatch):
    decoded = []
    monkeypatch.setattr(screen, "probe", lambda *args, **kwargs: {
        "stop": True, "projected_hours": 13.4, "horizon_hours": 12.0})
    monkeypatch.setattr(screen, "run", lambda *args, **kwargs: decoded.append(args[:2]))
    dirs = {set_name: tmp_path / set_name for set_name in screen.SETS}
    exit_code = screen.batch(tmp_path, dirs=dirs, probe_record=tmp_path / "probe.json",
                             workers=2)
    assert exit_code == screen.PROBE_STOP_EXIT == 3
    assert decoded == []


def test_the_batch_decodes_every_job_of_every_set_when_the_projection_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "probe", lambda *args, **kwargs: {"stop": False})
    decoded = []
    monkeypatch.setattr(screen, "run",
                        lambda set_name, store, *args, **kwargs: decoded.append((set_name, store)))
    dirs = {set_name: tmp_path / set_name for set_name in screen.SETS}
    assert screen.batch(tmp_path, dirs=dirs, probe_record=tmp_path / "probe.json",
                        workers=2) == 0
    # the warm-started verdict arms only on the two sets Muraja's engine grades
    assert set(decoded) == {
        ("benchmark", "decode"), ("edits", "decode"), ("qdat", "decode"), ("qdat", "w1"),
        ("ship-a", "decode"), ("ship-a", "w1"), ("qud-other", "decode")}


def test_a_batch_that_leaves_a_set_without_its_directory_is_refused():
    args = SimpleNamespace(work_dir="fetch", edits_dir=None, qdat_cache="qdat",
                           ship_a_dir="clips", qud_cache="qud")
    with pytest.raises(SystemExit):
        screen.directories(args)
