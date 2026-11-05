"""Tests for #267's decode driver: CPU only, with the gates at batch boundaries.

``decode_set`` is torch-free: the model is a function the caller supplies, so the driver's
refusals and stops are testable without the runtime. The owner's PAUSE file and the RAM floor
are honoured at batch boundaries; ``--device cuda`` is refused, because the GPU belongs to
round 3 (#263) and this screen records 0 GPU-hours.
"""

from __future__ import annotations

import json

import pytest

from tadabur.fastconformer_screen.decode import (
    DEFAULT_DEVICE,
    decode_set,
    parse_args,
    preflight,
    require_cpu,
)
from tadabur.fastconformer_screen.gates import ScreenStop
from tadabur.fastconformer_screen.sets import decode_plan
from tadabur.fastconformer_screen.verdict import Rates
from tadabur.tlog_mining.give_way import Gates


def gates(tmp_path, ram_gb: float = 64.0, disk_gb: float = 500.0) -> Gates:
    return Gates(tmp_path / "PAUSE", free_disk_gb=lambda: disk_gb,
                 available_ram_gb=lambda: ram_gb)


class Spans:
    """The spans the driver asked the model for, in order."""

    def __init__(self, on_transcribe=None):
        self.recorded, self.on_transcribe = [], on_transcribe


def test_cpu_is_the_default_device():
    assert DEFAULT_DEVICE == "cpu"
    assert parse_args([]).device == "cpu"


def test_a_gpu_decode_is_refused_with_a_message():
    with pytest.raises(ScreenStop, match="yield helper"):
        require_cpu("cuda")
    with pytest.raises(ScreenStop, match="unknown device"):
        require_cpu("tpu")
    require_cpu("cpu")


def test_decode_set_refuses_cuda_before_reading_anything(tmp_path):
    with pytest.raises(ScreenStop, match="CPU only"):
        decode_set("edits", plan=[], directory=tmp_path, out_dir=tmp_path, nemo_path=tmp_path,
                   model=object(), gates=gates(tmp_path), device="cuda")


def test_an_unknown_device_is_refused_by_the_command_line():
    with pytest.raises(SystemExit):
        parse_args(["--device", "tpu"])


def test_the_driver_decodes_only_the_plans_items(tmp_path, monkeypatch):
    """A source that holds more than the plan decodes the plan's keys alone (G6 removal)."""
    rows = [{"key": key, "audio_filename": "a.wav", "start_sample": 0, "end_sample": 2}
            for key in ("a@0:2", "b@0:2", "g6@0:2")]
    spans = _stub_source(monkeypatch, [rows[:1], rows[1:]])
    summary = decode_set("ship-a", plan=["a@0:2", "b@0:2"], directory=tmp_path,
                         out_dir=tmp_path, nemo_path=tmp_path / "m.nemo", model=object(),
                         gates=gates(tmp_path))
    keys = [json.loads(line)["key"] for line in
            (tmp_path / "ship-a.fastconformer.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (summary["stopped"], keys) == ("done", ["a@0:2", "b@0:2"])


class _Recording:
    """One source recording: the shard reader's shape, without a shard."""

    def __init__(self, rows):
        self.keys = [row["key"] for row in rows]
        self._rows = rows

    def load(self):
        return [None, None, None], self._rows


def _stub_source(monkeypatch, groups, on_transcribe=None):
    """Replace the set's shard reader and the model with fakes: no shard, no audio, no NeMo."""
    from tadabur.fastconformer_screen import decode as module
    from training import zipformer_screen_decode

    def recordings(_set, _directory):
        return iter([_Recording(rows) for rows in groups])

    spans = Spans(on_transcribe)

    def transcribe(_model, samples):
        spans.recorded.append(len(samples))
        if spans.on_transcribe is not None:
            spans.on_transcribe(len(spans.recorded))
        return f"text-{len(spans.recorded)}"

    monkeypatch.setattr(zipformer_screen_decode, "recordings", recordings)
    monkeypatch.setattr(zipformer_screen_decode, "pending_items", _pending)
    monkeypatch.setattr(module, "transcribe", transcribe)
    return spans


def _pending(source, done, keep=None):
    for recording in source:
        if not set(recording.keys) - done:
            continue
        _, rows = recording.load()
        for row in rows:
            if row["key"] in done or (keep is not None and row["key"] not in keep):
                continue
            yield row, [0.0, 0.0]


def test_the_driver_stops_at_a_batch_boundary_when_pause_appears(tmp_path, monkeypatch):
    pause = tmp_path / "PAUSE"
    rows = [{"key": f"k{n}", "audio_filename": "a.wav", "start_sample": 0, "end_sample": 2}
            for n in range(4)]
    spans = _stub_source(monkeypatch, [rows],
                         on_transcribe=lambda count: pause.touch() if count == 2 else None)
    summary = decode_set("edits", plan=[row["key"] for row in rows], directory=tmp_path,
                         out_dir=tmp_path, nemo_path=tmp_path / "m.nemo", model=object(),
                         gates=gates(tmp_path), batch_size=2)
    assert summary["stopped"] == "pause"
    assert spans.recorded == [2, 2]     # the batch in flight finished; the next never began


def test_the_driver_stops_before_starting_below_the_ram_floor(tmp_path, monkeypatch):
    spans = _stub_source(monkeypatch, [[{"key": "k0", "audio_filename": "a.wav",
                                         "start_sample": 0, "end_sample": 2}]])
    summary = decode_set("edits", plan=["k0"], directory=tmp_path, out_dir=tmp_path,
                         nemo_path=tmp_path / "m.nemo", model=object(),
                         gates=gates(tmp_path, ram_gb=1.0), batch_size=2)
    assert (summary["stopped"], spans.recorded) == ("ram", [])


class _CountingRecording:
    """A source recording whose load is the shard read: the tests count it."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.loads = 0

    def load(self):
        self.loads += 1
        return [None, None, None], [{"key": key, "audio_filename": "a.wav",
                                     "start_sample": 0, "end_sample": 2} for key in self.keys]


def _real_pending(monkeypatch, recordings):
    """The driver's own pending_items, over counting recordings (no stubbed generator)."""
    from training import zipformer_screen_decode

    monkeypatch.setattr(zipformer_screen_decode, "recordings", lambda *a: iter(recordings))


def test_an_existing_pause_file_stops_before_the_recording_is_loaded(tmp_path, monkeypatch):
    """The gate is read before advancing the source: no load (and no shard read) happens."""
    recording = _CountingRecording(["k0"])
    _real_pending(monkeypatch, [recording])
    (tmp_path / "PAUSE").touch()
    summary = decode_set("edits", plan=["k0"], directory=tmp_path, out_dir=tmp_path,
                         nemo_path=tmp_path / "m.nemo", model=object(),
                         gates=gates(tmp_path))
    assert (summary["stopped"], summary["items"], recording.loads) == ("pause", 0, 0)


def test_a_low_disk_floor_stops_before_the_recording_is_loaded(tmp_path, monkeypatch):
    recording = _CountingRecording(["k0"])
    _real_pending(monkeypatch, [recording])
    summary = decode_set("edits", plan=["k0"], directory=tmp_path, out_dir=tmp_path,
                         nemo_path=tmp_path / "m.nemo", model=object(),
                         gates=gates(tmp_path, disk_gb=1.0))
    assert (summary["stopped"], recording.loads) == ("disk", 0)


def test_a_pause_between_batches_never_loads_the_next_recording(tmp_path, monkeypatch):
    pause = tmp_path / "PAUSE"
    first, second = _CountingRecording(["k0"]), _CountingRecording(["k1"])
    _real_pending(monkeypatch, [first, second])
    from tadabur.fastconformer_screen import decode as module

    calls = []

    def transcribe(_model, _samples):
        calls.append(1)
        if len(calls) == 1:
            pause.touch()
        return "text"

    monkeypatch.setattr(module, "transcribe", transcribe)
    summary = decode_set("edits", plan=["k0", "k1"], directory=tmp_path, out_dir=tmp_path,
                         nemo_path=tmp_path / "m.nemo", model=object(),
                         gates=gates(tmp_path), batch_size=1)
    assert (summary["stopped"], len(calls)) == ("pause", 1)
    assert (first.loads, second.loads) == (1, 0)


def test_main_checks_the_gates_before_restoring_the_model(tmp_path, monkeypatch):
    from tadabur.fastconformer_screen import decode as module

    pause = tmp_path / "PAUSE"
    pause.touch()
    monkeypatch.setattr(module, "load_ctc_model",
                        lambda *a, **k: pytest.fail("the model must not be restored"))
    with pytest.raises(SystemExit) as exit_:
        module.main(["--set", "edits", "--dir", str(tmp_path), "--nemo", str(tmp_path / "m.nemo"),
                     "--out-dir", str(tmp_path / "out"), "--pause-file", str(pause)])
    assert exit_.value.code == 3
    timing = (tmp_path / "out" / "edits.fastconformer.timing.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert [json.loads(line)["stopped"] for line in timing] == ["pause"]


def test_a_stopped_invocation_still_records_its_timing(tmp_path, monkeypatch):
    """A PAUSE stop must not leave the set's timing file claiming more than it holds."""
    pause = tmp_path / "PAUSE"
    rows = [{"key": f"k{n}", "audio_filename": "a.wav", "start_sample": 0, "end_sample": 2}
            for n in range(4)]
    _stub_source(monkeypatch, [rows],
                 on_transcribe=lambda count: pause.touch() if count == 2 else None)
    decode_set("edits", plan=[row["key"] for row in rows], directory=tmp_path,
               out_dir=tmp_path, nemo_path=tmp_path / "m.nemo", model=object(),
               gates=gates(tmp_path), batch_size=2)
    timing = [json.loads(line) for line in
              (tmp_path / "edits.fastconformer.timing.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    assert [(line["stopped"], line["items"]) for line in timing] == [("pause", 2)]
    assert sum(line["items"] for line in timing) == len(
        (tmp_path / "edits.fastconformer.jsonl").read_text(encoding="utf-8").splitlines())


def test_preflight_is_the_one_refusal_path(tmp_path):
    bars = Rates(mistakes_flagged=1.0, correct_flagged=2.0, edits_heard=3.0,
                 decoys_unchanged=4.0)
    erratum = tmp_path / "erratum.md"
    erratum.write_text("| 1.0% | 2.0% | 3.0% | 4.0% |\n", encoding="utf-8")
    assert preflight("edits", bars, erratum_path=erratum) == decode_plan("edits")
