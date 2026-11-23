"""The tracer's run and output stage (#307): what a complete run is, and where the list lands.

A clip list may already live in the output directory (the CLI examples draw it there), and a
run is a result only when every drawn clip produced both arms — P1b's App-only verdict says
nothing about the Oracle arm or a clip that failed before either arm ran.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tadabur.followalong_audit import tracer
from tadabur.followalong_audit.assets import RELEASE
from tadabur.followalong_audit.oracle import OracleArm
from tadabur.followalong_audit.tracer import ClipRun, run_tracer, write_outputs

P1B = {"gate": "P1b", "verdict": "pass", "changed": [], "log_overlap_characters": 30,
       "clips": {"drawn": 1, "replayed": 1}}
META = {"issue": 307, "clips": 1}


def clip_row(index: int) -> dict:
    return {"audio_path": f"107_{index}_2416882393.flac", "audio_sha256": "a" * 64,
            "split": "clean", "shard_file": "clean-00000-of-00411", "row": index,
            "filename_label": "107:4"}


def prepared(clip: dict, *, oracle: bool = True) -> ClipRun:
    run = ClipRun(clip=clip, samples=16_000, word_count=2, surah=107, ayah=4,
                  wav_sha256="b" * 64, wall_seconds=1.0, second_wall_seconds=1.0,
                  deterministic=True, inference_count=1, failure_count=0)
    run.app = [{"hop": "", "overlap": "بَ", "flush": False},
               {"hop": "بَ", "overlap": "تِ", "flush": False, "stop": True}]
    if oracle:
        run.oracle = OracleArm(checks=[{"hop": "", "overlap": "بَ", "flush": False}],
                               aligned=True)
    return run


def write_clip_list(path: Path, count: int) -> Path:
    path.write_text("".join(json.dumps(clip_row(index), sort_keys=True) + "\n"
                            for index in range(count)), encoding="utf-8")
    return path


def test_a_clip_list_already_in_the_output_directory_is_not_copied_onto_itself(tmp_path):
    out = tmp_path / "tracer"
    out.mkdir()
    clip_list = write_clip_list(out / "clips.jsonl", 1)
    before = clip_list.read_bytes()
    write_outputs(out, [], [prepared(clip_row(0))], P1B, META, 40, None, {}, clip_list)
    assert clip_list.read_bytes() == before
    assert (out / "clip_runs.jsonl").exists()
    assert (out / "run_note.md").exists()


def test_a_clip_list_outside_the_output_directory_is_copied_byte_for_byte(tmp_path):
    clip_list = write_clip_list(tmp_path / "clips.jsonl", 1)
    out = tmp_path / "out"
    out.mkdir()
    write_outputs(out, [], [prepared(clip_row(0))], P1B, META, 40, None, {}, clip_list)
    assert (out / "clips.jsonl").read_bytes() == clip_list.read_bytes()


class FakeHarness:
    quran_db = Path("quran.db")

    @classmethod
    def locate(cls, commit: str) -> "FakeHarness":
        return cls()

    def run(self, requests):
        return None, []


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """``run_tracer`` with the build, release, engine and per-clip preparation replaced: only
    the completion rule under test is real."""
    monkeypatch.setattr(tracer, "harness_record", lambda *args: {"hifzguide_release": RELEASE})
    monkeypatch.setattr(tracer, "stage_assets", lambda *args, **kwargs: SimpleNamespace(
        directory=tmp_path / "staged", model="model", model_sha256="model-sha", checksums={}))
    monkeypatch.setattr(tracer, "driver_path", lambda *args: tmp_path / "driver")
    monkeypatch.setattr(tracer, "Harness", FakeHarness)
    monkeypatch.setattr(tracer, "MurajaText", lambda quran_db: object())
    monkeypatch.setattr(tracer, "load_word_times", lambda *args: {})
    monkeypatch.setattr(tracer, "word_traces", lambda *args: [])
    return tmp_path


def tracer_run(clips, stubbed, monkeypatch):
    clip_list = write_clip_list(stubbed / "clips.jsonl", len(clips))
    monkeypatch.setattr(tracer, "changed_decisions", lambda full, clipped: [{"word": 1}])
    return run_tracer(clips, clip_list=clip_list, wav_dir=stubbed, assets_dir=stubbed,
                      manifest={}, wav_manifest={}, out=stubbed / "out", latency_ms=40)


def test_a_run_whose_clips_all_produced_both_arms_is_complete(stubbed, monkeypatch):
    clips = [clip_row(0), clip_row(1)]
    monkeypatch.setattr(tracer, "prepare_clip", lambda clip, *args: prepared(clip))
    summary = tracer_run(clips, stubbed, monkeypatch)
    assert summary["verdict"] == "pass"
    assert summary["complete"] is True
    assert summary["failed"] == 0


def test_an_oracle_failure_is_not_a_complete_run_even_when_p1b_passes(stubbed, monkeypatch):
    clips = [clip_row(0), clip_row(1)]
    failure = "oracle: the clip has 1 word times for 2 words"

    def prepare(clip, *args):
        if clip["audio_path"] == clips[1]["audio_path"]:
            run = prepared(clip, oracle=False)
            run.failure = failure
            return run
        return prepared(clip)

    monkeypatch.setattr(tracer, "prepare_clip", prepare)
    summary = tracer_run(clips, stubbed, monkeypatch)
    assert summary["verdict"] == "pass"
    assert summary["complete"] is False
    assert summary["failed"] == 1
    note = (stubbed / "out" / "run_note.md").read_text(encoding="utf-8")
    assert "1/2 clips produced both arms" in note
    assert clips[1]["audio_path"] in note
    rows = [json.loads(line) for line in
            (stubbed / "out" / "clip_runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["failure"] for row in rows][1] == failure
