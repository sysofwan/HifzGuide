"""Acceptance tests for ``distill_train``'s ``--mix`` wiring (#228), written before the code.

Locked: later turns may add tests but never weaken or rewrite these.

Contract this file pins
-----------------------
* ``TrainConfig`` gains ``mix``, ``tlog_schedule``, ``tlog_schedule_sha256``, ``frame_manifest``
  and ``frame_manifest_sha256`` (all strings, ``""`` when the run has no mix). ``mix``,
  ``tlog_schedule_sha256`` and ``frame_manifest_sha256`` are resume-critical, and a checkpoint
  that predates them reads as ``""`` (the all-Tadabur control); the two paths are operational.
* ``--mix``, ``--tlog-schedule`` and ``--frame-manifest`` flags (default ``""``). ``main`` hashes
  the schedule and the frame manifest into the config (``tadabur.tlog_index.sha256_file``) and
  refuses (``SystemExit``) a mix without ``--stream-shards``, ``--tlog-schedule`` or
  ``--frame-manifest``, a mix whose size is not ``--batch-size``, and a frame manifest whose shard
  list is not ``--stream-shards`` or whose seed is not ``--seed``.
* ``build_dataloaders``: without a mix the train dataset is a plain ``StreamingWindowDataset``,
  as today; with one it is a ``distill_mixed.MixedWindowDataset`` around exactly that Tadabur
  dataset, the committed schedule read from ``--tlog-schedule``, and the tlog cache and
  consumed-keys log under the out dir (``distill_mixed.TLOG_CACHE_DIRNAME`` and
  ``TLOG_RUN_LOG_NAME``).
* ``teacher_student_inputs`` accepts the mixed stream's ``(teacher, student, source)`` batch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tadabur.tlog_index import sha256_file  # noqa: E402
from tadabur.tlog_schedule import SCHEDULE_PATH, read_schedule  # noqa: E402
from training import distill_stream as ds  # noqa: E402
from training import distill_train as dt  # noqa: E402

#: The head of the control's frame, so the specs are valid (unreserved) shard numbers.
SPEC = "21-23"
MIX = "tlog=24,tadabur=8"


def _config(tmp_path: Path, **overrides) -> dt.TrainConfig:
    audio = tmp_path / "val_clips"
    audio.mkdir(exist_ok=True)
    (audio / "clip_0000.wav").write_bytes(b"RIFF")  # discovered, never decoded here
    values = dict(preset="h448", audio_root=str(audio), out_dir=str(tmp_path / "run"),
                  batch_size=32, stream_shards=SPEC, seed=2026, num_workers=1,
                  exposure_use="consonant2.training")
    values.update(overrides)
    return dt.TrainConfig(**values)


def _mixed_config(tmp_path: Path, **overrides) -> dt.TrainConfig:
    return _config(tmp_path, mix=MIX, tlog_schedule=str(SCHEDULE_PATH),
                   tlog_schedule_sha256=sha256_file(SCHEDULE_PATH), **overrides)


# --- the config ------------------------------------------------------------------------------


def test_a_run_without_a_mix_has_empty_mix_fields(tmp_path):
    config = _config(tmp_path)
    for name in ("mix", "tlog_schedule", "tlog_schedule_sha256", "frame_manifest",
                 "frame_manifest_sha256"):
        assert getattr(config, name) == ""


def test_the_mix_and_the_hashes_of_what_it_read_are_resume_critical_and_the_paths_are_not():
    for name in ("mix", "tlog_schedule_sha256", "frame_manifest_sha256"):
        assert name in dt.RESUME_CRITICAL_FIELDS
    for name in ("tlog_schedule", "frame_manifest"):
        assert name not in dt.RESUME_CRITICAL_FIELDS


#: What a legacy config reads as for the fields this run's streams set (#94): no exposure use
#: and a 1024 buffer. The mix fields are the ones under test.
LEGACY_STREAM = dict(exposure_use="", shuffle_buffer=1024)


def test_a_checkpoint_from_before_the_mix_resumes_only_as_the_all_tadabur_run_it_was(tmp_path):
    saved = {"preset": "h448"}  # a legacy config: none of the mix fields
    dt.check_resume_compatible(saved, _config(tmp_path, **LEGACY_STREAM))
    with pytest.raises(SystemExit):
        dt.check_resume_compatible(saved, _mixed_config(tmp_path, **LEGACY_STREAM))


def test_a_changed_schedule_or_frame_manifest_is_refused_on_resume(tmp_path):
    base = _mixed_config(tmp_path, frame_manifest_sha256="a" * 64, **LEGACY_STREAM)
    saved = {"mix": MIX, "tlog_schedule_sha256": base.tlog_schedule_sha256,
             "frame_manifest_sha256": "a" * 64}
    dt.check_resume_compatible(saved, base)
    for field in ("tlog_schedule_sha256", "frame_manifest_sha256"):
        with pytest.raises(SystemExit):
            dt.check_resume_compatible({**saved, field: "b" * 64}, base)


def test_the_run_config_records_the_mix_the_hashes_and_the_warm_start(tmp_path):
    config = _mixed_config(tmp_path, frame_manifest_sha256="c" * 64)
    Path(config.out_dir).mkdir()
    dt.write_run_config(Path(config.out_dir), config, Path("/box/issue-178/checkpoint.pt"), True)
    record = json.loads((Path(config.out_dir) / dt.CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert record["mix"] == MIX
    assert record["tlog_schedule_sha256"] == sha256_file(SCHEDULE_PATH)
    assert record["frame_manifest_sha256"] == "c" * 64
    assert record["initialisation"] == {"init_from": "/box/issue-178/checkpoint.pt",
                                        "init_weights": "ema"}


def test_the_mixed_streams_batch_unpacks_to_the_teachers_and_the_students_input():
    teacher, student, sources = torch.zeros(4, 250, 160), torch.ones(4, 250, 160), torch.zeros(4)
    got_teacher, got_student = dt.teacher_student_inputs((teacher, student, sources))
    assert got_teacher is teacher and got_student is student


# --- the loaders -----------------------------------------------------------------------------


def test_without_a_mix_the_train_dataset_is_the_controls_streaming_dataset(tmp_path):
    train_loader, _ = dt.build_dataloaders(_config(tmp_path, shuffle_buffer=512))
    dataset = train_loader.dataset
    assert type(dataset) is ds.StreamingWindowDataset
    assert (dataset.shard_indices, dataset.seed, dataset.shuffle_buffer, dataset.exposure_use) == (
        [21, 22, 23], 2026, 512, "consonant2.training")
    assert train_loader.batch_size == 32 and train_loader.drop_last


def test_with_a_mix_the_train_dataset_wraps_the_controls_tadabur_dataset(tmp_path):
    from training import distill_mixed as dm

    config = _mixed_config(tmp_path, shuffle_buffer=512)
    train_loader, _ = dt.build_dataloaders(config)
    dataset = train_loader.dataset
    assert type(dataset) is dm.MixedWindowDataset
    assert dataset.mix == dm.Mix(tlog=24, tadabur=8)
    assert type(dataset.tadabur) is ds.StreamingWindowDataset
    assert (dataset.tadabur.shard_indices, dataset.tadabur.seed, dataset.tadabur.shuffle_buffer,
            dataset.tadabur.exposure_use) == ([21, 22, 23], 2026, 512, "consonant2.training")
    assert dataset.schedule == read_schedule(SCHEDULE_PATH)
    assert Path(dataset.cache_dir) == Path(config.out_dir) / dm.TLOG_CACHE_DIRNAME
    assert Path(dataset.run_log) == Path(config.out_dir) / dm.TLOG_RUN_LOG_NAME
    assert train_loader.batch_size == 32 and train_loader.drop_last


# --- the command line ------------------------------------------------------------------------


def _manifest(tmp_path: Path, shards=SPEC, seed=2026) -> Path:
    from tadabur.shard_reader import parse_shard_spec
    from training.frame_manifest import frame_manifest, write_frame_manifest

    return write_frame_manifest(tmp_path / "frame", frame_manifest(
        parse_shard_spec(shards), shard_order_seed=seed, exposure_use="consonant2.training"))


def _run_main(tmp_path, monkeypatch, *extra: str) -> dt.TrainConfig:
    """``main`` up to the point it would train: the config it builds."""
    captured: list[dt.TrainConfig] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dt, "preflight", lambda config, device, budget: (1.0, True))
    monkeypatch.setattr(dt, "train", lambda config, **kwargs: captured.append(config))
    audio = tmp_path / "val_clips"
    audio.mkdir(exist_ok=True)
    monkeypatch.setattr(sys, "argv", [
        "distill_train", "--audio-root", str(audio), "--out-dir", str(tmp_path / "run"),
        "--stream-shards", SPEC, "--seed", "2026", "--batch-size", "32", "--num-workers", "1",
        *extra])
    dt.main()
    return captured[0]


def test_the_flags_default_to_the_controls_path():
    args = dt.build_parser().parse_args(["--audio-root", "clips"])
    assert (args.mix, args.tlog_schedule, args.frame_manifest) == ("", "", "")


def test_main_without_a_mix_builds_the_controls_config(tmp_path, monkeypatch):
    config = _run_main(tmp_path, monkeypatch)
    assert (config.mix, config.tlog_schedule_sha256, config.frame_manifest_sha256) == ("", "", "")


def test_main_hashes_the_schedule_and_the_frame_manifest_into_the_config(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    config = _run_main(tmp_path, monkeypatch, "--mix", MIX, "--tlog-schedule", str(SCHEDULE_PATH),
                       "--frame-manifest", str(manifest))
    assert config.mix == MIX
    assert config.tlog_schedule == str(SCHEDULE_PATH)
    assert config.tlog_schedule_sha256 == sha256_file(SCHEDULE_PATH)
    assert config.frame_manifest_sha256 == sha256_file(manifest)


@pytest.mark.parametrize("flags", [
    ("--mix", MIX),                                                     # no schedule, no manifest
    ("--mix", MIX, "--tlog-schedule", str(SCHEDULE_PATH)),              # no manifest
    ("--mix", MIX, "--frame-manifest", "MANIFEST"),                     # no schedule
    ("--mix", "tlog=24,tadabur=4", "--tlog-schedule", str(SCHEDULE_PATH),
     "--frame-manifest", "MANIFEST"),                                   # size is not the batch
])
def test_a_mix_without_what_it_needs_is_refused(tmp_path, monkeypatch, flags):
    manifest = _manifest(tmp_path)
    flags = tuple(str(manifest) if flag == "MANIFEST" else flag for flag in flags)
    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch, *flags)


def test_a_mix_without_a_streamed_frame_is_refused(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", [
        "distill_train", "--audio-root", str(tmp_path), "--mix", MIX,
        "--tlog-schedule", str(SCHEDULE_PATH), "--frame-manifest", str(manifest)])
    with pytest.raises(SystemExit):
        dt.main()


@pytest.mark.parametrize("manifest_kwargs", [{"shards": "21-24"}, {"seed": 178}])
def test_a_frame_manifest_that_is_not_this_runs_frame_is_refused(
        tmp_path, monkeypatch, manifest_kwargs):
    """The mixed stream must replay the control's frame: a manifest for other shards or
    another seed would record a frame the run does not stream."""
    manifest = _manifest(tmp_path, **manifest_kwargs)
    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch, "--mix", MIX, "--tlog-schedule", str(SCHEDULE_PATH),
                  "--frame-manifest", str(manifest))
