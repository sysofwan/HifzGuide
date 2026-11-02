"""Acceptance tests for ``--tlog-after`` (#250), written before the code.

Locked: later turns may add tests but never weaken or rewrite these.

Contract this file pins
-----------------------
* ``TrainConfig`` gains ``tlog_after`` and ``tlog_after_sha256`` (strings, ``""`` for a run that
  starts the tlog walk at its top). ``tlog_after_sha256`` is resume-critical and a checkpoint that
  predates it reads as ``""``; the path is operational. Both land in ``run_config.json``.
* ``--tlog-after PATH`` (default ``""``) names the consumed-keys file the tlog walk starts after
  (``tadabur/tlog/consumed/consonant2.jsonl.gz`` for round 3). ``main`` records the path and the
  file's SHA-256 (``tadabur.tlog_index.sha256_file``) and refuses (``SystemExit``) the flag
  without ``--mix``: a Tadabur-only run reads no tlog rows, so recording one would certify an
  input it never streamed.
* ``build_dataloaders``: with ``tlog_after`` the mixed dataset's ``after`` is the verified
  ``tadabur.tlog_schedule.ConsumedPrefix`` of that file over the committed schedule and pool;
  otherwise it is ``None``. ``MixedWindowDataset(..., after=None)`` hands ``after`` to
  ``iter_training_rows`` unchanged and exposes it as ``after``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tadabur import tlog_schedule  # noqa: E402
from tadabur.tlog_index import sha256_file  # noqa: E402
from tadabur.tlog_schedule import SCHEDULE_PATH  # noqa: E402
from training import distill_mixed as dm  # noqa: E402
from training import distill_train as dt  # noqa: E402

SPEC = "21-23"
MIX = "tlog=24,tadabur=8"
CONSUMED = Path(tlog_schedule.__file__).parent / "tlog" / "consumed" / "consonant2.jsonl.gz"
#: What a legacy config reads as for the fields the stream sets (#94); not under test here.
LEGACY_STREAM = dict(exposure_use="", shuffle_buffer=1024)


def _config(tmp_path: Path, **overrides) -> dt.TrainConfig:
    audio = tmp_path / "val_clips"
    audio.mkdir(exist_ok=True)
    (audio / "clip_0000.wav").write_bytes(b"RIFF")  # discovered, never decoded here
    values = dict(preset="h448", audio_root=str(audio), out_dir=str(tmp_path / "run"),
                  batch_size=32, stream_shards=SPEC, seed=2026, num_workers=1,
                  exposure_use="consonant2.training", mix=MIX, tlog_schedule=str(SCHEDULE_PATH),
                  tlog_schedule_sha256=sha256_file(SCHEDULE_PATH))
    values.update(overrides)
    return dt.TrainConfig(**values)


def _after_config(tmp_path: Path, **overrides) -> dt.TrainConfig:
    """A config naming an after file; the config only records it, so the digest is a stand-in."""
    return _config(tmp_path, tlog_after="/box/consonant2.jsonl.gz", tlog_after_sha256="a" * 64,
                   **overrides)


def _manifest(tmp_path: Path) -> Path:
    from tadabur.shard_reader import parse_shard_spec
    from training.frame_manifest import frame_manifest, write_frame_manifest

    return write_frame_manifest(tmp_path / "frame", frame_manifest(
        parse_shard_spec(SPEC), shard_order_seed=2026, exposure_use="consonant2.training"))


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


def _mixed_flags(tmp_path: Path) -> tuple[str, ...]:
    return ("--mix", MIX, "--tlog-schedule", str(SCHEDULE_PATH),
            "--frame-manifest", str(_manifest(tmp_path)))


# --- the config ---------------------------------------------------------------------------------


def test_a_run_without_an_after_file_has_empty_after_fields(tmp_path):
    config = _config(tmp_path)
    assert (config.tlog_after, config.tlog_after_sha256) == ("", "")


def test_the_after_hash_is_resume_critical_and_the_path_is_not():
    assert "tlog_after_sha256" in dt.RESUME_CRITICAL_FIELDS
    assert "tlog_after" not in dt.RESUME_CRITICAL_FIELDS
    assert dt.LEGACY_FIELD_VALUES["tlog_after_sha256"] == ""


def test_a_checkpoint_from_before_the_after_file_resumes_only_as_the_run_that_started_at_row_one(
        tmp_path):
    saved = {"preset": "h448", "mix": MIX, "tlog_schedule_sha256": sha256_file(SCHEDULE_PATH)}
    dt.check_resume_compatible(saved, _config(tmp_path, **LEGACY_STREAM))
    with pytest.raises(SystemExit):
        dt.check_resume_compatible(saved, _after_config(tmp_path, **LEGACY_STREAM))


def test_a_changed_after_file_is_refused_on_resume(tmp_path):
    base = _after_config(tmp_path, **LEGACY_STREAM)
    saved = {"mix": MIX, "tlog_schedule_sha256": base.tlog_schedule_sha256,
             "tlog_after_sha256": base.tlog_after_sha256}
    dt.check_resume_compatible(saved, base)
    with pytest.raises(SystemExit):
        dt.check_resume_compatible({**saved, "tlog_after_sha256": "b" * 64}, base)
    with pytest.raises(SystemExit):
        dt.check_resume_compatible({**saved, "tlog_after_sha256": ""}, base)


def test_the_run_config_records_the_after_file_and_its_hash(tmp_path):
    config = _after_config(tmp_path)
    Path(config.out_dir).mkdir()
    dt.write_run_config(Path(config.out_dir), config, None, False)
    record = json.loads((Path(config.out_dir) / dt.CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert record["tlog_after"] == "/box/consonant2.jsonl.gz"
    assert record["tlog_after_sha256"] == "a" * 64


# --- the command line ---------------------------------------------------------------------------


def test_the_flag_defaults_to_the_walk_from_row_one():
    args = dt.build_parser().parse_args(["--audio-root", "clips"])
    assert args.tlog_after == ""


def test_main_without_the_flag_records_no_after_file(tmp_path, monkeypatch):
    config = _run_main(tmp_path, monkeypatch, *_mixed_flags(tmp_path))
    assert hasattr(config, "tlog_after")
    assert (config.tlog_after, config.tlog_after_sha256) == ("", "")


def test_main_records_the_after_file_and_its_hash(tmp_path, monkeypatch):
    config = _run_main(tmp_path, monkeypatch, *_mixed_flags(tmp_path),
                       "--tlog-after", str(CONSUMED))
    assert config.tlog_after == str(CONSUMED)
    assert config.tlog_after_sha256 == sha256_file(CONSUMED)


def test_an_after_file_without_a_mix_is_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="--mix"):
        _run_main(tmp_path, monkeypatch, "--tlog-after", str(CONSUMED))


# --- the dataset --------------------------------------------------------------------------------


def test_without_an_after_file_the_mixed_dataset_starts_at_the_top(tmp_path):
    train_loader, _ = dt.build_dataloaders(_config(tmp_path))
    assert train_loader.dataset.after is None


def test_with_an_after_file_the_mixed_dataset_holds_its_verified_prefix(tmp_path):
    train_loader, _ = dt.build_dataloaders(_after_config(tmp_path))
    after = train_loader.dataset.after
    assert isinstance(after, tlog_schedule.ConsumedPrefix)
    assert after.keys == frozenset(tlog_schedule.read_consumed_keys(CONSUMED))
    assert after.length == len(after.keys) == 73_990


class _Rows:
    """A stand-in Tadabur dataset: no windows, and the attributes the tlog side reads."""

    augment, hop_seconds, shuffle_buffer, seed = None, 2.5, 4, 3
    rows_seen = windows_emitted = 0
    audio_seconds = 0.0

    def __iter__(self):
        yield from ()


def _dataset(tmp_path: Path, **options) -> dm.MixedWindowDataset:
    return dm.MixedWindowDataset(_Rows(), mix=dm.Mix(24, 8), schedule=None,
                                 cache_dir=tmp_path / "cache", run_log=tmp_path / "log.jsonl.gz",
                                 **options)


@pytest.mark.parametrize("passed", [False, True], ids=["omitted", "prefix"])
def test_the_mixed_dataset_hands_after_to_the_reader_unchanged(passed, tmp_path, monkeypatch):
    given = tlog_schedule.ConsumedPrefix(frozenset(), 0, True) if passed else None
    seen = {}

    def reader(*args, **kwargs):
        seen.update(kwargs)
        return iter(())

    monkeypatch.setattr(dm, "iter_training_rows", reader)
    dataset = _dataset(tmp_path, after=given) if passed else _dataset(tmp_path)
    assert dataset.after is given
    assert list(dataset) == []
    assert seen["after"] is given
