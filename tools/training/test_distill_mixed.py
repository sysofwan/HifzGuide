"""Acceptance tests for the round-2 mixed window stream (#228), written before the code.

Locked: later turns may add tests but never weaken or rewrite these.

Contract this file pins (``training.distill_mixed``, a new module)
-----------------------------------------------------------------
* ``Mix(tlog, tadabur)``: a frozen dataclass of two positive ints with ``size`` their sum;
  ``parse_mix(spec, batch_size)`` reads ``"tlog=24,tadabur=8"`` (either order) and refuses a
  missing, extra, non-integer or non-positive part and a ``size`` that is not ``batch_size``.
* ``SOURCE_TLOG`` and ``SOURCE_TADABUR``: the two distinct ints that tag an item's source.
* ``MixedWindowDataset(tadabur, *, mix, schedule, cache_dir, run_log, rows=None, stream=None,
  load_audio=None)``: an ``IterableDataset`` over ``(teacher, student, source)`` items.
  ``tadabur`` is the ``StreamingWindowDataset`` the control (#223) runs, consumed as it is:
  its shard order, shuffle buffer, seed, exclusion, augmentation and ``extract`` are the
  Tadabur side's, so the all-Tadabur items are exactly the control's. The tlog side is
  :func:`tadabur.tlog_pool.iter_training_rows` (``rows``, ``stream``, ``load_audio`` and
  ``run_log`` are that reader's) cut into windows like Tadabur's, shuffle-buffered by the same
  ``tadabur.shuffle_buffer``, augmented per clip by ``tadabur.augment`` under the key
  ``(f"{split}:{shard_file}", row)`` and extracted by ``tadabur.extract_pair``. Attributes
  ``tadabur``, ``mix``, ``schedule``, ``cache_dir`` and ``run_log`` hold the constructor's
  arguments. Every 32-batch is exactly ``mix`` by construction and the stream ends on a batch
  boundary: it stops when either side runs out, never emitting a partial or skewed batch.
* ``check_batch_composition(sources, mix)``: raises ``ValueError`` unless ``sources`` holds
  exactly ``mix.tlog`` and ``mix.tadabur`` tags; ``batch_counts(sources)`` is the metrics
  record's ``{"batch_tlog": n, "batch_tadabur": m}``.
* ``per_source_stats(student_logits, teacher_logits, sources, loss_config)``: a flat dict of
  ``kl_<source>``, ``teacher_blank_ratio_<source>``, ``student_blank_ratio_<source>`` and
  ``blank_collapse_margin_<source>`` for ``tlog`` and ``tadabur``.
* ``TLOG_CACHE_DIRNAME`` and ``TLOG_RUN_LOG_NAME``: where the run's tlog shards cache and its
  consumed-keys log go, under the run's out dir.
"""

from __future__ import annotations

import dataclasses
import gzip
import io
import json
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tadabur import exposure, tlog_pool, tlog_schedule  # noqa: E402
from training import distill_mixed as dm  # noqa: E402
from training import distill_stream as ds  # noqa: E402
from training.distill_loss import BLANK_ID, DistillLossConfig, frame_weights, weighted_kl  # noqa: E402

SR = 16_000
#: Clip lengths (s) of the fake Tadabur shards: 11 windows per shard.
SECONDS = (3.0, 7.3, 12.0, 1.5, 9.9)
RECITERS = (100_001, 100_002, 100_001, 100_003, 100_002)
CONTROL_SHARDS_PATH = (Path(__file__).resolve().parents[1] / "tadabur" / "exposure"
                       / "consonant2.control.training.shards.json")
CLEAN_A, CLEAN_B, UNCLEAN_A = "clean-00000-of-00411", "clean-00001-of-00411", "unclean-00000-of-00332"
SHARD_NAMES = {CLEAN_A: "data/clean-00000-of-00411.parquet",
               CLEAN_B: "data/clean-00001-of-00411.parquet",
               UNCLEAN_A: "data/unclean-00000-of-00332.parquet"}


def _wav(samples: np.ndarray) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, samples, SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture
def tadabur_shards(tmp_path, monkeypatch):
    """Every Tadabur shard is one synthetic parquet of five clips; each download is recorded."""
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(0)
    names = [f"tadabur_spk{r % 1000:04d}_S1_A{i}_ab_{i:06d}.wav" for i, r in enumerate(RECITERS)]
    path = tmp_path / "train-shard.parquet"
    pq.write_table(pa.table({
        "audio": [{"bytes": _wav((0.1 * rng.standard_normal(int(s * SR))).astype(np.float32)),
                   "path": name} for name, s in zip(names, SECONDS)],
        "audio_filename": names,
        "surah_id": [1] * len(names), "ayah_id": list(range(1, len(names) + 1)),
        "reciter_id": list(RECITERS),
    }), path)
    downloads: list[int] = []

    def download(dataset_id, filename, **kwargs):
        downloads.append(int(filename.rsplit("-", 1)[1].split(".")[0]))
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr("tadabur.shard_reader.remove_shard_blob", lambda p: None)
    # Features are the window itself, so a test compares audio, not a feature extractor. The
    # tlog side must go through the same extraction (`extract_pair` of the Tadabur dataset).
    monkeypatch.setattr(ds.StreamingWindowDataset, "extract",
                        lambda self, window: torch.from_numpy(np.array(window, copy=True)))
    return downloads


def _tadabur(shards=(21, 22, 23, 24), **kwargs) -> ds.StreamingWindowDataset:
    return ds.StreamingWindowDataset(list(shards), delete_after=False, seed=3, **{
        "shuffle_buffer": 4, **kwargs})


# --- the fake tlog pool --------------------------------------------------------------------------


def _key(split: str, shard: str, row: int) -> str:
    return f"{split}:{shard}:{row}"


def _pool_row(split: str, shard: str, row: int, seconds: float) -> dict:
    payload = _key(split, shard, row).encode()
    import hashlib

    return {"split": split, "shard_file": shard, "row": row, "audio_path": f"{row}.flac",
            "audio_sha256": hashlib.sha256(payload).hexdigest(), "duration_s": seconds,
            "role": "train", "usable": True, "group_key": f"g-{shard}-{row}",
            "eval_overlap": [], "duplicate_of": None}


class FakeTlog:
    """Tlog shards of 12 s clips served by a fake stream; each clip is a ramp on its own
    integer base, so ``floor(first sample)`` names the clip and no two windows are equal."""

    def __init__(self, layout: dict[tuple[str, str], int], *, width: int = 2,
                 order: tuple[tuple[str, str], ...] | None = None, seconds: float = 12.0) -> None:
        self.rows = [_pool_row(split, shard, row, seconds)
                     for (split, shard), count in layout.items() for row in range(count)]
        use = exposure.RowSetExposure("b" * 40, dict(tlog_pool.TRAINING_PREDICATE),
                                      len(self.rows), tlog_pool.rows_digest(self.rows))
        schedule = tlog_schedule.build_schedule(self.rows, use, width=width)
        self.schedule = schedule if order is None else replace(schedule, shards=order)
        self.base = {_key(r["split"], r["shard_file"], r["row"]): index + 1
                     for index, r in enumerate(self.rows)}
        self.seconds = seconds
        self.open_calls: list[str] = []

    def stream(self, shard: str):
        self.open_calls.append(shard)
        rows = sorted((r for r in self.rows if SHARD_NAMES[r["shard_file"]] == shard),
                      key=lambda r: r["row"])
        return iter([{"audio": {"path": f"{r['row']}.flac",
                                "bytes": _key(r["split"], r["shard_file"], r["row"]).encode()}}
                     for r in rows])

    def load_audio(self, data: bytes) -> np.ndarray:
        count = int(self.seconds * SR)
        return (self.base[data.decode()] + 0.5 * np.arange(count) / count).astype(np.float32)

    def reader(self, tmp_path: Path, run_log: Path):
        """The reader the mixed stream wraps, on its own: the unshuffled schedule order."""
        return tlog_pool.iter_training_rows(
            self.rows, cache_dir=tmp_path, schedule=self.schedule, stream=self.stream,
            load_audio=self.load_audio, run_log=run_log)

    def windows(self, tmp_path: Path):
        """``((split, shard, row), window)`` for every window, in schedule order."""
        out = []
        for key, samples in self.reader(tmp_path, tmp_path / "reference.jsonl.gz"):
            out += [(key, ds.cut_window(samples, start))
                    for start in ds.row_window_starts(len(samples))]
        return out


def _clip_of(window, fake: FakeTlog) -> tuple[str, str, int]:
    """The ``(split, shard_file, row)`` of the clip a window was cut from."""
    base = int(np.floor(float(window[0])))
    key = next(key for key, value in fake.base.items() if value == base)
    split, shard, row = key.split(":")
    return split, shard, int(row)


@pytest.fixture
def tlog():
    """Four clean-A, four clean-B and four unclean clips: 12 clips x 4 windows = 48 windows."""
    return FakeTlog({("clean", CLEAN_A): 4, ("clean", CLEAN_B): 4, ("unclean", UNCLEAN_A): 4},
                    order=(("unclean", UNCLEAN_A), ("clean", CLEAN_B), ("clean", CLEAN_A)))


def _big_tlog() -> FakeTlog:
    """Enough tlog windows (160) that the Tadabur side is the one that runs out."""
    return FakeTlog({("clean", CLEAN_A): 14, ("clean", CLEAN_B): 13, ("unclean", UNCLEAN_A): 13},
                    width=3)


def _mixed(tmp_path: Path, fake: FakeTlog, mix: dm.Mix = dm.Mix(24, 8), **tadabur) -> dm.MixedWindowDataset:
    return dm.MixedWindowDataset(
        _tadabur(**tadabur), mix=mix, schedule=fake.schedule, cache_dir=tmp_path / "cache",
        run_log=tmp_path / "consumed.jsonl.gz", rows=fake.rows, stream=fake.stream,
        load_audio=fake.load_audio)


def _items(dataset) -> list[tuple]:
    return list(dataset)


def _factor(shard, row) -> np.float32:
    """A stand-in augmentation factor keyed by the clip, for ints (Tadabur) and strings (tlog)."""
    return np.float32(2.0 + (zlib.crc32(f"{shard}:{row}".encode()) % 1000) / 1000)


def _keyed_augment(calls: list | None = None):
    def augment(clean, shard, row):
        if calls is not None:
            calls.append((shard, row))
        return clean * _factor(shard, row)

    return augment


def _of(items, source):
    return [item for item in items if int(item[2]) == source]


# --- the mix ----------------------------------------------------------------------------------


def test_the_mix_is_two_positive_counts_and_a_size():
    mix = dm.parse_mix("tlog=24,tadabur=8", 32)
    assert mix == dm.Mix(tlog=24, tadabur=8)
    assert mix.size == 32
    assert dm.parse_mix("tadabur=8,tlog=24", 32) == mix
    assert dataclasses.is_dataclass(mix)
    with pytest.raises(dataclasses.FrozenInstanceError):
        mix.tlog = 1
    assert dm.SOURCE_TLOG != dm.SOURCE_TADABUR


@pytest.mark.parametrize("spec", [
    "tlog=24", "tadabur=8", "tlog=24,tadabur=8,other=1", "tlog=24,tlog=8", "tlog=a,tadabur=8",
    "tlog=24.5,tadabur=7.5", "tlog=0,tadabur=32", "tlog=32,tadabur=0", "tlog=-8,tadabur=40", "",
])
def test_a_malformed_mix_is_refused(spec):
    with pytest.raises(ValueError):
        dm.parse_mix(spec, 32)


def test_a_mix_that_is_not_the_batch_size_is_refused():
    with pytest.raises(ValueError):
        dm.parse_mix("tlog=24,tadabur=8", 16)


def test_a_batch_that_is_not_the_mix_is_refused_and_a_batch_is_counted():
    mix = dm.Mix(24, 8)
    good = torch.tensor([dm.SOURCE_TLOG] * 24 + [dm.SOURCE_TADABUR] * 8)
    dm.check_batch_composition(good, mix)
    assert dm.batch_counts(good) == {"batch_tlog": 24, "batch_tadabur": 8}
    skewed = torch.tensor([dm.SOURCE_TLOG] * 23 + [dm.SOURCE_TADABUR] * 9)
    with pytest.raises(ValueError):
        dm.check_batch_composition(skewed, mix)
    assert dm.batch_counts(skewed) == {"batch_tlog": 23, "batch_tadabur": 9}
    with pytest.raises(ValueError):
        dm.check_batch_composition(good[:31], mix)


# --- every batch is exactly the mix ------------------------------------------------------------------


@pytest.mark.parametrize("mix", [dm.Mix(24, 8), dm.Mix(3, 1), dm.Mix(1, 1)])
def test_every_batch_has_exactly_the_mixs_windows_from_each_source(tadabur_shards, tmp_path, mix):
    loader = torch.utils.data.DataLoader(_mixed(tmp_path, _big_tlog(), mix),
                                         batch_size=mix.size, drop_last=True, num_workers=0)
    batches = list(loader)
    assert len(batches) >= 3
    for teacher, student, sources in batches:
        assert teacher.shape[0] == student.shape[0] == mix.size
        dm.check_batch_composition(sources, mix)
        assert int((sources == dm.SOURCE_TLOG).sum()) == mix.tlog
        assert int((sources == dm.SOURCE_TADABUR).sum()) == mix.tadabur


def test_the_stream_ends_on_a_batch_boundary_when_tadabur_runs_out(tadabur_shards, tmp_path):
    """44 Tadabur windows make five batches of eight; the sixth would be 4 + 24 and is never
    emitted, however many tlog windows are left."""
    items = _items(_mixed(tmp_path, _big_tlog()))
    assert len(items) == 5 * 32
    assert len(_of(items, dm.SOURCE_TADABUR)) == 40
    chunks = [items[i:i + 32] for i in range(0, len(items), 32)]
    assert all(sum(int(c[2]) == dm.SOURCE_TLOG for c in chunk) == 24 for chunk in chunks)


def test_the_stream_ends_on_a_batch_boundary_when_tlog_runs_out(tadabur_shards, tmp_path, tlog):
    """48 tlog windows make two batches of 24; the Tadabur side has more than enough."""
    items = _items(_mixed(tmp_path, tlog, shuffle_buffer=4))
    assert len(items) == 2 * 32
    assert len(_of(items, dm.SOURCE_TLOG)) == 48


# --- the Tadabur side is the control's stream -----------------------------------------------------------


def test_the_tadabur_items_are_the_controls_stream_through_the_mix(tadabur_shards, tmp_path):
    control = list(_tadabur(augment=_keyed_augment()))
    items = _items(_mixed(tmp_path, _big_tlog(), augment=_keyed_augment()))
    mine = _of(items, dm.SOURCE_TADABUR)
    assert len(mine) == 40 <= len(control)
    for (teacher, student, _), (control_teacher, control_student) in zip(mine, control):
        assert torch.equal(teacher, control_teacher)
        assert torch.equal(student, control_student)


def test_the_tadabur_substream_is_the_control_frame_and_stream_order(tadabur_shards, tmp_path):
    """The acceptance pin: #223's committed frame (211 shards, seed 2026) hashes to the digests
    its run recorded, and the mixed stream's Tadabur side walks exactly that order."""
    from training.distill_stream import window_shuffle_seed
    from training.frame_manifest import frame_manifest

    shards = json.loads(CONTROL_SHARDS_PATH.read_text(encoding="utf-8"))["shards"]
    assert len(shards) == 211
    manifest = frame_manifest(shards, shard_order_seed=2026, exposure_use="consonant2.training")
    assert manifest["shards_sha256"] == (
        "4bd00c2ba5d0979bcc017546e3beeea70807dad443b125fa293adfe7f5f5d7d8")
    assert manifest["stream_order_sha256"] == (
        "7f0f1533618d0fc4e32e1fa34e20c4d1d22f4995779ad816a58e9d54193674fe")
    assert (manifest["shard_order_seed"], manifest["window_shuffle_seed"]) == (2026, 2027)
    assert window_shuffle_seed(2026) == 2027

    fake = _big_tlog()
    tadabur = ds.StreamingWindowDataset(shards, delete_after=False, seed=2026, shuffle_buffer=4)
    mixed = dm.MixedWindowDataset(
        tadabur, mix=dm.Mix(24, 8), schedule=fake.schedule, cache_dir=tmp_path / "cache",
        run_log=tmp_path / "consumed.jsonl.gz", rows=fake.rows, stream=fake.stream,
        load_audio=fake.load_audio)
    assert mixed.tadabur is tadabur
    assert tadabur._my_shards() == manifest["stream_order"]
    # The shards actually fetched, in order, are the head of that order.
    taken = 0
    for item in mixed:
        taken += int(item[2]) == dm.SOURCE_TADABUR
        if taken >= 40:
            break
    assert tadabur_shards[:4] == manifest["stream_order"][:4]


# --- the tlog side follows the schedule through the buffer --------------------------------------------


def test_the_tlog_windows_are_the_schedules_order_when_the_buffer_holds_one(
        tadabur_shards, tmp_path, tlog):
    items = _of(_items(_mixed(tmp_path, tlog, shuffle_buffer=1)), dm.SOURCE_TLOG)
    reference = tlog.windows(tmp_path)
    assert len(items) == len(reference) == 48
    for (teacher, student, _), (key, window) in zip(items, reference):
        assert np.array_equal(teacher.numpy(), window)
        assert _clip_of(teacher.numpy(), tlog) == key
    # The interleave is the reader's: the schedule opens the unclean shard and then clean B.
    assert reference[0][0][1] == UNCLEAN_A


@pytest.mark.parametrize("buffer", [3, 8])
def test_the_shuffle_buffer_reorders_the_tlog_windows_without_dropping_one(
        tadabur_shards, tmp_path, tlog, buffer):
    items = _of(_items(_mixed(tmp_path, tlog, shuffle_buffer=buffer)), dm.SOURCE_TLOG)
    reference = [window.tobytes() for _, window in tlog.windows(tmp_path)]
    emitted = [teacher.numpy().tobytes() for teacher, _, _ in items]
    assert sorted(emitted) == sorted(reference)  # no row dropped, none seen twice
    assert emitted != reference
    # A buffer of B releases its k-th window no later than after B + k windows were read.
    position = {window: index for index, window in enumerate(reference)}
    assert all(position[window] <= k + buffer - 1 for k, window in enumerate(emitted))


def test_the_consumed_tlog_keys_are_logged_in_the_schedules_order(tadabur_shards, tmp_path, tlog):
    list(_mixed(tmp_path, tlog, shuffle_buffer=4))
    with gzip.open(tmp_path / "consumed.jsonl.gz", "rt", encoding="utf-8") as handle:
        logged = [json.loads(line) for line in handle]  # a log left open would not parse
    expected = [{"split": s, "shard_file": f, "row": r} for (s, f, r), _ in
                tlog.reader(tmp_path, tmp_path / "other.jsonl.gz")]
    assert logged == expected
    assert {line["split"] for line in logged} == {"clean", "unclean"}


def test_a_stream_that_stops_early_still_closes_the_tlog_log(tadabur_shards, tmp_path):
    """Tadabur runs out first, with tlog clips still unread: the reader is closed, so its run
    log is complete and its shards are released."""
    fake = _big_tlog()
    items = _items(_mixed(tmp_path, fake))
    assert len(items) == 5 * 32
    with gzip.open(tmp_path / "consumed.jsonl.gz", "rt", encoding="utf-8") as handle:
        logged = handle.read().splitlines()
    assert 0 < len(logged) < len(fake.rows)


# --- augmentation -------------------------------------------------------------------------------------


def test_tlog_clips_are_augmented_under_their_split_shard_and_row_and_tadabur_under_its_own(
        tadabur_shards, tmp_path, tlog):
    calls: list = []
    list(_mixed(tmp_path, tlog, shuffle_buffer=4, augment=_keyed_augment(calls)))
    tlog_calls = [call for call in calls if isinstance(call[0], str)]
    tadabur_calls = [call for call in calls if isinstance(call[0], int)]
    assert len(tlog_calls) + len(tadabur_calls) == len(calls)
    assert sorted(tlog_calls) == sorted((f"{r['split']}:{r['shard_file']}", r["row"])
                                        for r in tlog.rows)
    # Tadabur keeps the control's `(shard, row)` ints, so shared rows keep the control's draws.
    assert tadabur_calls and {shard for shard, _ in tadabur_calls} <= {21, 22, 23, 24}
    assert all(0 <= row < len(SECONDS) for _, row in tadabur_calls)


def test_each_student_window_is_its_tlog_clips_augmentation_of_the_teachers(
        tadabur_shards, tmp_path, tlog):
    items = _of(_items(_mixed(tmp_path, tlog, shuffle_buffer=4, augment=_keyed_augment())),
                dm.SOURCE_TLOG)
    assert len(items) == 48
    for teacher, student, _ in items:
        split, shard, row = _clip_of(teacher.numpy(), tlog)
        assert torch.equal(student, teacher * _factor(f"{split}:{shard}", row))
    assert all(student is not teacher for teacher, student, _ in items)


def test_the_teacher_reads_the_same_stream_with_augmentation_on_or_off(
        tadabur_shards, tmp_path, tlog):
    off = _items(_mixed(tmp_path, tlog, shuffle_buffer=4))
    on = _items(_mixed(tmp_path / "again", tlog, shuffle_buffer=4, augment=_keyed_augment()))
    assert len(on) == len(off) > 0
    for (clean, _, source), (teacher, _, augmented_source) in zip(off, on):
        assert int(source) == int(augmented_source)
        assert torch.equal(clean, teacher)


def test_a_mixed_stream_without_augmentation_gives_the_student_the_teachers_window(
        tadabur_shards, tmp_path, tlog):
    assert all(student is teacher for teacher, student, _ in
               _items(_mixed(tmp_path, tlog, shuffle_buffer=4)))


# --- per-source logging -------------------------------------------------------------------------------


def _logits(blank: bool, frames: int = 6, rows: int = 2, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, frames, 43, generator=generator)
    logits[..., BLANK_ID] = 20.0 if blank else -20.0
    return logits


def test_the_kl_and_blank_ratios_are_reported_per_source():
    config = DistillLossConfig()
    sources = torch.tensor([dm.SOURCE_TLOG, dm.SOURCE_TADABUR, dm.SOURCE_TLOG, dm.SOURCE_TADABUR])
    teacher = torch.cat([_logits(True, rows=1, seed=1), _logits(False, rows=1, seed=2),
                         _logits(True, rows=1, seed=3), _logits(False, rows=1, seed=4)])
    student = teacher.clone()  # tlog rows: identical to the teacher
    student[1::2] = _logits(True, rows=2, seed=9)  # Tadabur: collapsed to blank
    stats = dm.per_source_stats(student, teacher, sources, config)

    mask = sources == dm.SOURCE_TADABUR
    expected = weighted_kl(student[mask], teacher[mask], frame_weights(
        teacher[mask], nonblank_weight=config.nonblank_weight,
        confirm_timesteps=config.confirm_timesteps, confirm_weight=config.confirm_weight),
        temperature=config.temperature)
    assert stats["kl_tlog"] == pytest.approx(0.0, abs=1e-6)
    assert float(expected) > 0.1
    assert stats["kl_tadabur"] == pytest.approx(float(expected), rel=1e-4)
    assert stats["teacher_blank_ratio_tlog"] == pytest.approx(1.0)
    assert stats["student_blank_ratio_tlog"] == pytest.approx(1.0)
    assert stats["blank_collapse_margin_tlog"] == pytest.approx(0.0)
    assert stats["teacher_blank_ratio_tadabur"] == pytest.approx(0.0)
    assert stats["student_blank_ratio_tadabur"] == pytest.approx(1.0)
    assert stats["blank_collapse_margin_tadabur"] == pytest.approx(1.0)
    assert set(stats) == {f"{name}_{source}" for source in ("tlog", "tadabur")
                          for name in ("kl", "teacher_blank_ratio", "student_blank_ratio",
                                       "blank_collapse_margin")}


def test_a_clean_augmentation_draw_still_gives_the_student_the_teachers_window(
        tadabur_shards, tmp_path, tlog):
    """S1.5: ``Augmenter.__call__`` returns the input array itself for a clean draw (and for
    digital silence); the tlog path must keep that identity so ``extract_pair`` extracts the
    clip once instead of twice."""
    def clean_draw(clean, shard, row):
        return clean

    items = _of(_items(_mixed(tmp_path, tlog, shuffle_buffer=4, augment=clean_draw)),
                dm.SOURCE_TLOG)
    assert items
    assert all(student is teacher for teacher, student, _ in items)


def test_the_tadabur_arm_reports_its_clips_windows_and_seconds(
        tadabur_shards, tmp_path, capsys):
    """S1.3: the tlog reader reports its own per-split rows, clips, windows and seconds; the
    Tadabur arm must report its half of the same line."""
    mixed = _mixed(tmp_path, _big_tlog())
    assert len(_of(_items(mixed), dm.SOURCE_TADABUR)) == 40
    tadabur = mixed.tadabur
    assert tadabur.rows_seen == len(SECONDS) * 4  # 4 shards x 5 clips
    assert 40 <= tadabur.windows_emitted <= 44  # cut, including the buffer's tail
    assert tadabur.audio_seconds == pytest.approx(4 * sum(SECONDS))
    assert f"[data] tadabur: {len(SECONDS) * 4} clips, " in capsys.readouterr().out
