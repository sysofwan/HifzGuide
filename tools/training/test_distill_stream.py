"""Tests for the streamed training windows' pairing (#94): the teacher reads the clean window,
the student the same window of the augmented clip, and augmentation off is ``h448``'s stream.

Synthetic shards served by a patched hub download; features are the window itself, so a
test compares audio, not a feature extractor.
"""

from __future__ import annotations

import io
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tadabur.shard_reader import iter_shard_rows  # noqa: E402
from training import distill_stream as ds  # noqa: E402

SR = 16_000
#: Clip lengths (s): one short of a window, one window and a padded tail, several windows.
SECONDS = (3.0, 7.3, 12.0, 1.5, 9.9)
RECITERS = (100_001, 100_002, 100_001, 100_003, 100_002)


def _wav(samples: np.ndarray) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, samples, SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture
def shards(tmp_path, monkeypatch):
    """Every shard is one synthetic parquet of five clips of distinct noise."""
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(0)
    names = [f"tadabur_spk{r % 1000:04d}_S1_A{i}_ab_{i:06d}.wav"
             for i, r in enumerate(RECITERS)]
    path = tmp_path / "train-00021.parquet"
    pq.write_table(pa.table({
        "audio": [{"bytes": _wav((0.1 * rng.standard_normal(int(s * SR))).astype(np.float32)),
                   "path": name} for name, s in zip(names, SECONDS)],
        "audio_filename": names,
        "surah_id": [1] * len(names), "ayah_id": list(range(1, len(names) + 1)),
        "reciter_id": list(RECITERS),
    }), path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(path))
    monkeypatch.setattr("tadabur.shard_reader.remove_shard_blob", lambda p: None)
    monkeypatch.setattr(ds.StreamingWindowDataset, "extract",
                        lambda self, window: torch.from_numpy(np.array(window, copy=True)))
    return [21, 22]


def _factor(shard: int, row: int) -> np.float32:
    return np.float32(2.0 + shard / 100 + row / 10_000)


def _keyed_augment(calls: list | None = None):
    """A stand-in augmentation keyed by the clip's row, as the real one is: the clip scaled
    by a factor of ``(shard, row)``, so a window pair shows which clip it was cut from."""

    def augment(clean, shard, row):
        if calls is not None:
            calls.append((shard, row))
        return clean * _factor(shard, row)

    return augment


def _dataset(shard_list, **kwargs) -> ds.StreamingWindowDataset:
    return ds.StreamingWindowDataset(shard_list, delete_after=False, seed=3, **kwargs)


def _h448_stream(dataset: ds.StreamingWindowDataset):
    """The stream as ``h448`` was trained on it, before #94: windows of every row in shard
    order, shuffle-buffered by ``Random(seed + 1)``, one tensor per window."""
    rng = random.Random(dataset.seed + 1)
    buffer: list = []
    for row in iter_shard_rows(dataset._my_shards(), delete_after=False):
        for window in ds.iter_row_windows(ds.decode_row_waveform(row), dataset.hop_seconds):
            buffer.append(window)
            if len(buffer) >= dataset.shuffle_buffer:
                index = rng.randrange(len(buffer))
                buffer[index], buffer[-1] = buffer[-1], buffer[index]
                yield dataset.extract(buffer.pop())
    rng.shuffle(buffer)
    for window in buffer:
        yield dataset.extract(window)


def test_augmentation_off_is_h448s_stream_bit_for_bit(shards):
    dataset = _dataset(shards, shuffle_buffer=4)
    pairs = list(dataset)
    reference = list(_h448_stream(dataset))
    assert len(pairs) == len(reference) > 10
    for (teacher, student), window in zip(pairs, reference):
        assert student is teacher  # one tensor: the student reads exactly the teacher's input
        assert torch.equal(teacher, window)


def test_the_teacher_reads_the_same_stream_with_augmentation_on_or_off(shards):
    """The draws never touch the shuffle: the teacher's windows, in order, are the control's."""
    control = [teacher for teacher, _ in _dataset(shards, shuffle_buffer=4)]
    augmented = [teacher for teacher, _ in
                 _dataset(shards, shuffle_buffer=4, augment=_keyed_augment())]
    assert len(control) == len(augmented)
    assert all(torch.equal(a, b) for a, b in zip(control, augmented))


def test_the_student_reads_the_same_window_of_its_augmented_clip(shards):
    """Clean and augmented windows share their starts: each student window is its teacher
    window under the clip's own augmentation, through the shuffle and the padding."""
    factors = {(s, r): _factor(s, r) for s in shards for r in range(len(SECONDS))}
    pairs = list(_dataset(shards, shuffle_buffer=4, augment=_keyed_augment()))
    for teacher, student in pairs:
        matches = [key for key, f in factors.items() if torch.equal(student, teacher * f)]
        assert len(matches) == 1
    assert all(student is not teacher for teacher, student in pairs)


def test_a_clip_the_augmentation_leaves_clean_is_extracted_once(shards):
    pairs = list(_dataset(shards, augment=lambda clean, shard, row: clean))
    assert pairs and all(student is teacher for teacher, student in pairs)


def test_the_pairing_does_not_depend_on_the_shuffle(shards):
    def pairs(buffer: int) -> list:
        stream = _dataset(shards, shuffle_buffer=buffer, augment=_keyed_augment())
        return sorted((t.numpy().tobytes(), s.numpy().tobytes()) for t, s in stream)

    assert pairs(1) == pairs(3) == pairs(64)


def test_rows_are_excluded_by_reciter_and_recording_and_keyed_before_exclusion(
    shards, monkeypatch, capsys
):
    """Row 1's reciter is excluded, and shard 22's row 3 is tied to an evaluation recording:
    neither is decoded, and the rest keep their parquet row numbers."""
    from tadabur.exposure import AUG_TRAINING, EVALSET_DEV, RowExclusion

    monkeypatch.setattr(RowExclusion, "for_use", classmethod(
        lambda cls, use, registry=None: cls(use, frozenset({RECITERS[1]}),
                                            {(22, 3): (EVALSET_DEV,)})))
    calls: list = []
    list(_dataset(shards, exposure_use=AUG_TRAINING, augment=_keyed_augment(calls)))
    kept = [row for row, reciter in enumerate(RECITERS) if reciter != RECITERS[1]]
    assert sorted(calls) == sorted((shard, row) for shard in shards for row in kept
                                   if (shard, row) != (22, 3))
    log = capsys.readouterr().out
    assert "exposure exclusion through shard" in log  # logged as each shard finishes
    assert "'excluded_by': {'decode_evalset.dev': 1, 'reciter': 2}" in log
