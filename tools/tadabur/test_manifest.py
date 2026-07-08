"""Tests for the passing-subset manifest and its resumable, idempotent write sink."""

from __future__ import annotations

import json

from tadabur.manifest import FilterManifest, ManifestRecord


def _record(name: str, ratio: float = 0.9) -> ManifestRecord:
    return ManifestRecord(
        audio_filename=name,
        surah_ayah="3:82",
        match_ratio=ratio,
        ayah_duration_s=10.9,
        reciter_id=88,
    )


def _read_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_commit_writes_records_and_advances_checkpoint(tmp_path):
    manifest_path = tmp_path / "subset.jsonl"
    with FilterManifest.open(manifest_path) as manifest:
        manifest.commit_batch([_record("a.wav"), _record("b.wav")], num_clips=5)
        assert manifest.clips_processed == 5

    rows = _read_lines(manifest_path)
    assert [r["audio_filename"] for r in rows] == ["a.wav", "b.wav"]
    checkpoint = json.loads(
        (tmp_path / "subset.jsonl.progress.json").read_text(encoding="utf-8")
    )
    assert checkpoint["clips_processed"] == 5


def test_reopen_resumes_position_and_dedupes(tmp_path):
    manifest_path = tmp_path / "subset.jsonl"
    with FilterManifest.open(manifest_path) as manifest:
        manifest.commit_batch([_record("a.wav")], num_clips=3)

    # Reopen: resume position is restored and the prior key is remembered.
    with FilterManifest.open(manifest_path) as manifest:
        assert manifest.clips_processed == 3
        assert manifest.passers_written == 1
        manifest.commit_batch([_record("b.wav")], num_clips=2)
        assert manifest.clips_processed == 5

    assert [r["audio_filename"] for r in _read_lines(manifest_path)] == [
        "a.wav",
        "b.wav",
    ]


def test_replaying_a_batch_appends_no_duplicates(tmp_path):
    # Simulates the crash window: a batch's records were written but the
    # checkpoint bump is replayed. The seen-set must keep the manifest duplicate-free.
    manifest_path = tmp_path / "subset.jsonl"
    with FilterManifest.open(manifest_path) as manifest:
        manifest.commit_batch([_record("a.wav"), _record("b.wav")], num_clips=4)

    with FilterManifest.open(manifest_path) as manifest:
        manifest.commit_batch(
            [_record("a.wav"), _record("b.wav"), _record("c.wav")], num_clips=4
        )

    assert [r["audio_filename"] for r in _read_lines(manifest_path)] == [
        "a.wav",
        "b.wav",
        "c.wav",
    ]


def test_empty_batch_still_advances_position(tmp_path):
    # A batch with no passers must still move the checkpoint so those rejected
    # clips are not re-scored on resume.
    manifest_path = tmp_path / "subset.jsonl"
    with FilterManifest.open(manifest_path) as manifest:
        manifest.commit_batch([], num_clips=7)
        assert manifest.clips_processed == 7

    assert manifest_path.read_text(encoding="utf-8") == ""
    with FilterManifest.open(manifest_path) as manifest:
        assert manifest.clips_processed == 7
        assert manifest.passers_written == 0


def test_record_is_json_serializable_with_arabic_surah_ayah(tmp_path):
    manifest_path = tmp_path / "subset.jsonl"
    with FilterManifest.open(manifest_path) as manifest:
        manifest.commit_batch([_record("x.wav", ratio=0.73)], num_clips=1)

    (row,) = _read_lines(manifest_path)
    assert row == {
        "audio_filename": "x.wav",
        "surah_ayah": "3:82",
        "match_ratio": 0.73,
        "ayah_duration_s": 10.9,
        "reciter_id": 88,
        "contrasts": [],
    }


def test_record_carries_contrasts_round_trip(tmp_path):
    from tadabur.manifest import read_records

    manifest_path = tmp_path / "subset.jsonl"
    with FilterManifest.open(manifest_path) as manifest:
        manifest.commit_batch(
            [
                ManifestRecord("a.wav", "3:82", 0.9, 10.0, 88, contrasts=("shadda", "س↔ص")),
                ManifestRecord("b.wav", "3:82", 0.7, 9.0, 88),
            ],
            num_clips=2,
        )

    records = read_records(manifest_path)
    assert [r.audio_filename for r in records] == ["a.wav", "b.wav"]
    assert records[0].contrasts == ("shadda", "س↔ص")
    assert records[1].contrasts == ()


def test_read_records_tolerates_legacy_manifest_without_contrasts(tmp_path):
    from tadabur.manifest import read_records

    manifest_path = tmp_path / "legacy.jsonl"
    manifest_path.write_text(
        '{"audio_filename": "old.wav", "surah_ayah": "1:1", '
        '"match_ratio": 0.8, "ayah_duration_s": 5.0, "reciter_id": 3}\n',
        encoding="utf-8",
    )
    (record,) = read_records(manifest_path)
    assert record.audio_filename == "old.wav"
    assert record.contrasts == ()
