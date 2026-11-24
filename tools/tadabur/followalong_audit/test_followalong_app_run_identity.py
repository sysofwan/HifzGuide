"""A run cache directory belongs to exactly one configuration (#309, S1.2).

The per-clip cache is keyed by the clip's checksum alone, so without a recorded identity the P3
trap is open: ``--latency-ms 130`` added to the 40 ms run's ``--out`` would answer every clip
from the 40 ms results, and the cadence report would read a 0% change from two identical tables.
"""

from __future__ import annotations

import json

import pytest

from tadabur.followalong_audit.app_run import IDENTITY_NAME, run_clips

IDENTITY = {"latency_ms": 40, "release": "v1.0.7", "muraja_commit": "0" * 40}


def clip(index: int) -> dict:
    return {"audio_path": f"{index}_1_{index}.flac", "audio_sha256": f"{index:064x}"}


CLIPS = [clip(index) for index in range(3)]


def ok(clip_row: dict) -> list[dict]:
    return [{"item": clip_row["audio_path"], "word": 1}]


def test_a_rerun_with_the_same_identity_reuses_the_directory(tmp_path):
    first = run_clips(CLIPS, ok, tmp_path, workers=1, identity=IDENTITY)
    again = run_clips(CLIPS, lambda row: pytest.fail("a finished clip ran again"), tmp_path,
                      workers=1, identity=IDENTITY)
    assert again == first
    assert json.loads((tmp_path / IDENTITY_NAME).read_text(encoding="utf-8")) == IDENTITY


def test_another_latency_in_the_same_directory_is_refused_by_name(tmp_path):
    run_clips(CLIPS, ok, tmp_path, workers=1, identity=IDENTITY)
    with pytest.raises(ValueError, match="latency_ms"):
        run_clips(CLIPS, ok, tmp_path, workers=1, identity={**IDENTITY, "latency_ms": 130})


def test_another_harness_or_asset_in_the_same_directory_is_refused(tmp_path):
    run_clips(CLIPS, ok, tmp_path, workers=1, identity=IDENTITY)
    with pytest.raises(ValueError, match="release"):
        run_clips(CLIPS, ok, tmp_path, workers=1, identity={**IDENTITY, "release": "v1.0.6"})


def test_without_an_identity_the_directory_is_not_claimed(tmp_path):
    run_clips(CLIPS, ok, tmp_path, workers=1)
    assert not (tmp_path / IDENTITY_NAME).exists()
