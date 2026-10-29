"""Acceptance tests for round 2's exposure uses and the tlog disjointness check (#228), written
before the code.

Locked: later turns may add tests but never weaken or rewrite these.

Contract this file pins
-----------------------
* ``exposure.CONSONANT2_TRAINING == "consonant2.training"``: the candidate's Tadabur use, a
  member of ``USES``, ``PRD_USES`` and ``TRAINING_AND_TUNING_USES``, like the control's.
* ``exposure/consonant2.training.shards.json`` is committed: a shard use, membership
  ``exact``, not shared-baseline, over exactly the control's 211 shards (the candidate replays
  the control's frame); ``exposure/README.md`` has a row for it.
* ``python -m tadabur.exposure tlog-training-disjoint``: runs
  :func:`tadabur.exposure.check_row_set_disjoint` over :func:`tadabur.tlog_pool.load_training_rows`
  and never :func:`tadabur.exposure.check_disjoint`, which refuses tlog by design. A collision
  exits non-zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tadabur import exposure, tlog_pool

EXPOSURE_DIR = Path(__file__).parent / "exposure"
CONTROL_USE_PATH = EXPOSURE_DIR / "consonant2.control.training.shards.json"


def test_the_candidates_use_is_a_training_use_like_the_controls():
    assert exposure.CONSONANT2_TRAINING == "consonant2.training"
    for group in (exposure.USES, exposure.PRD_USES, exposure.TRAINING_AND_TUNING_USES):
        assert exposure.CONSONANT2_TRAINING in group
        assert exposure.CONSONANT2_CONTROL in group


def test_the_committed_use_is_the_controls_frame_as_an_exact_shard_use():
    registry = exposure.load_registry()
    use = registry.shard_uses[exposure.CONSONANT2_TRAINING]
    control = registry.shard_uses[exposure.CONSONANT2_CONTROL]
    assert len(use.shards) == 211
    assert use.shards == control.shards
    assert use.membership == "exact"
    assert use.shared_baseline is False
    assert sum(use.rows_per_reciter.values()) > 0
    assert json.loads((EXPOSURE_DIR / "consonant2.training.shards.json").read_text(
        encoding="utf-8"))["shards"] == json.loads(CONTROL_USE_PATH.read_text(
            encoding="utf-8"))["shards"]


def test_the_exposure_readme_documents_the_candidates_use():
    readme = (EXPOSURE_DIR / "README.md").read_text(encoding="utf-8")
    assert "| `consonant2.training` |" in readme


# --- the tlog disjointness command ---------------------------------------------------------


@pytest.fixture
def calls(monkeypatch):
    """The two disjointness functions and the loader, recording their use."""
    record: dict[str, list] = {"row_set": [], "disjoint": []}
    rows = [{"split": "clean", "shard_file": "clean-00000-of-00411", "row": 0}]
    monkeypatch.setattr(tlog_pool, "load_training_rows", lambda *a, **k: rows)

    def row_set(passed, **kwargs):
        record["row_set"].append(list(passed))

    def disjoint(*args, **kwargs):
        record["disjoint"].append(args)
        raise AssertionError("check_disjoint refuses tlog by design and must not be called")

    monkeypatch.setattr(exposure, "check_row_set_disjoint", row_set)
    monkeypatch.setattr(exposure, "check_disjoint", disjoint)
    record["rows"] = rows
    return record


def _main(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["tadabur.exposure", "tlog-training-disjoint"])
    exposure.main()


def test_the_command_checks_the_training_rows_with_the_row_set_function(calls, monkeypatch):
    _main(monkeypatch)
    assert calls["row_set"] == [calls["rows"]]
    assert calls["disjoint"] == []


def test_a_collision_stops_the_command(calls, monkeypatch):
    def collide(passed, **kwargs):
        raise exposure.ExposureOverlap("audio_sha256: a training row is an evaluation clip")

    monkeypatch.setattr(exposure, "check_row_set_disjoint", collide)
    with pytest.raises((SystemExit, exposure.ExposureOverlap)) as stopped:
        _main(monkeypatch)
    assert stopped.value.code != 0 if isinstance(stopped.value, SystemExit) else True
    assert calls["disjoint"] == []
