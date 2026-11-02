"""Acceptance tests for round 3's Tadabur use, ``consonant3.training`` (#252), written before the
code.

Locked: later turns may add tests but never weaken or rewrite these.

Contract this file pins
-----------------------
* ``exposure.CONSONANT3_TRAINING == "consonant3.training"``: a member of ``USES``, ``PRD_USES``
  and ``TRAINING_AND_TUNING_USES``, like round 2's two uses.
* ``exposure/consonant3.training.shards.json`` is committed: a shard use, membership ``exact``,
  not shared-baseline, over exactly the 211 shards of ``consonant2.training`` minus the first 17
  of their seed-2026 permutation (``training.consumed_prefix.shard_permutation``): round 2 completed
  16 shards and stopped inside the 17th, shard 378, so 194 shards are left. It is computed from the
  same 385-shard index as round 2's use, so its ``index_sha256`` is round 2's and its rows per
  reciter can only be a part of round 2's.
* The use is held to the same row exclusion as round 2's: the same excluded reciters and the same
  barred recordings (``RowExclusion.for_use``). A shard use covers every row of its shards, so
  ``check_disjoint`` cannot certify a training use; the row exclusion is the disjointness
  guarantee, and a use that is not in ``TRAINING_AND_TUNING_USES`` would silently skip it.
* ``exposure/README.md`` has a row for the use.
"""

from __future__ import annotations

from pathlib import Path

from tadabur import exposure
from training.consumed_prefix import monitoring_shard, shard_permutation

EXPOSURE_DIR = Path(__file__).parent / "exposure"
ROUND2_SEED = 2026
ROUND2_COMPLETED = 16
ROUND3_SEED = 2030


def _round2_shards() -> list[int]:
    return list(exposure.load_registry().shard_uses[exposure.CONSONANT2_TRAINING].shards)


def test_the_round_3_use_is_a_training_use_like_round_2s():
    assert exposure.CONSONANT3_TRAINING == "consonant3.training"
    for group in (exposure.USES, exposure.PRD_USES, exposure.TRAINING_AND_TUNING_USES):
        assert exposure.CONSONANT3_TRAINING in group
        assert exposure.CONSONANT2_TRAINING in group


def test_round_2_stopped_inside_shard_378_the_17th_of_its_seed_2026_order():
    order = shard_permutation(_round2_shards(), ROUND2_SEED)
    assert order[ROUND2_COMPLETED] == 378


def test_the_committed_use_is_round_2s_frame_minus_the_17_shards_it_entered():
    registry = exposure.load_registry()
    use = registry.shard_uses[exposure.CONSONANT3_TRAINING]
    entered = shard_permutation(_round2_shards(), ROUND2_SEED)[:ROUND2_COMPLETED + 1]
    assert len(use.shards) == 194
    assert list(use.shards) == sorted(set(_round2_shards()) - set(entered))
    assert 378 not in use.shards
    assert use.membership == "exact"
    assert use.shared_baseline is False


def test_round_2s_monitor_shard_is_training_audio_in_round_3():
    """Shard 72 was round 2's monitor shard, never streamed; round 3 streams it, so round 3's
    validation clips must come from a different shard."""
    shards = exposure.load_registry().shard_uses[exposure.CONSONANT3_TRAINING].shards
    assert 72 in shards
    monitor, position = monitoring_shard(shards, ROUND3_SEED)
    assert monitor != 72
    assert position == len(shards) - 1


def test_the_use_is_computed_from_round_2s_index_and_only_narrows_its_rows():
    registry = exposure.load_registry()
    use = registry.shard_uses[exposure.CONSONANT3_TRAINING]
    round2 = registry.shard_uses[exposure.CONSONANT2_TRAINING]
    assert use.index_sha256 == round2.index_sha256
    assert sum(use.rows_per_reciter.values()) > 0
    assert sum(use.rows_per_reciter.values()) < sum(round2.rows_per_reciter.values())
    for reciter, rows in use.rows_per_reciter.items():
        assert rows <= round2.rows_per_reciter[reciter]


def test_the_use_is_held_to_round_2s_row_exclusion():
    registry = exposure.load_registry()
    round3 = exposure.RowExclusion.for_use(exposure.CONSONANT3_TRAINING, registry)
    round2 = exposure.RowExclusion.for_use(exposure.CONSONANT2_TRAINING, registry)
    assert round3.reciters == round2.reciters
    assert round3.recordings == round2.recordings
    assert round3.recordings  # a training use bars evaluation recordings; an empty set is a bug


def test_no_admitted_reciter_of_the_use_is_an_excluded_reciter():
    registry = exposure.load_registry()
    use = registry.shard_uses[exposure.CONSONANT3_TRAINING]
    barred = exposure.RowExclusion.for_use(exposure.CONSONANT3_TRAINING, registry).reciters
    assert not barred & {int(reciter) for reciter in use.rows_per_reciter}


def test_the_exposure_readme_documents_the_round_3_use():
    readme = (EXPOSURE_DIR / "README.md").read_text(encoding="utf-8")
    assert "| `consonant3.training` |" in readme
