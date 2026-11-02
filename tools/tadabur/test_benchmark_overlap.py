"""Tests for the benchmark's disjointness plan (#245, step 1); offline, no fingerprints, no audio.

The 600 benchmark clips are fingerprinted against every tlog row (:mod:`tadabur.tlog_overlap`'s
index, line and vote threshold, with ``SETS`` untouched); each hit names the tlog row's role. This
module turns those hits into the plan every model's decode list follows, **before any clip is
decoded**:

* a clip matching a sealed row (``look`` or ``confirm.1``-``confirm.4``) is dropped from every
  model's decode list: decoding it would read a sealed tranche;
* a clip matching a row the candidate's run consumed is decoded but excluded from the verdict
  and reported apart (the candidate trained on it);
* a clip matching only pool rows the run never read stays in the verdict, and is named apart
  when its row's shard is one the run partly read: the sensitivity split;
* every other clip is decoded and counts, matches of unread rows outside those shards
  included.

The table counts clips per hit category (a clip is counted once per category it hits).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tadabur import exposure
from tadabur.benchmark_overlap import POOL_UNCONSUMED_SAME_GROUP, plan_disjointness, row_group
from tadabur.tlog_index import CONFIRM_1, CONFIRM_2, CONFIRM_3, CONFIRM_4, LOOK, ROLES, TRAIN
from tadabur.tlog_pool import read_run_log, rows_digest

#: The round-2 candidate's committed consumed-key log: the rows the run read, not the pool.
CONSUMED_LOG = (Path(__file__).parent / "consonant2_readout" / "tlog_consumed.jsonl.gz")
#: ``row-2`` was read, its shard ``shard-a`` is a read shard; ``row-1`` and ``row-9`` were not
#: and are outside it.
CONSUMED = {"shard-a:row-2"}
READ_GROUPS = {"shard-a"}
#: c0 no hit; c1 an unread train row outside any read shard; c2 a consumed one; c3 a look row;
#: c4 a confirm row and a consumed train row; c5 a consumed and an unread train row; c6 an
#: unread train row in a read shard.
CLIPS = ["c0", "c1", "c2", "c3", "c4", "c5", "c6"]
HITS = {
    "c0": [],
    "c1": [(TRAIN, "shard-b:row-1")],
    "c2": [(TRAIN, "shard-a:row-2")],
    "c3": [(LOOK, "shard-a:row-3")],
    "c4": [(CONFIRM_2, "shard-c:row-4"), (TRAIN, "shard-a:row-2")],
    "c5": [(TRAIN, "shard-a:row-2"), (TRAIN, "shard-b:row-1")],
    "c6": [(TRAIN, "shard-a:row-9")],
}


def _plan(hits=HITS, clips=CLIPS, consumed=CONSUMED, read_groups=READ_GROUPS):
    return plan_disjointness(clips, hits, consumed, read_groups)


def test_sealed_collisions_are_dropped_from_the_decode_list():
    assert _plan().decode == ("c0", "c1", "c2", "c5", "c6")


def test_the_dropped_clips_are_named_with_their_sealed_roles():
    assert _plan().sealed == {"c3": (LOOK,), "c4": (CONFIRM_2,)}


def test_a_sealed_clip_that_also_hit_a_consumed_row_is_still_dropped():
    assert "c4" not in _plan().decode


@pytest.mark.parametrize("role", [LOOK, CONFIRM_1, CONFIRM_2, CONFIRM_3, CONFIRM_4])
def test_every_sealed_role_drops_the_clip(role):
    plan = _plan({"c": [(role, "r")]}, ["c"])
    assert plan.decode == () and plan.sealed == {"c": (role,)}


def test_clips_matching_consumed_training_rows_are_decoded_but_not_in_the_verdict():
    plan = _plan()
    assert plan.trained == ("c2", "c5")
    assert plan.verdict == ("c0", "c1", "c6")


def test_a_match_of_an_unconsumed_train_row_does_not_exclude_the_clip():
    assert "c1" in _plan().verdict


def test_an_unread_row_in_a_shard_the_run_partly_read_is_the_sensitivity_split():
    plan = _plan()
    assert plan.same_group == ("c6",)
    assert plan.by_role[POOL_UNCONSUMED_SAME_GROUP] == 1


def test_a_clip_with_a_consumed_hit_is_not_the_sensitivity_split():
    assert "c5" not in _plan().same_group


def test_the_sensitivity_split_is_part_of_the_verdict():
    plan = _plan()
    assert set(plan.same_group) <= set(plan.verdict)


def test_the_verdict_and_trained_clips_are_exactly_the_decoded_ones():
    plan = _plan()
    assert sorted(plan.verdict + plan.trained) == sorted(plan.decode)


def test_the_table_counts_clips_per_hit_category():
    assert _plan().by_role == {
        "look": 1, "confirm.1": 0, "confirm.2": 1, "confirm.3": 0, "confirm.4": 0,
        "train_consumed": 3, "train_unconsumed": 3, "pool_unconsumed_same_group": 1,
        "no_hit": 1}


def test_a_clip_with_several_hits_in_one_category_counts_once():
    hits = [(TRAIN, "shard-b:row-1"), (TRAIN, "shard-b:row-9"), (TRAIN, "shard-b:row-8")]
    plan = _plan({"c": hits}, ["c"])
    assert plan.by_role["train_unconsumed"] == 1


def test_a_row_group_is_the_rows_shard():
    assert row_group("clean:clean-00303-of-00411:26") == "clean:clean-00303-of-00411"


def test_the_decode_list_keeps_the_input_order():
    plan = _plan({c: [] for c in ("z", "b", "m")}, ["z", "b", "m"])
    assert plan.decode == plan.verdict == ("z", "b", "m")


def test_a_hit_with_a_role_the_index_does_not_have_is_refused():
    assert "confirm.5" not in ROLES
    with pytest.raises(ValueError):
        _plan({"c": [("confirm.5", "r")]}, ["c"])


def test_a_clip_with_no_entry_in_the_hits_is_refused_not_assumed_clean():
    with pytest.raises(ValueError):
        _plan({"c0": []}, ["c0", "c1"])


# --- the committed consumed log, against the pool it must not be -----------------------------


def test_the_committed_consumed_log_is_the_rows_the_round_two_run_read():
    rows = read_run_log(CONSUMED_LOG)
    shards = {row_group(f"{row['split']}:{row['shard_file']}:{row['row']}") for row in rows}
    assert (len(rows), len(shards)) == (73_990, 130)
    assert rows_digest(rows) == (
        "3aee9273aaa9f533bc20cc1d604005c4e4c505f1e2741243c85719958f5ec970")


def test_the_consumed_rows_are_not_the_training_pool():
    """D3's regression guard: the "trained on" set is the run's read rows, not the pool."""
    use = exposure.load_registry().row_sets[exposure.TLOG_TRAINING]
    rows = read_run_log(CONSUMED_LOG)
    assert use.count == 426_825
    assert len(rows) < use.count
    assert rows_digest(rows) != use.rows_sha256
