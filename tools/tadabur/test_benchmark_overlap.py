"""Tests for the benchmark's disjointness plan (#245, step 1); offline, no fingerprints, no audio.

The 600 benchmark clips are fingerprinted against every tlog row (:mod:`tadabur.tlog_overlap`'s
index, line and vote threshold, with ``SETS`` untouched); each hit names the tlog row's role. This
module turns those hits into the plan every model's decode list follows, **before any clip is
decoded**:

* a clip matching a sealed row (``look`` or ``confirm.1``-``confirm.4``) is dropped from every
  model's decode list: decoding it would read a sealed tranche;
* a clip matching a row ``tlog.training`` consumed is decoded but excluded from the verdict and
  reported apart (the candidate may have trained on it);
* every other clip is decoded and counts, matches of unconsumed ``train`` rows included.

The table counts clips per hit category (a clip is counted once per category it hits).
"""

from __future__ import annotations

import pytest

from tadabur.benchmark_overlap import plan_disjointness
from tadabur.tlog_index import CONFIRM_1, CONFIRM_2, CONFIRM_3, CONFIRM_4, LOOK, ROLES, TRAIN

CONSUMED = {"row-2"}
#: c0 no hit; c1 an unconsumed train row; c2 a consumed one; c3 a look row; c4 a confirm row and
#: a consumed train row; c5 a consumed and an unconsumed train row.
CLIPS = ["c0", "c1", "c2", "c3", "c4", "c5"]
HITS = {
    "c0": [],
    "c1": [(TRAIN, "row-1")],
    "c2": [(TRAIN, "row-2")],
    "c3": [(LOOK, "row-3")],
    "c4": [(CONFIRM_2, "row-4"), (TRAIN, "row-2")],
    "c5": [(TRAIN, "row-2"), (TRAIN, "row-1")],
}


def _plan(hits=HITS, clips=CLIPS, consumed=CONSUMED):
    return plan_disjointness(clips, hits, consumed)


def test_sealed_collisions_are_dropped_from_the_decode_list():
    assert _plan().decode == ("c0", "c1", "c2", "c5")


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
    assert plan.verdict == ("c0", "c1")


def test_a_match_of_an_unconsumed_train_row_does_not_exclude_the_clip():
    assert "c1" in _plan().verdict


def test_the_verdict_and_trained_clips_are_exactly_the_decoded_ones():
    plan = _plan()
    assert sorted(plan.verdict + plan.trained) == sorted(plan.decode)


def test_the_table_counts_clips_per_hit_category():
    assert _plan().by_role == {
        "look": 1, "confirm.1": 0, "confirm.2": 1, "confirm.3": 0, "confirm.4": 0,
        "train_consumed": 3, "train_unconsumed": 2, "no_hit": 1}


def test_a_clip_with_several_hits_in_one_category_counts_once():
    plan = _plan({"c": [(TRAIN, "row-1"), (TRAIN, "row-9"), (TRAIN, "row-8")]}, ["c"])
    assert plan.by_role["train_unconsumed"] == 1


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
