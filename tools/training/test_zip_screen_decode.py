"""Tests for the screen's decode driver (#245): the plan's outcomes and the edit checks.

The driver decodes the benchmark and the edits with our three models, through the shared loop.
Two things it must not get wrong are testable without a GPU or audio: every clip the plan decodes
carries the exposure class the readout partitions on, and no edit audio is decoded unless its
bytes are the manifest's.
"""

from __future__ import annotations

import pytest

from tadabur.synthetic_edits import read_items
from training.zip_screen_decode import BENCHMARK_SET, EDITS_SET, STORES
from training.zip_screen_decode import _clip_outcome, edit_recordings


def test_every_decoded_clip_carries_its_exposure_class():
    plan = {"decode": ["a", "b", "c"], "verdict": ["a"], "trained": ["b"],
            "same_group": ["c"]}
    assert _clip_outcome(plan) == {"a": "verdict", "b": "trained", "c": "same_group"}


def test_a_decoded_clip_the_plan_does_not_classify_is_refused():
    plan = {"decode": ["a", "b"], "verdict": ["a"], "trained": [], "same_group": []}
    with pytest.raises(ValueError, match="no outcome"):
        _clip_outcome(plan)


def test_the_two_decoded_sets_each_name_a_store():
    assert sorted(STORES) == sorted([BENCHMARK_SET, EDITS_SET])
    assert [set(stores) for stores in STORES.values()] == [{"decode"}, {"decode"}]
    assert set(STORES[BENCHMARK_SET]["decode"].protocols) == {"stream_b0", "spans"}


def test_an_edit_booked_at_another_checksum_is_refused_before_it_is_decoded(tmp_path):
    item = read_items()[0]
    (tmp_path / item["output"]["audio_filename"]).write_bytes(b"not the manifest's audio")
    with pytest.raises(ValueError, match="not the manifest's"):
        next(edit_recordings(tmp_path))


def test_a_missing_edit_audio_is_refused(tmp_path):
    with pytest.raises(ValueError, match="is not there"):
        next(edit_recordings(tmp_path))
