"""Tests for the re-read audit's sampling and verdict arithmetic."""

from __future__ import annotations

from tadabur.reread_audit import (
    VERDICT_CLEAN,
    VERDICT_NONHAFS,
    sample_clip_ids,
    summarise_verdicts,
)


def test_the_same_bundle_and_seed_draw_the_same_clips():
    ids = [f"clip{i:03d}" for i in range(200)]

    assert sample_clip_ids(ids, 20, seed=0) == sample_clip_ids(ids, 20, seed=0)
    assert sample_clip_ids(ids, 20, seed=0) != sample_clip_ids(ids, 20, seed=1)


def test_the_draw_does_not_depend_on_bundle_write_order():
    # The bundle is written in clip_id order today; a sample that silently changed if
    # that ever stopped being true would not be reproducible from the seed alone.
    ids = [f"clip{i:03d}" for i in range(50)]

    assert sample_clip_ids(ids, 10, seed=3) == sample_clip_ids(list(reversed(ids)), 10, seed=3)


def test_asking_for_more_clips_than_exist_returns_the_whole_corpus():
    assert sample_clip_ids(["a", "b"], 50, seed=0) == ["a", "b"]


def test_divergence_is_counted_only_where_there_is_a_divergence_to_classify():
    rows = [
        {"verdict": VERDICT_CLEAN, "divergence": ""},
        {"verdict": VERDICT_CLEAN},
        {"verdict": VERDICT_NONHAFS, "divergence": "vowel_only"},
        {"verdict": VERDICT_NONHAFS, "divergence": "consonantal"},
    ]

    report = summarise_verdicts(rows)

    assert report["judged"] == 4
    assert report["nonhafs_rate"] == 0.5
    assert report["divergence_modes"] == {"consonantal": 1, "vowel_only": 1}


def test_an_unclassified_nonhafs_row_is_named_not_assumed_vowel_only():
    # The inertness argument rests on vowel-only dominance. Defaulting a blank mode to
    # vowel_only would make the audit confirm its own hypothesis.
    rows = [{"verdict": VERDICT_NONHAFS, "divergence": ""}, {"verdict": VERDICT_NONHAFS}]

    assert summarise_verdicts(rows)["divergence_modes"] == {"unclassified": 2}


def test_no_verdicts_yet_is_a_zero_rate_not_a_crash():
    assert summarise_verdicts([]) == {
        "judged": 0,
        "verdicts": {},
        "nonhafs_rate": 0.0,
        "divergence_modes": {},
    }
