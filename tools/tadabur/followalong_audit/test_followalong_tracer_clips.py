"""The seeded draw of tracer clips (#307): reproducible, provenance-complete, extendable."""

from __future__ import annotations

import random

import pytest

from tadabur.followalong_audit.clips import PROVENANCE_FIELDS, draw_clips

GATING_ROWS = [
    {"split": "clean", "shard_file": f"clean-{shard:05d}-of-00411", "row": row,
     "audio_path": f"{shard}_{row}_{shard * 1000 + row}.flac",
     "audio_sha256": f"{shard * 1000 + row:064x}", "filename_label": f"{1 + shard}:{1 + row % 7}",
     "stratum": "gating", "group_key": f"g{shard}-{row}"}
    for shard in range(4) for row in range(40)
]


def test_the_provenance_names_the_audio_and_where_it_lives():
    assert {"audio_path", "split", "shard_file", "row", "audio_sha256"} <= set(PROVENANCE_FIELDS)


def test_a_draw_is_the_requested_size_of_distinct_source_rows():
    drawn = draw_clips(GATING_ROWS, 20, seed=7)
    assert len(drawn) == 20
    assert len({(c["split"], c["shard_file"], c["row"]) for c in drawn}) == 20


def test_each_drawn_clip_carries_its_source_rows_provenance_unchanged():
    by_key = {(r["split"], r["shard_file"], r["row"]): r for r in GATING_ROWS}
    for clip in draw_clips(GATING_ROWS, 20, seed=7):
        source = by_key[(clip["split"], clip["shard_file"], clip["row"])]
        assert {name: clip[name] for name in PROVENANCE_FIELDS} == \
            {name: source[name] for name in PROVENANCE_FIELDS}


def test_the_same_seed_draws_the_same_clips_in_the_same_order_whatever_the_input_order():
    shuffled = list(GATING_ROWS)
    random.Random(1).shuffle(shuffled)
    assert draw_clips(shuffled, 20, seed=7) == draw_clips(GATING_ROWS, 20, seed=7)


def test_another_seed_draws_other_clips():
    assert draw_clips(GATING_ROWS, 20, seed=7) != draw_clips(GATING_ROWS, 20, seed=8)


def test_extending_to_a_hundred_clips_keeps_the_first_twenty():
    assert draw_clips(GATING_ROWS, 100, seed=7)[:20] == draw_clips(GATING_ROWS, 20, seed=7)


def test_drawing_more_clips_than_there_are_rows_is_refused():
    with pytest.raises(ValueError, match="160"):
        draw_clips(GATING_ROWS, 161, seed=7)
