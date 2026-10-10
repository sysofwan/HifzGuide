"""Ship A's mining extension (#126, D9): the frame, the hit rule and the committed record."""

from __future__ import annotations

import json

from tadabur.mining_pool import pool_shards
from tadabur.ship_a_mining import (
    ALREADY_STAGED,
    CLIPS_PATH,
    COPY_IN_H448_TRAINING,
    FILES,
    FRAME_PATH,
    SCAN_PATH,
    PANEL_RECITER,
    SUMMARY_PATH,
    frame_record,
    mining_frame,
    partner_commits,
    shard_order,
)
from tadabur.mining_pool import load_manifest
from tadabur.staged_audio import SHIP_A_MINING, IndexRow, load_staged_clips

FATHA = "َ"


def _row(name: str, shard: int, row: int, reciter: int, seconds: float = 5.0) -> IndexRow:
    return IndexRow(name, shard, row, reciter, "2:2", seconds)


def test_the_frame_leaves_out_staged_clips_panel_reciters_and_training_copies():
    shard = pool_shards()[0]
    rows = [_row("a.wav", shard, 0, 1), _row("b.wav", shard, 1, 2), _row("c.wav", shard, 2, 3),
            _row("d.wav", shard, 3, 4), _row("e.wav", shard, 4, 5, seconds=0.5),
            _row("f.wav", 21, 0, 6)]
    frame, excluded = mining_frame(rows, {(shard, 3)}, ["a.wav"], frozenset({2}), {(shard, 2)},
                                   [shard])
    assert [r.audio_filename for r in frame] == []
    reasons = {row.audio_filename: reason for row, reason in excluded}
    assert reasons == {"a.wav": ALREADY_STAGED, "b.wav": PANEL_RECITER,
                       "c.wav": COPY_IN_H448_TRAINING, "d.wav": "in_decode_evalset",
                       "e.wav": "duration"}
    frame, _ = mining_frame(rows, set(), [], frozenset(), set(), [shard])
    assert {r.audio_filename for r in frame} == {"a.wav", "b.wav", "c.wav", "d.wav"}


def test_a_hit_is_a_partner_letter_of_any_target_pair():
    reference = f"ك{FATHA}س{FATHA}ب{FATHA}ت"
    assert partner_commits(reference, reference) == {}
    assert partner_commits(reference.replace("س", "ش"), reference) == {"س↔ش": 1}
    assert partner_commits(reference.replace("س", "ص"), reference) == {"س↔ص": 1}


def test_the_shard_order_is_a_fixed_permutation():
    order = shard_order(pool_shards())
    assert sorted(order) == sorted(pool_shards()) and order == shard_order(reversed(pool_shards()))


def test_the_frame_record_gives_each_scanned_clip_its_probability():
    shards = shard_order(pool_shards())
    frame = [_row("a.wav", shards[0], 0, 1), _row("b.wav", shards[1], 0, 1)]
    record = frame_record(frame, [], shards, shards[:1])
    assert record["inclusion_probability"] == 1 / len(shards)
    assert record["per_reciter"]["1"] == {"frame": 2, "scanned": 1}


def test_the_committed_extension_agrees_with_its_scan_and_registry():
    frame = json.loads(FRAME_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    with open(SCAN_PATH, encoding="utf-8") as f:
        scan = [json.loads(raw) for raw in f if raw.strip()]
    assert len(scan) == summary["scanned_clips"]
    assert {row["shard"] for row in scan} <= set(frame["scanned_shards"])
    hits = {row["audio_filename"] for row in scan if row["h448"]}
    assert len(hits) == summary["hit_clips"]
    registry = load_staged_clips()
    assert hits == {n for n, c in registry.items() if SHIP_A_MINING in c.uses}
    clips = load_manifest(CLIPS_PATH, registry, FILES.use, FILES.strata)
    assert {c.audio_filename for c in clips} <= hits
    assert all(c.inclusion_probability == frame["inclusion_probability"] for c in clips)


def test_the_mined_clips_share_no_recording_with_h448_training_or_the_panel():
    from tadabur.exposure import H448_TRAINING, SEALED_PANEL, SHIP_A_MINING, check_disjoint

    check_disjoint(SHIP_A_MINING, SEALED_PANEL)
    check_disjoint(SHIP_A_MINING, H448_TRAINING, SEALED_PANEL, by="source")


def test_a_hit_that_repeats_a_staged_recording_is_mined_once():
    from tadabur.ship_a_mining import duplicate_recordings
    from tadabur.staged_audio import StagedClip

    def clip(name, sha, use):
        return StagedClip(name, 9, 0, 1, "2:2", 100, sha * 64, (use,))

    registry = {"a.wav": clip("a.wav", "a", "mining_pool"), "b.wav": clip("b.wav", "a", SHIP_A_MINING),
                "c.wav": clip("c.wav", "c", SHIP_A_MINING), "d.wav": clip("d.wav", "c", SHIP_A_MINING)}
    assert duplicate_recordings(registry) == {"b.wav", "d.wav"}
