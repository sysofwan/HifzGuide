"""Ship A's mining extension (#126, D9): the frame, the hit rule and the committed record."""

from __future__ import annotations

import json

import pytest

from tadabur.mining_pool import pool_shards
from tadabur.ship_a_mining import (
    ALREADY_STAGED,
    CLIPS_PATH,
    LINKED_RECORDING,
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
    frame, excluded = mining_frame(rows, {(shard, 3)}, ["a.wav"], frozenset({2}),
                                   frozenset({(shard, 2)}), [shard])
    assert [r.audio_filename for r in frame] == []
    reasons = {row.audio_filename: reason for row, reason in excluded}
    assert reasons == {"a.wav": ALREADY_STAGED, "b.wav": PANEL_RECITER,
                       "c.wav": LINKED_RECORDING, "d.wav": "in_decode_evalset",
                       "e.wav": "duration"}
    frame, _ = mining_frame(rows, set(), [], frozenset(), frozenset(), [shard])
    assert {r.audio_filename for r in frame} == {"a.wav", "b.wav", "c.wav", "d.wav"}


def test_a_hit_is_a_partner_letter_of_any_target_pair():
    reference = f"ك{FATHA}س{FATHA}ب{FATHA}ت"
    assert partner_commits(reference, reference) == {}
    assert partner_commits(reference.replace("س", "ش"), reference) == {"س↔ش": 1}
    assert partner_commits(reference.replace("س", "ص"), reference) == {"س↔ص": 1}


def test_the_shard_order_is_a_fixed_permutation():
    order = shard_order(pool_shards())
    assert sorted(order) == sorted(pool_shards()) and order == shard_order(reversed(pool_shards()))


def test_the_frame_record_counts_frame_and_scanned_clips_per_reciter():
    shards = shard_order(pool_shards())
    frame = [_row("a.wav", shards[0], 0, 1), _row("b.wav", shards[1], 0, 1)]
    record = frame_record(frame, [], shards, shards[:1])
    assert "inclusion_probability" not in record  # a time-stopped scan is no random sample
    assert record["per_reciter"]["1"] == {"frame": 2, "scanned": 1}


def test_registering_twice_changes_nothing_and_a_changed_row_is_refused():
    from tadabur.ship_a_mining import merge_hits
    from tadabur.staged_audio import StagedClip

    hit = StagedClip("h.wav", 9, 0, 1, "2:2", 100, "a" * 64, (SHIP_A_MINING,))
    other = StagedClip("p.wav", 9, 1, 1, "2:2", 100, "b" * 64, ("mining_pool",))
    once = merge_hits({"p.wav": other}, {"h.wav": hit})
    assert merge_hits(once, {"h.wav": hit}) == once == {"p.wav": other, "h.wav": hit}
    with pytest.raises(SystemExit):
        merge_hits(once, {"h.wav": StagedClip("h.wav", 9, 0, 1, "2:2", 100, "c" * 64,
                                              (SHIP_A_MINING,))})


def test_every_scanned_recording_is_an_exposure_with_its_checksum_where_staged():
    from tadabur.ship_a_mining import scanned_exposures
    from tadabur.staged_audio import StagedClip

    scan = [{"audio_filename": "h.wav", "shard": 9, "row_index": 0, "reciter_id": 1},
            {"audio_filename": "n.wav", "shard": 9, "row_index": 1, "reciter_id": 2}]
    staged = {"h.wav": StagedClip("h.wav", 9, 0, 1, "2:2", 100, "a" * 64, (SHIP_A_MINING,))}
    rows = scanned_exposures(scan, staged)
    assert [(r.audio_filename, r.audio_sha256) for r in rows] == [("h.wav", "a" * 64),
                                                                  ("n.wav", None)]


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
    assert all(c.inclusion_probability == 1.0 for c in clips)


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
