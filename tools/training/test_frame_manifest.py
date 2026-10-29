"""The frame manifest a streamed run writes so another run can replay its input (#223).

The round-2 candidate takes its Tadabur substream from the control's frame, so the manifest has
to name the input exactly: the ordered shard list and its digest, the seed that orders the
shards, the derived seed that drives the window shuffle, and the augmentation recipe -- which
keys its draws by the source row rather than by the stream.
"""

from __future__ import annotations

import json

#: A frame small enough to read by eye; the real one is #223's 211 shards.
SHARDS = [21, 22, 23, 40, 41, 90, 383]


def test_the_manifest_names_the_frame_both_seeds_and_the_recipe():
    from training.distill_stream import SHUFFLE_SEED_OFFSET
    from training.frame_manifest import frame_manifest, shards_digest

    manifest = frame_manifest(SHARDS, shard_order_seed=2026,
                              exposure_use="consonant2.control.training")
    assert manifest["shards"] == SHARDS
    # The frame's membership digest is over the sorted set; the order the run streams -- the
    # input a replay has to reproduce -- carries its own digest.
    assert manifest["shards_sha256"] == shards_digest(SHARDS)
    assert manifest["stream_order"] != manifest["shards"]
    assert sorted(manifest["stream_order"]) == SHARDS
    assert manifest["stream_order_sha256"] == shards_digest(manifest["stream_order"])
    assert manifest["shard_order_seed"] == 2026
    # The shuffle is not the shard order's RNG: recording one seed for both would silently
    # replay a different stream.
    assert manifest["window_shuffle_seed"] == 2026 + SHUFFLE_SEED_OFFSET == 2027
    # The recipe record is a shape a replay depends on, so assert its fields rather than
    # echoing the callee's return value.
    recipe = manifest["augmentation"]
    assert recipe["version"] == "aug94-mixture-v2"
    assert set(recipe["mixture"]) == {"clean", "noise", "reverb", "band_limit", "codec"}
    assert recipe["snr_db"] == [3.0, 20.0]
    # Keyed by the source row, so the replay's draws do not depend on arrival order.
    assert "<shard>:<row>" in recipe["draw_seed"]
    assert manifest["exposure_use"] == "consonant2.control.training"


def test_the_digest_is_of_the_order_not_just_the_set():
    from training.frame_manifest import shards_digest

    reordered = [SHARDS[1], SHARDS[0], *SHARDS[2:]]
    assert shards_digest(SHARDS) != shards_digest(reordered)


def test_two_seeds_over_one_frame_have_one_membership_digest_and_two_orders():
    """Candidate and control draw from the same frame: the set digest must match, and the
    stream digests must not."""
    from training.frame_manifest import frame_manifest

    control = frame_manifest(SHARDS, shard_order_seed=2026, exposure_use="u")
    other = frame_manifest(SHARDS, shard_order_seed=178, exposure_use="u")
    assert control["shards_sha256"] == other["shards_sha256"]
    assert control["stream_order_sha256"] != other["stream_order_sha256"]


def test_the_written_manifest_lands_in_the_run_dir_as_one_json_line(tmp_path):
    from training.frame_manifest import FILENAME, frame_manifest, write_frame_manifest

    manifest = frame_manifest(SHARDS, shard_order_seed=2026,
                              exposure_use="consonant2.control.training")
    path = write_frame_manifest(tmp_path / "runs" / "arm", manifest)
    assert path == tmp_path / "runs" / "arm" / FILENAME
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") == 1  # one record, so a re-write is byte-identical
    assert json.loads(text) == manifest
