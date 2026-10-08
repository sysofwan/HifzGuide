"""Tests for the staged-audio registry and the re-staging step (#83).

Shards are synthetic row iterators and audio is a few hundred samples of WAV bytes, so
nothing is downloaded and no model runs.
"""

from __future__ import annotations

import dataclasses
import io
import json

import numpy as np
import pytest
import soundfile as sf

from tadabur.audio import decode_to_mono_16k
from tadabur.staged_audio import (
    MINING_POOL,
    IndexRow,
    StagedClip,
    as_staged,
    fill_staging,
    load_staged_clips,
    read_shard_index,
    stage_clips,
    write_staged_clips,
)
from tadabur.truth_sites import TruthSite, audio_sha256

SHA = "a" * 64


def _clip(name="a.wav", **overrides) -> StagedClip:
    fields = dict(
        audio_filename=name, shard=3, row_index=7, reciter_id=12, surah_ayah="2:255",
        num_samples=16000, audio_sha256=SHA, uses=("waqf_boundary",),
    )
    fields.update(overrides)
    return StagedClip(**fields)


def _site(name="a.wav", **overrides) -> TruthSite:
    fields = dict(
        site_id=f"s:{name}", source="waqf_boundary", assumes_competent_reciter=False,
        audio_filename=name, shard=None, start_sample=0, end_sample=None, audio_sha256=None,
        surah_ayah="2:255", reference="كَتَبَ", reference_index=4, mark="fatha",
        prescribed="fatha", heard="fatha", stratum="x", stratum_population=1,
    )
    fields.update(overrides)
    return TruthSite(**fields)


# --- the registry ----------------------------------------------------------------------


def test_the_registry_round_trips_sorted_by_filename(tmp_path):
    path = tmp_path / "clips.jsonl"
    clips = [_clip("b.wav", row_index=1), _clip("a.wav", uses=("mining_pool", "p35_fixture"))]
    write_staged_clips(clips, path)
    loaded = load_staged_clips(path)
    assert list(loaded) == ["a.wav", "b.wav"]
    assert loaded["a.wav"] == clips[1]
    assert path.read_text(encoding="utf-8").splitlines()[0].startswith('{"audio_filename": "a.wav"')


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"audio_sha256": "A" * 64}, "audio_sha256"),
        ({"shard": 385}, "shard"),
        ({"num_samples": 0}, "num_samples"),
        ({"surah_ayah": "2-255"}, "surah_ayah"),
        ({"uses": ("waqf_boundary", "mining_pool")}, "uses"),
        ({"uses": ("training",)}, "uses"),
        ({"uses": ()}, "uses"),
        ({"audio_filename": "dir/a.wav"}, "bare"),
    ],
)
def test_an_invalid_clip_is_refused_before_anything_is_written(tmp_path, overrides, message):
    path = tmp_path / "clips.jsonl"
    with pytest.raises(ValueError, match=message):
        write_staged_clips([_clip(**overrides)], path)
    assert not path.exists()


def test_a_bool_is_not_an_int(tmp_path):
    path = tmp_path / "clips.jsonl"
    row = {**dataclasses.asdict(_clip()), "uses": ["waqf_boundary"], "shard": True}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="shard must be int"):
        load_staged_clips(path)


def test_two_clips_cannot_claim_one_shard_row(tmp_path):
    with pytest.raises(ValueError, match="same shard row"):
        write_staged_clips([_clip("a.wav"), _clip("b.wav")], tmp_path / "clips.jsonl")


# --- filling truth sites ---------------------------------------------------------------


def test_a_whole_clip_item_takes_the_clips_length_shard_and_checksum():
    (site,), missing = fill_staging([_site()], {"a.wav": _clip()})
    assert (site.shard, site.end_sample, site.audio_sha256) == (3, 16000, SHA)
    assert missing == []


def test_a_span_item_keeps_its_span():
    (site,), _ = fill_staging(
        [_site(start_sample=100, end_sample=900)], {"a.wav": _clip()}
    )
    assert (site.start_sample, site.end_sample, site.shard) == (100, 900, 3)


def test_a_site_on_an_unstaged_clip_stays_null_and_is_listed():
    sites = [_site("b.wav"), _site("a.wav"), _site("b.wav", site_id="s2")]
    filled, missing = fill_staging(sites, {"a.wav": _clip()})
    assert missing == ["b.wav"]
    assert filled[0] == sites[0] and filled[1].audio_sha256 == SHA


def test_a_span_past_the_staged_clip_is_refused():
    with pytest.raises(ValueError, match="past the staged clip"):
        fill_staging([_site(start_sample=0, end_sample=16001)], {"a.wav": _clip()})


def test_a_span_item_without_an_end_is_refused():
    with pytest.raises(ValueError, match="needs its own end_sample"):
        fill_staging([_site(start_sample=5)], {"a.wav": _clip()})


# --- re-staging ------------------------------------------------------------------------


def _wav_bytes(n: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    sf.write(buffer, 0.1 * rng.standard_normal(n).astype(np.float32), 16000, format="WAV",
             subtype="FLOAT")
    return buffer.getvalue()


def _shard(names: list[str], reciter: int = 12) -> list[dict]:
    return [
        {"audio": {"bytes": _wav_bytes(400 + 10 * i, i), "path": name}, "reciter_id": reciter}
        for i, name in enumerate(names)
    ]


def _index(name: str, shard: int, row: int, reciter: int = 12) -> IndexRow:
    return IndexRow(name, shard, row, reciter, "2:255", 1.0)


def test_requested_rows_are_staged_once_per_shard_with_their_provenance(tmp_path):
    shards = {4: _shard(["x.wav", "a.wav", "y.wav"]), 2: _shard(["b.wav"])}
    reads: list[int] = []

    def shard_rows(shard):
        reads.append(shard)
        return iter(shards[shard])

    index = {"a.wav": _index("a.wav", 4, 1), "b.wav": _index("b.wav", 2, 0)}
    requests = {
        "a.wav": frozenset({MINING_POOL}),
        "b.wav": frozenset({"waqf_boundary", "p35_fixture"}),
        "gone.wav": frozenset({"p35_fixture"}),
    }
    staged, unlocatable = stage_clips(requests, index, tmp_path, shard_rows)

    assert reads == [2, 4]
    assert unlocatable == ["gone.wav"]
    a = staged["a.wav"]
    assert (a.shard, a.row_index, a.reciter_id, a.uses) == (4, 1, 12, (MINING_POOL,))
    assert a.num_samples == 410 and sf.info(tmp_path / "a.wav").subtype == "PCM_16"
    assert a.audio_sha256 == audio_sha256(tmp_path / "a.wav")
    assert staged["b.wav"].uses == ("p35_fixture", "waqf_boundary")


def test_as_staged_gives_the_samples_a_staged_file_holds(tmp_path):
    index = {"a.wav": _index("a.wav", 4, 0)}
    stage_clips({"a.wav": frozenset({MINING_POOL})}, index, tmp_path,
                lambda shard: iter(_shard(["a.wav"])))
    in_memory = as_staged(decode_to_mono_16k(_shard(["a.wav"])[0]["audio"]["bytes"]))
    on_disk, _ = sf.read(tmp_path / "a.wav", dtype="float32")
    assert np.array_equal(in_memory, on_disk)


def test_a_row_that_is_not_the_indexed_clip_is_refused(tmp_path):
    index = {"a.wav": _index("a.wav", 4, 0)}
    with pytest.raises(ValueError, match="the index says a.wav"):
        stage_clips({"a.wav": frozenset({MINING_POOL})}, index, tmp_path,
                    lambda shard: iter(_shard(["other.wav"])))


def test_a_reciter_that_disagrees_with_the_index_is_refused(tmp_path):
    index = {"a.wav": _index("a.wav", 4, 0, reciter=99)}
    with pytest.raises(ValueError, match="reciter"):
        stage_clips({"a.wav": frozenset({MINING_POOL})}, index, tmp_path,
                    lambda shard: iter(_shard(["a.wav"])))


def test_a_resumed_run_skips_staged_clips_and_merges_their_uses(tmp_path):
    index = {"a.wav": _index("a.wav", 4, 0)}
    staged, _ = stage_clips({"a.wav": frozenset({MINING_POOL})}, index, tmp_path,
                            lambda shard: iter(_shard(["a.wav"])))

    def no_reads(shard):
        raise AssertionError("a staged clip must not be read again")

    again, _ = stage_clips({"a.wav": frozenset({"p35_fixture"})}, index, tmp_path, no_reads,
                           registry=staged)
    assert again["a.wav"].uses == (MINING_POOL, "p35_fixture")
    assert again["a.wav"].audio_sha256 == staged["a.wav"].audio_sha256


def test_re_staging_must_reproduce_the_recorded_checksum(tmp_path):
    index = {"a.wav": _index("a.wav", 4, 0)}
    recorded = {"a.wav": _clip("a.wav", shard=4, row_index=0)}  # file absent, checksum "aaa…"
    with pytest.raises(ValueError, match="re-staged with sha256"):
        stage_clips({"a.wav": frozenset({MINING_POOL})}, index, tmp_path,
                    lambda shard: iter(_shard(["a.wav"])), registry=recorded)


def test_a_shard_that_ends_before_a_wanted_row_is_refused(tmp_path):
    index = {"a.wav": _index("a.wav", 4, 5)}
    with pytest.raises(ValueError, match="ended before"):
        stage_clips({"a.wav": frozenset({MINING_POOL})}, index, tmp_path,
                    lambda shard: iter(_shard(["a.wav"])))


def test_the_shard_index_canonicalizes_the_ayah(tmp_path):
    path = tmp_path / "index.jsonl"
    row = {"audio_filename": "a.wav", "shard": 4, "row_index": 1, "reciter_id": 12,
           "surah_id": 0, "ayah_id": 7, "ayah_duration_s": 2.5}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert read_shard_index(path)["a.wav"] == IndexRow("a.wav", 4, 1, 12, "1:7", 2.5)


# --- the committed data ----------------------------------------------------------------


@pytest.mark.parametrize("name, use", [("waqf_boundaries", "waqf_boundary"),
                                       ("p35_fixtures", "p35_fixture")])
def test_every_committed_truth_site_is_staged_with_the_registrys_provenance(name, use):
    from tadabur.truth_sites import TRUTH_SITES_DIR, load_truth_sites

    registry = load_staged_clips()
    for site in load_truth_sites(TRUTH_SITES_DIR / f"{name}.jsonl"):
        clip = registry[site.audio_filename]
        assert use in clip.uses
        assert (site.shard, site.audio_sha256) == (clip.shard, clip.audio_sha256)
        assert site.end_sample <= clip.num_samples and site.surah_ayah == clip.surah_ayah
