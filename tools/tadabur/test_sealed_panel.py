"""Tests for the sealed held-out panel (#89): the seal, the frame, the manifest, and the
committed panel's disjointness from every other use in the exposure registry.

None of these tests scores the panel, and none passes the ship criterion's flag.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tadabur.exposure import (
    H448_TRAINING,
    MINING_POOL,
    SEALED_PANEL,
    TRUTH_SITE_USES,
    Exposure,
    ShardExposure,
    check_disjoint,
    load_registry,
    write_shard_use,
    write_use,
)
from tadabur.mining_pool import PoolSegment, segment_key
from tadabur.sealed_panel import (
    ALLOWED_RECITER_OVERLAP,
    STAGED_PATH,
    PanelClip,
    SealedPanelError,
    build_manifest,
    exposed_reciters,
    frame_record,
    load_manifest,
    load_teacher_decodes,
    open_for_scoring,
    panel_frame,
    reference_capacity,
    unseen_shards,
    write_manifest,
)
from tadabur.staged_audio import REGISTRY_PATH, IndexRow, StagedClip, load_staged_clips

TOOLS_DIR = Path(__file__).resolve().parent.parent
SHA = "d" * 64


# --- the seal --------------------------------------------------------------------------


@pytest.mark.parametrize("authorization", [None, "", "issue-97", "yes, score it"])
def test_scoring_the_panel_without_the_ship_criterions_flag_fails_loudly(tmp_path, authorization):
    with pytest.raises(SealedPanelError, match="scored only by the ship criterion"):
        open_for_scoring(tmp_path, authorization=authorization)


#: Spelled in two halves so this test does not name the flag it pins.
FLAG_NAME = "SHIP_CRITERION_" + "AUTHORIZATION"


def _mentions(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8")


def test_only_the_panels_own_modules_reach_the_panel():
    """The seal is structural: no module may import the panel module, read its directory or
    name its flag except these. #97 adds its scoring module here, and only #97."""
    allowed = {
        "tadabur/sealed_panel.py",       # the panel itself
        "tadabur/test_sealed_panel.py",  # this test
        "tadabur/exposure.py",           # registers the panel's recordings as a use
        "tadabur/test_exposure.py",
        "tadabur/staged_audio.py",       # the staging use the panel's clips carry
    }
    sources = sorted(TOOLS_DIR.rglob("*.py"))
    assert len(sources) > 50  # the scan really walks tools/
    reaching = {
        str(p.relative_to(TOOLS_DIR)) for p in sources
        if _mentions(p, "sealed_panel") or _mentions(p, FLAG_NAME)
    }
    assert reaching <= allowed, sorted(reaching - allowed)
    naming_the_flag = {
        str(p.relative_to(TOOLS_DIR)) for p in sources
        if _mentions(p, FLAG_NAME)
    }
    assert naming_the_flag == {"tadabur/sealed_panel.py"}  # not even this test passes it


# --- the frame -------------------------------------------------------------------------


def _index_row(name, reciter, shard=39, row=0, seconds=8.0, surah_ayah="2:255") -> IndexRow:
    return IndexRow(name, shard, row, reciter, surah_ayah, seconds)


def test_the_unseen_shards_are_the_held_out_block_and_the_strided_reserve():
    shards = unseen_shards()
    assert shards[:21] == list(range(21)) and shards[21:] == list(range(39, 385, 19))


def test_panel_frame_bars_used_reciters_whole_then_applies_the_row_bounds():
    rows = [
        _index_row("a.wav", reciter=1, row=0),
        _index_row("b.wav", reciter=2, row=1),                       # a used reciter
        _index_row("c.wav", reciter=1, row=2, seconds=60.0),          # too long
        _index_row("d.wav", reciter=1, row=3, surah_ayah="106:1"),    # no phonetization
        _index_row("e.wav", reciter=1, shard=21, row=0),              # h448 trained on it
    ]
    panel, excluded = panel_frame(rows, {2: [MINING_POOL]})
    assert [r.audio_filename for r in panel] == ["a.wav"]
    assert {row.audio_filename: reason for row, reason in excluded} == {
        "b.wav": "reciter_exposed", "c.wav": "duration", "d.wav": "phonetizer_unsupported",
    }


def test_only_h448_training_and_the_panel_itself_leave_a_reciter_eligible(tmp_path):
    row = Exposure("a.wav", 39, 0, 1, None, None, SHA)
    write_use(MINING_POOL, [row], tmp_path)
    write_use(SEALED_PANEL, [Exposure("b.wav", 40, 0, 2, None, None, SHA)], tmp_path)
    write_shard_use(H448_TRAINING, ShardExposure((21,), {1: 4, 3: 2}, SHA), tmp_path)
    assert exposed_reciters(load_registry(tmp_path)) == {1: [MINING_POOL]}
    assert ALLOWED_RECITER_OVERLAP == {H448_TRAINING}


def test_frame_record_counts_rows_and_the_training_overlap(tmp_path):
    write_use(MINING_POOL, [Exposure("x.wav", 39, 9, 2, None, None, SHA)], tmp_path)
    write_shard_use(H448_TRAINING, ShardExposure((21,), {1: 4}, SHA), tmp_path)
    registry = load_registry(tmp_path)
    rows = [_index_row("a.wav", 1, row=0), _index_row("b.wav", 2, row=1)]
    panel, excluded = panel_frame(rows, exposed_reciters(registry))
    record = frame_record(panel, excluded, exposed_reciters(registry), registry)
    assert record["per_shard"]["39"] == {"excluded_reciter_exposed": 1, "panel": 1}
    assert record["per_reciter"] == {"1": {"clips": 1, "h448_training_rows": 4}}
    assert record["unseen_reciters_barred_by_use"] == {MINING_POOL: 1}


# --- the manifest ----------------------------------------------------------------------


def _staged(name="a.wav", reciter=1, samples=48000) -> StagedClip:
    return StagedClip(name, 39, 0, reciter, "2:255", samples, SHA, (SEALED_PANEL,))


def _status(name="a.wav", reciter=1) -> dict:
    return {
        "audio_filename": name, "surah_ayah": "2:255", "reciter_id": reciter, "n_words": 3,
        "skip_reason": None, "re_reads": 0, "recited_words": 3, "recitation_start_s": 0.0,
        "recitation_end_s": 3.0, "word_times": [0.0, 1.0, 2.0, 3.0],
    }


def test_build_manifest_converts_spans_and_round_trips(tmp_path):
    seg = {"segment_index": 0, "word_start": 0, "word_end": 3, "start_s": 0.5, "end_s": 2.0,
           "reference": "قَاالَ", "raw_word_offsets": [0], "kept": True}
    staged = {"a.wav": _staged()}
    (clip,) = build_manifest([_status()], [{"audio_filename": "a.wav", "segments": [seg]}], staged)
    assert (clip.segments[0].start_sample, clip.segments[0].end_sample) == (8000, 32000)
    write_manifest([clip], tmp_path / "clips.jsonl")
    assert load_manifest(tmp_path / "clips.jsonl", staged) == [clip]
    with pytest.raises(ValueError, match="different clips"):
        build_manifest([_status()], [], {**staged, "b.wav": _staged("b.wav")})
    with pytest.raises(ValueError, match="different clips"):
        load_manifest(tmp_path / "clips.jsonl", {**staged, "b.wav": _staged("b.wav")})


def test_reference_capacity_reads_only_kept_references():
    def segment(index, reference, kept=True):
        return PoolSegment(index, 0, 1, 0, 100, reference, (0,), kept)

    clip = PanelClip(
        "a.wav", "2:255", 1, 2, None, 0, 2, 0.0, 1.0, (0.0, 1.0),
        (segment(0, "رَببِ"), segment(1, "ذَذَ", kept=False)),
    )
    capacity = reference_capacity([clip])
    assert capacity["haraka_carriers"] == {"damma": 0, "fatha": 1, "kasra": 1}
    assert capacity["reference_geminates"] == 1
    assert capacity["pair_carriers"]["ذ↔ز"] == 0


# --- the committed panel ---------------------------------------------------------------


def test_the_committed_panel_loads_against_its_staging_registry():
    staged = load_staged_clips(STAGED_PATH)
    clips = load_manifest(staged=staged)
    assert len(clips) == len(staged) > 0
    assert all(c.uses == (SEALED_PANEL,) and c.shard in unseen_shards() for c in staged.values())
    fingerprint, decodes = load_teacher_decodes()
    assert fingerprint["model"] == "obadx/muaalem-model-v3_2"  # the base teacher, nothing else
    kept = {segment_key(c.audio_filename, s.segment_index) for c in clips for s in c.segments if s.kept}
    assert set(decodes) == kept


def test_the_panel_is_reciter_disjoint_from_every_other_use():
    registry = load_registry()
    staged = load_staged_clips(STAGED_PATH)
    assert registry.reciters(SEALED_PANEL) == {c.reciter_id for c in staged.values()}
    others = [u for u in registry.uses() if u != SEALED_PANEL and u not in ALLOWED_RECITER_OVERLAP]
    assert set(TRUTH_SITE_USES.values()) & set(others) and MINING_POOL in others
    check_disjoint(SEALED_PANEL, *others, by="reciter", registry=registry)


def test_the_panel_is_source_disjoint_from_every_use_h448_training_included():
    registry = load_registry()
    check_disjoint(SEALED_PANEL, *registry.uses(), by="source", registry=registry)
    shared = registry.reciters(SEALED_PANEL) & registry.reciters(H448_TRAINING)
    assert shared  # the overlap the panel allows is real, and recorded rather than hidden


def test_no_panel_clip_is_in_the_shared_staging_registry():
    assert not set(load_staged_clips(STAGED_PATH)) & set(load_staged_clips(REGISTRY_PATH))


def test_the_frame_and_summary_agree_with_the_manifest():
    panel_dir = STAGED_PATH.parent
    frame = json.loads((panel_dir / "frame.json").read_text(encoding="utf-8"))
    summary = json.loads((panel_dir / "summary.json").read_text(encoding="utf-8"))
    staged = load_staged_clips(STAGED_PATH)
    assert sum(r["clips"] for r in frame["per_reciter"].values()) == len(staged)
    assert {int(r) for r in frame["per_reciter"]} == {c.reciter_id for c in staged.values()}
    assert summary["clips"] == len(staged) and summary["reciters"] == frame["panel_reciters"]
