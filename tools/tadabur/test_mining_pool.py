"""Tests for the h448-unseen mining pool: its draw, its manifest and its capacity count (#83)."""

from __future__ import annotations

import json
import random

import pytest

from tadabur.mining_pool import (
    CONSONANT_PAIR,
    GEMINATE,
    UNIFORM,
    PoolClip,
    PoolSegment,
    Selected,
    build_manifest,
    capacity,
    census_frame,
    eligible,
    event_strata,
    frame_record,
    inclusion_probabilities,
    load_manifest,
    pool_shards,
    read_evalset_rows,
    read_selection,
    scan_events,
    scan_identity,
    segment_key,
    select_pool,
    stratify,
    write_manifest,
)
from tadabur.staged_audio import IndexRow, StagedClip

SHA = "c" * 64


def _row(name: str, reciter: int, shard: int = 39, row: int = 0, seconds: float = 8.0,
         ayah: str = "2:2") -> IndexRow:
    return IndexRow(name, shard, row, reciter, ayah, seconds)


# --- the draw --------------------------------------------------------------------------


def test_the_frame_is_the_strided_reserve_past_the_held_out_block():
    shards = pool_shards()
    assert shards[0] == 39 and shards[-1] == 381 and len(shards) == 19
    assert all(b - a == 19 for a, b in zip(shards, shards[1:]))


def test_ineligible_rows_are_counted_by_reason():
    rows = [
        _row("ok.wav", 1),
        _row("held_out_block.wav", 1, shard=20),
        _row("training_shard.wav", 1, shard=40),
        _row("evalset.wav", 1, row=3),
        _row("long.wav", 1, seconds=51.0),
        _row("short.wav", 1, seconds=1.0),
        _row("unsupported.wav", 1, ayah="106:1"),
    ]
    frame, excluded = eligible(rows, evalset_rows={(39, 3)})
    assert [r.audio_filename for r in frame] == ["ok.wav"]
    assert [(row.audio_filename, reason) for row, reason in excluded] == [
        ("evalset.wav", "in_decode_evalset"), ("long.wav", "duration"),
        ("short.wav", "duration"), ("unsupported.wav", "phonetizer_unsupported")]


def _frame() -> list[IndexRow]:
    # 30 reciters with 1..12 clips each.
    return [
        _row(f"r{reciter}_c{clip}.wav", reciter, row=reciter * 100 + clip)
        for reciter in range(30)
        for clip in range(1 + reciter % 12)
    ]


def test_the_draw_caps_clips_per_reciter_and_stops_at_the_size():
    pool = select_pool(_frame(), size=50, cap=4)
    assert len(pool) == 50
    per_reciter: dict[int, int] = {}
    for row in pool:
        per_reciter[row.reciter_id] = per_reciter.get(row.reciter_id, 0) + 1
    assert max(per_reciter.values()) <= 4
    # Reciters are taken whole, in order: only the last one drawn may be short of its cap.
    short = [r for r, n in per_reciter.items() if n < min(4, 1 + r % 12)]
    assert len(short) <= 1


def test_the_draw_depends_on_content_not_order():
    rows = _frame()
    shuffled = rows[:]
    random.Random(7).shuffle(shuffled)
    assert select_pool(rows, size=40, cap=3) == select_pool(shuffled, size=40, cap=3)


def test_a_frame_smaller_than_the_size_yields_every_capped_clip():
    pool = select_pool(_frame(), size=10_000, cap=2)
    assert len(pool) == sum(min(2, 1 + r % 12) for r in range(30))


def test_evalset_rows_are_read_from_its_filenames(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"clips": [
        {"filename": "tadabur_sh039_i00012_S1_A2_.wav", "shard": 39, "split": "dev"},
        {"filename": "tadabur_sh381_i00999_S0_A1_.wav", "shard": 381, "split": "test"},
    ]}), encoding="utf-8")
    assert read_evalset_rows(path) == {(39, 12), (381, 999)}


def test_an_evalset_filename_that_disagrees_with_its_shard_is_refused(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"clips": [
        {"filename": "tadabur_sh039_i00012_S1_A2_.wav", "shard": 58, "split": "dev"}]}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read"):
        read_evalset_rows(path)


# --- the manifest ----------------------------------------------------------------------

CLIP = "tadabur_spk0003_S1_A2_y_000002.wav"
FATHA, KASRA = "َ", "ِ"
RAA, BAA, LAM, DHAL, ZAI = "ر", "ب", "ل", "ذ", "ز"
REFERENCE = RAA + FATHA + BAA + BAA + KASRA + LAM + DHAL + FATHA + LAM
DECODE = RAA + FATHA + BAA + KASRA + LAM + ZAI + LAM


def _staged(uses=("mining_pool",)) -> dict[str, StagedClip]:
    return {CLIP: StagedClip(CLIP, 39, 4, 3, "2:2", 32000, SHA, uses)}


def _status() -> dict:
    return {"audio_filename": CLIP, "surah_ayah": "2:2", "reciter_id": 3, "n_words": 2,
            "duration_s": 2.0, "skip_reason": None, "re_reads": 0, "recited_words": 2,
            "recitation_start_s": 0.1, "recitation_end_s": 1.9, "word_times": [0.1, 1.0, 1.9]}


def _segmentation() -> dict:
    return {"audio_filename": CLIP, "drops": {}, "segments": [
        {"segment_index": 0, "word_start": 0, "word_end": 2, "start_s": 0.1, "end_s": 2.5,
         "reference": REFERENCE, "raw_word_offsets": [0, 5, 9], "kept": True}]}


def test_the_manifest_carries_sample_spans_clamped_to_the_staged_clip(tmp_path):
    (clip,) = build_manifest([_status()], [_segmentation()], {CLIP: Selected((UNIFORM,), 0.5)}, _staged())
    (seg,) = clip.segments
    assert (seg.start_sample, seg.end_sample) == (1600, 32000)
    assert clip.word_times == (0.1, 1.0, 1.9) and seg.raw_word_offsets == (0, 5, 9)

    path = tmp_path / "clips.jsonl"
    write_manifest([clip], path)
    assert load_manifest(path, registry=_staged()) == [clip]


def test_a_clip_not_staged_for_the_pool_is_refused():
    with pytest.raises(ValueError, match="not staged for the mining pool"):
        build_manifest([_status()], [_segmentation()], {CLIP: Selected((UNIFORM,), 0.5)},
                       _staged(uses=("p35_fixture",)))


def test_loading_refuses_a_clip_whose_reciter_disagrees_with_the_registry(tmp_path):
    (clip,) = build_manifest([_status()], [_segmentation()], {CLIP: Selected((UNIFORM,), 0.5)}, _staged())
    path = tmp_path / "clips.jsonl"
    write_manifest([PoolClip(**{**clip.__dict__, "reciter_id": 4})], path)
    with pytest.raises(ValueError, match="reciter or ayah"):
        load_manifest(path, registry=_staged())


def test_capacity_counts_each_stratum_from_the_base_decode():
    segment = PoolSegment(0, 0, 2, 0, 100, REFERENCE, (0, 5, 9), True)
    dropped = PoolSegment(1, 2, 3, 100, 200, REFERENCE, (0, 9), False)
    clip = PoolClip(CLIP, (UNIFORM,), 0.5, "2:2", 3, 2, None, 0, 2, 0.0, 1.0, (0.0, 1.0), (segment, dropped))
    counts = capacity([clip], {segment_key(CLIP, 0): DECODE})

    # The dropped segment has no decode and is not counted.
    assert counts["haraka"] == {"damma": {}, "fatha": {"matched": 1, "omitted": 1},
                                "kasra": {"matched": 1}}
    assert counts["shaddah"]["reference_geminates"] == 1
    assert counts["shaddah"]["base_single_at_geminate"] == 1
    assert counts["pairs"]["ذ↔ز"] == {"base_other_letter": 1, "reference_carriers": 1}
    assert counts["pairs"]["ذ↔ظ"] == {"base_other_letter": 0, "reference_carriers": 1}
    assert counts["sukun_mid_word_carriers"] == 1  # the ل before ذ; not the geminate ب


def test_a_selection_without_segmentation_is_refused():
    with pytest.raises(ValueError, match="different clips"):
        build_manifest([_status()], [_segmentation()], {"other.wav": Selected((UNIFORM,), 0.5)}, _staged())


def test_loading_refuses_an_unknown_stratum(tmp_path):
    (clip,) = build_manifest([_status()], [_segmentation()], {CLIP: Selected((UNIFORM,), 0.5)}, _staged())
    path = tmp_path / "clips.jsonl"
    write_manifest([PoolClip(**{**clip.__dict__, "strata": ("enriched",)})], path)
    with pytest.raises(ValueError, match="strata"):
        load_manifest(path, registry=_staged())


# --- the census strata -----------------------------------------------------------------


def test_scan_events_count_pair_substitutions_and_gemination_mismatches():
    events = scan_events(DECODE, REFERENCE)
    assert events == {"pairs": {"ذ↔ز": 1}, "shaddah": {"dropped": 1}}
    assert event_strata(events) == (CONSONANT_PAIR, GEMINATE)
    assert event_strata(scan_events(REFERENCE, REFERENCE)) == ()


def test_the_census_frame_is_every_frame_clip_of_the_drawn_reciters():
    frame = [_row("a.wav", 1), _row("b.wav", 2, row=1), _row("c.wav", 1, row=2)]
    assert [r.audio_filename for r in census_frame(frame, [frame[0]])] == ["a.wav", "c.wav"]


def test_census_clips_with_an_event_join_the_uniform_draw():
    uniform = [_row("a.wav", 1), _row("b.wav", 1, row=1)]
    census = uniform + [_row("c.wav", 1, row=2), _row("d.wav", 1, row=3)]
    none = {"pairs": {}, "shaddah": {}}
    scanned = {
        "a.wav": {"pairs": {"ذ↔ز": 1}, "shaddah": {}},
        "b.wav": none,
        "c.wav": {"pairs": {}, "shaddah": {"added": 2}},
        "d.wav": none,
    }
    assert stratify(uniform, census, scanned) == {
        "a.wav": (UNIFORM, CONSONANT_PAIR),
        "b.wav": (UNIFORM,),
        "c.wav": (GEMINATE,),
    }


def test_a_scan_that_misses_part_of_the_census_frame_is_refused():
    uniform = [_row("a.wav", 1)]
    census = uniform + [_row("c.wav", 1, row=2)]
    with pytest.raises(ValueError, match="unscanned"):
        stratify(uniform, census, {"a.wav": {"pairs": {}, "shaddah": {}}})


def test_a_selection_reads_back_as_index_rows(tmp_path):
    path = tmp_path / "selection.jsonl"
    row = _row("a.wav", 4, shard=58, row=9)
    path.write_text(json.dumps({**row.__dict__, "strata": ["uniform"]}) + "\n", encoding="utf-8")
    assert read_selection(path) == [row]


# --- the committed pool ----------------------------------------------------------------


def test_the_committed_pool_loads_and_matches_its_summary_and_decodes():
    from tadabur.mining_pool import SUMMARY_PATH, load_base_decodes

    clips = load_manifest()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    _, decodes = load_base_decodes()
    assert len(clips) == summary["clips"]
    assert {s: sum(s in c.strata for c in clips) for s in (UNIFORM, CONSONANT_PAIR, GEMINATE)} \
        == summary["clips_per_stratum"]
    kept = {segment_key(c.audio_filename, s.segment_index) for c in clips for s in c.segments
            if s.kept}
    assert kept == set(decodes)


# --- inclusion probabilities and the frame ---------------------------------------------


def test_inclusion_is_the_within_reciter_share_or_one_in_a_census():
    frame = [_row(f"r1_{i}.wav", 1, row=i) for i in range(4)] + [_row("r2.wav", 2, row=9)]
    uniform = [frame[0], frame[1], frame[4]]
    strata = {"r1_0.wav": (UNIFORM,), "r1_1.wav": (UNIFORM, GEMINATE),
              "r1_3.wav": (CONSONANT_PAIR,), "r2.wav": (UNIFORM,)}
    assert inclusion_probabilities(frame, uniform, strata) == {
        "r1_0.wav": 0.5, "r1_1.wav": 1.0, "r1_3.wav": 1.0, "r2.wav": 1.0}


def test_the_frame_record_counts_eligible_excluded_and_drawn_clips():
    frame = [_row("a.wav", 1), _row("b.wav", 1, row=1), _row("c.wav", 2, shard=58)]
    excluded = [(_row("x.wav", 3, row=2), "duration")]
    record = frame_record(frame, excluded, [frame[0]])
    assert record["per_shard"]["39"] == {"eligible": 2, "excluded_duration": 1}
    assert record["per_shard"]["58"] == {"eligible": 1}
    assert record["per_reciter"]["1"] == {"eligible": 2, "uniform": 1, "drawn": True}
    assert record["per_reciter"]["2"] == {"eligible": 1, "drawn": False}
    assert (record["reciters_eligible"], record["reciters_drawn"]) == (2, 1)


def test_a_scan_identity_changes_with_the_census_frame_and_the_decoder():
    census = [_row("a.wav", 1), _row("b.wav", 1, row=1)]
    base = scan_identity(census, {"model": "m"})
    assert base == scan_identity(list(census), {"model": "m"})
    assert base != scan_identity(census[:1], {"model": "m"})
    assert base != scan_identity([census[0], _row("b.wav", 1, row=2)], {"model": "m"})
    assert base != scan_identity(census, {"model": "other"})
