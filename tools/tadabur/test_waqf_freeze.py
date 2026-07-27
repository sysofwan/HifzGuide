"""Unit tests for the waqf eval-set freeze (``tadabur.waqf_freeze``)."""

from __future__ import annotations

import json

from tadabur import waqf_freeze as wf


def _clip(reciter: int, idx: int = 0) -> str:
    return f"tadabur_spk{reciter:04d}_S1_A1_x_{idx:06d}.wav"


def _cand(clip: str, boundary: int, predicted: str) -> dict:
    return {
        "clip_id": clip,
        "audio_ref": clip,
        "surah_ayah": "1:1",
        "boundary_index": boundary,
        "word_index": boundary,
        "start_s": float(boundary),
        "end_s": float(boundary),
        "predicted": predicted,
    }


def _write_candidates(path, rows):
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_materialize_applies_overrides_and_keeps_baseline(tmp_path):
    clip = _clip(1)
    cands = tmp_path / "cand.jsonl"
    _write_candidates(cands, [_cand(clip, 0, "wasl"), _cand(clip, 1, "wasl")])
    overrides = {(clip, 1): {"clip_id": clip, "boundary_index": 1, "verdict": "waqf", "note": "n"}}

    by_clip, stale = wf.materialize(cands, {clip}, overrides)

    assert stale == []
    entries = {e.boundary_index: e for e in by_clip[clip]}
    assert entries[0].verdict == "wasl"  # baseline preserved
    assert entries[1].verdict == "waqf" and entries[1].note == "n"  # override applied
    assert entries[1].predicted == "wasl"  # detector class retained for the metric


def test_stale_override_is_reported_not_placed(tmp_path):
    clip = _clip(1)
    cands = tmp_path / "cand.jsonl"
    _write_candidates(cands, [_cand(clip, 0, "wasl")])
    # boundary 5 no longer exists in the baseline -> stale
    overrides = {(clip, 5): {"clip_id": clip, "boundary_index": 5, "verdict": "waqf"}}

    by_clip, stale = wf.materialize(cands, {clip}, overrides)

    assert len(by_clip[clip]) == 1
    assert [o["boundary_index"] for o in stale] == [5]


def test_freeze_is_reciter_disjoint_and_lists_exclusions(tmp_path):
    clips = {_clip(r) for r in (1, 2, 3, 4)}
    cands = tmp_path / "cand.jsonl"
    _write_candidates(cands, [_cand(c, 0, "wasl") for c in clips])
    by_clip, _ = wf.materialize(cands, clips, {})

    calibration, test, report = wf.freeze(by_clip, clips, test_fraction=0.5, seed=0)

    cal_reciters = set(report["calibration_reciters"])
    test_reciters = set(report["test_reciters"])
    assert cal_reciters.isdisjoint(test_reciters)
    assert report["must_exclude_reciters"] == sorted(cal_reciters | test_reciters)
    assert len(calibration) + len(test) == sum(len(v) for v in by_clip.values())
    # every frozen entry lands on the side matching its reciter partition
    for entry in calibration:
        assert wf.reciter_of(entry.clip_id) in cal_reciters
    for entry in test:
        assert wf.reciter_of(entry.clip_id) in test_reciters


def test_binary_block_counts_waqf_as_positive_mwc_as_diagnostic_tag(tmp_path):
    # one reciter per class so the split is deterministic; mix all three verdicts
    clip = _clip(1)
    cands = tmp_path / "cand.jsonl"
    _write_candidates(
        cands,
        [
            _cand(clip, 0, "waqf"),
            _cand(clip, 1, "wasl"),
            _cand(clip, 2, "mid_word_closure"),
            _cand(clip, 3, "wasl"),
        ],
    )
    by_clip, _ = wf.materialize(cands, {clip}, {})

    _, _, report = wf.freeze(by_clip, {clip}, test_fraction=0.5, seed=0)

    # the single clip lands in exactly one partition; combine both to read its counts
    combined = {
        "waqf": report["binary"]["calibration"]["waqf"] + report["binary"]["test"]["waqf"],
        "not_waqf": report["binary"]["calibration"]["not_waqf"]
        + report["binary"]["test"]["not_waqf"],
        "closure_tag": report["binary"]["calibration"]["closure_tag"]
        + report["binary"]["test"]["closure_tag"],
    }
    assert combined["waqf"] == 1  # only verdict=='waqf' is a positive
    assert combined["not_waqf"] == 3  # wasl + wasl + mid_word_closure
    assert combined["closure_tag"] == 1  # mwc reported only as a diagnostic tag

