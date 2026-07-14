"""Unit tests for the waqf adjudication UI's pure logic (``tadabur.waqf_audit_ui``).

The UI is correction-based and per-clip: the candidate manifest is the assumed-correct
baseline, the reviewer overrides only false positives / false negatives, and a per-clip
``reviewed`` flag admits a clip to the eval set. These tests cover the override store's
persist/resume round-trip, the clip view (predicted ⊕ override → ``truth``), the review-stats
tallies, the reviewed-clip store, and the request paths — all without binding a socket.
"""

from __future__ import annotations

import json

import pytest

from .audit_http import sniff_audio_content_type
from .waqf_event_fixtures import MID_WORD_CLOSURE, WAQF, WASL, load_waqf_events
from .waqf_audit_ui import (
    ReviewedClipStore,
    WaqfAuditServer,
    WaqfEventStore,
    boundary_view,
    load_candidates_by_clip,
    load_clip_worklist,
    review_stats,
    reviewed_path_for,
)
from .waqf_segments import _save_local_clip


def _cand(clip, idx, predicted=WASL, surah_ayah="2:5"):
    """One candidate-manifest row (a word edge) as a plain dict."""
    return {
        "clip_id": clip, "audio_ref": f"{clip}.wav", "surah_ayah": surah_ayah,
        "boundary_index": idx, "word_index": idx, "start_s": float(idx),
        "end_s": float(idx), "predicted": predicted,
    }


def _server(tmp_path, by_clip, uthmani=None, reviewed=None):
    store = WaqfEventStore.load(tmp_path / "waqf_events.jsonl")
    rev = ReviewedClipStore(tmp_path / "waqf_reviewed_clips.json", set(reviewed or []))
    clips = list(by_clip.keys())
    return WaqfAuditServer(clips, by_clip, uthmani or {}, store, rev, tmp_path)


def test_sniff_shared_helper_reachable():
    assert sniff_audio_content_type(b"RIFF\x00\x00\x00\x00WAVEfmt ") == "audio/wav"


def test_store_persists_and_reloads(tmp_path):
    path = tmp_path / "waqf_events.jsonl"
    server = _server(tmp_path, {"a": [_cand("a", 0, WASL)]})
    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": WAQF, "note": "missed stop"})

    # File holds exactly the one correction.
    assert len(load_waqf_events(path)) == 1
    reloaded = WaqfEventStore.load(path)
    assert reloaded.verdict_of(("a", 0)) == WAQF
    assert reloaded.note_of(("a", 0)) == "missed stop"


def test_override_equal_to_prediction_clears_no_redundant_confirmation(tmp_path):
    # Assume-correct-by-default: setting a verdict equal to the predicted class stores
    # nothing (no "confirmation" line), and an explicit None clears an override.
    path = tmp_path / "waqf_events.jsonl"
    server = _server(tmp_path, {"a": [_cand("a", 0, WAQF)]})
    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": WAQF})
    assert server.store.verdict_of(("a", 0)) is None
    assert load_waqf_events(path) == []

    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": WASL})
    assert server.store.verdict_of(("a", 0)) == WASL
    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": None})
    assert server.store.verdict_of(("a", 0)) is None
    assert load_waqf_events(path) == []


def test_apply_label_rejects_unknown_boundary(tmp_path):
    server = _server(tmp_path, {"a": [_cand("a", 0, WAQF)]})
    with pytest.raises(KeyError):
        server.apply_label({"clip_id": "ghost", "boundary_index": 0, "verdict": WAQF})
    # A boundary index not in the clip's candidate set is equally rejected.
    with pytest.raises(KeyError):
        server.apply_label({"clip_id": "a", "boundary_index": 9, "verdict": WAQF})


def test_apply_label_accepts_any_candidate_boundary_not_just_stops(tmp_path):
    # False negatives: the reviewer marks a plain wasl word edge as a stop. Any word edge
    # in the manifest is a legal override target, not only the VAD-predicted stops.
    server = _server(tmp_path, {"a": [_cand("a", i, WASL) for i in range(4)] + [_cand("a", 4, WAQF)]})
    server.apply_label({"clip_id": "a", "boundary_index": 2, "verdict": WAQF})
    assert server.store.verdict_of(("a", 2)) == WAQF
    # The stored entry recovers its fields from the manifest, not the request.
    entry = server.store.entries[("a", 2)]
    assert entry.predicted == WASL and entry.word_index == 2 and entry.audio_ref == "a.wav"


def test_invalid_verdict_leaves_store_and_state_unchanged(tmp_path):
    path = tmp_path / "waqf_events.jsonl"
    server = _server(tmp_path, {"a": [_cand("a", 0, WASL)]})
    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": WAQF})
    before_entries = dict(server.store.entries)
    before_state = server.state()

    with pytest.raises(ValueError):
        server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": "bogus"})

    assert server.store.entries == before_entries
    assert server.store.verdict_of(("a", 0)) == WAQF
    assert server.state() == before_state
    assert WaqfEventStore.load(path).verdict_of(("a", 0)) == WAQF
    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": MID_WORD_CLOSURE})
    assert server.store.verdict_of(("a", 0)) == MID_WORD_CLOSURE


def test_review_stats_classifies_false_positive_negative_and_class_fix(tmp_path):
    by_clip = {"a": [_cand("a", 0, WAQF), _cand("a", 1, WASL), _cand("a", 2, WAQF), _cand("a", 3, WASL)]}
    server = _server(tmp_path, by_clip)
    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": WASL})            # FP
    server.apply_label({"clip_id": "a", "boundary_index": 1, "verdict": WAQF})            # FN
    server.apply_label({"clip_id": "a", "boundary_index": 2, "verdict": MID_WORD_CLOSURE})  # class fix
    # boundary 3 left at predicted wasl → no correction.
    stats = review_stats(server)
    assert stats["false_positive"] == 1
    assert stats["false_negative"] == 1
    assert stats["class_fix"] == 1
    assert stats["clips_total"] == 1 and stats["clips_reviewed"] == 0


def test_clip_view_truth_is_prediction_overlaid_with_override(tmp_path):
    by_clip = {"a": [_cand("a", 0, WAQF), _cand("a", 1, WASL), _cand("a", 2, MID_WORD_CLOSURE)]}
    server = _server(tmp_path, by_clip, uthmani={"2:5": "نص الآية هنا"})
    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": WASL})  # FP: waqf→wasl

    clip = server.state()["clips"][0]
    assert clip["clip_id"] == "a" and clip["surah_ayah"] == "2:5"
    assert clip["uthmani"] == "نص الآية هنا"
    assert clip["audio_url"] == "/audio/a.wav"
    assert clip["reviewed"] is False
    bounds = {b["boundary_index"]: b for b in clip["boundaries"]}
    assert bounds[0]["predicted"] == WAQF and bounds[0]["verdict"] == WASL and bounds[0]["truth"] == WASL
    assert bounds[1]["verdict"] is None and bounds[1]["truth"] == WASL      # untouched
    assert bounds[2]["verdict"] is None and bounds[2]["truth"] == MID_WORD_CLOSURE
    assert server.state()["classes"] == [WAQF, WASL, MID_WORD_CLOSURE]


def test_state_is_one_page_per_clip(tmp_path):
    by_clip = {"a": [_cand("a", 0, WAQF), _cand("a", 1, WASL)], "b": [_cand("b", 0, WAQF)]}
    server = _server(tmp_path, by_clip)
    clips = {c["clip_id"]: c for c in server.state()["clips"]}
    assert set(clips) == {"a", "b"}
    assert [b["boundary_index"] for b in clips["a"]["boundaries"]] == [0, 1]
    assert len(clips["b"]["boundaries"]) == 1


def test_reviewed_flag_persists_and_counts(tmp_path):
    server = _server(tmp_path, {"a": [_cand("a", 0, WAQF)], "b": [_cand("b", 0, WAQF)]})
    out = server.apply_review({"clip_id": "a", "reviewed": True})
    assert out["stats"]["clips_reviewed"] == 1
    assert server.state()["clips"][0]["reviewed"] is True
    # Persisted to the sibling file and reloadable.
    reloaded = ReviewedClipStore.load(tmp_path / "waqf_reviewed_clips.json")
    assert reloaded.is_reviewed("a") and not reloaded.is_reviewed("b")
    # Un-reviewing clears it.
    server.apply_review({"clip_id": "a", "reviewed": False})
    assert review_stats(server)["clips_reviewed"] == 0
    with pytest.raises(KeyError):
        server.apply_review({"clip_id": "ghost", "reviewed": True})


def test_boundary_view_shape(tmp_path):
    view = boundary_view(_cand("a", 3, WAQF), verdict=WASL, note="no stop")
    assert view["boundary_index"] == 3 and view["word_index"] == 3
    assert view["predicted"] == WAQF and view["verdict"] == WASL and view["truth"] == WASL
    assert view["note"] == "no stop"
    # Untouched → truth falls back to prediction.
    assert boundary_view(_cand("a", 3, WAQF), verdict=None, note="")["truth"] == WAQF


def test_load_candidates_by_clip_and_clip_worklist(tmp_path):
    manifest = tmp_path / "candidates.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for idx, predicted in [(2, WASL), (0, WAQF), (1, WASL)]:
            f.write(json.dumps(_cand("a", idx, predicted)) + "\n")
        f.write(json.dumps(_cand("b", 0, WAQF)) + "\n")
    by_clip = load_candidates_by_clip(manifest)
    assert set(by_clip) == {"a", "b"}
    assert [r["boundary_index"] for r in by_clip["a"]] == [0, 1, 2]  # sorted

    clips_path = tmp_path / "clips.jsonl"
    clips_path.write_text('{"clip_id": "b"}\n{"clip_id": "a"}\n{"clip_id": "b"}\n', encoding="utf-8")
    assert load_clip_worklist(clips_path) == ["b", "a"]  # order preserved, de-duped


def test_reviewed_path_is_beside_fixtures(tmp_path):
    assert reviewed_path_for(tmp_path / "waqf_events.jsonl") == tmp_path / "waqf_reviewed_clips.json"


def test_ui_audio_dir_is_populated_by_waqf_segments_staging(tmp_path):
    # End-to-end audio contract: a clip staged by tadabur.waqf_segments (under its raw
    # audio_filename) is served by the UI under the same name the manifest carries.
    import numpy as np

    audio_ref = "tadabur_spk0106_S77_A30_000.wav"
    row = _cand(audio_ref, 0, WAQF)
    row["audio_ref"] = audio_ref
    _save_local_clip(tmp_path, audio_ref, np.zeros(16, dtype=np.float32))

    server = _server(tmp_path, {audio_ref: [row]})
    clip = server.state()["clips"][0]
    assert clip["audio_url"] == f"/audio/{audio_ref}"
    assert clip["audio_available"] is True
    assert (tmp_path / audio_ref).is_file()
