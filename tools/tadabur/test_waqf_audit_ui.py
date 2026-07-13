"""Unit tests for the waqf adjudication UI's pure logic (``tadabur.waqf_audit_ui``).

Covers the event store's persist/resume round-trip through the fixture file, the
per-class progress + verdict-confusion stats, worklist loading, the item view, and
the apply-verdict request path (set / clear / unknown-boundary reject) — all without
binding a socket.
"""

from __future__ import annotations

import json

import pytest

from .audit_http import sniff_audio_content_type
from .waqf_event_fixtures import MID_WORD_CLOSURE, WAQF, WASL, load_waqf_events
from .waqf_event_sampler import WaqfCandidateItem, write_worklist
from .waqf_audit_ui import (
    WaqfAuditServer,
    WaqfEventStore,
    class_stats,
    load_worklist,
)


def _item(clip, idx, predicted=WAQF) -> WaqfCandidateItem:
    return WaqfCandidateItem(
        clip_id=clip, audio_ref=f"{clip}.wav", surah_ayah="2:5", boundary_index=idx,
        word_index=idx + 1, start_s=float(idx), end_s=idx + 0.3, predicted=predicted,
        local_audio_path=f"aa_{clip}",
    )


def _server(tmp_path, items, uthmani=None):
    store = WaqfEventStore.load(tmp_path / "waqf_events.jsonl")
    return WaqfAuditServer(items, uthmani or {}, store, tmp_path)


def test_sniff_shared_helper_reachable():
    assert sniff_audio_content_type(b"RIFF\x00\x00\x00\x00WAVEfmt ") == "audio/wav"


def test_store_persists_and_reloads(tmp_path):
    path = tmp_path / "waqf_events.jsonl"
    server = _server(tmp_path, [_item("a", 0, WASL)])
    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": WAQF, "note": "missed stop"})

    # File holds exactly the one adjudicated boundary.
    assert len(load_waqf_events(path)) == 1
    reloaded = WaqfEventStore.load(path)
    assert reloaded.verdict_of(("a", 0)) == WAQF
    assert reloaded.note_of(("a", 0)) == "missed stop"


def test_store_overwrite_and_clear(tmp_path):
    path = tmp_path / "waqf_events.jsonl"
    server = _server(tmp_path, [_item("a", 0, WAQF)])
    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": WAQF})
    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": MID_WORD_CLOSURE})
    assert server.store.verdict_of(("a", 0)) == MID_WORD_CLOSURE
    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": None})
    assert server.store.verdict_of(("a", 0)) is None
    assert load_waqf_events(path) == []


def test_apply_label_rejects_unknown_boundary(tmp_path):
    server = _server(tmp_path, [_item("a", 0)])
    with pytest.raises(KeyError):
        server.apply_label({"clip_id": "ghost", "boundary_index": 0, "verdict": WAQF})
    # A boundary index not in the worklist is equally rejected.
    with pytest.raises(KeyError):
        server.apply_label({"clip_id": "a", "boundary_index": 9, "verdict": WAQF})


def test_class_stats_confusion(tmp_path):
    items = [_item("a", 0, WASL), _item("b", 0, WASL), _item("c", 0, WASL), _item("d", 0, WAQF)]
    server = _server(tmp_path, items)
    # Two of the wasl-predicted boundaries were actually stops (false-wasl), one wasl.
    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": WAQF})
    server.apply_label({"clip_id": "b", "boundary_index": 0, "verdict": WAQF})
    server.apply_label({"clip_id": "c", "boundary_index": 0, "verdict": WASL})

    stats = {s["predicted"]: s for s in class_stats(items, server.store)}
    wasl = stats[WASL]
    assert wasl["total"] == 3 and wasl["labelled"] == 3
    assert wasl["verdicts"][WAQF] == 2 and wasl["verdicts"][WASL] == 1
    assert stats[WAQF]["labelled"] == 0
    # Every class appears even with no candidates.
    assert stats[MID_WORD_CLOSURE]["total"] == 0


def test_state_item_view_carries_boundary_context(tmp_path):
    items = [_item("a", 2, MID_WORD_CLOSURE)]
    server = _server(tmp_path, items, uthmani={"2:5": "نص الآية"})
    view = server.state()["items"][0]
    assert view["clip_id"] == "a" and view["boundary_index"] == 2
    assert view["predicted"] == MID_WORD_CLOSURE
    assert view["word_index"] == 3 and view["start_s"] == 2.0
    assert view["uthmani"] == "نص الآية"
    assert view["audio_url"] == "/audio/aa_a"
    assert view["audio_available"] is False
    assert server.state()["classes"] == [WAQF, WASL, MID_WORD_CLOSURE]


def test_load_worklist_roundtrip(tmp_path):
    items = [_item("a", 0, WAQF), _item("a", 1, WASL)]
    path = tmp_path / "worklist.jsonl"
    write_worklist(items, path)
    assert load_worklist(path) == items


def test_apply_label_enriches_fixture_from_worklist(tmp_path):
    server = _server(tmp_path, [_item("a", 0, WASL)])
    server.apply_label({"clip_id": "a", "boundary_index": 0, "verdict": WAQF, "note": "n"})
    entry = server.store.entries[("a", 0)]
    # Fields the POST omits (audio_ref, surah_ayah, word_index, times, predicted)
    # are recovered from the worklist row, not trusted from the request.
    assert entry.audio_ref == "a.wav" and entry.surah_ayah == "2:5"
    assert entry.word_index == 1 and entry.predicted == WASL
