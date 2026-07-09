"""Unit tests for the P3.5 audit UI's pure logic (``tadabur.audit_ui``).

Covers the label store's persist/resume round-trip through the eval-fixture files,
the per-contrast poison-rate stats, audio content-type sniffing, worklist loading,
and the apply-label request path — all without binding a socket.
"""

from __future__ import annotations

import json

import pytest

from . import eval_fixtures
from .audit_sampler import WorklistItem, write_worklist
from .audit_ui import (
    AuditServer,
    LabelStore,
    contrast_stats,
    load_worklist,
    sniff_audio_content_type,
    surah_ayah_index,
)
from .eval_fixtures import ACCEPT, REJECT, EvalFixtureEntry
from .manifest import ManifestRecord


def _item(clip_id: str, contrast: str, ratio: float = 0.8) -> WorklistItem:
    return WorklistItem(
        clip_id=clip_id,
        contrast=contrast,
        match_ratio=ratio,
        audio_ref=clip_id,
        local_audio_path=f"aa_{clip_id}",
    )


def _paths(tmp_path):
    return tmp_path / "should_accept.jsonl", tmp_path / "should_reject.jsonl"


def test_sniff_audio_content_type():
    assert sniff_audio_content_type(b"RIFF\x00\x00\x00\x00WAVEfmt ") == "audio/wav"
    assert sniff_audio_content_type(b"ID3\x03\x00rest") == "audio/mpeg"
    assert sniff_audio_content_type(b"\xff\xfb\x90\x00frame") == "audio/mpeg"
    assert sniff_audio_content_type(b"OggS\x00\x02") == "audio/ogg"
    assert sniff_audio_content_type(b"fLaC\x00\x00") == "audio/flac"
    assert sniff_audio_content_type(b"nonsense") == "application/octet-stream"


def test_label_store_persists_and_reloads(tmp_path):
    accept, reject = _paths(tmp_path)
    store = LabelStore.load(accept, reject)
    store.set(EvalFixtureEntry("c1", "c1", "2:5", "shadda", ACCEPT, "clean but soft"))
    store.set(EvalFixtureEntry("c2", "c2", "3:1", "ح↔ه", REJECT, "clearly ه for ح"))

    # Files hold exactly one entry each, in the right verdict set.
    assert len(eval_fixtures.load_eval_fixtures(accept, ACCEPT)) == 1
    assert len(eval_fixtures.load_eval_fixtures(reject, REJECT)) == 1

    # A fresh store resumes the same verdicts.
    reloaded = LabelStore.load(accept, reject)
    assert reloaded.verdict_of("c1", "shadda") == ACCEPT
    assert reloaded.note_of("c1", "shadda") == "clean but soft"
    assert reloaded.verdict_of("c2", "ح↔ه") == REJECT


def test_label_store_overwrite_moves_between_sets(tmp_path):
    accept, reject = _paths(tmp_path)
    store = LabelStore.load(accept, reject)
    store.set(EvalFixtureEntry("c1", "c1", "2:5", "shadda", ACCEPT))
    store.set(EvalFixtureEntry("c1", "c1", "2:5", "shadda", REJECT))
    assert eval_fixtures.load_eval_fixtures(accept, ACCEPT) == []
    assert len(eval_fixtures.load_eval_fixtures(reject, REJECT)) == 1
    assert store.verdict_of("c1", "shadda") == REJECT


def test_label_store_clear(tmp_path):
    accept, reject = _paths(tmp_path)
    store = LabelStore.load(accept, reject)
    store.set(EvalFixtureEntry("c1", "c1", "2:5", "shadda", ACCEPT))
    store.clear("c1", "shadda")
    assert store.verdict_of("c1", "shadda") is None
    assert eval_fixtures.load_eval_fixtures(accept, ACCEPT) == []


def test_label_store_rejects_bad_verdict(tmp_path):
    accept, reject = _paths(tmp_path)
    store = LabelStore.load(accept, reject)
    with pytest.raises(ValueError):
        store.set(EvalFixtureEntry("c1", "c1", "2:5", "shadda", "maybe"))


def test_contrast_stats_poison_rate(tmp_path):
    accept, reject = _paths(tmp_path)
    store = LabelStore.load(accept, reject)
    items = [_item("a", "ح↔ه"), _item("b", "ح↔ه"), _item("c", "ح↔ه"), _item("d", "shadda")]
    store.set(EvalFixtureEntry("a", "a", "1:1", "ح↔ه", REJECT))
    store.set(EvalFixtureEntry("b", "b", "1:2", "ح↔ه", ACCEPT))
    # "c" and "d" left unlabelled.

    stats = {s["contrast"]: s for s in contrast_stats(items, store)}
    ha = stats["ح↔ه"]
    assert ha["total"] == 3 and ha["labelled"] == 2
    assert ha["accept"] == 1 and ha["reject"] == 1
    assert ha["poison_rate"] == pytest.approx(0.5)
    assert stats["shadda"]["poison_rate"] is None  # nothing labelled yet


def test_load_worklist_roundtrip(tmp_path):
    items = [_item("a", "shadda"), _item("b", "ح↔ه")]
    path = tmp_path / "worklist.jsonl"
    write_worklist(items, path)
    assert load_worklist(path) == items


def test_surah_ayah_index(tmp_path):
    manifest = tmp_path / "m.jsonl"
    recs = [
        ManifestRecord("a", "2:255", 0.9, 3.0, 7, ("shadda",)),
        ManifestRecord("b", "112:1", 0.7, 2.0, 7, ("ح↔ه",)),
    ]
    with open(manifest, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({
                "audio_filename": r.audio_filename, "surah_ayah": r.surah_ayah,
                "match_ratio": r.match_ratio, "ayah_duration_s": r.ayah_duration_s,
                "reciter_id": r.reciter_id, "contrasts": list(r.contrasts),
            }) + "\n")
    idx = surah_ayah_index(manifest)
    assert idx == {"a": "2:255", "b": "112:1"}


def test_apply_label_enriches_and_rejects_unknown(tmp_path):
    accept, reject = _paths(tmp_path)
    items = [_item("a", "shadda")]
    store = LabelStore.load(accept, reject)
    server = AuditServer(items, {"a": "2:5"}, store, tmp_path)

    res = server.apply_label({"clip_id": "a", "contrast": "shadda", "verdict": ACCEPT, "note": "ok"})
    assert any(s["contrast"] == "shadda" and s["accept"] == 1 for s in res["stats"])
    # surah_ayah recovered from the manifest index, not the worklist.
    assert store.entries[("a", "shadda")].surah_ayah == "2:5"

    with pytest.raises(KeyError):
        server.apply_label({"clip_id": "ghost", "contrast": "shadda", "verdict": ACCEPT})


def test_apply_label_clear(tmp_path):
    accept, reject = _paths(tmp_path)
    items = [_item("a", "shadda")]
    store = LabelStore.load(accept, reject)
    server = AuditServer(items, {"a": "2:5"}, store, tmp_path)
    server.apply_label({"clip_id": "a", "contrast": "shadda", "verdict": ACCEPT})
    server.apply_label({"clip_id": "a", "contrast": "shadda", "verdict": None})
    assert store.verdict_of("a", "shadda") is None
