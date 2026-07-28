"""Unit tests for the P3.5 audit UI's pure logic (``tadabur.audit_ui``).

Covers the label store's persist/resume round-trip through the eval-fixture files,
the per-contrast poison-rate stats, audio content-type sniffing, worklist loading,
and the apply-label request path — all without binding a socket.
"""

from __future__ import annotations

import json

import pytest

from . import eval_fixtures
from .audit_http import sniff_audio_content_type
from .audit_sampler import WorklistItem, write_worklist
from .audit_ui import (
    AuditServer,
    LabelStore,
    align_phonemes,
    contrast_stats,
    load_worklist,
    predicted_phoneme_index,
    surah_ayah_index,
    uthmani_index,
    raw_reference_index,
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


def _write_manifest(path, recs):
    from dataclasses import asdict
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def test_predicted_phoneme_index(tmp_path):
    manifest = tmp_path / "m.jsonl"
    _write_manifest(manifest, [
        ManifestRecord("a", "2:255", 0.9, 3.0, 7, ("shadda",), "بتثج"),
        ManifestRecord("b", "112:1", 0.7, 2.0, 7, (), ""),
    ])
    assert predicted_phoneme_index(manifest) == {"a": "بتثج", "b": ""}


def test_align_phonemes_marks_match_sub_and_gaps():
    cols = align_phonemes("بتثج", "بتثج")
    assert cols and all(c["kind"] == "match" for c in cols)
    # A substituted phoneme is flagged, not silently matched.
    mixed = align_phonemes("بتشج", "بتثج")
    kinds = {c["kind"] for c in mixed}
    assert "match" in kinds and (kinds & {"sub", "del", "ins"})
    # Empty input yields no columns rather than raising.
    assert align_phonemes("", "بتثج") == []


def test_uthmani_index_reads_quran_db(tmp_path):
    import sqlite3
    db = tmp_path / "quran.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE ayahs (surah INTEGER, ayah INTEGER, text TEXT)")
    conn.execute("INSERT INTO ayahs VALUES (2, 77, 'نص')")
    conn.commit()
    conn.close()
    assert uthmani_index(db, {"2:77", "9:99"}) == {"2:77": "نص"}
    # Missing DB degrades to empty, never raises.
    assert uthmani_index(tmp_path / "nope.db", {"2:77"}) == {}


def test_raw_reference_index_reads_phonemes_column(tmp_path):
    import sqlite3
    db = tmp_path / "quran.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE ayahs (surah INTEGER, ayah INTEGER, text TEXT, phonemes TEXT)"
    )
    conn.execute("INSERT INTO ayahs VALUES (2, 77, 'نص', 'فَذُۥۥقُۥۥ')")
    conn.commit()
    conn.close()
    # Returns the raw phonemes (with madd/idgham markers), not the normalized form.
    assert raw_reference_index(db, {"2:77", "9:99"}) == {"2:77": "فَذُۥۥقُۥۥ"}
    # Missing DB degrades to empty, never raises.
    assert raw_reference_index(tmp_path / "nope.db", {"2:77"}) == {}


def test_state_includes_ayah_text_and_phoneme_diff(tmp_path):
    accept, reject = _paths(tmp_path)
    items = [_item("a", "shadda")]
    store = LabelStore.load(accept, reject)
    server = AuditServer(
        items, {"a": "2:77"}, store, tmp_path,
        uthmani={"2:77": "نص الآية"},
        predicted={"a": "بتشج"},
        reference={"2:77": "بتثج"},
        raw_reference={"2:77": "بتثثج"},
    )
    view = server.state()["items"][0]
    assert view["uthmani"] == "نص الآية"
    assert view["predicted_phonemes"] == "بتشج"
    assert view["reference_phonemes"] == "بتثج"
    # Raw reference (with tajweed markers) is surfaced alongside the normalized one.
    assert view["raw_reference_phonemes"] == "بتثثج"
    assert view["alignment"]  # non-empty aligned columns


def test_item_view_prefers_per_clip_reference_in_segment_mode(tmp_path):
    # In waqf-segment mode reference/raw_reference/uthmani are keyed by clip_id
    # (the segment id), not surah_ayah — two segments of one ayah must not collapse
    # to the same full-ayah reference.
    accept, reject = _paths(tmp_path)
    items = [_item("seg0", "shadda"), _item("seg1", "shadda")]
    store = LabelStore.load(accept, reject)
    server = AuditServer(
        items,
        {"seg0": "2:77", "seg1": "2:77"},
        store,
        tmp_path,
        uthmani={"seg0": "الأولى", "seg1": "الثانية"},
        predicted={"seg0": "بتج", "seg1": "بتشج"},
        reference={"seg0": "بتج", "seg1": "بتثج"},
        raw_reference={"seg0": "بتج", "seg1": "بتثثج"},
    )
    views = {v["clip_id"]: v for v in server.state()["items"]}
    assert views["seg0"]["uthmani"] == "الأولى"
    assert views["seg1"]["uthmani"] == "الثانية"
    assert views["seg0"]["reference_phonemes"] == "بتج"
    assert views["seg1"]["reference_phonemes"] == "بتثج"
    assert views["seg1"]["raw_reference_phonemes"] == "بتثثج"


def test_segment_display_index_keys_by_clip_id(tmp_path):
    from .audit_ui import segment_display_index

    manifest = tmp_path / "segments.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "audio_filename": "clip__seg0.wav", "surah_ayah": "2:77",
            "predicted_phonemes": "بتج", "reference_phonemes": "بتج",
            "raw_reference_phonemes": "بتج", "uthmani": "الأولى",
        }, ensure_ascii=False) + "\n")
        f.write(json.dumps({
            "audio_filename": "clip__seg1.wav", "surah_ayah": "2:77",
            "predicted_phonemes": "بتشج", "reference_phonemes": "بتثج",
            "raw_reference_phonemes": "بتثثج", "uthmani": "الثانية",
        }, ensure_ascii=False) + "\n")
    idx = segment_display_index(manifest)
    assert idx["surah_ayah"] == {"clip__seg0.wav": "2:77", "clip__seg1.wav": "2:77"}
    assert idx["reference"]["clip__seg1.wav"] == "بتثج"
    assert idx["uthmani"]["clip__seg0.wav"] == "الأولى"



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


def test_whole_clip_state_absent_by_default(tmp_path):
    # No --clip-status: the state flags the whole-clip view unavailable rather than 500ing.
    accept, reject = _paths(tmp_path)
    store = LabelStore.load(accept, reject)
    server = AuditServer([_item("a", "shadda")], {"a": "2:5"}, store, tmp_path)
    assert server.state()["whole_clip_available"] is False
    assert server.whole_clip_state() == {"available": False}


def test_whole_clip_state_serializes_audit(tmp_path):
    from tadabur.clip_status import ClipStatus, write_clip_status
    from training.whole_clip_audit import build_whole_clip_audit

    manifest = tmp_path / "segments.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "audio_filename": "a__seg0", "clip_audio_filename": "a.wav", "surah_ayah": "78:2",
            "reciter_id": 1, "segment_index": 0, "word_start": 0, "word_end": 3,
            "start_s": 0.0, "end_s": 4.0, "reference_phonemes": "ءبت", "uthmani": "أ ب ت",
            "raw_reference_phonemes": "ءَبِتُ", "raw_word_offsets": [0, 2, 4, 6],
        }, ensure_ascii=False) + "\n")
    status_path = tmp_path / "clip_status.jsonl"
    write_clip_status(status_path, [ClipStatus(
        audio_filename="a.wav", surah_ayah="78:2", reciter_id=1, n_words=3,
        duration_s=4.0, recitation_start_s=0.0, recitation_end_s=4.0,
    )])

    audit = build_whole_clip_audit(manifest, status_path)
    accept, reject = _paths(tmp_path)
    store = LabelStore.load(accept, reject)
    server = AuditServer([], {}, store, tmp_path, whole_clip_audit=audit)

    assert server.state()["whole_clip_available"] is True
    payload = server.whole_clip_state()
    assert payload["available"] is True
    assert payload["summary"]["clips_included"] == 1
    clip = payload["clips"][0]
    assert clip["clip_id"] == "a.wav" and clip["included"] is True
    assert clip["whole_clip_label"] == "ءَبِتُ"  # the raw, tashkeel-bearing label (ADR-0003)
    assert clip["windows"][0]["phoneme_label"] == "ءَبِتُ"
    # asdict serializes cleanly to JSON (segment_indices tuple -> array).
    assert json.dumps(payload, ensure_ascii=False)


def test_signoff_state_unavailable_without_reports(tmp_path):
    from .audit_ui import SignoffReports

    accept, reject = _paths(tmp_path)
    store = LabelStore.load(accept, reject)
    server = AuditServer([], {}, store, tmp_path)
    assert server.state()["signoff_available"] is False
    assert server.signoff_state() == {"available": False}


def test_signoff_state_reads_reports(tmp_path):
    from .audit_ui import SignoffReports

    event = tmp_path / "f2.json"
    event.write_text(json.dumps({
        "calibrated_silence_threshold": 0.5, "duration_gate_ms": 300,
        "test": {"silence_threshold": 0.5, "counts": {}, "metrics": {"f1": 0.9}},
        "calibration": {"silence_threshold": 0.5, "counts": {}, "metrics": {"f1": 0.9}},
        "blank_run_reference": {"available": False, "reason": "no decode"},
    }), encoding="utf-8")
    integration = tmp_path / "h.json"
    integration.write_text(json.dumps({"passed": True, "summary": "ok"}), encoding="utf-8")

    accept, reject = _paths(tmp_path)
    store = LabelStore.load(accept, reject)
    server = AuditServer([], {}, store, tmp_path,
                         signoff_reports=SignoffReports(event_eval=event, integration=integration))
    assert server.state()["signoff_available"] is True
    payload = server.signoff_state()
    assert payload["available"] is True
    assert payload["event_eval"]["available"] is True
    assert payload["integration"]["passed"] is True
    # E absent → pending, so not ready.
    assert payload["readiness"]["ready"] is False
    assert json.dumps(payload, ensure_ascii=False)
