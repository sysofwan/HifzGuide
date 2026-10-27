"""The S6 listening page's load guard and its verdict mapping (``tadabur.tlog_audit_ui``).

The page grades blind: the owner reads the candidate ayat, never the label. A worklist label
with no Uthmani text would serve an empty candidate and silently degrade an hour of listening,
so the session refuses to start instead. ``Session.record`` is the one place a click becomes one
of the five frozen verdicts, so its referent per stratum and its ayah bounds are pinned here.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from . import tlog_ident_run as run
from .tlog_audit_ui import Session


def _worklist(tmp_path, filename_label="2:1", class_label="2:2", **overrides):
    row = {"population": "look", "split": "clean", "shard_file": "clean-00000-of-00411",
           "row": 7, "role": "look", "audio_path": "audio/clean-00000-of-00411.parquet",
           "audio_sha256": "0" * 64, "duration_s": 12.5, "filename_label": filename_label,
           "assigned_label": filename_label, "class_label": class_label,
           "label_source": "both", "id_floor_ok": True, "overlaps_filename": False,
           "blocking_reason": None, "stratum": "both", "local_audio": "ab_00000000.wav",
           **overrides}
    path = tmp_path / "worklist.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path, row


@pytest.fixture
def quran_db(tmp_path):
    path = tmp_path / "quran.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE ayahs (surah INTEGER, ayah INTEGER, text TEXT)")
        conn.executemany("INSERT INTO ayahs VALUES (?, ?, ?)",
                         [(2, 1, "الم"), (2, 2, "ذلك الكتاب")])
    return path


def _session(tmp_path, quran_db, **overrides):
    worklist, row = _worklist(tmp_path, **overrides)
    return Session.load(worklist, tmp_path, tmp_path / "verdicts.jsonl", quran_db), row


def test_a_label_without_text_refuses_to_serve(tmp_path, quran_db):
    worklist, _row = _worklist(tmp_path, filename_label="2:1", class_label="3:7")
    with pytest.raises(ValueError, match=r"no Uthmani text for 1 of 2 ayah labels \(3:7\)"):
        Session.load(worklist, tmp_path, tmp_path / "verdicts.jsonl", quran_db)


def test_a_missing_database_refuses_to_serve(tmp_path):
    worklist, _row = _worklist(tmp_path)
    with pytest.raises(ValueError, match="no Uthmani text for 2 of 2"):
        Session.load(worklist, tmp_path, tmp_path / "verdicts.jsonl", tmp_path / "absent.db")


def test_the_loaded_session_serves_both_candidate_texts(tmp_path, quran_db):
    state, _row = _session(tmp_path, quran_db)
    items = state.payload()["items"]
    assert sorted(candidate["text"] for candidate in items[0]["candidates"]) == sorted(
        ["الم", "ذلك الكتاب"])
    assert items[0]["audio_url"] == "/audio/ab_00000000.wav"


def test_a_relabelled_click_scores_against_the_assigned_label(tmp_path, quran_db):
    state, row = _session(tmp_path, quran_db, stratum="relabelled", filename_label="2:1",
                          assigned_label="2:2", class_label="2:2", label_source="identified")
    key = run._item_key(row)
    assert state.record(key, "candidate", "2:2", "")["verdict"] == run.RIGHT
    assert state.record(key, "candidate", "2:1", "")["verdict"] == run.OTHER


def test_another_ayah_must_be_a_reference_the_mushaf_has(tmp_path, quran_db):
    state, row = _session(tmp_path, quran_db)
    key = run._item_key(row)
    for bad in ("0:1", "115:1", "2:3", "2:0", "two:1", "2:1:3"):
        with pytest.raises(ValueError, match="the mushaf has"):
            state.record(key, "another", bad, "")
    saved = state.record(key, "another", "2:2", "")
    assert saved["verdict"] == run.OTHER and saved["ayah"] == "2:2"


def test_the_note_the_owner_types_is_persisted_with_the_verdict(tmp_path, quran_db):
    state, row = _session(tmp_path, quran_db)
    state.record(run._item_key(row), "candidate", "2:1", "repeats the first ayah twice")
    stored = json.loads((tmp_path / "verdicts.jsonl").read_text(encoding="utf-8"))
    assert stored["note"] == "repeats the first ayah twice"
    assert set(stored) >= {"split", "shard_file", "row", "audio_path", "audio_sha256",
                           "verdict", "ayah", "note"}
