"""The S6 listening page's load guard (``tadabur.tlog_audit_ui.Session``).

The page grades blind: the owner reads the candidate ayat, never the label. A worklist label
with no Uthmani text would serve an empty candidate and silently degrade an hour of listening,
so the session refuses to start instead.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from .tlog_audit_ui import Session


def _worklist(tmp_path, *labels: str):
    row = {"population": "look", "split": "clean", "shard_file": "clean-00000-of-00411",
           "row": 7, "role": "look", "audio_path": "audio/clean-00000-of-00411.parquet",
           "audio_sha256": "0" * 64, "duration_s": 12.5, "filename_label": labels[0],
           "assigned_label": labels[0], "class_label": labels[-1], "label_source": "both",
           "id_floor_ok": True, "overlaps_filename": False, "blocking_reason": None,
           "stratum": "both", "local_audio": "ab_00000000.wav"}
    path = tmp_path / "worklist.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def quran_db(tmp_path):
    path = tmp_path / "quran.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE ayahs (surah INTEGER, ayah INTEGER, text TEXT)")
        conn.executemany("INSERT INTO ayahs VALUES (?, ?, ?)",
                         [(2, 1, "الم"), (2, 2, "ذلك الكتاب")])
    return path


def test_a_label_without_text_refuses_to_serve(tmp_path, quran_db):
    worklist = _worklist(tmp_path, "2:1", "3:7")
    with pytest.raises(ValueError, match=r"no Uthmani text for 1 of 2 ayah labels \(3:7\)"):
        Session.load(worklist, tmp_path, tmp_path / "verdicts.jsonl", quran_db)


def test_a_missing_database_refuses_to_serve(tmp_path):
    worklist = _worklist(tmp_path, "2:1", "2:2")
    with pytest.raises(ValueError, match="no Uthmani text for 2 of 2"):
        Session.load(worklist, tmp_path, tmp_path / "verdicts.jsonl", tmp_path / "absent.db")


def test_the_loaded_session_serves_both_candidate_texts(tmp_path, quran_db):
    worklist = _worklist(tmp_path, "2:1", "2:2")
    state = Session.load(worklist, tmp_path, tmp_path / "verdicts.jsonl", quran_db)
    items = state.payload()["items"]
    assert sorted(candidate["text"] for candidate in items[0]["candidates"]) == sorted(
        ["الم", "ذلك الكتاب"])
    assert items[0]["audio_url"] == "/audio/ab_00000000.wav"
