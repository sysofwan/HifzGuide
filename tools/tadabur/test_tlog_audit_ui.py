"""The S6 listening page's session: load guards, class scoring, verdict persistence.

The page grades blind: the owner reads the candidate ayat, never the label. Two labels that are
the same recitation (one text class) are one candidate and score the same however they are cited,
and every save is serialized and replaces the answer the client saw, so two quick clicks cannot
lose one another and a stale tab cannot overwrite a newer answer unseen.
"""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from . import tlog_ident as ident
from . import tlog_audit as run
from .tlog_audit_ui import Session, StaleAnswer, stored_answer

#: 2:1 and 2:2 are the same recitation (one text class); 3:1 is a different one.
DATABASE = ident.ReferenceDatabase.build([
    ident.AyahReference(2, 1, "بجح", (0, 3)),
    ident.AyahReference(2, 2, "بجح", (0, 3)),
    ident.AyahReference(3, 1, "دذر", (0, 3)),
])


def _row(row: int = 7, *, filename_label="2:1", class_label="2:2", stratum="both",
         assigned_label=None, **overrides) -> dict:
    return {"population": "look", "split": "clean", "shard_file": "clean-00000-of-00411",
            "row": row, "role": "look", "audio_path": "audio/clean-00000-of-00411.parquet",
            "audio_sha256": "0" * 64, "duration_s": 12.5, "filename_label": filename_label,
            "assigned_label": filename_label if assigned_label is None else assigned_label,
            "class_label": class_label, "label_source": "both", "id_floor_ok": True,
            "overlaps_filename": False, "blocking_reason": None, "stratum": stratum,
            "local_audio": f"{row:016x}_00000000.wav", **overrides}


def _worklist(tmp_path, *rows):
    path = tmp_path / "worklist.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")
    return path


@pytest.fixture
def quran_db(tmp_path):
    path = tmp_path / "quran.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE ayahs (surah INTEGER, ayah INTEGER, text TEXT)")
        conn.executemany("INSERT INTO ayahs VALUES (?, ?, ?)",
                         [(2, 1, "الم"), (2, 2, "الم"), (3, 1, "ذلك الكتاب")])
    return path


def _session(tmp_path, quran_db, *rows, verdicts=None):
    rows = rows or (_row(),)
    return Session.load(_worklist(tmp_path, *rows), tmp_path,
                        tmp_path / "verdicts.jsonl" if verdicts is None else verdicts,
                        quran_db, database=DATABASE)


def test_a_label_without_text_refuses_to_serve(tmp_path, quran_db):
    rows = (_row(class_label="3:1"),)
    with pytest.raises(ValueError, match="no Uthmani text for 2 of 2 ayah labels"):
        Session.load(_worklist(tmp_path, *rows), tmp_path, tmp_path / "verdicts.jsonl",
                     tmp_path / "empty.db", database=DATABASE)


def test_a_label_off_the_reference_database_refuses_to_serve(tmp_path, quran_db):
    rows = (_row(class_label="4:1"),)
    with pytest.raises(ValueError, match="4:1 is not an ayah of the reference database"):
        Session.load(_worklist(tmp_path, *rows), tmp_path, tmp_path / "verdicts.jsonl",
                     quran_db, database=DATABASE)


def test_the_loaded_session_serves_one_candidate_per_text_class(tmp_path, quran_db):
    state = _session(tmp_path, quran_db)
    items = state.payload()["items"]
    candidates = items[0]["candidates"]
    assert [candidate["ayah"] for candidate in candidates] == ["2:1"]
    assert candidates[0]["text"] == "الم"
    assert items[0]["audio_url"] == "/audio/0000000000000007_00000000.wav"


def test_a_cited_identical_recitation_scores_the_label_under_test(tmp_path, quran_db):
    state = _session(tmp_path, quran_db)
    key = run._item_key(_row())
    saved = state.record(key, "another", "2:2", "", None)
    assert saved["verdict"] == run.RIGHT
    assert (saved["heard_ayah"], saved["ayah"]) == ("2:2", "2:1")
    assert state.record(key, "another", "3:1", "", stored_answer(saved))["verdict"] == run.OTHER


def test_a_relabelled_click_scores_against_the_assigned_label(tmp_path, quran_db):
    row = _row(8, filename_label="2:1", assigned_label="3:1", class_label="3:1",
               stratum="relabelled", label_source="identified")
    state = _session(tmp_path, quran_db, row)
    key = run._item_key(row)
    saved = state.record(key, "candidate", "3:1", "", None)
    assert saved["verdict"] == run.RIGHT
    assert state.record(key, "candidate", "2:1", "", stored_answer(saved))["verdict"] == run.OTHER


def test_an_unknown_reference_is_refused(tmp_path, quran_db):
    state = _session(tmp_path, quran_db)
    key = run._item_key(_row())
    for bad in ("0:1", "115:1", "2:3", "2:0", "two:1", "2:1:3"):
        with pytest.raises(ValueError, match="the mushaf has"):
            state.record(key, "another", bad, "", None)


def test_the_note_the_owner_types_is_persisted_with_the_verdict(tmp_path, quran_db):
    state = _session(tmp_path, quran_db)
    state.record(run._item_key(_row()), "candidate", "2:1", "repeats the first ayah twice", None)
    stored = json.loads((tmp_path / "verdicts.jsonl").read_text(encoding="utf-8"))
    assert stored["note"] == "repeats the first ayah twice"
    assert set(stored) >= {"split", "shard_file", "row", "audio_path", "audio_sha256",
                           "verdict", "heard_ayah", "ayah", "note"}


def test_concurrent_saves_of_different_items_both_survive(tmp_path, quran_db):
    rows = [_row(row) for row in (11, 12)]
    state = _session(tmp_path, quran_db, *rows)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def save(row):
        try:
            barrier.wait()
            state.record(run._item_key(row), "candidate", "2:1", "", None)
        except BaseException as error:  # noqa: BLE001 - collected for the assertion
            errors.append(error)

    threads = [threading.Thread(target=save, args=(row,)) for row in rows]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    persisted = [json.loads(line) for line in
                 (tmp_path / "verdicts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["row"] for row in persisted} == {11, 12}
    assert len(state.verdicts) == 2


def test_a_stale_answer_cannot_replace_one_the_client_did_not_see(tmp_path, quran_db):
    state = _session(tmp_path, quran_db)
    key = run._item_key(_row())
    first = state.record(key, "candidate", "2:1", "", None)
    with pytest.raises(StaleAnswer):
        state.record(key, "candidate", "3:1", "", None)  # the tab saw no answer
    with pytest.raises(StaleAnswer):
        state.record(key, "silence", None, "", [run.OTHER, "3:1"])  # an unseen answer
    assert state.verdicts[key]["verdict"] == run.RIGHT
    replaced = state.record(key, "candidate", "3:1", "", stored_answer(first))
    assert replaced["verdict"] == run.OTHER
    assert json.loads((tmp_path / "verdicts.jsonl").read_text(encoding="utf-8"))["ayah"] == "3:1"
