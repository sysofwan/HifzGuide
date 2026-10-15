"""The owner's ``h448`` sessions from bug-report exports (#153): splitting an export, linking
truncated tails, the device-parity checks, what is read from a session, the owner's
annotations, and the committed report and doc, which must carry no log text. Torch-free; the
replay itself runs only where the Muraja log replay is built (``muraja_harness/build.sh``)."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from training import device_exports as x
from training import device_logs as dl
from training.device_exports_doc import render
from training.muraja_policy import LOG_REPLAY_BINARY, MURAJA_COMMIT, SHIPPED_DEFAULT, Harness

REPORT = json.loads(x.REPORT_PATH.read_text(encoding="utf-8"))
TARGETS = [s for s in REPORT["sessions"] if s["target"]]


def line(stamp: str, category: str, message: str) -> str:
    return f"[{stamp}] [INFO] [{category}] {message}"


def export(*body: str, mode: str = "Strict") -> str:
    return "\n".join(["Bug Report", "Device: phone", "Settings:", f"  Pronunciation: {mode}", "",
                      "[… truncated …]", *body]) + "\n"


def app_run(started: str, *body: str) -> list[str]:
    return ["=== Current Session ===", "=== Hifz Guide Session Log ===", f"Started: {started}",
            "Device: phone", "App Version: 1.0.27 (1)", "===========================", "",
            line("00:00:00.000", "MuaalemInference", "[MuaalemStudent_h448_6BIT] ✅ Loaded (ANE)"), *body]


SESSION_A = [line(f"00:00:0{i}.000", "FollowAlongManager", f"Update: type=preview, hop=, overlap=a{i}, didFlush=false")
             for i in range(1, 9)]
SESSION_A = [line("00:00:00.500", "ScoreCollector", "Tilawah session pending on page 4"), *SESSION_A]
SESSION_B = [line("00:01:00.000", "ScoreCollector", "Tilawah session pending on page 9"),
             line("00:01:01.000", "FollowAlongManager", "Update: type=preview, hop=, overlap=b, didFlush=false")]


# --- an export, split ---------------------------------------------------------------------------------
def test_an_export_splits_at_each_session_and_a_truncated_tail_is_linked_to_its_whole_session():
    whole = x.split_sessions("report 1.txt", export(*app_run("t0", *SESSION_A)))
    assert [(s.name, s.page, s.kind, s.model, s.app_version) for s in whole] == [
        ("1#1", 4, "Tilawah", "MuaalemStudent_h448_6BIT", "1.0.27")]
    # the later export keeps only the tail of the same app run: no header, no model line
    tail = ["update=… cut mid-line", *SESSION_A[2:], *SESSION_B]
    later = x.split_sessions("report 2.txt", export(*tail))
    assert [(s.name, s.page, s.model) for s in later] == [("2#1", None, None), ("2#2", 9, None)]
    linked = {s.name: s for s in x.link([*whole, *later])}
    assert linked["2#1"].copy_of == "1#1" and not linked["2#1"].whole
    assert (linked["2#2"].model, linked["2#2"].app_version) == ("MuaalemStudent_h448_6BIT", "1.0.27")
    assert linked["2#2"].run_basis == "the app run of 1#1 (2#1 is its tail)"


def test_a_stretch_with_no_update_before_a_runs_first_session_is_no_session():
    sessions = x.split_sessions("r 3.txt", export(*app_run("t0", line("0", "X", "Follow-along ready."), *SESSION_B)))
    assert [s.page for s in sessions] == [9]


# --- what the device logged -----------------------------------------------------------------------------
GRADE = "Grade end word 2:19:{}: cov=1.00 acc=1.00 score=1.00 mm=0 gap=0 last=false → {}"


def test_parity_counts_the_decisions_a_replay_reproduces_and_names_each_word_that_differs():
    device = [GRADE.format(1, "correct"), GRADE.format(2, "minor"), GRADE.format(3, "correct")]
    assert x.parity(device, device) | {"differing_words": []} == {
        "device": 3, "replay": 3, "matched": 3, "exact": True, "words": 3, "differing_words": []}
    other = x.parity(device, [GRADE.format(1, "correct"), GRADE.format(2, "correct"), GRADE.format(3, "correct")])
    assert (other["matched"], other["exact"], other["differing_words"]) == (2, False, ["2:19:2"])
    text = "\n".join(line("0", "QuranFollowAlong", g) for g in device)
    assert x.grade_lines(text) == device
    assert x.decision_counts(text, "2:19:2") == {"minor": 1}


def test_a_committed_grade_may_differ_from_the_kept_one_only_where_the_device_regraded_the_word():
    text = "\n".join([
        line("1", "ScoreCollector", "Commit 2:61:30 quality=tashkeelError score=1.000000 errors=1 [g2:tash]"),
        line("2", "ScoreCollector", "Commit 2:61:31 quality=minor score=0.5 errors=1 [g0:ل→ء]"),
        line("3", "FollowAlong", "Upgraded: 2:61:30 1.00→0.75"),
        line("4", "ScoreCollector", "Commit 2:61:32 quality=correct score=1.0 errors=0 []"),
    ])
    replay = dl.Replay("2:61:33", {"2:61:30": {"quality": "minor"}, "2:61:31": {"quality": "correct"},
                                   "2:61:32": {"quality": "correct"}})
    check = x.commit_parity(text, replay)
    assert (check["committed"], check["equal"]) == (3, 1)
    assert [r["word"] for r in check["regraded_after_commit"]] == ["2:61:30"]
    assert [r["word"] for r in check["unexplained"]] == ["2:61:31"]
    assert x.committed_flags(text) == [
        {"word": "2:61:30", "quality": "tashkeelError", "errors": ["tashkeel"]},
        {"word": "2:61:31", "quality": "minor", "errors": ["ل→ء"]}]


def test_an_added_phrase_is_the_detectors_classification_with_its_history():
    text = "\n".join([
        line("1", "FollowAlong", "Shadow speech: #2 unresolved cycle=154 2:253:21→2:253:22 said=7 «x»"),
        line("2", "FollowAlong", "Shadow speech: #2 added cycle=165 2:253:21→2:253:22 said=7 «x»"),
        line("3", "FollowAlong", "Shadow speech: #3 noise(short) cycle=170 2:253:26→2:253:27 said=1 «y»"),
        line("4", "ScoreCollector", "Added words saved 2:253:21→2:253:22"),
    ])
    assert x.added_phrases(text) == [x.Added(2, ("unresolved", "added"), "2:253:21", "2:253:22", True)]


# --- the owner's annotations ------------------------------------------------------------------------------
class Text:
    def phonemes(self, surah: int, ayah: int) -> str:
        return " ".join(["w"] * 24)


def test_the_annotations_resolve_every_statement_and_explanation():
    notes = x.load_annotations(Text())
    assert set(notes.exports) == {e["name"] for e in REPORT["exports"]}
    assert [(m.first, m.last, m.kind, m.basis) for m in notes.mistakes["71#2"]] == [
        ("2:256:1", "2:256:24", x.MISTAKE_AYAH, x.OWNER)]
    assert [(m.first, m.kind) for m in notes.mistakes["70#2"]] == [("2:60:11", x.MISTAKE)]
    assert all(v["explained_by"] in x.EXPLAINED_BY and v["basis"] == x.OWNER
               for by_word in notes.model_errors.values() for v in by_word.values())
    assert {s for s in (*notes.mistakes, *notes.model_errors, *notes.added)} <= {s["session"] for s in TARGETS}


# --- the committed report and doc ----------------------------------------------------------------------------
def test_the_report_and_doc_carry_no_log_text_device_info_or_settings():
    """Positions, counts, causes and the owner's words only: no Arabic but a one-letter label or
    a pair's or substitution's, no log timestamp, no device or settings line."""
    label = re.compile("[\u0600-\u06ff][↔→][\u0600-\u06ff∅]")
    arabic_run = re.compile("[\u0600-\u06ff]{2,}")
    for text in (x.REPORT_PATH.read_text(encoding="utf-8"), x.DOC_PATH.read_text(encoding="utf-8"),
                 x.SESSIONS_PATH.read_text(encoding="utf-8")):
        assert not arabic_run.search(label.sub("", text))
        assert not re.search(r"\[\d\d:\d\d:\d\d\.\d{3}\]", text)
        assert not re.search(r"iPhone\d|iOS \d|Device: |App Language|Pronunciation: |Started: ", text)


def test_each_session_reproduces_the_device_in_exactly_one_mode_at_both_commits():
    assert {(s["page"], s["model"], s["app_version"]) for s in TARGETS} == {
        (4, "MuaalemStudent_h448_6BIT", "1.0.27"), (9, "MuaalemStudent_h448_6BIT", "1.0.27"),
        (42, "MuaalemStudent_h448_6BIT", "1.0.27")}
    for s in TARGETS:
        for run in ("pin/lengthened", "main/lengthened"):
            assert [m for m, p in s["parity"][run].items() if p["exact"]] == [s["mode"]]
        assert not s["commit_parity"]["unexplained"]
    assert all(s["copy_of"] for s in REPORT["sessions"] if not s["target"])


def test_every_word_the_phone_showed_as_not_correct_has_the_owners_verdict():
    shown = [c for s in TARGETS for c in s["committed_flags"]]
    assert shown and all(c["verdict"]["status"] in (x.MISTAKE, x.IN_MISTAKE_AYAH, x.MODEL_ERROR) for c in shown)
    kept = [r for r in REPORT["words"] if r["scoring"] == x.AS_RECORDED]
    assert all(r["verdict"]["status"] != x.ASSUMED for r in kept)


def test_the_owners_mistakes_are_out_of_the_false_flag_rates_and_reported_apart():
    catch = REPORT["catch"][x.AS_RECORDED]
    every, rate = (REPORT["rates"][x.AS_RECORDED][v]["false_flags"] for v in (dl.ALL_WORDS, dl.UNEXPLAINED))
    ayah_words = sum(int(m["last"].rsplit(":", 1)[1]) for s in TARGETS for m in s["mistakes"] if m["kind"] == x.MISTAKE_AYAH)
    assert every["words"] - rate["words"] == catch["words"]["graded"] + ayah_words
    assert every["flags"] - rate["flags"] == catch["words"]["flagged"] + catch["ayat"]["flagged"]
    rows = [r for r in REPORT["words"] if r["scoring"] == x.AS_RECORDED]
    assert {w["word"] for w in catch["words"]["flagged_words"]} == {
        r["word"] for r in rows if r["verdict"]["status"] == x.MISTAKE}
    assert {w["word"] for w in catch["ayat"]["flagged_words"]} == {
        r["word"] for r in rows if r["verdict"]["status"] == x.IN_MISTAKE_AYAH}


def test_an_ayah_the_owner_says_holds_mistakes_names_no_word_as_one():
    """2:256's words are left out of the rates but none is the owner's mistake, and the ayah has
    no catch-rate denominator (#154 review)."""
    assert all(r["verdict"]["status"] != x.MISTAKE for r in REPORT["words"] if r["word"].startswith("2:256:"))
    assert set(REPORT["catch"][x.AS_RECORDED]["ayat"]) == {"flagged", "causes", "flagged_words"}


def test_the_exports_are_pinned_by_name_and_sha256():
    pinned = {e["name"]: e["sha256"] for e in json.loads(x.SESSIONS_PATH.read_text(encoding="utf-8"))["exports"]}
    assert {e["name"]: e["sha256"] for e in REPORT["exports"]} == pinned
    assert (REPORT["muraja"]["pin"], REPORT["muraja"]["main"]) == (MURAJA_COMMIT, dl.MURAJA_MAIN)


def test_the_doc_is_rendered_from_the_report():
    assert x.DOC_PATH.read_text(encoding="utf-8") == render(REPORT)


# --- the replay itself ------------------------------------------------------------------------------------
LOG_REPLAY = Harness.locate(binary=LOG_REPLAY_BINARY)


@pytest.mark.skipif(LOG_REPLAY is None, reason="the Muraja log replay is not built here")
def test_the_log_replay_runs_each_mode_and_keeps_the_scorers_grade_lines():
    log = "surah2_page4_strict_shaddah"
    path = dl.log_dir(MURAJA_COMMIT) / f"{log}.log"
    scorings = [replace(SHIPPED_DEFAULT, name=m, mode=m) for m in ("strict", "balanced")]
    _, replays = dl.replay_logs(LOG_REPLAY, {log: path}, scorings)
    lines = {s.mode: replays[log][s.engine_name].grade_lines for s in scorings}
    assert all(lines.values()) and all(g.startswith("Grade end word ") for g in lines["strict"])
    device = x.grade_lines(path.read_text(encoding="utf-8"))
    assert {m: x.parity(device, g)["matched"] for m, g in lines.items()}["strict"] > 0
