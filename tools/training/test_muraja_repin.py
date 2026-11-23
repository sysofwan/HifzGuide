"""The Muraja re-pin (#306): the commit as a value the caller passes, strict as a scoring of its
own, the per-word grade trace, and the phoneme-group rows ``WordError.groupIndex`` names at
Muraja 028bc51. Torch-free and harness-free; the engine itself is exercised in
``test_muraja_repin_harness.py``."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from muraja_harness.make_parity_fixtures import FIXTURES_PATH
from training import muraja_policy as mp
from training import word_flags as w
from training.muraja_policy import (
    ALLOWANCES_ON,
    HARNESS_PROTOCOL,
    MURAJA_COMMIT,
    SCORINGS,
    SHIPPED_DEFAULT,
    Harness,
    SiteGrade,
)
from training.test_muraja_policy import AYAH, W3_LETTERS, fake_binary, text  # noqa: F401  (text: fixture)

PIN = "99c326f3fb7c6c53ff525568976c7e2442ccde3c"
CURRENT = "028bc513eeb233caa7240a5c37c423da59f3d9d7"
SCORING_SWIFT = Path(mp.BUILD_SCRIPT).parent / "scoring.swift"
FIXTURES = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


# --- the commit is a value, never a switch ---------------------------------------------------------
def test_the_frozen_pin_stays_and_the_audits_commit_is_028bc51():
    assert MURAJA_COMMIT == PIN
    assert mp.CURRENT_COMMIT == CURRENT


def test_the_request_protocol_is_unchanged_so_the_frozen_binary_is_still_spoken_to():
    """No request or response shape changes (strict is a mode the binary already maps; the word
    trace is built here from what it already prints), so the 99c326f binary the frozen readouts
    pin by SHA-256 stays valid."""
    assert HARNESS_PROTOCOL == 3
    assert re.search(r"^let harnessProtocol = 3$", SCORING_SWIFT.read_text(encoding="utf-8"), re.M)


def test_a_scoring_names_the_commit_it_is_asked_for():
    assert SHIPPED_DEFAULT.as_dict()["muraja_commit"] == PIN
    assert SHIPPED_DEFAULT.as_dict(commit=CURRENT)["muraja_commit"] == CURRENT
    assert mp.STRICT.as_dict(commit=CURRENT)["muraja_commit"] == CURRENT


def test_a_harness_is_required_for_the_commit_asked_for(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mp, "HARNESS_ROOT", tmp_path)
    unbuilt = "0" * 40
    with pytest.raises(SystemExit, match=unbuilt):
        mp.require_harness(unbuilt)


@pytest.mark.parametrize("commit", [PIN, CURRENT])
def test_a_build_record_naming_another_commit_is_refused(tmp_path: Path, commit: str):
    other = CURRENT if commit == PIN else PIN
    root = tmp_path / commit
    root.mkdir()
    build = {"muraja_commit": commit, "compiler": "swift", "harness": "h",
             "protocol_version": HARNESS_PROTOCOL, "scoring_keys": ["name", "mode"]}
    for name in ("quran.db", "layout.db"):
        (tmp_path / name).write_text("")
    record = {"quran_db": str(tmp_path / "quran.db"), "layout_db": str(tmp_path / "layout.db"),
              "muraja_commit": commit, "compiler": "swift"}
    honest = fake_binary(tmp_path / "honest", build)
    liar = fake_binary(tmp_path / "liar", build | {"muraja_commit": other})
    (root / "harness.json").write_text(json.dumps(record | {"binary": str(honest)}))
    assert Harness.locate(tmp_path, commit=commit).build == build
    # the binary was built for another commit than the manifest says
    (root / "harness.json").write_text(json.dumps(record | {"binary": str(liar)}))
    assert Harness.locate(tmp_path, commit=commit) is None
    # asking for the other commit finds no harness at all
    (root / "harness.json").write_text(json.dumps(record | {"binary": str(honest)}))
    assert Harness.locate(tmp_path, commit=other) is None


def test_a_run_refuses_a_binary_whose_build_record_names_another_commit(tmp_path: Path):
    build = {"muraja_commit": CURRENT, "compiler": "swift", "harness": "h",
             "protocol_version": HARNESS_PROTOCOL, "scoring_keys": ["name", "mode"]}
    binary = fake_binary(tmp_path / "bin", build | {"muraja_commit": PIN})
    harness = Harness(binary, tmp_path, tmp_path, CURRENT, "swift", build=build)
    with pytest.raises(ValueError, match="build record"):
        harness.run([])


def test_the_build_script_compiles_the_replay_session_where_muraja_has_one_and_edits_nothing_new():
    script = Path(mp.BUILD_SCRIPT).read_text(encoding="utf-8")
    # replay_log.swift at 028bc51 calls makeReplaySession (tools/replay_session.swift); a checkout
    # at 99c326f has no such file, and its build must not change
    assert re.search(r'(-f|-e)\s+"?\$CHECKOUT/tools/replay_session\.swift"?', script)
    # the scratch copy's edits stay the four it has: forMode, isSoftMismatch, @main, the logger
    assert len(re.findall(r"^\s+edit_once \"", script, re.M)) == 4


# --- strict is a scoring of its own ----------------------------------------------------------------
def test_strict_is_muraja_s_strict_mode_with_tashkeel_shown_and_no_flag_overridden():
    strict = mp.STRICT
    assert (strict.name, strict.mode, strict.tashkeel_shown) == ("strict", "strict", True)
    assert strict.soft_pairs_enabled is None and strict.shaddah_suppression is None
    assert strict.soft_pairs_disabled == ()


def test_strict_is_not_one_of_the_frozen_scorings_nor_the_balanced_engine_run():
    assert mp.STRICT not in SCORINGS and mp.STRICT.name not in {s.name for s in SCORINGS}
    assert len(SCORINGS) == 5
    assert mp.STRICT.engine_name != SHIPPED_DEFAULT.engine_name
    runs = mp.engine_runs([mp.STRICT, SHIPPED_DEFAULT, ALLOWANCES_ON])
    assert {r["mode"] for r in runs} == {"strict", "balanced"} and len(runs) == 2
    strict_run = next(r for r in runs if r["mode"] == "strict")
    assert strict_run["name"] == mp.STRICT.engine_name
    assert strict_run["soft_pairs_enabled"] is None and strict_run["shaddah_suppression"] is None


def test_the_kept_quality_is_shown_in_strict_as_the_engine_keeps_it():
    assert mp.displayed("tashkeelError", mp.STRICT) == "tashkeelError"
    assert mp.displayed("minor", mp.STRICT) == "minor"


# --- the word trace ----------------------------------------------------------------------------------
def tashkeel(group: int) -> dict:
    return {"kind": {"type": "tashkeel", "expected": "fatha", "heard": None}, "group_index": group}


def status(word: int, quality: str, score: float, errors: list, withheld: str | None = None) -> dict:
    return {"word": word, "quality": quality, "score": score, "withheld": withheld, "errors": errors}


#: A hop and an overlap longer than the 30 Characters a device log keeps (Fable A1).
LONG_HOP = "ل" * 12 + "َ" * 12
LONG_OVERLAP = "تَمَسسَنَ" * 6
CHECKS = [
    {"hop": "", "overlap": "تَمَ", "flush": False},
    {"hop": LONG_HOP, "overlap": LONG_OVERLAP, "flush": False},
    {"hop": "", "overlap": "تَمَسسَنَ", "flush": True},
    {"hop": "نَ", "overlap": "", "flush": False, "stop": True},
]
STRICT_GRADE = status(3, "minor", 0.7, [tashkeel(2)])
BALANCED_GRADE = status(3, "correct", 0.9, [])


def harness_results() -> tuple[dict, list[dict]]:
    """A request and the results a harness prints for it: strict keeps word 3 as ``minor``
    (set by checks 1 and 2), balanced as ``correct`` (set by check 2)."""
    scorings = [mp.STRICT, SHIPPED_DEFAULT]
    request = {"item": "arm#1", "surah": 2, "ayah": 80, "start_word": 2, "report_words": [3, 5],
               "checks": CHECKS, "scorings": mp.engine_runs(scorings)}

    def result(scoring: mp.Scoring, kept: dict, per_check: list[list[dict]]) -> dict:
        return {"item": "arm#1", "scoring": scoring.engine_name, "page": 10, "parameters": {},
                "checks": [{"position": "2:80:2", "statuses": s} for s in per_check],
                "final": [kept], "letters": [W3_LETTERS]}

    strict = result(mp.STRICT, STRICT_GRADE, [
        [], [status(3, "wrong", 0.2, [tashkeel(1)])], [STRICT_GRADE], [STRICT_GRADE]])
    balanced = result(SHIPPED_DEFAULT, BALANCED_GRADE, [
        [status(3, "pending", 0.3, [])], [status(3, "minor", 0.5, [])], [BALANCED_GRADE], []])
    return request, [strict, balanced]


def rows_of() -> list[dict]:
    request, results = harness_results()
    return mp.word_traces(request, results, [mp.STRICT, SHIPPED_DEFAULT, ALLOWANCES_ON])


def test_a_trace_has_one_row_per_graded_word_and_scoring_in_the_scorings_order():
    rows = rows_of()
    assert [(r["item"], r["scoring"], r["word"]) for r in rows] == [
        ("arm#1", "strict", 3), ("arm#1", "shipped_default", 3), ("arm#1", "allowances_on", 3)]
    # word 5 was reported but never graded: no row
    assert not [r for r in rows if r["word"] == 5]


def test_a_row_carries_the_kept_grade_of_its_scorings_own_engine_run():
    strict, balanced, _ = rows_of()
    assert (strict["quality"], strict["score"], strict["errors"]) == ("minor", 0.7, [tashkeel(2)])
    assert (balanced["quality"], balanced["score"], balanced["errors"]) == ("correct", 0.9, [])
    assert strict["withheld"] is None


def test_a_row_names_the_letters_the_app_marks_for_the_kept_errors():
    strict, balanced, _ = rows_of()
    assert strict["marked"] == [2]  # the group-2 error is drawn on the third letter of تَمَسَّنَ
    assert balanced["marked"] == []
    request, results = harness_results()
    results[0]["final"] = [status(3, "minor", 0.7, [tashkeel(0), tashkeel(3), tashkeel(9)])]
    (row, *_) = mp.word_traces(request, results, [mp.STRICT])
    assert row["marked"] == [0, 3]  # a group past the printed word marks nothing


def test_a_row_carries_every_checks_own_status_of_the_word_and_its_full_text():
    strict, balanced, _ = rows_of()
    for row, per_check in ((strict, [None, ("wrong", 0.2), ("minor", 0.7), ("minor", 0.7)]),
                           (balanced, [("pending", 0.3), ("minor", 0.5), ("correct", 0.9), None])):
        assert len(row["checks"]) == len(CHECKS)
        for check, request_check, own in zip(row["checks"], CHECKS, per_check):
            assert (check["hop"], check["overlap"], check["flush"]) == (
                request_check["hop"], request_check["overlap"], request_check["flush"])
            if own is None:
                assert check["status"] is None
            else:
                assert (check["status"]["quality"], check["status"]["score"]) == own
    # the text of a check is the request's, never clipped to a log's 30 Characters
    assert strict["checks"][1]["hop"] == LONG_HOP and len(strict["checks"][1]["overlap"]) > 30
    assert strict["checks"][1]["overlap"] == LONG_OVERLAP
    assert strict["checks"][1]["status"]["errors"] == [tashkeel(1)]
    assert strict["checks"][3]["status"] == STRICT_GRADE


def test_a_row_names_the_checks_that_set_the_kept_grade():
    strict, balanced, _ = rows_of()
    assert strict["setting_checks"] == [2, 3] and balanced["setting_checks"] == [2]
    _, results = harness_results()
    assert strict["setting_checks"] == list(mp.setting_checks(results[0]["checks"], 3, STRICT_GRADE))
    assert balanced["setting_checks"] == list(mp.setting_checks(results[1]["checks"], 3, BALANCED_GRADE))


def test_the_display_toggle_does_not_change_the_row_of_a_shared_engine_run():
    _, balanced, allowances_on = rows_of()
    assert {k: v for k, v in balanced.items() if k != "scoring"} == {k: v for k, v in allowances_on.items() if k != "scoring"}


def test_a_trace_is_plain_json():
    assert json.loads(json.dumps(rows_of(), ensure_ascii=False)) == rows_of()


# --- setting_checks agrees with the kept grade on the recorded fixtures --------------------------------
def fixture_results():
    return [(case["case"], result) for case in FIXTURES["cases"] for result in case["expected"]]


def test_the_setting_checks_of_every_recorded_kept_grade_hold_that_grade():
    seen = 0
    for _, result in fixture_results():
        for kept in result["final"]:
            for index in mp.setting_checks(result["checks"], kept["word"], kept):
                own = next(s for s in result["checks"][index]["statuses"] if s["word"] == kept["word"])
                assert (own.get("withheld") or own["quality"], own["score"], own["errors"]) == (
                    kept["quality"], kept["score"], kept["errors"])
                seen += 1
    assert seen > 30


def test_every_recorded_settled_grade_was_set_by_some_check_and_a_skip_marking_by_none():
    for case, result in fixture_results():
        for kept in result["final"]:
            setting = mp.setting_checks(result["checks"], kept["word"], kept)
            if kept["quality"] == "skipped":
                assert setting == (), (case, result["scoring"])  # the engine's skip marking, no check's grade
            elif kept["quality"] != "pending":
                assert setting, (case, result["scoring"], kept["word"])


# --- MurajaText.word_rows and the row-indexed flags (Muraja e0b8b18) --------------------------------------
def test_a_words_phoneme_group_rows_are_the_texts_of_its_phoneme_groups_rows(text):  # noqa: F811
    assert text.word_rows(2, 80, 3) == ["تَ", "مَ", "سسَ", "نَ"]
    assert text.word_rows(2, 80, 1) == ["وَ", "قَ", "اا", "لُ", "ۥۥ"]
    assert "".join(text.word_rows(2, 80, 2)) == AYAH.split(" ")[1]


#: 2:2:7 لِلْمُتَّقِينَ as Muraja's quran.db holds it, and its phoneme_groups rows: the first two ل
#: are one geminate row (للِ), the third a row of its own; the scorer's group numbering counts
#: the geminate as two groups, the rows as one.
WORD = "للِلمُتتَقِۦۦن"
ROWS = ["للِ", "ل", "مُ", "تتَ", "قِ", "ۦۦ", "ن"]


def deleted(letter: str, group: int) -> dict:
    return {"kind": {"type": "deletion", "expected": letter}, "group_index": group}


def consonants(error: dict, rows=None) -> tuple:
    grade = SiteGrade("flagged", "minor", (error,))
    return (w.flag(grade, WORD) if rows is None else w.flag(grade, WORD, rows=rows)).consonants


def test_at_99c326f_an_error_names_the_scorers_group_and_the_numbering_is_unchanged():
    assert consonants(deleted("ل", 1)) == ((w.SHADDAH_CAUSE, "ل→∅"),)


def test_at_028bc51_an_error_names_a_row_so_the_geminate_row_is_the_doubled_letter():
    assert consonants(deleted("ل", 0), ROWS) == ((w.SHADDAH_CAUSE, "ل→∅"),)
    assert consonants(deleted("ت", 3), ROWS) == ((w.SHADDAH_CAUSE, "ت→∅"),)
    # row 1 is the third ل on its own, not the second half of the geminate
    assert consonants(deleted("ل", 1), ROWS) == ((w.OTHER_CONSONANT, "ل→∅"),)
    # the madd row after قِ is row 5, a row the scorer does not number
    assert consonants(deleted("ن", 6), ROWS) == ((w.OTHER_CONSONANT, "ن→∅"),)


def test_a_mismatch_is_read_the_same_in_either_numbering():
    error = {"kind": {"type": "mismatch", "expected": "ح", "heard": "ه"}, "group_index": 2}
    assert consonants(error) == consonants(error, ROWS) == ((w.SOFT_PAIR, "ح→ه"),)
