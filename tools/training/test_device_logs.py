"""The device-log false-flag report (#151): the inventory, what is read from a log, the replay's
parity check, attribution and rates, and the committed report and doc. Torch-free; the replay
itself runs only where the Muraja log replay is built (``muraja_harness/build.sh``)."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from training import device_logs as d
from training.device_logs_doc import render
from training.muraja_policy import (
    ALLOWANCES_ON,
    LOG_REPLAY_BINARY,
    MURAJA_COMMIT,
    SCORINGS,
    SHIPPED_DEFAULT,
    Harness,
)
from training.word_flags import OTHER_CONSONANT, SKIPPED, SOFT_PAIR, TASHKEEL

FATHA = "َ"
REPORT = json.loads(d.REPORT_PATH.read_text(encoding="utf-8"))


def update(kind: str, hop: str, overlap: str, flush: bool = False) -> str:
    return (f"[10:00:00.000] [INFO] [FollowAlongManager] Update: type={kind}, hop={hop}, "
            f"overlap={overlap}, didFlush={'true' if flush else 'false'}")


# --- the inventory ------------------------------------------------------------------------------------
def test_the_committed_inventory_is_well_formed():
    entries = d.load_inventory()
    assert len({e.log for e in entries}) == len(entries) == 66
    assert not any(e.log.startswith("generated_") for e in entries)
    for e in entries:
        assert e.included == (e.exclusion is None)
        assert e.exclusion is None or e.exclusion_detail
        assert e.included is False or e.model in d.MODELS


def test_an_explained_run_covers_its_ends_and_what_lies_between():
    run = d.Explained("2:7:3", "2:8:2", "skip", "docs", "x")
    assert [run.covers(w) for w in ("2:7:2", "2:7:3", "2:7:30", "2:8:2", "2:8:3")] == [
        False, True, True, True, False]


def test_a_malformed_inventory_is_refused(tmp_path):
    record = {"log": "a", "commit": "c", "committed": "2026-01-01", "model": "h448", "model_basis": None,
              "app_version": None, "purpose": "p", "sources": [], "excluded": None, "explained": [],
              "slips": []}
    path = tmp_path / "inventory.json"
    for bad in ({"model": "other"}, {"model": None}, {"excluded": {"reason": "why not", "detail": "x"}},
                {"explained": [{"first": "2:1:2", "last": "2:1:1", "kind": "skip", "basis": "docs", "source": ""}]},
                {"excluded": {"reason": "documented_mistake", "detail": "x"}},
                {"slips": [{"from": "2:1:2", "to": "2:1:1", "twin": None, "basis": "docs", "source": ""}]}):
        path.write_text(json.dumps([record | bad]))
        with pytest.raises(ValueError):
            d.load_inventory(path)
    path.write_text(json.dumps([record, record]))
    with pytest.raises(ValueError):
        d.load_inventory(path)


# --- what a log says -----------------------------------------------------------------------------------
def test_a_one_second_log_gives_its_start_pages_cut_overlaps_and_inference_time():
    text = "\n".join([
        "[00:00:00.000] [INFO] [FollowAlongManager] Start: position=2:17:0, page=4",
        "[10:00:00.000] [INFO] [Transcription] Preview 40ms — بب",
        update("preview", "", "ب" * 30),
        "[10:00:00.000] [INFO] [Transcription] Window 50ms rms=0.1 — hop: ب, overlap: ب",
        update("window", "ب", "بَ" * 29),
        "[10:00:01.000] [INFO] [FollowAlongManager] Page: 5",
        "[10:00:02.000] [INFO] [FollowAlongManager] Page: 4",
    ])
    facts = d.read_facts(text)
    assert facts == d.LogFacts(start="2:17:0", pages=(4, 5), protocol=d.ONE_SECOND, updates=2,
                               capped_overlaps=1, median_inference_ms=45)


def test_a_legacy_log_is_a_three_second_session_with_no_cut_overlaps():
    text = "\n".join([
        "Preview inference 335ms — قَ",
        "Match: ratio=1.00 pos=12:5:1 expected=12:5 near=true",
        "Window inference 302ms — confirmed: قَاالَ, unconfirmed: يَاا",
    ])
    facts = d.read_facts(text)
    assert (facts.start, facts.pages, facts.protocol, facts.updates, facts.capped_overlaps,
            facts.median_inference_ms) == ("12:5:0", (), d.THREE_SECOND, 2, 0, 318.5)
    old_updates = "Start: position=2:6:0, page=3\nUpdate: type=preview, confirmed=, unconfirmed=ءَعُ, didFlush=false"
    assert d.read_facts(old_updates).protocol == d.THREE_SECOND


def test_a_preview_takes_the_longer_text_its_transcriber_line_logged():
    lines = [
        "[1] [INFO] [Transcription] Preview 40ms — ابتث",
        update("preview", "", "ابت"),
        "[2] [INFO] [Transcription] Preview 40ms — جحخ",
        update("preview", "", "دذر"),   # not a prefix: kept
        update("preview", "", "زس"),    # no transcriber line since the last update: kept
        "[3] [INFO] [Transcription] Window 40ms rms=0.1 — hop: ا, overlap: ابتثجح",
        update("window", "ا", "بت"),     # a window: kept
    ]
    text, n = d.lengthened_previews("\n".join(lines) + "\n")
    assert n == 1
    assert text.splitlines()[1] == update("preview", "", "ابتث")
    assert text.splitlines()[2:] == lines[2:]


# --- replay parity ---------------------------------------------------------------------------------------
def test_a_replay_matches_a_baseline_only_word_for_word_and_at_its_final_position():
    replay = d.Replay("2:1:3", {"2:1:1": {"quality": "correct"}, "2:1:2": {"quality": "minor"}})
    baseline = {"final_position": "2:1:3", "word_qualities": {"2:1:1": "correct", "2:1:2": "minor"}}
    assert d.baseline_differences(replay, baseline) == []
    assert d.baseline_differences(replace(replay, final_position="2:1:2"), baseline) == [
        "final position 2:1:2, baseline 2:1:3"]
    moved = {"final_position": "2:1:3", "word_qualities": {"2:1:1": "correct", "2:1:3": "minor"}}
    assert d.baseline_differences(replay, moved) == [
        "2:1:2: minor, baseline None", "2:1:3: None, baseline minor"]


# --- attribution and rates -----------------------------------------------------------------------------
class Text:
    """Muraja's reference for 2:1, three phoneme words."""

    def phonemes(self, surah: int, ayah: int) -> str:
        assert (surah, ayah) == (2, 1)
        return "ذَل بَب كَتَبَ"


def status(quality: str, *errors: dict) -> dict:
    return {"quality": quality, "score": 1.0, "withheld": None, "errors": list(errors)}


def test_each_scoring_shows_and_attributes_the_grade_the_engine_kept():
    entry = d.LogEntry("log", "c", "2026-01-01", d.TEACHER, None, None, "p", (), None, None,
                       (d.Explained("2:1:3", "2:1:3", "skip", "docs", "s"),), {})
    tashkeel = {"kind": {"type": "tashkeel", "expected": FATHA, "heard": None}, "group_index": 0}
    swap = {"kind": {"type": "mismatch", "expected": "ذ", "heard": "ز"}, "group_index": 0}
    words = {"2:1:1": status("tashkeelError", tashkeel), "2:1:2": status("minor", swap),
             "2:1:3": status("skipped")}
    replays = {s.engine_name: d.Replay("2:1:3", words) for s in SCORINGS}
    grades = {(g.word, g.scoring): g for g in d.word_grades(entry, replays, (ALLOWANCES_ON, SHIPPED_DEFAULT), Text())}
    assert not grades[("2:1:1", "allowances_on")].flagged  # the Tashkeel toggle hides it
    assert grades[("2:1:1", "shipped_default")].causes == (TASHKEEL,)
    assert grades[("2:1:1", "shipped_default")].marks == ("fatha dropped",)
    assert (grades[("2:1:2", "allowances_on")].causes, grades[("2:1:2", "allowances_on")].pairs) == (
        (SOFT_PAIR,), ("ذ↔ز",))
    assert grades[("2:1:3", "allowances_on")].causes == (SKIPPED,)
    assert grades[("2:1:3", "allowances_on")].explained.kind == "skip"
    view = d.views(list(grades.values()), (ALLOWANCES_ON, SHIPPED_DEFAULT))
    assert view["shipped_default"][d.ALL_WORDS]["false_flags"]["flags"] == 3
    unexplained = view["shipped_default"][d.UNEXPLAINED]["false_flags"]
    assert (unexplained["flags"], unexplained["words"], unexplained["per_1000"]) == (2, 2, 1000.0)
    assert view["allowances_on"][d.UNEXPLAINED]["causes"][SOFT_PAIR]["flags"] == 1
    assert view["allowances_on"][d.UNEXPLAINED]["pairs"] == {"ذ↔ز": 1}


def test_the_slip_sensitivity_keeps_every_h448_session_but_its_slip_region():
    def entry(log, exclusion, model=d.H448):
        region = {w: (d.Explained("2:1:2", "2:1:2" if w == "narrow" else "2:1:3", d.SLIP, "docs", "s"),)
                  for w in d.SLIP_WIDTHS}
        return d.LogEntry(log, "c", "2026-01-01", model, None, None, "p", (), exclusion, None, (), region)

    def grade(log, word):
        return d.WordGrade(log, word, "shipped_default", "minor", (OTHER_CONSONANT,), (), (), None)

    entries = [entry("kept", None), entry("slipped", d.DOCUMENTED_MISTAKE),
               entry("synthetic", "synthetic"), entry("teacher", None, d.TEACHER)]
    grades = {e.log: [grade(e.log, w) for w in ("2:1:1", "2:1:2", "2:1:3")] for e in entries}
    view = d.slip_sensitivity(entries, grades, (SHIPPED_DEFAULT,))[d.H448]
    assert view["narrow"]["shipped_default"]["false_flags"]["words"] == 4
    assert view["wide"]["shipped_default"]["false_flags"]["words"] == 2


def test_a_slip_region_runs_by_one_rule_in_two_widths():
    lengths = {(3, 64): 31, (3, 65): 15, (3, 7): 9}
    slip = d.Slip("3:64:11", "3:64:13", "3:7", "docs", "s")
    regions = d.regions(slip, lambda surah, ayah: lengths[(surah, ayah)])
    assert [(r.first, r.last) for r in regions["narrow"]] == [("3:64:11", "3:64:31"), ("3:7:1", "3:7:9")]
    assert [(r.first, r.last) for r in regions["wide"]] == [("3:64:1", "3:65:15"), ("3:7:1", "3:7:9")]
    entry = d.with_regions(d.LogEntry("s", "c", "d", d.H448, None, None, "p", (), d.DOCUMENTED_MISTAKE, "x", (),
                                      slips=(slip,)), lambda surah, ayah: lengths[(surah, ayah)])
    assert entry.in_slip("3:64:20", "narrow") and not entry.in_slip("3:64:10", "narrow")
    assert entry.in_slip("3:64:10", "wide") and entry.in_slip("3:7:3", "narrow")


def test_an_allowance_adds_the_words_it_alone_flags_and_names_them():
    entry = d.LogEntry("s", "c", "2026-01-01", d.TEACHER, None, None, "p", (), None, None,
                       (d.Explained("2:1:3", "2:1:3", "skip", "docs", "x"),), {})

    def grade(word, scoring, flagged, pairs=()):
        return d.WordGrade("s", word, scoring, "minor" if flagged else "correct",
                           (SOFT_PAIR,) if flagged else (), (), pairs, entry.explanation(word))

    off = "not_forgiven:ح↔ه"
    grades = {"s": [grade("2:1:1", "allowances_on", False), grade("2:1:2", "allowances_on", True, ("س↔ش",)),
                    grade("2:1:3", "allowances_on", False),
                    grade("2:1:1", off, True, ("ح↔ه",)), grade("2:1:2", off, True), grade("2:1:3", off, True)]}
    added = d.allowance_effect([entry], grades, ("soft pair ح↔ه", off), None)
    assert (added["flags"], added["words"], added["removed"]) == (1, 2, 0)  # 2:1:3 is explained
    assert added["carriers"] == [["2:1:1", 1]]
    present = d.allowance_effect([entry], grades, ("target pair س↔ش", None), None)
    assert (present["allowance"], present["flags"]) == (False, 1)


def test_a_rate_is_per_thousand_words_and_has_no_interval_from_one_log():
    one = d.rate([True, False, False, False], ["a"] * 4)
    assert (one["flags"], one["words"], one["logs"], one["per_1000"], one["method"]) == (1, 4, 1, 250.0, "none")
    assert d.rate([], [])["per_1000"] is None


def test_covered_ayat_are_runs_of_consecutive_ayat():
    assert d.covered_ayat(["2:17:1", "2:18:3", "2:19:1", "2:21:2", "3:1:1"]) == "2:17-2:19, 2:21, 3:1"


def test_the_simulation_is_read_from_the_word_sample_report():
    sim = d.simulation_rates(json.loads(d.truth_baseline.REPORT_PATH.read_text(encoding="utf-8")))
    cell = sim[d.H448]["shipped_default"]
    assert (cell["words"], cell["flags"]) == (155, 3)
    assert cell["causes"][OTHER_CONSONANT]["flags"] == 2


# --- the committed report and doc ------------------------------------------------------------------------
def test_an_explained_skip_is_the_skipped_ayah_not_the_cursors_jump():
    """ADR-0020's 12:6:1 to 12:7:7 is the cursor's jump; the reader skipped 12:6 only, and 12:7:4
    and 12:7:6, which the engine marks skipped, were recited: they stay false flags."""
    entry = next(e for e in d.load_inventory() if e.log == "surah12_page236_ayah5_v2")
    assert all(entry.explanation(f"12:6:{w}") for w in range(1, 26))
    assert not any(entry.explanation(f"12:7:{w}") for w in range(1, 8))
    flagged = {(r["word"], r["explained"]) for r in REPORT["words"]
               if r["log"] == entry.log and r["scoring"] == "shipped_default"}
    assert {("12:7:4", None), ("12:7:6", None)} <= flagged


def test_every_session_whose_docs_record_a_slip_is_out_of_the_headline():
    for e in d.load_inventory():
        assert (e.exclusion == d.DOCUMENTED_MISTAKE) == bool(e.slips)
    for log in ("surah3_page58_far_reread_pause_residue", "surah3_page52_jump_reread",
                "surah36_page442_cross_ayah_skip"):
        assert next(e for e in d.load_inventory() if e.log == log).exclusion == d.DOCUMENTED_MISTAKE


def test_the_report_covers_the_inventory_and_every_log_reproduces_its_baseline():
    assert [e["log"] for e in REPORT["logs"]] == [e.log for e in d.load_inventory()]
    assert all(e["parity"] == {"pin": True, "main": True} for e in REPORT["logs"])
    assert (REPORT["muraja"]["pin"], REPORT["muraja"]["main"]) == (MURAJA_COMMIT, d.MURAJA_MAIN)


def test_the_report_carries_no_log_text():
    """Positions, counts, causes, commits and docs only: no Arabic but a pair's or a
    substitution's label."""
    arabic = re.compile("[\u0600-\u06ff]")
    pair = re.compile("[\u0600-\u06ff][↔→][\u0600-\u06ff∅]")

    def strings(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    assert not [s for s in strings(REPORT) if arabic.search(pair.sub("", s))]


def test_the_doc_is_rendered_from_the_report():
    assert d.DOC_PATH.read_text(encoding="utf-8") == render(REPORT)


# --- the replay itself ------------------------------------------------------------------------------------
LOG_REPLAY = Harness.locate(binary=LOG_REPLAY_BINARY)


@pytest.mark.skipif(LOG_REPLAY is None, reason="the Muraja log replay is not built here")
def test_the_log_replay_reproduces_committed_baselines():
    logs = ["surah2_page4_strict_shaddah", "surah70_page569_ayah27_v2"]
    directory = d.log_dir(MURAJA_COMMIT)
    _, replays = d.replay_logs(LOG_REPLAY, {log: directory / f"{log}.log" for log in logs}, SCORINGS)
    for log in logs:
        baseline = json.loads((directory / f"{log}.baseline.json").read_text(encoding="utf-8"))
        assert d.baseline_differences(replays[log][SHIPPED_DEFAULT.engine_name], baseline) == []
    # soft-pair forgiveness off shows 2:19:15's ح→ه (the committed report's per-word table)
    soft_off = next(s for s in SCORINGS if s.name == "soft_pair_forgiveness_off")
    assert replays["surah2_page4_strict_shaddah"][soft_off.engine_name].words["2:19:15"]["quality"] == "minor"


@pytest.mark.skipif(LOG_REPLAY is None, reason="the Muraja log replay is not built here")
def test_the_committed_report_word_rows_regenerate_byte_identically():
    """The shared serialisation (``word_grades``, ``word_row``) still writes #151's committed rows
    byte for byte, so later readers of the shared code leave its frozen output alone."""
    from training.muraja_policy import MurajaText

    logs = ["surah2_page4_strict_shaddah", "surah12_page236_ayah5_v2"]
    entries = {e.log: e for e in d.load_inventory()}
    directory = d.log_dir(MURAJA_COMMIT)
    _, replays = d.replay_logs(LOG_REPLAY, {log: directory / f"{log}.log" for log in logs}, SCORINGS)
    text = MurajaText(LOG_REPLAY.quran_db)
    for log in logs:
        rows = [d.word_row(g) for g in d.word_grades(entries[log], replays[log], SCORINGS, text) if g.flagged]
        committed = [r for r in REPORT["words"] if r["log"] == log]
        assert committed and json.dumps(rows, ensure_ascii=False) == json.dumps(committed, ensure_ascii=False)
