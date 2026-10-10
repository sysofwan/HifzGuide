"""The plumbing around Muraja's engine (``muraja_policy``): scorings and allowances, the checks one
decode stands for, sites on Muraja's reference, requests out and results in. Torch-free and
harness-free; the engine itself is exercised in ``test_muraja_harness.py``."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from training.muraja_policy import (
    ALLOWANCES,
    ALLOWANCES_ON,
    FLAGGED,
    FLAGGED_ELSEWHERE,
    FLAGGED_UNATTRIBUTED,
    MURAJA_COMMIT,
    NOT_FLAGGED,
    NOT_GRADED,
    SAMPLE_RATE,
    SCORINGS,
    SHIPPED_DEFAULT,
    UNPLACED,
    Harness,
    Item,
    MurajaSite,
    MurajaText,
    Scoring,
    all_off,
    allowance_off,
    build_request,
    clusters,
    counterpart,
    displayed,
    graphemes,
    highlights,
    read_results,
    single_decode_checks,
    site_outcome,
)
from training.test_site_outcomes import DHAKARA, DHAL, KATABA, MADDA, make

# --- scorings and allowances ------------------------------------------------------------------


def test_the_baseline_has_adr_0012s_three_allowances_on():
    assert ALLOWANCES_ON.mode == "balanced" and not ALLOWANCES_ON.tashkeel_shown
    assert ALLOWANCES_ON.soft_pairs_enabled is None and ALLOWANCES_ON.shaddah_suppression is None
    assert SHIPPED_DEFAULT.tashkeel_shown and SHIPPED_DEFAULT.mode == "balanced"
    assert SHIPPED_DEFAULT.as_dict()["muraja_commit"] == MURAJA_COMMIT


def test_each_allowance_switches_one_muraja_flag_and_the_scorings_cover_them_all():
    assert [a.name for a in ALLOWANCES] == ["soft_pair_forgiveness", "shaddah_suppression", "tashkeel_toggle"]
    assert [allowance_off(a).name for a in ALLOWANCES] == [
        "soft_pair_forgiveness_off", "shaddah_suppression_off", "shipped_default",
    ]
    off = all_off(ALLOWANCES_ON)
    assert (off.soft_pairs_enabled, off.shaddah_suppression, off.tashkeel_shown) == (False, False, True)
    assert replace(next(s for s in SCORINGS if s.name == "allowances_off"), name=off.name) == off
    assert len({s.name for s in SCORINGS}) == len(SCORINGS)


def test_the_tashkeel_toggle_needs_no_engine_run_of_its_own():
    assert ALLOWANCES_ON.engine_name == SHIPPED_DEFAULT.engine_name
    assert len({s.engine_name for s in SCORINGS}) == 4


def test_affected_populations_come_from_the_truth_record():
    pair = make(DHAKARA, 2, "ذ↔ز", DHAL, DHAL)
    held = make(MADDA, 2, "shaddah", "held", "held")
    in_geminate = make(MADDA, 3, "fatha", "fatha", "fatha")
    plain = make(KATABA, 2, "fatha", "fatha", "fatha")
    zah = replace(pair, mark="ذ↔ظ")

    def names(site):
        return {a.name for a in ALLOWANCES if a.affects(site)}

    assert names(pair) == {"soft_pair_forgiveness"} and names(zah) == set()
    assert names(held) == {"shaddah_suppression"}
    assert names(in_geminate) == {"shaddah_suppression", "tashkeel_toggle"}
    assert names(plain) == {"tashkeel_toggle"}


def test_display_filters():
    lenient = Scoring("l", mode="lenient")
    assert displayed("tashkeelError", ALLOWANCES_ON) == "correct"
    assert displayed("tashkeelError", SHIPPED_DEFAULT) == "tashkeelError"
    assert displayed("minor", lenient) == "correct" and displayed("minor", SHIPPED_DEFAULT) == "minor"
    assert displayed("wrong", lenient) == "wrong"


# --- the checks one decode stands for ------------------------------------------------------------


def test_clusters_keep_marks_on_their_letter():
    assert clusters("كَتَبَ ق") == ["كَ", "تَ", "بَ", " ", "ق"]


def test_single_decode_checks_follow_the_transcriber_cadence():
    decode = "بَ" * 80  # 80 Characters over 8 s: 10 per second, 2 per 200 ms step
    checks = single_decode_checks(decode, 8 * SAMPLE_RATE)
    previews = [c for c in checks if not c["hop"] and not c["flush"]]
    windows = [c for c in checks if c["hop"] and not c["flush"]]
    # a preview every 200 ms from 2 s on (31 steps), and a full window at 5, 6, 7 and 8 s
    assert len(windows) == 4 and len(previews) == 31
    assert previews[0]["overlap"] == "بَ" * 20
    assert all(w["hop"] == "بَ" * 10 and w["overlap"] == "بَ" * 40 for w in windows)
    confirmed = "".join(c["hop"] for c in checks)
    assert confirmed == decode and checks[-1]["flush"] and not checks[-1]["overlap"]


@pytest.mark.parametrize(
    ("seconds", "previews", "windows"),
    [(1.91, 0, 0), (2.3095, 2, 0), (4.91, 15, 0), (5.0, 16, 1)],
)
def test_checks_are_scheduled_on_exact_samples(seconds, previews, windows):
    """Nothing is rounded to 200 ms: 1.91 s is too short for a preview, 4.91 s for a window,
    and the remainder after the last whole step goes to the flush."""
    n_samples = round(seconds * SAMPLE_RATE)
    decode = "بَ" * n_samples  # one Character per sample keeps the cut points exact
    checks = single_decode_checks(decode, n_samples)
    assert sum(1 for c in checks if not c["hop"] and not c["flush"]) == previews
    assert sum(1 for c in checks if c["hop"] and not c["flush"]) == windows
    last_preview = max((len(c["overlap"]) for c in checks if not c["flush"]), default=0) // 2
    assert last_preview <= n_samples - n_samples % 3200
    assert "".join(c["hop"] for c in checks) == decode


def test_a_short_item_is_one_flush():
    assert single_decode_checks("بَتَ", SAMPLE_RATE) == [{"hop": "بَتَ", "overlap": "", "flush": True}]


# --- sites on Muraja's reference ----------------------------------------------------------------

AYAH = "وَقَاالُۥۥ لَںںں تَمَسسَنَ"  # 2:80 w1-w3 as Muraja's quran.db holds them


@pytest.fixture
def text(tmp_path: Path) -> MurajaText:
    db = tmp_path / "quran.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE ayahs (surah INT, ayah INT, text TEXT, phonemes TEXT)")
    con.execute("CREATE TABLE word_map (surah INT, ayah INT, phoneme_word INT, text_word INT)")
    con.execute(
        "CREATE TABLE phoneme_groups (surah INT, ayah INT, group_idx INT, group_text TEXT,"
        " ph_start INT, ph_end INT, uthmani_word INT)"
    )
    con.execute("INSERT INTO ayahs VALUES (2, 80, '', ?)", (AYAH,))
    con.executemany("INSERT INTO word_map VALUES (2, 80, ?, ?)", [(1, 1), (2, 2), (3, 3)])
    groups = [("وَ", 1), ("قَ", 1), ("اا", 1), ("لُ", 1), ("ۥۥ", 1), ("لَ", 2), ("ںںں", 2),
              ("تَ", 3), ("مَ", 3), ("سسَ", 3), ("نَ", 3)]
    position = 0
    for index, (group, word) in enumerate(groups):
        position = AYAH.index(group, position)
        con.execute("INSERT INTO phoneme_groups VALUES (2, 80, ?, ?, ?, ?, ?)",
                    (index, group, position, position + len(group), word))
        position += len(group)
    con.commit()
    con.close()
    return MurajaText(db)


#: How the harness reports 2:80 w3's printed letters: تَمَسَّنَ (scalars ت َ م َ س ّ َ ن َ) and,
#: per phoneme group, the scalars it covers (``phonemeGroupCharIndices``).
W3_LETTERS = {"word": 3, "text": "تَمَسَّنَ", "group_chars": [[0, 1], [2, 3], [4, 5, 6], [7, 8]]}
W3_MARKS = (frozenset({0}), frozenset({1}), frozenset({2}), frozenset({3}))


def test_highlights_take_scalars_to_graphemes():
    assert highlights(W3_LETTERS) == W3_MARKS
    # 5:73 w11 إِلَٰهٍ: the char map counts eight scalars, the printed word has seven
    tanween = {"text": "إِلَٰهٍ", "group_chars": [[0, 1], [2, 3], [5], [6, 7], [7]]}
    assert highlights(tanween) == (frozenset({0}), frozenset({1}), frozenset({2}), frozenset({2}), frozenset())
    assert graphemes("إِلَٰهٍ") == [0, 0, 1, 1, 1, 2, 2]


def test_counterpart_aligns_across_spaces_and_waqf_forms():
    assert counterpart("لَں تَمَ", 4, "لَںںں تَمَسسَنَ") == 6
    assert counterpart("زَيد", 0, "بَكر") is None
    # a pausal ه against the wasl spellings of a taa marbuta's tanween
    assert counterpart("دَه", 2, "دَتَںںں") == 2
    assert counterpart("ثَه وَ", 2, "ثَتِوو وَ") == 2
    assert counterpart("دَهُم", 2, "دَتَںںں") is None  # two letters against one group


def test_a_site_maps_to_its_word_and_the_letter_the_app_marks(text):
    site = make("لَںںںتَمَسسَنَ", 10, "fatha", "fatha", "fatha", surah_ayah="2:80")  # the sin's fatha
    assert text.locate(site) == MurajaSite(2, 80, 3, 2)
    assert text.start_word(site) == 2
    lam = make("لَںںںتَمَسسَنَ", 0, "fatha", "fatha", "fatha", surah_ayah="2:80")
    assert text.locate(lam).word == 2


def test_an_unmatched_carrier_keeps_its_word(text):
    # the realized word ends in ب where Muraja's has ن: the carrier has no counterpart, but
    # the rest of its word does
    site = make("لَںںںتَمَسسَبَ", 12, "fatha", "fatha", "fatha", surah_ayah="2:80")
    where = text.locate(site)
    assert where.word == 3 and where.group is None
    assert text.locate(make("زِيدِ", 0, "kasra", "kasra", "kasra", surah_ayah="2:80")) is None


# --- what a kept grade means for a site ----------------------------------------------------------


def kept(quality: str, *groups: int) -> dict:
    return {"quality": quality, "errors": [{"group_index": g, "kind": {"type": "tashkeel"}} for g in groups]}


def test_site_outcomes_from_the_kept_grade():
    where = MurajaSite(2, 80, 3, 2)

    def outcome(site, grade, scoring=SHIPPED_DEFAULT):
        return site_outcome(site, grade, W3_MARKS, scoring).outcome

    assert outcome(where, None) == NOT_GRADED
    assert outcome(None, kept("minor", 2)) == UNPLACED
    assert outcome(where, kept("pending")) == NOT_GRADED
    assert outcome(where, kept("correct", 2)) == NOT_FLAGGED
    assert outcome(where, kept("minor", 2)) == FLAGGED
    assert outcome(where, kept("minor", 1)) == FLAGGED_ELSEWHERE
    assert outcome(where, kept("minor", 9)) == FLAGGED_ELSEWHERE
    assert outcome(where, kept("skipped")) == FLAGGED_ELSEWHERE
    assert outcome(where, kept("tashkeelError", 2)) == FLAGGED
    assert outcome(where, kept("tashkeelError", 2), ALLOWANCES_ON) == NOT_FLAGGED
    unattributed = replace(where, group=None)
    assert outcome(unattributed, kept("minor", 2)) == FLAGGED_UNATTRIBUTED
    assert outcome(unattributed, kept("correct")) == NOT_FLAGGED


def test_two_groups_printed_on_one_letter_mark_the_same_letter():
    marks = highlights({"text": "إِلَٰهٍ", "group_chars": [[0, 1], [2, 3], [5], [6, 7], [7]]})
    site_on_he = MurajaSite(5, 73, 11, 3)
    assert site_outcome(site_on_he, kept("minor", 2), marks, SHIPPED_DEFAULT).outcome == FLAGGED
    assert site_outcome(site_on_he, kept("minor", 4), marks, SHIPPED_DEFAULT).outcome == FLAGGED_ELSEWHERE


def test_requests_out_and_results_in(text):
    site = make("لَںںںتَمَسسَنَ", 10, "fatha", "fatha", "fatha", surah_ayah="2:80")
    item = Item("arm#item", (site,), "لَںںںتَمَسَنَ", 2 * SAMPLE_RATE)
    request, where = build_request(item, text, SCORINGS)
    assert (request["surah"], request["ayah"], request["start_word"], request["report_words"]) == (2, 80, 2, [3])
    assert len(request["scorings"]) == 4 and request["checks"][-1]["flush"]
    json.dumps(request, ensure_ascii=False)
    results = [
        {"item": "arm#item", "scoring": s["name"], "final": [{"word": 3, **kept("tashkeelError", 2)}],
         "letters": [W3_LETTERS]}
        for s in request["scorings"]
    ]
    out = read_results([item], {"arm#item": where}, results, SCORINGS, {"muraja_commit": MURAJA_COMMIT})
    grades = {name: g[("arm#item", site.site_id)].outcome for name, g in out.grades.items()}
    assert grades["allowances_on"] == NOT_FLAGGED and grades["shipped_default"] == FLAGGED
    assert out.unplaced == () and out.unattributed == ()


def test_a_harness_is_located_only_for_the_pinned_commit(tmp_path: Path):
    assert Harness.locate(tmp_path) is None
    root = tmp_path / MURAJA_COMMIT
    root.mkdir()
    files = {name: tmp_path / name for name in ("bin", "quran.db", "layout.db")}
    for path in files.values():
        path.write_text("")
    record = {"binary": str(files["bin"]), "quran_db": str(files["quran.db"]),
              "layout_db": str(files["layout.db"]), "muraja_commit": MURAJA_COMMIT, "compiler": "swift"}
    (root / "harness.json").write_text(json.dumps(record))
    assert Harness.locate(tmp_path).muraja_commit == MURAJA_COMMIT
    (root / "harness.json").write_text(json.dumps(record | {"muraja_commit": "other"}))
    assert Harness.locate(tmp_path) is None
