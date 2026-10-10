"""Muraja's own engine, through the harness, on the parity cases (``muraja_harness/
parity_fixtures.json``): GPT-6 Astra's counterexamples to the dropped Python port, and one case
per ADR-0012 allowance flag. Skipped where the harness is not built (``muraja_harness/build.sh``
needs macOS and a Muraja checkout)."""

from __future__ import annotations

import json

import pytest

from muraja_harness.make_parity_fixtures import CASES, FIXTURES_PATH, request
from tadabur.truth_sites import TRUTH_SITES_DIR, load_truth_sites
from training.muraja_policy import MURAJA_COMMIT, Harness, MurajaText, highlights

HARNESS = Harness.locate()
pytestmark = pytest.mark.skipif(HARNESS is None, reason="the Muraja harness is not built here")
FIXTURES = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


def _rounded(value):
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def test_the_fixtures_were_recorded_from_the_pinned_commit():
    assert FIXTURES["build"]["muraja_commit"] == MURAJA_COMMIT
    assert [c["case"] for c in FIXTURES["cases"]] == [c["case"] for c in CASES]


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["case"])
def test_the_engine_reproduces_its_recorded_grades(case):
    build, results = HARNESS.run([request(case)])
    assert build["muraja_commit"] == MURAJA_COMMIT
    assert _rounded(results) == _rounded(case["expected"])


def _final(case: str, scoring: str) -> dict:
    fixture = next(c for c in FIXTURES["cases"] if c["case"] == case)
    result = next(r for r in fixture["expected"] if r["scoring"] == scoring)
    return {s["word"]: s for s in result["final"]}


def test_what_the_parity_cases_show():
    # a bare double drops the geminate's group into a gap: balanced forgives, strict and
    # shaddah suppression off do not
    assert _final("bare_double_geminate", "balanced")[3]["quality"] == "correct"
    assert _final("bare_double_geminate", "strict")[3]["quality"] == "minor"
    assert _final("bare_double_geminate", "shaddah_off")[3]["quality"] == "minor"
    # a half-heard end word stays pending, its minor withheld
    partial = _final("partial_word", "balanced")[4]
    assert partial["quality"] == "pending" and partial["withheld"] == "minor"
    # placement reads harakat: only w15 (للَااهُ) is graded
    assert set(_final("placement_by_harakat", "balanced")) == {15}
    # switching the allowances off leaves the و exemption on
    assert _final("protected_waw_exemption", "allowances_off")[1]["quality"] == "correct"
    # held wrong grades are replaced by the engine's skip marking
    assert _final("hold_buffer", "balanced")[9]["quality"] == "skipped"
    # the soft pair is forgiven only with softPairsEnabled
    assert _final("soft_pair_flag", "balanced")[9]["quality"] == "correct"
    assert _final("soft_pair_flag", "soft_pairs_off")[9]["quality"] == "minor"


@pytest.mark.parametrize(
    ("case", "word", "strict", "balanced", "lenient"),
    [
        ("missing_haraka", 3, "tashkeelError", "tashkeelError", "correct"),
        ("wrong_haraka", 3, "tashkeelError", "tashkeelError", "tashkeelError"),
        ("haraka_added_at_sukun", 9, "tashkeelError", "tashkeelError", "tashkeelError"),
        ("word_initial_geminate_single", 11, "correct", "correct", "correct"),
        ("single_decoded_double", 14, "correct", "correct", "correct"),
        ("dhal_heard_as_zah", 9, "minor", "minor", "correct"),
    ],
)
def test_the_amended_adr_0011_table(case, word, strict, balanced, lenient):
    """ADR-0011 §2's amended table, cell by cell, as the engine keeps it."""
    kept = {mode: _final(case, mode)[word]["quality"] for mode in ("strict", "balanced", "lenient")}
    assert kept == {"strict": strict, "balanced": balanced, "lenient": lenient}


# --- regressions that do not depend on the recorded fixtures --------------------------------------


def test_the_engine_starts_where_the_request_says():
    """setPage moves the engine to the page's first ayah; the harness moves it back to the
    declared start (start 2:80:15, four previews of its word)."""
    request = {
        "item": "start", "surah": 2, "ayah": 80, "start_word": 15, "report_words": [11, 15, 20],
        "checks": [{"hop": "", "overlap": "للَااهُ", "flush": False}] * 4,
        "scorings": [{"name": "balanced", "mode": "balanced"}],
    }
    _, (result,) = HARNESS.run([request])
    assert {check["position"] for check in result["checks"]} == {"2:80:15"}
    assert [(s["word"], s["quality"]) for s in result["final"]] == [(15, "correct")]


def _letters(surah: int, ayah: int, word: int) -> dict:
    request = {
        "item": "letters", "surah": surah, "ayah": ayah, "start_word": word, "report_words": [word],
        "checks": [], "scorings": [{"name": "balanced", "mode": "balanced"}],
    }
    _, (result,) = HARNESS.run([request])
    (letters,) = result["letters"]
    return letters


def test_highlights_follow_the_apps_renderer():
    # 5:73 w11 إِلَٰهٍ: phoneme_char_map counts a scalar (a tatweel) the printed word lacks, so
    # the renderer lands the madd group (اا) and the ه group on the printed ه and the tanween
    # group (ن) on nothing
    assert highlights(_letters(5, 73, 11)) == (
        frozenset({0}), frozenset({1}), frozenset({2}), frozenset({2}), frozenset()
    )
    # 2:80 w18 تَقُۥۥلُۥۥنَ: the word scorer skips the ۥۥ groups, so its error on ل (scorer group 2)
    # is drawn on the printed و (sysofwan/Muraja#260); the mapping reproduces what the app shows
    marks = highlights(_letters(2, 80, 18))
    assert marks[2] == frozenset({2}) and marks[3] == frozenset({3})


def test_a_pausal_taa_keeps_its_word_and_letter():
    site = next(
        s for s in load_truth_sites(TRUTH_SITES_DIR / "waqf_boundaries.jsonl")
        if s.site_id.endswith("tadabur_spk0043_S1_A80_60bc490d_000002.wav#6")
    )
    where = MurajaText(HARNESS.quran_db).locate(site)
    assert where.word == 7 and where.group is not None


@pytest.mark.parametrize(
    "tail",
    [
        "tadabur_spk0093_S39_A22_1d52b201_000017.wav#7",  # 40:22, a repeated فَكَفَرُۥۥ
        "tadabur_spk0033_S20_A34_6b098b87_000024.wav#4",  # 21:34, a repeated قَبڇلِكَ run on
    ],
)
def test_a_repeated_word_is_never_placed_on_another_words_grade(tail):
    site = next(
        s for s in load_truth_sites(TRUTH_SITES_DIR / "waqf_boundaries.jsonl") if s.site_id.endswith(tail)
    )
    assert MurajaText(HARNESS.quran_db).locate(site) is None
