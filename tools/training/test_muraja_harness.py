"""Muraja's own engine, through the harness, on the parity cases (``muraja_harness/
parity_fixtures.json``): GPT-6 Astra's counterexamples to the dropped Python port, and one case
per ADR-0012 allowance flag. Skipped where the harness is not built (``muraja_harness/build.sh``
needs macOS and a Muraja checkout)."""

from __future__ import annotations

import json

import pytest

from muraja_harness.make_parity_fixtures import CASES, FIXTURES_PATH, request
from training.muraja_policy import MURAJA_COMMIT, Harness

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
        ("haraka_added_at_sukun", 9, "tashkeelError", "tashkeelError", "correct"),
        ("word_initial_geminate_single", 11, "correct", "correct", "correct"),
        ("single_decoded_double", 14, "correct", "correct", "correct"),
        ("dhal_heard_as_zah", 9, "minor", "minor", "correct"),
    ],
)
def test_the_amended_adr_0011_table(case, word, strict, balanced, lenient):
    """ADR-0011 §2's amended table, cell by cell, as the engine grades it (the kept grade; the
    lenient sukun cell is locked correct by an earlier partial check)."""
    kept = {mode: _final(case, mode)[word]["quality"] for mode in ("strict", "balanced", "lenient")}
    assert kept == {"strict": strict, "balanced": balanced, "lenient": lenient}
