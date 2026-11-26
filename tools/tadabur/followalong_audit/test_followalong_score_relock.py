"""Re-lock after self-check round 2 (#311 PR2, findings S2.1-S2.3).

Committed **beside** the locked ``test_followalong_score_*`` files, never editing them, and
locked in turn: later turns may add tests but never weaken or rewrite these.

* S2.1 — ``s5_draw/verdicts.jsonl`` is the owner's human data; ``*.jsonl`` is ignored by default, so
  it needs its own ``!`` entry in ``.gitignore`` (AGENTS.md "Commit the data this repo depends on").
* S2.2 — the locked contract of ``attribute_listened`` lists ``verdict`` among the returned fields
  and nothing asserted it; each line of ``attribution.jsonl`` carries the verdict the owner gave.
* S2.3 — ``test_followalong_score_committed.test_the_five_s1_pending_words_have_no_oracle_grade``
  intersects the five words with the *drawn* arms and none is drawn, so it checks nothing. The claim
  is about the committed tables: the five S1-pending words are App strict flags the oracle left
  ungraded. They are asserted here against ``oracle_sweep/word_grades.jsonl.gz`` and the App table,
  and "none of the five is drawn" is stated as a fact of the draw, not left implicit.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tadabur.followalong_audit.s5_score import attribute_listened
from training.muraja_policy import STRICT

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools"
AUDIT = Path(__file__).resolve().parent
DRAW = AUDIT / "s5_draw"

#: #311 comment 7158861900: (item, word) -> the App strict causes.
S1_PENDING = {
    ("55_28_6162424172.flac#0", 4): ["tashkeel"],
    ("56_34_2147808602.flac#0", 1): ["tashkeel"],
    ("79_28_420573590.flac#0", 1): ["unrecorded"],
    ("84_19_9013039918.flac#0", 4): ["soft_pair"],
    ("84_5_2752243060.flac#0", 3): ["other_consonant", "shaddah"],
}


def _table(path: Path) -> dict[tuple[str, int], dict]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return {(row["item"], row["word"]): row for row in map(json.loads, source)}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# --- S2.1 -----------------------------------------------------------------------------------------
def test_the_verdicts_file_is_not_gitignored():
    ignored = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q",
         "tools/tadabur/followalong_audit/s5_draw/verdicts.jsonl"]).returncode
    assert ignored == 1, "verdicts.jsonl is the owner's data: add a ! entry to .gitignore"


# --- S2.2 -----------------------------------------------------------------------------------------
def _arms(app_quality: str) -> dict:
    def trace(quality):
        return {"quality": quality, "score": 1.0, "withheld": None, "errors": [], "checks": [],
                "setting_checks": []}

    def arm(quality):
        return {"strict": trace(quality), "shipped_default": trace(quality)}

    return {"item": "1_2_1.flac#0", "word": 1, "app": arm(app_quality),
            "causes": {"strict": [], "shipped_default": []}, "oracle": arm("correct"),
            "teacher": arm("correct"), "student": arm("correct"), "reference": None}


@pytest.mark.parametrize("verdict, app_quality", [
    ("correct", "correct"), ("correct", "minor"), ("mistake", "minor"), ("mistake", "correct"),
    ("not sure", "correct"), ("different text", "minor"), ("unusable audio", "correct")])
def test_an_attributed_word_carries_the_verdict_the_owner_gave(verdict, app_quality):
    key = {"id": "0123456789abcdef", "item": "1_2_1.flac#0", "word": 1, "surah_ayah": "1:2",
           "stratum": "S3", "cell": "all", "inclusion_probability": 0.1}
    row = {"id": key["id"], "verdict": verdict, "kinds": ["letter"] if verdict == "mistake" else [],
           "sub_reason": None, "whole_played": False, "note": ""}
    result = attribute_listened(key, _arms(app_quality), row, suspects=frozenset(), scoring=STRICT)
    assert result["verdict"] == verdict


def test_every_line_of_the_attribution_file_carries_the_verdict_it_was_given(tmp_path):
    originals = sorted(row["id"] for row in _jsonl(DRAW / "key.jsonl") if row["repeat_of"] is None)
    given = {originals[0]: ("correct", []), originals[1]: ("mistake", ["haraka"]),
             originals[2]: ("not sure", []), originals[3]: ("different text", []),
             originals[4]: ("correct", [])}
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text("".join(
        json.dumps({"id": ident, "verdict": verdict, "kinds": kinds, "sub_reason": None,
                    "whole_played": False, "note": ""}) + "\n"
        for ident, (verdict, kinds) in given.items()), encoding="utf-8")
    out = tmp_path / "out"
    done = subprocess.run(
        [sys.executable, "-m", "tadabur.followalong_audit", "s5-score", "--verdicts",
         str(verdicts), "--out", str(out), "--iterations", "20"],
        cwd=TOOLS, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    lines = _jsonl(out / "attribution.jsonl")
    assert {line["id"]: line["verdict"] for line in lines} == {
        ident: verdict for ident, (verdict, _) in given.items()}


# --- S2.3 -----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def oracle() -> dict:
    return _table(AUDIT / "oracle_sweep" / "word_grades.jsonl.gz")


@pytest.fixture(scope="module")
def app() -> dict:
    return _table(AUDIT / "app_arm" / "word_grades.jsonl.gz")


def test_the_oracle_table_leaves_each_s1_pending_word_ungraded_in_both_scorings(oracle):
    for word in S1_PENDING:
        row = oracle[word]
        for scoring in ("strict", "shipped_default"):
            assert (row[scoring]["graded"], row[scoring]["flagged"]) == (False, False), word


def test_the_app_flags_each_s1_pending_word_strictly_for_the_issues_causes(app):
    for word, causes in S1_PENDING.items():
        cell = app[word]["strict"]
        assert (cell["graded"], cell["flagged"], sorted(cell["causes"])) == (
            True, True, sorted(causes)), word


def test_none_of_the_five_s1_pending_words_is_among_the_drawn_words():
    drawn = {(row["item"], row["word"]) for row in _jsonl(DRAW / "key.jsonl")}
    assert drawn.isdisjoint(S1_PENDING)
