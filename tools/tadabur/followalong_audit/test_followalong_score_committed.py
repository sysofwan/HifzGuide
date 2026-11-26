"""The committed arm traces, the draw run note and the scorer CLI (#311 PR2).

Committed before the PR2 code exists and **locked**: later turns may add tests but never weaken or
rewrite these. None runs the engine or the model; the arms are the committed
``s5_draw/arms.jsonl.gz`` (built on the Mac by ``s5-arms`` from the gate's regenerated App traces,
the pinned Teacher and Student decodes and the oracle arm) and the CLI reads only committed files.

Contract this file pins
-----------------------
* ``s5_draw/arms.jsonl.gz`` (<= 10 MB): one arms row (see ``test_followalong_score_attribution``)
  per drawn ``(item, word)`` of ``key.jsonl``, fields exactly ``item, word, app, causes, oracle,
  teacher, student, reference``. App rows are full ``word_traces`` rows; the other arms are compact
  ``{quality, score, withheld, errors}``. They must agree with the committed word tables: the App
  rows' kept outcomes and ``causes`` with ``app_arm/word_grades.jsonl.gz``, the oracle's kept outcome
  with ``oracle_sweep/word_grades.jsonl.gz``. At most 5 drawn words have ``reference`` ``None``
  (architect r7: more is a stop for the owner).
* ``s5_score.candidate_pattern_words(report) -> int``: the most words a *candidate* pattern of the
  S2 report (``docs/followalong-oracle-sweep.json``) covers, 0 with none (rule 1 reads S2's committed
  report, it does not recompute).
* ``docs/followalong-audit-draw-run.md``: the draw's run note: the seed, 0 GPU-hours, the snip
  fallback rate, the five S1-pending words with a note each, §5's wording for F1-align's reference,
  and the count of drawn words with no mapped canonical word.
* ``python -m tadabur.followalong_audit s5-score --verdicts PATH --out DIR [--iterations N]`` (run
  from ``tools/``): reads ``s5_draw/`` and the verdicts, never writes into ``s5_draw/``, and writes
  ``DIR/summary.json`` and ``DIR/attribution.jsonl`` (one line per first verdict, sorted by id,
  the :func:`attribute_listened` dict under strict). Exit 0 on no verdicts (an empty or missing file);
  non-zero on a verdict for an id the key lacks. ``summary.json``::

      {"verdicts": {"served", "first_verdicts", "excluded": {"not sure", "different text",
                    "unusable audio", "not_in_clip"}, "agreement": {"pairs", "agree"}},
       "bootstrap": {"seed": subseed(draw seed, "bootstrap"), "iterations": N},
       "strict" | "balanced": {"attribution": {"false_flag_buckets": {F1..F3: n},
                                "miss_labels": {primary label: n}, "correct_catches", "true_passes"},
                               "rates": {"false_flags_per_1000": R, "misses_per_100": {"headline": R,
                                "s4": R, "s5_edge": R, "full_frame": R}},
                               "rules": evaluate_rules(...)}}

  with ``R = {"estimate", "interval", "without_edges": {"estimate", "interval"}}``, estimates and
  intervals ``None`` when undefined.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tadabur.followalong_audit.attribution import kept_outcome
from tadabur.followalong_audit.draw import derive_seed, subseed
from tadabur.followalong_audit.s5_score import candidate_pattern_words
from training.muraja_policy import AUDIT_SCORINGS

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools"
AUDIT = Path(__file__).resolve().parent
DRAW = AUDIT / "s5_draw"
RUN_NOTE = ROOT / "docs" / "followalong-audit-draw-run.md"
SEED = 9665042067186208708
ARM_FIELDS = {"item", "word", "app", "causes", "oracle", "teacher", "student", "reference"}
COMPACT = {"quality", "score", "withheld", "errors"}
F_BUCKETS = ("F1", "F1-align", "F-ref", "F2", "F3")
M_LABELS = ("M1r", "M1", "M1u", "M-nograde", "M2", "M3")
S1_PENDING = ("55_28_6162424172.flac#0", "56_34_2147808602.flac#0", "79_28_420573590.flac#0",
              "84_19_9013039918.flac#0", "84_5_2752243060.flac#0")


def _lines(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


@pytest.fixture(scope="module")
def key() -> list[dict]:
    return _lines(DRAW / "key.jsonl")


@pytest.fixture(scope="module")
def arms() -> list[dict]:
    return _lines(DRAW / "arms.jsonl.gz")


def _outcome(cell: dict) -> str:
    if not cell["graded"]:
        return "ungraded"
    return "flagged" if cell["flagged"] else "passed"


# --- the arms -------------------------------------------------------------------------------------
def test_the_arms_file_is_small_enough_to_commit():
    assert (DRAW / "arms.jsonl.gz").stat().st_size <= 10_000_000


def test_there_is_one_arms_row_for_each_drawn_word_and_no_other(arms, key):
    drawn = {(row["item"], row["word"]) for row in key if row["repeat_of"] is None}
    assert len(arms) == len(drawn) == 540
    assert {(row["item"], row["word"]) for row in arms} == drawn
    assert all(set(row) == ARM_FIELDS for row in arms)


def test_the_app_rows_are_full_traces_and_the_other_arms_compact(arms):
    for row in arms:
        for scoring in AUDIT_SCORINGS:
            app = row["app"][scoring.name]
            assert app is None or {"quality", "checks", "setting_checks", "errors"} <= set(app)
            for arm in ("oracle", "teacher", "student"):
                cell = row[arm][scoring.name]
                assert cell is None or set(cell) == COMPACT
        assert set(row["app"]) == set(row["oracle"]) == {"strict", "shipped_default"}


def test_the_app_rows_agree_with_the_committed_word_table(arms):
    committed = {(row["item"], row["word"]): row for row in _lines(AUDIT / "app_arm"
                                                                   / "word_grades.jsonl.gz")}
    for row in arms:
        table = committed[(row["item"], row["word"])]
        for scoring in AUDIT_SCORINGS:
            assert kept_outcome(row["app"][scoring.name], scoring) == _outcome(
                table[scoring.name]), (row["item"], row["word"], scoring.name)
            assert row["causes"][scoring.name] == table[scoring.name]["causes"]


def test_the_oracle_rows_agree_with_the_committed_oracle_table(arms):
    committed = {(row["item"], row["word"]): row for row in _lines(AUDIT / "oracle_sweep"
                                                                   / "word_grades.jsonl.gz")}
    for row in arms:
        table = committed[(row["item"], row["word"])]
        for scoring in AUDIT_SCORINGS:
            assert kept_outcome(row["oracle"][scoring.name], scoring) == _outcome(
                table[scoring.name]), (row["item"], row["word"], scoring.name)


def test_a_reference_is_the_oracles_text_and_at_most_five_words_have_none(arms):
    missing = [row for row in arms if row["reference"] is None]
    assert len(missing) <= 5
    for row in arms:
        reference = row["reference"]
        if reference is not None:
            assert set(reference) == {"previous", "word", "next"}
            assert isinstance(reference["word"], str) and reference["word"]


# The five S1-pending words are not drawn, so nothing here reads them; the claim that the oracle
# left each of them ungraded is asserted against the committed oracle table in
# ``test_followalong_score_relock.py``.


# --- the S2 patterns ------------------------------------------------------------------------------
def test_the_candidate_pattern_words_are_the_largest_candidate_patterns_count():
    report = {"patterns": [{"candidate": False, "words": 6}, {"candidate": True, "words": 12},
                           {"candidate": True, "words": 10}]}
    assert candidate_pattern_words(report) == 12
    assert candidate_pattern_words({"patterns": [{"candidate": False, "words": 9}]}) == 0
    assert candidate_pattern_words({"patterns": []}) == 0


def test_the_committed_s2_report_has_no_candidate_pattern():
    report = json.loads((ROOT / "docs" / "followalong-oracle-sweep.json").read_text(
        encoding="utf-8"))
    assert candidate_pattern_words(report) == 0


# --- the run note ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def note() -> str:
    return RUN_NOTE.read_text(encoding="utf-8")


def test_the_run_note_states_the_seed_the_gpu_time_and_the_snip_fallback_rate(note):
    report = json.loads((DRAW / "snips.report.json").read_text(encoding="utf-8"))
    assert str(derive_seed(json.loads((DRAW / "draw_seed.json").read_text())["amendment_commit"])) in note
    assert "0 GPU-hours" in note
    assert f"{report['fallback_rate']:.4f}" in note or f"{report['fallback_rate']:.1%}" in note
    assert f"{report['alignment_fallback_rate']:.4f}" in note or \
        f"{report['alignment_fallback_rate']:.1%}" in note
    assert f"{report['alignment_fallbacks']}" in note and f"{report['whole_clip']}" in note


def test_the_run_note_records_each_s1_pending_word_with_a_note(note):
    for item in S1_PENDING:
        assert item in note
    assert "S1-pending" in note


def test_the_run_note_cites_section_five_and_counts_words_with_no_mapped_canonical_word(note, arms):
    count = sum(1 for row in arms if row["reference"] is None)
    assert "§5" in note and "canonical" in note
    assert re.search(rf"\b{count}\b[^\n]*no mapped canonical word|no mapped canonical word[^\n]*\b{count}\b",
                     note)


# --- the CLI --------------------------------------------------------------------------------------
def _digest_draw() -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(DRAW.iterdir()) if path.is_file()}


def _score(tmp_path, rows, *, iterations=20):
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
    out = tmp_path / "out"
    before = _digest_draw()
    done = subprocess.run(
        [sys.executable, "-m", "tadabur.followalong_audit", "s5-score", "--verdicts",
         str(verdicts), "--out", str(out), "--iterations", str(iterations)],
        cwd=TOOLS, capture_output=True, text=True)
    assert _digest_draw() == before, "the scorer rewrote s5_draw/"
    return done, out


def _verdict(ident, kind="correct", kinds=()):
    return {"id": ident, "verdict": kind, "kinds": list(kinds), "sub_reason": None,
            "whole_played": False, "note": ""}


def test_the_scorer_runs_on_no_verdicts_and_reports_direction_only(tmp_path, key):
    done, out = _score(tmp_path, [])
    assert done.returncode == 0, done.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdicts"] == {
        "served": len(key), "first_verdicts": 0,
        "excluded": {"not sure": 0, "different text": 0, "unusable audio": 0, "not_in_clip": 0},
        "agreement": {"pairs": 0, "agree": 0}}
    assert summary["bootstrap"] == {"seed": subseed(SEED, "bootstrap"), "iterations": 20}
    assert (out / "attribution.jsonl").read_text(encoding="utf-8") == ""
    for name in ("strict", "balanced"):
        block = summary[name]
        assert block["attribution"] == {
            "false_flag_buckets": dict.fromkeys(F_BUCKETS, 0),
            "miss_labels": dict.fromkeys(M_LABELS, 0), "correct_catches": 0, "true_passes": 0}
        rates = block["rates"]
        assert rates["false_flags_per_1000"]["estimate"] is None
        assert rates["false_flags_per_1000"]["interval"] is None
        assert set(rates["misses_per_100"]) == {"headline", "s4", "s5_edge", "full_frame"}
        assert all(rate["estimate"] is None and rate["without_edges"]["estimate"] is None
                   for rate in rates["misses_per_100"].values())
        assert block["rules"]["rule_4"] == {"confirmed_misses": 0, "threshold": 60,
                                            "decides": False, "status": "direction only"}
        assert block["rules"]["rule_1"]["follow_along_first"] is False
        assert block["rules"]["rule_5"] is None


def test_a_missing_verdicts_file_is_the_same_as_none(tmp_path):
    out = tmp_path / "out"
    done = subprocess.run(
        [sys.executable, "-m", "tadabur.followalong_audit", "s5-score", "--verdicts",
         str(tmp_path / "absent.jsonl"), "--out", str(out), "--iterations", "20"],
        cwd=TOOLS, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads((out / "summary.json").read_text())["verdicts"]["first_verdicts"] == 0


def test_the_scorer_attributes_every_first_verdict_and_counts_the_rest(tmp_path, key):
    originals = sorted(row["id"] for row in key if row["repeat_of"] is None)
    rows = ([_verdict(ident) for ident in originals[:30]]
            + [_verdict(ident, "mistake", ["letter"]) for ident in originals[30:50]]
            + [_verdict(originals[50], "not sure"), _verdict(originals[51], "different text"),
               {**_verdict(originals[52], "unusable audio"), "sub_reason": "not_in_clip"}])
    done, out = _score(tmp_path, rows)
    assert done.returncode == 0, done.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdicts"]["first_verdicts"] == 53
    assert summary["verdicts"]["excluded"] == {"not sure": 1, "different text": 1,
                                               "unusable audio": 1, "not_in_clip": 1}
    lines = _lines(out / "attribution.jsonl")
    assert [line["id"] for line in lines] == sorted(row["id"] for row in rows)
    for name in ("strict", "balanced"):
        block = summary[name]["attribution"]
        assert sum(block["false_flag_buckets"].values()) + block["true_passes"] == 30
        assert sum(block["miss_labels"].values()) + block["correct_catches"] == 20
        assert summary[name]["rules"]["rule_4"]["confirmed_misses"] == sum(
            block["miss_labels"].values())
    assert summary["strict"]["rules"]["rule_4"]["status"] == "direction only"


def test_a_repeats_verdict_never_enters_the_rates_but_feeds_the_agreement(tmp_path, key):
    repeat = next(row for row in key if row["repeat_of"] is not None)
    rows = [_verdict(repeat["repeat_of"]), _verdict(repeat["id"])]
    done, out = _score(tmp_path, rows)
    assert done.returncode == 0, done.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdicts"]["agreement"] == {"pairs": 1, "agree": 1}
    assert summary["verdicts"]["first_verdicts"] == 1
    assert [line["id"] for line in _lines(out / "attribution.jsonl")] == [repeat["repeat_of"]]


def test_a_verdict_for_an_id_the_key_does_not_hold_stops_the_scorer(tmp_path):
    done, _ = _score(tmp_path, [_verdict("0000000000000000")])
    assert done.returncode != 0
