"""The committed oracle sweep (#308): both sweeps complete, the doc is generated, drafts are in the note.

``docs/followalong-oracle-sweep.json`` is the report; ``docs/followalong-oracle-sweep.md`` is
rendered from it by :func:`tadabur.followalong_audit.sweep_doc.render` and never edited by hand;
the Muraja issue drafts for every pattern of ten words or more sit in the run note beside the
sweep's other provenance. Regenerating the report from the harness reproduces it exactly (the
slow tier), because the sweep is seeded and the engine deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tadabur.followalong_audit.patterns import CANDIDATE_WORDS, EXAMPLES, issue_body
from tadabur.followalong_audit.plant import KINDS
from tadabur.followalong_audit.sweep import SWEEP_SEED, build_report
from tadabur.followalong_audit.sweep_doc import render
from training.muraja_policy import CORRECT, CURRENT_COMMIT, Harness

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / "docs" / "followalong-oracle-sweep.md"
REPORT = REPO / "docs" / "followalong-oracle-sweep.json"
RUN_NOTE = Path(__file__).parent / "oracle_sweep" / "run_note.md"

#: The population the issue names: confirm.1/2 gating rows less #248's twenty unreadable clips,
#: and every ayah of the mushaf.
TLOG_CLIPS = 5_742
MUSHAF_AYAT = 6_236
MODES = ["strict", "shipped_default"]


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_the_report_names_its_seed_commit_and_modes(report):
    assert report["seed"] == SWEEP_SEED
    assert report["muraja_commit"] == CURRENT_COMMIT
    assert report["modes"] == MODES


def test_the_clean_sweep_covers_every_eligible_tlog_clip_in_both_modes(report):
    tlog = report["clean"]["tlog"]
    assert tlog["clips"] == TLOG_CLIPS
    assert sorted(tlog["graded_words"]) == sorted(MODES)
    assert all(words > TLOG_CLIPS for words in tlog["graded_words"].values())
    assert sorted(tlog["false_flags_per_1000"]) == sorted(MODES)
    assert sorted(tlog["not_graded"]) == sorted(MODES)


def test_the_mushaf_sweep_is_reported_apart_and_accounts_for_every_ayah(report):
    mushaf = report["clean"]["mushaf"]
    assert mushaf["ayat"] == MUSHAF_AYAT
    assert mushaf["graded"] + mushaf["unreadable"] == MUSHAF_AYAT
    assert mushaf["graded"] > 6_000  # the phonetizer reads all but a handful
    assert sorted(mushaf["false_flags_per_1000"]) == sorted(MODES)


def test_the_mushaf_pace_is_the_tlog_speech_span_median_and_recorded(report):
    pace = report["pace"]
    assert pace["basis"] == "speech_span_median"
    assert pace["clips"] > 5_000
    assert pace["characters_per_second"] == pytest.approx(7.17, abs=0.05)
    assert report["clean"]["mushaf"]["characters_per_second"] == pace["characters_per_second"]


def test_every_kind_is_planted_and_its_catch_rates_are_reported(report):
    planted = report["planted"]
    seen = {cell["kind"] for cell in planted["catch_rates"]}
    assert seen == set(KINDS)
    assert sorted({cell["mode"] for cell in planted["catch_rates"]}) == sorted(MODES)
    assert set(planted["unplantable"]) <= set(KINDS)
    assert all({"kind", "mode", "position", "length_bin", "n", "caught", "not_graded", "rate"}
               <= set(cell) for cell in planted["catch_rates"])


def test_haraka_drops_report_the_protected_letters_apart(report):
    drops = [cell for cell in report["planted"]["protected_split"] if cell["kind"] == "haraka_drop"]
    assert {cell["protected"] for cell in drops} == {True, False}


def test_a_pattern_is_a_candidate_exactly_when_it_covers_ten_words_and_keeps_its_evidence(report):
    assert report["patterns"], "the sweeps found nothing to group: the report is empty or the grouping is not wired"
    for pattern in report["patterns"]:
        assert pattern["candidate"] is (pattern["words"] >= CANDIDATE_WORDS)
        assert 1 <= len(pattern["examples"]) <= EXAMPLES
        assert all(example["statuses"] for example in pattern["examples"])


def test_patterns_pool_only_the_clean_sweeps_findings(report):
    # A planted miss is graded correct on purpose: pooling it here would draft an issue that
    # claims the perfect decode flagged a word it did not.
    for pattern in report["patterns"]:
        assert all(example["quality"] != CORRECT for example in pattern["examples"])


def test_each_candidate_pattern_has_its_issue_draft_in_the_run_note(report):
    note = RUN_NOTE.read_text(encoding="utf-8")
    for pattern in (p for p in report["patterns"] if p["candidate"]):
        assert issue_body(pattern) in note
    assert "not filed" in note.lower()


def test_the_doc_is_exactly_the_rendering_of_the_report():
    text = DOC.read_text(encoding="utf-8")
    assert text == render(json.loads(REPORT.read_text(encoding="utf-8")))
    assert "do not edit" in text.lower()
    assert "generated" in text.lower()


def test_the_doc_states_the_rules_a_reader_needs_to_read_a_catch():
    text = DOC.read_text(encoding="utf-8").lower()
    for phrase in ("not_graded", "protected", "inserted", "neighbour", "speech span"):
        assert phrase in text


@pytest.mark.slow
def test_regenerating_the_sweep_from_the_harness_reproduces_the_committed_report():
    harness = Harness.locate(commit=CURRENT_COMMIT)
    if harness is None:
        pytest.skip(f"no Muraja harness for {CURRENT_COMMIT}: build it with muraja_harness/build.sh")
    rebuilt = build_report(harness, seed=SWEEP_SEED)
    assert json.loads(json.dumps(rebuilt, ensure_ascii=False)) == json.loads(
        REPORT.read_text(encoding="utf-8"))
