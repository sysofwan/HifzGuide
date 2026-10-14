"""The commit-block replay (#148): its b = 1 store, the pairing with the committed b = 0
sequences, the comparisons, the flips and the committed report. Torch-free."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tadabur.ship_a import HELD_OUT, TRAIN
from tadabur.ship_a_verified import REJECT_PILE, SHIP_A, Design
from training import commit_block as cb
from training.commit_block_doc import CODE_READING, render
from training.muraja_policy import FLAGGED, MINOR, NOT_FLAGGED, SiteGrade
from training.truth_baseline import CHECK_STORE, DEFAULT_MODELS, KeptGrades, read_cache

WINDOW = {"hop": "بَ", "overlap": "تَ", "flush": False}
STOP = {"hop": "تَ", "overlap": "", "flush": False, "stop": True}


def test_the_b1_store_is_its_own_protocol_beside_the_baseline():
    assert cb.CHECK_B1_STORE.protocols == {"checks_b1": "muraja-app-checks-v2-b1"}
    assert cb.CHECK_B1_STORE.directory.parent.name == "commit_block"
    assert cb.CHECK_B1_STORE.directory != CHECK_STORE.directory


def test_a_b1_record_keeps_the_digest_of_the_b0_sequence_its_inferences_cut():
    class Recorder:
        def app_checks_at(self, samples, blocks):
            assert tuple(blocks) == (0, 1)
            return {0: [WINDOW, STOP], 1: [STOP]}

    record = cb._b1_sequence(Recorder(), None)
    assert record == {"checks_b1": [STOP], "checks_sha256": cb.sequence_digest([WINDOW, STOP])}


def test_the_digest_reads_content_not_key_order():
    assert cb.sequence_digest([{"hop": "a", "overlap": ""}]) == cb.sequence_digest([{"overlap": "", "hop": "a"}])
    assert cb.sequence_digest([WINDOW, STOP]) != cb.sequence_digest([STOP, WINDOW])


def test_an_item_not_cut_from_the_committed_inferences_is_refused():
    b0 = {"a": {"checks": [WINDOW, STOP]}, "b": {"checks": [STOP]}}
    b1 = {"a": {"checks_sha256": cb.sequence_digest([WINDOW, STOP])},
          "b": {"checks_sha256": cb.sequence_digest([WINDOW])}}
    cb.check_pairing(b0, b1, ["a"])
    with pytest.raises(ValueError, match="1 item"):
        cb.check_pairing(b0, b1, ["a", "b"])


def test_the_comparisons_are_each_block_change_and_each_model_gap():
    assert cb.comparisons(["base", "h448"]) == [
        ("base/checks_b1", "base/checks"), ("h448/checks_b1", "h448/checks"),
        ("h448/checks", "base/checks"), ("h448/checks_b1", "base/checks_b1"),
    ]
    assert cb.comparisons(["h448"]) == [("h448/checks_b1", "h448/checks")]


def test_only_ship_a_held_out_sites_are_replayed():
    designs = {
        "a": Design("a", SHIP_A, HELD_OUT, 0.5),
        "b": Design("b", SHIP_A, TRAIN, 0.5),
        "c": Design("c", REJECT_PILE, HELD_OUT, 0.5),
    }
    sites = [SimpleNamespace(site_id=i) for i in ("a", "b", "c", "d")]
    assert [s.site_id for s in cb.held_out_ship_a(sites, designs)] == ["a"]


def test_a_flip_is_a_word_flagged_at_one_block_only():
    minor = SiteGrade(FLAGGED, MINOR, ({"kind": {"type": "mismatch", "expected": "ط", "heard": "ق"},
                                        "group_index": 0},))
    clean = SiteGrade(NOT_FLAGGED, "correct")
    kept = KeptGrades(
        {"s": {"m/checks": {"x": minor, "y": clean, "z": minor},
               "m/checks_b1": {"x": clean, "y": clean, "z": minor}}},
        {"x": "طَ", "y": "بَ", "z": "طَ"},
    )
    flips = cb.flips({"x": {"mark": "ط↔ق"}, "y": {}, "z": {}}, kept, ["m"])
    assert [(f["id"], f["mark"]) for f in flips["m"]["s"]] == [("x", "ط↔ق")]
    assert flips["m"]["s"][0]["b0"] == {"outcome": FLAGGED, "quality": MINOR,
                                        "causes": ["target_pair"], "errors": ["g0:ط→ق"]}
    assert flips["m"]["s"][0]["b1"]["causes"] == []


def test_the_transcripts_count_checks_and_changed_items():
    sequences = {"m/checks": {"a": [WINDOW, STOP], "b": [STOP]},
                 "m/checks_b1": {"a": [STOP], "b": [STOP]}}
    assert cb.transcripts(sequences, ["m"]) == {
        "m": {"items": 2, "checks": {"checks": 3, "checks_b1": 2}, "different_transcripts": 1}}


def test_the_code_reading_cites_the_pinned_commit():
    assert "`99c326f`" in CODE_READING and "GradeStore.swift:182-207" in CODE_READING


def test_the_runner_needs_no_torch():
    code = "import sys; import training.commit_block; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent.parent)


@pytest.mark.parametrize("model", sorted(DEFAULT_MODELS))
def test_the_committed_b1_sequences_are_paired_with_the_committed_b0(model):
    b1 = read_cache(cb.CHECK_B1_STORE.path(model))
    b0 = read_cache(CHECK_STORE.path(model))
    assert b1["items"] and b1["identity"] == b0["identity"]
    cb.check_pairing(b0["items"], b1["items"], sorted(b1["items"]))


def test_the_committed_doc_is_the_render_of_the_committed_report():
    report = json.loads(cb.REPORT_PATH.read_text(encoding="utf-8"))
    assert cb.DOC_PATH.read_text(encoding="utf-8") == render(report)
