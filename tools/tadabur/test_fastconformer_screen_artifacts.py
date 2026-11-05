"""Tests for #267's committed run: decodes with identity and the CC-BY-4.0 NOTICE, report, doc.

They read what the run committed under ``tools/tadabur/fastconformer_screen/`` and ``docs/``, so
they stay red until the screen has run, and then pin that the artifacts agree with each other
and with the frozen rule: the verdict is the rule's answer on the report's own numbers, the
teacher's bars are the ones the erratum announced before any decode, no G6 or round-3 item was
decoded, and no audio is committed.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tadabur.fastconformer_screen.gates import NEMO_SHA256, REVISION
from tadabur.fastconformer_screen.sets import (
    ERRATUM_PATH,
    decode_plan,
    require_erratum,
    teacher_bars,
)
from tadabur.fastconformer_screen.verdict import GO, PRIOR, STOP, Rates, verdict
from training.consonant_readout import load_g6_sites

HERE = Path(__file__).resolve().parent / "fastconformer_screen"
DECODES = HERE / "decodes"
DOC = Path(__file__).resolve().parents[2] / "docs" / "fastconformer-screen.md"
QDAT_ITEMS = 1505
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".npy", ".npz", ".nemo", ".onnx"}


def read_gz_lines(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def report() -> dict:
    return json.loads((HERE / "report.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("set_name", ["ship-a", "edits", "qdat"])
def test_each_set_has_decodes_identity_and_timings(set_name):
    for suffix in (".fastconformer.jsonl.gz", ".fastconformer.identity.json",
                   ".fastconformer.timing.jsonl.gz"):
        assert (DECODES / f"{set_name}{suffix}").is_file(), f"{set_name}{suffix}"


@pytest.mark.parametrize("set_name", ["ship-a", "edits", "qdat"])
def test_the_identity_pins_the_revision_and_the_full_sha(set_name):
    record = json.loads((DECODES / f"{set_name}.fastconformer.identity.json").read_text(encoding="utf-8"))
    assert record["identity"]["revision"] == REVISION
    assert record["identity"]["nemo_sha256"] == NEMO_SHA256
    assert record["fingerprint"]["device_type"] == "cpu"            # CPU is the default and the run
    assert record["fingerprint"]["decoder"] == "ctc-greedy"


@pytest.mark.parametrize("set_name", ["ship-a", "edits"])
def test_the_decodes_are_exactly_the_plan(set_name):
    keys = [line["key"] for line in read_gz_lines(DECODES / f"{set_name}.fastconformer.jsonl.gz")]
    assert sorted(keys) == decode_plan(set_name)


def test_qdat_is_decoded_whole():
    lines = read_gz_lines(DECODES / "qdat.fastconformer.jsonl.gz")
    assert len(lines) == len({line["key"] for line in lines}) == QDAT_ITEMS


def test_no_g6_item_and_no_round_3_input_was_decoded():
    g6_files = {site_id.split("#")[0].split(":", 1)[1] for site_id in load_g6_sites()}
    lines = read_gz_lines(DECODES / "ship-a.fastconformer.jsonl.gz")
    assert not g6_files & {line["audio_filename"] for line in lines}
    decoded_sets = {p.name.split(".")[0] for p in DECODES.glob("*.fastconformer.*")}
    assert decoded_sets == {"ship-a", "edits", "qdat"}


def test_every_decode_line_carries_its_ctc_text_and_provenance():
    for set_name in ("ship-a", "edits", "qdat"):
        for line in read_gz_lines(DECODES / f"{set_name}.fastconformer.jsonl.gz"):
            assert isinstance(line["ctc"], str)
            assert line["audio_sha256"] and "start_sample" in line and "end_sample" in line


def test_the_notice_attributes_nvidia_under_cc_by_4_0():
    text = (DECODES / "NOTICE").read_text(encoding="utf-8")
    for needed in ("CC-BY-4.0", "NVIDIA", "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0", REVISION):
        assert needed in text


def test_the_mapping_gate_refuses_a_report_below_its_bar(monkeypatch):
    """Step 2's gate is a refusal: too few located sites and no report is written at all."""
    from tadabur.fastconformer_screen import readout
    from tadabur.fastconformer_screen.gates import ScreenStop

    monkeypatch.setattr(readout, "located_ship_a_sites", lambda: ())
    with pytest.raises(ScreenStop, match="mapping gate"):
        readout.site_report()


def test_the_edits_are_scored_on_the_located_items_only():
    """Step 2 excludes an unlocated edit; it is not a FastConformer miss."""
    record = report()
    unlocated = len(record["mapping"]["edits"]["unlocated"])
    scored = record["mapping"]["edits"]["total"] - unlocated
    assert record["edits"]["edit"]["n"] == record["edits"]["decoy"]["n"] == scored // 2


def test_no_audio_or_weights_are_committed():
    assert not [p for p in HERE.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES]


# --- the report ---------------------------------------------------------------------------------


def test_the_verdict_is_the_frozen_rule_on_the_reports_own_numbers():
    record = report()
    expected = verdict(Rates(**record["teacher"]), Rates(**record["fastconformer"]),
                       len(record["unique_catches"]))
    assert record["verdict"] == expected and expected in (GO, PRIOR, STOP)


def test_the_teachers_bars_are_the_ones_the_erratum_announced():
    bars = teacher_bars()
    assert Rates(**report()["teacher"]) == bars
    require_erratum(bars, ERRATUM_PATH)


def test_the_mapping_gate_is_reported_with_its_unlocated_items():
    mapping = report()["mapping"]
    assert mapping["located"] + len(mapping["unlocated"]) == mapping["total"]
    assert mapping["located"] >= 0.95 * mapping["total"]


def test_the_run_note_says_cpu_and_no_time_yielded():
    note = report()["run_note"]
    assert note["device"] == "cpu" and note["gpu_hours"] == 0
    assert "yielded_seconds" in note


def test_unique_catches_are_mistake_sites_outside_g6():
    sites = report()["unique_catches"]
    assert sites == sorted(sites)
    assert not set(sites) & set(load_g6_sites())


# --- the generated doc --------------------------------------------------------------------------


def test_the_doc_opens_with_a_plain_language_summary_and_the_verdict():
    text = DOC.read_text(encoding="utf-8")
    first_section = text.split("\n## ")[1]
    assert first_section.lower().startswith("summary")
    assert report()["verdict"] in first_section


def test_the_doc_lists_every_unique_catch_by_site():
    text = DOC.read_text(encoding="utf-8")
    for site_id in report()["unique_catches"]:
        assert site_id in text


def test_the_doc_says_what_the_small_catch_sets_cannot_carry():
    text = DOC.read_text(encoding="utf-8").lower()
    assert "interval" in text and "cannot" in text


def test_a_go_says_only_worth_trying():
    if report()["verdict"] != GO:
        pytest.skip("the verdict is not GO")
    text = DOC.read_text(encoding="utf-8").lower()
    assert "worth trying" in text and "better" not in text
