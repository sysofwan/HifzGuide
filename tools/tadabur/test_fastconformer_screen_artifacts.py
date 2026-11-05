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
import os
import re
import subprocess
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


@pytest.mark.parametrize("set_name", ["ship-a", "edits", "qdat"])
def test_the_timing_lines_cover_every_decoded_item(set_name):
    """A timing line per invocation, and their items sum to the set's lines: a stop is not
    a gap in the set's provenance."""
    lines = read_gz_lines(DECODES / f"{set_name}.fastconformer.jsonl.gz")
    timing = read_gz_lines(DECODES / f"{set_name}.fastconformer.timing.jsonl.gz")
    assert sum(line["items"] for line in timing) == len(lines)
    decoded_seconds = sum((line["end_sample"] - line["start_sample"]) / 16_000 for line in lines)
    assert abs(sum(line["audio_seconds"] for line in timing) - decoded_seconds) < 0.1


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


def test_the_doc_is_the_generators_output():
    """A hand-edited number in the doc would otherwise pass every other doc test."""
    from tadabur.fastconformer_screen.readout import report_doc

    assert DOC.read_text(encoding="utf-8") == report_doc(report())


@pytest.mark.slow
def test_the_report_is_the_generators_output():
    """The whole frozen report, re-derived from the committed decodes (the slow tier)."""
    from tadabur.fastconformer_screen.readout import site_report

    assert json.loads((HERE / "report.json").read_text(encoding="utf-8")) == site_report()


def test_the_mapping_gate_refuses_a_report_below_its_bar(monkeypatch):
    """Step 2's gate is a refusal: too few located sites and no report is written at all."""
    from tadabur.fastconformer_screen import readout
    from tadabur.fastconformer_screen.gates import ScreenStop

    monkeypatch.setattr(readout, "located_ship_a_sites", lambda: ())
    with pytest.raises(ScreenStop, match="mapping gate"):
        readout.site_report()


def failing_card(monkeypatch, wer: float = 20.0) -> None:
    from tadabur.fastconformer_screen import readout

    monkeypatch.setattr(readout, "card_line_record", lambda: {
        "card": {"wer": 7.61, "clips": 200},
        "measured": {"clips": 200, "wer": wer, "source": "everyayah_heldout"},
        "gate_failed": True})


def test_a_failed_card_gate_is_a_stop_record_and_needs_no_decodes(monkeypatch):
    """Step 1's gate is evaluated before any decode is read: a failed runtime still reports."""
    from tadabur.fastconformer_screen import readout

    failing_card(monkeypatch)
    monkeypatch.setattr(readout, "decode_lines",
                        lambda *a, **k: pytest.fail("a failed card gate must not read decodes"))
    record = readout.site_report()
    assert (record["verdict"], record["stopped_by"]) == (STOP, "card sanity")
    assert "card sanity" in record["refusal"] and record["card_sanity"]["gate_failed"] is True


def test_a_failed_mapping_gate_writes_its_refusal_before_exiting(tmp_path, monkeypatch):
    """The entry point persists the STOP record and every unlocated item, then re-raises."""
    from tadabur.fastconformer_screen import readout
    from tadabur.fastconformer_screen.gates import ScreenStop

    monkeypatch.setattr(readout, "located_ship_a_sites", lambda: ())
    monkeypatch.setattr(readout, "located_edits", lambda: [])
    report_path = tmp_path / "report.json"
    doc_path = tmp_path / "screen.md"
    with pytest.raises(ScreenStop, match="mapping gate"):
        readout.write(report_path=report_path, doc_path=doc_path)
    record = json.loads(report_path.read_text(encoding="utf-8"))
    assert (record["verdict"], record["stopped_by"]) == (STOP, "mapping gate")
    assert "mapping gate" in record["refusal"]
    assert len(record["mapping"]["unlocated"]) == record["mapping"]["total"] > 0
    assert len(record["mapping"]["edits"]["unlocated"]) == record["mapping"]["edits"]["total"]
    text = doc_path.read_text(encoding="utf-8")
    assert "Verdict: STOP" in text and "mapping gate" in text
    assert record["mapping"]["unlocated"][0] in text
    assert record["mapping"]["edits"]["unlocated"][0] in text


def test_a_failed_card_gate_writes_a_stop_doc_and_exits_unsuccessfully(tmp_path, monkeypatch):
    from tadabur.fastconformer_screen import readout

    failing_card(monkeypatch)
    monkeypatch.setattr(readout, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(readout, "DOC_PATH", tmp_path / "screen.md")
    with pytest.raises(SystemExit) as exit_:
        readout.main()
    assert exit_.value.code == 3
    record = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert (record["verdict"], record["stopped_by"]) == (STOP, "card sanity")
    assert "Verdict: STOP" in (tmp_path / "screen.md").read_text(encoding="utf-8")


def test_the_edits_are_scored_on_the_located_items_only():
    """Step 2 excludes an unlocated edit; it is not a FastConformer miss."""
    record = report()
    unlocated = len(record["mapping"]["edits"]["unlocated"])
    scored = record["mapping"]["edits"]["total"] - unlocated
    assert record["edits"]["edit"]["n"] == record["edits"]["decoy"]["n"] == scored // 2


def test_the_edit_bootstrap_clusters_the_pairs_by_reciter():
    """#267 resamples the reciter, so a reciter's pairs move together, not one pair each."""
    from tadabur.fastconformer_screen.readout import edit_rows

    items = [
        {"item_id": "a", "source": {"reciter_id": 7}, "pair_id": "p1", "role": "edit"},
        {"item_id": "b", "source": {"reciter_id": 7}, "pair_id": "p2", "role": "edit"},
        {"item_id": "c", "source": {"reciter_id": 8}, "pair_id": "p3", "role": "edit"},
    ]
    rows = edit_rows(items, lambda arm, item: item["item_id"] == "a")
    assert rows["base"] == [(7, 1, 1), (7, 0, 1), (8, 0, 1)]


def test_the_committed_pairs_have_fewer_reciters_than_pairs():
    """The cluster choice is observable on the committed set: 150 pairs, 44 reciters."""
    from collections import Counter

    from tadabur.fastconformer_screen.sets import edit_items

    counts = Counter(item["source"]["reciter_id"]
                     for item in edit_items() if item["role"] == "edit")
    assert max(counts.values()) > 1 and len(counts) < 150


def test_no_audio_or_weights_are_committed():
    assert not [p for p in HERE.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES]


# --- the qdat word-disagreement measure ----------------------------------------------------------


@pytest.mark.parametrize(("words", "hypothesis", "expected"), [
    (["كُتُب"], "كُتُب", (0, 1)),                 # unchanged
    (["كُتُب"], "كتتب", (1, 1)),                  # an inserted letter inside the word
    (["كُتُب"], "اكتتب", (1, 1)),                 # a leading insertion belongs to the first word
    (["كُتُب"], "كتُبا", (1, 1)),                  # a trailing insertion
    (["مِن"], "مِن", (0, 1)),
    (["مِن", "هُوَ"], "منه هو", (1, 2)),          # between-word insertion: the word it follows
    (["كُتُب"], "كت", (1, 1)),                    # a deletion
    (["كُتُب"], "كثب", (1, 1)),                   # a substitution
    (["كُتُب"], "كُتّب", (1, 1)),                 # a shadda the reference does not carry
])
def test_word_disagreement_counts_inserted_letters(words, hypothesis, expected):
    """An inserted glyph is a disagreement, never silently dropped (the write-up's measure)."""
    from tadabur.fastconformer_screen.readout import word_disagreement

    assert word_disagreement(words, hypothesis) == expected


# --- the report ---------------------------------------------------------------------------------


def test_the_verdict_is_the_frozen_rule_on_the_reports_own_numbers():
    record = report()
    expected = verdict(Rates(**record["teacher"]), Rates(**record["fastconformer"]),
                       len(record["unique_catches"]))
    if record["card_sanity"]["gate_failed"]:
        # a step-1 gate failure is the rule's own first STOP, whatever the screen's numbers say
        assert (record["verdict"], record["stopped_by"]) == (STOP, "card sanity")
    else:
        assert record["stopped_by"] is None
        assert record["verdict"] == expected and expected in (GO, PRIOR, STOP)


def test_the_card_line_is_measured_and_gated():
    """Step 1's line is a measurement, and its gate's outcome follows from its own WER."""
    from tadabur.fastconformer_screen.gates import CARD_CLIPS, CARD_WER

    card = report()["card_sanity"]
    measured = card["measured"]
    assert measured["clips"] == CARD_CLIPS
    assert measured["source"] == "everyayah_heldout" and len(measured["clip_keys"]) == CARD_CLIPS
    assert card["gate_failed"] == (measured["wer"] > 2 * CARD_WER)


def test_the_teachers_bars_are_the_ones_the_erratum_announced():
    bars = teacher_bars()
    assert Rates(**report()["teacher"]) == bars
    require_erratum(bars, ERRATUM_PATH)


def test_the_mapping_gate_is_reported_on_both_denominators():
    """The gate's denominator is every site (the pre-registered wording); the scored set is
    reported beside it, and both must clear the bar."""
    record = report()
    gates = record["mapping"]["gates"]
    scored = (record["sections"]["mistakes"]["base"]["n"]
              + record["sections"]["correct"]["base"]["n"])
    assert gates["scored"]["located"] == scored              # the sites the table is read on
    assert gates["scored"]["total"] >= scored                # every site with a side
    assert gates["sites"]["total"] >= gates["scored"]["total"]
    assert min(gates["sites"]["share"], gates["scored"]["share"]) >= 95.0


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


# --- the launcher -------------------------------------------------------------------------------

LAUNCHER = Path(__file__).resolve().parents[2] / "scripts" / "fastconformer-screen.sh"


def launcher(*args, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(LAUNCHER), *args], capture_output=True, text=True,
                          env={**os.environ, **(env or {})}, check=False)


def test_the_launcher_fetches_the_run_output_not_the_staged_checkout():
    """A fetch that read the checkout could silently return the committed decodes."""
    decode = launcher("decode", "print").stdout
    out_dir = re.search(r"--out-dir '([^']+)'", decode).group(1)
    fetch = launcher("fetchout", "print").stdout
    assert f"rsync -a 'root@cuda-dev:{out_dir}/' DEST/" in fetch
    assert "root@cuda-dev:/root/scratch/fc-screen/card_line.json" in fetch
    assert "repo/tools/tadabur/fastconformer_screen/decodes" not in fetch


def test_the_launcher_refuses_to_fetch_without_a_run_output(tmp_path):
    """No run output means a refusal, never the staged checkout's committed decodes."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "ssh"
    fake.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake.chmod(0o755)
    destination = tmp_path / "fetched"
    result = launcher("fetchout", str(destination),
                      env={"PATH": f"{bin_dir}:{os.environ['PATH']}"})
    assert result.returncode != 0
    assert "no run output" in result.stderr
    assert not destination.exists()
