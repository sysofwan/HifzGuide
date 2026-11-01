"""Tests for the zipformer screen's run drivers (#245): no Hub, no audio, no ONNX file.

The fetch record is the run's pinned provenance, so a changed file or a second, different
identity must refuse rather than resume; the benchmark manifest is validated once and every
later step reads it; the card's own PER reference (its table key, its longest-match units)
must reproduce the card's segmentation; and the export's symbolic batch axis must become a
concrete cache shape before the first ONNX call.
"""

from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from training.external_decode import record_identity
from training.zipformer_decode import BLANK_ID, CARD_ONNX_FILE, ONNX_FILE, ZipformerDecoder
from training.zipformer_gate import card_norm, card_table, reference_units
from training.zipformer_run import (
    CARD_SOURCES,
    DEFAULT_GATE_MODEL,
    FETCH_NAME,
    main,
    _file_entry,
    _read_plan,
    consumed_rows,
    files_from_record,
    load_fetch,
    manifest_rows,
    name_screen,
)


# --- the shared identity writer --------------------------------------------------------------


def test_an_identity_is_written_once_and_an_identical_rerun_passes(tmp_path):
    path = tmp_path / "identity.json"
    record = {"model": "zipformer", "identity": {"sample_scale": 1.0}}
    record_identity(path, record, label="zipformer")
    record_identity(path, dict(record), label="zipformer")
    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_a_different_identity_names_the_record_and_leaves_it_untouched(tmp_path):
    path = tmp_path / "identity.json"
    first = {"model": "zipformer", "identity": {"sample_scale": 1.0}}
    record_identity(path, first, label="zipformer")
    with pytest.raises(ValueError, match="identity.json"):
        record_identity(path, {**first, "identity": {"sample_scale": 32_768.0}},
                        label="zipformer")
    assert json.loads(path.read_text(encoding="utf-8")) == first


# --- the benchmark manifest and the pinned fetch record --------------------------------------


def _manifest(tmp_path, rows=None):
    rows = rows if rows is not None else [
        {"id": f"{source}/1", "source": source, "reference_text": "ب",
         "audio": f"audio/{source}/1.wav"} for source in CARD_SOURCES]
    path = tmp_path / "benchmark.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_the_manifest_rows_are_read_with_their_sources(tmp_path):
    rows = manifest_rows(_manifest(tmp_path))
    assert [row["source"] for row in rows] == list(CARD_SOURCES)


def test_an_exact_duplicate_row_is_the_same_clip_counted_once(tmp_path):
    row = {"id": "qul_alnufais/1", "source": "qul_alnufais", "reference_text": "ب",
           "audio": "a"}
    path = _manifest(tmp_path, [row, dict(row)])
    assert manifest_rows(path) == [row]


def test_one_id_with_two_different_rows_is_refused(tmp_path):
    path = _manifest(tmp_path, [
        {"id": "qul_alnufais/1", "source": "qul_alnufais", "reference_text": "ب", "audio": "a"},
        {"id": "qul_alnufais/1", "source": "qul_alnufais", "reference_text": "ت", "audio": "b"},
    ])
    with pytest.raises(ValueError, match="twice with different rows"):
        manifest_rows(path)


def test_an_unknown_source_is_refused(tmp_path):
    path = _manifest(tmp_path, [{"id": "x/1", "source": "elsewhere",
                                 "reference_text": "ب", "audio": "a"}])
    with pytest.raises(ValueError, match="unknown source"):
        manifest_rows(path)


def _pinned_work_dir(tmp_path):
    entries = {}
    for name in ("zipformer_p_arabic_v3.2.onnx", "zipformer_p_arabic_v3.2.int8.onnx",
                 "zipformer_p_arabic_v3.onnx", "quran_text2phoneme.json",
                 "benchmark/benchmark.jsonl", "tadabur/sheikh_dict.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))
        entries[name] = _file_entry(tmp_path, path)
    record = {
        "model": {"revision": "a" * 40},
        "benchmark": {"revision": "b" * 40},
        "files": {name: entries[name] for name in (
            "zipformer_p_arabic_v3.2.onnx", "zipformer_p_arabic_v3.2.int8.onnx",
            "zipformer_p_arabic_v3.onnx", "quran_text2phoneme.json")},
        "manifest": entries["benchmark/benchmark.jsonl"],
        "sheikh_dict": entries["tadabur/sheikh_dict.json"],
    }
    (tmp_path / FETCH_NAME).write_text(json.dumps(record), encoding="utf-8")
    return record


def test_a_fetch_record_names_the_files_the_loader_takes(tmp_path):
    record = _pinned_work_dir(tmp_path)
    files = files_from_record(tmp_path, load_fetch(tmp_path))
    assert files.onnx == tmp_path / "zipformer_p_arabic_v3.2.onnx"
    assert files.manifest == tmp_path / "benchmark/benchmark.jsonl"
    assert (files.revision, files.benchmark_revision) == ("a" * 40, "b" * 40)


def test_the_card_gates_decoder_is_the_export_the_card_describes(tmp_path):
    record = _pinned_work_dir(tmp_path)
    files = files_from_record(tmp_path, record, onnx=CARD_ONNX_FILE)
    assert files.onnx == tmp_path / "zipformer_p_arabic_v3.onnx"
    assert files.int8 == tmp_path / "zipformer_p_arabic_v3.2.int8.onnx"
    assert record["files"][CARD_ONNX_FILE]["sha256"] != (
        record["files"][ONNX_FILE]["sha256"])


def test_a_changed_fetched_file_is_refused_not_decoded(tmp_path):
    _pinned_work_dir(tmp_path)
    (tmp_path / "zipformer_p_arabic_v3.2.onnx").write_bytes(b"other bytes")
    with pytest.raises(ValueError, match="does not match the SHA-256"):
        load_fetch(tmp_path)


def test_a_missing_fetch_record_names_the_step_to_run(tmp_path):
    with pytest.raises(ValueError, match="run the fetch step"):
        load_fetch(tmp_path)


# --- the card's own reference ----------------------------------------------------------------


def test_the_card_norm_folds_diacritics_tatweel_and_letter_variants():
    assert card_norm("أَنَّ ٱللَّهَ") == "ان الله"
    assert card_norm("  مُحَمَّـــدٌ  ") == "محمد"
    assert card_norm("إِلَىٰ") == "الي"


def test_the_card_table_keys_by_its_normalization():
    table = card_table({"أَنَّ": "ان", "بَ": "بَ"})
    assert table == {"ان": "ان", "ب": "بَ"}


def test_two_raw_keys_normalizing_to_different_phonemes_are_refused():
    with pytest.raises(ValueError, match="two phoneme strings"):
        card_table({"أَنَّ": "ان", "أَن": "انن"})


def test_a_reference_becomes_the_cards_units_by_longest_match():
    table = card_table({"أَنَّ": "انن"})
    assert reference_units(table, "أَنَّ", ["ا", "ن", "ان"]) == ["ان", "ن"]


def test_a_reference_the_table_lacks_is_a_miss_not_a_guess():
    assert reference_units(card_table({"بَ": "ب"}), "ت", ["ب", "ت"]) is None


# --- the name screen -------------------------------------------------------------------------


def test_the_qul_name_is_looked_up_in_the_tadabur_dictionary():
    rows = [{"id": "qul_alnufais/1", "source": "qul_alnufais", "reference_text": "ب",
             "audio": "a"},
            {"id": "everyayah_heldout/1", "source": "everyayah_heldout",
             "reference_text": "ب", "audio": "b"},
            {"id": "tlog_holdout/1", "source": "tlog_holdout", "reference_text": "ب",
             "audio": "c"}]
    screen = name_screen(rows, {"احمد النفيس": 106},
                         {106: {"consonant2.training": 625}})
    assert screen["qul_alnufais"]["tadabur"] == {
        "tadabur_id": 106, "uses": {"consonant2.training": 625}}
    assert screen["everyayah_heldout"]["tadabur"] is None
    assert screen["tlog_holdout"]["tadabur"] is None


# --- the plan's manifest binding -------------------------------------------------------------


def test_a_plan_screened_against_another_manifest_is_refused(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    monkeypatch.setattr("training.zipformer_run.DISJOINTNESS_PATH", tmp_path / "plan.json")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "benchmark": {"manifest_sha256": "0" * 64},
        "plan": {"decode": ["qul_alnufais/1"], "verdict": [], "trained": [], "sealed": {}},
    }), encoding="utf-8")
    files = SimpleNamespace(manifest=manifest)
    with pytest.raises(ValueError, match="another benchmark manifest"):
        _read_plan(files)


def test_a_plan_that_decodes_a_sealed_clip_is_refused(tmp_path, monkeypatch):
    from tadabur.tlog_index import sha256_file

    manifest = _manifest(tmp_path)
    monkeypatch.setattr("training.zipformer_run.DISJOINTNESS_PATH", tmp_path / "plan.json")
    (tmp_path / "plan.json").write_text(json.dumps({
        "benchmark": {"manifest_sha256": sha256_file(manifest)},
        "plan": {"decode": ["qul_alnufais/1"], "verdict": [],
                 "trained": [], "sealed": {"qul_alnufais/1": ["look"]}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed"):
        _read_plan(SimpleNamespace(manifest=manifest))


def test_each_subcommand_takes_the_work_dir_after_its_name(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("training.zipformer_run.fetch", lambda work: seen.append(work))
    assert main(["fetch", "--work-dir", str(tmp_path)]) == 0
    assert seen == [tmp_path]


def test_the_gate_takes_the_model_and_defaults_to_the_cards_own(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("training.zipformer_run.gate",
                        lambda work, *, model, workers: seen.append((work, model)))
    assert main(["gate", "--work-dir", str(tmp_path)]) == 0
    assert main(["gate", "--model", "v3.2", "--work-dir", str(tmp_path)]) == 0
    assert seen == [(tmp_path, DEFAULT_GATE_MODEL), (tmp_path, "v3.2")]


# --- the export's symbolic batch axis --------------------------------------------------------


class _SymbolicSession:
    """A cache-aware export whose cache batch axis is ``"N"``, as the card's export is."""

    def __init__(self) -> None:
        self.feeds: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="x", shape=["N", 61, 80], type="tensor(float)"),
                SimpleNamespace(name="cached_key", shape=[256, "N", 128],
                                type="tensor(float)")]

    def run(self, output_names, feed):
        self.feeds.append(feed["cached_key"])
        log_probs = np.full((1, 12, 251), -10.0, dtype=np.float32)
        log_probs[0, :, BLANK_ID] = 0.0
        log_probs[0, 0, 3] = 1.0
        return [log_probs, feed["cached_key"] + 1]


class _Fbank:
    def __call__(self, waveform, **options):
        return np.zeros((len(np.asarray(waveform)) // 160, 80), dtype=np.float32)


def test_the_symbolic_batch_axis_becomes_one_stream_of_concrete_caches():
    session = _SymbolicSession()
    decoder = ZipformerDecoder(session, {3: "ب", BLANK_ID: "<blank>"}, _Fbank(), scale=1.0)
    assert decoder.stream(np.zeros(16_000, dtype=np.float32))[0].text == "ب"
    assert session.feeds[0].shape == (256, 1, 128)
    assert session.feeds[0].dtype == np.float32


# --- the consumed log is the run's own, not the pool -----------------------------------------


def _run_log(tmp_path, rows):
    path = tmp_path / "consumed.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_the_consumed_rows_are_read_and_checked_against_the_pool(tmp_path):
    log = _run_log(tmp_path, [{"split": "clean", "shard_file": "clean-1", "row": 2}])
    rows, ids = consumed_rows({"clean:clean-1:2", "clean:clean-1:3"}, log)
    assert ids == {"clean:clean-1:2"}
    assert rows == [{"split": "clean", "shard_file": "clean-1", "row": 2}]


def test_a_log_naming_a_row_outside_the_training_pool_is_refused(tmp_path):
    log = _run_log(tmp_path, [{"split": "clean", "shard_file": "clean-1", "row": 2}])
    with pytest.raises(ValueError, match="outside tlog.training"):
        consumed_rows({"clean:clean-2:2"}, log)
