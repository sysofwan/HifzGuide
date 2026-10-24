"""Tests for ``tadabur.tlog_index``: the tlog metadata index and the hold-out rules (#185).

The index is metadata only: every row is one ``(split, shard, row)`` of ``tarteel-ai/tlog`` at its
pinned revision, read from the ``audio.path`` column alone. The pure rules — the filename parse, the
id groups, the Tarteel v1 overlap and the pre-registered role — are tested on synthetic rows; the
committed index is tested for coverage, determinism and the counts its ``meta.json`` records.

Test command: ``python -m pytest -q tools/tadabur/test_tlog_index.py`` from the repo root.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from tadabur import qud_source
from tadabur import tlog_index as ti
from tadabur.nonpro_sources import TLOG, tlog_split

#: The repo's ``data/quran.db`` (gitignored, present where the tests run).
QURAN_DB = Path(ti.__file__).resolve().parents[2] / "data" / "quran.db"
#: The frozen Tarteel v1 split (#160), whose clip ids the overlap reads.
TARTEEL_V1_SPLIT = (Path(ti.__file__).resolve().parent / "nonpro_eval" / "tarteel_v1"
                    / "split.json")
#: The full row schema, in no particular order: exactly these keys, no dataset label.
ROW_KEYS = frozenset({
    "schema_version", "split", "shard_file", "row", "audio_path", "filename_label",
    "recording_id", "id_group", "tarteel_v1_eval_overlap", "tarteel_v1_any_overlap",
})
ROWS = {"clean": 398490, "unclean": 321363}
TOTAL_ROWS = 719853
LABELLED = 398753
WITH_RECORDING_ID = 704491
ID_GROUPS = 710589
ROWS_ON_DUPLICATED_IDS = 15639


# --- the filename rule -----------------------------------------------------------------------


def test_the_ayah_counts_come_from_quran_db():
    counts = ti.ayah_counts(QURAN_DB)
    assert len(counts) == 114 and sum(counts.values()) == 6236
    assert (counts[1], counts[2], counts[114]) == (7, 286, 6)


@pytest.mark.parametrize(("path", "label"), [
    ("100_10_1086409507.flac", "100:10"),
    ("101_1_2000480248_JUw7QgO.flac", "101:1"),      # the id-bearing suffix form
    ("38_1_1895007097.wav", "38:1"),                 # unclean, numeric
    ("100_10_null.flac", "100:10"),                  # ayah known, recording id lost
    ("102_3_undefined_9l53SfT.flac", "102:3"),
    ("1_8_1.flac", None),                            # surah 1 has 7 ayat
    ("2_287_1.flac", None),                          # surah 2 has 286
    ("0_1_1.flac", None),
    ("115_1_1.flac", None),
    ("-1_-1_1000017196.wav", None),                  # unclean's lost-ayah form
    ("012a3c4f-2239-450e-bd73-e07d043a67dc_1610517908_float64_48000_1.wav", None),
    ("recording_05jCcB5.wav", None),
    ("__0.wav", None),
    ("sample.wav", None),
])
def test_a_filename_label_is_surah_colon_ayah_checked_against_quran_db(path, label):
    assert ti.filename_label(path, ti.ayah_counts(QURAN_DB)) == label


@pytest.mark.parametrize(("path", "rid"), [
    ("100_10_1086409507.flac", "1086409507"),
    ("101_1_2000480248_JUw7QgO.flac", "2000480248"),
    ("38_1_1895007097.wav", "1895007097"),
    ("-1_-1_1000017196.wav", "1000017196"),          # id kept where the ayah is lost
    ("100_10_null.flac", None),
    ("100_10_undefined_6UygQ4s.flac", None),
    # The uuid family's third field is the audio spec, never an id.
    ("012a3c4f-2239-450e-bd73-e07d043a67dc_1610517908_float64_48000_1.wav", None),
    ("recording.wav", None),
    ("recording_05jCcB5.wav", None),
    ("__0.wav", None),
])
def test_a_recording_id_is_the_third_name_field_only_when_it_is_all_digits(path, rid):
    assert ti.recording_id(path) == rid


# --- id groups and the Tarteel v1 overlap ----------------------------------------------------


def test_id_groups_union_paths_and_recording_ids_across_both_splits():
    rows = [
        {"split": "clean", "audio_path": "b_1_7.flac", "recording_id": "7"},
        {"split": "unclean", "audio_path": "a_1_7.wav", "recording_id": "7"},
        {"split": "clean", "audio_path": "b_1_7.flac", "recording_id": "7"},
        {"split": "clean", "audio_path": "x_1_8.flac", "recording_id": "8"},
        {"split": "unclean", "audio_path": "x_1_8.flac", "recording_id": "9"},  # path links 8, 9
        {"split": "unclean", "audio_path": "c_1_9.wav", "recording_id": "9"},
        {"split": "clean", "audio_path": "z_1_1.flac", "recording_id": None},
    ]
    assert ti.id_groups(rows) == {
        "a_1_7.wav": "a_1_7.wav", "b_1_7.flac": "a_1_7.wav",
        "c_1_9.wav": "c_1_9.wav", "x_1_8.flac": "c_1_9.wav",
        "z_1_1.flac": "z_1_1.flac",
    }


def test_the_tarteel_v1_ids_are_the_third_field_of_every_decoded_clip():
    record = {"clips": 2, "look": ["100_11_3215204702.wav"],
              "confirm": ["1_1_9.flac"]}
    assert ti.tarteel_v1_ids(record) == frozenset({"3215204702", "9"})
    split = json.loads(TARTEEL_V1_SPLIT.read_text(encoding="utf-8"))
    ids = ti.tarteel_v1_ids(split)
    assert len(split["look"]) + len(split["confirm"]) == 8950
    # Four decoded names (``36_35_null.wav`` and three others) carry no id.
    assert len(ids) == 8839 and all(value.isdigit() for value in ids)


def test_the_overlap_flags_report_eval_and_any_separately():
    assert ti.overlap_flags("10", frozenset({"10"}), frozenset({"10"})) == (True, True)
    assert ti.overlap_flags("11", frozenset({"10"}), frozenset({"11"})) == (False, True)
    assert ti.overlap_flags(None, frozenset({"10"}), frozenset({"10"})) == (False, False)


# --- the pre-registered role -----------------------------------------------------------------


def test_the_role_is_the_pre_registered_u_ranges_at_their_boundaries():
    assert ti.ROLES == ("look", "confirm.1", "confirm.2", "confirm.3", "confirm.4", "train")
    assert ti.ROLE_SALT == "tlog-holdout-2026-10"
    cutoffs = ti.ROLE_CUTOFFS
    assert cutoffs == tuple(2**64 * percent // 100 for percent in (2, 4, 6, 8, 10))
    assert ti.role_from_hash(0) == "look"
    assert ti.role_from_hash(2**64 - 1) == "train"
    for index, cutoff in enumerate(cutoffs):
        assert ti.role_from_hash(cutoff - 1) == ti.ROLES[index]
        assert ti.role_from_hash(cutoff) == ti.ROLES[index + 1]


def test_a_group_key_s_role_is_the_role_of_its_hash():
    key = "100_10_1086409507.flac"
    value = int.from_bytes(
        hashlib.sha256(f"tlog-holdout-2026-10:{key}".encode()).digest()[:8], "big")
    assert ti.group_hash(key) == value
    assert ti.role(key) == ti.role_from_hash(value)


# --- writing a shard, and the column read ----------------------------------------------------


def test_a_shard_written_twice_is_the_same_bytes(tmp_path):
    rows = [
        {"schema_version": ti.SCHEMA_VERSION, "split": "clean",
         "shard_file": "clean-00000-of-00411", "row": 0,
         "audio_path": "100_10_1086409507.flac", "filename_label": "100:10",
         "recording_id": "1086409507", "id_group": "100_10_1086409507.flac",
         "tarteel_v1_eval_overlap": False, "tarteel_v1_any_overlap": False},
        {"schema_version": ti.SCHEMA_VERSION, "split": "clean",
         "shard_file": "clean-00000-of-00411", "row": 1,
         "audio_path": "100_10_null.flac", "filename_label": "100:10",
         "recording_id": None, "id_group": "100_10_null.flac",
         "tarteel_v1_eval_overlap": False, "tarteel_v1_any_overlap": False},
    ]
    first, second = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    ti.write_shard(rows, first)
    ti.write_shard(rows, second)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[4:8] == b"\x00\x00\x00\x00"  # gzip MTIME 0
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert [json.loads(line) for line in handle] == rows


def test_the_index_reads_the_audio_path_column_and_nothing_else(monkeypatch):
    calls = []

    class Table:
        def to_pylist(self):
            return [{"audio": {"path": "1_1_1.flac"}}]

    def fake_parquet_table(repo, revision, path, columns=None):
        calls.append((repo, revision, path, columns))
        return Table()

    monkeypatch.setattr(qud_source, "parquet_table", fake_parquet_table)
    shard = TLOG.shards[0]
    assert ti.fetch_paths(shard) == ["1_1_1.flac"]
    # Never ``label`` or ``is_clean``: the dataset's labels are not read.
    assert calls == [(TLOG.repo, TLOG.revision, shard, ["audio.path"])]


# --- the committed index ---------------------------------------------------------------------


def _index_files() -> list[Path]:
    return sorted(ti.INDEX_DIR.glob("*/*.jsonl.gz"))


def _rows() -> list[dict]:
    rows = []
    for path in _index_files():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows += [json.loads(line) for line in handle]
    return rows


def test_the_committed_index_covers_every_shard_and_its_rows():
    expected = {ti.INDEX_DIR / tlog_split(shard) / f"{Path(shard).stem}.jsonl.gz":
                Path(shard).stem for shard in TLOG.shards}
    files = set(_index_files())
    assert files == set(expected) and len(files) == 743
    totals = Counter()
    for path in sorted(files):
        split = path.parent.name
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        assert rows
        assert [row["row"] for row in rows] == list(range(len(rows)))
        assert {row["split"] for row in rows} == {split}
        assert {row["shard_file"] for row in rows} == {expected[path]}
        assert {row["schema_version"] for row in rows} == {ti.SCHEMA_VERSION}
        assert all(set(row) == ROW_KEYS for row in rows)
        assert all(isinstance(row["audio_path"], str) and row["audio_path"] for row in rows)
        totals[split] += len(rows)
    assert dict(totals) == ROWS


def test_the_committed_rows_are_parsed_and_grouped_as_the_meta_records():
    rows = _rows()
    meta = json.loads((ti.INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == ti.SCHEMA_VERSION
    assert meta["source"]["repo"] == "tarteel-ai/tlog"
    assert meta["source"]["revision"] == "9fe4f7e6d1b372c2c11157d6d2f0f91ca344eb51"
    assert re.fullmatch(r"[0-9a-f]{40}", meta["code_sha"])
    counts = meta["counts"]
    assert counts["rows"] == {**ROWS, "total": TOTAL_ROWS}
    assert len(rows) == TOTAL_ROWS
    assert sum(1 for row in rows if row["filename_label"]) == LABELLED
    assert sum(1 for row in rows if row["recording_id"]) == WITH_RECORDING_ID

    recomputed = ti.id_groups(rows)
    assert all(row["id_group"] == recomputed[row["audio_path"]] for row in rows)
    assert all(row["id_group"] <= row["audio_path"] for row in rows)
    assert all(recomputed[row["id_group"]] == row["id_group"] for row in rows)
    row_sizes = Counter(recomputed[row["audio_path"]] for row in rows)
    duplicated = sum(size for size in row_sizes.values() if size > 1)
    assert len(row_sizes) == ID_GROUPS == counts["id_groups"]
    assert duplicated == ROWS_ON_DUPLICATED_IDS == counts["rows_on_duplicated_ids"]

    eval_ids = ti.tarteel_v1_ids(json.loads(TARTEEL_V1_SPLIT.read_text(encoding="utf-8")))
    assert all(row["tarteel_v1_eval_overlap"] == (row["recording_id"] in eval_ids)
               for row in rows)
    eval_rows = sum(1 for row in rows if row["tarteel_v1_eval_overlap"])
    any_rows = sum(1 for row in rows if row["tarteel_v1_any_overlap"])
    # All 8,950 Tarteel v1 metadata rows were decoded (#156), so the two flags coincide.
    assert eval_rows == any_rows > 0
    assert eval_rows == counts["tarteel_v1_eval_overlap"]
    assert any_rows == counts["tarteel_v1_any_overlap"]


def test_the_preregistration_is_committed_with_only_its_fields_filled():
    path = Path(ti.__file__).resolve().parents[2] / "docs" / "tlog-holdout-preregistration.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# tlog hold-out and cleaning: pre-registered rules (#173)\n")
    assert "**Status:** frozen 2026-10-24, owner-approved 2026-10-24 (calls 1–5)" in text
    assert re.search(r"Index commit: `[0-9a-f]{40}`\.", text)
    assert 'sha256("tlog-holdout-2026-10:" + group_key)' in text
    for row in ("| [0.00, 0.02) | `look` |", "| [0.02, 0.04) | `confirm.1` |",
                "| [0.04, 0.06) | `confirm.2` |", "| [0.06, 0.08) | `confirm.3` |",
                "| [0.08, 0.10) | `confirm.4` |", "| otherwise | `train` |"):
        assert row in text
    assert "<date>" not in text and "<sha>" not in text


# --- the frozen pre-registration, and the build that writes the index ------------------------
# Added after selfcheck-r1 (S1.1, S1.3): the fragment checks above cannot see a changed threshold,
# and the build-only logic had no test. Both live here so the brief's targeted command runs them.
from tadabur.nonpro_sources import Dataset

PREREGISTRATION_PATH = (
    Path(ti.__file__).resolve().parents[2] / "docs" / "tlog-holdout-preregistration.md")
#: SHA-256 of the committed text, the owner-approved hold-out of 2026-10-24.
PREREGISTRATION_SHA256 = "655c8c3e8c560a6e9a6b53b48b5e41c4bf45d3de270562b7140b6b757344096d"


def test_the_tlog_preregistration_is_the_pinned_document():
    digest = hashlib.sha256(PREREGISTRATION_PATH.read_bytes()).hexdigest()
    assert digest == PREREGISTRATION_SHA256, (
        "docs/tlog-holdout-preregistration.md changed: the hold-out is frozen, so a deliberate "
        "change needs a new dated pre-registration and its pin updated here"
    )


CLEAN_SHARD = "data/clean-00000-of-00001.parquet"
UNCLEAN_SHARD = "data/unclean-00000-of-00001.parquet"
TARTEEL_SHARD = "data/train-00000-of-00001.parquet"
#: Fake pins: two tlog shards and the Tarteel v1 metadata the "any" overlap reads.
TLOG_FAKE = Dataset("tlog", "fake/tlog", "0" * 40, (CLEAN_SHARD, UNCLEAN_SHARD), "cc-by-4.0",
                    False)
TARTEEL_FAKE = Dataset("tarteel_v1", "fake/tarteel_v1", "0" * 40, (TARTEEL_SHARD,),
                       "CC BY 4.0", True)
PATHS = {
    ("tlog", CLEAN_SHARD): ["1_1_111.flac", "1_2_111.flac", "1_3_null.flac",
                            "1_4_222_AbC.flac"],
    ("tlog", UNCLEAN_SHARD): ["-1_-1_111.wav", "-1_-1_222.wav", "-1_-1_222.wav", "2_9_1.wav",
                              "3_4_500.wav"],
    ("tarteel_v1", TARTEEL_SHARD): ["1_1_111.wav", "3_3_500.wav"],
}


def _read_shard(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_the_build_writes_the_index_and_counts_it_describes(tmp_path, monkeypatch):
    monkeypatch.setattr(ti, "TLOG", TLOG_FAKE)
    monkeypatch.setattr(ti, "TARTEEL_V1", TARTEEL_FAKE)
    monkeypatch.setattr(ti, "load_split", lambda: {"look": ["1_1_111.wav"], "confirm": []})
    monkeypatch.setattr(ti, "fetch_paths",
                        lambda shard, dataset=TLOG_FAKE: PATHS[(dataset.name, shard)])

    index_dir = tmp_path / "index"
    meta = ti.build(index_dir=index_dir, quran_db=QURAN_DB, code_sha="ab" * 20, workers=1)

    clean = _read_shard(index_dir / "clean" / "clean-00000-of-00001.jsonl.gz")
    unclean = _read_shard(index_dir / "unclean" / "unclean-00000-of-00001.jsonl.gz")
    assert [row["row"] for row in clean] == [0, 1, 2, 3]
    assert [row["row"] for row in unclean] == list(range(5))
    assert all(set(row) == ROW_KEYS for row in clean + unclean)
    assert {row["split"] for row in clean} == {"clean"}
    assert {row["shard_file"] for row in unclean} == {"unclean-00000-of-00001"}
    # One row's parse: the label, the id, and the cross-split group's smallest path.
    assert (clean[0]["audio_path"], clean[0]["filename_label"], clean[0]["recording_id"],
            clean[0]["id_group"]) == ("1_1_111.flac", "1:1", "111", "-1_-1_111.wav")
    assert (clean[2]["filename_label"], clean[2]["recording_id"]) == ("1:3", None)
    # A row on the "any" overlap but not among the decoded eval clips.
    flags = {row["audio_path"]: (row["tarteel_v1_eval_overlap"], row["tarteel_v1_any_overlap"])
             for row in unclean}
    assert flags["3_4_500.wav"] == (False, True) and flags["-1_-1_111.wav"] == (True, True)

    assert meta["source"]["repo"] == "fake/tlog"
    assert meta["source"]["splits"] == {"clean": 1, "unclean": 1}
    assert meta["code_sha"] == "ab" * 20
    assert meta["counts"] == {
        "rows": {"clean": 4, "unclean": 5, "total": 9},
        "filename_labels": 6,
        "recording_ids": 8,
        "id_groups": 5,
        "rows_on_duplicated_ids": 6,
        "tarteel_v1_eval_overlap": 3,
        "tarteel_v1_any_overlap": 4,
    }
    assert json.loads((index_dir / "meta.json").read_text(encoding="utf-8")) == meta
