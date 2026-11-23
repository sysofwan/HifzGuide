"""The committed provenance manifest of the App-arm population (#309).

``provenance_rows(rows, wav_manifest)`` is one line per clip — where the audio lives
(``split``/``shard_file``/``row``), what it is (``audio_path``/``audio_sha256``) and the WAV the
App arm graded (``wav_sha256``/``samples``) — so the committed file can be checked against the
claim-gated tranche rows. ``clips.app_arm_rows`` is those rows: the claim-gated phase A gating
rows of confirm.1/2 minus #248's unreadable ayat (5,742 of 5,762).
"""

from __future__ import annotations

import pytest

from tadabur.followalong_audit.clips import app_arm_rows, gating_rows, oracle_eligible
from tadabur.followalong_audit.provenance import (
    PROVENANCE_COLUMNS,
    PROVENANCE_PATH,
    check_provenance,
    provenance_rows,
    read_provenance,
    write_provenance,
)
from tadabur.followalong_audit.tracer import load_word_times
from training.nonpro_decode import item_key


def source_row(shard: int, index: int, label: str = "1:2") -> dict:
    return {"split": "clean", "shard_file": f"clean-{shard:05d}-of-00411", "row": index,
            "audio_path": f"{shard}_{index}.flac", "audio_sha256": f"{shard * 100 + index:064x}",
            "filename_label": label, "stratum": "gating", "group_key": f"g{shard}-{index}"}


ROWS = [source_row(1, 5), source_row(0, 3), source_row(0, 1)]
WAVS = {row["audio_sha256"]: {"wav_sha256": f"{int(row['audio_sha256'], 16) + 7:064x}",
                              "samples": 16_000 + row["row"]} for row in ROWS}


def test_a_line_names_the_audio_where_it_lives_and_the_wav_graded():
    assert set(PROVENANCE_COLUMNS) == {"audio_path", "split", "shard_file", "row", "samples",
                                       "audio_sha256", "wav_sha256"}
    for line in provenance_rows(ROWS, WAVS):
        assert set(line) == set(PROVENANCE_COLUMNS)


def test_lines_follow_the_source_order_split_shard_row_whatever_the_input_order():
    lines = provenance_rows(ROWS, WAVS)
    assert [(line["shard_file"], line["row"]) for line in lines] == [
        ("clean-00000-of-00411", 1), ("clean-00000-of-00411", 3), ("clean-00001-of-00411", 5)]
    assert provenance_rows(list(reversed(ROWS)), WAVS) == lines


def test_each_line_carries_its_wav_digest_and_sample_count():
    for line in provenance_rows(ROWS, WAVS):
        assert line["wav_sha256"] == WAVS[line["audio_sha256"]]["wav_sha256"]
        assert line["samples"] == WAVS[line["audio_sha256"]]["samples"]


def test_a_clip_the_fetch_never_wrote_is_refused_by_name():
    with pytest.raises(ValueError, match="0_1.flac"):
        provenance_rows(ROWS, {key: value for key, value in WAVS.items()
                               if key != ROWS[2]["audio_sha256"]})


def test_the_manifest_matching_its_tranche_rows_is_accepted():
    check_provenance(provenance_rows(ROWS, WAVS), ROWS)


def test_a_manifest_missing_a_tranche_row_is_refused():
    with pytest.raises(ValueError):
        check_provenance(provenance_rows(ROWS[:2], WAVS), ROWS)


def test_a_manifest_holding_a_row_the_tranche_does_not_is_refused():
    with pytest.raises(ValueError):
        check_provenance(provenance_rows(ROWS, WAVS), ROWS[:2])


def test_a_manifest_line_that_disagrees_with_its_source_row_is_refused():
    lines = provenance_rows(ROWS, WAVS)
    lines[0] = {**lines[0], "audio_sha256": "f" * 64}
    with pytest.raises(ValueError):
        check_provenance(lines, ROWS)


def test_the_manifest_round_trips_and_a_rewrite_is_byte_identical(tmp_path):
    lines = provenance_rows(ROWS, WAVS)
    write_provenance(lines, tmp_path / "a.jsonl.gz")
    write_provenance(lines, tmp_path / "b.jsonl.gz")
    assert read_provenance(tmp_path / "a.jsonl.gz") == lines
    assert (tmp_path / "a.jsonl.gz").read_bytes() == (tmp_path / "b.jsonl.gz").read_bytes()


# --- the population: claim-gated rows only -----------------------------------------------------


def test_the_population_reads_only_confirm_1_and_2_through_the_claim_gated_reader(monkeypatch):
    asked = []

    def phase_a_rows(tranche):
        asked.append(tranche)
        return [source_row(len(asked), 0)]

    monkeypatch.setattr("tadabur.tlog_mining.selection.phase_a_rows", phase_a_rows)
    records = {item_key(row["audio_path"]): {"reason": None} for row in (
        source_row(1, 0), source_row(2, 0))}
    assert [row["shard_file"] for row in app_arm_rows(records)] == [
        "clean-00001-of-00411", "clean-00002-of-00411"]
    assert asked == ["confirm.1", "confirm.2"]


def test_an_unclaimed_tranche_stops_the_population_instead_of_being_skipped(monkeypatch):
    def phase_a_rows(tranche):
        raise ValueError(f"{tranche} has no committed claim")

    monkeypatch.setattr("tadabur.tlog_mining.selection.phase_a_rows", phase_a_rows)
    with pytest.raises(ValueError, match="no committed claim"):
        app_arm_rows({})


def test_a_clip_whose_ayah_the_phonetizer_cannot_read_is_left_out(monkeypatch):
    rows = [source_row(0, 0), source_row(0, 1)]
    monkeypatch.setattr("tadabur.tlog_mining.selection.phase_a_rows", lambda tranche: rows)
    records = {item_key(rows[0]["audio_path"]): {"reason": None},
               item_key(rows[1]["audio_path"]): {
                   "reason": "the phonetizer cannot handle 106:1"}}
    assert {row["audio_path"] for row in app_arm_rows(records)} == {rows[0]["audio_path"]}


# --- the committed manifest ----------------------------------------------------------------------


def test_the_committed_manifest_is_exactly_the_5742_claim_gated_tranche_rows():
    rows = app_arm_rows()
    assert len(rows) == 5742
    assert rows == oracle_eligible(gating_rows(), load_word_times())
    check_provenance(read_provenance(PROVENANCE_PATH), rows)
