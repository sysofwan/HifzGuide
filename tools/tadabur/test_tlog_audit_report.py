"""The generated audit report's refusals and its §9 outcome (``tlog_ident_run.render_audit``).

A report rendered from zero verdicts would print "trigger fired: False" from no evidence, and a
verdict file from another worklist would silently drop rows. Both are refused. A partly answered
sitting still renders, with the S5b §9 outcome read against the drawn denominator.
"""

from __future__ import annotations

import json

import pytest

from . import tlog_ident_run as run


def _row(row: int, stratum: str, *, split: str = "clean", **fields) -> dict:
    return {"population": "look", "role": "look", "split": split,
            "shard_file": f"{split}-00000-of-00411", "row": row,
            "audio_path": f"audio/{split}-00000-of-00411.parquet", "audio_sha256": "0" * 64,
            "duration_s": 12.5, "filename_label": "2:1", "assigned_label": "2:1",
            "class_label": "2:1", "label_source": "both", "id_floor_ok": True,
            "overlaps_filename": False, "blocking_reason": None, "stratum": stratum,
            "local_audio": f"{row:016x}_00000000.wav", **fields}


def _paths(tmp_path, rows, verdicts=None):
    worklist = tmp_path / "worklist.jsonl"
    worklist.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
    verdicts_path = tmp_path / "verdicts.jsonl"
    if verdicts is not None:
        verdicts_path.write_text(
            "".join(json.dumps({**{field: row[field] for field in
                                   ("split", "shard_file", "row")}, **verdict},
                               ensure_ascii=False) + "\n"
                    for row, verdict in verdicts), encoding="utf-8")
    return worklist, verdicts_path


def _render(tmp_path, rows, verdicts=None, ident_dir=run.IDENT_DIR):
    worklist, verdicts_path = _paths(tmp_path, rows, verdicts)
    return run.render_audit(worklist_path=worklist, verdicts_path=verdicts_path,
                            ident_dir=ident_dir)


def test_a_report_without_verdicts_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="has no verdicts"):
        _render(tmp_path, [_row(1, "both")])


def test_a_verdict_off_another_worklist_is_refused(tmp_path):
    rows = [_row(1, "both")]
    stray = _row(999, "both")
    with pytest.raises(SystemExit, match="not in the worklist"):
        _render(tmp_path, rows, [(stray, {"verdict": run.RIGHT, "ayah": "2:1"})])


def test_a_partial_sitting_renders_and_its_small_n_is_not_read(tmp_path):
    rows = [_row(1, "both"), _row(2, "both"), _row(3, "non_recitation",
                                                   id_floor_ok=False)]
    document = _render(tmp_path, rows, [
        (rows[0], {"verdict": run.RIGHT, "ayah": "2:1"}),
        (rows[2], {"verdict": run.SILENCE, "ayah": None})])
    assert "2 of 3 drawn items answered" in document
    assert "n too small to read" in document
    assert "## The S5b §9 trigger" in document
    assert "**Suspended: False.**" in document


def test_the_ninth_section_suspends_the_consequence_at_three_of_the_drawn_items(tmp_path):
    rows = [_row(row, "non_recitation", id_floor_ok=False) for row in range(1, 4)]
    heard = [{**row, "filename_label": None, "assigned_label": None, "label_source": "none"}
             for row in rows]
    document = _render(tmp_path, heard, [
        (row, {"verdict": run.RIGHT, "ayah": "2:1"}) for row in heard])
    assert "**3 of 3 drawn** (3 answered)" in document
    assert "**Suspended: True.**" in document
    assert "do not start" in document


def test_the_five_percent_rule_and_the_s4_disclosure_are_generated_from_committed_files(
        tmp_path):
    rows = [_row(1, "both"), _row(2, "both")]
    document = _render(tmp_path, rows, [(rows[0], {"verdict": run.RIGHT, "ayah": "2:1"})])
    assert "## The 5% rule" in document
    assert "triggered = **" in document
    assert "## Disclosure: the S4 exclusions (not audited here)" in document
    assert "## Disclosure: the floor's calibration population" in document
    assert "tlog clean look retention is" in document
    assert "## Served order" in document

def test_an_unclear_verdict_in_a_labelled_stratum_is_not_a_wrong_ayah(tmp_path):
    rows = [_row(row, "both") for row in range(1, 41)]
    document = _render(tmp_path, rows, [
        *[(row, {"verdict": run.RIGHT, "ayah": "2:1"}) for row in rows[:39]],
        (rows[39], {"verdict": run.UNCLEAR, "ayah": None})])
    assert "Residual wrong-ayah rate in `both`: 0.00%" in document
    assert "Trigger fired: **False**" in document
    assert "| both | 40 | 40 | 39 | 39 | 100.00% |" in document


def test_the_best_class_readout_follows_the_heard_ayah(tmp_path):
    rows = [_row(row, "below-bar disagreement", class_label="2:2") for row in range(1, 16)]
    matching = _render(tmp_path, rows, [
        (row, {"verdict": run.OTHER, "ayah": "2:2"}) for row in rows])
    elsewhere = _render(tmp_path, rows, [
        (row, {"verdict": run.OTHER, "ayah": "3:1"}) for row in rows])
    assert "Best-class suggestions heard as the recitation: **15 of 15 answered**" in matching
    assert "Best-class suggestions heard as the recitation: **0 of 15 answered**" in elsewhere
    assert "Filename-label precision in the same items: 0.00%" in matching


def test_the_floor_sampling_frame_is_disclosed(tmp_path):
    frame = run._floor_frame(run.IDENT_DIR)
    assert frame == {"failing": 2_439, "both": 450}
    rows = [_row(1, "both")]
    document = _render(tmp_path, rows, [(rows[0], {"verdict": run.RIGHT, "ayah": "2:1"})])
    assert "Sampling frame of the `non_recitation` stratum" in document
    assert f"{frame['both']} also match their filename class" in document
    assert f"estimates the {frame['failing'] - frame['both']} floor-failing rows" in document
