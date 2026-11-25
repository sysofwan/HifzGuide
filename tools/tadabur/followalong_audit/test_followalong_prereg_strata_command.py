"""The ``strata`` command's cells output (#310, R1.3).

The count-only run writes the pools summary and no design artifact. The design artifact (S1's
frozen 200-snip allocation and S4's ungraded split) is written only when asked for, and both
records are computed before either output is replaced — an S1 pool the frozen allocation cannot
pay must not leave a new ``pools.json`` beside an old ``cells.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pytest

import tadabur.followalong_audit.__main__ as cli
import tadabur.followalong_audit.app_arm as app_arm
import tadabur.followalong_audit.strata_run as strata_run
from tadabur.followalong_audit.allocation import PRIMARY_CAUSES

DETECTORS = ("base", "c95", "r2_candidate_ema", "r3_candidate_ema")


_SERIAL = itertools.count()


def _row(stratum: str = "S5", causes=(), base: bool = False) -> dict:
    # Every row is its own clip (A1's coverage record reads the clip's words in order).
    return {
        "key": f"clip{next(_SERIAL)}#0",
        "position": 0,
        "app": "ungraded" if stratum == "S5" else "passed",
        "stratum": stratum,
        "catch_loss": False,
        "detectors": {name: name == "base" and base for name in DETECTORS},
        "detector_graded": dict.fromkeys(DETECTORS, True),
        "causes": list(causes),
    }


def _meta(words: int) -> dict:
    return {"items": 1, "words": words, "detector_ungraded": 0, "excluded_failed_clips": 0,
            "excluded_failed_words": 0, "harness": {}}


def _args(tmp_path: Path, *, cells: bool) -> argparse.Namespace:
    return argparse.Namespace(grades=Path("grades"), oracle=None, clip_runs=Path("clips"),
                              pools_out=tmp_path / "pools.json",
                              cells_out=(tmp_path / "cells.json") if cells else None)


def _run(monkeypatch, rows: list[dict]) -> None:
    monkeypatch.setattr(cli, "app_arm_rows", list)
    monkeypatch.setattr(cli, "_failed_items", lambda *_: [])
    monkeypatch.setattr(app_arm, "read_grades", lambda *_: [])
    monkeypatch.setattr(strata_run, "stratified_run",
                        lambda *_, **__: (rows, _meta(len(rows)), set()))


def test_the_count_only_run_writes_no_design_artifact(monkeypatch, tmp_path):
    _run(monkeypatch, [_row()])
    assert cli._strata(_args(tmp_path, cells=False)) == 0
    assert (tmp_path / "pools.json").is_file()
    assert not (tmp_path / "cells.json").exists()


def test_an_empty_s1_fails_before_replacing_either_output(monkeypatch, tmp_path):
    _run(monkeypatch, [_row()])
    with pytest.raises(ValueError):
        cli._strata(_args(tmp_path, cells=True))
    assert not (tmp_path / "pools.json").exists()
    assert not (tmp_path / "cells.json").exists()


def test_the_design_artifact_is_written_when_the_cells_can_pay_the_allocation(monkeypatch, tmp_path):
    rows = [_row("S1", [cause], base)
            for cause in PRIMARY_CAUSES for base in (False, True) for _ in range(20)]
    _run(monkeypatch, rows)
    assert cli._strata(_args(tmp_path, cells=True)) == 0
    cells = json.loads((tmp_path / "cells.json").read_text(encoding="utf-8"))
    assert len(cells["s1_primary_cells"]) == 14
    assert sum(cell["n"] for cell in cells["s1_primary_cells"]) == 200
