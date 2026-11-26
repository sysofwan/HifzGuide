"""The arm traces are pinned to the gate's own regeneration (#311 PR2).

``s5-arms`` reads the App arm from the gate's per-clip traces, and the gate's identity is pinned in
``reproduction.json``. A directory of traces from any other run would put a second App arm's
evidence beside the committed key, so the build refuses it before reading a word.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tadabur.followalong_audit.s5_arms import build_arms, write_arms

DRAW = Path(__file__).resolve().parent / "s5_draw"


def test_a_traces_directory_that_is_not_the_gates_is_refused(tmp_path):
    (tmp_path / "traces").mkdir()
    with pytest.raises(ValueError, match="traces"):
        build_arms(tmp_path / "traces", log=lambda _message: None)


def test_a_second_arms_table_is_never_written_over_the_committed_one():
    with pytest.raises(ValueError, match="arms.jsonl.gz"):
        write_arms([{"item": "other"}], DRAW / "arms.jsonl.gz")
