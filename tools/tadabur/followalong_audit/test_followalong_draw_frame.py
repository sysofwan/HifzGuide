"""The frame re-derives from the committed App table and detectors (#311, PR 1).

The rest of the draw's committed tests start at ``s5_draw/frame.jsonl.gz`` and only count its
cells against ``app_arm/cells.json``; a hand-edited frame with the right counts would pass them.
This module runs the frame's own derivation — the strata re-grade over the committed word table,
the Oracle and the failed-clip list — and requires the committed file byte for byte.

Only the frame can be re-derived here: the gate's regenerated table and traces are the second
App run on the GPU box and are not committed, so their identity is pinned by digest in
``reproduction.json`` instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tadabur.followalong_audit.build import _write_jsonl_gz, build_frame
from tadabur.followalong_audit.clips import app_arm_rows

DRAW = Path(__file__).resolve().parent / "s5_draw"


@pytest.mark.slow
def test_the_frame_is_the_committed_file_byte_for_byte(tmp_path):
    _write_jsonl_gz(build_frame(app_arm_rows()), tmp_path / "frame.jsonl.gz")
    assert (tmp_path / "frame.jsonl.gz").read_bytes() == (DRAW / "frame.jsonl.gz").read_bytes()
