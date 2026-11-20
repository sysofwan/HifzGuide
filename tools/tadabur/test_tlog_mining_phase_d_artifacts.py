"""Phase D's committed artifacts, pinned to the SHA-256 ``docs/tlog-mining-run.md`` records.

Phase A's four committed files are pinned in ``test_tlog_mining_phase_d.py`` so that a
regeneration cannot drift from the note. These three are phase D's own output — a re-merge (the
committed merge is byte-identical by construction) or a hand edit must not leave the note citing
checksums no file has.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

WORD_TIMES = Path(__file__).resolve().parents[1] / "tadabur" / "tlog_mining" / "word_times"
#: ``docs/tlog-mining-run.md``, "Phase D".
PHASE_D_SHA256 = {
    "mined.confirm.1.jsonl.gz": "5d8b275dd7f052da0d50598473326cf6b0327d7409b76c690409c52893caf541",
    "mined.confirm.2.jsonl.gz": "e2c801671faa9000d78faed29284b352821adb97f03ed65e0975a8b47057111a",
    "meta.mined.json": "01594bc682098d5149404a22bef5799f32a6947af0d664fa1ba7eb52490b2223",
}


def test_the_committed_phase_d_artifacts_are_the_ones_the_run_note_pins():
    found = {name: hashlib.sha256((WORD_TIMES / name).read_bytes()).hexdigest()
             for name in PHASE_D_SHA256}
    assert found == PHASE_D_SHA256
