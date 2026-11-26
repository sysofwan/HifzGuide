"""The committed S5 draw is servable by the listening UI (#311 PR3).

Committed before ``tadabur/followalong_audit/s5_ui.py`` exists and **locked**: later turns may add
tests but never weaken or rewrite these. They load the real ``s5_draw/`` files through
``s5_ui.load_rows`` (everything but the audio, which lives on the Mac and cuda-dev): the worklist's
words against the phonetizer's, the committed pilot and main orders, the snips file, and the
wasl-merge check over the real reference. They are what makes "the UI serves the pilot list" true of
the committed files, whatever box serves it.

Contract (see ``test_followalong_ui_session`` for ``load_rows``): ``load_rows(worklist, snips,
orders, *, uthmani_words, references=None, sittings=("pilot",))``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tadabur.followalong_audit.s5_ui import load_rows, read_audio_manifest
from tadabur.followalong_audit.sweep import ayah_references
from tadabur.waqf_segments import uthmani_words
from training.muraja_policy import MurajaText

AUDIT = Path(__file__).resolve().parent
DRAW = AUDIT / "s5_draw"
DB = AUDIT.parents[2] / "data" / "quran.db"
ORDERS = {"pilot": DRAW / "pilot.jsonl", "main": DRAW / "main.jsonl"}


def _ids(path: Path) -> list[str]:
    return [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


@pytest.fixture(scope="module")
def references():
    return ayah_references(MurajaText(DB))


def test_the_pilot_list_is_served_in_its_committed_order_from_the_blind_worklist():
    rows = load_rows(DRAW / "worklist.jsonl", DRAW / "snips.jsonl", ORDERS,
                     uthmani_words=uthmani_words)
    assert [row.site.site_id for row in rows] == _ids(DRAW / "pilot.jsonl")
    assert len(rows) == 157


def test_the_main_sitting_continues_it_with_the_other_408_items():
    rows = load_rows(DRAW / "worklist.jsonl", DRAW / "snips.jsonl", ORDERS,
                     uthmani_words=uthmani_words, sittings=("pilot", "main"))
    ids = [row.site.site_id for row in rows]
    assert ids == _ids(DRAW / "pilot.jsonl") + _ids(DRAW / "main.jsonl")
    assert len(ids) == len(set(ids)) == 565


def test_no_drawn_word_is_wasl_merged_so_the_startup_check_passes_on_the_real_draw(references):
    rows = load_rows(DRAW / "worklist.jsonl", DRAW / "snips.jsonl", ORDERS,
                     uthmani_words=uthmani_words, references=references,
                     sittings=("pilot", "main"))
    assert len(rows) == 565


def test_the_committed_manifest_vouches_for_every_served_clip_and_its_length():
    manifest = read_audio_manifest(AUDIT / "app_arm" / "audio_manifest.jsonl.gz")
    rows = [json.loads(line) for line in
            (DRAW / "worklist.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        entry = manifest[row["audio_sha256"]]
        assert entry["samples"] == row["end_sample"]
        assert len(entry["wav_sha256"]) == 64


def test_the_manifest_reader_takes_the_committed_gzipped_jsonl():
    with gzip.open(AUDIT / "app_arm" / "audio_manifest.jsonl.gz", "rt", encoding="utf-8") as source:
        lines = [json.loads(line) for line in source if line.strip()]
    assert len(read_audio_manifest(AUDIT / "app_arm" / "audio_manifest.jsonl.gz")) == len(
        {line["audio_sha256"] for line in lines}) == 5742
