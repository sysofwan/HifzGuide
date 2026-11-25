"""Amendment A1's committed coverage record, re-derived from its inputs (#323).

``test_followalong_prereg_coverage.py`` (locked) compares ``app_arm/cells.json`` with constants,
so a change to ``coverage.py``, the alignment or the reference would leave the committed record
stale with every test still green. This file re-derives the S5 classes through ``coverage.py`` from
the committed inputs, in about a second (inner loop, not ``slow``):

* the Teacher's decode: the pinned base decode ``consonant2_readout/decodes/tlog.confirm.base.jsonl.gz``;
* the App arm's word table ``app_arm/word_grades.jsonl.gz`` (which words the App left ungraded);
* ``MurajaText(data/quran.db)``. The audit's harness database is not in the repository; the
  phoneme text and groups the gate reads are the same in ``data/quran.db``.

Only S5 depends on the transcript rule. S1, S3 and S4 need the engine's detector grades, are never
uncovered, and come from the committed ``app_arm/pools.json``; ``s5_engine_graded`` needs the
engine and is not recomputed here.
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from tadabur.followalong_audit import coverage
from tadabur.followalong_audit.coverage import ayah_coverage
from tadabur.followalong_audit.strata import DETECTORS, stratify
from tadabur.followalong_audit.strata_run import coverage_record
from tadabur.tlog_mining.candidate_inventory import read_pinned
from training.muraja_policy import MurajaText

ROOT = Path(__file__).resolve().parents[3]
TADABUR = ROOT / "tools" / "tadabur"
ARM = TADABUR / "followalong_audit" / "app_arm"
CELLS = json.loads((ARM / "cells.json").read_text(encoding="utf-8"))
POOLS = json.loads((ARM / "pools.json").read_text(encoding="utf-8"))["pools"]["strata"]

BASE_DECODE = "consonant2_readout/decodes/tlog.confirm.base.jsonl.gz"
S5_CLASSES = ("S5-leading", "S5-trailing", "S5-interior")
WHOLE_CLASSES = ("S1", "S3", "S4")
RECORD_KEYS = ("frame", "not_covered", "frame_words", "not_covered_words")


@pytest.fixture(scope="module")
def teacher_rows() -> dict[str, dict]:
    lines = gzip.decompress(read_pinned(BASE_DECODE, TADABUR)).decode("utf-8").splitlines()
    return {row["key"]: row for row in map(json.loads, lines)}


@pytest.fixture(scope="module")
def app_words() -> dict[str, list[dict]]:
    """Each clip's App word table, in word order."""
    with gzip.open(ARM / "word_grades.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    clips: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda row: (row["item"], row["word"])):
        clips.setdefault(row["item"], []).append(row)
    return clips


@pytest.fixture(scope="module")
def text() -> MurajaText:
    return MurajaText(ROOT / "data" / "quran.db")


def app_outcome(cell: dict) -> str:
    if not cell["graded"]:
        return "ungraded"
    return "flagged" if cell["flagged"] else "passed"


def s5_record(clips: dict[str, list[dict]], teacher_rows: dict[str, dict], text: MurajaText) -> dict:
    """The coverage record's S5 classes, through ``ayah_coverage`` and ``coverage_record``.

    Words the App graded carry no detector flags here, so only S5's classes are meaningful; S5's
    edges are decided over every word of the clip, as the committed record decides them.
    """
    words = []
    for key, rows in clips.items():
        ungraded = [row["word"] for row in rows if app_outcome(row["strict"]) == "ungraded"]
        row = teacher_rows[key]
        surah, ayah = (int(part) for part in row["surah_ayah"].split(":"))
        covered = ayah_coverage(row, text, surah, ayah, ungraded) if ungraded else {}
        for item in rows:
            app = app_outcome(item["strict"])
            words.append({"key": key, "position": item["word"], "word": item["word"], "app": app,
                          "oracle": None, "detectors": dict.fromkeys(DETECTORS, False),
                          "detector_graded": dict.fromkeys(DETECTORS, False), "causes": [],
                          "transcript_covered": covered[item["word"]] if app == "ungraded" else None})
    record = coverage_record(stratify(words))
    return {section: {name: record[section][name] for name in S5_CLASSES}
            for section in ("frame", "not_covered")}


def committed_record(s5: dict) -> dict:
    """The record ``cells.json`` should hold: ``s5`` for S5, the pools' sizes for the whole strata."""
    whole = {"S1": POOLS["S1"] + POOLS["S1-pending"], "S3": POOLS["S3"], "S4": POOLS["S4"]}
    frame = {**whole, **s5["frame"]}
    not_covered = {**dict.fromkeys(WHOLE_CLASSES, 0), **s5["not_covered"]}
    return {"frame": frame, "not_covered": not_covered,
            "frame_words": sum(frame.values()), "not_covered_words": sum(not_covered.values())}


def committed_coverage() -> dict:
    return {key: CELLS["coverage"][key] for key in RECORD_KEYS}


def test_the_committed_coverage_record_is_what_the_inputs_produce(teacher_rows, app_words, text):
    s5 = s5_record(app_words, teacher_rows, text)
    assert s5["frame"] == {"S5-leading": 2038, "S5-trailing": 1553, "S5-interior": 196}
    assert s5["not_covered"] == {"S5-leading": 1895, "S5-trailing": 3550, "S5-interior": 110}
    assert committed_record(s5) == committed_coverage()


def test_a_changed_committed_count_is_caught(teacher_rows, app_words, text):
    recomputed = committed_record(s5_record(app_words, teacher_rows, text))
    for section, name in (("frame", "S5-leading"), ("not_covered", "S5-interior"),
                          ("frame", "S1")):
        edited = copy.deepcopy(committed_coverage())
        edited[section][name] += 1
        assert recomputed != edited, (section, name)
    edited = copy.deepcopy(committed_coverage())
    edited["frame_words"] += 1
    assert recomputed != edited


def test_the_recompute_runs_through_coverage_py(teacher_rows, app_words, text, monkeypatch):
    """A change to the rule in ``coverage.py`` moves the recomputed record off the committed one."""
    some = dict(list(app_words.items())[:200])
    honest = s5_record(some, teacher_rows, text)
    monkeypatch.setattr(coverage, "covered_words", lambda text, reference, spans: [True] * len(spans))
    assert s5_record(some, teacher_rows, text) != honest
