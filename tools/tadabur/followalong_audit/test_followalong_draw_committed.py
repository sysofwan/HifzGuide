"""The committed S5 draw re-derives from its seed and its inputs (#311, PR 1).

Committed before the draw exists and **locked**: later turns may add tests but never weaken or
rewrite these. The files below land in ``tadabur/followalong_audit/s5_draw/`` before any
listening; each test is about one of them and none runs the engine or the model.

Contract this file pins
-----------------------
* ``frame.jsonl.gz``: the pre-gate draw frame, one :func:`draw.frame_rows` line per covered word
  (49,000), sorted by ``(key, position)``.
* ``reproduction.json``: the F-ref gate's result::

      {"committed_words": 57935, "regenerated_words": 57935, "flips": int,
       "flips_in_s1_cells": int,
       "did_not_reproduce": [{"item", "word"}, ...],           # frame words, sorted
       "cells": [{"stratum", "cell", "frame", "did_not_reproduce", "population"}, ...]}

  ``flips`` counts every word of the population whose regenerated row differs from
  ``app_arm/word_grades.jsonl.gz``; ``did_not_reproduce`` is those that are in the frame.
* ``app_errors.jsonl.gz``: the regenerated App strict flagged words as
  :func:`attribution.reference_suspects` observations: ``{"item", "group_key", "surah", "ayah",
  "word", "errors"}``, one line per word.
* ``key.jsonl``: one line per served item (originals and repeats), never served to the page::

      {"id", "item", "word", "word_index", "surah_ayah", "stratum", "cell", "edge",
       "whole_played", "n", "population", "inclusion_probability", "in_pilot", "repeat_of"}

  An original's ``inclusion_probability`` is ``n / population``; a repeat carries its original's
  fields except ``id``, ``repeat_of`` (its original's id) and ``inclusion_probability`` (``None``).
  ``in_pilot`` says which sitting serves the item.
* ``worklist.jsonl``: the blind file, exactly ``catch_loss_worklist.WORKLIST_FIELDS``, sorted by
  id, one line per key line; a repeat repeats its original's line under its own id.
* ``pilot.jsonl`` / ``main.jsonl``: the serve orders, ``{"id", "position"}`` per served item,
  position from 1.
* ``snips.jsonl`` and ``snips.report.json``: :func:`draw.snips_for_draw`'s output and its counts
  (``rows``, ``alignment``, ``whole_clip``, ``fallback_rate``).
"""

from __future__ import annotations

import gzip
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from tadabur.catch_loss_worklist import WORKLIST_FIELDS
from tadabur.followalong_audit.attribution import reference_suspects
from tadabur.followalong_audit.draw import (
    AMENDMENT_COMMIT,
    derive_seed,
    item_id,
    plan_sitting,
    repeat_id,
    select_words,
)
from tadabur.word_snips import WHOLE_CLIP, read_snips

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
DRAW = AUDIT / "s5_draw"
SEED = derive_seed(AMENDMENT_COMMIT)
TRANCHES = ("confirm.1", "confirm.2")
CELLS = json.loads((AUDIT / "app_arm" / "cells.json").read_text(encoding="utf-8"))
KEY_FIELDS = {"id", "item", "word", "word_index", "surah_ayah", "stratum", "cell", "edge",
              "whole_played", "n", "population", "inclusion_probability", "in_pilot",
              "repeat_of"}
FRAME_FIELDS = {"key", "position", "word", "word_index", "stratum", "cell", "edge",
                "whole_played"}
EXPECTED_FRAME = {("S3", "all"): 1459, ("S4", "all"): 38598, ("S5", "leading"): 2038,
                  ("S5", "trailing"): 1553, ("S5", "interior"): 196}


def _lines(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


@pytest.fixture(scope="module")
def frame() -> list[dict]:
    return _lines(DRAW / "frame.jsonl.gz")


@pytest.fixture(scope="module")
def reproduction() -> dict:
    return json.loads((DRAW / "reproduction.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def key() -> list[dict]:
    return _lines(DRAW / "key.jsonl")


@pytest.fixture(scope="module")
def worklist() -> list[dict]:
    return _lines(DRAW / "worklist.jsonl")


@pytest.fixture(scope="module")
def draws(frame, reproduction):
    removed = {(entry["item"], entry["word"]) for entry in reproduction["did_not_reproduce"]}
    return select_words(frame, seed=SEED, removed=removed)


def _originals(key) -> dict[str, dict]:
    return {row["id"]: row for row in key if row["repeat_of"] is None}


# --- the frame ----------------------------------------------------------------------------------
def test_the_frame_is_the_49000_covered_words_in_the_frozen_order(frame):
    assert len(frame) == 49000
    assert all(set(row) == FRAME_FIELDS for row in frame)
    assert [(row["key"], row["position"]) for row in frame] == sorted(
        (row["key"], row["position"]) for row in frame)
    assert len({(row["key"], row["position"]) for row in frame}) == 49000


def test_the_frames_cells_are_the_committed_pools(frame):
    counts = Counter((row["stratum"], row["cell"]) for row in frame)
    s1 = {cell: words for (stratum, cell), words in counts.items() if stratum == "S1"}
    assert s1 == {f"{cell['cause']}:{'yes' if cell['base_flags'] else 'no'}": cell["words"]
                  for cell in CELLS["s1_primary_cells"]}
    assert {cell: words for cell, words in counts.items() if cell[0] != "S1"} == EXPECTED_FRAME


def test_only_s5_edge_words_are_served_whole_in_the_frame(frame):
    assert {(row["stratum"], row["cell"]) for row in frame if row["whole_played"]} == {
        ("S5", "leading"), ("S5", "trailing")}


# --- the gate -----------------------------------------------------------------------------------
def test_the_gate_compares_the_whole_population_with_the_committed_word_table(reproduction):
    assert reproduction["committed_words"] == reproduction["regenerated_words"] == 57935
    assert reproduction["flips"] >= len(reproduction["did_not_reproduce"]) >= 0
    assert 0 <= reproduction["flips_in_s1_cells"] <= len(reproduction["did_not_reproduce"])


def test_the_words_that_did_not_reproduce_are_frame_words_listed_in_order(reproduction, frame):
    listed = [(entry["item"], entry["word"]) for entry in reproduction["did_not_reproduce"]]
    assert listed == sorted(listed)
    assert set(listed) <= {(row["key"], row["word"]) for row in frame}


def test_each_cells_population_counts_only_the_words_that_reproduced(reproduction, frame):
    frame_counts = Counter((row["stratum"], row["cell"]) for row in frame)
    gone = Counter()
    by_word = {(row["key"], row["word"]): (row["stratum"], row["cell"]) for row in frame}
    for entry in reproduction["did_not_reproduce"]:
        gone[by_word[(entry["item"], entry["word"])]] += 1
    cells = {(entry["stratum"], entry["cell"]): entry for entry in reproduction["cells"]}
    assert set(cells) == set(frame_counts)
    for cell, entry in cells.items():
        assert entry["frame"] == frame_counts[cell]
        assert entry["did_not_reproduce"] == gone[cell]
        assert entry["population"] == entry["frame"] - entry["did_not_reproduce"]
    s1 = sum(entry["did_not_reproduce"] for (stratum, _), entry in cells.items()
             if stratum == "S1")
    assert reproduction["flips_in_s1_cells"] == s1


def test_the_regenerated_app_errors_feed_the_reference_suspect_rule(reproduction):
    rows = _lines(DRAW / "app_errors.jsonl.gz")
    assert all(set(row) == {"item", "group_key", "surah", "ayah", "word", "errors"}
               for row in rows)
    committed = _lines(AUDIT / "app_arm" / "word_grades.jsonl.gz")
    flagged = sum(1 for row in committed if row["strict"]["flagged"])
    assert abs(len(rows) - flagged) <= reproduction["flips"]
    assert len({(row["item"], row["word"]) for row in rows}) == len(rows)
    assert isinstance(reference_suspects(rows), set)


# --- the draw -----------------------------------------------------------------------------------
def test_the_key_holds_exactly_the_540_drawn_words_and_25_repeats(key):
    assert all(set(row) == KEY_FIELDS for row in key)
    originals = _originals(key)
    assert len(originals) == 540 and len(key) == 565
    assert len({row["id"] for row in key}) == 565
    assert Counter(row["stratum"] for row in originals.values()) == {
        "S1": 200, "S3": 250, "S4": 60, "S5": 30}


def test_the_drawn_words_are_the_seeded_permutations_prefixes(key, draws):
    expected = {item_id(SEED, row["key"], row["word"]): (cell, row["key"], row["word"])
                for cell, draw in draws.items() for row in draw.rows}
    originals = _originals(key)
    assert set(originals) == set(expected)
    for ident, row in originals.items():
        cell, item, word = expected[ident]
        assert ((row["stratum"], row["cell"]), row["item"], row["word"]) == (cell, item, word)


def test_the_pilot_flag_is_each_cells_prefix(key, draws):
    pilot = {item_id(SEED, row["key"], row["word"]) for draw in draws.values()
             for row in draw.pilot}
    originals = _originals(key)
    assert {ident for ident, row in originals.items() if row["in_pilot"]} == pilot
    assert Counter(row["stratum"] for row in originals.values() if row["in_pilot"]) == {
        "S1": 56, "S3": 69, "S4": 17, "S5": 8}


def test_the_inclusion_probability_is_the_cells_n_over_its_reproduced_population(key, draws):
    by_cell = {(draw.stratum, draw.cell): draw for draw in draws.values()}
    for row in _originals(key).values():
        draw = by_cell[(row["stratum"], row["cell"])]
        assert (row["n"], row["population"]) == (draw.n, draw.population)
        assert row["inclusion_probability"] == draw.n / draw.population


def test_the_cells_n_are_the_frozen_design(key):
    ns = {(row["stratum"], row["cell"]): row["n"] for row in _originals(key).values()}
    assert {cell: n for (stratum, cell), n in ns.items() if stratum == "S1"} == {
        f"{cell['cause']}:{'yes' if cell['base_flags'] else 'no'}": cell["n"]
        for cell in CELLS["s1_primary_cells"]}
    assert ns[("S3", "all")] == 250 and ns[("S4", "all")] == 60
    assert {ns[("S5", edge)] for edge in ("leading", "trailing", "interior")} == {10}


def test_a_drawn_word_never_comes_from_the_words_that_did_not_reproduce(key, reproduction):
    gone = {(entry["item"], entry["word"]) for entry in reproduction["did_not_reproduce"]}
    assert not gone & {(row["item"], row["word"]) for row in key}


def test_s5_edge_words_are_served_whole_and_interior_words_are_snips(key):
    for row in key:
        assert row["whole_played"] == (row["stratum"] == "S5" and row["cell"] != "interior")
        assert (row["edge"] is not None) == (row["stratum"] == "S5")


def test_the_repeats_re_ask_originals_and_carry_no_inclusion_probability(key):
    originals = _originals(key)
    repeats = [row for row in key if row["repeat_of"] is not None]
    assert len(repeats) == 25 == len({row["repeat_of"] for row in repeats})
    assert Counter(row["in_pilot"] for row in repeats) == {True: 7, False: 18}
    for row in repeats:
        original = originals[row["repeat_of"]]
        assert row["id"] == repeat_id(SEED, row["repeat_of"])
        assert row["inclusion_probability"] is None
        assert row["in_pilot"] == original["in_pilot"]
        assert {name: row[name] for name in KEY_FIELDS - {"id", "repeat_of",
                                                         "inclusion_probability"}} == {
            name: original[name] for name in KEY_FIELDS - {"id", "repeat_of",
                                                          "inclusion_probability"}}


# --- the blind worklist -------------------------------------------------------------------------
def test_the_worklist_carries_only_the_blind_fields_and_no_stratum_or_model(worklist):
    assert all(set(row) == set(WORKLIST_FIELDS) for row in worklist)
    text = (DRAW / "worklist.jsonl").read_text(encoding="utf-8")
    for leaked in ("stratum", "inclusion", "whole_played", "teacher", "student", "oracle",
                   "pilot", "repeat", "cell", "edge"):
        assert leaked not in text


def test_the_worklist_is_sorted_by_id_and_describes_the_keys_items(worklist, key):
    ids = [row["id"] for row in worklist]
    assert ids == sorted(ids) and len(set(ids)) == 565
    assert set(ids) == {row["id"] for row in key}
    assert all(len(ident) == 16 and ident == ident.lower() for ident in ids)


def test_a_repeats_worklist_line_is_its_originals_under_its_own_id(worklist, key):
    by_id = {row["id"]: row for row in worklist}
    for row in key:
        if row["repeat_of"] is not None:
            original = by_id[row["repeat_of"]]
            assert {name: value for name, value in by_id[row["id"]].items() if name != "id"} == {
                name: value for name, value in original.items() if name != "id"}


def test_every_worklist_line_names_its_keys_word(worklist, key):
    by_id = {row["id"]: row for row in key}
    for row in worklist:
        assert (row["surah_ayah"], row["word_index"]) == (
            by_id[row["id"]]["surah_ayah"], by_id[row["id"]]["word_index"])
        assert row["end_sample"] > 0


# --- the serve orders ---------------------------------------------------------------------------
def test_the_serve_orders_are_the_amendments_plans_over_the_pilot_and_the_main_items(key):
    originals = _originals(key)
    for name, sitting, repeats in (("pilot", "pilot", 7), ("main", "main", 18)):
        items = [ident for ident, row in originals.items()
                 if row["in_pilot"] == (sitting == "pilot")]
        plan = plan_sitting(items, seed=SEED, sitting=sitting, repeats=repeats)
        served = [repeat_id(SEED, entry.item) if entry.repeat else entry.item for entry in plan]
        written = _lines(DRAW / f"{name}.jsonl")
        assert [row["id"] for row in written] == served
        assert [row["position"] for row in written] == list(range(1, len(served) + 1))
    assert len(_lines(DRAW / "pilot.jsonl")) == 157 and len(_lines(DRAW / "main.jsonl")) == 408


def test_the_pilot_and_the_main_sitting_serve_every_item_exactly_once_between_them(key):
    served = [row["id"] for name in ("pilot", "main") for row in _lines(DRAW / f"{name}.jsonl")]
    assert sorted(served) == sorted(row["id"] for row in key)


# --- the snips ----------------------------------------------------------------------------------
def test_there_is_a_snip_for_every_worklist_item_and_edge_words_play_whole(worklist, key):
    snips = read_snips(DRAW / "snips.jsonl")
    assert set(snips) == {row["id"] for row in worklist}
    for row in key:
        if row["whole_played"]:
            assert snips[row["id"]].snip_source == WHOLE_CLIP


def test_the_snips_re_derive_from_the_committed_word_times(worklist, key):
    from tadabur.followalong_audit.draw import snips_for_draw

    directory = AUDIT.parent / "tlog_mining" / "word_times"
    records = [record for tranche in TRANCHES for record in _lines(directory / f"{tranche}.jsonl.gz")]
    assert snips_for_draw(worklist, key, records) == [
        snip for _, snip in sorted(read_snips(DRAW / "snips.jsonl").items())]


def test_the_snip_report_gives_the_fallback_rate_the_run_note_quotes(key):
    report = json.loads((DRAW / "snips.report.json").read_text(encoding="utf-8"))
    assert report["rows"] == 565 == report["alignment"] + report["whole_clip"]
    assert report["fallback_rate"] == report["whole_clip"] / report["rows"]
    assert report["whole_clip"] >= sum(1 for row in key if row["whole_played"])


# --- the files are tracked ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["worklist.jsonl", "key.jsonl", "pilot.jsonl", "main.jsonl",
                                  "snips.jsonl"])
def test_a_label_file_is_not_gitignored(name):
    ignored = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q",
         f"tools/tadabur/followalong_audit/s5_draw/{name}"]).returncode
    assert ignored == 1, f"{name} would be ignored by *.jsonl: add a ! entry to .gitignore"
