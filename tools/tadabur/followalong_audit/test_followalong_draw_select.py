"""Which words the draw picks: the frame's cells, the sizes, the pilot prefix and the gate (#311).

Committed before ``tadabur/followalong_audit/draw.py`` exists and **locked**: later turns may add
tests but never weaken or rewrite these. They pin ``docs/followalong-audit-preregistration.md``
§3 and ``docs/followalong-audit-draw-amendment.md`` §1-§3.

Contract this file pins (``tadabur.followalong_audit.draw``)
-----------------------------------------------------------
* ``frame_rows(stratified)``: :func:`strata_run.draw_frame`'s rows (same words, same ``(key,
  position)`` order) each with ``word``, ``word_index`` (copied from the stratified row) and a
  ``cell``: S1 ``"<primary cause>:<yes|no>"`` (``yes`` when the base detector flags it), S3 and S4
  ``"all"``, S5 its edge (``leading``/``trailing``/``interior``). Row keys are exactly ``key,
  position, word, word_index, stratum, cell, edge, whole_played``.
* ``design_totals(frame) -> {(stratum, cell): n}``: S1's 200 by :func:`allocation.allocate` (floor
  3) over its cells' word counts, S3 250, S4 60, S5 10 per bucket.
* ``pilot_split(totals) -> {(stratum, cell): pilot_n}``: each stratum's pilot (S1 56, S3 69, S4 17,
  S5 8) over its cells by :func:`allocation.allocate` with no floor.
* ``select_words(frame, *, seed, removed=frozenset(), totals=None, pilot=None) ->
  {(stratum, cell): CellDraw}``. ``removed`` is a set of ``(key, word)``. A cell's frame rows are
  sorted ascending by ``(key, position)`` and shuffled with
  ``seeded_order(..., purpose=f"cell:{stratum}:{cell}")``; removed words are skipped when the
  prefix is taken. ``CellDraw`` has ``stratum, cell, n, pilot_n, population, rows`` (the first
  ``n`` surviving frame rows, drawn order) and ``.pilot`` / ``.main`` (``rows[:pilot_n]`` /
  ``rows[pilot_n:]``) and ``.inclusion`` (``n / population``). ``population`` counts the words
  that survived. ``totals``/``pilot`` default to :func:`design_totals` / :func:`pilot_split`.
  A cell with fewer surviving words than ``n`` raises ``ValueError``.
* ``non_reproducing(committed, regenerated) -> [(item, word), ...]``: the sorted words whose
  regenerated ``word_grades`` row differs from the committed one in either scoring (``strict`` or
  ``shipped_default``: ``graded``, ``flagged`` or ``causes``). The two tables must hold the same
  ``(item, word)`` keys, else ``ValueError``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from tadabur.followalong_audit.allocation import allocate
from tadabur.followalong_audit.draw import (
    design_totals,
    frame_rows,
    non_reproducing,
    pilot_split,
    select_words,
    seeded_order,
    subseed,
)
from tadabur.followalong_audit.strata import stratify
from tadabur.followalong_audit.strata_run import draw_frame

AUDIT = Path(__file__).resolve().parent
CELLS = json.loads((AUDIT / "app_arm" / "cells.json").read_text(encoding="utf-8"))
SEED = 9665042067186208708
DETECTORS = ("base", "c95", "r2_candidate_ema", "r3_candidate_ema")
FIELDS = {"key", "position", "word", "word_index", "stratum", "cell", "edge", "whole_played"}
S5_EDGES = ("interior", "leading", "trailing")


def stratified_word(key, position, app, *, flagging=(), oracle="passed", causes=(),
                    covered_by_audio=True) -> dict:
    return {"key": key, "position": position, "word": position + 1, "word_index": position + 10,
            "app": app, "oracle": oracle,
            "detectors": {name: name in flagging for name in DETECTORS},
            "detector_graded": dict.fromkeys(DETECTORS, True), "causes": list(causes),
            "transcript_covered": covered_by_audio if app == "ungraded" else None}


def stratified() -> list[dict]:
    return stratify([
        stratified_word("a#0", 0, "ungraded"),                                      # S5 leading
        stratified_word("a#0", 1, "flagged", causes=["tashkeel", "shaddah"],        # S1 shaddah
                        flagging=["base"]),
        stratified_word("a#0", 2, "flagged", causes=["other_consonant", "soft_pair"]),  # S1 soft
        stratified_word("a#0", 3, "passed", flagging=["base"]),                     # S3
        stratified_word("a#0", 4, "passed"),                                        # S4
        stratified_word("a#0", 5, "ungraded"),                                      # S5 interior
        stratified_word("a#0", 6, "passed"),
        stratified_word("a#0", 7, "ungraded"),                                      # S5 trailing
        stratified_word("a#0", 8, "ungraded", covered_by_audio=False),              # not covered
        stratified_word("b#0", 0, "flagged", oracle=None, causes=["skipped"]),      # S1-pending
        stratified_word("b#0", 1, "flagged", oracle="flagged", causes=["tashkeel"]),  # explained
        stratified_word("b#0", 2, "passed", flagging=["c95"]),                      # residual
    ])


# --- the frame's cells --------------------------------------------------------------------------
def test_the_frame_rows_are_the_draw_frame_with_a_word_a_word_index_and_a_cell():
    rows = frame_rows(stratified())
    assert all(set(row) == FIELDS for row in rows)
    assert [(row["key"], row["position"]) for row in rows] == [
        (row["key"], row["position"]) for row in draw_frame(stratified())]
    expected = {(row["key"], row["position"]): (row["stratum"], row["edge"], row["whole_played"])
                for row in draw_frame(stratified())}
    assert {(row["key"], row["position"]): (row["stratum"], row["edge"], row["whole_played"])
            for row in rows} == expected


def test_a_frame_row_carries_the_stratified_words_own_word_and_word_index():
    row = next(row for row in frame_rows(stratified())
               if (row["key"], row["position"]) == ("a#0", 3))
    assert (row["word"], row["word_index"]) == (4, 13)


def test_an_s1_word_is_celled_by_its_primary_cause_and_the_base_detector():
    cells = {(row["key"], row["position"]): row["cell"] for row in frame_rows(stratified())
             if row["stratum"] == "S1"}
    # shaddah outranks tashkeel; soft_pair outranks other_consonant; base flags only a#0/1.
    assert cells == {("a#0", 1): "shaddah:yes", ("a#0", 2): "soft_pair:no",
                     ("b#0", 0): "skipped:no"}


def test_s3_and_s4_are_one_cell_each_and_s5_cells_by_its_edge():
    cells = {(row["key"], row["position"]): row["cell"] for row in frame_rows(stratified())
             if row["stratum"] != "S1"}
    assert cells == {("a#0", 0): "leading", ("a#0", 3): "all", ("a#0", 4): "all",
                     ("a#0", 5): "interior", ("a#0", 6): "all", ("a#0", 7): "trailing"}


# --- the sizes ----------------------------------------------------------------------------------
def frame_with(s1: dict[str, int], s3=300, s4=100, s5=20) -> list[dict]:
    """A synthetic frame: ``s1`` words per ``"<cause>:<yes|no>"`` cell and the other strata."""
    rows = []

    def add(stratum, cell, count):
        for number in range(count):
            key = f"{stratum}-{cell}#{number // 7}"
            rows.append({"key": key, "position": number % 7, "word": number % 7 + 1,
                         "word_index": number % 7, "stratum": stratum, "cell": cell,
                         "edge": cell if stratum == "S5" else None,
                         "whole_played": stratum == "S5" and cell != "interior"})

    for cell, count in s1.items():
        add("S1", cell, count)
    add("S3", "all", s3)
    add("S4", "all", s4)
    for edge in S5_EDGES:
        add("S5", edge, s5)
    return rows


def committed_s1() -> dict[str, int]:
    return {f"{cell['cause']}:{'yes' if cell['base_flags'] else 'no'}": cell["words"]
            for cell in CELLS["s1_primary_cells"]}


def test_the_design_totals_are_the_frozen_540_with_s1_from_the_committed_cells():
    totals = design_totals(frame_with(committed_s1()))
    assert {cell: n for (stratum, cell), n in totals.items() if stratum == "S1"} == {
        f"{cell['cause']}:{'yes' if cell['base_flags'] else 'no'}": cell["n"]
        for cell in CELLS["s1_primary_cells"]}
    assert totals[("S3", "all")] == 250 and totals[("S4", "all")] == 60
    assert {totals[("S5", edge)] for edge in S5_EDGES} == {10}
    assert sum(totals.values()) == 540


def test_the_design_totals_are_allocate_with_a_floor_of_three_over_the_s1_cells():
    counts = committed_s1()
    expected = allocate(counts, 200, floor=3)
    totals = design_totals(frame_with(counts))
    assert {cell: totals[("S1", cell)] for cell in counts} == expected


def test_the_pilot_split_reproduces_the_frozen_56_69_17_8_and_the_amendments_ties():
    totals = design_totals(frame_with(committed_s1()))
    pilot = pilot_split(totals)
    by_stratum = {stratum: sum(n for (name, _), n in pilot.items() if name == stratum)
                  for stratum in ("S1", "S3", "S4", "S5")}
    assert by_stratum == {"S1": 56, "S3": 69, "S4": 17, "S5": 8}
    assert {edge: pilot[("S5", edge)] for edge in S5_EDGES} == {
        "interior": 3, "leading": 3, "trailing": 2}
    assert {cell: n for (stratum, cell), n in pilot.items() if stratum == "S1"} == {
        "other_consonant:no": 12, "other_consonant:yes": 17,
        "tashkeel:no": 3, "tashkeel:yes": 5,
        "soft_pair:no": 1, "soft_pair:yes": 4,
        "skipped:no": 1, "skipped:yes": 3,
        "unrecorded:no": 1, "unrecorded:yes": 4,
        "shaddah:no": 1, "shaddah:yes": 2,
        "target_pair:no": 1, "target_pair:yes": 1}


def test_a_pilot_never_exceeds_its_cells_total():
    totals = design_totals(frame_with(committed_s1()))
    pilot = pilot_split(totals)
    assert all(pilot[cell] <= totals[cell] for cell in totals)


# --- the permutation, the prefix and the removals -----------------------------------------------
def small_frame() -> list[dict]:
    return [{"key": f"k{number // 3}#0", "position": number % 3, "word": number % 3 + 1,
             "word_index": number % 3, "stratum": "S3", "cell": "all", "edge": None,
             "whole_played": False} for number in range(40)]


def test_a_cell_is_drawn_as_the_seeded_permutation_of_its_sorted_frame_words():
    frame = small_frame()
    draw = select_words(list(reversed(frame)), seed=SEED, totals={("S3", "all"): 12},
                        pilot={("S3", "all"): 5})[("S3", "all")]
    ordered = sorted(frame, key=lambda row: (row["key"], row["position"]))
    random.Random(subseed(SEED, "cell:S3:all")).shuffle(ordered)
    assert draw.rows == ordered[:12]
    assert (draw.stratum, draw.cell, draw.n, draw.pilot_n) == ("S3", "all", 12, 5)


def test_the_pilot_is_the_permutations_prefix_and_the_main_sitting_continues_it():
    draw = select_words(small_frame(), seed=SEED, totals={("S3", "all"): 12},
                        pilot={("S3", "all"): 5})[("S3", "all")]
    assert draw.pilot == draw.rows[:5] and draw.main == draw.rows[5:] and len(draw.main) == 7
    bigger = select_words(small_frame(), seed=SEED, totals={("S3", "all"): 20},
                          pilot={("S3", "all"): 5})[("S3", "all")]
    assert bigger.rows[:12] == draw.rows and bigger.pilot == draw.pilot


def test_the_inclusion_probability_is_n_over_the_frame_without_removals():
    draw = select_words(small_frame(), seed=SEED, totals={("S3", "all"): 12},
                        pilot={("S3", "all"): 5})[("S3", "all")]
    assert draw.population == 40 and draw.inclusion == 12 / 40


def test_a_removed_word_is_skipped_in_the_frozen_order_and_lowers_the_population():
    frame = small_frame()
    plain = select_words(frame, seed=SEED, totals={("S3", "all"): 12},
                         pilot={("S3", "all"): 5})[("S3", "all")]
    first, fifth = plain.rows[0], plain.rows[4]
    removed = {(first["key"], first["word"]), (fifth["key"], fifth["word"])}
    draw = select_words(frame, seed=SEED, removed=removed, totals={("S3", "all"): 12},
                        pilot={("S3", "all"): 5})[("S3", "all")]
    ordered = sorted(frame, key=lambda row: (row["key"], row["position"]))
    random.Random(subseed(SEED, "cell:S3:all")).shuffle(ordered)
    survivors = [row for row in ordered if (row["key"], row["word"]) not in removed]
    assert draw.rows == survivors[:12]
    assert not removed & {(row["key"], row["word"]) for row in draw.rows}
    assert draw.population == 38 and draw.inclusion == 12 / 38
    assert len(draw.pilot) == 5


def test_a_removal_in_one_cell_leaves_every_other_cell_untouched():
    frame = small_frame() + [{**row, "key": "s4-" + row["key"], "stratum": "S4"}
                                  for row in small_frame()]
    totals = {("S3", "all"): 12, ("S4", "all"): 12}
    pilot = {("S3", "all"): 5, ("S4", "all"): 5}
    plain = select_words(frame, seed=SEED, totals=totals, pilot=pilot)
    gone = plain[("S3", "all")].rows[0]
    draw = select_words(frame, seed=SEED, removed={(gone["key"], gone["word"])}, totals=totals,
                        pilot=pilot)
    assert draw[("S4", "all")].rows == plain[("S4", "all")].rows
    assert draw[("S4", "all")].population == 40


def test_a_cell_with_fewer_surviving_words_than_its_n_is_refused():
    frame = small_frame()
    removed = {(row["key"], row["word"]) for row in frame[:30]}
    with pytest.raises(ValueError):
        select_words(frame, seed=SEED, removed=removed, totals={("S3", "all"): 12},
                     pilot={("S3", "all"): 5})


def test_the_draw_defaults_to_the_frozen_design_and_is_reproducible():
    frame = frame_with(committed_s1())
    first, second = select_words(frame, seed=SEED), select_words(list(reversed(frame)), seed=SEED)
    assert first == second
    assert sum(len(draw.rows) for draw in first.values()) == 540
    assert sum(draw.pilot_n for draw in first.values()) == 150
    assert sum(len(draw.pilot) for draw in first.values()) == 150
    assert len({(row["key"], row["position"]) for draw in first.values()
                for row in draw.rows}) == 540
    assert select_words(frame, seed=SEED + 1) != first


def test_every_cells_words_come_from_that_cell_and_stratum():
    for (stratum, cell), draw in select_words(frame_with(committed_s1()), seed=SEED).items():
        assert all((row["stratum"], row["cell"]) == (stratum, cell) for row in draw.rows)


# --- the reproduction gate ----------------------------------------------------------------------
def grade_row(item, word, *, strict=(False, False, ()), shipped=(False, False, ())) -> dict:
    def cell(values):
        graded, flagged, causes = values
        return {"graded": graded, "flagged": flagged, "causes": list(causes)}

    return {"item": item, "word": word, "strict": cell(strict), "shipped_default": cell(shipped)}


def table() -> list[dict]:
    return [grade_row("a#0", 1, strict=(True, False, ()), shipped=(True, False, ())),
            grade_row("a#0", 2, strict=(True, True, ("tashkeel",)),
                      shipped=(True, True, ("tashkeel",))),
            grade_row("b#0", 1)]


def test_an_identical_regeneration_reproduces_every_word():
    assert non_reproducing(table(), [dict(row) for row in table()]) == []


def test_a_word_whose_strict_flag_changed_does_not_reproduce():
    regenerated = table()
    regenerated[0] = grade_row("a#0", 1, strict=(True, True, ("skipped",)),
                               shipped=(True, False, ()))
    assert non_reproducing(table(), regenerated) == [("a#0", 1)]


def test_a_word_that_became_graded_or_lost_a_cause_does_not_reproduce():
    regenerated = table()
    regenerated[2] = grade_row("b#0", 1, strict=(True, False, ()))
    regenerated[1] = grade_row("a#0", 2, strict=(True, True, ()), shipped=(True, True, ("tashkeel",)))
    assert non_reproducing(table(), regenerated) == [("a#0", 2), ("b#0", 1)]


def test_a_difference_in_the_balanced_scoring_alone_is_still_a_flip():
    regenerated = table()
    regenerated[1] = grade_row("a#0", 2, strict=(True, True, ("tashkeel",)),
                               shipped=(True, False, ()))
    assert non_reproducing(table(), regenerated) == [("a#0", 2)]


def test_the_flips_are_sorted_whatever_order_the_tables_come_in():
    regenerated = [grade_row("b#0", 1, strict=(True, True, ("skipped",))),
                   grade_row("a#0", 1, strict=(True, True, ("skipped",))), table()[1]]
    flips = non_reproducing(list(reversed(table())), regenerated)
    assert flips == sorted(flips) == [("a#0", 1), ("b#0", 1)]


def test_a_regeneration_missing_or_adding_a_word_is_refused_not_ignored():
    with pytest.raises(ValueError):
        non_reproducing(table(), table()[:2])
    with pytest.raises(ValueError):
        non_reproducing(table()[:2], table())


def test_the_seeded_order_helper_is_what_the_selection_uses():
    # the selection and the public helper agree on one cell, so there is a single idiom
    frame = small_frame()
    draw = select_words(frame, seed=SEED, totals={("S3", "all"): 40},
                        pilot={("S3", "all"): 0})[("S3", "all")]
    assert draw.rows == seeded_order(frame, SEED, "cell:S3:all",
                                     key=lambda row: (row["key"], row["position"]))
