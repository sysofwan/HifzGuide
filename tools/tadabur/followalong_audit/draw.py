"""The S5 draw (#311): the seed, the cells, the pilot prefixes, the blind ids and the snips.

The frozen design is ``docs/followalong-audit-preregistration.md`` §3 and its amendment
``docs/followalong-audit-draw-amendment.md`` §1–§4. This module holds only the pure steps: the
seed formula, the cell arithmetic, the seeded permutation, the reproduction gate and the sitting
plans. The committed tables in ``s5_draw/`` are written beside it (``build.py``).

Every random choice is one ``random.Random(subseed(...))`` over items sorted first, so a draw
re-derives from the amendment alone and never from the order a caller happened to read. The seed
is the first eight bytes, big-endian, of ``sha256("followalong-audit-draw:" + M)``, where M is
the first commit on ``main`` that adds the amendment; every use hashes the decimal seed and its
own purpose name.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from tadabur.followalong_audit.allocation import (
    S1_CELL_FLOOR, S1_SNIPS, allocate, primary_cause,
)
from tadabur.followalong_audit.strata import S1, S3, S4, S5
from tadabur.followalong_audit.strata_run import draw_frame
from tadabur.word_snips import MIN_LL_PER_FRAME, Snip, SnipPlacement, placements

#: M: PR #326's squash commit, the first commit on ``main`` that adds the draw amendment.
AMENDMENT_COMMIT = "03bc6d071a37a9bb264fcd6513381dbedee1c96e"

#: The pre-registered sizes the draw allocates (S1's 200 is split over its 14 cells, floor 3).
STRATUM_SIZES = {S3: 250, S4: 60}
#: S5's snips, one cell per edge class.
S5_SNIPS_PER_EDGE = 10
#: The pilot's size per stratum, allocated over that stratum's cells by the same rule.
PILOT_SIZES = {S1: 56, S3: 69, S4: 17, S5: 8}
#: The repeats: 7 in the pilot, 18 in the main sitting (amendment §4).
PILOT_REPEATS = 7
MAIN_REPEATS = 18
#: A repeat sits at least this many served places after its original (amendment §4).
REPEAT_GAP = 20

S5_EDGES = ("leading", "trailing", "interior")
#: The snip report's cause for an S5 edge word, which the draw serves whole by rule (§3).
EDGE_WHOLE_REASON = "the S5 edge word plays whole"
SITTINGS = ("pilot", "main")
_SCORINGS = ("strict", "shipped_default")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")


def _prefix(text: str) -> int:
    """The first eight bytes of ``sha256(text)``, big-endian, as an unsigned integer."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def require_commit_hex(commit_hex: str) -> str:
    """The commit's 40 lowercase hex characters, refusing anything else."""
    if not _HEX40.match(commit_hex):
        raise ValueError(f"{commit_hex!r} is not a 40-character lowercase commit SHA")
    return commit_hex


def derive_seed(commit_hex: str) -> int:
    """The draw's seed for the amendment commit M (amendment §1)."""
    return _prefix(f"followalong-audit-draw:{require_commit_hex(commit_hex)}")


def subseed(seed: int, purpose: str) -> int:
    """One use's seed: ``sha256(str(seed) + ":" + purpose)``'s first eight bytes, big-endian."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"the seed must be an unsigned integer, not {seed!r}")
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("a sub-seed needs a purpose name")
    return _prefix(f"{seed}:{purpose}")


def seeded_order(items: Iterable, seed: int, purpose: str, *, key=None) -> list:
    """``items`` sorted ascending (by ``key`` when given), then shuffled on the sub-seed.

    The sort makes the order a function of the set and the seed only; the input is left alone.
    """
    ordered = sorted(items, key=key) if key is not None else sorted(items)
    random.Random(subseed(seed, purpose)).shuffle(ordered)
    return ordered


# --- the frame's cells ---------------------------------------------------------------------------

def frame_rows(stratified: Sequence[Mapping]) -> list[dict]:
    """The draw frame with each row's word, word index and cell.

    The frame is :func:`strata_run.draw_frame`'s — the covered words the design samples — and
    the cell is S1's ``<primary cause>:<yes|no>`` (the base detector's flag), S3's and S4's
    ``all``, and S5's edge class. ``word`` and ``word_index`` are the stratified word's own.
    """
    by_position = {(row["key"], row["position"]): row for row in stratified}
    rows = []
    for frame in draw_frame(stratified):
        word = by_position[(frame["key"], frame["position"])]
        rows.append({
            "key": frame["key"], "position": frame["position"],
            "word": word["word"], "word_index": word["word_index"],
            "stratum": frame["stratum"], "cell": _cell(frame, word),
            "edge": frame["edge"], "whole_played": frame["whole_played"],
        })
    return rows


def _cell(frame: Mapping, word: Mapping) -> str:
    """The cell a frame row belongs to."""
    if frame["stratum"] == S1:
        return f"{primary_cause(word['causes'])}:{'yes' if word['detectors']['base'] else 'no'}"
    if frame["stratum"] == S5:
        return frame["edge"]
    return "all"


def _cells_by_stratum(frame: Iterable[Mapping]) -> dict[str, dict[str, int]]:
    """Each stratum's cell sizes in the frame."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in frame:
        counts[row["stratum"]][row["cell"]] += 1
    return counts


def design_totals(frame: Sequence[Mapping]) -> dict[tuple[str, str], int]:
    """The frozen design's snips per ``(stratum, cell)``.

    S1's 200 are :func:`allocation.allocate`'s split with the pre-registered floor of 3, which
    reproduces the 14 tabulated cells; S3 takes 250, S4 60, and S5 10 per edge class.
    """
    counts = _cells_by_stratum(frame)
    s1 = counts.get(S1, {})
    if not s1:
        raise ValueError("the frame holds no S1 words: its 200 snips have no cells")
    totals = {(S1, cell): n for cell, n in allocate(s1, S1_SNIPS, floor=S1_CELL_FLOOR).items()}
    totals.update({(stratum, "all"): size for stratum, size in STRATUM_SIZES.items()})
    totals.update({(S5, edge): S5_SNIPS_PER_EDGE for edge in S5_EDGES})
    return totals


def pilot_split(totals: Mapping[tuple[str, str], int]) -> dict[tuple[str, str], int]:
    """The pilot's size per cell: each stratum's pilot over its cells, no floor."""
    pilot: dict[tuple[str, str], int] = {}
    for stratum, size in PILOT_SIZES.items():
        cells = {cell: n for (name, cell), n in totals.items() if name == stratum}
        if not cells:
            continue
        pilot.update({(stratum, cell): n for cell, n in allocate(cells, size).items()})
    return pilot


# --- the draw ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CellDraw:
    """One cell's draw: its size, its prefix, its reproduced population and the drawn rows."""

    stratum: str
    cell: str
    n: int
    pilot_n: int
    population: int
    rows: list[dict]

    @property
    def pilot(self) -> list[dict]:
        return self.rows[:self.pilot_n]

    @property
    def main(self) -> list[dict]:
        return self.rows[self.pilot_n:]

    @property
    def inclusion(self) -> float:
        """The cell's inclusion probability: its ``n`` over its reproduced population."""
        return self.n / self.population


def select_words(frame: Sequence[Mapping], *, seed: int, removed: Iterable[tuple] = frozenset(),
                 totals: Mapping[tuple[str, str], int] | None = None,
                 pilot: Mapping[tuple[str, str], int] | None = None) -> dict[tuple[str, str],
                                                                              CellDraw]:
    """Every cell's permutation, its ``n`` prefix and its pilot prefix (amendment §1–§3).

    A cell's frame rows are sorted by ``(key, position)`` and shuffled on
    ``cell:<stratum>:<cell>``; words in ``removed`` (``(key, word)``, the gate's did-not-reproduce
    pool) are skipped, so ``population`` counts only the words that reproduced and ``inclusion``
    is ``n / population``. A cell with fewer surviving words than its ``n`` is refused.
    """
    totals = design_totals(frame) if totals is None else dict(totals)
    pilot = pilot_split(totals) if pilot is None else dict(pilot)
    unknown = sorted(set(pilot) - set(totals))
    if unknown:
        raise ValueError(f"pilot cells with no design total: {unknown}")
    by_cell: dict[tuple[str, str], list] = defaultdict(list)
    for row in frame:
        by_cell[(row["stratum"], row["cell"])].append(row)
    removed = set(removed)

    draws: dict[tuple[str, str], CellDraw] = {}
    for (stratum, cell), n in totals.items():
        ordered = seeded_order(by_cell.get((stratum, cell), []), seed,
                               f"cell:{stratum}:{cell}",
                               key=lambda row: (row["key"], row["position"]))
        survivors = [row for row in ordered if (row["key"], row["word"]) not in removed]
        if len(survivors) < n:
            raise ValueError(f"{stratum}/{cell} has {len(survivors)} reproduced words, below "
                             f"its {n} snips")
        if pilot.get((stratum, cell), 0) > n:
            raise ValueError(f"{stratum}/{cell}'s pilot {pilot[(stratum, cell)]} exceeds its {n}")
        draws[(stratum, cell)] = CellDraw(stratum, cell, n, pilot.get((stratum, cell), 0),
                                          len(survivors), survivors[:n])
    return draws


def non_reproducing(committed: Sequence[Mapping],
                    regenerated: Sequence[Mapping]) -> list[tuple[str, int]]:
    """The words the regenerated App table did not reproduce, sorted (amendment §2).

    A word reproduces when both scorings agree on ``graded``, ``flagged`` and the cause set. The
    two tables must hold the same ``(item, word)`` keys: a gate over a different population would
    otherwise call every missing word a flip and every extra one a clean word.
    """
    committed_by = _by_word(committed, "committed")
    regenerated_by = _by_word(regenerated, "regenerated")
    if set(committed_by) != set(regenerated_by):
        missing = sorted(set(committed_by) - set(regenerated_by))
        extra = sorted(set(regenerated_by) - set(committed_by))
        raise ValueError(f"the regenerated table is another population: {len(missing)} words "
                         f"missing ({missing[:3]}), {len(extra)} extra ({extra[:3]})")
    return sorted(word for word, row in committed_by.items()
                  if any(_outcome(row[scoring]) != _outcome(regenerated_by[word][scoring])
                         for scoring in _SCORINGS))


def _by_word(rows: Iterable[Mapping], where: str) -> dict[tuple[str, int], Mapping]:
    by_word: dict[tuple[str, int], Mapping] = {}
    for row in rows:
        key = (row["item"], row["word"])
        if key in by_word:
            raise ValueError(f"the {where} table repeats {key}")
        by_word[key] = row
    return by_word


def _outcome(cell: Mapping) -> tuple:
    """One scoring's outcome for one word, order-insensitive in its causes."""
    return bool(cell["graded"]), bool(cell["flagged"]), tuple(sorted(cell["causes"]))


# --- the blind ids and the sitting plans ---------------------------------------------------------

def item_id(seed: int, key: str, word: int) -> str:
    """The page's opaque id for a word: 16 lowercase hex, a salted hash of its clip and word."""
    if not isinstance(key, str) or not key:
        raise ValueError("a word's item id needs its clip key")
    if not isinstance(word, int) or isinstance(word, bool) or word < 1:
        raise ValueError(f"{key}: a word is a positive integer, not {word!r}")
    return hashlib.sha256(f"{subseed(seed, 'ids')}:{key}:{word}".encode("utf-8")).hexdigest()[:16]


def repeat_id(seed: int, original: str) -> str:
    """The id a repeat of ``original`` is served under: 16 lowercase hex, never the original."""
    if not _HEX16.match(original):
        raise ValueError(f"{original!r} is not a 16-character lowercase item id")
    salt = subseed(seed, "ids")
    attempt = 0
    while True:
        candidate = hashlib.sha256(
            f"{salt}:repeat:{attempt}:{original}".encode("utf-8")).hexdigest()[:16]
        if candidate != original:
            return candidate
        attempt += 1


@dataclass(frozen=True)
class Served:
    """One served item: the original's id, and whether this line is its repeat."""

    item: str
    repeat: bool


def plan_sitting(items: Iterable[str], *, seed: int, sitting: str,
                 repeats: int) -> list[Served]:
    """One sitting's served order with its repeats (amendment §4).

    The originals appear in ``seeded_order(items, seed, "order:<sitting>")``; ``repeats``
    distinct originals drawn from the first ``len(items) - 20`` positions are served a second
    time, each at a seeded later position at least 20 served places after its original. The
    choices come from ``repeats:<sitting>``.
    """
    if sitting not in SITTINGS:
        raise ValueError(f"{sitting!r} is not a sitting: {SITTINGS}")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 0:
        raise ValueError(f"a sitting cannot serve {repeats!r} repeats")
    items = list(items)
    if len(set(items)) != len(items):
        raise ValueError("a sitting's items repeat an id")
    eligible = len(items) - REPEAT_GAP
    if repeats > max(eligible, 0):
        raise ValueError(f"{len(items)} items leave {max(eligible, 0)} originals with room for "
                         f"a repeat, not {repeats}")

    originals = seeded_order(items, seed, f"order:{sitting}")
    rng = random.Random(subseed(seed, f"repeats:{sitting}"))
    chosen = [originals[index] for index in sorted(rng.sample(range(eligible), repeats))]
    plan = [Served(item, False) for item in originals]
    for item in chosen:
        original = next(served for served in plan if served.item == item and not served.repeat)
        place = rng.randint(_place(plan, original) + REPEAT_GAP, len(plan))
        plan.insert(place, Served(item, True))
    return plan


def _place(plan: Sequence[Served], original: Served) -> int:
    """The original's position in the plan being built (each object is created once)."""
    return next(index for index, served in enumerate(plan) if served is original)


# --- the snips -----------------------------------------------------------------------------------

def snips_for_draw(worklist: Sequence[Mapping], key: Sequence[Mapping],
                   records: Iterable[Mapping], *,
                   min_ll_per_frame: float = MIN_LL_PER_FRAME) -> list[Snip]:
    """One snip per worklist row, with S5's edge words served whole (amendment §3, #264's rule).

    The snips of :func:`snip_placements`, which also says why a row plays whole.
    """
    return [placement.snip
            for placement in snip_placements(worklist, key, records,
                                             min_ll_per_frame=min_ll_per_frame)]


def snip_placements(worklist: Sequence[Mapping], key: Sequence[Mapping],
                    records: Iterable[Mapping], *,
                    min_ll_per_frame: float = MIN_LL_PER_FRAME) -> list[SnipPlacement]:
    """Every worklist row's placement, sorted by id, and for a whole-clip row its cause.

    ``word_snips.placements`` places every word it can; a row whose key says ``whole_played``
    is a whole-clip snip whatever its alignment, under :data:`EDGE_WHOLE_REASON`. The blind
    file and the key must describe the same items, and a repeat (its own id, its original's
    clip and word) gets the same excerpt.
    """
    whole = {}
    for row in key:
        if row["id"] in whole:
            raise ValueError(f"the key repeats {row['id']}")
        whole[row["id"]] = bool(row["whole_played"])
    ids = {row["id"] for row in worklist}
    missing, extra = sorted(ids - set(whole)), sorted(set(whole) - ids)
    if missing or extra:
        raise ValueError(f"the worklist and the key describe different items: {len(missing)} "
                         f"without a key row ({missing[:3]}), {len(extra)} with no worklist row")
    return sorted(placements(worklist, records, min_ll_per_frame,
                             forced_whole={site_id: EDGE_WHOLE_REASON
                                           for site_id, edge in whole.items() if edge}),
                  key=lambda placement: placement.snip.site_id)
