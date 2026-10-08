"""The numbers of the haraka-gap diagnosis (#85): rates, paired differences, class
contributions and the gap arithmetic, with reciter-clustered intervals.

Every quantity is a ratio of sums over :class:`~training.haraka_gap_arms.Observation`
cells, or a signed sum of such ratios plus a constant (:class:`Statistic`), so the point
estimate and every bootstrap replicate are computed by the same arithmetic. The cells are
summed per reciter once (:class:`Ledger`); a replicate is a reciter resample's counts
times those sums. Weighted cells carry 1 / inclusion probability (times a window share);
unweighted cells carry the share alone, so their totals are site counts.

A **contribution** restricts the numerator of a paired arm-minus-segment difference to
some classes but keeps the arm's whole denominator, so the contributions of a partition
of an arm's classes add up to its difference exactly. That is what lets the report say how
much of a difference each class carries without any residual.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from training.haraka_gap_arms import (
    ALL_HARAKAT,
    ARMS,
    BLOCKS,
    EDGE_CLASSES,
    EMPTY,
    HARAKAT,
    MATCHED,
    OUTCOME_GROUPS,
    SEAM_CLASSES,
    SEGMENT,
    SEGMENT_ALL,
    STREAM,
    STREAM_CLASSES,
    WINDOW,
    WINDOW_CLASSES,
    WINDOW_OCC,
    WINDOWED,
    WORD_IN_NO_WINDOW,
    WORD_POSITIONS,
    JointKey,
    Observation,
)

#: The two numbers this report explains: ADR-0005's full-corpus ``matched`` rate on whole
#: segments and ADR-0007's base-teacher recall on held-out windows.
ADR0005_SEGMENT_RECALL = 0.9617
ADR0007_WINDOW_RECALL = 0.841

#: Reciter-clustered percentile bootstrap, with ``docs/acceptance-rules.md`` §1's B, seed
#: and percentile convention (NumPy's default linear interpolation).
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261008

# --- Statistics: ratios of per-reciter sums, bootstrapped by reciter -----------------------

Predicate = Callable[[JointKey], bool]


def cells(model: str, arm: str, haraka: str = ALL_HARAKAT,
          classes: frozenset[str] | None = None, **fixed: str) -> Predicate:
    """Joint cells of one model and arm, optionally one haraka, some classes, and fixed
    values of other :class:`JointKey` fields (``segment``, ``outcome``, ``word_position``)."""
    def predicate(key: JointKey) -> bool:
        return (
            key.model == model
            and key.arm == arm
            and (haraka == ALL_HARAKAT or key.haraka == haraka)
            and (classes is None or key.cls in classes)
            and all(getattr(key, name) == value for name, value in fixed.items())
        )

    return predicate


@dataclass(frozen=True)
class Term:
    """``sign * sum(coefficient * numerator cells) / sum(denominator cells)``."""

    sign: float
    numerator: tuple[tuple[Predicate, float], ...]
    denominator: Predicate


@dataclass(frozen=True)
class Statistic:
    """``constant + sum of terms``: a rate, a paired difference, or a gap term."""

    terms: tuple[Term, ...]
    constant: float = 0.0

    def __neg__(self) -> "Statistic":
        return Statistic(
            tuple(Term(-t.sign, t.numerator, t.denominator) for t in self.terms), -self.constant
        )

    def __sub__(self, other: "Statistic") -> "Statistic":
        negated = -other
        return Statistic(self.terms + negated.terms, self.constant + negated.constant)


def constant(value: float) -> Statistic:
    return Statistic((), value)


def rate(model: str, arm: str, haraka: str, group: str,
         classes: frozenset[str] | None = None, column: str = "outcome",
         **fixed: str) -> Statistic:
    """Share of the arm's (restricted) weight whose ``column`` group is ``group``:
    ``column="segment"`` gives the whole-segment rate on the same occurrences."""
    return Statistic((Term(
        1.0,
        ((cells(model, arm, haraka, classes, **{column: group}, **fixed), 1.0),),
        cells(model, arm, haraka, classes, **fixed),
    ),))


def contribution(model: str, arm: str, haraka: str, group: str,
                 classes: frozenset[str] | None = None, **fixed: str) -> Statistic:
    """``classes``' part of the arm-minus-segment difference in ``group``'s rate.

    The numerator is restricted to ``classes``; the denominator is the arm's whole weight,
    so the contributions of a partition of the classes add up exactly to the paired
    difference (``classes=None``), which is itself one.
    """
    return Statistic((Term(
        1.0,
        (
            (cells(model, arm, haraka, classes, outcome=group, **fixed), 1.0),
            (cells(model, arm, haraka, classes, segment=group, **fixed), -1.0),
        ),
        cells(model, arm, haraka),
    ),))


class Ledger:
    """Observations summed per reciter and joint cell, weighted and unweighted."""

    def __init__(self, observations: Iterable[Observation]) -> None:
        sums: dict[tuple[int, JointKey], list[float]] = defaultdict(lambda: [0.0, 0.0])
        for obs in observations:
            cell = sums[(obs.reciter_id, obs.key)]
            cell[0] += obs.weight
            cell[1] += obs.share
        self.reciters = sorted({reciter for reciter, _ in sums})
        self.keys = sorted({key for _, key in sums})
        r_index = {r: i for i, r in enumerate(self.reciters)}
        k_index = {k: i for i, k in enumerate(self.keys)}
        self.weighted = np.zeros((len(self.reciters), len(self.keys)))
        self.unweighted = np.zeros_like(self.weighted)
        for (reciter, key), (w, u) in sums.items():
            self.weighted[r_index[reciter], k_index[key]] = w
            self.unweighted[r_index[reciter], k_index[key]] = u

    def _mask(self, predicate: Predicate) -> np.ndarray:
        return np.fromiter((predicate(key) for key in self.keys), dtype=float, count=len(self.keys))

    def total(self, predicate: Predicate, weighted: bool = True) -> float:
        matrix = self.weighted if weighted else self.unweighted
        return float(matrix.sum(axis=0) @ self._mask(predicate))

    def _per_reciter(self, statistic: Statistic, matrix: np.ndarray):
        """Each term's sign and per-reciter numerator and denominator sums."""
        for term in statistic.terms:
            coefficients = sum(c * self._mask(p) for p, c in term.numerator)
            yield term.sign, matrix @ coefficients, matrix @ self._mask(term.denominator)

    def value(self, statistic: Statistic, weighted: bool = True) -> float | None:
        """The statistic on the full sample; ``None`` if a denominator is zero."""
        total = statistic.constant
        for sign, numerator, denominator in self._per_reciter(
            statistic, self.weighted if weighted else self.unweighted
        ):
            if denominator.sum() == 0:
                return None
            total += sign * numerator.sum() / denominator.sum()
        return total

    def intervals(self, statistics: list[Statistic]) -> list[tuple[float, float] | None]:
        """Weighted 95% percentile intervals, every statistic on the same reciter draws.

        A statistic with an undefined replicate (a zero denominator) gets ``None``.
        """
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        n = len(self.reciters)
        draws = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
        counts = np.zeros((BOOTSTRAP_DRAWS, n))
        np.add.at(counts, (np.arange(BOOTSTRAP_DRAWS)[:, None], draws), 1.0)
        result = []
        for statistic in statistics:
            replicate = np.full(BOOTSTRAP_DRAWS, statistic.constant)
            with np.errstate(divide="ignore", invalid="ignore"):
                for sign, numerator, denominator in self._per_reciter(statistic, self.weighted):
                    replicate += sign * (counts @ numerator) / (counts @ denominator)
            if np.all(np.isfinite(replicate)):
                low, high = np.percentile(replicate, [2.5, 97.5])
                result.append((float(low), float(high)))
            else:
                result.append(None)
        return result


# --- The report -------------------------------------------------------------------------


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 5)


class _Collector:
    """Evaluates statistics into report cells, then fills every interval in one bootstrap."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self._pending: list[tuple[dict, Statistic]] = []

    def put(self, target: dict, name: str, statistic: Statistic) -> None:
        target[name] = {
            "weighted": _round(self.ledger.value(statistic)),
            "unweighted": _round(self.ledger.value(statistic, weighted=False)),
        }
        self._pending.append((target[name], statistic))

    def count(self, predicate: Predicate) -> float:
        return round(self.ledger.total(predicate, weighted=False), 2)

    def resolve(self) -> None:
        intervals = self.ledger.intervals([statistic for _, statistic in self._pending])
        for (cell, _), interval in zip(self._pending, intervals):
            cell["ci95"] = None if interval is None else [_round(v) for v in interval]


def discordance(collector: _Collector, model: str, arm: str, haraka: str, group: str,
                classes: frozenset[str] | None = None, **fixed: str) -> dict:
    """Unweighted counts behind a paired difference: occurrences (window shares count
    fractionally) in ``group`` on the whole segment only, in this arm only, and the net.
    A net under one is less than a single site, whatever the weighted difference says."""
    segment_only = sum(collector.count(cells(model, arm, haraka, classes, segment=group,
                                             outcome=other, **fixed))
                       for other in OUTCOME_GROUPS if other != group)
    arm_only = sum(collector.count(cells(model, arm, haraka, classes, segment=other,
                                         outcome=group, **fixed))
                   for other in OUTCOME_GROUPS if other != group)
    return {
        "segment_only": round(segment_only, 2),
        "arm_only": round(arm_only, 2),
        "net": round(arm_only - segment_only, 2),
        "at_least_one_site": abs(arm_only - segment_only) >= 1,
    }


#: The class partition of each arm with classes, and the named unions reported beside it.
CLASS_PARTITIONS = {
    WINDOW: (WINDOW_CLASSES, {"edge": EDGE_CLASSES}),
    WINDOW_OCC: (WINDOW_CLASSES, {"edge": EDGE_CLASSES}),
    **{STREAM[b]: (STREAM_CLASSES, {"seam": SEAM_CLASSES}) for b in BLOCKS},
}


def _class_cell(collector: _Collector, model: str, arm: str, haraka: str,
                members: frozenset[str], **fixed: str) -> dict:
    """Counts, rates on the arm and on the whole segment, and contributions to the arm's
    difference from the whole segment, for some classes (and fixed key fields)."""
    cell = {
        "sites": collector.count(cells(model, arm, haraka, members, **fixed)),
        "empty_count": collector.count(cells(model, arm, haraka, members, outcome=EMPTY, **fixed)),
        "segment_empty_count": collector.count(
            cells(model, arm, haraka, members, segment=EMPTY, **fixed)),
    }
    collector.put(cell, "share", Statistic((Term(
        1.0, ((cells(model, arm, haraka, members, **fixed), 1.0),), cells(model, arm, haraka)),)))
    for group in (MATCHED, EMPTY):
        collector.put(cell, f"{group}_rate", rate(model, arm, haraka, group, members, **fixed))
        collector.put(cell, f"segment_{group}_rate",
                      rate(model, arm, haraka, group, members, column="segment", **fixed))
        collector.put(cell, f"{group}_contribution",
                      contribution(model, arm, haraka, group, members, **fixed))
        cell[f"{group}_contribution"]["counts"] = discordance(
            collector, model, arm, haraka, group, members, **fixed)
    return cell


def model_section(collector: _Collector, model: str) -> dict:
    """One model's arms, paired differences against the whole segment, classes, and the
    b=1 vs b=0 stream difference."""
    section: dict = {"arms": {}, "vs_segment": {}, "classes": {}, "b1_minus_b0": {}}
    harakat = (ALL_HARAKAT, *HARAKAT)
    for arm in (SEGMENT_ALL, *ARMS):
        section["arms"][arm] = {}
        for haraka in harakat:
            cell: dict = {"sites": collector.count(cells(model, arm, haraka))}
            for group in OUTCOME_GROUPS:
                cell[f"{group}_count"] = collector.count(cells(model, arm, haraka, outcome=group))
                collector.put(cell, group, rate(model, arm, haraka, group))
            section["arms"][arm][haraka] = cell
    for arm in ARMS[1:]:
        section["vs_segment"][arm] = {}
        for haraka in harakat:
            cell = {}
            for group in (MATCHED, EMPTY):
                collector.put(cell, group, contribution(model, arm, haraka, group))
                cell[group]["counts"] = discordance(collector, model, arm, haraka, group)
            section["vs_segment"][arm][haraka] = cell
    for arm, (partition, unions) in CLASS_PARTITIONS.items():
        section["classes"][arm] = {}
        for haraka in harakat:
            section["classes"][arm][haraka] = {
                name: _class_cell(collector, model, arm, haraka, members)
                for name, members in (*((c, frozenset({c})) for c in partition), *unions.items())
            }
    section["window_occ_by_word_position"] = {
        f"{cls}/{position}": _class_cell(collector, model, WINDOW_OCC, ALL_HARAKAT,
                                         frozenset({cls}), word_position=position)
        for cls in WINDOW_CLASSES for position in WORD_POSITIONS
    }
    for haraka in harakat:
        cell = {}
        for group in (MATCHED, EMPTY):
            collector.put(cell, group, rate(model, STREAM[1], haraka, group)
                          - rate(model, STREAM[0], haraka, group))
        section["b1_minus_b0"][haraka] = cell
    return section


def gap_section(collector: _Collector, model: str) -> dict:
    """How ``0.9617 - 0.841`` splits, for ``model`` (the base teacher), all harakat.

    ``0.9617 - 0.841 = (0.9617 - S_all) + (S_all - S_elig) + (S_elig - S) + (S - S_occ)
    + (S_occ - W_occ) + (W_occ - 0.841)``: ``S_all`` is every kept segment of the pool
    decoded whole, ``S_elig`` those of the clips the window build accepts, ``S`` the
    windowed sites among them, ``S_occ`` the same counted once per window
    occurrence (ADR-0007's weighting) and ``W_occ`` the window decodes of those
    occurrences. ``S_occ - W_occ`` is what decoding a window instead of a segment costs on
    identical occurrences, and splits exactly over the window classes. The outer two terms
    are not measured here: they are whatever differs between this pool and the ADR-0005
    corpus or the ADR-0007 val windows (reciters, segmentation revision, decode numerics).
    """
    seg_all = rate(model, SEGMENT_ALL, ALL_HARAKAT, MATCHED)
    seg_eligible = rate(model, SEGMENT_ALL, ALL_HARAKAT, MATCHED,
                        frozenset({WORD_IN_NO_WINDOW, WINDOWED}))
    seg = rate(model, SEGMENT, ALL_HARAKAT, MATCHED)
    seg_occ = rate(model, WINDOW_OCC, ALL_HARAKAT, MATCHED, column="segment")
    win_occ = rate(model, WINDOW_OCC, ALL_HARAKAT, MATCHED)
    gap: dict = {"adr0005": ADR0005_SEGMENT_RECALL, "adr0007": ADR0007_WINDOW_RECALL,
                 "total": round(ADR0005_SEGMENT_RECALL - ADR0007_WINDOW_RECALL, 5)}
    for name, statistic in (
        ("S_all", seg_all), ("S_elig", seg_eligible), ("S", seg), ("S_occ", seg_occ),
        ("W_occ", win_occ),
        ("adr0005_minus_S_all", constant(ADR0005_SEGMENT_RECALL) - seg_all),
        ("clips_not_windowed", seg_all - seg_eligible),
        ("words_in_no_window", seg_eligible - seg),
        ("overlap_recount", seg - seg_occ),
        ("window_decode", seg_occ - win_occ),
        *((f"window_decode:{c}",
           -contribution(model, WINDOW_OCC, ALL_HARAKAT, MATCHED, frozenset({c})))
          for c in WINDOW_CLASSES),
        ("window_decode:edge",
         -contribution(model, WINDOW_OCC, ALL_HARAKAT, MATCHED, EDGE_CLASSES)),
        ("W_occ_minus_adr0007", win_occ - constant(ADR0007_WINDOW_RECALL)),
    ):
        collector.put(gap, name, statistic)
    return gap


def build_report(ledger: Ledger, models: list[str], meta: dict, gap_model: str = "base") -> dict:
    """Every number the write-up quotes, with reciter-clustered intervals."""
    collector = _Collector(ledger)
    report = {
        "meta": meta,
        "models": {model: model_section(collector, model) for model in models},
        "gap": gap_section(collector, gap_model),
    }
    collector.resolve()
    return report




# --- Markdown ----------------------------------------------------------------------------


def _pct(cell: dict, key: str = "weighted") -> str:
    """A rate in percent without its interval; a dash when it is undefined."""
    value = cell[key]
    return "—" if value is None else f"{100 * value:.2f}"


def _signed(value: float) -> str:
    return f"{100 * value:+.2f}".replace("-", "−")


def _pts(cell: dict) -> str:
    """A weighted difference in points with its interval, e.g. ``−1.23 [−1.50, −0.97]``."""
    if cell["weighted"] is None:
        return "—"
    interval = cell["ci95"]
    bounds = "" if interval is None else f" [{_signed(interval[0])}, {_signed(interval[1])}]"
    return _signed(cell["weighted"]) + bounds


def _rate(cell: dict) -> str:
    """A weighted rate in percent with its interval."""
    if cell["weighted"] is None:
        return "—"
    interval = cell["ci95"]
    bounds = "" if interval is None else f" [{100 * interval[0]:.2f}, {100 * interval[1]:.2f}]"
    return f"{100 * cell['weighted']:.2f}{bounds}"


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return lines + [""]


def _arm_rows(section: dict) -> list[list]:
    rows = []
    for arm in (SEGMENT_ALL, *ARMS):
        for haraka in (ALL_HARAKAT, *HARAKAT):
            cell = section["arms"][arm][haraka]
            rows.append([
                f"`{arm}`", haraka, f"{cell['sites']:g}",
                _rate(cell[MATCHED]), _rate(cell[EMPTY]),
                _pct(cell["swapped"]), _pct(cell["unanchored"]),
                _pct(cell[MATCHED], "unweighted"), f"{cell[f'{EMPTY}_count']:g}",
            ])
    return rows


def _difference_rows(section: dict) -> list[list]:
    rows = []
    for arm in ARMS[1:]:
        for haraka in (ALL_HARAKAT, *HARAKAT):
            cell = section["vs_segment"][arm][haraka]
            counts = cell[EMPTY]["counts"]
            rows.append([
                f"`{arm}`", haraka, _pts(cell[MATCHED]), _pts(cell[EMPTY]),
                f"{counts['segment_only']:g} / {counts['arm_only']:g} / {counts['net']:+g}",
            ])
    return rows


def _class_rows(table: dict, names) -> list[list]:
    rows = []
    for name in names:
        cell = table[name]
        rows.append([
            name, f"{cell['sites']:g}", _pct(cell["share"]),
            _pct(cell[f"{EMPTY}_rate"]), _pct(cell[f"segment_{EMPTY}_rate"]),
            _pts(cell[f"{EMPTY}_contribution"]), _pts(cell[f"{MATCHED}_contribution"]),
        ])
    return rows


_CLASS_HEADER = [
    "class", "sites", "share %", "empty % (arm)", "empty % (segment, same sites)",
    "Δ empty, contribution (pts)", "Δ matched, contribution (pts)",
]


def render_markdown(report: dict) -> str:
    """The generated tables of ``docs/haraka-gap-tables.md``, from the report JSON."""
    meta = report["meta"]
    lines = [
        "# Haraka gap: generated tables (#85)",
        "",
        "> Generated by `python -m training.haraka_gap report`; do not edit by hand. The",
        "> write-up is [`haraka-gap.md`](haraka-gap.md). Every rate is **agreement with the",
        "> mushaf** (the realized reference), not truth.",
        "",
        "Rates are weighted by 1 / inclusion probability, in percent, with reciter-clustered",
        f"95% intervals (B = {meta['bootstrap']['draws']:,}, seed {meta['bootstrap']['seed']}).",
        "Differences are arm minus whole segment on the same sites, in percentage points of",
        "reference harakat. `sites` are unweighted (a window share counts fractionally).",
        "",
        "## Population",
        "",
        f"- pool clips {meta['pool_clips']:,}; eligible for windowing {meta['eligible_clips']:,} "
        f"({meta['reciters']} reciters); excluded {meta['exclusions']}",
        f"- {meta['windows']:,} windows; {meta['segments_eligible']:,} segments in eligible "
        f"clips ({meta['segments_all']:,} kept in the whole pool); "
        f"{meta['stream_seconds']:,} s streamed",
        f"- {meta['population_sites']:,} sites; {meta['eligible_sites_in_no_window']:,} haraka "
        "of eligible clips sit in no window and are left out of every arm",
        f"- site times (stream regions): {meta['site_time_sources']}",
        f"- base whole-segment decodes identical to the #83 cache: "
        f"{meta['base_segments_identical_to_pool_cache'][0]:,} / "
        f"{meta['base_segments_identical_to_pool_cache'][1]:,}",
        f"- decode: weights {meta['weights_dtype']}, batch size {meta['decode_batch_size']}; "
        f"models {meta['model_refs']}",
        "",
    ]
    for model, section in report["models"].items():
        lines += [f"## `{model}`", "", "### Arms", ""]
        lines += _table(
            ["arm", "haraka", "sites", "matched %", "empty %", "swapped %", "unanchored %",
             "matched % (unweighted)", "empty (count)"],
            _arm_rows(section),
        )
        lines += ["### Arm minus whole segment, same sites", ""]
        lines += _table(
            ["arm", "haraka", "Δ matched (pts)", "Δ empty (pts)",
             "empty on segment only / in arm only / net (sites)"],
            _difference_rows(section),
        )
        for arm, (partition, unions) in CLASS_PARTITIONS.items():
            lines += [f"### `{arm}` by class", ""]
            lines += _table(_CLASS_HEADER, _class_rows(
                section["classes"][arm][ALL_HARAKAT], [*partition, *unions]))
            # Per haraka: each union beside the classes it leaves out.
            names = [*unions, *(c for c in partition if not any(c in u for u in unions.values()))]
            rows = [
                [haraka, *row]
                for haraka in HARAKAT
                for row in _class_rows(section["classes"][arm][haraka], names)
            ]
            lines += _table(["haraka", *_CLASS_HEADER], rows)
        lines += ["### `window_occ` by class and the haraka's position in its word", ""]
        by_position = section["window_occ_by_word_position"]
        lines += _table(_CLASS_HEADER, _class_rows(by_position, list(by_position)))
        lines += ["### Stream b=1 minus b=0", ""]
        lines += _table(
            ["haraka", "Δ matched (pts)", "Δ empty (pts)"],
            [[h, _pts(section["b1_minus_b0"][h][MATCHED]), _pts(section["b1_minus_b0"][h][EMPTY])]
             for h in (ALL_HARAKAT, *HARAKAT)],
        )
    gap = report["gap"]
    total = gap["total"]
    lines += [
        "## Gap arithmetic (base teacher, all harakat, weighted)",
        "",
        f"`{gap['adr0005']} − {gap['adr0007']} = {total:.4f}`, split as",
        "`(0.9617 − S_all) + (S_all − S_elig) + (S_elig − S) + (S − S_occ) + (S_occ − W_occ)"
        " + (W_occ − 0.841)`.",
        "",
    ]
    rows = [[name, _rate(gap[name]), ""] for name in ("S_all", "S_elig", "S", "S_occ", "W_occ")]
    for name in ("adr0005_minus_S_all", "clips_not_windowed", "words_in_no_window",
                 "overlap_recount",
                 "window_decode", *(f"window_decode:{c}" for c in (*WINDOW_CLASSES, "edge")),
                 "W_occ_minus_adr0007"):
        value = gap[name]["weighted"]
        share = "—" if value is None else f"{100 * value / total:.1f}%".replace("-", "−")
        rows.append([name, _pts(gap[name]), share])
    lines += _table(["term", "value (% or pts)", "share of the gap"], rows)
    return "\n".join(lines)

