"""What the owner heard at ``h448``'s excess flags on Tarteel v1 (#160), run after the sitting.

Reads the frozen worklist (:mod:`tadabur.tarteel_excess`) and the page's ``verdicts.jsonl``:
per stratum and over the whole frame, the share of the excess flags (words ``h448`` flags and
the base teacher grades correct, look half) the owner heard as

* **correct** (``Yes``): recited correctly, so the flag is a **model error**;
* **mistake** (``No``): the user's real mistake, which the teacher missed;
* **unclear**;

and the same shares on the **control** words (both models grade them correct): the owner's
baseline, how often a word nobody flagged is heard as a mistake (or unclear). Each excess share
is also given **net of that baseline**, the excess share minus the control share: a mistake
share no higher than the controls' says the flags found no more mistakes than chance would.

Each word weighs ``1 / inclusion_probability`` (its place in its clip, then its clip's draw),
so a share is ``Σ w·[verdict] / Σ w`` over the answered words; the pooled one estimates the
share among every excess flag of the look half, not among the words drawn. The three shares are
over every answer; ``model_error_of_clear`` leaves ``unclear`` out of the denominator, as every
other evaluation does with an unclear verdict.

Intervals follow acceptance rules §1 (:mod:`training.acceptance_stats`): the bootstrap clustered
by surah (#156's clusters, as Tarteel v1 names no user) where a cell is not sparse, a Wilson
bound for a share whose sparse cell is independent and equally weighted, none otherwise. A
difference resamples the surahs of both groups in one draw (``difference_interval``), and has
a bound only where **each** group alone would have a bootstrap one: neither sparse, and
neither's replicates (those that define it) all equal. Pooled support must not stand in for a
six-word group's, and its two groups are disjoint, never paired, so there is no score-interval
fallback. A stratum holds about
six words, so its interval is wide or absent: read the strata as where the model errs, and the
pooled share as how often.

Usage (from ``tools/``; torch-free)::

  python -m tadabur.tarteel_excess_report
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from training.acceptance_stats import (
    NO_BOUND,
    Interval,
    Resample,
    difference_interval,
    is_sparse,
    ratio,
    ratio_interval,
    unit_if_equal,
)
from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE
from training.tashkeel_eval import write_text_atomically

from .listening_session import Verdict, read_verdicts
from .tarteel_excess import (
    CONTROL,
    EXCESS_DIR,
    EXCESS_STRATA,
    VERDICTS_PATH,
    ListeningWord,
    load_worklist,
)
from .truth_sites import UNCLEAR
from .word_sample import WORD_VERDICTS

REPORT_PATH = EXCESS_DIR / "report.json"
POOLED = "pooled"
MODEL_ERROR_OF_CLEAR = "model_error_of_clear"


def _answered(heard: str | None) -> bool:
    return heard is not None


def _clear(heard: str | None) -> bool:
    return heard in (CORRECT_SIDE, MISTAKE_SIDE)


#: Each share reported: the verdict counted, and which answers form its denominator.
MEASURES: dict[str, tuple[str, Callable[[str | None], bool]]] = {
    **{v: (v, _answered) for v in WORD_VERDICTS},
    MODEL_ERROR_OF_CLEAR: (CORRECT_SIDE, _clear),
}


def _heard(word: ListeningWord, verdicts: Mapping[str, Verdict]) -> str | None:
    verdict = verdicts.get(word.site_id)
    return None if verdict is None else verdict.heard


def _share(words: Sequence[ListeningWord], verdicts: Mapping[str, Verdict], measure: str) -> dict:
    """One measure's weighted share over ``words``, its §1 interval and its counts."""
    hit, among = MEASURES[measure]
    kept = [w for w in words if among(_heard(w, verdicts))]
    if not kept:
        return {"words": 0, "of": 0, "share": None, "interval": None}
    weights = unit_if_equal([w.weight for w in kept])
    hits = np.array([_heard(w, verdicts) == hit for w in kept], dtype=float)
    return {"words": int(hits.sum()), "of": len(kept),
            "share": round(ratio(weights * hits, weights), 4),
            "interval": ratio_interval([w.cluster for w in kept], weights * hits, weights,
                                       weights).as_dict()}


def _difference(words: Sequence[ListeningWord], controls: Sequence[ListeningWord],
                verdicts: Mapping[str, Verdict], measure: str) -> dict:
    """``words``' share minus the controls' for one measure, and its interval: every answered
    word of either group is one observation, in the first ratio or the second."""
    hit, among = MEASURES[measure]
    kept = [(w, first) for group, first in ((words, True), (controls, False)) for w in group
            if among(_heard(w, verdicts))]
    if not any(first for _, first in kept) or all(first for _, first in kept):
        return {"difference": None, "interval": None}
    weights = np.array([w.weight for w, _ in kept])
    hits = np.array([_heard(w, verdicts) == hit for w, _ in kept], dtype=float)
    first = np.array([f for _, f in kept], dtype=float)
    arm = (weights * hits * first, weights * first)
    baseline = (weights * hits * (1 - first), weights * (1 - first))
    clusters = [w.cluster for w, _ in kept]
    resample = Resample.by_cluster(clusters)
    supported = all(_supported(resample, terms, [c for (_, f), c in zip(kept, clusters) if f == own])
                    for terms, own in ((arm, True), (baseline, False)))
    interval = (difference_interval(clusters, arm, baseline, resample=resample) if supported
                else Interval(None, None, NO_BOUND, "a group is sparse or degenerate on its own"))
    return {"difference": round(ratio(*arm) - ratio(*baseline), 4), "interval": interval.as_dict()}


def _supported(resample: Resample, terms: tuple[np.ndarray, np.ndarray],
               clusters: Sequence[int]) -> bool:
    """Whether one group of a difference would have a bootstrap bound on its own: not sparse
    (its own words and surahs), and its replicates, where defined, not all equal."""
    replicates = resample.ratios(*terms)
    defined = replicates[~np.isnan(replicates)]
    return not is_sparse(len(clusters), len(set(clusters))) and bool(
        defined.size) and not np.all(defined == defined[0])


def cell(words: Sequence[ListeningWord], verdicts: Mapping[str, Verdict]) -> dict:
    """One group's verdicts: counts and each measure's weighted share."""
    return {
        "drawn": len(words),
        "answered": sum(_answered(_heard(w, verdicts)) for w in words),
        "estimated_frame_words": round(sum(w.weight for w in words), 1),
        **{m: _share(words, verdicts, m) for m in MEASURES},
    }


def report(rows: Sequence[ListeningWord], verdicts: Mapping[str, Verdict]) -> dict:
    """Per excess stratum, pooled over the excess words, the controls, and every excess cell
    net of the controls (module docstring). A verdict outside the word question's answers, or
    for a word the worklist does not hold, is an error."""
    ids = {r.site_id for r in rows}
    for verdict in verdicts.values():
        if verdict.site_id not in ids or verdict.heard not in WORD_VERDICTS:
            raise ValueError(f"{verdict.site_id}: {verdict.heard!r} is no answer to a drawn word")
    controls = [r for r in rows if r.stratum == CONTROL]
    groups = {**{s: [r for r in rows if r.stratum == s] for s in EXCESS_STRATA},
              POOLED: [r for r in rows if r.stratum != CONTROL]}
    return {
        "issue": 160,
        "verdicts": {v: sum(x.heard == v for x in verdicts.values()) for v in WORD_VERDICTS},
        "strata": {s: cell(groups[s], verdicts) for s in EXCESS_STRATA},
        POOLED: cell(groups[POOLED], verdicts),
        CONTROL: cell(controls, verdicts),
        "net_of_control": {name: {m: _difference(words, controls, verdicts, m) for m in MEASURES}
                           for name, words in groups.items()},
    }


def _bounds(interval: dict | None) -> str:
    if not interval or interval["lower"] is None:
        return ""
    return f" [{100 * float(interval['lower']):.0f}, {100 * float(interval['upper']):.0f}]"


def _fmt(share: dict) -> str:
    if share["share"] is None:
        return "–"
    return f"{100 * share['share']:.0f}%{_bounds(share['interval'])} ({share['words']}/{share['of']})"


def _fmt_net(net: dict) -> str:
    if net["difference"] is None:
        return "–"
    return f"{100 * net['difference']:+.0f} pts{_bounds(net['interval'])}"


def table(result: dict) -> str:
    """The report as Markdown: each excess stratum, the pooled excess words and the controls;
    then each excess cell net of the controls."""
    head = ("| group | drawn | answered | correct (model error) | mistake | unclear "
            "| model error of clear |\n|---|---|---|---|---|---|---|\n")
    cells = {**result["strata"], POOLED: result[POOLED], CONTROL: result[CONTROL]}
    shares = "".join(
        f"| {name} | {c['drawn']} | {c['answered']} | {_fmt(c[CORRECT_SIDE])} | "
        f"{_fmt(c[MISTAKE_SIDE])} | {_fmt(c[UNCLEAR])} | {_fmt(c[MODEL_ERROR_OF_CLEAR])} |\n"
        for name, c in cells.items())
    net_head = ("\nNet of the controls (excess share − control share):\n\n"
                "| group | correct | mistake | unclear | model error of clear |\n"
                "|---|---|---|---|---|\n")
    nets = "".join(f"| {name} | " + " | ".join(_fmt_net(n[m]) for m in MEASURES) + " |\n"
                   for name, n in result["net_of_control"].items())
    return head + shares + net_head + nets


def main(verdicts_path: Path = VERDICTS_PATH) -> None:
    result = report(load_worklist(), read_verdicts(verdicts_path))
    write_text_atomically(REPORT_PATH, json.dumps(result, ensure_ascii=False, indent=1,
                                                  sort_keys=True) + "\n")
    print(table(result))
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
