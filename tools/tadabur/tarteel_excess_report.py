"""What the owner heard at ``h448``'s excess flags on Tarteel v1 (#160), run after the sitting.

Reads the frozen worklist (:mod:`tadabur.tarteel_excess`) and the page's ``verdicts.jsonl``:
per stratum and over the whole frame, the share of the excess flags (words ``h448`` flags and
the base teacher grades correct, look half) the owner heard as

* **correct** (``Yes``): recited correctly, so the flag is a **model error**;
* **mistake** (``No``): the user's real mistake, which the teacher missed;
* **unclear**.

Each word weighs ``1 / inclusion_probability`` (its clip's representative, then the stratum's
draw), so a share is ``Σ w·[verdict] / Σ w`` over the answered words and the pooled one
estimates the share among every excess flag of the look half, not among the words drawn. The
three shares are over every answer; ``model_error_of_clear`` leaves ``unclear`` out of the
denominator, as every other evaluation does with an unclear verdict.

Intervals follow acceptance rules §1 (:func:`training.acceptance_stats.ratio_interval`): the
bootstrap clustered by surah (#156's clusters, as Tarteel v1 names no user) where a cell is not
sparse, a Wilson bound where it is sparse but its words are independent and equally weighted,
none otherwise. A stratum holds about ten words, so its interval is wide or absent: read the
strata as where the model errs, and the pooled share as how often.

Usage (from ``tools/``; torch-free)::

  python -m tadabur.tarteel_excess_report
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from training.acceptance_stats import ratio, ratio_interval, unit_if_equal
from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE
from training.tashkeel_eval import write_text_atomically

from .listening_session import Verdict, read_verdicts
from .tarteel_excess import EXCESS_DIR, STRATA, VERDICTS_PATH, ListeningWord, load_worklist
from .truth_sites import UNCLEAR
from .word_sample import WORD_VERDICTS

REPORT_PATH = EXCESS_DIR / "report.json"
POOLED = "pooled"


def _share(words: Sequence[ListeningWord], hits: Sequence[bool], among: Sequence[bool]) -> dict:
    """The weighted share of ``among`` words that are ``hits``, its §1 interval and counts."""
    kept = [(w, h) for w, h, a in zip(words, hits, among) if a]
    if not kept:
        return {"words": 0, "of": 0, "share": None, "interval": None}
    weights = unit_if_equal([w.weight for w, _ in kept])
    hit = np.array([h for _, h in kept], dtype=float)
    clusters = [w.cluster for w, _ in kept]
    point = ratio(weights * hit, weights)
    return {"words": int(hit.sum()), "of": len(kept), "share": round(point, 4),
            "interval": ratio_interval(clusters, weights * hit, weights, weights).as_dict()}


def cell(words: Sequence[ListeningWord], verdicts: Mapping[str, Verdict]) -> dict:
    """One stratum's (or the frame's) verdicts: counts, each verdict's weighted share of the
    answers, and the model-error share of the clear ones."""
    heard = [verdicts[w.site_id].heard if w.site_id in verdicts else None for w in words]
    answered = [h is not None for h in heard]
    clear = [h in (CORRECT_SIDE, MISTAKE_SIDE) for h in heard]
    return {
        "drawn": len(words),
        "answered": sum(answered),
        "estimated_frame_words": round(sum(w.weight for w in words), 1),
        **{v: _share(words, [h == v for h in heard], answered) for v in WORD_VERDICTS},
        "model_error_of_clear": _share(words, [h == CORRECT_SIDE for h in heard], clear),
    }


def report(rows: Sequence[ListeningWord], verdicts: Mapping[str, Verdict]) -> dict:
    """Per stratum and pooled (module docstring). A verdict outside the word question's
    answers, or for a word the worklist does not hold, is an error."""
    ids = {r.site_id for r in rows}
    for verdict in verdicts.values():
        if verdict.site_id not in ids or verdict.heard not in WORD_VERDICTS:
            raise ValueError(f"{verdict.site_id}: {verdict.heard!r} is no answer to a drawn word")
    return {
        "issue": 160,
        "verdicts": {v: sum(x.heard == v for x in verdicts.values()) for v in WORD_VERDICTS},
        "strata": {s: cell([r for r in rows if r.stratum == s], verdicts) for s in STRATA},
        POOLED: cell(rows, verdicts),
    }


def _fmt(share: dict) -> str:
    if share["share"] is None:
        return "–"
    interval = share["interval"]
    bounds = ("" if interval["lower"] is None
              else f" [{100 * interval['lower']:.0f}, {100 * interval['upper']:.0f}]")
    return f"{100 * share['share']:.0f}%{bounds} ({share['words']}/{share['of']})"


def table(result: dict) -> str:
    """The report as a Markdown table, one row per stratum and the pooled row."""
    head = ("| stratum | drawn | answered | correct (model error) | mistake | unclear "
            "| model error of clear |\n|---|---|---|---|---|---|---|\n")
    cells = {**result["strata"], POOLED: result[POOLED]}
    return head + "".join(
        f"| {name} | {c['drawn']} | {c['answered']} | {_fmt(c[CORRECT_SIDE])} | "
        f"{_fmt(c[MISTAKE_SIDE])} | {_fmt(c[UNCLEAR])} | {_fmt(c['model_error_of_clear'])} |\n"
        for name, c in cells.items())


def main(verdicts_path: Path = VERDICTS_PATH) -> None:
    result = report(load_worklist(), read_verdicts(verdicts_path))
    write_text_atomically(REPORT_PATH, json.dumps(result, ensure_ascii=False, indent=1,
                                                  sort_keys=True) + "\n")
    print(table(result))
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
