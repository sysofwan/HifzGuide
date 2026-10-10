"""Render ``docs/ship-a-sizing.md`` from Ship A's committed results (#126).

The page is a function of ``tadabur/ship_a/sizing.json`` (:mod:`training.ship_a_sizing`),
``sites.summary.json``, ``worklist.summary.json`` and ``frame.json``; a test checks the
committed page equals :func:`render` of them, so no number in it can drift from its source.

Usage (from ``tools/``)::

  python -m training.ship_a_sizing_doc
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from tadabur.power_frame import FRAME_PATH
from tadabur.ship_a import SITES_SUMMARY_PATH
from tadabur.ship_a_worklist import SUMMARY_PATH as WORKLIST_SUMMARY_PATH
from training.power_simulation import TARGET_POWER
from training.ship_a_sizing import (
    EVERY_SITE,
    GEMINATES,
    ONE_PER_RECITER,
    RECOVERY,
    RETENTION,
    SIZING_PATH,
)
from training.tashkeel_eval import write_text_atomically

DOC_PATH = Path(__file__).parent.parent.parent / "docs" / "ship-a-sizing.md"

#: The planning point the summary tables are read at (the full grids are in sizing.json).
SUMMARY_THETA = 0.15
SUMMARY_ICC = 0.15
SUMMARY_NEW_ERROR_SHARE = 0.5


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _n(value: float) -> str:
    return f"{value:,}" if isinstance(value, int) else f"{value:,.1f}"


def frame_totals(frame: Mapping) -> dict[str, dict[str, int]]:
    """Per frame stratum, each field summed over its reciters."""
    fields = frame["fields"]
    return {name: {f: round(sum(v[i] for v in s["per_reciter"].values()))
                   for i, f in enumerate(fields)}
            for name, s in frame["strata"].items()}


def _certified_floor(row: Mapping) -> str:
    """The highest floor the lower bound reaches with the target power, or ``—``."""
    reached = [float(f) for f, p in row["power"].items() if p >= TARGET_POWER]
    return f"{max(reached):.1f}" if reached else "—"


def _power_rows(sizing: Mapping, side: str, split: str, rule: str = ONE_PER_RECITER,
                theta: float = SUMMARY_THETA, icc: float = SUMMARY_ICC) -> list[list]:
    rows: dict[str, dict] = {}
    for r in sizing["held_out_power"]:
        if (r["side"], r["split"], r["rule"], r["theta"], r["icc"]) == (side, split, rule,
                                                                          theta, icc):
            entry = rows.setdefault(r["cell"], {"n": r["mean_reciters"]})
            entry[r["rate"]] = _certified_floor(r)
    rates = sizing["grids"]["true_rates"]
    return [[cell, _n(round(e["n"], 1)), *(e[rate] for rate in rates)] for cell, e in rows.items()]


def _section_counts(sizing: Mapping, sites: Mapping, totals: Mapping) -> str:
    directions = sizing["budget"]["directions"]
    rows = []
    for name, entry in directions.items():
        if name == "shaddah":
            continue
        e = entry["proposed"]
        rows.append([name, e["sites"], e["base_already_right"]])
    partner = sites["pair_commits"]["partner"]
    other = sites["pair_commits"]["other_letter"]
    held = totals["shaddah:held:in_word"]
    cross = totals["shaddah:held:cross_word"]
    geminates = sites["families"]["shaddah"]
    pair_flagged = partner["sites"] + other["sites"]
    harakat = {field: sum(totals[f"tashkeel:{h}"][field] for h in ("fatha", "damma", "kasra"))
               for field in ("sites", "h448_empty_on_haraka")}
    letters = sorted(sites["pair_commits"]["by_letter"].items(), key=lambda kv: (-kv[1], kv[0]))
    commonest = ", ".join(f"`{name}` {count}" for name, count in letters[:6])
    return f"""## The counts, re-derived from the decodes

Every count below is decode level: what `h448` (streamed at b = 0, no bias) and the base
teacher commit at a site, read by the #84 scorer's rules against the mushaf. No Muraja grade
enters (`tadabur.power_frame`; `training.muraja_policy` is not imported).

- **Pair carriers.** `h448` commits a letter other than the mushaf's at **{pair_flagged}
  unique carriers** (the issue's ~550 counted the `ظ` carriers twice). Only
  **{partner["sites"]}** of them are **partner swaps**, the confusion the soft-pair allowance
  forgives; the other **{other["sites"]}** are other letters, which no allowance forgives (the
  commonest commits: {commonest}). The base
  teacher already commits the mushaf letter at {partner["base_already_right"]} of the swaps
  and {other["base_already_right"]} of the others.
- **In-word geminates.** Of {_n(held["sites"])}, `h448` decodes **{held["h448_single_on_held"]}**
  as one consonant (the base teacher {held["base_commit"] - held["base_mushaf"]}). The base
  holds **{geminates["base_already_right"]}** of those {held["h448_single_on_held"]}, so
  **{geminates["listened"]}** need listening. Cross-word geminates ({_n(cross["sites"])}, of
  which `h448` collapses {cross["h448_single_on_held"]}) are out: the app never scores them.
- **Goal 3 (Ship B), counted only.** `h448` leaves the slot empty at
  {_n(totals["tashkeel:sukun"]["h448_empty_on_sukun"])} of {_n(totals["tashkeel:sukun"]["sites"])}
  mid-word prescribed sukun and at {_n(harakat["h448_empty_on_haraka"])} of
  {_n(harakat["sites"])} mid-word harakat.

Off-letter commits per direction (a partner swap is named by the letter committed; any other
letter is grouped per carrier):

{_table(["direction", "sites", "base already right"], rows)}
"""


def _section_budget(sizing: Mapping) -> str:
    rows = []
    for name, entry in sizing["budget"]["families"].items():
        p, a = entry["proposed"], entry["alternative"]
        status = "insufficient evidence" if p["held_out_sparse"] else "sufficient"
        rows.append([name, p["sites"], p["base_already_right"], p["listened"],
                     f'{p["train"]["listened"]} ({p["train"]["listened_reciters"]})',
                     f'{p["held_out"]["listened"]} ({p["held_out"]["listened_reciters"]})',
                     status,
                     f'{a["held_out"]["listened"]} ({a["held_out"]["listened_reciters"]})'])
    swaps = sizing["budget"]["cells"]["soft-pair partner swaps"]["proposed"]
    rule = sizing["budget"]["sparse_rule"]
    directions = {name: entry["proposed"]["sites"]
                  for name, entry in sizing["budget"]["directions"].items() if "partner" in name}
    largest = max(directions, key=lambda name: (directions[name], name))
    sparse_both = [name for name, entry in sizing["budget"]["families"].items()
                   if all(entry[split]["held_out_sparse"] for split in entry)]
    return f"""## 1. Verification budget

Every pair site is listened to (D6). Geminates the base teacher holds take its decode as the
target (ADR-0011 §3) and leave the worklist, so the **~300-geminate slot becomes
{sizing["budget"]["cells"][GEMINATES]["proposed"]["listened"]}**: every collapsed geminate the
base does not hold, a census. Listened sites (reciters) per half, under the proposed 50/50
split and the alternative (two thirds held out). A held-out cell with fewer than
{rule["min_reciters"]} reciters or {rule["min_sites"]} sites is sparse (§1): its family is
reported as insufficient evidence and pooled (§7).

{_table(["family", "flagged", "base already right", "listened", "train (50/50)",
         "held out (50/50)", "50/50 status", "held out (⅓ / ⅔)"], rows)}

Per direction, the largest partner-swap direction ({largest}) has {directions[largest]} sites
in all, so no direction supports a claim of its own; the partner swaps pooled hold
{swaps["held_out"]["listened"]} held-out sites from {swaps["held_out"]["listened_reciters"]}
reciters. Sparse on the held-out half under both splits: {", ".join(sparse_both) or "none"}.
"""


def _section_power(sizing: Mapping) -> str:
    rates = sizing["grids"]["true_rates"]
    head = ["cell", "held-out reciters", *(f"true {r:.0%}" for r in rates)]
    needed = [[f'{r["rate"]:.0%}', f'{r["floor"]:.0%}', r["size"], f'{r["power"]:.2f}']
              for r in sizing["sites_needed"]]
    recovery = _power_rows(sizing, RECOVERY, "proposed")
    middle = len(rates) // 2
    reading = "; ".join(f"{row[0]} {row[2 + middle]}" for row in recovery)
    every = _power_rows(sizing, RECOVERY, "proposed", EVERY_SITE)
    every0 = _power_rows(sizing, RECOVERY, "proposed", EVERY_SITE, icc=0.0)
    return f"""## 2. Power on the held-out half

A cell's held-out sites are listened to; each is a real mistake with probability θ. The
candidate **recovers** a verified-correct site (commits the mushaf letter, or holds the
geminate) with a true rate, or **retains** a verified mistake (keeps committing what was
said). Each of {sizing["repetitions"]:,} repetitions takes the §1 lower bound of that rate on
one site per reciter (Wilson where sparse, else the reciter bootstrap); reciter effects at
latent ICC 0, 0.05 and 0.15 on both whether a site is a mistake and whether it is recovered.
One site per reciter makes the bound blind to the ICC; the pooled cells are also read on every
site with the reciter-clustered bootstrap, which the ICC does move.

**No floor is chosen here** (#128). Each cell shows the **highest floor the lower bound
reaches with {TARGET_POWER:.0%} power**, at θ = {SUMMARY_THETA} and ICC = {SUMMARY_ICC}
(every θ, ICC and floor is in `sizing.json`).

Reciters (one site each) a true rate needs for its lower bound to reach a floor with
{TARGET_POWER:.0%} power (no clustering; a size the grid steps to, so an upper estimate):

{_table(["true rate", "floor", "reciters", "power"], needed)}

**Recovery**, 50/50:

{_table(head, recovery)}

At a true recovery of {rates[middle]:.0%}, the floor each cell reaches: {reading}.

**Recovery**, two thirds held out:

{_table(head, _power_rows(sizing, RECOVERY, "alternative"))}

**Retention** of verified mistakes, 50/50 (θ = {SUMMARY_THETA}; held-out mistakes are few):

{_table(head, _power_rows(sizing, RETENTION, "proposed"))}

Every site with the reciter bootstrap, recovery, 50/50, at ICC 0.15 and at ICC 0:

{_table(head, every)}

{_table(head, every0)}
"""


def _section_harm(sizing: Mapping) -> str:
    harm = sizing["harm"]
    rates = sizing["grids"]["harm_rates"]
    rows = [[f'{h["churn"]:.2%}', f'{h["new_error_share"]:.0%}', h["n"],
             _n(round(h["mean_sites_heard"], 1)), _n(round(h["mean_minutes"], 1)),
             f'{h["true_rate_per_1000"]:.2f}', f'{h["median_bound_per_1000"]:.2f}',
             *(f'{h["power"][str(x)]:.2f}' for x in rates)]
            for h in harm["scenarios"] if h["icc"] == SUMMARY_ICC]
    words = [[f'{w["rate"]:.0%}', f'{w["reduction"]:.0%}',
              *(f'{w["power"][str(m)]:.2f}' for m in sizing["grids"]["word_margins"])]
             for w in sizing["word_sample"]]
    n_words = sizing["word_sample"][0]["correct_words"]
    grid = sizing["grids"]
    top = next(h for h in harm["scenarios"] if h["icc"] == SUMMARY_ICC
               and h["churn"] == max(grid["churns"])
               and h["new_error_share"] == min(grid["new_error_shares"])
               and h["n"] == max(grid["cluster_reciters"]))
    return f"""## 3. New-error harm on sites `h448` gets right

Once a candidate exists, the **changed sites** (its decode ≠ `h448`'s at a pair carrier or an
in-word geminate where `h448` commits the mushaf value) are a census, computed without
listening. On the held-out half that is {harm["held_out_carriers"]:,} carriers of
{harm["reciters"]} reciters. A blind, candidate-locked **PPS cluster sample** draws reciters in
proportion to their changed sites and adjudicates every changed site of each reciter drawn;
`cluster_mistake_bound` gives a design-valid 97.5% bound on the new errors among all of them.
Simulated at ICC {SUMMARY_ICC} (the grid is in `sizing.json`): churn is the share of those
carriers the candidate changes, and a share of the changed sites are new errors (the rest are
corrections of real mistakes, or abstentions). Minutes at {harm["seconds_per_site"]:.1f} s per
site. **Report only: not on #61's worklist.**

{_table(["churn", "new errors", "reciters drawn", "sites heard", "minutes", "true /1,000",
         "median bound /1,000", *(f"P(bound ≤ {x:g})" for x in rates)], rows)}

- The census of changed sites bounds the new errors **without listening**: they cannot exceed
  the changes (churn × 1,000 per 1,000). Listening tightens that only where many changes are
  *not* errors, and only by reaching most changed reciters: at the highest churn
  ({top["churn"]:.1%}) with {top["new_error_share"]:.0%} of changes new errors, {top["n"]}
  reciters ({_n(round(top["mean_minutes"], 1))} minutes) bring the median bound from
  {top["churn"] * 1000:.1f} to {top["median_bound_per_1000"]:.1f} per 1,000 (truth
  {top["true_rate_per_1000"]:.1f}).
- **No separate draw on unflagged strata is needed** for decode-level harm at pair and
  geminate carriers: the census of changes covers every carrier, flagged or not, and costs
  nothing to enumerate. What it cannot see is harm the candidate does *elsewhere* (other
  letters, tashkeel) and harm in the app outcome, where the ratchet decides.
- **The random word sample** is the model-independent denominator for that: {n_words}
  correctly recited words (planning: {sizing["grids"]["word_correct_share"]:.0%} of the 160
  heard as correct), one per held-out reciter. It covers false flags from any cause, as the app
  grades them in #123's replay, and new errors anywhere in a word. It leaves out the rare
  events: per-family harm at a few per 1,000 carriers, real mistakes (a few words), cross-word
  geminates. Its power to show the candidate's false-flag rate below `h448`'s, with the
  candidate adding no flag (paired, one word per reciter; `h448`'s app false-flag rate is
  unknown until #123):

{_table(["`h448` false flags", "reduction", *(f"P(upper bound ≤ {m:+.0%})" for m in
                                               sizing["grids"]["word_margins"])], words)}
"""


def _section_mix(sizing: Mapping, worklist: Mapping) -> str:
    mix = sizing["mistake_mix"]
    carried = worklist["carried_rows_already_ship_a_sites"]
    pile = worklist["parts"]["reject_pile"]["rows"]
    p35 = [[family, e["should_accept"], e["should_reject"], f'{e["reject_share_upper"]:.0%}']
           for family, e in mix["p35_h448_off"].items()]
    thetas = sizing["grids"]["thetas"]
    train = [[cell, mix["train_listened"][cell], *(mix["expected_train_mistakes"][cell][str(t)]
                                                   for t in thetas)]
             for cell in mix["train_listened"]]
    return f"""## 4. Mix of real mistakes

No site has been listened to yet: the 23 P3.5 re-listens and the reject-pile sites are on
#61's worklist (#87's {pile + carried} reject-pile sites, less the {carried} that are Ship A
sites already), so
the mix is **measured in the first sitting**. The only prior is the P3.5 fixtures where `h448`
commits an off value: a should-reject fixture is a clip-level verdict, so its share bounds the
share of mistakes from above. The fixtures were selected on the base teacher's flags, so they
over-represent mistakes against the pool's `h448` flags.

{_table(["family", "should-accept", "should-reject (pending)", "reject share (upper)"], p35)}

Planning θ = {", ".join(str(t) for t in thetas)}. Verified mistakes the "keep both outcomes"
term would get on the train half (50/50) at each θ:

{_table(["cell", "train sites listened", *(f"θ = {t}" for t in thetas)], train)}
"""


def _section_worklist(worklist: Mapping) -> str:
    """The worklist's parts and sittings."""
    parts = [[part, e["rows"], e["reciters"], e["rows_by_half"]["train"],
              e["rows_by_half"]["held_out"], _n(e["minutes"])]
             for part, e in worklist["parts"].items()]
    sittings = [[k, f'{s["queue_positions"][0]}-{s["queue_positions"][1]}', s["rows"],
                 _n(s["minutes"])] for k, s in enumerate(worklist["sittings"], 1)]
    return f"""## The worklist for #61

`tools/tadabur/ship_a/worklist.jsonl` (site rows) and `word_sample.jsonl` (words), frozen by
the SHA-256 in `worklist.summary.json`; launch commands in
[`tools/tadabur/ship_a/README.md`](../tools/tadabur/ship_a/README.md). The blind UI shows no
model output and no target. Minutes by #87's model (each excerpt twice plus 4 s; a word twice
plus 6 s).

{_table(["part", "rows", "reciters", "train", "held out", "minutes"], parts)}

**{_n(worklist["listening_minutes_without_edit_check"])} minutes** of Ship A listening, plus
the synthetic-edit check the UI queues with it: **{_n(worklist["listening_minutes"])} minutes**
in all, in three sittings (runs of the queue; the words come last because a sampled word is
shown whole):

{_table(["sitting", "queue positions", "rows", "minutes"], sittings)}
"""


def _section_questions(sites: Mapping, worklist: Mapping, seconds: float) -> str:
    other = worklist["parts"]["pair_other_letter"]
    held = sites["strata"]["ship_a:shaddah:held:h448_single:base_held"]["held_out"]
    minutes = round(held["reciters"] * seconds / 60, 1)
    return f"""## For the owner

- **Partner swaps or every off letter?** The soft-pair allowance forgives partner swaps; the
  {other["rows"]} other-letter pair sites ({_n(other["minutes"])} minutes) are flags it never
  forgives. They are on the worklist as #126 asks; listening to the partner swaps alone
  would save those minutes and lose the training labels for other-letter false flags.
- **The split.** Confirm 50/50 or the ⅓ / ⅔ alternative before the first sitting.
- **Geminate evaluation.** The {held["sites"]} held-out geminates the base holds get the
  teacher's target but no truth, so their recovery is unmeasured unless some are listened to:
  one per held-out reciter is {held["reciters"]} sites, about {_n(minutes)} minutes.
- **Floors and margins** (#128): every power here is over a grid; none is chosen.
"""


def render(sizing: Mapping, sites: Mapping, worklist: Mapping, frame: Mapping) -> str:
    totals = frame_totals(frame)
    return "\n".join([
        f"""# Ship A sizing (#126)

Generated by `python -m training.ship_a_sizing_doc` from `tools/tadabur/ship_a/sizing.json`
(`python -m training.ship_a_sizing run`), `sites.summary.json`, `worklist.summary.json` and
`frame.json`; do not edit by hand.

Ship A (ADR-0012, goals 1-2) trains on sites the owner verifies by ear: where `h448` commits
the wrong letter at a target-pair carrier, or collapses an in-word geminate. This page sizes
that listening and says what the held-out half can certify. It reads only `h448` and the base
teacher, never a candidate, and **recommends no threshold, floor or margin**: those are the
owner's (#128). The app outcome itself is #123's Muraja replay; nothing here models Muraja.
""",
        _section_counts(sizing, sites, totals),
        _section_budget(sizing),
        _section_power(sizing),
        _section_harm(sizing),
        _section_mix(sizing, worklist),
        _section_worklist(worklist),
        _section_questions(sites, worklist, sizing["harm"]["seconds_per_site"]),
    ])


def load_inputs() -> tuple[dict, dict, dict, dict]:
    return tuple(json.loads(path.read_text(encoding="utf-8")) for path in (
        SIZING_PATH, SITES_SUMMARY_PATH, WORKLIST_SUMMARY_PATH, FRAME_PATH))


def main() -> None:
    write_text_atomically(DOC_PATH, render(*load_inputs()))
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
