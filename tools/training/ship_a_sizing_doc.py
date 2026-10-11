"""Render ``docs/ship-a-sizing.md`` from Ship A's committed results (#126).

The page is a function of ``tadabur/ship_a/sizing.json`` (:mod:`training.ship_a_sizing`),
``sites.summary.json``, ``worklist.summary.json``, ``frame.json``, the mining extension's
summary and frame, and ``waqf_guard.summary.json``; a test checks the
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
from tadabur.ship_a_mining import FRAME_PATH as MINING_FRAME_PATH
from tadabur.ship_a_mining import SUMMARY_PATH as MINING_SUMMARY_PATH
from tadabur.ship_a_worklist import SITTING_MINUTES
from tadabur.ship_a_worklist import SUMMARY_PATH as WORKLIST_SUMMARY_PATH
from tadabur.waqf_guard import SUMMARY_PATH as GUARD_SUMMARY_PATH
from training.power_simulation import TARGET_POWER
from training.ship_a_sizing import (
    EVERY_SITE,
    GEMINATES,
    ONE_PER_RECITER,
    RECOVERY,
    RETENTION,
    SIZING_PATH,
    SOFT_PAIRS_POOLED,
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


def _power_rows(sizing: Mapping, side: str, rule: str = ONE_PER_RECITER,
                theta: float = SUMMARY_THETA, icc: float = SUMMARY_ICC) -> list[list]:
    rows: dict[str, dict] = {}
    for r in sizing["held_out_power"]:
        if (r["side"], r["rule"], r["theta"], r["icc"]) == (side, rule, theta, icc):
            entry = rows.setdefault(r["cell"], {"n": r["mean_reciters"]})
            entry[r["rate"]] = _certified_floor(r)
    rates = sizing["grids"]["true_rates"]
    return [[cell, _n(round(e["n"], 1)), *(e[rate] for rate in rates)] for cell, e in rows.items()]


def _section_counts(sizing: Mapping, sites: Mapping, frame: Mapping, totals: Mapping) -> str:
    rows = [[name, e["sites"], e["base_already_right"], "yes" if e["listened"] else "no"]
            for name, e in sizing["budget"]["directions"].items() if name != "shaddah"]
    partner = sites["pool_pair_commits"]["partner"]
    other = sites["pool_pair_commits"]["other_letter"]
    held = totals["shaddah:held:in_word"]
    cross = totals["shaddah:held:cross_word"]
    geminates = sites["families"]["shaddah"]
    pair_flagged = partner["sites"] + other["sites"]
    harakat = {field: sum(totals[f"tashkeel:{h}"][field] for h in ("fatha", "damma", "kasra"))
               for field in ("sites", "h448_empty_on_haraka")}
    letters = sorted(sites["pool_pair_commits"]["by_letter"].items(),
                     key=lambda kv: (-kv[1], kv[0]))
    commonest = ", ".join(f"`{name}` {count}" for name, count in letters[:6])
    return f"""## The counts, re-derived from the decodes

Every count below is decode level: what `h448` (streamed at b = 0, no bias) and the base
teacher commit at a site, read by the #84 scorer's rules against the mushaf. No Muraja grade
enters (`tadabur.power_frame`; `training.muraja_policy` is not imported).

The target pairs are nine (D9, `CONTEXT.md`): the six Muraja soft pairs, `ذ↔ظ`, and the
owner's `س↔ش` and `ط↔ق`, which the app does not forgive yet. They are reported as their own
families and never pooled with the soft pairs.

Every population here is `h448`-unseen: no clip whose recording is tied, even as an alias or
a probable copy, to what `h448` trained on, `decode_evalset` or the sealed panel
(`exposure.unseen_violations`). That leaves out {frame["linked_clips_left_out"]} of the pool's
clips.

- **Pair carriers in the mining pool.** `h448` commits a letter other than the mushaf's at
  **{pair_flagged} unique carriers**. **{partner["sites"]}** of them are **partner swaps** (the
  pair's other letter), the sites Ship A's listening is drawn from (below); the other
  **{other["sites"]}** commit a letter outside every target pair and are not asked (D9). The commonest commits:
  {commonest}. The owner traced `ت`→`ه` to a ta marbuta the reciter paused on where the
  realized reference has wasl `ت`: a reference problem (#130), not a model one. The base teacher
  already commits the mushaf letter at {partner["base_already_right"]} of the swaps.
- **In-word geminates.** Of {_n(held["sites"])}, `h448` decodes **{held["h448_single_on_held"]}**
  as one consonant (the base teacher {held["base_commit"] - held["base_mushaf"]}). The base
  holds **{geminates["base_already_right"]}** of those {held["h448_single_on_held"]}, whose
  target is its decode with no listening (D10); **{geminates["listened"]}** are listened to.
  Cross-word geminates ({_n(cross["sites"])}, of which `h448` collapses
  {cross["h448_single_on_held"]}) are out: the app never scores them.
- **Goal 3 (Ship B), counted only.** `h448` leaves the slot empty at
  {_n(totals["tashkeel:sukun"]["h448_empty_on_sukun"])} of {_n(totals["tashkeel:sukun"]["sites"])}
  mid-word prescribed sukun and at {_n(harakat["h448_empty_on_haraka"])} of
  {_n(harakat["sites"])} mid-word harakat.

Off-letter commits in the pool and the mining extension per direction (a partner swap is
named by the letter committed; any other letter is grouped per carrier):

{_table(["direction", "sites", "base already right", "listened"], rows)}
"""


def _section_mining(mining: Mapping, mining_frame: Mapping, sites: Mapping) -> str:
    excluded = ", ".join(f"{reason} {n:,}" for reason, n in mining_frame["excluded"].items())
    rows = [[pair, e["h448"]["partner_commits"], e["h448"]["clips"], e["base"]["partner_commits"],
             sites["extension_families"].get(pair, {}).get("sites", 0)]
            for pair, e in mining["pairs"].items()]
    return f"""## Mining more partner swaps (D9)

`tadabur.ship_a_mining` decoded more audio `h448` never trained on: every shard outside
`h448`'s training shards (the held-out block 0-20 and the strided reserve), every row the
pool's own rule admits, less the clips already staged, the sealed
panel's reciters and recordings the exposure registry ties to `h448`'s training shards
({excluded}). That frame holds {mining_frame["frame_clips"]:,} clips of
{mining_frame["frame_reciters"]} reciters. Whole shards were scanned in a salted order until
the time budget was spent: **{mining["scanned_shards"]} of {len(mining_frame["shard_order"])}
shards, {mining["scanned_clips"]:,} clips of {mining["scanned_reciters"]} reciters, in
{mining["scan_gpu_hours"]} GPU-hours**, each clip decoded whole by `h448` (b = 0 stream) and
the base teacher. **{mining["hit_clips"]} clips** ({mining["hit_clips_per_gpu_hour"]} per
GPU-hour) hold an `h448` partner swap; they were staged, segmented as the pool was, and
mined like it. The build leaves out {len(mining["duplicate_recordings_dropped"])} hits that
repeat a staged recording byte for byte and {len(mining["linked_recordings_dropped"])} tied to a
`decode_evalset` recording (the scan's frame followed links to `h448`'s training shards
only); {mining["kept_hit_clips"]} of the rest kept a segment. Every scanned recording, hit or
not, is in the exposure registry as `ship_a.mining`.

The scan stopped when its GPU-time budget was spent, and that stopping rule depends on how
long each shard takes, so the scanned shards are **not** a random sample of the frame's.
**Inference from the extension is restricted to the scanned population**, of which the hits
are a census: its clips carry inclusion probability 1, and nothing here extends it to the
unscanned shards.

Per target pair, the partner commits in the scanned clips (whole-ayah alignment) and the
partner swaps mined from the hit clips' kept segments:

{_table(["pair", "h448 partner commits", "h448 clips", "base partner commits",
         "partner swaps mined"], rows)}
"""


def _section_checks(sites: Mapping) -> str:
    checks = sites["partner_swap_checks"]
    rows = [[name, e["sites"], e["reciters"], e["max_per_reciter"], e["ayahs"],
             e["base_already_right"], f'`{e["top_word"]}` {e["top_word_sites"]}']
            for name, e in sorted(checks.items(), key=lambda kv: (-kv[1]["sites"], kv[0]))]
    agreed = [f'{name} ({e["sites"] - e["base_already_right"]} of {e["sites"]}; commonest word '
              f'`{e["top_word"]}`, {e["top_word_sites"]} sites)'
              for name, e in sorted(checks.items()) if 2 * e["base_already_right"] < e["sites"]]
    return f"""## Checks on the partner swaps

What could make a direction's count misleading: one reciter or one word holding most of it,
or the base teacher hearing the same partner (then the "swap" may be what was said, or a
shared model habit). Pool and extension together:

{_table(["direction", "sites", "reciters", "most per reciter", "ayat", "base already right",
         "commonest carrier word"], rows)}

Directions where the base teacher also commits the partner at most sites: {"; ".join(agreed) or "none"}.
Listening decides them; until then they are not evidence of an `h448`-only error.
"""


def _section_listening(sites: Mapping) -> str:
    rows = [[kind, e["sites"], e["listened"]] for kind, e in sites["listening"].items()]
    return f"""## What is listened to (owner decision, 2026-10-10)

The owner chose the lean set over every partner swap (decision A) and a sample of the
`كسب` swaps (decision B). `tadabur.ship_a.select_listening` applies it:

- **A partner swap the base teacher gets wrong** is always listened to: its training label
  needs the ear.
- **A partner swap the base teacher gets right** is listened to on held-out reciters only,
  one per reciter and family (the first by salted rank, so with chance `1 / k` among that
  reciter's `k`), for evaluation. **The rest are not listened to.** As D10 does for the
  geminates the teacher holds, their training target is the teacher's own decode (the
  mushaf letter; ADR-0011 §3), and they are no verified site in any evaluation. On held-out
  reciters, recovery of teacher-right swaps is read on the one-per-reciter sample, weighted
  by the inverse of its chance.
- **A `كسب` swap** (`س`→`ش` on the `س` of the consonants `كسب`, which both models hear as
  `ش`) is listened to only in a sample of 20: reciters in salted order take one site each per
  round, so the sample spreads over reciters. A site's chance is its reciter's expected picks
  over every salted order of the reciters, divided by the reciter's sites
  (`ship_a.round_robin_shares`: full rounds, then `R` of the `E` eligible reciters in the
  partial round); here one partial round, so `(20 / 80) / n`. **The rest take the sample's verdict**, by this rule, fixed before
  any is heard: if a majority of the sampled sites with a verdict (`unclear` aside) are
  heard as one letter, every unsampled `كسب` site takes that letter as its label; with no
  majority they get no label and stay out of training and evaluation. Which letter that is,
  is left to the listening.
- **A geminate** is listened to when the teacher does not hold it (D10); **another letter**
  never (D9).

**What a score describes (owner decision, 2026-10-10).** The estimand is the **frozen
verified site set**: the pool's and the extension's Ship A sites as mined here, and the
frozen word sample. Scores describe that set, not all of Tadabur, so no weight expands a site
to the pool's frame or the extension's. The only weights are the inverse listening
probabilities (`ship_a.listening_weights`): 1 where a stratum is listened to whole, `k` for a
teacher-right swap, `1 / chance` for a `كسب` swap; and for a sampled word the inverse of its
selection probability within the sample's frame. Acceptance rules §1's weighting row is
amended to say so, and the #84 scorer takes these per-site weights
(`truth_scorer.score(design_weights=…)`).

{_table(["site", "mined", "listened"], rows)}
"""


def _rates(point: Mapping) -> tuple[str, str]:
    """One detector's precision (tp/flagged) and recall (tp/positives) on one split."""
    def value(x: float | None) -> str:
        return "—" if x is None else f"{x:.3f}"

    flagged = point["tp"] + point["fp"]
    return (f'{value(point["precision"])} ({point["tp"]}/{flagged})',
            f'{value(point["recall"])} ({point["tp"]}/{point["positives"]})')


def _section_guard(sizing: Mapping, guard: Mapping) -> str:
    calibration = guard["calibration"]
    chosen = calibration["operating_point"]
    detectors = (("segmenter", "the segmenter (today's, re-run on the fixture clips)"),
                 ("local_check", "the local check"),
                 ("segmenter_or_local_check", "either (the reference the guard leaves)"),
                 ("local_check_where_segmenter_did_not_cut",
                  "the local check where the segmenter did not cut"))
    rates = [[split, label, *_rates(report[name])]
             for split, report in calibration["splits"].items() for name, label in detectors]
    recorded = "; ".join("{} precision {}, recall {}".format(split, *_rates(p))
                         for split, p in guard["segmenter_on_fixture_rows"].items())
    strata = [[name, e["items"], e["wasl_dependent"], e["phonetizer_unsupported"],
               e["detector_paused"], e["unplaced"], e["decided_by_owner_note"],
               e["waqf_realized"]]
              for name, e in guard["strata"].items()
              if e["wasl_dependent"] or e["phonetizer_unsupported"]]
    taken = sizing["waqf_guard"]
    fixture_realized = sizing["fixture_report"]["waqf_realized"]
    cells = [[cell, e["sites"], e["held_out_listened"]] for cell, e in taken["cells"].items()]
    unplaced = ", ".join(f"{n} {split}" for split, n in
                         guard["fixture_boundaries_unplaced"].items())
    dropped = guard["fixture_test_boundaries_dropped_for_a_reciter_in_calibration"]
    overlap = "; ".join(f"reciter {r} lost {n} test boundaries" for r, n in dropped.items())
    guarded_test = calibration["splits"]["test"]["local_check_where_segmenter_did_not_cut"]
    test_precision, test_recall = _rates(guarded_test)
    return f"""## The waqf guard (#134)

A geminate made by joining two words (tanween or noon/meem idgham, a letter assimilated into
the next word), a ta marbuta read `ت`, or a word-final haraka exists only in wasl; a sun-letter
`لام` keeps its shaddah after a pause. The segmenter misses about one real pause in five, so
`tadabur.waqf_guard` re-realizes every site, worklist row and sampled word with a waqf at each
boundary of its word (through the phonetizer, both sides) and, where its prescribed mark
changes there, asks `tadabur.local_pause` whether the reciter paused: the VAD's frame-level
silence probability between the two words' aligned times, more sensitive than the segmenter.
An owner note `waqf` (or `wasl`) on a verdict decides instead. A site judged paused is
`waqf_realized`: it leaves every cell below (its verdict stays a training target).

**The two errors cost differently.** A false pause takes one site out of scoring. A missed
pause leaves a reference that only exists in wasl, so a correct waqf is scored as a missed
shaddah or a real mistake.

**Calibration** on the owner's waqf fixtures, one unit per adjudicated boundary (a pause when
any of its rows says `waqf`), on the clips whose word times place their words ({unplaced}
boundaries are on re-read or skipped clips). The fixtures' split predates the canonical
reciter ids; a reciter with boundaries in both splits keeps only its calibration ones, so the
splits are reciter-disjoint ({overlap or "no reciter was in both"}). The operating point is
selected on the
population the guard applies the check to, the {calibration["selected_on"]}: the pre-stated
rule ({calibration["rule"]}) picks silence probability ≥ {chosen["threshold"]} for
≥ {chosen["duration"]} s. Both curves (that population, and every calibration boundary for
reference) are in `waqf_guard.summary.json`. The segmenter row by row, as the fixtures
recorded it: {recorded}.

{_table(["split", "detector", "precision", "recall"], rates)}

**On test, where the guard applies the check** (the boundaries the segmenter did not cut):
precision {test_precision}, recall {test_recall}. The test n is small:
{guarded_test["positives"]} missed pauses, so these two numbers are rough.

Two failure patterns show in the errors:

- **A missed pause spills into the next boundary's window.** When the alignment places the
  next word's onset before the silence, the pause lands in the following boundary's window:
  a miss at the pause's own boundary and a false pause at the next (on test, clip
  `spk0018_S32_A65` at boundaries 3 and 4, and `spk0031_S6_A168` at 4 and 5).
- **Long silences at boundaries marked wasl.** Some false pauses are silences of 1.4 to
  2.4 s at boundaries the fixtures call wasl. They may be real pauses the correction-based
  review missed (the reviewer marked only the detector's errors), or more misplaced word
  times.

Per stratum (items wasl-dependent, the phonetizer unable to realize the waqf, judged paused
by the check, unplaced, decided by an owner note, and realized), at the verdicts committed when
the guard was built:

{_table(["stratum", "items", "wasl-dependent", "unsupported", "check: paused", "unplaced",
         "owner note", "waqf_realized"], strata)}

The sizing below leaves out {taken["sites"]} realized sites and {taken["words"]} sampled words
(per cell, sites and held-out listened sites); question 4 reads the P3.5 fixtures without the
{len(fixture_realized)} realized there:

{_table(["cell", "sites", "held-out listened"], cells)}
"""


def _section_budget(sizing: Mapping) -> str:
    rows = []
    for name, e in sizing["budget"]["families"].items():
        status = "insufficient evidence" if e["held_out_sparse"] else "sufficient"
        rows.append([name, e["sites"], e["partner_swaps"], e["base_already_right"], e["listened"],
                     f'{e["train"]["listened"]} ({e["train"]["listened_reciters"]})',
                     f'{e["held_out"]["listened"]} ({e["held_out"]["listened_reciters"]})',
                     status])
    rule = sizing["budget"]["sparse_rule"]
    directions = {name: entry["sites"]
                  for name, entry in sizing["budget"]["directions"].items() if "partner" in name}
    largest = max(directions, key=lambda name: (directions[name], name))
    sparse = [name for name, e in sizing["budget"]["families"].items() if e["held_out_sparse"]]
    soft = sizing["budget"]["cells"][SOFT_PAIRS_POOLED]
    return f"""## 1. Verification budget

Listened: the sites the section above selects ({sizing["budget"]["cells"][GEMINATES]["listened"]}
geminates, a census of those the base does not hold; the ~300-geminate slot of the first
plan). Listened sites (reciters) per half of the confirmed 50/50 split
(D11). A held-out cell with fewer than {rule["min_reciters"]} reciters or {rule["min_sites"]}
sites is sparse (§1): its family is reported as insufficient evidence (§7).

{_table(["family", "flagged", "partner swaps", "base already right", "listened",
         "train", "held out", "held-out status"], rows)}

The six soft pairs pooled hold {soft["held_out"]["listened"]} held-out sites from
{soft["held_out"]["listened_reciters"]} reciters. The largest single direction is {largest}
({directions[largest]} sites). Sparse on the held-out half: {", ".join(sparse) or "none"}.
"""


def _section_power(sizing: Mapping) -> str:
    rates = sizing["grids"]["true_rates"]
    head = ["cell", "held-out reciters", *(f"true {r:.0%}" for r in rates)]
    needed = [[f'{r["rate"]:.0%}', f'{r["floor"]:.0%}', r["size"], f'{r["power"]:.2f}']
              for r in sizing["sites_needed"]]
    recovery = _power_rows(sizing, RECOVERY)
    middle = len(rates) // 2
    reading = "; ".join(f"{row[0]} {row[2 + middle]}" for row in recovery)
    every = _power_rows(sizing, RECOVERY, EVERY_SITE)
    every0 = _power_rows(sizing, RECOVERY, EVERY_SITE, icc=0.0)
    return f"""## 2. Power on the held-out half

A cell's held-out sites are listened to; each is a real mistake with probability θ. The
candidate **recovers** a verified-correct site (commits the mushaf letter, or holds the
geminate) with a true rate, or **retains** a verified mistake (keeps committing what was
said). Each listened site weighs the inverse of its listening probability (the estimand is
the frozen site set): teacher-wrong swaps and geminates are a census (weight 1), teacher-right
swaps one per held-out reciter and family, the كسب swaps a sample. θ is the same in every
stratum (a planning assumption). Each of {sizing["repetitions"]:,} repetitions takes the §1
lower bound of the weighted rate on one site per reciter (chosen in proportion to its weight,
counted with its reciter's total weight), with the reciter bootstrap, or the Wilson bound
where the cell is sparse and its observations are equally weighted (one common weight,
whatever its value, as the #84 scorer reads it); a sparse cell with unequal weights has no
bound (§1). Reciter effects at latent ICC 0, 0.05 and 0.15 act on both
whether a site is a mistake and whether it is recovered. The pooled cells are also read on
every site with the reciter-clustered bootstrap, which the ICC does move.

Kish's effective size per cell under those weights (the word sample's is in §3):

{_table(["cell", "held-out sites", "effective size"],
        [[cell, e["sites"], e["effective"]] for cell, e in sizing["effective_sizes"].items()])}

**No floor is chosen here** (#128). Each cell shows the **highest floor the lower bound
reaches with {TARGET_POWER:.0%} power**, at θ = {SUMMARY_THETA} and ICC = {SUMMARY_ICC}
(every θ, ICC and floor is in `sizing.json`).

Reciters (one site each) a true rate needs for its lower bound to reach a floor with
{TARGET_POWER:.0%} power (no clustering; a size the grid steps to, so an upper estimate):

{_table(["true rate", "floor", "reciters", "power"], needed)}

**Recovery** on the held-out half:

{_table(head, recovery)}

At a true recovery of {rates[middle]:.0%}, the floor each cell reaches: {reading}.

**Retention** of verified mistakes (θ = {SUMMARY_THETA}; held-out mistakes are few):

{_table(head, _power_rows(sizing, RETENTION))}

Every site with the reciter bootstrap, recovery, at ICC 0.15 and at ICC 0:

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
    word_sizes = sizing["word_sample_effective"]
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
- **The random word sample** is the model-independent denominator for that: {word_sizes["words"]}
  words, one per held-out reciter (planning: {sizing["grids"]["word_correct_share"]:.0%} heard as
  correct), each weighing the inverse of its selection probability within the sample's frame
  (Kish effective size {word_sizes["effective"]}). It covers false flags from any cause, as the app
  grades them in #123's replay, and new errors anywhere in a word. It leaves out the rare
  events: per-family harm at a few per 1,000 carriers, real mistakes (a few words), cross-word
  geminates. Its power to show the candidate's weighted false-flag rate below `h448`'s, with
  the candidate adding no flag (paired, one word per reciter; `h448`'s app false-flag rate is
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
in all, in {len(worklist["sittings"])} sittings (runs of the queue of at most {SITTING_MINUTES} minutes; the words come last because a sampled word is
shown whole):

{_table(["sitting", "queue positions", "rows", "minutes"], sittings)}
"""


def _section_questions(worklist: Mapping) -> str:
    return f"""## Notes for the owner

- **§7 amended.** Acceptance rules §7 now names nine target pairs (D9, 2026-10-10); the #84
  scorer's frozen required cells follow `truth_sites.TARGET_PAIRS`. No threshold, margin or
  ceiling moved. The committed truth-baseline report (`docs/truth-baseline.md`) carries the
  two pairs' cells, all without support yet.
- **P3.5 safeguards on linked clips.** {worklist["p35_rows_on_linked_clips"]} of the 23 P3.5
  re-listens sit on clips tied to a recording `h448` trained on or `decode_evalset`. They
  stay on the worklist as §8's targeted safeguards; they were never `h448`-unseen evaluation.
- **Training on held-out reciters' other audio.** The exposure registry lists the split's two
  halves (`ship_a.verified_train`, `ship_a.verified_held_out`). Keeping the held-out reciters
  out of the probe's full-shard training stream is #94's decision: every one of them also
  has rows in `h448`'s training shards.
- **Floors and margins** (#128): every power here is over a grid; none is chosen.
"""


HEADER = """# Ship A sizing (#126)

Generated by `python -m training.ship_a_sizing_doc` from `tools/tadabur/ship_a/sizing.json`
(`python -m training.ship_a_sizing run`), `sites.summary.json`, `worklist.summary.json`,
`frame.json`, `mining/summary.json` / `mining/frame.json` and `waqf_guard.summary.json`
(`python -m tadabur.waqf_guard build`); do not edit by hand.

Ship A (ADR-0012, goals 1-2) trains on sites the owner verifies by ear: where `h448` swaps the
letters of a target pair, or collapses an in-word geminate. This page sizes
that listening and says what the held-out half can certify. It reads only `h448` and the base
teacher, never a candidate, and **recommends no threshold, floor or margin**: those are the
owner's (#128). The app outcome itself is #123's Muraja replay; nothing here models Muraja.
"""


def render(sizing: Mapping, sites: Mapping, worklist: Mapping, frame: Mapping,
           mining: Mapping, mining_frame: Mapping, guard: Mapping) -> str:
    totals = frame_totals(frame)
    return "\n".join([
        HEADER,
        _section_counts(sizing, sites, frame, totals),
        _section_mining(mining, mining_frame, sites),
        _section_checks(sites),
        _section_listening(sites),
        _section_guard(sizing, guard),
        _section_budget(sizing),
        _section_power(sizing),
        _section_harm(sizing),
        _section_mix(sizing, worklist),
        _section_worklist(worklist),
        _section_questions(worklist),
    ])


def load_inputs() -> tuple[dict, ...]:
    return tuple(json.loads(path.read_text(encoding="utf-8")) for path in (
        SIZING_PATH, SITES_SUMMARY_PATH, WORKLIST_SUMMARY_PATH, FRAME_PATH,
        MINING_SUMMARY_PATH, MINING_FRAME_PATH, GUARD_SUMMARY_PATH))


def main() -> None:
    write_text_atomically(DOC_PATH, render(*load_inputs()))
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
