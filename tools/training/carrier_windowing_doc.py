"""Render ``docs/carrier-windowing.md`` from :func:`training.carrier_windowing.run`'s report (#146).

Every number on the page is read from the report; the prose states only what the report's
construction fixes (the population rule, the arms, the statistics). Nothing here recommends a
protocol or a threshold.
"""

from __future__ import annotations

from training.carrier_windowing import (
    CACHE,
    DIAGNOSTIC,
    UNSEEN,
    WEIGHTS,
    WINDOWING,
    WORD_SAMPLE,
)
from training.direction_table import (
    ALL_DIRECTIONS,
    GEMINATE,
    MUSHAF,
    OTHER,
    SOFT_PAIRS_FAMILY,
    SWAP,
)

#: The rows the cross-checks and the sensitivity show: the two pairs the owner hears
#: reliably, the soft pairs pooled, and the geminates.
HEADLINE_PAIRS = (("ط↔ق", "ط→ق"), ("ط↔ق", "ق→ط"), ("س↔ش", "س→ش"), ("س↔ش", "ش→س"),
                  (SOFT_PAIRS_FAMILY, ALL_DIRECTIONS))
HEADLINE = (*HEADLINE_PAIRS, (GEMINATE, GEMINATE))
SPARSE_MARK = "*"
_METHOD_MARKS = {"bootstrap": "", "wilson": " W", "tango": " T"}
_PROTOCOL_NAMES = {"whole": "whole", "stream_b0": "b0", "stream_b1": "b1",
                   "spans": "spans", "checks": "checks"}


def _pct(value, signed: bool = False) -> str:
    """Percent to one decimal, a rounded zero unsigned (never ``-0.0``)."""
    if value is None:
        return "–"
    if isinstance(value, str):  # an infinite adverse endpoint
        return "−∞" if value.startswith("-") else "+∞"
    rounded = round(100 * value, 1) + 0.0
    return f"{rounded:+.1f}" if signed else f"{rounded:.1f}"


def _bounds(entry: dict) -> str:
    if entry["method"] == "none":
        return " (n/c)"
    return f" [{_pct(entry['lower'])}, {_pct(entry['upper'])}]{_METHOD_MARKS[entry['method']]}"


def rate_cell(entry: dict | None, count: int | None) -> str:
    """``point [lower, upper] (count)`` in percent, the count unweighted."""
    if not entry or entry["point"] is None:
        return "–"
    return f"{_pct(entry['point'])}{_bounds(entry)} ({count})"


def gap_cell(entry: dict | None) -> str:
    if not entry or entry["point"] is None:
        return "–"
    return f"{_pct(entry['point'], signed=True)}{_bounds(entry)}"


def row_label(row: dict) -> str:
    if row["direction"] == GEMINATE:
        return "in-word geminate → single"
    if row["direction"] != ALL_DIRECTIONS:
        return row["direction"]
    if row["family"] == SOFT_PAIRS_FAMILY:
        return "**soft pairs, pooled**"
    return f"**{row['family']}, both**"


def _size(row: dict) -> str:
    size = f"{row['sites']} / {row['reciters']}"
    return f"{size} {SPARSE_MARK}" if row["insufficient_evidence"] else size


def _arm_name(arm_name: str) -> str:
    model, protocol = arm_name.split("/", 1)
    return f"{model} {_PROTOCOL_NAMES[protocol]}"


def _gap_name(name: str) -> str:
    first, second = name.split(" − ")
    return f"{_arm_name(first)} − {_arm_name(second)}"


def rate_table(rows: list[dict], arms: list[str], rate: str) -> list[str]:
    lines = ["| row | carriers / reciters | " + " | ".join(map(_arm_name, arms)) + " |",
             "|---|---|" + "---|" * len(arms)]
    for row in rows:
        cells = [rate_cell(row.get("arms", {}).get(a, {}).get(rate),
                           row.get("counts", {}).get(a, {}).get(rate)) for a in arms]
        lines.append(f"| {row_label(row)} | {_size(row)} | " + " | ".join(cells) + " |")
    return lines


def gap_table(rows: list[dict], gaps: list[str]) -> list[str]:
    lines = ["| row | carriers / reciters | " + " | ".join(map(_gap_name, gaps)) + " |",
             "|---|---|" + "---|" * len(gaps)]
    for row in rows:
        cells = [gap_cell(row.get("gaps", {}).get(g)) for g in gaps]
        lines.append(f"| {row_label(row)} | {_size(row)} | " + " | ".join(cells) + " |")
    return lines


def headline_rows(rows: list[dict], keys=HEADLINE) -> list[dict]:
    by_key = {(row["family"], row["direction"]): row for row in rows}
    return [by_key[key] for key in keys]


def uncommitted_table(uncommitted: dict) -> list[str]:
    lines = ["| arm | pair carriers with nothing committed | geminates with nothing aligned |",
             "|---|---|---|"]
    for arm_name, kinds in uncommitted.items():
        cells = [f"{k['uncommitted']} / {k['carriers']}"
                 + (f" ({100 * k['uncommitted'] / k['carriers']:.1f}%)" if k["carriers"] else "")
                 for k in (kinds["pair"], kinds[GEMINATE])]
        lines.append(f"| {_arm_name(arm_name)} | " + " | ".join(cells) + " |")
    return lines


def verdict_table(verdicts: dict) -> list[str]:
    lines = ["| pair | answered | heard the mushaf letter | heard the partner | another letter "
             "| unclear |", "|---|---|---|---|---|---|"]
    for pair, counts in verdicts.items():
        answered = sum(counts.values())
        cells = [str(counts.get(k, 0)) for k in (MUSHAF, SWAP, OTHER, "unclear")]
        name = "**soft pairs, pooled**" if pair == SOFT_PAIRS_FAMILY else pair
        lines.append(f"| {name} | {answered} | " + " | ".join(cells) + " |")
    return lines


def _population(report: dict) -> list[str]:
    population = report["population"]
    pool, cache, unseen = population["pool"], population[CACHE], population[UNSEEN]
    refused = ", ".join(f"`{reason}` {n}" for reason, n in population["not_in_cache"].items())
    return [
        "## Population",
        "",
        f"- **Pool.** The #83 mining pool: {pool['clips']:,} clips of {pool['reciters']} "
        "reciters, drawn from shards `h448` never trained on.",
        f"- **Cache.** The #85 haraka-gap cache streams the {cache['clips']:,} clips "
        f"({cache['reciters']} reciters, {cache['segments']:,} kept segments) that ADR-0007's "
        f"window build accepts. Left out with their reasons: {refused}. Those clips have no "
        "stream decode, so nothing here covers re-read, partly dropped, unphonetizable or "
        "over-long recitations.",
        f"- **`h448`-unseen.** Of the cached clips, {population['h448_linked_clips']} hold a "
        "recording tied, as itself, an alias or a probable copy, to `h448`'s training shards, "
        "`decode_evalset` or the sealed panel (`tadabur.power_frame.unseen_clips`, Ship A's "
        "rule). They are left out of the main tables: a weights comparison on audio one model "
        f"was trained or selected on is not one. That leaves **{unseen['clips']:,} clips, "
        f"{unseen['reciters']} reciters, {unseen['segments']:,} kept segments**; "
        f"{unseen['inclusion_probability_1']:,} of the clips have inclusion probability 1. "
        "The whole cache is reported as a sensitivity below.",
        "- **Carriers.** Every target-pair carrier and in-word geminate of those kept segments, "
        "from the realized reference alone: no carrier is chosen by a decode.",
        "- **Weights.** 1 / the clip's inclusion probability: the pool over-samples clips where "
        "the base teacher swapped a pair or collapsed a geminate in the census of the drawn "
        "reciters, so inference is to those reciters' frame, not to all Tadabur reciters.",
        "",
        "**The stream's reference.** A stream decode covers the clip's whole recitation span; "
        "it is read against the clip's kept-segment references joined by one space "
        f"(`{report['segment_separator']!r}`), the word separator they already use, so two "
        "segments' letters never merge into one run or one normalized group. Every cached "
        "clip keeps all its segments, so no recited word is missing from that reference. The "
        "alignment is one local Smith-Waterman over up to 40 s: a carrier outside its span, "
        "and audio past the last full window (never decoded), commit nothing and count as "
        "`other`. How often that happens, per arm (unweighted, `h448`-unseen carriers):",
        "",
        *uncommitted_table(report["pool"][UNSEEN]["uncommitted"]),
        "",
    ]


def _verdicts(report: dict) -> list[str]:
    return [
        "## Is a swap a model error? (#90's verdicts)",
        "",
        "Every rate above is against the mushaf. At Ship A's partner swaps where the base "
        "teacher's whole-segment decode was right, the owner's answered verdicts "
        "(`listening_session/verdicts.jsonl`, before any exclusion). These sites are selected "
        "and enter no estimate here:",
        "",
        *verdict_table(report["owner_verdicts"]),
        "",
        "Where the owner hears the mushaf letter at nearly every such site (`ط↔ق`, `س↔ش`), a "
        "swap against the mushaf reads as a model error. The soft pairs are the ones the owner "
        "finds hard to hear, so for them a swap rate here is agreement with the mushaf only.",
        "",
    ]


#: What each cross-check group's items are, for the page.
_APP_GROUPS = {
    DIAGNOSTIC: ("the diagnostic items", "`truth_sites/waqf_boundaries.jsonl`: whole clips chosen "
                 "for their waqf boundaries, not on any model outcome; `spans` is the whole "
                 "clip, not its waqf segments, so it is not the pool's `whole`"),
    WORD_SAMPLE: ("the word sample's segments", "`ship_a/word_sample.jsonl`: the pool segment of "
                  "each sampled word, one word per held-out reciter drawn; every pair carrier "
                  "of the segment, not only the sampled word's; `spans` is the segment whole"),
}


def _app_check(name: str, checks: dict) -> list[str]:
    title, what = _APP_GROUPS[name]
    rows = headline_rows(checks["rows"], HEADLINE_PAIRS)
    return [
        f"### {title[0].upper()}{title[1:]}",
        "",
        f"{checks['items']} items ({what}). {checks['h448_linked_items']} are tied to "
        f"`h448`'s training or selection and left out, leaving {checks['kept_items']} items of "
        f"{checks['reciters']} reciters. Swap rate:",
        "",
        *rate_table(rows, checks["arms"], SWAP),
        "",
        "Paired gaps in the swap rate (points):",
        "",
        *gap_table(rows, [*checks["gaps"][WINDOWING], *checks["gaps"][WEIGHTS]]),
        "",
    ]


def _app_checks(report: dict) -> list[str]:
    return [
        "## Cross-check: Muraja's own checks on the truth baseline's unselected items",
        "",
        "The truth baseline's items have committed decodes in three arms per model: `spans` "
        "(the item whole), `stream_b0` and `checks`, the transcript Muraja commits from its own "
        "check sequence (`training.truth_baseline`). Pair carriers only (the diagnostic items "
        "carry no word offsets to tell in-word geminates); each item weighs 1, so these are "
        "unweighted, labelled cross-checks, not population estimates.",
        "",
        *[line for name, checks in report["app_checks"].items()
          for line in _app_check(name, checks)],
    ]


def render(report: dict) -> str:
    pool = report["pool"][UNSEEN]
    main_rows = pool["rows"]
    settings = report["decode_settings"]
    bootstrap = report["bootstrap"]
    lines = [
        "# Windowing vs weights at target-pair carriers, unselected (#146)",
        "",
        "Generated by `python -m training.carrier_windowing` (from `tools/`); do not edit by "
        "hand. The machine-readable report is [`carrier-windowing.json`](carrier-windowing.json).",
        "",
        "> **Against the mushaf, not truth.** Every rate compares a decode with the realized "
        "reference. Nothing here was listened to, no threshold is chosen and no protocol is "
        "recommended: it measures how much of a swap rate moves with the window protocol and "
        "how much with the weights, on carriers no model outcome selected.",
        "",
        "## What was measured",
        "",
        "| | |",
        "|---|---|",
        "| decodes | `tools/training/haraka_gap_cache/{base,h448}.json` (#85), checked against "
        "their recorded checksums and source manifests |",
        f"| models | base teacher `{report['inputs']['models']['base']['ref']}` "
        f"(hub revision `{report['inputs']['models']['base'].get('hub_revision', '?')[:12]}`); "
        f"`h448` (checkpoint sha256 "
        f"`{report['inputs']['models']['h448'].get('checkpoint_sha256', '?')[:12]}`) |",
        f"| decode settings | {settings['weights_dtype']} weights, batch size "
        f"{settings['batch_size']}, {settings['device_type']}, policy `{settings['policy']}`, "
        "the same for every arm and both models |",
        "| arms | `whole`: each kept waqf segment decoded whole; `b0` / `b1`: the recitation "
        "streamed at commit block b (5 s windows, 1 s hop, startup rule, tail flush) |",
        "| carriers | the first of each run of a target-pair letter (a geminate is one carrier); "
        "a `ذ`, `ظ`, `س`, `ط` or `ق` carrier counts in both its pairs. In-word geminates: the "
        "first of a doubled consonant whose run stays inside one word |",
        "| outcome | what the #84 scorer reads at the carrier (`training.site_outcomes`, the "
        "`contrast_attribution.aligned_consonants` alignment), rated by `training.direction_table` "
        "as #90's verified sites are: **swap** = the partner letter "
        "(a geminate: one consonant), **mushaf** = the mushaf's letter (geminate held), "
        "**other** = anything else, another letter or nothing aligned |",
        "| weights | 1 / the clip's inclusion probability (`mining_pool/clips.jsonl`) |",
        f"| intervals | acceptance rules §1: reciter-clustered bootstrap, B = "
        f"{bootstrap['replicates']:,}, seed {bootstrap['seed']}, type-7 percentiles. A cell "
        "too sparse or degenerate for the bootstrap shows **W** (Wilson, only for independent "
        "equally weighted carriers) or **(n/c)** (no admissible bound) |",
        "| gaps | paired on the same carriers: both swap rates recomputed in every replicate "
        "of one reciter draw. **Windowing** = stream − whole per model; **weights** = `h448` − "
        "base per protocol |",
        "",
        "A rate is `percent [95% interval] (unweighted carriers with that outcome)`; a gap is "
        f"in points. `{SPARSE_MARK}` marks a row with fewer than 10 reciters or 20 carriers: it "
        "cannot support a claim.",
        "",
        *_population(report),
        "## Swap rate",
        "",
        *rate_table(main_rows, pool["arms"], SWAP),
        "",
        "## Paired gaps in the swap rate",
        "",
        "**Windowing** (stream − whole, per model), points:",
        "",
        *gap_table(main_rows, pool["gaps"][WINDOWING]),
        "",
        "**Weights** (`h448` − base, per protocol), points:",
        "",
        *gap_table(main_rows, pool["gaps"][WEIGHTS]),
        "",
        "## Mushaf rate",
        "",
        *rate_table(main_rows, pool["arms"], MUSHAF),
        "",
        "## Other rate",
        "",
        *rate_table(main_rows, pool["arms"], OTHER),
        "",
        "## Sensitivity: the whole cache",
        "",
        f"The same rows on all {report['population'][CACHE]['clips']:,} cached clips, the "
        f"{report['population']['h448_linked_clips']} `h448`-linked ones included. Swap rate:",
        "",
        *rate_table(headline_rows(report["pool"][CACHE]["rows"]), pool["arms"], SWAP),
        "",
        "Paired gaps (points):",
        "",
        *gap_table(headline_rows(report["pool"][CACHE]["rows"]),
                   [*pool["gaps"][WINDOWING], *pool["gaps"][WEIGHTS]]),
        "",
        *_verdicts(report),
        *_app_checks(report),
        "## Inputs",
        "",
        "The cache's files and the manifests it was decoded against, as its "
        "`provenance.json` records them (each checked before use; the staged-clip registry "
        "has grown since, so it is checked through the pool clips' audio, "
        f"`pool_audio` `{report['inputs']['pool_audio']}`), then the files read beside it:",
        "",
        "| file | sha256 |",
        "|---|---|",
        *[f"| `haraka_gap_cache/{name}` | `{sha}` |"
          for name, sha in report["inputs"]["haraka_gap_cache"].items()],
        *[f"| `{name}` | `{sha}` |" for name, sha in report["inputs"]["sources"].items()],
        *[f"| `{name}` | `{sha}` |" for name, sha in report["inputs"]["other"].items()],
        "",
    ]
    return "\n".join(lines)

