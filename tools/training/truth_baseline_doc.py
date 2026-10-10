"""Render the truth-site baseline report (``docs/truth-baseline.md``) from the scorer's JSON.

Presentation only: every number comes from :func:`training.truth_scorer.score`, and the doc
is regenerated with it by :mod:`training.truth_baseline`, never edited by hand.
"""

from __future__ import annotations

from collections import defaultdict

from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE
from training.truth_scorer import DIAGNOSTIC, HEADLINE, PAUSE, POPULATIONS, SAFEGUARD, SYNTHETIC

_POPULATION_TITLES = {
    HEADLINE: "Headline population (listening-session sites)",
    PAUSE: "Sukun at a pause (its own row, not the in-scope sukun population)",
    SAFEGUARD: "Targeted safeguards: the re-located P3.5 fixtures",
    SYNTHETIC: "Synthetic edits",
    DIAGNOSTIC: "Diagnostics only: weak labels (in no gate)",
}
_RATES = {
    CORRECT_SIDE: ("commit_rate", "committed_accuracy", "false_flags", "false_flags@allowances_off",
                   "coverage", "coverage@allowances_off"),
    MISTAKE_SIDE: ("commit_rate", "committed_accuracy", "missed_mistakes", "silent_corrections",
                   "muraja_missed", "muraja_missed@allowances_off"),
}
_RATE_TITLES = {
    "commit_rate": "commit rate",
    "committed_accuracy": "committed accuracy",
    "false_flags": "Muraja false flags (shipped default, balanced)",
    "false_flags@allowances_off": "Muraja false flags, every allowance off",
    "coverage": "Muraja coverage (shipped default, balanced)",
    "coverage@allowances_off": "Muraja coverage, every allowance off",
    "missed_mistakes": "missed mistakes",
    "silent_corrections": "silent corrections",
    "muraja_missed": "Muraja missed (shipped default, balanced)",
    "muraja_missed@allowances_off": "Muraja missed, every allowance off",
    "spurious_haraka": "spurious haraka",
}
#: The rates the single-decode bound is shown for: today's view under each approximation.
_BOUND_RATES = {CORRECT_SIDE: ("false_flags", "coverage"), MISTAKE_SIDE: ("muraja_missed",)}
SPARSE_MARK = "too small"


_METHOD_MARKS = {"bootstrap": "", "wilson": " W", "tango": " T"}
NOT_CERTIFIABLE = "n/c"


def _pct(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, str):  # an infinite adverse endpoint
        return "−∞" if value.startswith("-") else "+∞"
    return f"{100 * value:.1f}"


def _bounds(entry: dict) -> str:
    """`` [lower, upper]`` and the method mark, or the not-certifiable mark."""
    if entry["method"] == "none":
        return f" ({NOT_CERTIFIABLE})"
    return f" [{_pct(entry['lower'])}, {_pct(entry['upper'])}]{_METHOD_MARKS[entry['method']]}"


def format_estimate(estimate: dict | None) -> str:
    """``point [lower, upper]`` in percent; ``–`` for an undefined value."""
    if not estimate or estimate["point"] is None:
        return "–"
    return _pct(estimate["point"]) + _bounds(estimate)


def _delta(delta: dict | None) -> str:
    if not delta or delta["point"] is None:
        return "–"
    return f"{100 * delta['point']:+.1f}" + _bounds(delta)


def _cell_name(cell: dict) -> str:
    name = cell["label"]
    if cell["provisional"]:
        name += " (provisional)"
    return name


def _size(cell: dict) -> str:
    size = f"{cell['sites']} / {cell['reciters']}"
    return f"{size} ({SPARSE_MARK})" if cell["sparse"] else size


def _rate_table(cells: list[dict], arms: list[str], rate: str) -> list[str]:
    lines = [
        "| cell | sites / reciters | " + " | ".join(arms) + " |",
        "|---|---|" + "---|" * len(arms),
    ]
    for cell in cells:
        values = [format_estimate(cell["arms"][a]["rates"].get(rate)) for a in arms]
        lines.append(f"| {_cell_name(cell)} | {_size(cell)} | " + " | ".join(values) + " |")
    return lines


def _difference_table(cells: list[dict], comparisons: list[str], rate: str) -> list[str]:
    lines = [
        "| cell | sites / reciters | " + " | ".join(comparisons) + " |",
        "|---|---|" + "---|" * len(comparisons),
    ]
    for cell in cells:
        values = [_delta(cell["differences"][c].get(rate)) for c in comparisons]
        lines.append(f"| {_cell_name(cell)} | {_size(cell)} | " + " | ".join(values) + " |")
    return lines


def _population_section(population: str, cells: list[dict], arms: list[str]) -> list[str]:
    lines = [f"### {_POPULATION_TITLES[population]}", ""]
    for side in (CORRECT_SIDE, MISTAKE_SIDE):
        scored = [c for c in cells if c["side"] == side and c["sites"]]
        title = "Correct recitation" if side == CORRECT_SIDE else "Real mistakes"
        if not scored:
            lines += [f"**{title}:** no site with a verdict yet.", ""]
            continue
        lines += [f"**{title}**", ""]
        rates = list(_RATES[side])
        if any("spurious_haraka" in c["arms"][arms[0]]["rates"] for c in scored):
            rates.append("spurious_haraka")
        for rate in rates:
            rows = [c for c in scored if rate in c["arms"][arms[0]]["rates"]]
            lines += [f"*{_RATE_TITLES[rate]}*, % [95% interval]", ""]
            lines += _rate_table(rows, arms, rate) + [""]
        comparisons = list(scored[0]["differences"])
        for rate in ("commit_rate", "committed_accuracy", "false_flags" if side == CORRECT_SIDE else "missed_mistakes"):
            lines += [f"*{_RATE_TITLES[rate]}, paired difference*, points [95% interval]", ""]
            lines += _difference_table(scored, comparisons, rate) + [""]
    return lines


def _counts_table(report: dict) -> list[str]:
    lines = [
        "| population | stratum | stratum population | sites | with a verdict | items | reciters |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["sites"]:
        lines.append(
            f"| {row['population']} | `{row['stratum']}` | {row['stratum_population']} | "
            f"{row['sites']} | {row['with_verdict']} | {row['items']} | {row['reciters']} |"
        )
    return lines


def _all_census(report: dict) -> bool:
    """Whether every stratum's population equals the sites sampled in it (weights all 1)."""
    sampled: dict[str, int] = defaultdict(int)
    population: dict[str, int] = {}
    for row in report["sites"]:
        sampled[row["stratum"]] += row["sites"]
        population[row["stratum"]] = row["stratum_population"]
    return all(sampled[stratum] == n for stratum, n in population.items())


def _allowance_section(report: dict, arms: list[str]) -> list[str]:
    lines = [
        "For each allowance, on the sites it affects: Muraja's false-flag rate on correct",
        "recitation and missed-mistake rate on real mistakes, with the allowance on (today) and",
        "switched off alone, everything else today's configuration, under the single-decode",
        "approximation. % [95% interval].",
        "",
        "| population | allowance | side | sites / reciters | "
        + " | ".join(f"{a} on → off" for a in arms) + " |",
        "|---|---|---|---|" + "---|" * len(arms),
    ]
    for entry in report["allowances"]:
        for rate, block in entry["sides"].items():
            if not block["sites"]:
                continue
            size = f"{block['sites']} / {block['reciters']}"
            if block["sparse"]:
                size += f" ({SPARSE_MARK})"
            values = [
                f"{format_estimate(block[a]['on'])} → {format_estimate(block[a]['off'])}" for a in arms
            ]
            lines.append(
                f"| {entry['population']} | {entry['allowance']} | {rate.replace('_', ' ')} | "
                f"{size} | " + " | ".join(values) + " |"
            )
    lines += [
        "",
        "Allowances and sides with no affected site in a population are omitted. A",
        "missed-mistake rate needs real-mistake sites, and `ح↔ه` cannot receive a retirement",
        "verdict (§4) whatever its correct-side numbers.",
    ]
    return lines


def _muraja_section(report: dict, arms: list[str]) -> list[str]:
    """Which Muraja the rates replay, and the single-decode approximation they rest on."""
    config, views = report["muraja_config"], report["muraja_views"]
    off = views["allowances_off"]["config"]
    lines = [
        f"Muraja's grades replay Muraja `{config['revision'][:12]}` (v1.0.27; every rule is cited",
        "to its file and line in `tools/training/muraja_policy.py`). Two configurations are",
        "reported side by side:",
        "",
        f"- **shipped default**: mode `{config['mode']}`, tashkeel error detection on, soft pairs",
        "  and shaddah suppression on, the dropped-haraka exemption on و ا ء ي, `suppressHarakaDrop`",
        "  off, the waqf-final exemptions, the geminate-gap tashkeel discard and the word-initial",
        "  assimilation skip on (`muraja_policy.TODAY`). The acceptance rules read these rates;",
        "- **every allowance off**: the same scores and thresholds with no soft pair, no",
        "  dropped-haraka letter, and " + ", ".join(f"`{a}`" for a in _OFF_NAMES if not off[a]),
        "  switched off (`muraja_policy.allowances_off`).",
        "",
        "A false flag is a correct-recitation site whose word shows as not correct because of the",
        "site; coverage is the share of sites Muraja checked; Muraja missed is the share of real",
        "mistakes nothing showed against.",
        "",
        "**These rates rest on a single-decode approximation.** Muraja grades a word in every",
        "check that reaches it (each 1 s hop and each 200 ms preview) and keeps its best grade;",
        "here each item has one decode per arm. "
        + views["today"]["statement"],
        "",
        "The bound below grades today's configuration as if every word were an end word once"
        f" (`{views['every_word_ends']['approximation']}`): "
        + views["every_word_ends"]["statement"],
        "",
    ]
    by_population: dict[str, list[dict]] = defaultdict(list)
    for cell in report["cells"]:
        if cell["sites"] and cell["family"] == "all":
            by_population[cell["population"]].append(cell)
    lines += [
        "| population | side | rate | sites / reciters | "
        + " | ".join(f"{a} run ends → every word ends" for a in arms) + " |",
        "|---|---|---|---|" + "---|" * len(arms),
    ]
    for population in POPULATIONS:
        for cell in by_population.get(population, []):
            for rate in _BOUND_RATES[cell["side"]]:
                values = [
                    f"{format_estimate(cell['arms'][a]['rates'][rate])} → "
                    f"{format_estimate(cell['arms'][a]['rates'][rate + '@every_word_ends'])}"
                    for a in arms
                ]
                lines.append(
                    f"| {population} | {cell['side']} | {rate.replace('_', ' ')} | {_size(cell)} | "
                    + " | ".join(values) + " |"
                )
    return lines


_OFF_NAMES = (
    "shaddah_suppression", "suppress_haraka_drop", "final_at_waqf_tashkeel",
    "final_at_waqf_consonant", "group_has_gap", "leading_assimilation",
)


def _exclusions_table(report: dict) -> list[str]:
    if not report["exclusions"]:
        return ["No site is `unclear` or `pending`."]
    lines = [
        "| population | stratum | mark | prescribed | heard | sites |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["exclusions"]:
        lines.append(
            f"| {row['population']} | `{row['stratum']}` | {row['mark']} | {row['prescribed']} | "
            f"{row['heard']} | {row['sites']} |"
        )
    return lines


def _required_table(report: dict) -> list[str]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in report["required_cells"]:
        groups[(row["gate"], row["rule"], row["side"])].append(row)
    lines = [
        "| gate | rule | side | cells | cells with any site | cells not too small |",
        "|---|---|---|---|---|---|",
    ]
    for (gate, rule, side), rows in groups.items():
        with_sites = sum(r["sites"] > 0 for r in rows)
        supported = sum(r["sites"] >= 20 and r["reciters"] >= 10 for r in rows)
        lines.append(f"| {gate} | {rule} | {side} | {len(rows)} | {with_sites} | {supported} |")
    return lines


def render(report: dict) -> str:
    """The whole markdown document."""
    arms = report["arms"]
    by_population: dict[str, list[dict]] = defaultdict(list)
    for cell in report["cells"]:
        by_population[cell["population"]].append(cell)
    reproduction = report.get("p35_base_reproduction")
    fingerprints = report["fingerprints"]
    first = next(iter(fingerprints.values()))["spans"]

    lines = [
        "# Truth-site baseline: the base teacher and `h448` (#84)",
        "",
        "Generated by `python -m training.truth_baseline` (from `tools/`); do not edit by hand.",
        "The scorer and every definition it applies are frozen in `tools/training/truth_scorer.py`,",
        "`site_outcomes.py`, `muraja_policy.py` and `acceptance_stats.py` under",
        "[acceptance rules](acceptance-rules.md) §1 and §9. The machine-readable report is",
        "`tools/tadabur/truth_baseline/report.json`; the §8 power simulation's inputs are",
        "`tools/tadabur/truth_baseline/power_inputs.json`.",
        "",
        "This is the first time `h448` is measured against what reciters said. The numbers are",
        "reported as measured, with no recommendation.",
        "",
        "## What was scored",
        "",
        f"Truth-site files: {', '.join(f'`{name}`' for name in report['truth_site_files'])}.",
        "",
    ]
    lines += _counts_table(report)
    lines += [
        "",
        "P3.5 fixtures lost to re-location (#83) are listed in",
        "`tools/tadabur/truth_sites/p35_fixtures.summary.json`; weights are not inflated to cover",
        "them. A site's weight is its stratum population over the sites sampled in the stratum"
        + (": every stratum here is a census, so every weight is 1." if _all_census(report) else "."),
        f"Records sharing one physical site (audio checksum, span, carrier, mark): "
        f"{len(report['physical_sites_merged'])} merged"
        + (", each under its directly adjudicated verdict (`report.json`)."
           if report["physical_sites_merged"] else "; every record is its own site."),
        "",
        "Every site is scored under four **arms**: each model, decoded two ways from the same",
        "staged audio. `spans` decodes the item (the whole clip, or the P3.5 segment) in one pass;",
        "`stream_b0` replays the deployed streaming protocol on the item (5 s windows from its",
        "first sample, each normalized on its own, 1 s hop, block 0 committed, startup rule and",
        "tail flush; `training.decoding`). Every arm has an outcome at every site, so each",
        "comparison is paired on the same sites. Both models are decoded with",
        f"`{first['weights_dtype']}` weights at batch size {first['batch_size']} on "
        f"{first['device_type']} (inference policy `{first['policy']}`):",
        "",
        "| model | reference | modes |",
        "|---|---|---|",
    ]
    for name, records in fingerprints.items():
        modes = ", ".join(f"`{record['mode']}`" for record in records.values())
        lines.append(f"| {name} | `{records['spans']['model']}` | {modes} |")
    lines += [
        "",
        "Intervals follow §1. Unmarked, the reciter-clustered bootstrap (B = "
        f"{report['bootstrap']['replicates']:,}, seed {report['bootstrap']['seed']}, type-7",
        "percentiles, an undefined replicate at the adverse endpoint, shown as ±∞). A cell",
        "that is sparse (fewer than 10 reciters or 20 sites) or whose replicates are all",
        "identical gets an exact bound only where its sites are independent and equally",
        "weighted: **W** marks a Wilson bound, **T** a Tango bound. Otherwise the value carries",
        f"**({NOT_CERTIFIABLE})**: no bound is admissible and any rule read from it is",
        f"`cannot_certify`. A cell is also marked **{SPARSE_MARK}** when it has fewer than 10",
        "reciters or 20 sites: it cannot support a claim. `–` is an undefined value (a zero",
        "denominator). Rates are in percent; differences in points.",
        "",
        "## How Muraja is replayed",
        "",
    ]
    lines += _muraja_section(report, arms)
    lines += [
        "",
        "## Read this first",
        "",
    ]
    if not any(c["sites"] for c in by_population.get(HEADLINE, [])):
        lines += [
            "- **No headline-population site exists yet.** Every headline cell, and therefore",
            "  every gate of §2, §3 and §5, is empty until the listening session (#87) adds",
            "  directly adjudicated sites. Nothing here is a headline rate.",
        ]
    if not any(c["sites"] for c in report["cells"] if c["side"] == MISTAKE_SIDE):
        lines += [
            "- **No real-mistake site with a verdict exists yet.** Missed mistakes, silent",
            "  corrections and every mistake-side rate are undefined.",
        ]
    lines += [
        "- **Sukun commit rate is 0 by construction** while a model has no sukun class: at a",
        "  sukun site the empty slot is all it can emit (§1, C). The informative sukun number is",
        "  the spurious-haraka rate.",
        "- **The P3.5 sites were selected on the base teacher's errors.** #83 kept a fixture's",
        "  site only where the base decode (whole span, bf16, batch 1) showed the labelled",
        "  contrast against the realized reference, so on these correct-recitation sites the base",
        "  teacher's committed accuracy is fixed near 0 by selection, not measured.",
    ]
    if reproduction:
        lines += [
            f"  The base `spans` decode here reproduces the re-location decode on "
            f"{reproduction['identical']} of {reproduction['items']}",
            "  P3.5 items. `h448` was not selected on, so its numbers there are its behaviour at",
            "  known base-teacher errors, not a rate over correct recitation.",
        ]
    lines += [
        "- **Shaddah is provisional** until #92 freezes its held / not held / unsure state",
        "  table: today a single consonant counts as committing `not_held`.",
        "",
        "## Results",
        "",
    ]
    for population in POPULATIONS:
        cells = by_population.get(population, [])
        if not any(c["sites"] for c in cells):
            if population == HEADLINE:
                lines += [f"### {_POPULATION_TITLES[population]}", "", "No site yet.", ""]
            continue
        lines += _population_section(population, cells, arms)

    lines += ["## Per-allowance view (§4)", ""] + _allowance_section(report, arms)
    lines += [
        "",
        "## Excluded sites",
        "",
        "`unclear` and `pending` sites leave every denominator. Each cell in `report.json` carries",
        "the best and worst case of its commit rate and committed accuracy with the exclusions",
        "that could belong to it counted as successes or as failures.",
        "",
    ]
    lines += _exclusions_table(report)
    lines += [
        "",
        "## Support for the required cells",
        "",
        "The required cells of §2, §3 and §5 are frozen in `truth_scorer.REQUIRED_CELLS`",
        "(directional pair cells, `ذ↔ظ` included; shaddah provisional). Their support today, all",
        "in the headline population except the pause row of §5's spurious haraka (the mistake",
        "side of that guard, sukun said where a haraka was prescribed, is its own required row):",
        "",
    ]
    lines += _required_table(report)
    lines += [
        "",
        "## Where each §9 definition is frozen",
        "",
        "| §9 item | frozen in | pinned by |",
        "|---|---|---|",
        "| flagged at decode level (F), and an empty or unaligned slot | `site_outcomes` (module doc, `SiteOutcome.flagged`) | `test_site_outcomes.py` |",
        "| substitutions, several marks, insertions, unaligned and wrong carriers | `site_outcomes`, `tashkeel_eval.carrier_readings` | `test_site_outcomes.py`, `test_tashkeel_eval.py` |",
        "| consonant commitment | `contrast_attribution.aligned_consonants` | `test_contrast_attribution.py`, `test_site_outcomes.py` |",
        "| Muraja configuration, word grading, ratchet and single-decode approximation | `muraja_policy.TODAY`, `muraja_policy.grade_item`, `muraja_policy.single_decode_cycles` | `test_muraja_policy.py` |",
        "| allowance-affected populations | `muraja_policy.ALLOWANCES` | `test_muraja_policy.py` |",
        "| streaming: startup, tail flush, per-window normalization | `decoding` (`confirmed-stream-v2-flush`) | `test_decoding.py` |",
        "| streaming: window phase (the item's first sample) and pairing (every arm scores every site) | `truth_baseline`, `truth_scorer.outcomes_by_arm` | `test_truth_scorer.py` |",
        "| teacher agreement | `acceptance_stats.agreement_terms`, `project_to_teacher` | `test_acceptance_stats.py` |",
        "| reciter split | `acceptance_stats.reciter_half` | `test_acceptance_stats.py` |",
        "| required cells | `truth_scorer.REQUIRED_CELLS` | `test_truth_scorer.py` |",
        "| intervals, sparse cells, verdicts, aggregation (§1) | `acceptance_stats` | `test_acceptance_stats.py` |",
        "",
        "## Re-running",
        "",
        "```bash",
        "# from tools/, on the GPU box with the staged clips; decodes only what the cache lacks",
        "python -m training.truth_baseline --audio-dir /root/scratch/issue-83/stage/clips",
        "# anywhere, torch-free, from the committed decodes",
        "python -m training.truth_baseline",
        "```",
        "",
        "Every `tools/tadabur/truth_sites/<name>.jsonl` is a truth-site file, so new sites are",
        "scored by the same command.",
        "",
    ]
    return "\n".join(lines)
