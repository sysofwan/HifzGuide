"""Render the truth-site baseline report (``docs/truth-baseline.md``) from the scorer's JSON.

Presentation only: every number comes from :func:`training.truth_scorer.score`, and the doc
is regenerated with it by :mod:`training.truth_baseline`, never edited by hand.
"""

from __future__ import annotations

from collections import defaultdict

from tadabur.exclusions import REASONS
from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE
from training.muraja_policy import APPROXIMATIONS, RECORDED_CHECKS, SINGLE_DECODE
from training.truth_scorer import DIAGNOSTIC, HEADLINE, PAUSE, POPULATIONS, SAFEGUARD, SYNTHETIC

_POPULATION_TITLES = {
    HEADLINE: "Headline population (listening-session sites)",
    PAUSE: "Sukun at a pause (its own row, not the in-scope sukun population)",
    SAFEGUARD: "Targeted safeguards: the re-located P3.5 fixtures",
    SYNTHETIC: "Synthetic edits",
    DIAGNOSTIC: "Diagnostics only: weak labels (in no gate)",
}
_RATES = {
    CORRECT_SIDE: ("commit_rate", "committed_accuracy", "word_flags@allowances_on",
                   "word_flags@shipped_default", "word_flags@allowances_off", "coverage@shipped_default"),
    MISTAKE_SIDE: ("commit_rate", "committed_accuracy", "missed_mistakes", "silent_corrections",
                   "muraja_missed@allowances_on", "muraja_missed@allowances_off"),
}
_RATE_TITLES = {
    "commit_rate": "commit rate",
    "committed_accuracy": "committed accuracy",
    "word_flags@allowances_on": "app: flagged words, allowances on",
    "word_flags@shipped_default": "app: flagged words, shipped default",
    "word_flags@allowances_off": "app: flagged words, ADR-0012 allowances off",
    "coverage@shipped_default": "app: words graded, shipped default",
    "missed_mistakes": "missed mistakes",
    "silent_corrections": "silent corrections",
    "muraja_missed@allowances_on": "app: missed mistakes, allowances on",
    "muraja_missed@allowances_off": "app: missed mistakes, ADR-0012 allowances off",
    "spurious_haraka": "spurious haraka",
}
_APP_RATES = {
    CORRECT_SIDE: ("word_flags", "letter_flags", "coverage"),
    MISTAKE_SIDE: ("word_flags", "letter_flags"),
}
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
            if not rows:
                continue
            lines += [f"*{_RATE_TITLES[rate]}*, % [95% interval]", ""]
            lines += _rate_table(rows, arms, rate) + [""]
        comparisons = list(scored[0]["differences"])
        for rate in ("commit_rate", "committed_accuracy", rates[2]):
            if rate not in scored[0]["arms"][arms[0]]["rates"]:
                continue
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
        sampled[row["stratum"]] += row["sites"] + sum(row.get(r, 0) for r in REASONS)
        population[row["stratum"]] = row["stratum_population"]
    return all(sampled[stratum] == n for stratum, n in population.items())


_APPROXIMATION_TITLES = {
    RECORDED_CHECKS: "Recorded check sequence",
    SINGLE_DECODE: "Single-decode approximation",
}


def _approximations(muraja: dict) -> list[str]:
    """Each way the arms' checks were made, with the arms it applies to."""
    by_approximation: dict[str, list[str]] = defaultdict(list)
    for arm, approximation in muraja["approximations"].items():
        by_approximation[approximation].append(arm)
    lines = []
    for approximation in APPROXIMATIONS:
        if approximation in by_approximation:
            arms = ", ".join(f"`{a}`" for a in by_approximation[approximation])
            lines += [
                f"**{_APPROXIMATION_TITLES[approximation]}** ({arms}). {APPROXIMATIONS[approximation]}",
                "",
            ]
    return lines


def _recorded_vs_single(report: dict) -> list[str]:
    """Per pooled cell, model and scoring: the recorded check sequence's flagged words minus
    the single-decode approximation's on the same stream, paired."""
    pooled = [c for c in report["cells"] if c["sites"] and c["family"] == "all"]
    compared = pooled[0]["differences"] if pooled else {}
    pairs = [c for c in compared if c.endswith("/stream_b0") and "/checks - " in c]
    if not pairs:
        return []
    names = [s["name"] for s in report["muraja"]["scorings"]]
    lines = [
        "### The recorded check sequence against the single-decode approximation",
        "",
        "Flagged words (the site's word shows as not correct) with each item's recorded check",
        "sequence (`checks`) minus the same with the single-decode approximation on the deployed",
        "stream (`stream_b0`), paired on the same sites, points [95% interval]. On correct",
        "recitation a negative difference is fewer false flags; on real mistakes, fewer catches.",
        "",
        "Three things differ between the two, and this comparison does not separate them. The",
        "transcript differs: `checks` commits its own hops, featurised and stripped as the app",
        "does, and ends with the app's stop rather than a silence flush (its decode-level rates sit",
        "next to `stream_b0`'s in the results below). The placement differs: a single-decode check",
        "is cut from the one decode by spreading its characters evenly over time, a recorded check",
        "decodes exactly its own audio. And the checks are separate inferences: every single-decode",
        "check of a word repeats the one decode's reading of it, while recorded checks are",
        "inferences in their own context (a preview ends where the audio ends; an overlap has the",
        "window's own right context), so a word's checks can disagree, and `GradeStore` keeps the",
        "best grade any of them gave. That ratchet can hide real mistakes as well as false flags;",
        "these sites cannot show it until real-mistake sites exist.",
        "",
    ]
    for cell in pooled:
        lines += [f"**{cell['population']}, {cell['side']}** ({_size(cell)})", ""]
        lines += ["| paired difference | " + " | ".join(f"`{n}`" for n in names) + " |",
                  "|---|" + "---|" * len(names)]
        for pair in pairs:
            values = [_delta(cell["differences"][pair].get(f"word_flags@{n}")) for n in names]
            lines.append(f"| {pair} | " + " | ".join(values) + " |")
        lines.append("")
    return lines


def _app_section(report: dict, arms: list[str]) -> list[str]:
    """Pooled app outcomes per population, side and scoring."""
    muraja = report["muraja"]
    build = muraja["build"]
    lines = [
        f"Graded by Muraja's own engine at `{build['muraja_commit'][:12]}` (v1.0.27), compiled with",
        f"{build['compiler']} into `tools/muraja_harness` (see `tools/training/muraja_policy.py`).",
        "",
        *_approximations(muraja),
        f"Sites whose word has no counterpart on Muraja's reference (unplaced, never graded): "
        f"{muraja['unplaced_sites']}; sites placed on their word but whose own letter has no "
        f"counterpart (a flag on the word counts, its letter is unattributed): "
        f"{muraja['unattributed_sites']}.",
        "",
        "Scorings (balanced throughout; `None` keeps balanced's own flag):",
        "",
        "| scoring | `softPairsEnabled` | `shaddahSuppression` | tashkeel shown |",
        "|---|---|---|---|",
    ]
    for s in muraja["scorings"]:
        lines.append(
            f"| `{s['name']}` | {s['soft_pairs_enabled']} | {s['shaddah_suppression']} | {s['tashkeel_shown']} |"
        )
    lines += [
        "",
        "Pooled per population and side: **flagged words** (the site's word shows as not correct),",
        "**flagged letters** (the app marks the site's own letter) and, on correct recitation,",
        "**graded words** (the word shows any grade). % [95% interval].",
        "",
    ]
    names = [s["name"] for s in muraja["scorings"]]
    for cell in report["cells"]:
        if not cell["sites"] or cell["family"] != "all":
            continue
        lines += [f"**{cell['population']}, {cell['side']}** ({_size(cell)})", ""]
        lines += ["| arm | rate | " + " | ".join(f"`{n}`" for n in names) + " |",
                  "|---|---|" + "---|" * len(names)]
        for arm in arms:
            rates = cell["arms"][arm]["rates"]
            for rate in _APP_RATES[cell["side"]]:
                values = [format_estimate(rates.get(f"{rate}@{n}")) for n in names]
                lines.append(f"| {arm} | {rate.replace('_', ' ')} | " + " | ".join(values) + " |")
        lines.append("")
    return lines + _recorded_vs_single(report)


def _allowance_rows(entries: list[dict], arms: list[str]) -> list[str]:
    """The per-allowance view's table rows: on → off per arm, per allowance and side."""
    lines = [
        "| population | allowance | side | sites / reciters | "
        + " | ".join(f"{a} on → off" for a in arms) + " |",
        "|---|---|---|---|" + "---|" * len(arms),
    ]
    for entry in entries:
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
    return lines


def _allowance_section(report: dict, arms: list[str]) -> list[str]:
    lines = [
        "For each ADR-0012 allowance, on the sites it affects: the share of sites whose word the",
        "app flags, on correct recitation (false flags) and on real mistakes (caught), with the",
        "allowance on (`allowances_on`) and switched off alone. % [95% interval].",
        "",
        *_allowance_rows(report["allowances"], arms),
        "",
        "Allowances and sides with no affected site in a population are omitted. A caught rate",
        "needs real-mistake sites. Soft pairs are pooled, as Muraja has one switch (ADR-0012);",
        "the directional pair cells above keep the per-direction harm checks. The headline rows",
        "here pool both halves and frames of the verified sites; the per-half view is under",
        "\"The verified sites per half\".",
    ]
    return lines


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


_CAUSE_TITLES = {
    "tashkeel": "tashkeel",
    "soft_pair": "soft-pair consonant",
    "target_pair": "non-soft target pair (ذ↔ظ, س↔ش, ط↔ق)",
    "other_consonant": "another consonant",
    "shaddah": "shaddah (collapsed geminate)",
    "skipped": "skipped by the engine",
    "unrecorded": "no error recorded",
}


def _count_rate(entry: dict, total: int) -> str:
    """``k / n · weighted % [interval]``."""
    return f"{entry['words']} / {total} · {format_estimate(entry)}"


def _word_cause_table(view: dict, names: list[str]) -> list[str]:
    """One arm: per scoring, the false flags and their causes."""
    lines = ["| | " + " | ".join(f"`{n}`" for n in names) + " |",
             "|---|" + "---|" * len(names)]
    total = {n: view[n]["correct_words"] for n in names}
    lines.append("| **false flags** | " + " | ".join(
        _count_rate(view[n]["false_flags"], total[n]) for n in names) + " |")
    for cause, title in _CAUSE_TITLES.items():
        lines.append(f"| {title} | " + " | ".join(
            _count_rate(view[n]["causes"][cause], total[n]) for n in names) + " |")
    marks = sorted({m for n in names for m in view[n]["tashkeel_marks"]})
    for mark in marks:
        lines.append(f"| — tashkeel: {mark} | " + " | ".join(
            str(view[n]["tashkeel_marks"].get(mark, 0)) for n in names) + " |")
    return lines


def _word_sample_section(report: dict) -> list[str]:
    """The word sample's app outcome: what correctly recited words are flagged for."""
    sample = report["word_sample"]
    names = [s["name"] for s in report["muraja"]["scorings"]]
    arms = sample["arms"]
    first = arms[next(iter(arms))][names[0]]
    verdicts = ", ".join(f"{n} {v}" for v, n in sample["verdicts"].items())
    taken = "; ".join(f"{len(sample[r])} {r}" for r in REASONS if sample.get(r))
    lines = [
        "## What correctly recited words are flagged for: the word sample (#126, #61)",
        "",
        f"The {sample['words']} sampled words (one per held-out reciter, drawn with no model) were",
        f"judged whole by the owner: {verdicts}. The shared exclusion takes out "
        f"{taken or 'none'}, leaving {first['correct_words']} correctly recited words. Each word",
        "was replayed through Muraja's engine with the rest of its segment, and each false flag",
        "(a correctly recited word the app shows as not correct) is attributed to the errors of",
        "the grade the engine kept for it (`training/word_flags.py`): a word shown as a tashkeel",
        "error to its tashkeel errors (the mark the mushaf prescribes, and whether the decode",
        "dropped it or put another mark); a word shown as minor or worse to its consonant errors",
        "(a soft pair, a non-soft target pair, another consonant, or a collapsed geminate); a",
        "word the engine's skip marking passed over is `skipped`. A word can carry several",
        "causes, so the causes can sum to more than the false flags.",
        "",
        "Each cell is the count over the correctly recited words, then the weighted rate (each",
        "word weighs the inverse of its selection probability, from "
        f"{sample['weight_range'][0]:g} to {sample['weight_range'][1]:g} here) with",
        "its §1 interval. **The sample is small: one false flag moves a count by one in "
        f"{first['correct_words']}, and the",
        "intervals are wide.** A cause no word carries has no admissible interval (n/c).",
        "",
    ]
    for model in ("h448", "base"):
        arm = f"{model}/checks"
        if arm not in arms:
            continue
        lines += [f"**`{arm}`** (the recorded check sequence: the app outcome to read)", ""]
        lines += _word_cause_table(arms[arm], names) + [""]
    lines += [
        "False flags per arm (the `spans` and `stream_b0` arms are the single-decode",
        "approximation, for comparison; count, weighted % [interval]):",
        "",
        "| arm | " + " | ".join(f"`{n}`" for n in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for arm, view in arms.items():
        lines.append(f"| {arm} | " + " | ".join(
            _count_rate(view[n]["false_flags"], view[n]["correct_words"]) for n in names) + " |")
    mistakes = sorted({w for view in arms.values() for n in names for w in view[n]["mistakes"]})
    for word in mistakes:
        lines += ["", f"The word heard as a mistake (`{word}`), its app outcome per arm:", "",
                  "| arm | " + " | ".join(f"`{n}`" for n in names) + " |",
                  "|---|" + "---|" * len(names)]
        for arm, view in arms.items():
            lines.append(f"| {arm} | " + " | ".join(view[n]["mistakes"][word] for n in names) + " |")
    lines += ["", "Every false flag of the `checks` arms under the shipped default, with the "
              "kept grade's errors as Muraja logs them:", "",
              "| arm | word | causes | quality | errors |", "|---|---|---|---|---|"]
    for model in ("h448", "base"):
        view = arms.get(f"{model}/checks", {}).get("shipped_default")
        for word, f in (view or {}).get("flags", {}).items():
            causes = ", ".join(f["causes"] + [f"({m})" for m in f["marks"]])
            lines.append(f"| {model}/checks | `{word}` | {causes} | {f['quality']} | "
                         f"{' '.join(f['errors']) or '–'} |")
    return lines + [""]


_FRAME_TITLES = {
    "ship_a": "Ship A's verified sites (where `h448` swapped the pair or collapsed the geminate)",
    "reject_pile": "Reject-pile sites (#87: where the base teacher heard the partner)",
}
_HALF_TITLES = {"held_out": "held-out half", "train": "train half"}


def _entry_size(entry: dict) -> str:
    size = f"{entry['sites']} / {entry['reciters']}"
    return size + " (insufficient evidence)" if entry["insufficient_evidence"] else size


def _swap_rows(entries: list[dict], arms: list[str], label: list[str]) -> list[str]:
    """One row per non-empty entry: its ``label`` fields, size and swap rate per arm."""
    lines = ["| " + " | ".join(label) + " | sites / reciters | " + " | ".join(arms) + " |",
             "|" + "---|" * (len(label) + 1 + len(arms))]
    for e in entries:
        if not e["sites"]:
            continue
        names = ["**all**" if e[k] == "all" else str(e[k]) for k in label]
        values = [format_estimate(e["arms"][a]["swap"]) for a in arms]
        lines.append("| " + " | ".join(names) + f" | {_entry_size(e)} | " + " | ".join(values) + " |")
    return lines


def _gap_rows(entries: list[tuple[str, dict]], label: str) -> list[str]:
    names = list(entries[0][1]["gaps"])
    lines = ["| half | " + label + " | sites / reciters | " + " | ".join(f"`{n}`" for n in names)
             + " |", "|---|---|---|" + "---|" * len(names)]
    for half, e in entries:
        lines.append(f"| {_HALF_TITLES[half]} | {e[label]} | {e['sites']} / {e['reciters']} | "
                     + " | ".join(_delta(e["gaps"][n]) for n in names) + " |")
    return lines


def _verified_app(views: dict, names: list[str], arms: list[str]) -> list[str]:
    lines = [
        "### The verified sites per half: the app outcome",
        "",
        "The halves and frames were selected by different rules, so each is scored alone",
        "(`truth_scorer.score` over that subset, with the design weights). Flagged words per",
        "side, arm and scoring: on correct recitation a false flag, on a real mistake a catch.",
        "**Held-out is the half to read**; the train half was listened to only where the base",
        "teacher was wrong. % [95% interval].",
        "",
    ]
    for frame, by_half in views.items():
        for half, view in by_half.items():
            cells = view["app"]["cells"]
            if not cells:
                continue
            lines += [f"**{_FRAME_TITLES[frame]}, {_HALF_TITLES[half]}**", "",
                      "| side | sites / reciters | arm | " + " | ".join(f"`{n}`" for n in names)
                      + " |", "|---|---|---|" + "---|" * len(names)]
            for cell in cells:
                for arm in arms:
                    rates = cell["arms"][arm]["rates"]
                    lines.append(f"| {cell['side']} | {cell['sites']} / {cell['reciters']} | {arm} | "
                                 + " | ".join(format_estimate(rates.get(f"word_flags@{n}"))
                                              for n in names) + " |")
            lines += ["", "Per allowance, on the sites it affects:", ""]
            lines += _allowance_rows(view["app"]["allowances"], arms) + [""]
    return lines


def _directions_section(report: dict) -> list[str]:
    """The verified sites per frame and half: the app outcome, the direction table, the
    per-stratum and per-teacher-outcome rates, and the arm gaps at ط→ق."""
    views = report["verified"]
    arms = report["arms"]
    names = [s["name"] for s in report["muraja"]["scorings"]]
    lines = [
        "## The verified sites per half (#90)",
        "",
        "Ship A's verified sites (where `h448` swapped a target pair or collapsed an in-word",
        "geminate) and #87's reject-pile sites (where the base teacher heard the partner) are two",
        "frames, and each is split by the confirmed reciter split (D11). Every number here is",
        "read within one frame and half, weighted by the design weights",
        "(`ship_a/verified_design.jsonl`: a listened site weighs the inverse of its chance of",
        "being listened to).",
        "",
    ]
    lines += _verified_app(views, names, arms)
    lines += [
        "### Decode level, per direction",
        "",
        "Per target-pair direction (the mushaf's letter → its partner) and for in-word geminates,",
        "by verdict side, the share of sites where the decode **swapped**: committed the partner",
        "(a geminate: a single consonant). On the correct side that is a mishearing; on the",
        "mistake side it is the model hearing the mistake. `training/direction_table.py` is the",
        "single-model core (#57's diff compares two of its arms); `report.json` also has each",
        "row's `mushaf` and `other` rates. A direction with fewer than 10 reciters or 20 sites is",
        "insufficient evidence: read its family row. Directions with no verified site are left",
        "out.",
        "",
        "**These rates are conditional on selection.** Ship A's sites were mined where `h448`,",
        "streamed at b = 0 over the whole pool clip, committed the partner (or a single",
        "consonant), so on the correct side `h448`'s stream arms are high by construction; its",
        "`stream_b0` here streams the segment from its own first sample, so it need not reproduce",
        "the mining decode. The base teacher's whole-segment decode at mining defined each pair",
        "stratum (`base_right` / `base_wrong`): on the held-out half both are listened to and",
        "re-weighted to their mined proportions, so `base/spans` is low there by design; on the",
        "train half only `base_wrong` was listened to (owner decisions D10 and 2026-10-10 A), so",
        "`base/spans` is high (not 100%: the mining's base decode and this one differ on some",
        "items). The per-stratum rows below keep the strata apart.",
        "",
    ]
    for frame, by_half in views.items():
        for half, view in by_half.items():
            lines += [f"#### {_FRAME_TITLES[frame]}: {_HALF_TITLES[half]}", "",
                      "*swap rate*, % [95% interval]", ""]
            lines += _swap_rows(view["directions"], arms, ["family", "direction", "side"]) + [""]
    lines += ["### Decode level, per recorded stratum", "",
              "The same swap rates per stratum as mined (`ship_a:` the pool, `ship_a_mining:` the",
              "extension; the last field is the base teacher's outcome at mining), per side. "
              "% [95% interval].", ""]
    for half, view in views["ship_a"].items():
        lines += [f"#### {_FRAME_TITLES['ship_a']}: {_HALF_TITLES[half]}", ""]
        lines += _swap_rows(view["strata"], arms, ["stratum", "side"]) + [""]
    teacher = [(half, e) for half, view in views["ship_a"].items() for e in view["teacher"]
               if e["direction"] == "ط→ق" and e["side"] == "correct" and e.get("gaps")]
    if teacher:
        lines += [
            "### The arms at ط→ق (correct recitation), by the base teacher's outcome",
            "",
            "The sizing's \"base already right\" compares the base teacher decoding the whole",
            "span with `h448` streamed at b = 0. Both of those arms are selected here. Every site",
            "was mined where `h448`'s stream swapped. Each stratum is also defined by the base's",
            "whole-segment decode: in `base_right` it was right, so `base/spans` is selected low",
            "and a stream − spans difference is biased up; in `base_wrong` it was wrong, so",
            "`base/spans` is selected high and the difference is biased down. **These",
            "within-stratum differences are conditional on selections that bias them in opposite",
            "directions, and they cannot attribute the gap to windowing or to weights.** Whether",
            "windowing or weights drive `ط→ق` swaps in the population is a question for an",
            "unselected measurement, outside this report. Swap rates per arm, then the paired",
            "differences (`h448/spans - base/spans`: both whole; `h448/stream_b0 - h448/spans` and",
            "`base/stream_b0 - base/spans`: each model streamed against whole; `h448/checks -",
            "base/checks`: the app's checks). % and points [95% interval].",
            "",
        ]
        lines += _swap_rows([e for _, e in teacher], arms, ["teacher"]) + [""]
        lines += _gap_rows(teacher, "teacher") + [""]
    return lines


def _edit_check_section(report: dict) -> list[str]:
    """The synthetic-edit blind check per operation, with #90's call."""
    check = report["edit_check"]
    lines = [
        "## Edit types (#107's blind check, for #94)",
        "",
        f"{check['answered']} of {check['items']} blind-check items answered "
        "(`python -m tadabur.edit_check_summary`). Per operation, edits and their unedited",
        "decoys apart: the share judged natural (of natural or unnatural) and the share heard",
        "as labelled (an edit's changed state; a decoy's unchanged one). The call is #90's",
        "(`tadabur.edit_check_summary.CALLS`): only an operation that passes may feed the",
        "training probe (#94); a type nobody checked is `not checked` and may not either.",
        "",
        "| operation | call | role | items | natural | heard as labelled |",
        "|---|---|---|---|---|---|",
    ]

    def share(value) -> str:
        return "–" if value is None else f"{100 * value:.0f}%"

    for operation, call in check["type_verdicts"].items():
        roles = check["operations"].get(operation, {})
        if not roles:
            lines.append(f"| `{operation}` | {call} | – | 0 | – | – |")
        for role, cell in roles.items():
            answered = cell["items"] - cell["pending"]
            lines.append(
                f"| `{operation}` | {call} | {role} | {answered} | "
                f"{share(cell['natural_rate'])} ({cell['natural']} / "
                f"{cell['natural'] + cell['unnatural']}) | {share(cell['heard_as_labelled_rate'])} "
                f"({cell['heard_as_labelled']} / {cell['heard_as_labelled'] + cell['heard_otherwise']}) |")
    return lines + [""]


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
        "# Truth-site baseline: the base teacher and `h448` (#84, #90)",
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
        *(_word_sample_section(report) if report.get("word_sample") else []),
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
        *([f"{len(report['waqf_realized'])} site(s) Ship A's waqf guard realizes (#134: the "
           "mark only exists in wasl and the reciter paused there) are left out of every "
           "denominator, at their frozen weights: "
           + ", ".join(f"`{i}`" for i in report["waqf_realized"]) + "."]
          if report.get("waqf_realized") else []),
        *([f"{len(report['onset_trimmed'])} site(s) Ship A's onset guard judges trimmed (#138: "
           "the clip's opening sound is cut) are left out the same way: "
           + ", ".join(f"`{i}`" for i in report["onset_trimmed"]) + "."]
          if report.get("onset_trimmed") else []),
        *([f"{len(report['not_hafs'])} site(s) whose owner note denies the recitation is Hafs "
           "(#90, owner decision 2026-10-13) are left out the same way: "
           + ", ".join(f"`{i}`" for i in report["not_hafs"]) + "."]
          if report.get("not_hafs") else []),
        f"Records sharing one physical site (audio checksum, span, carrier, mark): "
        f"{len(report['physical_sites_merged'])} merged"
        + (", each under its directly adjudicated verdict (`report.json`)."
           if report["physical_sites_merged"] else "; every record is its own site."),
        "",
        "Every site is scored under three **arms** per model, all from the same staged audio.",
        "`spans` decodes the item (the whole clip, or the P3.5 segment) in one pass; `stream_b0`",
        "replays the deployed streaming protocol on the item (5 s windows from its first sample,",
        "each normalized on its own, 1 s hop, block 0 committed, startup rule and tail flush);",
        "`checks` records every update Muraja's transcriber would hand its engine over the item",
        "(each full window's hop and overlap, each 200 ms preview, and the stop's decode for the",
        "final alignment pass), one inference per check featurised as the app does, and its",
        "decode is the transcript the",
        "app commits from them (every hop; `training.decoding`, `Decoder.app_checks`). Every arm",
        "has an outcome at every site, so each comparison is paired on the same sites. Both models",
        "are decoded with",
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
        "## The app outcome",
        "",
    ]
    lines += _app_section(report, arms) if report["muraja"] else ["No app outcome in this report.", ""]
    lines += [
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
    if report["muraja"] and RECORDED_CHECKS in report["muraja"]["approximations"].values():
        lines += [
            "- **The `checks` arms are the app outcome to read.** They replay the checks Muraja's",
            "  transcriber would make from the audio (bypasses stated under \"The app outcome\";",
            "  they make it an estimate of the device's grades, not a bound on them); the `spans`",
            "  and `stream_b0` app outcomes are the single-decode approximation, kept as a",
            "  fallback and for comparison.",
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
    ]
    if report.get("verified"):
        lines += [
            "- **The headline population is Ship A's verified site set** (acceptance rules §1 as",
            "  amended 2026-10-10): sites mined where `h448` swapped a target pair or collapsed an",
            "  in-word geminate, plus #87's reject-pile sites where the base teacher heard the",
            "  partner, each weighted by the inverse of its chance of being listened to. It is",
            "  not a sample of all recitation: a correct-side rate there is a rate at sites a",
            "  model already got wrong. The word sample above is the model-free denominator.",
        ]
    lines += [""]
    if report.get("verified"):
        lines += _directions_section(report)
    if report.get("edit_check"):
        lines += _edit_check_section(report)
    lines += [
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

    lines += ["## Per-allowance view (ADR-0012)", ""] + _allowance_section(report, arms)
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
        "| Muraja configuration (the scorings replayed) and the app outcome | `muraja_policy.SCORINGS`, `muraja_policy.site_outcome`, `tools/muraja_harness` | `test_muraja_policy.py` |",
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
        "# from tools/, on the GPU box with the staged clips (the pool's and Ship A's mining",
        "# extension's in one directory); decodes only what the cache lacks",
        "python -m training.truth_baseline --audio-dir <clips> --decode-only",
        "# anywhere, torch-free, from the committed decodes",
        "python -m training.truth_baseline",
        "```",
        "",
        "Every `tools/tadabur/truth_sites/<name>.jsonl` is a truth-site file, so new sites are",
        "scored by the same command.",
        "",
    ]
    return "\n".join(lines)
