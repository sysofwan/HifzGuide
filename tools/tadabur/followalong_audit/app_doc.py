"""``docs/followalong-app-arm.md`` is generated from the committed summary, never hand-edited (#309).

The summary the run commits (``app_arm/summary.json``) holds every number the doc shows: the
population and its failures, the two scorings' flag rates per 1,000 graded words by cause with
their clip-bootstrap intervals, S2's oracle share (``null`` until #308 has run), P2's bridge, P3's
cadence and the candidate pools. :func:`render` is the only writer of the doc, and the doc's test
re-renders the committed summary and compares bytes — so a hand edit is a failed test, not a
silent divergence.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from training.word_flags import CAUSES

#: Where the run's summary and the doc it renders live.
APP_ARM_DIR = Path(__file__).parent / "app_arm"
SUMMARY_PATH = APP_ARM_DIR / "summary.json"
DOC_PATH = Path(__file__).resolve().parents[3] / "docs" / "followalong-app-arm.md"

#: What the doc calls each scoring of ``training.muraja_policy.AUDIT_SCORINGS``.
SCORING_LABELS = {"strict": "strict", "shipped_default": "balanced (shipped default)"}

#: How each stratum reads in the pools table.
STRATUM_LABELS = {
    "S5": "the App left the word ungraded",
    "S1": "the App flags and the oracle passes",
    "S1-pending": "the App flags, oracle pending S2 (#308)",
    "oracle-explained": "the App flags and the oracle flags too",
    "S3": "the App passes and the base or two detectors flag",
    "S4": "nothing flags",
    "residual": "the App passes and one non-base detector flags",
}


def render(summary: Mapping) -> str:
    """The doc as text, from the summary alone; deterministic, ending with one newline."""
    lines = [
        "# Follow-along audit: the App arm on placeable phone audio (#309)",
        "",
        "Generated from `app_arm/summary.json` by "
        "`python -m tadabur.followalong_audit report`; never edited by hand.",
        "",
    ]
    lines += _population(summary["population"])
    lines += _rates(summary["rates"])
    lines += _oracle(summary.get("oracle"))
    lines += _p2(summary["p2"])
    lines += _p3(summary["p3"])
    lines += _pools(summary["pools"], summary["s1_by_cause_and_base"])
    return "\n".join(_trim(lines)) + "\n"


def _population(population: Mapping) -> list[str]:
    failed = population.get("failed") or []
    clips = population["clips"]
    graded = clips - len(failed)
    lines = [
        "## Population",
        "",
        f"{clips:,} claim-gated confirm.1/confirm.2 gating clips — the ones #248's"
        " exclusion leaves (its ayat the phonetizer cannot read have no reference to grade).",
        f"{graded:,} of {clips:,} clips produced a word table "
        f"({(graded / clips if clips else 0.0):.1%})" + (":" if failed else "."),
    ]
    lines += [f"- {entry['item']}: {entry['reason']}" for entry in failed]
    return [*lines, ""]


def _rates(rates: Mapping) -> list[str]:
    lines = [
        "## Flags per 1,000 graded words",
        "",
        "The App arm ran the app's own CoreML decoder at 40 ms per inference; **strict** is"
        " Muraja's strict mode, **balanced** its shipped default. The denominator is the graded"
        " words (an ungraded word is stratum S5, not a pass); each interval resamples whole clips.",
        "",
        "| scoring | graded words | flagged words | per 1,000 | 95% CI |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for scoring, label in SCORING_LABELS.items():
        entry = rates[scoring]
        lines.append(f"| {label} | {entry['graded']:,} | {entry['flagged']:,} | "
                     f"{entry['per_1000']:.1f} | {_interval(entry['interval'])} |")
    lines += ["", "### By cause", "",
              "| cause | strict per 1,000 | 95% CI | balanced per 1,000 | 95% CI |",
              "| --- | ---: | --- | ---: | --- |"]
    for cause in CAUSES:
        cells = []
        for scoring in SCORING_LABELS:
            entry = rates[scoring]["causes"][cause]
            cells += [f"{entry['per_1000']:.1f}", _interval(entry["interval"])]
        lines.append(f"| {cause} | {' | '.join(cells)} |")
    return [*lines, ""]


def _oracle(oracle: Mapping | None) -> list[str]:
    if oracle is None:
        return ["## Oracle-explained share", "",
                "pending S2 (#308): S2's oracle word grades fill in the App flags the oracle"
                " shares and the per-cause share it explains.", ""]
    return ["## Oracle-explained share", "",
            f"The oracle flags {oracle['flagged']:,} of the App arm's {oracle['app_flagged']:,}"
            f" flagged words: {oracle['share']:.1%} of the App flags are oracle-explained.", ""]


def _p2(p2: Mapping) -> list[str]:
    subset, everything = p2["subset"], p2.get("all")
    bar = f"{p2['bar']:.0%}"
    lines = [
        "## P2: the CoreML-vs-PyTorch bridge",
        "",
        "The bridge reuses the committed round-3 candidate EMA PyTorch decode "
        f"`{Path(p2['source']).name}` — the same weights v1.0.7 ships, so no GPU run is spent"
        " re-deriving it — graded through the same session request as the App arm. Agreement is"
        " on the strict outcome (flagged / passed / ungraded), word by word.",
        "",
        f"Bar: {bar}.",
        "",
        f"Seeded subset: {subset['words']:,} words, {subset['agree']:,} agree "
        f"({subset['rate']:.1%}) — "
        + ("at or above the bar." if p2["verdict"] == "met" else f"below the {bar} bar."),
    ]
    if everything is not None:
        lines.append(f"All clips (secondary): {everything['words']:,} words, "
                     f"{everything['agree']:,} agree ({everything['rate']:.1%}).")
    disagreements = subset["disagreements"]
    app_only = [entry for entry in disagreements
                if entry["app"] == "flagged" and entry["pytorch"] != "flagged"]
    torch_only = [entry for entry in disagreements
                  if entry["app"] != "flagged" and entry["pytorch"] == "flagged"]
    lines += ["", f"Disagreeing words: {len(disagreements)}.",
              f"- the App arm flags what PyTorch passes: {len(app_only)}",
              f"- PyTorch flags what the App arm passes: {len(torch_only)}"
              + (" — the App arm's flag is below the PyTorch decode at these words."
                 if torch_only else "")]
    if disagreements:
        lines += ["", "| item | word | App arm | PyTorch |", "| --- | ---: | --- | --- |"]
        lines += [f"| {entry['item']} | {entry['word']} | {entry['app']} | {entry['pytorch']} |"
                  for entry in disagreements]
    return [*lines, ""]


def _p3(p3: Mapping) -> list[str]:
    lines = ["## P3: cadence", "", "Device logs, previews per window:", "",
             "| log | preview:window |", "| --- | ---: |"]
    for log in sorted(p3["device_ratios"]):
        ratio = p3["device_ratios"][log]
        lines.append(f"| {log} | {'—' if ratio is None else f'{ratio:.2f}'} |")
    latencies = ", ".join(f"{value} ms" for value in p3["latencies_ms"])
    lines += ["", f"The App arm ran at {latencies} per inference on {p3['clips']:,} clips.", "",
              "| scoring | words | changed | share |", "| --- | ---: | ---: | ---: |"]
    for scoring, label in SCORING_LABELS.items():
        report = p3["cadence"][scoring]
        lines.append(f"| {label} | {report['words']:,} | {report['changed']:,} | "
                     f"{report['share']:.1%} |")
    findings = [SCORING_LABELS[scoring] for scoring in SCORING_LABELS
                if p3["cadence"][scoring]["finding"]]
    lines += [""]
    if findings:
        lines.append(f"Follow-along finding: {', '.join(findings)} change more than 2% of word"
                     " outcomes between the latency settings.")
    else:
        lines.append("No scoring changes more than 2% of word outcomes between the latency"
                     " settings.")
    return [*lines, ""]


def _pools(pools: Mapping, s1: Sequence[Mapping]) -> list[str]:
    lines = ["## Candidate strata", "",
             "| stratum | words | catch-loss words |", "| --- | ---: | ---: |"]
    for stratum in sorted(pools["strata"]):
        size = pools["strata"][stratum]
        lines.append(f"| {stratum}: {STRATUM_LABELS[stratum]} | {size:,} | "
                     f"{pools['catch_loss'][stratum]:,} |")
    lines += ["", "The App-flag pool by cause and base-detector agreement — strata S1 (the App"
                  " flags and the oracle passes) and S1-pending — pending S2 (#308):", "",
              "| cause | base detector flags | words |", "| --- | --- | ---: |"]
    lines += [f"| {entry['cause']} | {entry['base_flags']} | {entry['words']:,} |"
              for entry in s1]
    return [*lines, ""]


def _interval(interval: Sequence[float]) -> str:
    low, high = interval
    return f"{low:.1f}–{high:.1f}"


def _trim(lines: list[str]) -> list[str]:
    while lines and not lines[-1]:
        lines.pop()
    return lines


def write_summary(summary: Mapping, path: Path = SUMMARY_PATH) -> None:
    """The summary as sorted, indented JSON, atomically: the doc's one input."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_doc(summary: Mapping, path: Path = DOC_PATH) -> None:
    """The doc rendered from ``summary``, atomically."""
    _write(Path(path), render(summary))


def _write(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
