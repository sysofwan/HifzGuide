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
from collections.abc import Mapping, Sequence
from pathlib import Path

from tadabur.followalong_audit.atomic import write_text_atomically
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
    run = summary.get("run")
    lines += _population(summary["population"])
    lines += _run_identity(run)
    lines += _rates(summary["rates"], run)
    lines += _oracle(summary.get("oracle"))
    lines += _p2(summary["p2"])
    lines += _p3(summary["p3"])
    lines += _pools(summary["pools"], summary["s1_by_cause_and_base"],
                    summary.get("catch_loss"), oracle_pending=summary.get("oracle") is None)
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


def _run_identity(run: Mapping | None) -> list[str]:
    """What produced the numbers: weights, engine, latency, seeds, determinism and the digests."""
    if run is None:
        return []
    harness = run.get("harness") or {}
    seeds = run.get("seeds") or {}
    lines = [
        "## Run identity",
        "",
        f"- Muraja harness commit `{run['muraja_commit']}`; assets release `{run['release']}`.",
        f"- App-arm latency {run['latency_ms']} ms per inference; P2 seed {seeds.get('p2')};"
        f" P3 seed {seeds.get('p3')}.",
        f"- Driver sha256 `{run['driver_sha256']}`; harness binary sha256"
        f" `{harness.get('binary_sha256', '—')}`.",
        f"- `audio_manifest.jsonl.gz` sha256 `{run['audio_manifest_sha256']}`;"
        f" `word_grades.jsonl.gz` sha256 `{run['word_grades_sha256']}`.",
    ]
    if run.get("determinism") is not None:
        lines.append(f"- {_determinism(run['determinism'])}")
    if run.get("payload_probe") is not None:
        lines.append(f"- {_probe(run['payload_probe'])}")
    return [*lines, ""]


def _probe(probe: Mapping) -> str:
    """One clip repeated: the counterexample a seeded pair that agreed cannot show.

    ``_determinism`` reports one comparison, and a comparison that agrees bounds the decoder's
    stability without establishing it. The probe is the audit's own observation of that gap and
    is rendered beside the pair whichever way the pair came out: the doc must disclose a known
    variation even when the latest seeded sample is identical.

    The repetitions' outcome comes first, because a probe whose runs failed has no payloads to
    compare and must not read as one that agreed: fewer than two successful repetitions is
    reported as insufficient evidence, and every stability claim is about the runs that ran.
    """
    runs, failed = probe["runs"], probe["failed_runs"]
    succeeded = runs - len(failed)
    head = (f"`{probe['item']}` ran {runs} times at {probe['latency_ms']} ms: {succeeded}"
            f" succeeded, {len(failed)} failed")
    if succeeded < 2:
        return (f"One clip repeated (`app_arm/payload_probe.json`): {head} — too few successful"
                " repetitions to compare its payloads.")
    variants = len(probe["payloads"])
    if variants < 2:
        return (f"One clip repeated (`app_arm/payload_probe.json`): {head}; the successful runs'"
                " payload was identical in every one of them.")
    counts = ", ".join(f"{count}× `{digest[:12]}`" for digest, count in probe["payloads"].items())
    finding = (f"Known counterexample (`app_arm/payload_probe.json`): {head}; the successful runs'"
               f" decode took {variants} distinct payloads ({counts})")
    if probe["changed_words"]:
        words = ", ".join(f"word {entry['word']}" for entry in probe["changed_words"])
        finding += f", changing {len(probe['changed_words'])} word outcome(s) ({words})"
    else:
        finding += ", and their word table was identical"
    return finding + (" — the decoder is not bit-stable, and diagnosing that is S7's, not this"
                      " audit's.")


def _determinism(report: Mapping) -> str:
    """The seeded two-run check, as its sample found it: payloads compared, not just grades."""
    size = report["size"]
    differing = report.get("differences") or []
    failed = report.get("failed") or []
    rows_differ = report.get("rows_differ") or []
    if not differing and not failed:
        return (f"Determinism: the {size} seeded clips ran twice and all {size} driver payloads"
                " were byte-identical in this sample; a sample bounds, it does not prove,"
                " bit-identity (`app_arm/determinism.json`).")
    identical = size - len(differing) - len(failed)
    finding = (f"Determinism: the {size} seeded clips ran twice; {identical} of {size} driver"
               " payloads were byte-identical")
    if differing:
        changed = (f"{len(rows_differ)} of them changed a word's outcome"
                   if rows_differ else "their word tables did not change")
        finding += f", {len(differing)} differed in the decode and {changed}"
    if failed:
        finding += f", {len(failed)} failed to run"
    return finding + " (`app_arm/determinism.json`)."


def _rates(rates: Mapping, run: Mapping | None = None) -> list[str]:
    latency = f" at {run['latency_ms']} ms per inference" if run is not None else ""
    lines = [
        "## Flags per 1,000 graded words",
        "",
        f"The App arm ran the app's own CoreML decoder{latency}; **strict** is"
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
    presence = [entry for entry in disagreements
                if (entry["app"] == "ungraded") != (entry["pytorch"] == "ungraded")]
    lines += ["", f"Disagreeing words: {len(disagreements)} (the table below is exhaustive).",
              f"- the App arm flags what PyTorch passes: {len(app_only)}",
              f"- PyTorch flags what the App arm passes: {len(torch_only)}"
              + (" — the App arm's flag is below the PyTorch decode at these words."
                 if torch_only else ""),
              f"- graded on one side only (ungraded against a grade): {len(presence)}"]
    skipped = p2.get("skipped") or []
    if skipped:
        lines += ["", f"Skipped (no word table): {len(skipped)} clip(s): "
                      + ", ".join(skipped[:20]) + ("…" if len(skipped) > 20 else "") + "."]
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
    lines += _app_cadence(p3.get("app_cadence") or [])
    latencies = ", ".join(f"{value} ms" for value in p3["latencies_ms"])
    clips = f"{p3['clips']:,} clips"
    if p3.get("seed") is not None:
        clips += f" (seed {p3['seed']})"
    lines += ["", f"The App arm ran at {latencies} per inference on {clips}.", "",
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
    skipped = p3.get("skipped") or []
    if skipped:
        lines += ["", f"Skipped (no rows in one latency run): {len(skipped)} clip(s): "
                      + ", ".join(skipped[:20]) + ("…" if len(skipped) > 20 else "") + "."]
    return [*lines, ""]


def _app_cadence(entries: Sequence[Mapping]) -> list[str]:
    """The App arm's own previews per window, beside the device logs it is compared with.

    The ratio is pooled over clips (``sum(previews) / sum(windows)``), from the subset traces each
    latency run committed; flushes and the closing stop are neither, a clip that ran no full
    window still contributes its previews, and a clip whose trace is empty contributes nothing.
    Each exclusion has its own column, so the subset drawn is the clips pooled plus the traces
    without rows. The comparison bounds the cadence the audit actually ran — S1's caveat still
    holds: clip length and the output gate also move a device log's ratio, so a gap is not all
    the App arm's.
    """
    if not entries:
        return []
    lines = ["", "The App arm's own cadence on its seeded subsets — previews per full window, "
                 "pooled over the subset (flushes and the closing stop are neither). A clip that "
                 "ran no window still counts its previews and is listed in its own column; a clip "
                 "whose trace is empty — its session graded no word — has no word to pool and is "
                 "listed beside the clips that do:", "",
             "| traces | latency | clips with rows | previews | windows | preview:window | no "
             "window | no rows |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for entry in sorted(entries, key=lambda item: item["latency_ms"]):
        ratio = entry["ratio"]
        lines.append(f"| {entry['traces']} | {entry['latency_ms']} ms | {entry['clips']:,} | "
                     f"{entry['previews']:,} | {entry['windows']:,} | "
                     f"{'—' if ratio is None else f'{ratio:.2f}'} | "
                     f"{entry['clips_without_window']} | {entry['clips_without_rows']} |")
    lines += ["", "Device logs and the App arm read the same scheduler facet, but not under the "
                  "same conditions: the logs are whole device sessions (their length sets how "
                  "many previews fit), so a larger App ratio means these clips had more previews "
                  "per window, not that the scheduler differs."]
    return lines


def _pools(pools: Mapping, s1: Sequence[Mapping],
           catch_loss: Mapping | None = None, *, oracle_pending: bool = True) -> list[str]:
    lines = ["## Candidate strata", "",
             "| stratum | words | catch-loss words |", "| --- | ---: | ---: |"]
    for stratum in sorted(pools["strata"]):
        size = pools["strata"][stratum]
        lines.append(f"| {stratum}: {STRATUM_LABELS[stratum]} | {size:,} | "
                     f"{pools['catch_loss'][stratum]:,} |")
    pending = (" and S1-pending — pending S2 (#308)"
               if oracle_pending else ", with S1-pending emptied by S2's oracle grades")
    lines += ["", "The App-flag pool by cause and base-detector agreement — strata S1 (the App"
                  f" flags and the oracle passes){pending}:", "",
              "| cause | base detector flags | words |", "| --- | --- | ---: |"]
    lines += [f"| {entry['cause']} | {entry['base_flags']} | {entry['words']:,} |"
              for entry in s1]
    if catch_loss is not None:
        lines += ["", f"Round 2's catch-loss worklist (A5b) marks {catch_loss['marked']} words;"
                      f" {catch_loss['matched']} of them are in this population. The worklist's"
                      " words are look-tranche rows, so a confirm.1/confirm.2 population can"
                      " match none."]
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
    write_text_atomically(path, text)
