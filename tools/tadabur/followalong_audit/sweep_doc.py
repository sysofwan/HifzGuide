"""``docs/followalong-oracle-sweep.md`` is rendered from the committed report (#308).

The report (``docs/followalong-oracle-sweep.json``) holds every number the doc shows: the
speech-span pace, the clean sweeps' false flags per mode/cause/position, the planted catch
rates, the protected and inserted splits, and the pattern list. :func:`render` is the only
writer, and the doc's test re-renders the committed report and compares bytes — a hand edit is
a failed test, not a silent divergence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

#: Where the rendered doc and the report it renders live.
REPORT_PATH = Path(__file__).resolve().parents[3] / "docs" / "followalong-oracle-sweep.json"
DOC_PATH = Path(__file__).resolve().parents[3] / "docs" / "followalong-oracle-sweep.md"

#: How the doc names each scoring of ``training.muraja_policy.AUDIT_SCORINGS``.
MODE_LABELS = {"strict": "strict", "shipped_default": "balanced (shipped default)"}


def render(report: Mapping) -> str:
    """The doc as text, from the report alone; deterministic, ending with one newline."""
    lines = [
        "# Follow-along audit: the oracle and planted-mistake sweeps (#308)",
        "",
        "Generated from `docs/followalong-oracle-sweep.json` by "
        "`python -m tadabur.followalong_audit oracle-sweep`; generated, do not edit it by hand.",
        "",
        "The engine is fed a perfect decode: the reference phonemes placed on the clock by the"
        " Oracle arm, so no model and no ear is involved. Every word the engine does not grade"
        " correct is therefore a follow-along or reference fault, not a reciter's mistake.",
        "",
        f"Seed {report['seed']}, Muraja `{report['muraja_commit']}`, modes "
        + ", ".join(MODE_LABELS[mode] for mode in report["modes"]) + ".",
        "",
    ]
    lines += _pace(report["pace"])
    lines += _clean("the tlog clips", report["clean"]["tlog"], modes=report["modes"])
    lines += _clean("the mushaf at the tlog pace", report["clean"]["mushaf"],
                    modes=report["modes"], mushaf=True)
    lines += _planted(report["planted"])
    lines += _patterns(report["patterns"])
    return "\n".join(_trim(lines)) + "\n"


def _pace(pace: Mapping) -> list[str]:
    return [
        "## Pace",
        "",
        f"{pace['characters_per_second']:.3f} Characters per second — the median over the"
        f" speech span of {pace['clips']:,} aligned clips (a clip's first word start to its"
        " last word end). The mushaf sweep lays every ayah's words end to end at this pace.",
        "",
    ]


def _clean(title: str, section: Mapping, *, modes: Sequence[str],
           mushaf: bool = False) -> list[str]:
    lines = [f"## Clean sweep: {title}", ""]
    if mushaf:
        lines.append(
            f"{section['ayat']:,} ayat; {section['graded']:,} graded and"
            f" {section['unreadable']} the phonetizer cannot read (tallied, never dropped), at"
            f" {section['characters_per_second']:.3f} Characters per second.")
        unreadable = section.get("unreadable_ayat") or []
        if unreadable:
            lines += ["", "Unreadable ayat: " + ", ".join(
                f"{entry['ayah']} ({entry['reason']})" for entry in unreadable) + "."]
    else:
        lines.append(f"{section['clips']:,} gating clips, each at its own phase A word times.")
    lines += ["A word with no kept grade — no row in the engine's `final`, or only `pending` —"
              " is counted as not graded, never as a finding."]
    lines += ["", "| mode | graded words | not graded | false flags per 1,000 words |",
              "| --- | ---: | ---: | ---: |"]
    for mode in modes:
        lines.append(
            f"| {MODE_LABELS[mode]} | {section['graded_words'][mode]:,} |"
            f" {section['not_graded'][mode]:,} |"
            f" {_rate(section['false_flags_per_1000'][mode]['all'])} |")
    lines += ["", "By cause and position (per 1,000 graded words):", "",
              "| mode | error-kind rates | position rates |",
              "| --- | --- | --- |"]
    for mode in modes:
        rates = section["false_flags_per_1000"][mode]
        causes = ", ".join(f"{cause} {_rate(rate)}"
                           for cause, rate in sorted(rates["error_kind"].items())) or "—"
        positions = ", ".join(f"{position} {_rate(rate)}"
                              for position, rate in sorted(rates["position"].items())) or "—"
        lines.append(f"| {MODE_LABELS[mode]} | {causes} | {positions} |")
    return [*lines, ""]


def _planted(planted: Mapping) -> list[str]:
    lines = [
        "## Planted mistakes",
        "",
        "One seeded mistake per kind per clip, planted before the reference is placed. **Caught**"
        " means the planted word's kept grade is below correct. A word with no kept grade — no"
        " row in the engine's `final`, or only `pending` — is `not_graded`: reported apart and"
        " counted as neither caught nor missed. Rates are caught over graded planted words.",
        "",
        "| kind | mode | position | word length | graded | caught | not_graded | rate |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for cell in planted["catch_rates"]:
        lines.append(
            f"| {cell['kind']} | {MODE_LABELS.get(cell['mode'], cell['mode'])} |"
            f" {cell['position']} | {cell['length_bin']} | {cell['n']:,} |"
            f" {cell['caught']:,} | {cell['not_graded']:,} | {_rate(cell['rate'])} |")
    unplantable = planted["unplantable"]
    lines += ["", "Kinds no clip could carry, with the number of clips that could not: "
              + (", ".join(f"{kind} ({count:,})" for kind, count in unplantable.items())
                 or "none") + "."]
    lines += ["", "### Haraka drops: protected exemptions", "",
              "A drop on و ا ء ي is labelled `protected`; every other letter is `other`.", "",
              "| mode | protected | graded | caught | not_graded | rate |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for cell in planted["protected_split"]:
        lines.append(
            f"| {MODE_LABELS.get(cell['mode'], cell['mode'])} | {cell['protected']} |"
            f" {cell['n']:,} | {cell['caught']:,} | {cell['not_graded']:,} |"
            f" {_rate(cell['rate'])} |")
    exemptions = planted["observed_exemptions"]
    lines += ["", "Further letters the engine never flagged a drop on (observed, not assumed "
              "exempt): " + (", ".join(
                  f"{entry['letter']} ({entry['planted']:,} planted)" for entry in exemptions)
                  or "none") + "."]
    lines += ["", "### Added words: copied word vs the word after it", "",
              "An added word is a copy of its neighbour inserted right after it. The harness"
              " reports the ayah's own phoneme words, so the copy has no word number of its own:"
              " the insertion can displace the grade of the copied word and of the word after"
              " it, and caught means a non-correct grade on either; the split is reported"
              " apart.", "",
              "| mode | graded | caught | on the copied word | on the word after | not_graded |"
              " rate |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cell in planted["inserted_split"]:
        lines.append(
            f"| {MODE_LABELS.get(cell['mode'], cell['mode'])} | {cell['n']:,} |"
            f" {cell['caught']:,} | {cell['caught_on_copied']:,} |"
            f" {cell['caught_on_following']:,} | {cell['not_graded']:,} |"
            f" {_rate(cell['rate'])} |")
    return [*lines, ""]


def _patterns(patterns: Sequence[Mapping]) -> list[str]:
    lines = [
        "## Patterns",
        "",
        "The clean sweeps' findings — every word the engine did not grade correct under a"
        " perfect decode — are pooled by cause (every grouping) and by reference form, where"
        " only the forms covering ten or more words are listed: a form seen once is not a"
        " pattern. A planted mistake the engine misses is catch capacity, not a finding, so it"
        " never enters a pattern. A pattern covering ten or more distinct ayah words is a"
        " candidate Muraja issue. The drafts are in the run note"
        " (`tadabur/followalong_audit/oracle_sweep/run_note.md`) and are not filed by this repo.",
        "",
        "| pattern | words | findings | candidate |",
        "| --- | ---: | ---: | --- |",
    ]
    for pattern in patterns:
        key = ", ".join(f"{dimension}: {pattern['key'][dimension]}"
                        for dimension in sorted(pattern["key"]))
        lines.append(f"| {key} | {pattern['words']:,} | {pattern['findings']:,} |"
                     f" {pattern['candidate']} |")
    for pattern in patterns:
        key = ", ".join(f"{dimension}: {pattern['key'][dimension]}"
                        for dimension in sorted(pattern["key"]))
        lines += ["", f"### {key} — {pattern['words']:,} words", ""]
        for example in pattern["examples"]:
            lines.append(
                f"- {example['surah']}:{example['ayah']} word {example['word']}"
                f" ({MODE_LABELS.get(example['mode'], example['mode'])}):"
                f" {example['quality']}, {example['error_kind']}, {example['position']},"
                f" boundary {example['boundary']}; form `{example['form']}`")
        lines += ["", "Replay evidence (first example, per-check statuses):", "", "```json",
                  json.dumps(pattern["examples"][0]["statuses"], ensure_ascii=False,
                             sort_keys=True),
                  "```"]
    return [*lines, ""]


def _rate(value: float | None) -> str:
    """A rate as text: ``None`` is a dash, and a small rate keeps three decimals (0.018, not 0.0)."""
    if value is None:
        return "—"
    return f"{value:.3f}" if value <= 1 else f"{value:.1f}"


def _trim(lines: list[str]) -> list[str]:
    while lines and not lines[-1]:
        lines.pop()
    return lines
