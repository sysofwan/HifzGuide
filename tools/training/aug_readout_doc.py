"""Render ``docs/augmentation-readout.md`` from the readout's ``look.json`` and ``report.json`` (#95).

Torch-free and harness-free: numbers in, Markdown out. Every table shows each model's rate and
the round's paired differences (#95: candidate − ``h448`` and candidate − control; the consonant
round: candidate − C95, with C95's own row beside the candidate's).
"""

from __future__ import annotations

from collections.abc import Mapping

from training.aug_readout_metrics import PER_1000
from training.aug_readout_spec import AUG95, RoundSpec
from training.aug_readout_verdict import BAR_TEXT

ORDER = ("candidate", "c95", "control", "h448", "base")
LABELS = {"c95": "c95 (the baseline)", "base": "base (the teacher)"}
VERDICT_TEXT = {"pass": "pass", "fail": "fail", "cannot_certify": "cannot certify"}


def _number(value, sign: str = "") -> str:
    """A value to two decimals (``sign="+"`` writes a sign), an infinite end, or a dash."""
    if value is None:
        return "–"
    if isinstance(value, str):
        return {"inf": "+∞", "-inf": "−∞"}.get(value, value)
    return f"{value:{sign}.2f}"


def _signed(value) -> str:
    return _number(value, "+")


def interval_text(interval: Mapping, signed: bool = False) -> str:
    """One interval as a reader sees it: its two ends, or why it has no bound."""
    if interval["method"] == "none":
        return f"(no bound: {interval['reason']})"
    show = _signed if signed else _number
    return f"[{show(interval['lower'])}, {show(interval['upper'])}]"


def table(title: str, view: Mapping | None) -> list[str]:
    """One compare() view: a row per model, then a row per paired difference."""
    if view is None:
        return [f"**{title}**: not read (an input is missing); cannot certify.", ""]
    unit = "per 1,000" if view["scale"] == PER_1000 else "points"
    lines = [f"**{title}** ({unit}; {view['observations']} observations in "
             f"{view['clusters']} clusters)", "", "| | value | 95% interval | numerator / denominator |",
             "|---|---|---|---|"]
    for model in ORDER:
        if model in view["models"]:
            row = view["models"][model]
            lines.append(f"| {LABELS.get(model, model)} | {_number(row['point'])} | {interval_text(row['interval'])} | "
                         f"{row['num']:g} / {row['den']:g} |")
    for name, row in view["comparisons"].items():
        lines.append(f"| {name} | {_signed(row['point'])} | "
                     f"{interval_text(row['interval'], signed=True)} | paired |")
        if "veto_interval" in row:
            lines.append(f"| {name} (99.5%) | {_signed(row['point'])} | "
                         f"{interval_text(row['veto_interval'], signed=True)} | paired |")
    return lines + [""]


def coverage_section(coverage: Mapping, look: Mapping, report: Mapping) -> list[str]:
    """Each frozen population's decoded and evaluated size, its excluded keys and the statement
    of what the verdict does not cover (review R1.1)."""
    order = ("tarteel_v1.look", "tarteel_v1.confirm", "qud_half")
    if set(coverage["populations"]) != set(order):
        raise ValueError(f"coverage records {sorted(coverage['populations'])}, not {list(order)}")
    populations = {name: coverage["populations"][name] for name in order}
    evaluated = {"tarteel_v1.look": look["items"],
                 "tarteel_v1.confirm": report["sizes"]["tarteel_v1_confirm"],
                 "qud_half": report["sizes"]["qud_half"]}
    for name, row in populations.items():
        if row["evaluated"] != evaluated[name]:
            raise ValueError(f"coverage {name}: {row['evaluated']} evaluated, the readout read "
                             f"{evaluated[name]}")
    out = ["## 3. Coverage of the frozen populations", "", coverage["statement"], "",
           "| population | decoded | evaluated | excluded | excluded canonical word "
           "occurrences |", "|---|---|---|---|---|"]
    for name, row in populations.items():
        out.append(f"| {name} | {row['decoded']} | {row['evaluated']} | {row['excluded']} | "
                   f"{row['excluded_words']} |")
    out.append("")
    for name, row in populations.items():
        out += [f"**{name}**, the {row['excluded']} excluded: {row['reason']}.", ""]
    out += ["The excluded keys, one per line:", ""]
    for name, row in populations.items():
        out += [f"`{name}`:", "", "```text", *row["excluded_keys"], "```", ""]
    return out


def _condition_reading(condition: Mapping) -> str:
    """One verdict row's reading, from whatever the condition recorded."""
    if "bound" in condition:
        return (f"point {_signed(condition.get('point'))}, {condition.get('end')} bound "
                f"{_signed(condition['bound'])} ({condition.get('method')})")
    if "gap" in condition:
        return (f"reduction {_number(condition['reduction'])} of a gap of "
                f"{_number(condition['gap'])}; needs {_number(condition['required_reduction'])}")
    if "reduction" in condition:
        return (f"reduction {_number(condition['reduction'])}; needs "
                f"{_number(condition['required'])}")
    if "difference" in condition:
        return f"candidate − C95 {condition['difference']:+d} sites"
    if "verdict" in condition and condition.get("reason"):
        return condition["reason"]
    if "candidate" in condition:
        return f"candidate {_number(condition['candidate'])}, C95 {_number(condition['c95'])}" \
            if "c95" in condition else \
            f"candidate {_number(condition['candidate'])}, h448 {_number(condition['h448'])}"
    return condition.get("reason", "")


def render(look: Mapping, report: Mapping, coverage: Mapping | None = None,
           spec: RoundSpec = AUG95) -> str:
    """The document. ``coverage`` is :func:`training.aug_readout.coverage`'s record (committed
    as ``coverage.json``); without one the scope section is omitted."""
    if spec.name == "consonant1":
        return _render_consonant(look, report, coverage, spec)
    verdict = report["verdict"]
    out = [
        "# The augmentation experiment's readout (#95)",
        "",
        "Generated by `python -m training.aug_readout doc` from `tools/tadabur/aug_readout/"
        "look.json` and `report.json`. The bars are the owner's, pre-registered on #128 on "
        "2026-10-15 before any candidate result existed; they are read as written. "
        "Candidate: `h448` continued with augmented input (#94; `aug94_candidate_v2`, recipe "
        "`aug94-mixture-v2`, the re-run after the first candidate crashed); control: the same "
        "run without augmentation (`aug94_control`). Nothing here trains, sets a threshold or touches the sealed panel.",
        "",
        f"## Verdict: {VERDICT_TEXT[verdict['verdict']]}",
        "",
        "| condition | bar (#128) | reading | verdict |",
        "|---|---|---|---|",
    ]
    for c in verdict["conditions"]:
        out.append(f"| {c['name']} | {BAR_TEXT[c['name']]} | {_condition_reading(c)} | "
                   f"{VERDICT_TEXT[c['verdict']]} |")
    attribution = verdict["attribution"]
    control = attribution.get("control_vs_h448")
    out += ["", f"**Against the control:** {attribution['reading']} (main bound against "
                f"`h448`: {VERDICT_TEXT[attribution['against_h448']]}; against the control: "
                f"{VERDICT_TEXT[attribution['against_control']]}). Described, never part of "
                "the verdict: control − `h448` "
                + (f"{_signed(control['point'])} per 1,000 "
                   f"{interval_text(control['interval'], signed=True)}." if control else "–."), "",
            "**Pre-registered edge rules** (#95, 2026-10-16, before any candidate decode): an "
            "`h448` − teacher gap on the confirm half that is not positive makes the quarter-gap "
            "bar \"cannot certify\", an additional condition the bar presumes; qdat's consistency "
            "is required for a pass. **A \"cannot certify\" is resolved by a new "
            "pre-registration on #128, never by a re-run.**", "",
            "## A caveat on Tarteel's absolute rates (the owner's listening, #160)", "",
            "The owner listened to 100 Tarteel v1 look-half words on 2026-10-21. Of the 80 "
            "excess flags (`h448` flags, the teacher does not), 77 were heard as correctly "
            "recited, 0 as mistakes and 3 as unclear; all 20 controls were correct. So `h448`'s "
            "excess flags on app-user audio are model errors. **The Tarteel clips also contain "
            "real reciter mistakes that no model highlighted**, so Tarteel's absolute rates on "
            "presumed-correct recitation include some real mistakes. The paired "
            "excess-over-teacher comparisons are unaffected.", ""]

    out += ["## 1. The look half: live or EMA", "",
            f"Tarteel v1's look half, {look['items']} clips, checks protocol. Rule: "
            f"{look['rule']}. **Chosen: {look['choice']}.** {look['control'].capitalize()}.",
            "",
            "| weights | graded words | flagged | false flags per 1,000 | word agreement "
            "with the teacher (points) |", "|---|---|---|---|---|"]
    for model, row in sorted(look["inputs"].items()):
        out.append(f"| {model} | {row['graded_words']:g} | {row['flagged_words']:g} | "
                   f"{_number(row['false_flags_per_1000'])} | "
                   f"{_number(row['word_agreement_points'])} |")
    out.append("")

    out += ["## 2. Confirm: the main gate, mechanism and consistency", ""]
    out += table("Tarteel v1 confirm half, false flags per 1,000 graded words (shipped default)",
                 report["main"]["tarteel_v1_confirm"])
    out += table("Tarteel v1 confirm half, word agreement with the teacher",
                 report["mechanism"]["tarteel_v1_confirm"])
    out += table("qdat, word agreement with the teacher", report["mechanism"]["qdat"])
    out += table("qdat, false flags per 1,000 graded words", report["consistency"]["qdat"])

    if coverage is not None:
        out += coverage_section(coverage, look, report)

    out += ["## 4. Guards", ""]
    guards = report["guards"]
    out += table("QUD, salted half of segments, false flags per 1,000 (evaluated share; §3)",
                 guards["qud_half"])
    out += table("Tadabur dev (no copy in `h448`'s training), teacher agreement at stream_b0",
                 guards["tadabur_dev_unseen"])
    for kind, view in guards["qdat_bench"].items():
        out += table(f"qdat_bench stream arm, {kind} agreement with the experts", view)

    out += ["## 5. Catch side (descriptive)", ""]
    catch = report["catch"]
    out += table("RetaSy `in_correct`, flagged words per 1,000 graded (teacher, `h448`, "
                 "candidate and control side by side)",
                 catch["retasy_in_correct"]["words"])
    out += table("RetaSy `in_correct`, clips with a flagged word", catch["retasy_in_correct"]["clips"])
    out += table("qdat_bench, flags on the words where the experts heard a departure",
                 catch["qdat_bench_departures"])

    out += ["## 6. Leniency (descriptive)", ""]
    leniency = report["leniency"]
    out += table("Tadabur dev, blank share of the argmax frames", leniency["blank_ratio"])
    same = leniency["baselines_redecoded_identical"]
    out += [f"The baselines decoded again for the blank share reproduce #159's committed "
            f"`stream_b0` decode on {same.get('h448', 0)} (`h448`) and {same.get('base', 0)} "
            f"(base) of {same['clips']} clips.", ""]
    for name, view in leniency["abstentions"].items():
        out += table(f"{name}, consonant carriers the decode commits nothing at", view)
    for name, view in leniency["ungraded"].items():
        out += table(f"{name}, words the engine leaves ungraded, per 1,000", view)

    audit = report["audit"]
    out += ["## Identity and audit", "",
            f"- `choice.json`: SHA-256 `{audit['choice_sha256']}`, commit "
            f"`{audit['choice_commit']}`.",
            *(f"- `{m}`: `{identity}`" for m, identity in sorted(audit["identities"].items())),
            *(f"- `{m}` (committed decodes): `{identity}`"
              for m, identity in sorted(report["baselines"].items())),
            f"- Muraja harness: `{audit['harness']}`",
            f"- Input files read, each with its SHA-256 in `report.json`: {len(audit['inputs'])}.",
            ""]
    return "\n".join(out)


# --- the consonant round's document -------------------------------------------------------------


def _render_consonant(look: Mapping, report: Mapping, coverage: Mapping | None,
                      spec: RoundSpec) -> str:
    """#174's readout in #95's layout: the verdict table, every reported-only table, the G6
    catch arm and the frozen populations' coverage."""
    verdict = report["verdict"]
    out = [
        f"# {spec.title}",
        "",
        "Generated by `python -m training.aug_readout doc --spec consonant1` from "
        "`tools/tadabur/consonant_readout/`'s `look.json` and `report.json`. The bars are #174's "
        "pre-registration, frozen 2026-10-22 before any training of the round and read exactly "
        "as written. Candidate: C95 continued toward the base teacher; baseline: C95. Nothing "
        "here trains, sets a threshold or touches the sealed panel.",
        "",
        f"## Verdict: {VERDICT_TEXT[verdict['verdict']]}",
        "",
        "| condition | bar (#174) | reading | verdict |",
        "|---|---|---|---|",
    ]
    for condition in verdict["conditions"]:
        out.append(f"| {condition['name']} | {condition['bar']} | "
                   f"{_condition_reading(condition)} | "
                   f"{VERDICT_TEXT[condition['verdict']]} |")
    out += ["", "**A \"cannot certify\" is resolved by a new dated pre-registration, never by "
                "a re-run.**", ""]

    out += ["## 1. The look half: live or EMA", "",
            f"Tarteel v1's look half, {look['items']} clips. Rule: {look['rule']}. "
            f"**Chosen: {look['choice']}.**", "",
            "| weights | graded words | flagged | false flags per 1,000 | word agreement "
            "with the teacher (points) |", "|---|---|---|---|---|"]
    for model, row in sorted(look["inputs"].items()):
        out.append(f"| {model} | {row['graded_words']:g} | {row['flagged_words']:g} | "
                   f"{_number(row['false_flags_per_1000'])} | "
                   f"{_number(row['word_agreement_points'])} |")
    out.append("")

    main = report["main"]
    out += ["## 2. Confirm: bars 1–5", ""]
    out += table("Tarteel v1 confirm, all-cause false flags per 1,000 graded words (bar 1; bar 2 "
                 "reads the point reduction)", main["all_cause"])
    out += table("Tarteel v1 confirm, consonant-caused false flags per 1,000 (bar 3)",
                 main["consonant_cause"])
    out += table("Tarteel v1 confirm, tashkeel-caused false flags per 1,000 (G4)",
                 main["tashkeel_cause"])
    out += table("QUD other half, tashkeel-caused false flags per 1,000 (G4's companion, "
                 "reported)", main["qud_other_tashkeel_cause"])
    out += table("Tarteel v1 confirm, word agreement with the teacher (bar 4a)",
                 report["mechanism"]["tarteel_v1_confirm"])
    out += table("qdat, word agreement with the teacher (bar 4a)", report["mechanism"]["qdat"])
    out += table("Tarteel v1 confirm, agreement on the teacher-departure words (bar 4b)",
                 report["mechanism"]["tarteel_v1_confirm_departs"])
    out += table("qdat, agreement on the teacher-departure words (bar 4b)",
                 report["mechanism"]["qdat_departs"])
    out += table("Tarteel v1 confirm, agreement where the teacher equals the reference (bar 4b's "
                 "companion, reported)", report["mechanism"]["tarteel_v1_confirm_equals"])
    out += table("qdat, agreement where the teacher equals the reference (bar 4b's companion, "
                 "reported)", report["mechanism"]["qdat_equals"])
    out += table("qdat, all-cause false flags per 1,000 (bar 5)", report["consistency"]["qdat"])

    out += ["## 3. Substitutions (G5 vetoes the frozen list only)", "",
            "Every substitution a model carried on the confirm half, flagged words per 1,000 "
            "graded words. G5 reads only the frozen list (marked ‡); the rest are reported, "
            "never vetoed. G5 trips only on a large regression; a pass does not mean no "
            "consonant got worse.", ""]
    frozen = set(report.get("g5_frozen", ()))
    for substitution, view in report["substitutions"].items():
        mark = " ‡ (frozen, G5)" if substitution in frozen else ""
        out += table(f"`{substitution}`{mark}", view)
    if not report["substitutions"]:
        out += ["No substitution was observed on the confirm half: G5 cannot certify.", ""]
    elif not frozen:
        out += ["No frozen substitution list: G5 cannot certify.", ""]

    out += ["## 4. Guards", ""]
    guards = report["guards"]
    out += table("QUD other half, all-cause false flags per 1,000 (G1)", guards["qud_other"])
    out += table("Tadabur dev, teacher agreement at stream_b0 (G2)",
                 guards["tadabur_dev_unseen"])
    for kind, view in guards["qdat_bench"].items():
        out += table(f"qdat_bench stream arm, {kind} agreement with the experts (G3)", view)

    out += ["## 5. Catch side: G6", ""]
    catch = report["catch"]["ship_a_verified"]
    if catch is None:
        out += ["No catch input (C95's and the candidate's caches): G6 cannot certify.", ""]
    else:
        out += [f"Ship A's held-out verified real mistakes at ط↔ق and س↔ش: {len(catch['sites'])} "
                f"sites, words the app flags (checks, shipped default).", ""]
        if "scope" in catch:
            scope = catch["scope"]
            out += [f"Of the {scope['candidates']} held-out mistakes at the two pairs, the "
                    f"shared exclusions remove {len(scope['excluded'])}; G6 reads "
                    f"{len(catch['sites'])}.", ""]
            if scope["excluded"]:
                out += ["Excluded: " + ", ".join(f"`{key}`" for key in scope["excluded"]), ""]
        out += ["| model | graded sites | flagged |", "|---|---|---|"]
        out += [f"| {model} | {row['graded']} | {row['flagged']} |"
                for model, row in catch["models"].items()]
        difference = catch.get("difference")
        out += ["", (f"candidate − {spec.baseline}: {difference:+d} sites; the bar is ≥ −2."
                     if difference is not None else
                     f"candidate − {spec.baseline}: not read (one side's caches are missing); "
                     "G6 cannot certify."), ""]
    report_only = report["catch"].get("ship_a_verified_report_only")
    if report_only:
        out += ["The real-mistake marks #174 reports only (noisier verdicts): each mark's sites "
                "after the shared exclusions, how many both models' caches hold, and the flags "
                "there. They never enter G6's denominator.", "",
                "| mark | sites after exclusions | read | model | graded | flagged |",
                "|---|---|---|---|---|---|"]
        for mark, row in report_only.items():
            if not row["models"]:
                out += [f"| {mark} | {row['sites']} | {row['read']} | – | – | – |"]
            for name, model_row in row["models"].items():
                out += [f"| {mark} | {row['sites']} | {row['read']} | {name} | "
                        f"{model_row['graded']} | {model_row['flagged']} |"]
        out += [""]
    out += table("qdat_bench, flags on the words where the experts heard a departure (the catch "
                 "side as #95 read it, reported)",
                 report["catch"]["qdat_bench_departures"])

    out += ["## 6. Reported, not gating", ""]
    out += table("QUD, the salted half #95 read, false flags per 1,000 (reported)",
                 report["guards"]["qud_half"])
    for name, view in report["leniency"]["abstentions"].items():
        out += table(f"{name}, consonant carriers the decode commits nothing at", view)
    for name, view in report["leniency"]["ungraded"].items():
        out += table(f"{name}, words the engine leaves ungraded, per 1,000", view)
    out += table("Tadabur dev, blank share of the argmax frames",
                 report["leniency"]["blank_ratio"])
    if coverage is not None:
        out += _round_coverage(coverage, look, report)
    if report.get("not_run"):
        out += ["**Not run:**", ""]
        out += [f"- {name}: {row['status']} — {row['reason']}."
                for name, row in report["not_run"].items()]
        out += [""]
    out += ["**Reported, not gating:** every substitution's change on the confirm half (§3), "
            "the false flags by cause (§2), `h448` and the teacher columns in every table, the "
            "catch side (§5), abstentions and coverage (§6–7), the #95 QUD half (§6), the "
            "teacher-equals subset beside bar 4b (§2), and G4's QUD-other-half companion (§2). "
            "RetaSy is #171's, absent here.", ""]

    if report.get("absent"):
        out += ["**Absent inputs:** the decodes of " + ", ".join(
            f"`{name}`" for name in report["absent"]) + " are not committed; every rule that "
            "reads them cannot certify.", ""]

    audit = report.get("audit")
    if audit:
        out += ["## Identity and audit", "",
                f"- `choice.json`: SHA-256 `{audit['choice_sha256']}`, commit "
                f"`{audit['choice_commit']}`.",
                f"- Muraja harness: `{audit['harness']}`",
                f"- Input files read, each with its SHA-256 in `report.json`: "
                f"{len(audit['inputs'])}.", ""]
    return "\n".join(out)


def _round_coverage(coverage: Mapping, look: Mapping, report: Mapping) -> list[str]:
    """The round's frozen populations: every set's decoded, evaluated and excluded counts, with
    the reasons, each checked against the size the readout actually read."""
    populations = dict(coverage["populations"])
    evaluated = {"tarteel_v1.look": look["items"]}
    for name in populations:
        evaluated.setdefault(name, report["sizes"].get(name.replace(".", "_")))
    for name, row in populations.items():
        if evaluated[name] is not None and row["evaluated"] != evaluated[name]:
            raise ValueError(f"coverage {name}: {row['evaluated']} evaluated, the readout read "
                             f"{evaluated[name]}")
    out = ["## 7. Coverage of the frozen populations", "", coverage["statement"], "",
           "| population | decoded | evaluated | excluded | excluded canonical word "
           "occurrences |", "|---|---|---|---|---|"]
    out += [f"| {name} | {row['decoded']} | {row['evaluated']} | {row['excluded']} | "
            f"{row['excluded_words']} |" for name, row in populations.items()]
    out += [""]
    for name, row in populations.items():
        out += [f"**{name}**, the {row['excluded']} excluded: {row['reason']}.", ""]
    return out
