"""The augmentation experiment's pre-registered bars and verdict (#95, bars from #128).

The owner set these bars on 2026-10-15, before any candidate result existed (#128, "Owner
decision, 2026-10-15: the bars for the augmentation experiment"). They are frozen here and read
by :func:`verdict` exactly as written; nothing in this repo may change or reinterpret them.

**Main gate (both parts must pass)**

1. *App outcome, Tarteel v1 confirm half*: the candidate's false flags per 1,000 graded words
   (shipped default) lower than ``h448``'s, the paired 95% upper bound **< 0**; and the point
   reduction **at least a quarter** of the ``h448`` − teacher gap on that half.
2. *Mechanism*: word-level agreement with the teacher's decode on Tarteel v1 (the confirm half)
   and on qdat must rise, each paired 95% lower bound **> 0**.

**Consistency**: on qdat, the candidate's false flags per 1,000 lower than ``h448``'s (point).

**Guards (all must hold)**: QUD false flags per 1,000, candidate − ``h448`` paired upper bound
**≤ +1.0**; Tadabur ``decode_evalset`` dev (the clips ``h448`` holds no copy of), decode agreement
with the teacher at ``stream_b0``, paired lower bound **≥ −1.0 pt**; qdat_bench, the stream arm's
consonant, haraka and shaddah agreement with the experts, each paired lower bound **≥ −2.0 pt**.

Every condition is ``pass``, ``fail`` or ``cannot_certify`` (no admissible bound, or an undefined
statistic), and the verdict folds them by §1's truth table
(:func:`training.acceptance_stats.aggregate`): any fail is a fail, else any cannot_certify is
"cannot certify", else pass. "Cannot certify" is never a pass.

**Edge rules** (#95, settled 2026-10-16 before any candidate decode): an ``h448`` − teacher gap on
the confirm half that is not positive makes the quarter-gap bar "cannot certify" (an additional
condition the bar presumes); qdat's consistency is required for a pass.

**Attribution** (#95): every result is also read as candidate − control. A candidate whose main
app-outcome bound beats ``h448`` but whose bound against the control does not is reported as a
**null for augmentation** (:func:`attribution`), with control − ``h448`` beside it as
description; this reading never changes the verdict.

A "cannot certify" is resolved by a new pre-registration on #128, never by a re-run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from training.acceptance_stats import CANNOT_CERTIFY, FAIL, NO_BOUND, PASS, aggregate
from training.aug_readout_metrics import CANDIDATE, CONTROL, H448, BASE, comparison_name
from training.aug_readout_spec import AUG95, RoundSpec

#: #128's bars, on the scale each table reports (false flags per 1,000; agreement in points).
BARS = {
    "main_upper_below": 0.0,
    "main_gap_share": 0.25,
    "mechanism_lower_above": 0.0,
    "qud_upper_at_most": 1.0,
    "tadabur_lower_at_least": -1.0,
    "bench_lower_at_least": -2.0,
}
#: The bars as #128 states them, for the report.
BAR_TEXT = {
    "main.app_outcome": "Tarteel v1 confirm half, false flags per 1,000 (shipped default): "
                        "candidate − h448 paired 95% upper bound < 0",
    "main.quarter_gap": "the point reduction at least a quarter of the h448 − teacher gap on "
                        "the confirm half",
    "mechanism.tarteel_v1": "word-level agreement with the teacher on Tarteel v1 (confirm half) "
                            "rises: paired 95% lower bound > 0",
    "mechanism.qdat": "word-level agreement with the teacher on qdat rises: paired 95% lower "
                      "bound > 0",
    "consistency.qdat": "qdat false flags per 1,000: candidate point estimate lower than h448's",
    "guard.qud": "QUD false flags per 1,000: candidate − h448 paired upper bound ≤ +1.0",
    "guard.tadabur": "Tadabur decode_evalset dev (no copy in h448's training), decode agreement "
                     "with the teacher at stream_b0: paired lower bound ≥ −1.0 pt",
    "guard.qdat_bench.consonant": "qdat_bench stream arm, consonant agreement with the experts: "
                                  "paired lower bound ≥ −2.0 pt",
    "guard.qdat_bench.haraka": "qdat_bench stream arm, haraka agreement with the experts: "
                               "paired lower bound ≥ −2.0 pt",
    "guard.qdat_bench.shaddah": "qdat_bench stream arm, shaddah agreement with the experts: "
                                "paired lower bound ≥ −2.0 pt",
}
#: The consonant round's bars, as #174 states them (#175).
CONSONANT_BAR_TEXT = {
    "bar1.all_cause": "Tarteel v1 confirm half, all-cause false flags per 1,000 (shipped "
                      "default): candidate − C95 paired 95% upper bound < 0",
    "bar2.reduction": "the same difference as a point reduction of at least 4.7 per 1,000",
    "bar3.consonant_cause": "consonant-caused false flags per 1,000: candidate − C95 paired "
                            "95% upper bound < 0",
    "bar4a.tarteel_v1_confirm": "word agreement with the teacher on Tarteel v1 confirm: "
                                "paired 95% lower bound > 0",
    "bar4a.qdat": "word agreement with the teacher on qdat: paired 95% lower bound > 0",
    "bar4b.tarteel_v1_confirm": "the same on the teacher-departure words of Tarteel v1 "
                                "confirm: paired 95% lower bound ≥ −0.5 pt",
    "bar4b.qdat": "the same on the teacher-departure words of qdat: paired 95% lower bound "
                  "≥ −0.5 pt",
    "bar5.qdat": "qdat all-cause false flags per 1,000: candidate point estimate < C95's",
    "g1.qud_other": "QUD other half, all-cause false flags per 1,000: candidate − C95 paired "
                    "95% upper bound ≤ +1.0",
    "g2.tadabur_dev_unseen": "Tadabur dev, decode agreement with the teacher at stream_b0: "
                             "paired 95% lower bound ≥ −1.0 pt",
    "g3.qdat_bench.consonant": "qdat_bench stream arm, consonant agreement: paired 95% lower "
                               "bound ≥ −2.0 pt",
    "g3.qdat_bench.haraka": "qdat_bench stream arm, haraka agreement: paired 95% lower "
                            "bound ≥ −2.0 pt",
    "g3.qdat_bench.shaddah": "qdat_bench stream arm, shaddah agreement: paired 95% lower "
                             "bound ≥ −2.0 pt",
    "g4.tashkeel_cause": "Tarteel v1 confirm, tashkeel-caused false flags per 1,000: "
                         "candidate − C95 paired 95% upper bound ≤ +1.0",
    "g5": "no substitution's paired 99.5% lower bound > 0",
    "g6.ship_a_verified": "Ship A held-out verified mistakes at ط↔ق and س↔ش: candidate flags "
                          "≥ C95 − 2 sites (paired count)",
}
CANDIDATE_VS_H448 = comparison_name(CANDIDATE, H448)
CANDIDATE_VS_CONTROL = comparison_name(CANDIDATE, CONTROL)
CONTROL_VS_H448 = comparison_name(CONTROL, H448)
NULL_FOR_AUGMENTATION = "null for augmentation"
BEATS_CONTROL = "beats the control"
NOT_APPLICABLE = "not applicable: the candidate does not beat h448"


def _finite(value) -> float | None:
    """A bound as the table holds it: a number, an infinite end, or none."""
    if value is None:
        return None
    if value in ("inf", "-inf"):
        return math.inf if value == "inf" else -math.inf
    return float(value)


def bound_condition(name: str, table: Mapping | None, comparison: str, end: str,
                    threshold: float, strict: bool, text: str | None = None) -> dict:
    """One bound of a table's paired difference against ``threshold``: ``end`` is ``lower``
    (it must be above, or at least, the threshold) or ``upper`` (below, or at most). A table
    that was not read (its input is missing) cannot certify."""
    if table is None:
        return {"name": name, "bar": text or BAR_TEXT[name], "verdict": CANNOT_CERTIFY,
                "reason": "no input: the set's decodes are not committed"}
    difference = table["comparisons"].get(comparison)
    if difference is None:
        return {"name": name, "bar": text or BAR_TEXT[name], "verdict": CANNOT_CERTIFY,
                "reason": f"no {comparison} comparison"}
    interval = difference["interval"]
    bound = _finite(interval[end])
    out = {"name": name, "bar": text or BAR_TEXT[name], "comparison": comparison,
           "point": difference["point"], "bound": interval[end], "end": end,
           "threshold": threshold, "strict": strict, "method": interval["method"]}
    if interval["method"] == NO_BOUND or bound is None:
        return {**out, "verdict": CANNOT_CERTIFY, "reason": interval["reason"] or "no bound"}
    if end == "lower":
        holds = bound > threshold if strict else bound >= threshold
    else:
        holds = bound < threshold if strict else bound <= threshold
    return {**out, "verdict": PASS if holds else FAIL}


def quarter_gap_condition(table: Mapping | None) -> dict:
    """The candidate's point reduction against ``h448`` at least :data:`BARS` ``main_gap_share``
    of the ``h448`` − teacher gap, all on the confirm half; an undefined rate, or a gap that is
    not positive (the bar presumes ``h448`` flags more than the teacher), cannot certify."""
    if table is None:
        return {"name": "main.quarter_gap", "bar": BAR_TEXT["main.quarter_gap"],
                "verdict": CANNOT_CERTIFY,
                "reason": "no input: the set's decodes are not committed"}
    models = table["models"]
    points = {m: models[m]["point"] if m in models else None for m in (CANDIDATE, H448, BASE)}
    out = {"name": "main.quarter_gap", "bar": BAR_TEXT["main.quarter_gap"], **{
        f"{m}_per_1000": p for m, p in points.items()}}
    if any(p is None for p in points.values()):
        return {**out, "verdict": CANNOT_CERTIFY, "reason": "an undefined rate"}
    gap = points[H448] - points[BASE]
    required = BARS["main_gap_share"] * gap
    reduction = points[H448] - points[CANDIDATE]
    out |= {"gap": gap, "required_reduction": required, "reduction": reduction}
    if gap <= 0:
        return {**out, "verdict": CANNOT_CERTIFY, "reason": "the h448 − teacher gap is not positive"}
    return {**out, "verdict": PASS if reduction >= required else FAIL}


def point_lower_condition(name: str, table: Mapping | None, text: str | None = None,
                          baseline: str = H448) -> dict:
    """The candidate's point estimate strictly lower than the baseline's."""
    if table is None:
        return {"name": name, "bar": text or BAR_TEXT[name], "verdict": CANNOT_CERTIFY,
                "reason": "no input: the set's decodes are not committed"}
    models = table["models"]
    a, b = (models.get(m, {}).get("point") for m in (CANDIDATE, baseline))
    out = {"name": name, "bar": text or BAR_TEXT[name], "candidate": a, baseline: b}
    if a is None or b is None:
        return {**out, "verdict": CANNOT_CERTIFY, "reason": "an undefined rate"}
    return {**out, "verdict": PASS if a < b else FAIL}


def conditions(report: Mapping) -> list[dict]:
    """Every pre-registered condition, from the confirm report's tables."""
    main = report["main"]["tarteel_v1_confirm"]
    mechanism = report["mechanism"]
    guards = report["guards"]
    return [
        bound_condition("main.app_outcome", main, CANDIDATE_VS_H448, "upper",
                        BARS["main_upper_below"], strict=True),
        quarter_gap_condition(main),
        bound_condition("mechanism.tarteel_v1", mechanism["tarteel_v1_confirm"],
                        CANDIDATE_VS_H448, "lower", BARS["mechanism_lower_above"], strict=True),
        bound_condition("mechanism.qdat", mechanism["qdat"], CANDIDATE_VS_H448, "lower",
                        BARS["mechanism_lower_above"], strict=True),
        point_lower_condition("consistency.qdat", report["consistency"]["qdat"]),
        bound_condition("guard.qud", guards["qud_half"], CANDIDATE_VS_H448, "upper",
                        BARS["qud_upper_at_most"], strict=False),
        bound_condition("guard.tadabur", guards["tadabur_dev_unseen"], CANDIDATE_VS_H448,
                        "lower", BARS["tadabur_lower_at_least"], strict=False),
        *(bound_condition(f"guard.qdat_bench.{kind}", guards["qdat_bench"][kind],
                          CANDIDATE_VS_H448, "lower", BARS["bench_lower_at_least"], strict=False)
          for kind in ("consonant", "haraka", "shaddah")),
    ]


def attribution(report: Mapping) -> dict:
    """The main app-outcome bound read against the control: when the candidate beats ``h448``
    (its upper bound below zero) but not the control, a null for augmentation."""
    main = report["main"]["tarteel_v1_confirm"]
    against_h448 = bound_condition("main.app_outcome", main, CANDIDATE_VS_H448, "upper",
                                   BARS["main_upper_below"], strict=True)
    against_control = bound_condition("main.app_outcome", main, CANDIDATE_VS_CONTROL, "upper",
                                      BARS["main_upper_below"], strict=True)
    if against_h448["verdict"] != PASS:
        reading = NOT_APPLICABLE
    else:
        reading = BEATS_CONTROL if against_control["verdict"] == PASS else NULL_FOR_AUGMENTATION
    control_vs_h448 = None if main is None else main["comparisons"].get(CONTROL_VS_H448)
    return {"reading": reading, "against_h448": against_h448["verdict"],
            "against_control": against_control["verdict"],
            "control_upper_bound": against_control.get("bound"),
            "control_point": against_control.get("point"),
            # Description only: how far the extra steps alone moved h448.
            "control_vs_h448": control_vs_h448}


def verdict(report: Mapping, spec: RoundSpec | None = None) -> dict:
    """The pre-registered verdict of ``spec``'s round: every condition, their fold, and (#95
    only) the attribution. Bars are read as the round's pre-registration states them."""
    specification = spec or AUG95
    if specification.name == "aug95":
        parts = conditions(report)
        return {"bars": dict(BARS), "conditions": parts,
                "verdict": aggregate([c["verdict"] for c in parts]),
                "attribution": attribution(report)}
    parts = consonant_conditions(report, specification)
    return {"bars": dict(specification.bars), "conditions": parts,
            "reported": consonant_reported(report, specification),
            "verdict": aggregate([c["verdict"] for c in parts])}


# --- the consonant round's bars (#174) ----------------------------------------------------------


def reduction_condition(table: Mapping | None, spec: RoundSpec) -> dict:
    """Bar 2: the point reduction against the round's baseline at least the pre-registered
    number of flags per 1,000 (4.7 for C95's published 47.41)."""
    if table is None:
        return {"name": "bar2.reduction", "bar": CONSONANT_BAR_TEXT["bar2.reduction"],
                "verdict": CANNOT_CERTIFY,
                "reason": "no input: the set's decodes are not committed"}
    models = table["models"]
    points = {m: models[m]["point"] if m in models else None
              for m in (CANDIDATE, spec.baseline)}
    out = {"name": "bar2.reduction", "bar": CONSONANT_BAR_TEXT["bar2.reduction"],
           "candidate_per_1000": points[CANDIDATE],
           "baseline_per_1000": points[spec.baseline]}
    if any(point is None for point in points.values()):
        return {**out, "verdict": CANNOT_CERTIFY, "reason": "an undefined rate"}
    # At the report's precision: 47.41 − 42.71 is 4.699999999999996 in binary floating point.
    reduction = round(points[spec.baseline] - points[CANDIDATE], 9)
    required = spec.bars["reduction_at_least"]
    return {**out, "reduction": reduction, "required": required,
            "verdict": PASS if reduction >= required else FAIL}


def veto_conditions(substitutions: Mapping, frozen: Sequence[str], spec: RoundSpec) -> list[dict]:
    """G5: every **frozen** substitution's paired 99.5 % lower bound at most 0. Every other
    observed substitution is reported descriptively and never vetoes (#174). No list, or a
    frozen substitution the confirm half did not observe, cannot certify."""
    if not frozen:
        return [{"name": "g5", "bar": CONSONANT_BAR_TEXT["g5"], "verdict": CANNOT_CERTIFY,
                 "reason": "no frozen substitution list"}]
    comparison = comparison_name(CANDIDATE, spec.baseline)
    out = []
    for substitution in sorted(frozen):
        item = {"name": f"g5.{substitution}", "bar": CONSONANT_BAR_TEXT["g5"],
                "substitution": substitution}
        view = substitutions.get(substitution)
        if view is None:
            out.append({**item, "verdict": CANNOT_CERTIFY,
                        "reason": "not observed on the confirm half"})
            continue
        row = view["comparisons"].get(comparison)
        veto = row.get("veto_interval") if row else None
        if veto is None:
            out.append({**item, "verdict": CANNOT_CERTIFY, "reason": "no 99.5% interval"})
            continue
        bound = _finite(veto["lower"])
        if veto["method"] == NO_BOUND or bound is None:
            out.append({**item, "verdict": CANNOT_CERTIFY,
                        "reason": veto["reason"] or "no bound"})
            continue
        out.append({**item, "bound": veto["lower"], "end": "lower", "method": veto["method"],
                    "verdict": PASS if bound <= 0 else FAIL})
    return out


def catch_condition(table: Mapping | None, spec: RoundSpec) -> dict:
    """G6: the candidate flags at least the baseline's count less two sites."""
    item = {"name": "g6.ship_a_verified", "bar": CONSONANT_BAR_TEXT["g6.ship_a_verified"]}
    difference = None if table is None else table.get("difference")
    if difference is None:
        return {**item, "verdict": CANNOT_CERTIFY, "reason": "no catch input"}
    threshold = spec.bars["catch_at_least"]
    return {**item, "difference": difference, "threshold": threshold,
            "verdict": PASS if difference >= threshold else FAIL}


def consonant_conditions(report: Mapping, spec: RoundSpec) -> list[dict]:
    """Bars 1–5 and guards G1–G6, exactly as #174 states them, in the pre-registration's order."""
    bars = spec.bars
    comparison = comparison_name(CANDIDATE, spec.baseline)
    main, mechanism, guards = report["main"], report["mechanism"], report["guards"]

    def bound(name: str, table: Mapping, end: str, threshold: float, strict: bool) -> dict:
        return bound_condition(name, table, comparison, end, threshold, strict,
                               text=CONSONANT_BAR_TEXT[name])

    return [
        bound("bar1.all_cause", main["all_cause"], "upper",
              bars["all_cause_upper_below"], True),
        reduction_condition(main["all_cause"], spec),
        bound("bar3.consonant_cause", main["consonant_cause"], "upper",
              bars["consonant_upper_below"], True),
        bound("bar4a.tarteel_v1_confirm", mechanism["tarteel_v1_confirm"], "lower",
              bars["agreement_lower_above"], True),
        bound("bar4a.qdat", mechanism["qdat"], "lower", bars["agreement_lower_above"], True),
        bound("bar4b.tarteel_v1_confirm", mechanism["tarteel_v1_confirm_departs"], "lower",
              bars["departs_lower_at_least"], False),
        bound("bar4b.qdat", mechanism["qdat_departs"], "lower",
              bars["departs_lower_at_least"], False),
        point_lower_condition("bar5.qdat", report["consistency"]["qdat"],
                              text=CONSONANT_BAR_TEXT["bar5.qdat"], baseline=spec.baseline),
        bound("g1.qud_other", guards["qud_other"], "upper", bars["qud_upper_at_most"], False),
        bound("g2.tadabur_dev_unseen", guards["tadabur_dev_unseen"], "lower",
              bars["tadabur_lower_at_least"], False),
        *(bound(f"g3.qdat_bench.{kind}", guards["qdat_bench"][kind], "lower",
                bars["bench_lower_at_least"], False)
          for kind in spec.g3_kinds),
        bound("g4.tashkeel_cause", main["tashkeel_cause"], "upper",
              bars["tashkeel_upper_at_most"], False),
        # The round's report always writes ``g5_frozen`` (empty before #177). A report that
        # omits it entirely is the pre-#175 shape: its substitutions are the list it gates.
        *veto_conditions(report["substitutions"],
                         report.get("g5_frozen", tuple(report["substitutions"])), spec),
        catch_condition(report["catch"]["ship_a_verified"], spec),
    ]


def consonant_reported(report: Mapping, spec: RoundSpec) -> list[dict]:
    """The readings computed beside the verdict and shown with it, never folded into it: round
    2's shaddah G3 line (the owner reports it rather than gating it), same table and bar. An
    absent reading is reported as cannot-certify, never as a pass."""
    comparison = comparison_name(CANDIDATE, spec.baseline)
    guards = report["guards"]
    return [bound_condition(f"g3.qdat_bench.{kind}", guards["qdat_bench"][kind], comparison,
                            "lower", spec.bars["bench_lower_at_least"], False,
                            text=CONSONANT_BAR_TEXT[f"g3.qdat_bench.{kind}"])
            for kind in spec.g3_reported_kinds]
