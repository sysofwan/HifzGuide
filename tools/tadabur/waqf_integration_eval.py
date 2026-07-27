"""P7.H conditional-reference integration eval — the product gate (#35, ADR-0004).

The event-level waqf metrics (#34, :mod:`tadabur.waqf_event_eval`) can look good while the
*product* goal still fails, so ADR-0004
(``docs/adr/0004-waqf-head-and-joint-whole-clip-fine-tune.md``) demands a dedicated end-to-end
gate: consume the **phoneme decode** and the **predicted waqf events** *together* — snap →
per-run **realized-reference selection** (waqf vs wasl form) → **`.strict`** scoring — on
adjudicated wasl/waqf cases that turn on a **final haraka (i'raab)** or a **cross-word idgham**,
and show it regains the wasl-sensitive discrimination that today's *ignore-end-word-tashkeel*
hack throws away. "This is what the sign-off (#10) must actually clear — not frame-F1, not
event-F1 alone" (ADR-0004).

**The mechanism.** A boundary the reciter **stops** at realizes its terminal word in the *pausal*
form (a tanwin becomes a madd, the final haraka drops, no cross-word idgham); a boundary they
**continue** through keeps the continuation form (the tanwin's noon/ghunna carries onto the next
word, the desinence is realized). The *same* phoneme realization is therefore **correct** after a
genuine stop but a **dropped-i'raab / missed-idgham error** mid-continuation. The waqf head is what
tells the two apart, so its prediction **selects which realized reference** the `.strict` gate
scores against. The frozen fixtures (:mod:`tadabur.waqf_integration_fixtures`) carry, per
adjudicated boundary realization, the two pre-resolved normalized references and the reciter's
decode; this harness is a pure function of the scorer over them.

**Four scenarios, one `.strict` gate.** Each case is scored under four reference-selection rules,
so the eval demonstrates *both* the win and the two failure directions ADR-0004 names:

* **conditional (oracle waqf head)** — reference = the realized form of the **true** boundary
  class. This is the product path with a perfect waqf head (the ceiling the trained head is judged
  against). When the trained head lands, its per-boundary prediction (from
  :mod:`tadabur.waqf_event_eval` / :mod:`tadabur.waqf_postprocess`) drops in here unchanged.
* **baseline (ignore-end-word-tashkeel)** — today's behaviour: score against the continuation
  reference but **forgive the terminal edge** (discount the local aligner's ``trailing_trim`` from
  the denominator), so a legitimate pause is never punished — and neither is a boundary error.
* **false-waqf** — a spurious stop: always select the **waqf** reference. ADR-0004's dangerous
  error — it **forgives** a dropped haraka / missed idgham exactly as the baseline does.
* **false-wasl** — a missed stop: always select the **wasl** reference under full strict. It
  **rejects** a legitimately-paused recitation.

**The gate (`passed`).** The conditional path *regains discrimination the baseline lacks without
regressing legitimate-pause acceptance*: (1) conditional scores every case correctly
(``accept`` for a correct recitation, ``reject`` for a dropped/interior error); (2) it **rejects
every** error case the **baseline forgives** (≥1 such case, so the regain is real); (3) it keeps
**accepting every legitimate pause** the baseline accepts. The two failure scenarios are surfaced
as evidence, not gated on. Nothing here is torch or model dependent; identical fixtures yield an
identical report.

Consumer contract (:mod:`tadabur.signoff_results`): the report carries top-level ``passed`` (bool)
and ``summary`` (str); the full detail is passed through.

Usage:
  python -m tadabur.waqf_integration_eval [--cases PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .scorer import MIN_QUERY_PHONEMES, STRICT, STRICT_SCORER
from .waqf_integration_fixtures import (
    ACCEPT,
    CORRECT,
    REJECT,
    WAQF,
    WASL,
    IntegrationCase,
    load_integration_cases,
)

# The four reference-selection scenarios the eval scores every case under (see the module docstring).
CONDITIONAL = "conditional"
BASELINE = "baseline"
FALSE_WAQF = "false_waqf"
FALSE_WASL = "false_wasl"
SCENARIOS: tuple[str, ...] = (CONDITIONAL, BASELINE, FALSE_WAQF, FALSE_WASL)

# Recorded verbatim on the report so a reader cannot mistake this `.strict` product gate for the
# frame-F1 or event-F1 checks (ADR-0004: "This is what the sign-off (#10) must actually clear").
PRODUCT_GATE_NOTE = (
    "The product gate is `.strict` conditional-reference scoring, not frame-F1 or event-F1. It "
    "consumes the phoneme decode and the predicted waqf events together (snap -> per-run realized-"
    "reference selection -> strict scoring) on adjudicated i'raab / cross-word-idgham boundaries, "
    "and asks whether selecting the realized (waqf vs wasl) reference regains the wasl-sensitive "
    "discrimination the ignore-end-word-tashkeel baseline throws away (ADR-0004). The conditional "
    "scenario uses an oracle waqf head (the ceiling); the trained head's per-boundary prediction "
    "from tadabur.waqf_event_eval / tadabur.waqf_postprocess drops into it unchanged."
)


def strict_accepts(decode: str, reference: str) -> bool:
    """Whether ``STRICT_SCORER`` accepts ``decode`` against a normalized ``reference``."""
    return STRICT_SCORER.gate(decode, reference).passed


def baseline_accepts(decode: str, reference: str) -> bool:
    """Today's ignore-end-word-tashkeel verdict: strict, but forgiving the terminal edge.

    Scores ``decode`` against the continuation (wasl) ``reference`` and discounts the terminal
    phonemes the local aligner trims (``trailing_trim``) from the match denominator, so the
    boundary word's pausal ending never counts. A legitimate pause is not punished — and a
    dropped-i'raab / missed-idgham error at that edge is equally forgiven, which is exactly the
    discrimination ADR-0004 is regaining. Interior (non-edge) errors are *not* forgiven: they lower
    the alignment score, which the trailing-trim discount cannot recover.

    ``decode`` is the raw model-decode phoneme string; ``STRICT_SCORER.gate`` normalizes it (the
    fixtures store the decode un-normalized, per the scorer's non-idempotent-normalization
    contract), so ``query_count`` is taken from that same normalized view.
    """
    gate = STRICT_SCORER.gate(decode, reference)
    query_count = sum(1 for ch in _normalized_query(decode) if ch != " ")
    if query_count < MIN_QUERY_PHONEMES:
        return False
    score = gate.match_ratio * query_count
    forgiven = max(query_count - gate.trailing_trim, 1)
    return score / forgiven >= STRICT.correct_threshold


def _normalized_query(decode: str) -> str:
    """The scorer's normalized view of a raw ``decode`` (matches what ``gate`` aligns)."""
    from .normalization import normalize_phonemes

    return normalize_phonemes(decode).normalized


def _reference_for(case: IntegrationCase, scenario: str) -> str:
    """The realized reference ``scenario`` selects for ``case`` (baseline is handled separately)."""
    if scenario == CONDITIONAL:
        return case.waqf_reference if case.true_class == WAQF else case.wasl_reference
    if scenario == FALSE_WAQF:
        return case.waqf_reference
    if scenario == FALSE_WASL:
        return case.wasl_reference
    raise ValueError(f"scenario {scenario!r} has no single selected reference")


def verdict(case: IntegrationCase, scenario: str) -> str:
    """The `.strict` verdict (``accept`` / ``reject``) ``scenario`` reaches on ``case``."""
    if scenario == BASELINE:
        accepted = baseline_accepts(case.decode, case.wasl_reference)
    else:
        accepted = strict_accepts(case.decode, _reference_for(case, scenario))
    return ACCEPT if accepted else REJECT


@dataclass(frozen=True)
class ScenarioResult:
    """One reference-selection scenario's outcome over all cases.

    ``correct`` is the number of cases whose `.strict` verdict matched ``expected_strict``.
    ``forgave_errors`` counts error cases (expected reject) it wrongly **accepted**;
    ``rejected_valid`` counts correct cases (expected accept) it wrongly **rejected** — the two
    failure directions ADR-0004 names (false-waqf forgives, false-wasl rejects).
    """

    scenario: str
    total: int
    correct: int
    forgave_errors: int
    rejected_valid: int

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None

    def to_json_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "forgave_errors": self.forgave_errors,
            "rejected_valid": self.rejected_valid,
        }


def score_scenario(cases: list[IntegrationCase], scenario: str) -> ScenarioResult:
    """Confuse ``scenario``'s `.strict` verdicts against the adjudicated expected verdicts."""
    correct = forgave = rejected_valid = 0
    for case in cases:
        got = verdict(case, scenario)
        if got == case.expected_strict:
            correct += 1
        elif case.expected_strict == REJECT and got == ACCEPT:
            forgave += 1
        elif case.expected_strict == ACCEPT and got == REJECT:
            rejected_valid += 1
    return ScenarioResult(scenario, len(cases), correct, forgave, rejected_valid)


@dataclass(frozen=True)
class CaseOutcome:
    """One case's per-scenario verdicts, for the report's per-case audit trail."""

    case_id: str
    phenomenon: str
    true_class: str
    recitation: str
    expected_strict: str
    verdicts: dict[str, str]

    def to_json_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "phenomenon": self.phenomenon,
            "true_class": self.true_class,
            "recitation": self.recitation,
            "expected_strict": self.expected_strict,
            "verdicts": self.verdicts,
        }


@dataclass(frozen=True)
class IntegrationReport:
    """The full product-gate report: the go/no-go plus every scenario and case."""

    passed: bool
    summary: str
    scenarios: dict[str, ScenarioResult]
    cases: list[CaseOutcome]
    regained_discrimination: int
    baseline_forgave: int

    def to_json_dict(self) -> dict:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "product_gate_note": PRODUCT_GATE_NOTE,
            "regained_discrimination": self.regained_discrimination,
            "baseline_forgave": self.baseline_forgave,
            "scenarios": {
                name: result.to_json_dict() for name, result in self.scenarios.items()
            },
            "cases": [case.to_json_dict() for case in self.cases],
        }


def _error_cases_regained(cases: list[IntegrationCase]) -> tuple[int, int]:
    """(# error cases the baseline forgives that conditional rejects, # baseline forgives).

    The regain is the discrimination ADR-0004 is after: an error case (expected reject) the
    end-word-forgiven baseline **accepts** but the conditional realized-reference path correctly
    **rejects**.
    """
    baseline_forgave = regained = 0
    for case in cases:
        if case.expected_strict != REJECT:
            continue
        if verdict(case, BASELINE) == ACCEPT:
            baseline_forgave += 1
            if verdict(case, CONDITIONAL) == REJECT:
                regained += 1
    return regained, baseline_forgave


def _legitimate_pause_preserved(cases: list[IntegrationCase]) -> bool:
    """Whether every legitimate pause the baseline accepts, the conditional path also accepts.

    The waqf head must regain discrimination *without* reintroducing the punished-pause the hack
    exists to avoid: for every genuine waqf/correct case the baseline accepts, conditional must too.
    """
    for case in cases:
        if not (case.true_class == WAQF and case.recitation == CORRECT):
            continue
        if verdict(case, BASELINE) == ACCEPT and verdict(case, CONDITIONAL) != ACCEPT:
            return False
    return True


def run_eval(cases: list[IntegrationCase]) -> IntegrationReport:
    """Score every scenario and decide the product gate over the adjudicated cases.

    ``passed`` iff the conditional realized-reference path (1) scores every case correctly,
    (2) rejects every error case the baseline forgives — with at least one such case, so the regain
    is demonstrated, not vacuous — and (3) preserves legitimate-pause acceptance.
    """
    if not cases:
        raise ValueError("no integration cases — run tadabur.waqf_integration_gen first")

    scenarios = {name: score_scenario(cases, name) for name in SCENARIOS}
    outcomes = [
        CaseOutcome(
            case_id=case.case_id,
            phenomenon=case.phenomenon,
            true_class=case.true_class,
            recitation=case.recitation,
            expected_strict=case.expected_strict,
            verdicts={name: verdict(case, name) for name in SCENARIOS},
        )
        for case in cases
    ]

    regained, baseline_forgave = _error_cases_regained(cases)
    conditional = scenarios[CONDITIONAL]
    passed = (
        conditional.correct == conditional.total
        and baseline_forgave > 0
        and regained == baseline_forgave
        and _legitimate_pause_preserved(cases)
    )

    baseline = scenarios[BASELINE]
    if passed:
        summary = (
            f"PASS: conditional-reference (.strict) scored {conditional.correct}/"
            f"{conditional.total} adjudicated i'raab/idgham cases correctly and regained "
            f"discrimination on all {regained} boundary errors the ignore-end-word-tashkeel "
            f"baseline forgave (baseline {baseline.correct}/{baseline.total}); false-waqf forgave "
            f"{scenarios[FALSE_WAQF].forgave_errors}, false-wasl rejected "
            f"{scenarios[FALSE_WASL].rejected_valid} legitimate pauses."
        )
    else:
        summary = (
            f"FAIL: conditional-reference (.strict) scored {conditional.correct}/"
            f"{conditional.total}; regained {regained} of {baseline_forgave} baseline-forgiven "
            f"boundary errors; legitimate-pause acceptance "
            f"{'preserved' if _legitimate_pause_preserved(cases) else 'regressed'}."
        )

    return IntegrationReport(
        passed=passed,
        summary=summary,
        scenarios=scenarios,
        cases=outcomes,
        regained_discrimination=regained,
        baseline_forgave=baseline_forgave,
    )


def _print_summary(report: IntegrationReport) -> None:
    print(report.summary)
    for name in SCENARIOS:
        result = report.scenarios[name]
        print(
            f"  {name:12} accuracy {result.accuracy}  "
            f"forgave_errors {result.forgave_errors}  rejected_valid {result.rejected_valid}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cases", type=Path, default=None, help="Frozen integration cases (JSONL)."
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Report output path (JSON)."
    )
    args = parser.parse_args()

    from .waqf_integration_fixtures import WAQF_INTEGRATION_PATH

    cases_path = args.cases or WAQF_INTEGRATION_PATH
    cases = load_integration_cases(cases_path)
    if not cases:
        raise SystemExit(
            f"No integration cases at {cases_path} — run tadabur.waqf_integration_gen first."
        )

    report = run_eval(cases)
    out = args.out or (WAQF_INTEGRATION_PATH.parent / "waqf_integration_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report.to_json_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    _print_summary(report)
    print(f"Wrote report to {out}")


if __name__ == "__main__":
    main()
