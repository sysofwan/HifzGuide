"""Round 2's pre-claim gates, sequential claims and pooled verdict (#230).

Round 2's pre-registration (``docs/consonant-round2-preregistration.md``) splits its readout in
two reads: the non-tlog guards and the owner's catch-loss audit before any tranche is claimed
(:func:`run_preclaim`), then the pooled ``confirm.1``+``confirm.2`` read
(:func:`run_verdict`). The module holds both halves in the pre-registration's order:

* :data:`PRECLAIM_GATES` — the audit (:func:`audit_condition`), read from
  ``catch_loss/result.json`` exactly as :func:`tadabur.catch_loss_audit.score` wrote it, then
  G3's consonant and haraka readings, G6, bars 4a.qdat, 4b.qdat and 5, G1 and G2. A gate whose
  table was not read cannot certify; G3's shaddah part is reported only (owner, #173).
  :func:`run_preclaim` writes ``preclaim.json`` and, on anything but an overall pass, the
  round's final verdict and doc: no claim is then possible.
* :func:`claim` opens ``confirm.1`` and then ``confirm.2`` through
  :mod:`tadabur.tlog_tranches`, only when ``preclaim.json`` is committed in HEAD with an
  overall pass; every other role is refused for this round.
* :data:`POOLED_BARS` — bars 1, 2 (absolute 4.7), 3, 4a.tlog, 4b.tlog, G4 and G5 on the pooled
  gating grades — and :func:`final_verdict`, the decision rule over both reads.
  :func:`run_verdict` reads the two claimable tranches once, pools them before resampling
  (:func:`training.tlog_eval.confirm_views`: the cluster is the surah), and writes
  ``report.json`` and ``verdict.json``; :func:`render` writes the readout doc.

Every verdict carries :data:`STATEMENT`: the tranches are a clip-level split, so the same users
may sit on both sides.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from tadabur.tlog_tranches import TRANCHES_DIR, claim_tranche
from training import aug_readout as ar
from training.acceptance_stats import PASS, aggregate
from training.aug_readout_gate import (
    Layout,
    committed_choice,
    is_committed,
    sha256_file,
)
from training.aug_readout_metrics import (
    BASE,
    CANDIDATE,
    CONSONANT,
    CONTROL,
    DEPARTS,
    PER_1000,
    POINTS,
    TASHKEEL_CAUSE,
    bench_terms,
    compare,
    comparison_name,
    false_flag_terms,
    observed_substitutions,
    substitution_terms,
    substitution_views,
    teacher_agreement_terms,
    word_agreement_terms,
)
from training.aug_readout_spec import C95, RoundSpec
from training.aug_readout_verdict import (
    CONSONANT_BAR_TEXT,
    bound_condition,
    catch_condition,
    consonant_reported,
    point_lower_condition,
    reduction_condition,
    veto_conditions,
)
from training.consonant_readout import frozen_substitutions
from training.muraja_policy import SHIPPED_DEFAULT
from training.tashkeel_eval import write_text_atomically
from training.tlog_eval import (
    CONFIRM_TRANCHES,
    ID_FLOOR_OK,
    POOLED_CONFIRM,
    confirm_views,
)

#: What every verdict and doc states of the split (the pre-registration's own sentence).
STATEMENT = "clip-level split, same users possible on both sides"
#: The pre-claim gates, in the pre-registration's order: the audit first, G3's shaddah aside.
PRECLAIM_GATES = (
    "catch_loss", "g3.qdat_bench.consonant", "g3.qdat_bench.haraka", "g6.ship_a_verified",
    "bar4a.qdat", "bar4b.qdat", "bar5.qdat", "g1.qud_other", "g2.tadabur_dev_unseen")
#: The pooled confirm gates, in the pre-registration's order; G5 adds one entry per frozen
#: substitution (:func:`pooled_conditions`).
POOLED_BARS = ("bar1.all_cause", "bar2.reduction", "bar3.consonant_cause",
               "bar4a.tlog_confirm", "bar4b.tlog_confirm", "g4.tashkeel_cause")
#: The roles the pooled read grades, in report order.
ROLES = (CANDIDATE, CONTROL, C95, BASE)
#: The control comparison's caveat: one training run per arm, so it never gates.
CONTROL_CAVEAT = ("n = 1 per arm: one training run each, so candidate − control is "
                  "descriptive, not a test")
#: The reported strata's coverage: P3 decodes gating rows only (the pre-registration's budget).
NOT_DECODED = {
    "status": "not decoded",
    "reason": "P3 decodes the gating rows only; the reported strata and the span-trimmed "
              "sensitivity are recorded here, never decoded, so nothing estimates them",
}
AUDIT_BAR = ("the catch-loss audit: at most 20% of the decisive removed verdicts are real "
             "mistakes, and at least 80 decisive removed verdicts are heard")
TLOG_BAR_TEXT = {
    **CONSONANT_BAR_TEXT,
    "bar4a.tlog_confirm": "word agreement with the teacher on the pooled tlog confirm "
                          "tranches: paired 95% lower bound > 0",
    "bar4b.tlog_confirm": "the same on the teacher-departure words of the pooled tlog confirm "
                          "tranches: paired 95% lower bound ≥ −0.5 pt",
}


# --- the pre-claim gates ------------------------------------------------------------------------


def audit_condition(result: Mapping) -> dict:
    """The ``catch_loss`` gate: the scorer's own verdict, read, never re-derived."""
    return {"name": "catch_loss", "bar": AUDIT_BAR, "verdict": result["verdict"],
            "removed": result["removed"], "added": result.get("added"),
            "bars": result.get("bars")}


def preclaim_conditions(report: Mapping, spec: RoundSpec) -> list[dict]:
    """The eight non-audit pre-claim gates, in :data:`PRECLAIM_GATES` order.

    Each reads the frozen number in ``spec.bars``; a table that was not read, or an interval
    with no bound, cannot certify. G3's shaddah part is reported by :func:`consonant_reported`,
    never folded in here.
    """
    bars = spec.bars
    comparison = comparison_name(CANDIDATE, spec.baseline)
    guards = report["guards"]
    mechanism = report["mechanism"]

    def bound(name: str, table: Mapping | None, end: str, threshold: float,
              strict: bool) -> dict:
        return bound_condition(name, table, comparison, end, threshold, strict,
                               text=TLOG_BAR_TEXT[name])

    return [
        *(bound(f"g3.qdat_bench.{kind}", guards["qdat_bench"][kind], "lower",
                bars["bench_lower_at_least"], False)
          for kind in ("consonant", "haraka")),
        catch_condition(report["catch"]["ship_a_verified"], spec),
        bound("bar4a.qdat", mechanism["qdat"], "lower", bars["agreement_lower_above"], True),
        bound("bar4b.qdat", mechanism["qdat_departs"], "lower",
              bars["departs_lower_at_least"], False),
        point_lower_condition("bar5.qdat", report["consistency"]["qdat"],
                              text=TLOG_BAR_TEXT["bar5.qdat"], baseline=spec.baseline),
        bound("g1.qud_other", guards["qud_other"], "upper", bars["qud_upper_at_most"], False),
        bound("g2.tadabur_dev_unseen", guards["tadabur_dev_unseen"], "lower",
              bars["tadabur_lower_at_least"], False),
    ]


def _bound_set(data, roles: Mapping[str, str]):
    """A set's decode lines and items, keyed by role rather than by decode name."""
    decodes = {roles[name]: lines for name, lines in data.decodes.items() if name in roles}
    return ar.SetData(data.items, decodes)


def _role_grades(sources, data, roles: Mapping[str, str]) -> ar.SetData:
    """One set's decodes bound to roles and graded under the shipped default."""
    bound = _bound_set(data, roles)
    return ar.SetData(bound.items, bound.decodes,
                      sources.grade(bound.items, bound.decodes, tuple(roles.values())))


def preclaim_report(sources, spec: RoundSpec, choice: Mapping, workers: int = 1) -> dict:
    """The non-tlog sets read once for the chosen candidate and control, as round 1's report.

    Reads qdat (bars 4a, 4b, 5 and G3's bench kinds), the QUD other half (G1), Tadabur dev (G2)
    and G6's caches, plus the Tarteel v1 confirm half only for the substitution tally G5 needs
    later. ``control`` holds the control's columns beside the candidate's. A set whose decodes
    are not committed is ``None`` and cannot certify.
    """
    from training.consonant_readout import catch_arm, g6_scope

    weights = choice["weights"]
    chosen = ar.choice_models(spec, weights)
    students = tuple(chosen.values())
    models = (*students, spec.baseline, BASE)
    roles = {**{chosen[CANDIDATE]: CANDIDATE, chosen[CONTROL]: CONTROL},
             spec.baseline: spec.baseline, BASE: BASE}
    present = tuple(roles.values())
    absent = set(ar.absent_sets(sources, "confirm", weights, spec))
    shipped = SHIPPED_DEFAULT.name
    comparisons = spec.comparisons

    def flags(data, cause: str | None = None, against=None) -> dict | None:
        if data is None:
            return None
        return compare(false_flag_terms(data.grades, present, shipped, cause=cause), PER_1000,
                       comparisons if against is None else against)

    def agreement(data, subset=None, against=None) -> dict | None:
        if data is None:
            return None
        return compare(word_agreement_terms(
            data.items, ar.checks_text(data, present), (CANDIDATE, spec.baseline),
            subset=subset, workers=workers), POINTS,
            comparisons if against is None else against)

    qdat = None if ar.QDAT_SET in absent else _role_grades(sources, sources.qdat(models), roles)
    qud_other = (None if ar.QUD_OTHER_SET in absent
                 else _role_grades(sources, sources.qud_other(models), roles))
    tadabur = None
    if ar.TADABUR_SET not in absent:
        raw = sources.tadabur(models)
        tadabur = ar.TadaburData(
            raw.clips,
            {roles[name]: decodes for name, decodes in raw.decodes.items() if name in roles},
            {roles[name]: blanks for name, blanks in raw.blanks.items() if name in roles},
            raw.redecoded_identical)
    bench = [] if qdat is None else ar.bench_items(qdat, sources.bench())
    bench_views = {} if qdat is None else {
        kind: compare(terms, POINTS, comparisons) for kind, terms in bench_terms(
            bench, ar.stream_text(ar.SetData(bench, qdat.decodes), present),
            sources.bench(), present).items()}

    identity = choice.get("identities", {})
    identities = ({chosen[CANDIDATE]: identity[chosen[CANDIDATE]]}
                  if chosen[CANDIDATE] in identity else {})
    scope = g6_scope()
    catch = catch_arm(scope["sites"], (chosen[CANDIDATE], spec.baseline), spec=spec,
                      identities=identities, scope=scope)

    all_cause_qdat = flags(qdat)
    qud_table = flags(qud_other)
    tadabur_table = (None if tadabur is None else compare(
        teacher_agreement_terms(tadabur.clips, tadabur.decodes, present), POINTS, comparisons))
    report = {
        "main": {"all_cause": all_cause_qdat},
        "mechanism": {"qdat": agreement(qdat),
                      "qdat_departs": agreement(qdat, DEPARTS)},
        "consistency": {"qdat": all_cause_qdat},
        "guards": {"qud_other": flags(qud_other),
                   "tadabur_dev_unseen": tadabur_table,
                   "qdat_bench": bench_views or {kind: None
                                                 for kind in ("consonant", "haraka",
                                                              "shaddah")}},
        "catch": {"ship_a_verified": catch},
        "sizes": {"qdat": None if qdat is None else len(qdat.items),
                  "qud_other": None if qud_other is None else len(qud_other.items),
                  "tadabur_dev_unseen": (None if tadabur is None else len(tadabur.clips))},
        "control": {
            "qdat": {"all_cause_per_1000": _control_point(all_cause_qdat)},
            "qud_other": {"all_cause_per_1000": _control_point(qud_table)},
            "tadabur_dev_unseen": {"teacher_agreement_points":
                                   _control_point(tadabur_table)},
            "qdat_bench": {kind: _control_point(view)
                           for kind, view in (bench_views or {}).items()},
        },
    }
    return report


def _control_point(table: Mapping | None):
    """The control's point estimate in one table, or ``None`` when the table was not read."""
    return None if table is None else table["models"][CONTROL]["point"]


def _catch_loss_result(layout: Layout) -> Path:
    from tadabur.catch_loss_audit import RESULT

    return layout.root / "catch_loss" / RESULT


def require_passing_preclaim(layout: Layout) -> dict:
    """``preclaim.json`` committed in HEAD, unedited, with an overall pass."""
    path = layout.preclaim
    if not is_committed(path):
        raise SystemExit(f"{path.name} is not committed: the pre-claim phase has not run")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("verdict") != PASS:
        raise SystemExit(f"{path.name} is {record.get('verdict')!r}, not a pass: no claim is "
                         "possible and no confirm tranche is read")
    return record


def run_preclaim(sources, spec: RoundSpec, layout: Layout | None = None,
                 workers: int = 1) -> dict:
    """Read the audit and the non-tlog guards once, write ``preclaim.json``, and end the round
    on anything but a pass. Refused without the committed choice and the committed audit
    result, before anything is read, and once its own record exists."""
    layout = layout or Layout(spec.root)
    choice = committed_choice(layout, spec=spec)
    result_path = _catch_loss_result(layout)
    if not is_committed(result_path):
        raise SystemExit(f"{result_path.name} is not committed: the catch-loss audit has not "
                         "run, or its result changed since its commit")
    record = layout.preclaim
    if record.exists():
        raise SystemExit(f"{record.name} exists: the pre-claim phase reads once")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    report = preclaim_report(sources, spec, choice.choice, workers)
    conditions = [audit_condition(result), *preclaim_conditions(report, spec)]
    preclaim = {
        "issue": spec.issue,
        "phase": "preclaim",
        "verdict": aggregate([condition["verdict"] for condition in conditions]),
        "gates": {condition["name"]: condition for condition in conditions},
        "reported": {condition["name"]: condition
                     for condition in consonant_reported(report, spec)},
        "control": report.get("control", {}),
        "audit": result,
        "bars": dict(spec.bars),
        "choice_commit": choice.commit,
        "choice_sha256": choice.sha256,
        "harness": sources.harness_identity(),
        "inputs": _digests(sources, spec, choice.choice),
    }
    write_text_atomically(record, ar.canonical(preclaim))
    if preclaim["verdict"] != PASS:
        final = {"issue": spec.issue, "phase": "preclaim", "verdict": preclaim["verdict"],
                 "statement": STATEMENT, "gates": preclaim["gates"],
                 "control": preclaim["control"]}
        write_text_atomically(layout.verdict, ar.canonical(final))
        write_text_atomically(spec.doc_path, render(preclaim, None))
    return preclaim


def _digests(sources, spec: RoundSpec, choice: Mapping) -> dict:
    from training.aug_readout_gate import digests

    weights = choice["weights"]
    return digests(path for path in sources.input_files("confirm", weights) if path.exists())


# --- the sequential claims ----------------------------------------------------------------------


def _head_commit(path: Path) -> str:
    """The commit that last touched ``path`` in HEAD."""
    found = subprocess.run(
        ["git", "-C", str(path.parent), "log", "-1", "--format=%H", "--", path.name],
        capture_output=True, text=True, check=True).stdout.strip()
    if not found:
        raise SystemExit(f"{path.name} is in no commit: a claim names the commit it read")
    return found


def claim(spec: RoundSpec, role: str, *, tranches_dir: Path = TRANCHES_DIR, date: str,
          layout: Layout | None = None) -> Path:
    """Open one confirm tranche for this round; the operator commits the returned file.

    Only ``confirm.1`` and ``confirm.2`` are claimable here — confirm.3 and confirm.4 are
    refused under any input — and only when ``preclaim.json`` is committed in HEAD with an
    overall pass. ``commit`` is the commit that holds that preclaim record; ``confirm.2`` also
    needs ``confirm.1``'s committed claim (:func:`tadabur.tlog_tranches.require_claim_order`).
    """
    if role not in CONFIRM_TRANCHES:
        raise SystemExit(f"{role!r} cannot be claimed for {spec.name}: the round reads "
                         f"{list(CONFIRM_TRANCHES)}")
    layout = layout or Layout(spec.root)
    require_passing_preclaim(layout)
    return claim_tranche(role, issue=spec.issue, commit=_head_commit(layout.preclaim),
                         date=date, tranches_dir=tranches_dir)


# --- the pooled verdict -------------------------------------------------------------------------


def pooled_conditions(report: Mapping, spec: RoundSpec) -> list[dict]:
    """Bars 1, 2, 3, 4a.tlog, 4b.tlog and G4 from the pooled tables, then G5 per frozen
    substitution. Each reads the frozen number in ``spec.bars``; an unread table, an unbounded
    interval or an unobserved frozen substitution cannot certify."""
    bars = spec.bars
    comparison = comparison_name(CANDIDATE, spec.baseline)
    main = report["main"]
    mechanism = report["mechanism"]

    def bound(name: str, table: Mapping | None, end: str, threshold: float,
              strict: bool) -> dict:
        return bound_condition(name, table, comparison, end, threshold, strict,
                               text=TLOG_BAR_TEXT[name])

    return [
        bound("bar1.all_cause", main["all_cause"], "upper",
              bars["all_cause_upper_below"], True),
        reduction_condition(main["all_cause"], spec),
        bound("bar3.consonant_cause", main["consonant_cause"], "upper",
              bars["consonant_upper_below"], True),
        bound("bar4a.tlog_confirm", mechanism["tlog_confirm"], "lower",
              bars["agreement_lower_above"], True),
        bound("bar4b.tlog_confirm", mechanism["tlog_confirm_departs"], "lower",
              bars["departs_lower_at_least"], False),
        bound("g4.tashkeel_cause", main["tashkeel_cause"], "upper",
              bars["tashkeel_upper_at_most"], False),
        *veto_conditions(report["substitutions"], report.get("g5_frozen", ()), spec),
    ]


def final_verdict(preclaim: Mapping, conditions: Sequence[Mapping]) -> str:
    """The decision rule: any fail fails, else any cannot-certify cannot certify, else pass."""
    return aggregate([preclaim["verdict"], *(c["verdict"] for c in conditions)])


def _tables(grades: Sequence, comparisons: Sequence[tuple[str, str]]) -> dict:
    """All-cause, consonant-caused and tashkeel-caused false flags per 1,000."""
    shipped = SHIPPED_DEFAULT.name
    return {
        "all_cause": compare(false_flag_terms(grades, ROLES, shipped), PER_1000, comparisons),
        "consonant_cause": compare(false_flag_terms(grades, ROLES, shipped, cause=CONSONANT),
                                   PER_1000, comparisons),
        "tashkeel_cause": compare(false_flag_terms(grades, ROLES, shipped,
                                                   cause=TASHKEEL_CAUSE),
                                  PER_1000, comparisons),
    }


def _bar2_companions(main: Mapping) -> dict:
    """The reported companions of bar 2: 10% of C95's rate, the share of the C95 − teacher gap
    the candidate closes, and how many words the candidate and the teacher both flag."""
    models = main["models"]
    rate = {role: models.get(role, {}).get("point") for role in (CANDIDATE, C95, BASE)}
    gap = None
    if all(value is not None for value in rate.values()) and rate[C95] != rate[BASE]:
        gap = (rate[C95] - rate[CANDIDATE]) / (rate[C95] - rate[BASE])
    return {"ten_percent_of_c95": None if rate[C95] is None else 0.1 * rate[C95],
            "gap_closed": gap}


def _teacher_shared_flags(grades: Sequence) -> int:
    """Words both the candidate and the teacher flag, in the pooled gating grades."""
    both: dict[tuple[str, int], set[str]] = {}
    for grade in grades:
        if grade.model in (CANDIDATE, BASE):
            both.setdefault((grade.item.key, grade.position), set()).add(grade.model)
    return sum(1 for models in both.values() if {CANDIDATE, BASE} <= models)


def run_verdict(sources, spec: RoundSpec, layout: Layout | None = None, workers: int = 1,
                tranches_dir: Path = TRANCHES_DIR) -> dict:
    """Read ``confirm.1`` and ``confirm.2`` once, pool them, and write the report and verdict.

    Refused before any tranche is read unless the committed pre-claim record is a pass and both
    claims are committed in HEAD, and once ``verdict.json`` exists (the sitting reads once).
    """
    from tadabur.tlog_tranches import require_claim

    layout = layout or Layout(spec.root)
    preclaim = require_passing_preclaim(layout)
    for role in CONFIRM_TRANCHES:
        require_claim(role, tranches_dir)
    if layout.verdict.exists():
        raise SystemExit(f"{layout.verdict.name} exists: the pooled confirm read happens once")
    committed = committed_choice(layout, spec=spec)
    weights = committed.choice["weights"]
    chosen = ar.choice_models(spec, weights)
    models = (*chosen.values(), spec.baseline, BASE)
    roles = {**{chosen[CANDIDATE]: CANDIDATE, chosen[CONTROL]: CONTROL},
             C95: C95, BASE: BASE}

    sitting = {"phase": "confirm", "preclaim_sha256": sha256_file(layout.preclaim),
               "choice_commit": committed.commit, "choice_sha256": committed.sha256,
               "inputs": _digests(sources, spec, committed.choice),
               "harness": sources.harness_identity()}
    from training.aug_readout_gate import claim as hold

    with hold(layout.claim("confirm"), sitting):
        tranches = {}
        for role in CONFIRM_TRANCHES:
            data = sources.tlog(role, models)
            bound = _bound_set(data, roles)
            tranches[role] = ar.SetData(
                bound.items, bound.decodes,
                sources.grade(bound.items, bound.decodes, ROLES))
        pooled = confirm_views({role: data.items for role, data in tranches.items()})
        items = pooled[POOLED_CONFIRM]
        decodes = {role: {**tranches["confirm.1"].decodes[role],
                          **tranches["confirm.2"].decodes[role]} for role in ROLES}
        grades = sources.grade(items, decodes, ROLES)
        comparisons = spec.comparisons
        main = _tables(grades, comparisons)
        mechanism = {
            "tlog_confirm": compare(
                word_agreement_terms(items, ar.checks_text(ar.SetData(items, decodes), ROLES),
                                     (CANDIDATE, C95), workers=workers), POINTS, comparisons),
            "tlog_confirm_departs": compare(
                word_agreement_terms(items, ar.checks_text(ar.SetData(items, decodes), ROLES),
                                     (CANDIDATE, C95), subset=DEPARTS, workers=workers),
                POINTS, comparisons),
        }
        shipped = SHIPPED_DEFAULT.name
        frozen = frozen_substitutions(spec)
        substitutions = substitution_views(
            substitution_terms(grades, ROLES, shipped,
                               observed_substitutions(grades, ROLES, shipped)),
            PER_1000, comparisons)
        conditions = pooled_conditions(
            {"main": main, "mechanism": mechanism, "substitutions": substitutions,
             "g5_frozen": list(frozen)}, spec)
        floor_keys = {item.key for item in items
                      if (item.labels or {}).get("id_floor_ok") is True}
        floor_grades = [grade for grade in grades if grade.item.key in floor_keys]
        companions = _bar2_companions(main["all_cause"])
        companions["teacher_shared_flags"] = _teacher_shared_flags(grades)
        report = {
            "issue": spec.issue,
            "statement": STATEMENT,
            "main": main,
            "mechanism": mechanism,
            "substitutions": substitutions,
            "g5_frozen": list(frozen),
            "tranches": {role: _tables(data.grades, comparisons)
                         for role, data in tranches.items()},
            "bar2_companions": companions,
            "candidate_vs_control": {
                "table": compare(false_flag_terms(grades, ROLES, shipped), PER_1000,
                                 ((CANDIDATE, CONTROL),)),
                "caveat": CONTROL_CAVEAT},
            "strata": {ID_FLOOR_OK: _tables(floor_grades, comparisons)},
            "coverage": {"reported_strata": dict(NOT_DECODED)},
            "sizes": {"pooled": len(items),
                      **{role: len(data.items) for role, data in tranches.items()}},
            "verdict": {"bars": dict(spec.bars), "conditions": conditions,
                        "preclaim": preclaim["verdict"],
                        "verdict": final_verdict(preclaim, conditions)},
        }
        write_text_atomically(layout.report, ar.canonical(report))
        write_text_atomically(layout.verdict, ar.canonical(
            {"issue": spec.issue, "statement": STATEMENT,
             "preclaim": preclaim["verdict"], **report["verdict"]}))
        write_text_atomically(spec.doc_path, render(preclaim, report))
    return report


# --- the document -------------------------------------------------------------------------------


def _condition_lines(conditions: Sequence[Mapping]) -> list[str]:
    return [f"- {condition['name']}: {condition['verdict']} — "
            f"{condition.get('bar') or ''}" for condition in conditions]


def render(preclaim: Mapping, report: Mapping | None = None) -> str:
    """The round's readout doc as text: the verdict, every gate with its frozen bar, and the
    clip-level-split statement. ``report=None`` is a round the pre-claim ended."""
    lines = [f"# The tlog consonant round's readout (#{preclaim['issue']})", "",
             STATEMENT, "",
             f"Pre-claim verdict: {preclaim['verdict']}", ""]
    lines += ["## The pre-claim gates", ""]
    lines += _condition_lines(list(preclaim["gates"].values()))
    if preclaim.get("reported"):
        lines += ["", "Reported beside the gates, never folded into them:", ""]
        lines += _condition_lines(list(preclaim["reported"].values()))
    if preclaim.get("control"):
        lines += ["", "## The control's columns", "",
                  json.dumps(preclaim["control"], ensure_ascii=False, sort_keys=True)]
    if report is None:
        lines += ["", "No confirm tranche was claimed: confirm.1 and confirm.2 are not "
                      "decoded, and the round ends here."]
        return "\n".join(lines) + "\n"
    verdict = report["verdict"]
    lines += ["", "## The pooled confirm verdict", "",
              f"Verdict: {verdict['verdict']} (pre-claim {verdict['preclaim']})", "",
              "### Bars", ""]
    lines += _condition_lines(verdict["conditions"])
    lines += ["", "### Each tranche alone", ""]
    for role, table in report["tranches"].items():
        lines.append(f"- {role}: {table['all_cause']['observations']} observations, "
                     f"{table['all_cause']['clusters']} surah clusters")
    lines += ["", "### Reported companions", ""]
    for name, value in report["bar2_companions"].items():
        lines.append(f"- {name}: {value}")
    lines += ["", f"Candidate − control: {report['candidate_vs_control']['caveat']}",
              f"Gating ∧ {ID_FLOOR_OK}: "
              f"{report['strata'][ID_FLOOR_OK]['all_cause']['observations']} observations",
              f"Reported strata: {report['coverage']['reported_strata']['status']} — "
              f"{report['coverage']['reported_strata']['reason']}"]
    return "\n".join(lines) + "\n"
