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
* :data:`POOLED_BARS` — bars 1, 2 (absolute 4.7), 3, 4a.tlog, 4b.tlog and G4 on the pooled
  gating grades, with G5 appended per frozen substitution — and :func:`final_verdict`, the
  decision rule over both reads.
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

from tadabur.tarteel_split import CONFIRM
from tadabur.tlog_tranches import TRANCHES_DIR, claim_tranche
from training import aug_readout as ar
from training.acceptance_stats import CANNOT_CERTIFY, PASS, aggregate
from training.aug_readout_gate import (
    Layout,
    claim as hold_sitting,
    committed_choice,
    input_name,
    is_committed,
    needs,
    read_manifest,
    require_manifest,
    require_preregistration,
    require_preregistration_text,
    round_decodes_file,
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
    SHADDAH_CAUSE,
    TASHKEEL_CAUSE,
    abstention_terms,
    bench_terms,
    compare,
    comparison_name,
    false_flag_terms,
    observed_substitutions,
    shaddah_agreement_terms,
    substitution_terms,
    substitution_views,
    teacher_agreement_terms,
    word_agreement_terms,
)
from training.aug_readout_spec import TLOG_CONFIRM, RoundSpec, check_pins
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
#: The pooled confirm gates, in the pre-registration's order; :func:`pooled_conditions` appends
#: one ``g5.<substitution>`` per frozen substitution.
POOLED_BARS = ("bar1.all_cause", "bar2.reduction", "bar3.consonant_cause",
               "bar4a.tlog_confirm", "bar4b.tlog_confirm", "g4.tashkeel_cause")
#: The control comparison's caveat: one training run per arm, so it never gates.
CONTROL_CAVEAT = ("n = 1 per arm: one training run each, so candidate − control is "
                  "descriptive, not a test")
#: The reported strata's coverage: P3 decodes gating rows only (the pre-registration's budget).
#: The span-trimmed sensitivity is not in this record: it is read from the gating decodes.
NOT_DECODED = {
    "status": "not decoded",
    "reason": "P3 decodes the gating rows only: the reported:* strata are recorded here, "
              "never decoded, so nothing estimates them",
}
#: How the reported per-reciter points are read (one number per reciter defines no interval).
DOSE_CAVEAT = ("one point per reciter on Tadabur dev; a per-reciter cell is descriptive and "
               "defines no interval")
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
    return ar.SetData(data.items, decodes, excluded=data.excluded)


def _role_grades(sources, data, roles: Mapping[str, str]) -> ar.SetData:
    """One set's decodes bound to roles and graded under the shipped default."""
    bound = _bound_set(data, roles)
    return ar.SetData(bound.items, bound.decodes,
                      sources.grade(bound.items, bound.decodes, tuple(roles.values())))


def preclaim_report(sources, spec: RoundSpec, choice: Mapping, workers: int = 1) -> dict:
    """The non-tlog sets read once for the chosen candidate and control, as round 1's report.

    Reads qdat (bars 4a, 4b, 5 and G3's bench kinds), the QUD other half (G1), Tadabur dev (G2)
    and G6's caches. ``control`` holds the control's columns beside the candidate's. A set
    whose decodes are not committed is ``None`` and cannot certify.
    """
    from training.consonant_readout import catch_arm, g6_scope

    weights = choice["weights"]
    chosen = ar.choice_models(spec, weights)
    models = (*chosen.values(), spec.baseline, BASE)
    roles = {**{chosen[CANDIDATE]: CANDIDATE, chosen[CONTROL]: CONTROL},
             spec.baseline: spec.baseline, BASE: BASE}
    present = tuple(roles.values())
    absent = set(ar.absent_sets(sources, CONFIRM, weights, spec))
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
    transcripts = {} if qdat is None else sources.bench()
    bench = [] if qdat is None else ar.bench_items(qdat, transcripts)
    bench_views = {} if qdat is None else {
        kind: compare(terms, POINTS, comparisons) for kind, terms in bench_terms(
            bench, ar.stream_text(ar.SetData(bench, qdat.decodes), present), transcripts,
            present).items()}

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


def require_passing_preclaim(layout: Layout, spec: RoundSpec) -> dict:
    """``preclaim.json`` committed in HEAD, unedited, with an overall pass **under the round's
    pins**: the record's recorded harness and the committed pre-registration text must be the
    round's, so a pass written under another engine cannot open a tranche."""
    path = layout.preclaim
    if not is_committed(path):
        raise SystemExit(f"{path.name} is not committed: the pre-claim phase has not run")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("verdict") != PASS:
        raise SystemExit(f"{path.name} is {record.get('verdict')!r}, not a pass: no claim is "
                         "possible and no confirm tranche is read")
    prereg = require_preregistration_text(spec)
    check_pins(spec, record.get("harness") or {}, prereg)
    return record


def _freeze_path(spec: RoundSpec) -> Path:
    """The round's frozen substitution list: the committed ``freeze.json`` the G5 veto reads."""
    return Path(spec.root) / "freeze.json"


def require_frozen_list(spec: RoundSpec) -> tuple[Path, str]:
    """The round's freeze, refused unless it is committed and unchanged at HEAD, with its
    SHA-256: the G5 list is frozen, so a working-tree edit is never a new selection."""
    path = _freeze_path(spec)
    if not path.exists():
        raise SystemExit(f"{path.name} is absent: the round's frozen substitutions are not "
                         "recorded")
    if not is_committed(path):
        raise SystemExit(f"{path.name} is not committed unchanged: the frozen substitution "
                         "list is never edited")
    return path, sha256_file(path)


def _manifest_records(layout: Layout, spec: RoundSpec) -> dict:
    """The identities the sitting validates: the round's ``models.json`` plus the pooled confirm
    merge's own (``meta.confirm.json``), which the confirm decoder writes beside its decodes."""
    records = dict(read_manifest(layout))
    meta = round_decodes_file(spec, TLOG_CONFIRM, BASE).parent / "meta.confirm.json"
    if meta.exists():
        records.update(json.loads(meta.read_text(encoding="utf-8")).get("identities", {}))
    return records


def validate_inputs(sources, spec: RoundSpec, layout: Layout, choice: Mapping,
                    weights) -> tuple[dict, tuple[str, ...]]:
    """Refuse a read set whose producer identity is not the pinned, chosen model, before any
    grade is read: every present set's records are validated by the canonical
    :func:`training.aug_readout_gate.require_manifest`. Sets whose decodes are absent are
    returned as such — they cannot certify, and nothing reads them."""
    absent = ar.absent_sets(sources, CONFIRM, weights, spec)
    required = {name: models for name, models in needs(CONFIRM, weights, spec).items()
                if name not in absent}
    identities = require_manifest(_manifest_records(layout, spec), required, choice,
                                  spec=spec)
    return identities, absent


def run_preclaim(sources, spec: RoundSpec, layout: Layout | None = None,
                 workers: int = 1) -> dict:
    """Read the audit and the non-tlog guards once, write ``preclaim.json``, and end the round
    on anything but a pass. Refused without the committed choice, the committed audit result
    and the round's pinned harness and pre-registration, before anything is read, and once its
    own record exists."""
    layout = layout or Layout(spec.root)
    choice = committed_choice(layout, spec=spec)
    result_path = _catch_loss_result(layout)
    if not is_committed(result_path):
        raise SystemExit(f"{result_path.name} is not committed: the catch-loss audit has not "
                         "run, or its result changed since its commit")
    record = layout.preclaim
    if record.exists():
        raise SystemExit(f"{record.name} exists: the pre-claim phase reads once")
    prereg = require_preregistration(spec, sources.harness_identity())
    validate_inputs(sources, spec, layout, choice.choice, choice.choice["weights"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    report = preclaim_report(sources, spec, choice.choice, workers)
    conditions = [audit_condition(result), *preclaim_conditions(report, spec)]
    inputs, missing = _inputs(sources, spec, choice.choice)
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
        "prereg_sha256": prereg,
        "harness": sources.harness_identity(),
        "inputs": inputs,
        "missing_inputs": missing,
    }
    write_text_atomically(record, ar.canonical(preclaim))
    if preclaim["verdict"] != PASS:
        final = {"issue": spec.issue, "phase": "preclaim", "verdict": preclaim["verdict"],
                 "statement": STATEMENT, "gates": preclaim["gates"],
                 "control": preclaim["control"]}
        write_text_atomically(layout.verdict, ar.canonical(final))
        write_text_atomically(spec.doc_path, render(preclaim, None))
    return preclaim


def _inputs(sources, spec: RoundSpec, choice: Mapping) -> tuple[dict, list[str]]:
    """The confirm sitting's input digests and the declared inputs that are absent: a record
    smaller than its roster names what is missing instead of dropping it."""
    from training.aug_readout_gate import digests

    files = list(sources.input_files("confirm", choice["weights"]))
    missing = [path for path in files if not path.exists()]
    return (digests(path for path in files if path not in missing),
            sorted(input_name(path) for path in missing))


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
    require_preregistration_text(spec)
    require_passing_preclaim(layout, spec)
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


#: A cell with no graded word: reported, defined by nothing, and never a statistic.
EMPTY_CELL = {"status": "reported empty", "reason": "no graded word in this cell"}


def _tables(grades: Sequence, roles: Sequence[str],
            comparisons: Sequence[tuple[str, str]]) -> dict:
    """All-cause, consonant-caused, tashkeel-caused and shaddah-caused false flags per 1,000.
    A cell with no graded word is marked empty rather than raising: a reported-only subset can
    be empty without blocking the gating outcome."""
    if not grades:
        return dict(EMPTY_CELL)
    shipped = SHIPPED_DEFAULT.name
    return {
        "all_cause": compare(false_flag_terms(grades, roles, shipped), PER_1000, comparisons),
        "consonant_cause": compare(false_flag_terms(grades, roles, shipped, cause=CONSONANT),
                                   PER_1000, comparisons),
        "tashkeel_cause": compare(false_flag_terms(grades, roles, shipped,
                                                   cause=TASHKEEL_CAUSE),
                                  PER_1000, comparisons),
        "shaddah_cause": compare(false_flag_terms(grades, roles, shipped, cause=SHADDAH_CAUSE),
                                 PER_1000, comparisons),
    }


def _bar2_companions(main: Mapping, baseline: str) -> dict:
    """The reported companions of bar 2: 10% of the baseline's rate, the share of the
    baseline − teacher gap the candidate closes, and how many words both the candidate and the
    teacher flag."""
    models = main["models"]
    rate = {role: models.get(role, {}).get("point") for role in (CANDIDATE, baseline, BASE)}
    gap = None
    if all(value is not None for value in rate.values()) and rate[baseline] != rate[BASE]:
        gap = (rate[baseline] - rate[CANDIDATE]) / (rate[baseline] - rate[BASE])
    return {"ten_percent_of_c95": None if rate[baseline] is None else 0.1 * rate[baseline],
            "gap_closed": gap}


def _teacher_shared_flags(grades: Sequence, baseline: str) -> dict[str, int]:
    """Words each student and the teacher both flag, in the pooled gating grades: the
    candidate's and the baseline's, the pre-registration's two readings."""
    flagged: dict[tuple[str, int], set[str]] = {}
    for grade in grades:
        if grade.flagged:
            flagged.setdefault((grade.item.key, grade.position), set()).add(grade.model)
    return {student: sum(1 for models in flagged.values() if {student, BASE} <= models)
            for student in (CANDIDATE, baseline)}


def _partial_span(data) -> dict:
    """How many of a tranche's gating clips name a span shorter than their filename ayah (S6's
    ``word_from``/``word_to`` on the decode line; the readout grades the whole ayah)."""
    partial = 0
    for item in data.items:
        line = data.decodes[BASE][item.key]
        words = item.reference.reference.split(" ")
        span = (line.get("word_from") or 1, line.get("word_to") or len(words))
        partial += span != (1, len(words))
    return {"clips": len(data.items), "partial_span": partial,
            "partial_span_share": partial / len(data.items) if data.items else None}


def _tranche_coverage(data, present: Sequence[str], comparisons, workers: int) -> dict:
    """One tranche's declared and graded population, its partial-span share and the rate at
    which each model commits nothing on a consonant carrier (an empty slot)."""
    return {**_partial_span(data), "excluded": len(data.excluded),
            "empty_slots": compare(
                abstention_terms(data.items, ar.checks_text(data, present), present, workers),
                POINTS, comparisons)}


def dose_response(tadabur, present: Sequence[str], baseline: str) -> dict:
    """Per reciter on Tadabur dev: each model's teacher-agreement point and the candidate −
    baseline difference. Descriptive: one point per reciter defines no interval."""
    rows = []
    for reciter in sorted({clip.reciter_id for clip in tadabur.clips}):
        clips = [clip for clip in tadabur.clips if clip.reciter_id == reciter]
        terms = teacher_agreement_terms(clips, tadabur.decodes, present)
        points = {}
        for model in present:
            num, den = terms.by_model[model]
            total = float(den.sum())
            points[model] = None if not total else float(num.sum()) / total * POINTS
        rows.append({"reciter_id": reciter, "clips": len(clips), "points": points,
                     "candidate_vs_baseline": (None if points[CANDIDATE] is None
                                               or points[baseline] is None
                                               else points[CANDIDATE] - points[baseline])})
    return {"caveat": DOSE_CAVEAT, "reciters": rows}


def _pooled_report(sources, spec: RoundSpec, models, roles: Mapping[str, str],
                   present: Sequence[str], preclaim: Mapping, workers: int) -> dict | None:
    """The pooled confirm read: both tranches once, their span-trimmed views, the reported
    tables. ``None`` when the pooled gating set holds no item (nothing can certify)."""
    tranches, trimmed = {}, {}
    for role in CONFIRM_TRANCHES:
        for data, table in zip(sources.tlog_views(role, models), (tranches, trimmed)):
            bound = _bound_set(data, roles)
            table[role] = ar.SetData(
                bound.items, bound.decodes,
                sources.grade(bound.items, bound.decodes, present),
                excluded=bound.excluded)
    pooled = confirm_views({role: data.items for role, data in tranches.items()})
    items = pooled[POOLED_CONFIRM]
    if not items:
        return None
    decodes = {role: {**tranches["confirm.1"].decodes[role],
                      **tranches["confirm.2"].decodes[role]} for role in present}
    grades = sources.grade(items, decodes, present)
    comparisons = spec.comparisons
    texts = ar.checks_text(ar.SetData(items, decodes), present)
    main = _tables(grades, present, comparisons)
    mechanism = {
        "tlog_confirm": compare(
            word_agreement_terms(items, texts, (CANDIDATE, spec.baseline),
                                 workers=workers), POINTS, comparisons),
        "tlog_confirm_departs": compare(
            word_agreement_terms(items, texts, (CANDIDATE, spec.baseline),
                                 subset=DEPARTS, workers=workers), POINTS, comparisons),
        "tlog_confirm_shaddah": compare(
            shaddah_agreement_terms(items, texts, (CANDIDATE, spec.baseline),
                                    workers=workers), POINTS, comparisons),
    }
    shipped = SHIPPED_DEFAULT.name
    frozen = frozen_substitutions(spec)
    substitutions = substitution_views(
        substitution_terms(grades, present, shipped,
                           observed_substitutions(grades, present, shipped)),
        PER_1000, comparisons)
    conditions = pooled_conditions(
        {"main": main, "mechanism": mechanism, "substitutions": substitutions,
         "g5_frozen": list(frozen)}, spec)
    floor_keys = {item.key for item in items
                  if (item.labels or {}).get("id_floor_ok") is True}
    floor_grades = [grade for grade in grades if grade.item.key in floor_keys]
    companions = _bar2_companions(main["all_cause"], spec.baseline)
    companions["teacher_shared_flags"] = _teacher_shared_flags(grades, spec.baseline)
    # The span-trimmed sensitivity: the same gating decodes, each item cut to the span its
    # decode records. Reported only, never graded into gating.
    span_pooled = confirm_views({role: data.items for role, data in trimmed.items()})
    span_items = span_pooled[POOLED_CONFIRM]
    span_decodes = {role: {**trimmed["confirm.1"].decodes[role],
                           **trimmed["confirm.2"].decodes[role]} for role in present}
    tarteel = _bound_set(sources.tarteel(CONFIRM, models), roles)
    dev = sources.tadabur(models)
    tadabur = ar.TadaburData(
        dev.clips,
        {roles[name]: lines for name, lines in dev.decodes.items() if name in roles},
        {roles[name]: blanks for name, blanks in dev.blanks.items() if name in roles},
        dev.redecoded_identical)
    return {
        "issue": spec.issue,
        "statement": STATEMENT,
        "baseline": spec.baseline,
        "main": main,
        "mechanism": mechanism,
        "substitutions": substitutions,
        "g5_frozen": list(frozen),
        "tranches": {role: _tables(data.grades, present, comparisons)
                     for role, data in tranches.items()},
        "sensitivity": {"span_trimmed": _tables(
            sources.grade(span_items, span_decodes, present), present, comparisons)},
        "tarteel_confirm": _tables(
            sources.grade(tarteel.items, tarteel.decodes, present), present, comparisons),
        "dose_response": dose_response(tadabur, present, spec.baseline),
        "bar2_companions": companions,
        "candidate_vs_control": {
            "table": compare(false_flag_terms(grades, present, shipped), PER_1000,
                             ((CANDIDATE, CONTROL),)),
            "caveat": CONTROL_CAVEAT},
        "strata": {ID_FLOOR_OK: _tables(floor_grades, present, comparisons)},
        "coverage": {"tranches": {
            role: _tranche_coverage(data, present, comparisons, workers)
            for role, data in tranches.items()},
            "reported_strata": dict(NOT_DECODED)},
        "sizes": {"pooled": len(items),
                  **{role: len(data.items) for role, data in tranches.items()}},
        "verdict": {"bars": dict(spec.bars), "conditions": conditions,
                    "preclaim": preclaim["verdict"],
                    "verdict": final_verdict(preclaim, conditions)},
    }


def _unreadable_report(spec: RoundSpec, preclaim: Mapping, reason: str,
                       missing: Sequence[str]) -> dict:
    """The round's record when nothing was read: every pooled bar cannot certify, and the
    reason (the absent inputs) is named rather than raised."""
    tables = {name: None for name in ("all_cause", "consonant_cause", "tashkeel_cause",
                                      "shaddah_cause")}
    mechanism = {name: None for name in ("tlog_confirm", "tlog_confirm_departs",
                                         "tlog_confirm_shaddah")}
    conditions = pooled_conditions(
        {"main": tables, "mechanism": mechanism, "substitutions": {}, "g5_frozen": []}, spec)
    return {
        "issue": spec.issue,
        "statement": STATEMENT,
        "baseline": spec.baseline,
        "unreadable": {"reason": reason, "missing_inputs": list(missing)},
        "tranches": {},
        "strata": {},
        "coverage": {"reported_strata": dict(NOT_DECODED)},
        "verdict": {"bars": dict(spec.bars), "conditions": conditions,
                    "preclaim": preclaim["verdict"], "verdict": CANNOT_CERTIFY},
    }


def run_verdict(sources, spec: RoundSpec, layout: Layout | None = None, workers: int = 1,
                tranches_dir: Path = TRANCHES_DIR) -> dict:
    """Read ``confirm.1`` and ``confirm.2`` once, pool them, and write the report and verdict.

    Refused before any tranche is read unless the pre-registration is committed at its pinned
    hash, the freeze is committed unchanged, the producers' identities are the pinned, chosen
    models, the committed pre-claim record is a pass under those same pins, both claims are
    committed in HEAD, and ``verdict.json`` is absent (the sitting reads once). A declared input
    that is absent, or a pooled tranche that holds no gating item, writes a cannot-certify
    verdict instead of reading nothing's file or grading an empty population.
    """
    from tadabur.tlog_tranches import require_claim

    layout = layout or Layout(spec.root)
    preclaim = require_passing_preclaim(layout, spec)
    tranches_dir = Path(tranches_dir)
    sources.tranches_dir = tranches_dir
    for role in CONFIRM_TRANCHES:
        require_claim(role, tranches_dir)
    if layout.verdict.exists():
        raise SystemExit(f"{layout.verdict.name} exists: the pooled confirm read happens once")
    committed = committed_choice(layout, spec=spec)
    prereg = require_preregistration(spec, sources.harness_identity())
    _, freeze_sha = require_frozen_list(spec)
    weights = committed.choice["weights"]
    chosen = ar.choice_models(spec, weights)
    models = (*chosen.values(), spec.baseline, BASE)
    roles = {**{chosen[CANDIDATE]: CANDIDATE, chosen[CONTROL]: CONTROL},
             spec.baseline: spec.baseline, BASE: BASE}
    present = tuple(roles.values())
    identities, absent = validate_inputs(sources, spec, layout, committed.choice, weights)
    inputs, missing = _inputs(sources, spec, committed.choice)

    sitting = {"phase": "confirm", "preclaim_sha256": sha256_file(layout.preclaim),
               "choice_commit": committed.commit, "choice_sha256": committed.sha256,
               "prereg_sha256": prereg, "freeze_sha256": freeze_sha,
               "identities": identities, "inputs": inputs, "missing_inputs": missing,
               "harness": sources.harness_identity()}
    with hold_sitting(layout.claim("confirm"), sitting):
        if absent:
            report = _unreadable_report(
                spec, preclaim, f"declared input absent: {', '.join(absent)}", missing)
        else:
            report = _pooled_report(sources, spec, models, roles, present, preclaim, workers)
            if report is None:
                report = _unreadable_report(
                    spec, preclaim, "the pooled gating tranches hold no item", missing)
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
    """The round's readout doc as text: the verdict, every gate with its frozen bar, the
    reported tables with their estimates and intervals, and the clip-level-split statement.
    ``report=None`` is a round the pre-claim ended; a report with ``unreadable`` renders why."""
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
    if report.get("unreadable"):
        lines += ["", "Nothing was read: " + report["unreadable"]["reason"],
                  *[f"- missing input: {name}" for name in report["unreadable"]["missing_inputs"]]]
        return "\n".join(lines) + "\n"
    baseline = report["baseline"]
    lines += ["", "### The pooled gating tables", ""]
    lines += _tables_lines("pooled confirm.1 + confirm.2", report["main"], baseline)
    lines += ["", "### Each tranche alone", ""]
    for role, table in report["tranches"].items():
        lines += _tables_lines(role, table, baseline)
    lines += ["", f"### Gating ∧ {ID_FLOOR_OK}", ""]
    lines += _tables_lines(ID_FLOOR_OK, report["strata"][ID_FLOOR_OK], baseline)
    lines += ["", "### Coverage and empty slots, per tranche", ""]
    for role, coverage in report["coverage"]["tranches"].items():
        lines.append(f"- {role}: {coverage['clips']} clips, {coverage['partial_span']} "
                     f"partial spans, {coverage['excluded']} excluded")
        lines += _single_lines("empty slots per 100 consonant carriers",
                               coverage["empty_slots"], baseline)
    lines += ["", "### Tarteel v1 confirm (the full round-1 table)", ""]
    lines += _tables_lines("Tarteel v1 confirm", report["tarteel_confirm"], baseline)
    lines += ["", "### Span-trimmed sensitivity", ""]
    lines += _tables_lines("span-trimmed confirm.1 + confirm.2",
                           report["sensitivity"]["span_trimmed"], baseline)
    lines += ["", "### Shaddah, reported only", ""]
    lines += _single_lines("shaddah agreement with the teacher",
                           report["mechanism"]["tlog_confirm_shaddah"], baseline)
    lines += ["", "### Reported companions", ""]
    for name, value in report["bar2_companions"].items():
        lines.append(f"- {name}: {value}")
    lines += ["", "### Per-reciter dose-response on Tadabur dev", "",
              report["dose_response"]["caveat"], "",
              "| reciter | clips | candidate | c95 | candidate − c95 |",
              "|---|---|---|---|---|"]
    for row in report["dose_response"]["reciters"]:
        points = row["points"]
        lines.append(f"| {row['reciter_id']} | {row['clips']} | "
                     f"{_number(points.get(CANDIDATE))} | {_number(points.get(baseline))} | "
                     f"{_number(row['candidate_vs_baseline'])} |")
    lines += ["", "### Candidate − control", "",
              report["candidate_vs_control"]["caveat"], ""]
    lines += _single_lines("candidate − control", report["candidate_vs_control"]["table"],
                           baseline)
    lines += ["", f"Reported strata: {report['coverage']['reported_strata']['status']} — "
              f"{report['coverage']['reported_strata']['reason']}"]
    return "\n".join(lines) + "\n"


def _number(value) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def _interval(value: Mapping) -> str:
    if value.get("lower") is None and value.get("upper") is None:
        return "no interval"
    return f"[{_number(value.get('lower'))}, {_number(value.get('upper'))}]"


def _single_lines(title: str, table: Mapping, baseline: str) -> list[str]:
    """One :func:`training.aug_readout_metrics.compare` table: each model's point and each
    paired difference with its interval."""
    if "models" not in table:
        return [f"- {title}: {table.get('status')} — {table.get('reason')}"]
    out = [f"**{title}** (per 1,000 or points; n = {table['observations']})", "",
           "| model | point |", "|---|---|"]
    for name in (CANDIDATE, baseline, BASE, CONTROL):
        if name in table["models"]:
            out.append(f"| {name} | {_number(table['models'][name]['point'])} |")
    for name, versus in table["comparisons"].items():
        out.append(f"| {name} | {_number(versus['point'])} {_interval(versus['interval'])} |")
    return out


def _tables_lines(title: str, tables: Mapping, baseline: str) -> list[str]:
    """One cell's cause tables (all-cause, consonant, tashkeel, shaddah) with each model's
    point and the candidate − baseline difference with its interval."""
    if "all_cause" not in tables:
        return [f"- {title}: {tables.get('status')} — {tables.get('reason')}"]
    first = tables["all_cause"]
    rows = [name for name in (CANDIDATE, baseline, BASE, CONTROL) if name in first["models"]]
    out = [f"**{title}** (per 1,000 graded words; n = {first['observations']})", "",
           "| cause | " + " | ".join(rows) + " | candidate − baseline |",
           "|---" * (len(rows) + 2) + "|"]
    for cause, table in tables.items():
        points = " | ".join(_number(table["models"][name]["point"]) for name in rows)
        versus = ", ".join(f"{_number(value['point'])} {_interval(value['interval'])}"
                           for value in table["comparisons"].values())
        out.append(f"| {cause} | {points} | {versus} |")
    return out
