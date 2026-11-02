"""Round 3's freeze (#251): tlog look gating, the mechanical power rule, the known exclusions.

Round 3 (:mod:`training.aug_readout_spec.CONSONANT3`) reads the tlog tranches, so its first
grade of any tlog decode is this freeze (its pre-registration's data discipline 1). It reads
``tlog.look`` gating only — the confirm tranches stay sealed — and writes, beside round 2's
cause split (:func:`training.consonant_readout.freeze_tables`) and G5's list:

* **the look exclusions**: the rows the item builder left out, each of which must be a known
  exclusion (:func:`training.tlog_eval.require_known_exclusions`), and the set of them must be
  exactly round 2's twelve (#248);
* **the mechanical power rule**: one row per guard, with its set, margin, SE source, SE and
  ``P(pass | true 0)``, refusing the freeze when any effective row's probability is under
  ``power_at_least``. G4's SE is the look gating paired bootstrap SE of round 2's EMA against
  C95 for tashkeel false flags, scaled by ρ — round 2's realised pooled SE over its freeze
  stand-in — and is printed at both the registered +2.5 (rejected by amendment A1) and the
  effective +3.5. G6 is the exact sign test over the truth-baseline caches' discordant sites;
* **G5's per-substitution SE**, reported, never tested, flagging a frozen substitution no model
  flags.

Every number is a width read from this round's grades, round 2's committed report, its freeze
and the truth-baseline check caches. Nothing here sets a threshold or reads a confirm row.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from tadabur.tarteel_split import LOOK
from training import aug_readout_metrics as am
from training.aug_readout_gate import digests, require_preregistration_text
from training.aug_readout_metrics import BASE
from training.aug_readout_spec import (C95, CONSONANT2_ROOT, EMA, R2, R2_ARM_PREFIX,
                                       TLOG_LOOK, RoundSpec, check_pins)
from training.consonant2_readout import DAL_TA
from training.muraja_policy import SHIPPED_DEFAULT

#: Round 2's committed artifacts: its report and freeze give ρ, its freeze fixes the look
#: exclusions and G5's list this round must reproduce, its models give G6 the EMA's identity.
ROUND2_REPORT_PATH = CONSONANT2_ROOT / "report.json"
ROUND2_FREEZE_PATH = CONSONANT2_ROOT / "freeze.json"
ROUND2_MODELS_PATH = CONSONANT2_ROOT / "models.json"
#: Round 2's candidate at its averaged weights, by the name its decodes and caches carry.
ROUND2_EMA = f"{R2_ARM_PREFIX}candidate_{EMA}"
#: The models the freeze grades on the look set: C95, round 2's EMA and the teacher.
FREEZE_MODELS = (C95, R2, BASE)
FREEZE_LABELS = {C95: "C95", R2: "round 2 EMA", BASE: "teacher"}
#: The models every power-rule SE pairs: round 2's EMA against C95.
POWER_MODELS = (R2, C95)
#: The non-tlog sets G1, G2 and G3 read, from round 2's committed decodes.
GUARD_SETS = ("qdat", "qud_other", "tadabur_dev_unseen")
#: The command that writes the freeze; recorded in the artifact for the PR's command log.
FREEZE_COMMAND = "python -m training.consonant_readout freeze --spec consonant3 --half look"
RULE = ("C95's and round 2's EMA's tlog look gating false flags per 1,000 graded words by "
        "word_flags cause at the shipped default, each filename ayah's full text graded through "
        "the Muraja harness at the round's pin; G5 is round 1's rule over this population plus "
        "د→ت, and the freeze is written only if every guard's effective-margin pass "
        "probability at true zero is at least the round's power bar")
#: The width of a two-sided 95 % interval in standard errors: the pre-registration's divisor.
Z_95 = 3.92
#: The pre-registration's reported scaling: look gating words over both halves' words.
LOOK_WORDS, HALVES_WORDS = 2853, 5670
#: The pre-registration's reported lines. Neither feeds the rule: they assume the two halves'
#: divergence equals the look half's, which round 2's report already measured.
SCALED_LABEL = (f"SE_look × √({LOOK_WORDS}/{HALVES_WORDS}), the look gating share of both "
                "halves (pre-registration); assumes no further divergence from C95")
REALISED_LABEL = ("round 2's realised pooled tashkeel SE, (UB − LB)/3.92 of its report "
                  "(pre-registration); assumes no further divergence from C95")
#: Where each row's width comes from, printed so no reader takes a projection for a reading.
G1_SOURCE = ("the paired clustered bootstrap of round 2's EMA − C95, all-cause false flags "
             "per 1,000 on qud_other")
G2_SOURCE = ("the paired clustered bootstrap of round 2's EMA − C95, teacher agreement on "
             "tadabur_dev_unseen")
G3_SOURCE = ("the paired clustered bootstrap of round 2's EMA − C95, qdat_bench {kind} "
             "agreement")
G4_SOURCE = ("the paired clustered bootstrap of round 2's EMA − C95, tashkeel false flags per "
             "1,000 on tlog.look, unscaled, × ρ")
G6_SOURCE = "the truth-baseline check caches' discordant sites (exact binomial sign test)"

#: The bar names a registered margin may name, and the guard each prints as.
_REGISTERED_GUARDS = {"tashkeel_upper_at_most": "G4"}


# --- the pass probabilities ---------------------------------------------------------------------


def _phi(value: float) -> float:
    """The standard normal distribution function."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def pass_probability(margin: float, se: float) -> float:
    """``Φ(|margin| / se − 1.96)``: the chance a guard whose true change is zero passes its
    95 % bound. Symmetric in the margin's sign, so a lower bound's margin counts by its size;
    ``se = 0`` is a certain pass: with no width the bound is exact at zero change."""
    if se == 0:
        return 1.0
    return _phi(abs(margin) / se - 1.96)


def catch_pass_probability(discordant: int) -> float:
    """G6's exact sign test at true zero: ``P(Bin(d, ½) ≥ ⌈(d − 2)/2⌉)`` over ``d`` discordant
    sites. Not monotone in ``d``, so it is never replaced by a threshold on ``d``."""
    threshold = max(math.ceil((discordant - 2) / 2), 0)
    tail = sum(math.comb(discordant, k) for k in range(threshold, discordant + 1))
    return tail / 2 ** discordant


# --- the power table ----------------------------------------------------------------------------


def power_table(spec: RoundSpec, *, g4_se: float, guard_ses: Mapping[str, float],
                discordant: int) -> list[dict]:
    """The pre-registration's power-rule table, in its order: G1, G2, G3's consonant and haraka
    parts, each registered G4 margin (:attr:`RoundSpec.registered_margins`, not effective), the
    effective G4 row, and G6. A row is ``pass`` at or above the spec's ``power_at_least`` and
    ``under`` below it; only effective rows are ever judged by :func:`require_power_rule`."""
    bar = spec.bars["power_at_least"]

    def row(guard: str, set_name: str, margin: float, se: float | None, se_source: str,
            p_pass: float, *, effective: bool = True, note: str = "", **extra) -> dict:
        return {"guard": guard, "set": set_name, "margin": margin, "se_source": se_source,
                "se": se, "p_pass": p_pass,
                "effective": effective, "result": "pass" if p_pass >= bar else "under",
                "note": note, **extra}

    rows = [
        row("G1", "qud_other", spec.bars["qud_upper_at_most"], guard_ses["g1"], G1_SOURCE,
            pass_probability(spec.bars["qud_upper_at_most"], guard_ses["g1"])),
        row("G2", "tadabur_dev_unseen", spec.bars["tadabur_lower_at_least"],
            guard_ses["g2"], G2_SOURCE,
            pass_probability(spec.bars["tadabur_lower_at_least"], guard_ses["g2"])),
    ]
    for kind in spec.g3_kinds:
        margin = spec.bars["bench_lower_at_least"]
        se = guard_ses[f"g3_{kind}"]
        rows.append(row(f"G3.{kind}", "qdat_bench", margin, se, G3_SOURCE.format(kind=kind),
                        pass_probability(margin, se)))
    for name, (margin, amendment) in spec.registered_margins.items():
        rows.append(row(_REGISTERED_GUARDS[name], "tlog.look", margin, g4_se, G4_SOURCE,
                        pass_probability(margin, g4_se), effective=False,
                        note=f"registered, rejected by the rule ({amendment})"))
    effective = spec.bars["tashkeel_upper_at_most"]
    rows.append(row("G4", "tlog.look", effective, g4_se, G4_SOURCE,
                    pass_probability(effective, g4_se)))
    rows.append(row("G6", "ship_a_verified", spec.bars["catch_at_least"], None, G6_SOURCE,
                    catch_pass_probability(discordant), discordant=discordant))
    return rows


def require_power_rule(rows: Sequence[Mapping]) -> None:
    """Refuse the freeze when any **effective** row is under the bar, naming its guard. A
    registered row the amendment rejected never refuses anything, and there is no escape
    hatch: a further change is a new dated amendment needing the owner and a spec change."""
    under = [row for row in rows if row["effective"] and row["result"] != "pass"]
    if under:
        raise SystemExit(
            "the power rule refuses the freeze: " + "; ".join(
                f"{row['guard']} at margin {row['margin']:g} has P(pass | true 0) "
                f"{row['p_pass']:.3f}" for row in under))


# --- round 2's committed widths -----------------------------------------------------------------


def _round2_freeze() -> dict:
    return json.loads(ROUND2_FREEZE_PATH.read_text(encoding="utf-8"))


def _round2_report() -> dict:
    return json.loads(ROUND2_REPORT_PATH.read_text(encoding="utf-8"))


def round2_realised_se() -> float:
    """Round 2's realised pooled tashkeel SE: the half-width of its candidate − C95 interval
    over the pre-registration's 3.92."""
    interval = _round2_report()["main"]["tashkeel_cause"]["comparisons"][
        am.comparison_name(am.CANDIDATE, am.C95)]["interval"]
    return (interval["upper"] - interval["lower"]) / Z_95


def rho() -> float:
    """Round 2's realised pooled SE over its freeze stand-in SE: the scale from the look
    gating SE to the pooled SE. Exactly ``0.762 / 0.764`` on round 2's committed files."""
    return round2_realised_se() / _round2_freeze()["power_check"]["tashkeel"]["se"]


def look_se(grades: Sequence) -> float:
    """The surah-clustered paired bootstrap SE of round 2's EMA − C95 for tashkeel false flags
    per 1,000 on the look gating words: G4's unscaled width."""
    terms = am.false_flag_terms(grades, POWER_MODELS, SHIPPED_DEFAULT.name,
                                cause=am.TASHKEEL_CAUSE)
    return am.paired_se(terms, *POWER_MODELS, am.PER_1000)


# --- G1, G2 and G3's widths ---------------------------------------------------------------------


def guard_ses(sources, spec: RoundSpec) -> dict[str, float]:
    """G1, G2 and G3's paired clustered bootstrap SEs of round 2's EMA against C95 on the
    verdict's own sets, from round 2's committed non-tlog decodes: measured widths, no
    projection. G1 reads QUD's other half's all-cause false flags, G2 Tadabur dev's teacher
    agreement, and G3 qdat_bench's letter agreement per gated kind."""
    from training import aug_readout as ar

    models = POWER_MODELS
    shipped = SHIPPED_DEFAULT.name
    qdat = sources.qdat(models)
    qud = sources.qud_other(models)
    tadabur = sources.tadabur(models)
    transcripts = sources.bench()
    bench = ar.bench_items(qdat, transcripts)
    bench_kinds = am.bench_terms(bench, ar.stream_text(ar.SetData(bench, qdat.decodes),
                                                       models), transcripts, models)

    def se(terms, scale: float) -> float:
        return am.paired_se(terms, *models, scale)

    return {
        "g1": se(am.false_flag_terms(
            sources.grade(qud.items, qud.decodes, models), models, shipped), am.PER_1000),
        "g2": se(am.teacher_agreement_terms(
            tadabur.clips, {model: tadabur.decodes[model] for model in models}, models),
            am.POINTS),
        **{f"g3_{kind}": se(bench_kinds[kind], am.POINTS) for kind in spec.g3_kinds},
    }


# --- G6's discordant sites ----------------------------------------------------------------------


def _round2_identity() -> dict:
    """Round 2's look sitting recorded its EMA's identity in ``models.json``; G6 binds its
    decodes and check caches to it (its pin alone cannot stand in for the chosen weights)."""
    return json.loads(ROUND2_MODELS_PATH.read_text(encoding="utf-8"))[
        f"{TLOG_LOOK}/{ROUND2_EMA}"]["identity"]


def _discordant(table: Mapping, baseline: str) -> int:
    """The sites exactly one of the two models flags: the sign test's discordant count."""
    from training.muraja_policy import WORD_FLAGGED

    flagged = {model: {site for site, outcome in table["models"][model]["outcomes"].items()
                       if outcome in WORD_FLAGGED}
               for model in (ROUND2_EMA, baseline)}
    return len(flagged[ROUND2_EMA] ^ flagged[baseline])


def discordant_sites(spec: RoundSpec) -> int:
    """G6's ``d``: the sites exactly one of round 2's EMA and C95 flags, read from the
    truth-baseline check caches through :func:`training.consonant_readout.catch_arm`. A missing
    cache cannot be read as zero discordant sites."""
    from training import consonant_readout as cr

    scope = cr.g6_scope()
    table = cr.catch_arm(scope["sites"], (ROUND2_EMA, spec.baseline), spec=spec,
                         identities={ROUND2_EMA: _round2_identity()}, scope=scope)
    if table is None:
        raise SystemExit("G6 cannot read: a truth-baseline check cache is missing, so the "
                         "discordant count is unknown, never zero")
    return _discordant(table, spec.baseline)


# --- G5's reported widths -----------------------------------------------------------------------


def g5_se(grades: Sequence, rows: Sequence[Mapping]) -> list[dict]:
    """Each frozen substitution's paired clustered SE of round 2's EMA − C95 per 1,000, with
    each model's flagged-word count: reported, never tested. A substitution neither model
    flags has a zero width and is flagged, so a dead entry is seen rather than read as power."""
    names = tuple(row["substitution"] for row in rows)
    terms = am.substitution_terms(grades, POWER_MODELS, SHIPPED_DEFAULT.name, names)
    out = []
    for name in names:
        one = terms[name]
        events = {model: int(one.by_model[model][0].sum()) for model in POWER_MODELS}
        out.append({"substitution": name,
                    "se": am.paired_se(one, *POWER_MODELS, am.PER_1000),
                    "events": events,
                    "zero_events": not any(events.values())})
    return out


# --- the record ---------------------------------------------------------------------------------


def freeze_inputs(sources) -> list[Path]:
    """Every committed file the freeze's decision reads: the look set's three decodes and its
    tranche manifest; G1–G3's six readout decodes with every committed baseline, frozen QUD
    half and Tadabur grid/manifest their populations come from (the qdat_bench experts'
    transcripts are read from the pinned dataset revision, not a local file); ρ's two round-2
    artifacts; G6's identity record and both truth-baseline caches (:func:`catch_required`). A
    changed one of any of these changes the record's `inputs`."""
    from tadabur.exposure import EVALSET_MANIFEST_PATH
    from tadabur.nonpro_sources import QDAT
    from tadabur.qud_overlap import PILOT
    from training.aug_readout_gate import BASELINES
    from training.consonant_readout import QUD_SELECTION_PATH, catch_required
    from training.nonpro_eval import decodes_file as committed
    from training.nonpro_gap import DECODES_DIR as GAP_DECODES, GRID_PATH
    from training.qud_pilot import decodes_file as qud_committed

    return [
        *(sources.decode_path(TLOG_LOOK, model) for model in FREEZE_MODELS),
        sources.tranches_dir / f"{LOOK}.jsonl",
        *(sources.decode_path(name, model) for name in GUARD_SETS for model in POWER_MODELS),
        committed(QDAT, BASE),
        *(qud_committed(slug, model) for slug in PILOT for model in BASELINES),
        QUD_SELECTION_PATH,
        *(GAP_DECODES / f"clean.{model}.jsonl.gz" for model in BASELINES),
        GRID_PATH, EVALSET_MANIFEST_PATH,
        ROUND2_REPORT_PATH, ROUND2_FREEZE_PATH, ROUND2_MODELS_PATH,
        *catch_required((ROUND2_EMA, C95)),
    ]


def _require_round2_exclusions(excluded: Mapping[str, str | None]) -> None:
    """The look exclusions must be exactly round 2's twelve (#248): a different population is a
    different freeze, and silently dropping a gradeable row is what the rule forbids."""
    frozen = _round2_freeze()["excluded"]["keys"]
    if sorted(excluded) != frozen:
        raise SystemExit(
            f"the left-out rows differ from round 2's {len(frozen)}: "
            f"{sorted(set(excluded) - set(frozen))} are new and "
            f"{sorted(set(frozen) - set(excluded))} are gone")


def _require_round2_g5(rows: Sequence[Mapping]) -> None:
    """G5's list must be round 2's frozen list, order included: the veto is frozen before
    training, so a different derivation cannot silently replace it."""
    frozen = [row["substitution"] for row in _round2_freeze()["substitutions"]]
    listed = [row["substitution"] for row in rows]
    if listed != frozen:
        raise SystemExit(f"G5's list differs from round 2's frozen list: {listed} is not "
                         f"{frozen}")


def freeze_record(sources, spec: RoundSpec) -> dict:
    """Round 3's freeze: the look set's cause split beside round 2's EMA and the teacher, G5's
    list and per-substitution widths, the look exclusions, the power-rule table and the two
    reported lines. Refuses unless the pre-registration is committed and pinned (before reading
    any decode) and unless every effective power-rule row passes."""
    from training.consonant_readout import freeze_tables
    from training.tlog_eval import exclusion_reasons, excluded_record, require_known_exclusions

    prereg = require_preregistration_text(spec)
    harness = sources.harness_identity()
    check_pins(spec, harness, prereg)
    data = sources.tlog(LOOK, FREEZE_MODELS)
    grades = sources.grade(data.items, data.decodes, FREEZE_MODELS)
    require_known_exclusions(data.excluded)
    _require_round2_exclusions(data.excluded)
    tables = freeze_tables(grades, models=FREEZE_MODELS, labels=FREEZE_LABELS,
                           baseline=spec.baseline, required=(DAL_TA,))
    _require_round2_g5(tables["substitutions"])
    se_look = look_se(grades)
    rows = power_table(spec, g4_se=se_look * rho(), guard_ses=guard_ses(sources, spec),
                       discordant=discordant_sites(spec))
    require_power_rule(rows)
    return {
        "issue": spec.issue,
        "round": spec.name,
        "half": LOOK,
        "command": FREEZE_COMMAND,
        "rule": RULE,
        "pins": {"muraja_commit": spec.pins.muraja_commit,
                 "harness_sha256": spec.pins.harness_sha256,
                 "prereg_sha256": prereg},
        "harness": harness,
        "inputs": digests(freeze_inputs(sources)),
        "items": len(data.items),
        "excluded": {**excluded_record(data.excluded),
                     "by_reason": exclusion_reasons(data.excluded)},
        **tables,
        "power_table": rows,
        "power_reported": [
            {"label": SCALED_LABEL, "se": se_look * math.sqrt(LOOK_WORDS / HALVES_WORDS)},
            {"label": REALISED_LABEL, "se": round2_realised_se()}],
        "g5_se": g5_se(grades, tables["substitutions"]),
        "power_rule": {"satisfied": True,
                       "amendments": sorted({amendment for _, amendment
                                             in spec.registered_margins.values()})},
    }


def run_freeze(spec: RoundSpec, sources, model: str | None = None) -> dict:
    """The freeze command's dry run: the whole record, computed and printed, nothing written.
    It refuses without the committed, pinned pre-registration before grading anything. The
    round reads its own three models, so a named model is refused (round 2's dry run summarises
    one model; round 3's prints the power rule)."""
    if model is not None:
        raise SystemExit(f"{model}: round 3's freeze reads {', '.join(FREEZE_MODELS)}, not one "
                         "named model")
    return freeze_record(sources, spec)


# --- the doc ------------------------------------------------------------------------------------


def _table(rows: Sequence[Mapping]) -> list[str]:
    out = ["| guard | set | margin | SE | SE source | P(pass at true 0) | result | note |",
           "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        se = "—" if row["se"] is None else f"{row['se']:.3f}"
        out.append(f"| {row['guard']} | {row['set']} | {row['margin']:g} | {se} | "
                   f"{row['se_source']} | {row['p_pass']:.3f} | {row['result']} | "
                   f"{row['note']} |")
    return out


def render(record: Mapping) -> str:
    """The freeze's doc, rendered from the record alone: the power-rule table with both G4
    margins, the two reported lines that never feed the rule, G5's frozen list and widths, the
    look exclusions by reason, and the inputs and pins that produced them."""
    out = [
        "# Round 3's frozen look readings (#251)",
        "",
        f"Generated by `{record['command']}` and "
        "`python -m training.consonant_readout doc --spec consonant3` from "
        "`tools/tadabur/consonant3_readout/freeze.json`; do not edit by hand.",
        "",
        "**The tlog look gating set only.** No confirm row was read: this record holds the "
        "look gating rows' grades, and the confirm tranches stay sealed until the pre-claim "
        "gates pass. It is the first grade of any round-3 tlog decode.",
        "",
        f"{record['rule']}.",
        "",
        "## The power rule",
        "",
        "One row per guard. `P(pass | true 0)` is the chance a candidate with no true change "
        "passes the guard's 95 % bound; the freeze is written only if every **effective** row "
        "is at least the round's power bar. The +2.5 G4 row is the margin the round registered "
        "and amendment A1 rejected — it is printed, never judged.",
        "",
        *_table(record["power_table"]),
        "",
        "### Reported lines, outside the rule",
        "",
        "Each assumes no further divergence from C95 and reports only:",
        "",
        *[f"- {line['label']}: SE {line['se']:.3f}" for line in record["power_reported"]],
        "",
        "## G5's veto list and its widths",
        "",
        "C95's top ten consonant substitutions behind look-gating false-flagged words, the "
        "madd alif (`ا→∅`) aside, ties by substitution (Unicode order), **plus `د→ت`** — the "
        "owner kept it whatever its rank (2026-10-24). The list is round 2's, reproduced "
        "exactly; the widths are reported and tested nowhere. A substitution no model flags "
        "has a zero width and is flagged.",
        "",
        "| substitution | C95 words | round 2 EMA words | SE | flag |",
        "|---|---|---|---|---|",
        *[f"| {row['substitution']} | {row['events'][C95]} | {row['events'][R2]} | "
          f"{row['se']:.3f} | {'zero events' if row['zero_events'] else ''} |"
          for row in record["g5_se"]],
        "",
        "## The look exclusions (#248)",
        "",
        f"{record['items']} gating rows carry an item. **{record['excluded']['count']} more "
        "have none**: the row names no ayah, or the phonetizer cannot handle the ayah it names "
        f"({record['excluded']['by_reason']['unphonetizable']} unphonetizable, "
        f"{record['excluded']['by_reason']['names_no_ayah']} naming no ayah). They are exactly "
        "round 2's, and none is a gradeable ayah.",
        "",
        "| model | label | graded words | false flags |",
        "|---|---|---|---|",
        *[f"| {model} | {entry['label']} | "
          f"{entry['scorings']['shipped_default']['graded']} | "
          f"{entry['scorings']['shipped_default']['flagged']} |"
          for model, entry in record["models"].items()],
        "",
        "## The inputs, harness and pins",
        "",
        "| input | SHA-256 |",
        "|---|---|",
        *[f"| `{name}` | `{digest}` |" for name, digest in record["inputs"].items()],
        "",
        f"{record['items']} tlog look gating clips were graded through the Muraja harness at "
        f"commit `{record['pins']['muraja_commit']}` (binary SHA-256 "
        f"`{record['pins']['harness_sha256']}`), whose build is recorded in the JSON. The "
        f"pre-registration doc's SHA-256 is `{record['pins']['prereg_sha256']}`.",
        "",
    ]
    return "\n".join(out)
