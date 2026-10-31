"""Round 2's freeze (#227): C95 on the tlog look gating rows, before any tranche is claimed.

Round 2 (:mod:`training.aug_readout_spec.CONSONANT2`) reads the tlog tranches, so its first
grade of any tlog decode is this freeze (its pre-registration's data discipline 1). It reads
``tlog.look`` gating only — the confirm tranches are sealed — and writes, beside round 1's
cause split (:func:`training.consonant_readout.freeze_tables`):

* **G5's list**: C95's top ten consonant substitutions behind look-gating false-flagged words,
  the madd alif aside, ∪ {``د→ت``} (owner 2026-10-24);
* **the ``د→ت`` reading**: C95 against #178 on the look set — reported, it changes neither the
  list nor the bar (the likeliest non-tlog reason to fail, #173);
* **the one-tranche power check**: the paired clustered SE of #178 − C95 for all-cause and
  tashkeel false flags per 1,000, labelled "decides nothing" (the pre-registration's
  informative line);
* **graded-word coverage**: the measured replacement for the 0.98 graded-share projection.

Every number is read from the look gating grades; nothing here sets a threshold or reads a
confirm row.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from tadabur.tarteel_split import LOOK
from training import aug_readout_metrics as am
from training.aug_readout_gate import digests, require_preregistration
from training.aug_readout_metrics import BASE
from training.aug_readout_spec import C178, C95, RoundSpec
from training.consonant2_freeze_doc import render
from training.muraja_policy import SHIPPED_DEFAULT

#: The models the freeze grades on the look set: C95, #178 and the teacher (round 2's context
#: columns; ``h448`` is not decoded on tlog).
FREEZE_MODELS = (C95, C178, BASE)
FREEZE_LABELS = {C95: "C95", C178: "#178", BASE: "teacher"}
#: The substitution the owner kept in G5 whatever C95's look tally ranks (#173, 2026-10-24).
DAL_TA = "د→ت"
#: The models the ``د→ت`` reading compares.
DAL_TA_MODELS = (C95, C178)
#: The command that writes the freeze; recorded in the artifact for the PR's command log.
FREEZE_COMMAND = "python -m training.consonant_readout freeze --spec consonant2 --half look"
#: The pair the power check reports: the candidate against the round's baseline.
POWER_MODELS = (C178, C95)
#: The power line's label: a width, never a test (the pre-registration's "informative").
POWER_LABEL = ("the paired clustered SE of #178 − C95 per 1,000 graded words on one tranche; "
               "informative only: this line decides nothing")
RULE = ("C95's tlog look gating false flags per 1,000 graded words by word_flags cause at the "
        "shipped default, each filename ayah's full text graded through the Muraja harness at "
        "the round's pin; G5 is round 1's rule over this population plus د→ت, and the power "
        "line and the coverage are measured here, not projected")


def freeze_inputs(sources) -> list[Path]:
    """The committed decodes the freeze reads, the look set's C95, #178 and teacher decodes, and
    the tranche manifest that selects the graded rows (a manifest edit moves the population, so
    its digest belongs beside theirs)."""
    from tadabur.tlog_tranches import TRANCHES_DIR

    set_name = f"tlog.{LOOK}"
    return [*(sources.decode_path(set_name, model) for model in FREEZE_MODELS),
            TRANCHES_DIR / f"{LOOK}.jsonl"]


def dal_ta_reading(grades: Sequence) -> dict[str, dict]:
    """How many words each model flags with ``د→ت`` on the look set: reported at the freeze,
    changing neither G5's list nor its bar."""
    return {model: {"words": sum(1 for grade in grades
                                if grade.model == model
                                and grade.scoring == SHIPPED_DEFAULT.name and grade.flagged
                                and grade.reason is not None
                                and any(substitution == DAL_TA
                                        for _, substitution in grade.reason.consonants))}
            for model in DAL_TA_MODELS}


def power_check(grades: Sequence) -> dict:
    """The paired clustered SE of #178 − C95 for all-cause and tashkeel false flags per 1,000
    on one look tranche: a width, labelled so no reader takes it for a test."""
    check = {"label": POWER_LABEL}
    for name, cause in (("all_cause", None), ("tashkeel", am.TASHKEEL_CAUSE)):
        terms = am.false_flag_terms(grades, POWER_MODELS, SHIPPED_DEFAULT.name, cause=cause)
        check[name] = {"se": am.paired_se(terms, *POWER_MODELS, am.PER_1000)}
    return check


def coverage(grades: Sequence) -> dict[str, dict]:
    """Each graded model's words, graded words and graded share on the look set: the measured
    replacement for the pre-registration's projected graded share."""
    rows = {}
    for model in FREEZE_MODELS:
        mine = [grade for grade in grades
                if grade.model == model and grade.scoring == SHIPPED_DEFAULT.name]
        graded = sum(grade.graded for grade in mine)
        rows[model] = {"words": len(mine), "graded": graded,
                       "graded_share": graded / len(mine) if mine else 0.0}
    return rows


def freeze_record(sources, spec: RoundSpec) -> dict:
    """Round 2's freeze: the look set's cause split beside #178 and the teacher, G5's list,
    the ``د→ت`` reading, the power line and the coverage. Refuses unless the pre-registration
    is committed and pinned, before reading any decode."""
    from training.consonant_readout import freeze_tables
    from training.tlog_eval import excluded_record

    harness = sources.harness_identity()
    prereg = require_preregistration(spec, harness)
    data = sources.tlog(LOOK, FREEZE_MODELS)
    grades = sources.grade(data.items, data.decodes, FREEZE_MODELS)
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
        "excluded": excluded_record(data.excluded),
        **freeze_tables(grades, models=FREEZE_MODELS, labels=FREEZE_LABELS,
                        baseline=spec.baseline, required=(DAL_TA,)),
        "dal_ta_reading": dal_ta_reading(grades),
        "power_check": power_check(grades),
        "coverage": coverage(grades),
    }


def run_freeze(spec: RoundSpec, sources, model: str | None = None) -> dict:
    """One of the round's three freeze models' look-set cause split and top substitutions, for
    the freeze command's dry run: it refuses without the committed pre-registration and refuses
    any other model by name (an unpinned role included), grades, and writes nothing."""
    from training.consonant_readout import freeze_report

    model = model or spec.baseline
    if model not in FREEZE_MODELS:
        raise SystemExit(f"{model} is not a freeze model: the freeze reads "
                         f"{', '.join(FREEZE_LABELS[m] for m in FREEZE_MODELS)}")
    require_preregistration(spec, sources.harness_identity())
    data = sources.tlog(LOOK, (model,))
    return freeze_report(sources.grade(data.items, data.decodes, (model,)), model)


def models_record(spec: RoundSpec) -> dict:
    """The look sitting's ``models.json``: each model it reads, its identity record from the
    round's pins (the teacher's from the truth baseline's committed identity). The run issues
    fill the four arms' pins; an unpinned arm is refused by name, never resolved to #94's
    checkpoint."""
    from training.aug_readout import look_graded_models
    from training.aug_readout_gate import (baseline_identities, expected_fingerprints,
                                           identity_record, model_spec)

    for model in look_graded_models(spec):
        if model not in spec.models:
            raise SystemExit(f"{model} is not pinned by {spec.name}: the run issue fills in its "
                             "checkpoint, and no manifest can name it before that")
    records = {}
    for model in (*look_graded_models(spec), BASE):
        pinned = spec.models.get(model)
        records[f"{spec.look_set}/{model}"] = {
            "model": model,
            "fingerprints": expected_fingerprints(model_spec(model, spec=spec).ref),
            "identity": identity_record(pinned) if pinned is not None
                        else baseline_identities()[model]}
    return records


def write_models(spec: RoundSpec, path: Path | None = None) -> dict:
    """Write the round's ``models.json``, once: the look sitting's identities from the spec's
    pins. It reads no decode and grades nothing; the look phase refuses without it."""
    from training.aug_readout_gate import Layout
    from training.tashkeel_eval import write_text_atomically

    path = Layout(spec.root).models if path is None else path
    if path.exists():
        raise SystemExit(f"{path.name} exists: the round's look identities are fixed")
    records = models_record(spec)
    write_text_atomically(path, json.dumps(records, ensure_ascii=False, indent=1) + "\n")
    return records


def input_files(sources, phase: str, weights=None) -> list[Path]:
    """Round 2's sitting inputs: the look set's decodes of every model it reads, or the confirm
    sitting's every read set, the teacher's decodes and the committed baselines it reuses, G6's
    caches, the round's own frozen selections and the two claimable tranches' manifests; and the
    pinned pre-registration (round 1's roster, ``aug_readout.Sources.input_files``)."""
    from tadabur.exposure import EVALSET_MANIFEST_PATH
    from tadabur.nonpro_sources import QDAT, TARTEEL_V1
    from tadabur.qud_overlap import PILOT
    from tadabur.tlog_tranches import TRANCHES_DIR
    from training.aug_readout import look_graded_models
    from training.consonant_readout import G6_SITES_PATH, MODELS_PATH, QUD_SELECTION_PATH
    from training.nonpro_eval import decodes_file as committed
    from training.nonpro_gap import GRID_PATH
    from training.qud_pilot import decodes_file as qud_committed
    from training.truth_baseline import CHECK_STORE, DECODE_STORE

    if phase == LOOK:
        decodes = [sources.decode_path(sources.spec.look_set, model)
                   for model in (*look_graded_models(sources.spec), BASE)]
        return [*decodes, TRANCHES_DIR / f"{LOOK}.jsonl", sources.spec.prereg_path]
    from training.aug_readout_gate import BASELINES, needs

    read = needs(phase, weights, sources.spec)
    models = sorted({model for names in read.values() for model in names})
    decodes = [sources.decode_path(set_name, model)
               for set_name, names in read.items() for model in names]
    # The teacher's decode of every set the sitting grades, and G6's caches: declared even when
    # absent, so a cannot-certify record names what is missing.
    teachers = [sources.decode_path(set_name, BASE) for set_name in read]
    caches = [store.path(model) for model in models for store in (CHECK_STORE, DECODE_STORE)]
    return [*decodes, *teachers,
            *(committed(dataset, model) for dataset in (TARTEEL_V1, QDAT)
              for model in (*BASELINES,)),
            *(qud_committed(slug, model) for slug in PILOT for model in BASELINES),
            *caches,
            QUD_SELECTION_PATH, G6_SITES_PATH, MODELS_PATH,
            sources.spec.root / "freeze.json",
            *(TRANCHES_DIR / f"{role}.jsonl" for role in ("confirm.1", "confirm.2")),
            EVALSET_MANIFEST_PATH, GRID_PATH, sources.spec.prereg_path]
