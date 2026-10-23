"""The consonant round's own data: its QUD half, freeze, catch arm and C95-against-C95 tracer (#175).

The bars and sets are #174's pre-registration (``docs/consonant-round-preregistration.md``); this
module holds what the round adds to :mod:`training.aug_readout`'s pipeline:

* **The unread QUD half** (:func:`qud_complement_record`): the pilot segments #95's salted half did
  not keep, frozen in ``tadabur/consonant_readout/qud_selection.json``.
* **The freeze** (:func:`run_freeze`, :func:`write_freeze`): a model's cause split and its top
  substitution list, read from the look half only — the confirm half is never frozen (data
  discipline 1). :func:`write_freeze` writes #177's committed ``freeze.json`` and its generated
  doc; the dry run prints one model's summary and writes nothing.
* **The catch arm** (G6, :func:`catch_arm`): Ship A's held-out verified real mistakes at
  ط↔ق and س↔ش, graded through :mod:`training.truth_baseline`'s recorded check sequences, read
  read-only from its caches. The site ids are frozen in ``g6_sites.json``.
* **The tracer** (:func:`tracer_text`): C95 against itself — every point difference 0.00, every
  interval "no bound", so nothing is certified.

Usage (from ``tools/``)::

  python -m training.consonant_readout select-qud
  python -m training.consonant_readout g6
  python -m training.consonant_readout models
  python -m training.consonant_readout freeze --half look
  python -m training.consonant_readout freeze --half look --dry-run
  python -m training.consonant_readout doc
  python -m training.consonant_readout tracer

Files (``tadabur/consonant_readout/``): ``qud_selection.json``, ``g6_sites.json``,
``models.json`` (C95's identity per set), ``freeze.json`` (#177's look-half freeze); its doc is
``docs/consonant-round-freeze.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from tadabur.tarteel_split import CONFIRM, LOOK, half
from training.aug_readout_metrics import (
    BASE,
    CANDIDATE,
    CONSONANT,
    CONSONANT_FLAG_CAUSES,
    DEPARTS,
    EQUALS,
    H448,
    MADD_ALIF,
    PER_1000,
    POINTS,
    TASHKEEL_CAUSE,
    carries_cause,
    compare,
    comparison_name,
    false_flag_terms,
    observed_substitutions,
    substitution_terms,
    substitution_views,
    teacher_agreement_terms,
    word_agreement_terms,
)
from training.aug_readout_spec import C95, CONSONANT1, RoundSpec
from training.muraja_policy import SHIPPED_DEFAULT
from training.tashkeel_eval import write_text_atomically
from training.word_flags import CAUSES

ROUND_DIR = Path(__file__).parent.parent / "tadabur" / "consonant_readout"
QUD_SELECTION_PATH = ROUND_DIR / "qud_selection.json"
G6_SITES_PATH = ROUND_DIR / "g6_sites.json"
MODELS_PATH = ROUND_DIR / "models.json"
#: #177's artifact, written by :func:`write_freeze`; its list fixes G5.
FREEZE_PATH = ROUND_DIR / "freeze.json"
#: #177's generated doc, rendered from ``freeze.json`` by :mod:`training.consonant_freeze_doc`.
FREEZE_DOC_PATH = Path(__file__).parent.parent.parent / "docs" / "consonant-round-freeze.md"
#: The command that writes the freeze; recorded in the artifact for the PR's command log.
FREEZE_COMMAND = "python -m training.consonant_readout freeze --half look"
#: The models the freeze reads on the look half: C95 and the two baseline columns (#174).
FREEZE_MODELS = (C95, H448, BASE)
FREEZE_LABELS = {C95: "C95", H448: "h448", BASE: "teacher"}
#: The committed decode file C95's identity and decodes are recorded under (#95's candidate).
C95_DECODE = "candidate_ema"
#: G6's pairs (#174): the two the pre-registration reads.
G6_MARKS = ("س↔ش", "ط↔ق")
#: The noisier real-mistake marks #174 reports but never gates ("reported only").
DESCRIPTIVE_MARKS = ("س↔ص", "ث", "ض↔ظ")


# --- the unread QUD half -----------------------------------------------------------------------


def qud_complement_record(keys_by_mushaf: Mapping[str, Sequence[str]]) -> dict:
    """Every decoded pilot segment #95's ``aug95-qud-half-2026-10`` rule did **not** keep."""
    from training.aug_readout import QUD_CUTOFF, QUD_SALT, qud_kept

    other = {slug: sorted(key for key in keys if not qud_kept(key))
             for slug, keys in sorted(keys_by_mushaf.items())}
    return {
        "issue": CONSONANT1.issue,
        "rule": "the complement of #95's half: kept when the first 8 bytes (big-endian) of "
                'sha256("<salt>:<key>") are at or above 2**63',
        "salt": QUD_SALT,
        "cutoff": QUD_CUTOFF,
        "segments": {slug: len(keys) for slug, keys in sorted(keys_by_mushaf.items())},
        "kept_counts": {slug: len(k) for slug, k in other.items()},
        "kept": other,
    }


def load_qud_complement(path: Path = QUD_SELECTION_PATH) -> set[str]:
    """The frozen other half, each key re-checked as one #95 kept out."""
    from training.aug_readout import QUD_CUTOFF, QUD_SALT, qud_kept

    record = json.loads(path.read_text(encoding="utf-8"))
    if record["salt"] != QUD_SALT or record["cutoff"] != QUD_CUTOFF:
        raise ValueError(f"{path} was drawn under another rule")
    keys = {key for keys in record["kept"].values() for key in keys}
    if any(qud_kept(key) for key in keys):
        raise ValueError(f"{path} holds a segment #95 kept: it is another half")
    return keys


def write_qud_complement(path: Path = QUD_SELECTION_PATH) -> dict:
    """Freeze the complement from every segment the QUD pilot decoded."""
    from tadabur.qud_overlap import PILOT
    from training.qud_pilot import read_decodes

    if path.exists():
        raise SystemExit(f"{path.name} exists: the half is fixed")
    record = qud_complement_record({slug: sorted(read_decodes(slug, BASE)) for slug in PILOT})
    write_text_atomically(path, json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    return record


# --- freeze: the look half only -----------------------------------------------------------------


def freeze_report(grades: Sequence, model: str, top: int = 10) -> dict:
    """One model's false flags by cause and its top substitutions over the graded words, read
    from the words whose kept grade is a flag (``training.word_flags.Flag``)."""
    causes: Counter = Counter()
    substitutions: Counter = Counter()
    for grade in grades:
        if grade.model != model or grade.scoring != SHIPPED_DEFAULT.name:
            continue
        if not grade.graded or not grade.flagged:
            continue
        for cause in (CONSONANT, TASHKEEL_CAUSE):
            if carries_cause(grade.reason, cause):
                causes[cause] += 1
        # G5's list is the round's consonant causes only: the madd alif is Muraja#263 and a
        # shaddah deletion is its own cause, so neither enters the veto (#174).
        substitutions.update({(cause, substitution)
                              for cause, substitution in grade.reason.consonants
                              if cause in CONSONANT_FLAG_CAUSES
                              and substitution != MADD_ALIF})
    return {
        "model": model,
        "causes": {cause: causes[cause] for cause in (CONSONANT, TASHKEEL_CAUSE)},
        "substitutions": [{"cause": cause, "substitution": substitution, "words": words}
                          for (cause, substitution), words in
                          sorted(substitutions.items(), key=lambda kv: (-kv[1], kv[0]))[:top]],
    }


def _require_look(half: str) -> None:
    """The freeze reads the look half only; the confirm half is the verdict phase's to read
    (:func:`confirm_report`), never the freeze's (data discipline 1)."""
    if half != LOOK:
        raise SystemExit(f"freeze reads the look half only; {half!r} belongs to the verdict "
                         "phase")


def run_freeze(spec: RoundSpec, sources, half: str, model: str | None = None) -> dict:
    """A model's cause split and top substitutions on the **look** half, for the freeze command's
    dry run: it grades and prints, and writes nothing."""
    _require_look(half)
    model = model or spec.baseline
    data = sources.tarteel(LOOK, (model,))
    return freeze_report(sources.grade(data.items, data.decodes, (model,)), model)


def _per_1000(count: int, graded: int) -> float:
    return round(1000 * count / graded, 2) if graded else 0.0


def freeze_tables(grades: Sequence) -> dict:
    """The freeze's two readings from the look half's grades: each model's false flags by
    ``word_flags`` cause under both scorings (with their rates) and C95's canonical
    substitution tally, whole, with each cause's top ten reported beside it — shaddah and the
    madd alif (``ا→∅``) included — plus G5, derived separately from the untruncated
    shipped-default tally: the ten consonant substitutions behind the most C95 false-flagged
    words, the madd alif aside, ties by substitution (Unicode order)."""
    from training.external_corpus import app_scorings, flag_view

    models: dict[str, dict] = {}
    g5_rows: list[dict] = []
    for model in FREEZE_MODELS:
        scorings = {}
        for scoring in (s.name for s in app_scorings()):
            mine = [grade for grade in grades
                    if grade.model == model and grade.scoring == scoring]
            if not mine:
                continue
            view = flag_view(mine)
            graded = view["graded"]
            by_cause = {cause: [row for row in view["substitutions"]
                                if row["cause"] == cause]
                        for cause in CAUSES}
            scorings[scoring] = {
                "graded": graded,
                "flagged": view["flagged"],
                "per_1000": _per_1000(view["flagged"], graded),
                "causes": {cause: {"words": view["causes"][cause],
                                   "per_1000": _per_1000(view["causes"][cause], graded)}
                           for cause in CAUSES},
                # The diagnostic is the canonical tally whole: a shaddah deletion is its own
                # cause and the madd alif is counted (Muraja#263), so both stay in. The record
                # keeps the entire tally, so G5's derivation is auditable from the artifact;
                # only the reported view is truncated, per cause, to ten.
                "substitution_tally": by_cause,
                "substitutions": {cause: rows[:10] for cause, rows in by_cause.items()},
            }
            if model == C95 and scoring == SHIPPED_DEFAULT.name:
                # G5 alone applies the round's exclusions (#174), and it reads the whole tally:
                # the madd alif still takes a display slot, so a truncated list can be short.
                g5_rows = [row for cause in sorted(CONSONANT_FLAG_CAUSES)
                           for row in by_cause[cause]
                           if row["substitution"] != MADD_ALIF]
        if scorings:
            models[model] = {"label": FREEZE_LABELS[model], "scorings": scorings}
    g5 = sorted(g5_rows, key=lambda row: (-row["words"], row["substitution"]))[:10]
    if len(g5) < 10:
        raise SystemExit(f"the look half yields {len(g5)} consonant substitutions; G5 needs 10")
    return {"models": models, "substitutions": g5}


def freeze_input_paths(sources) -> list[Path]:
    """The committed decodes the freeze reads: C95's look-half decodes and the baselines'
    full-corpus decodes, which :meth:`Sources.tarteel` filters to the look half."""
    from tadabur.nonpro_sources import TARTEEL_V1
    from training.aug_readout_gate import TARTEEL_LOOK
    from training.nonpro_eval import decodes_file

    return [sources.decode_path(TARTEEL_LOOK, C95),
            decodes_file(TARTEEL_V1, H448), decodes_file(TARTEEL_V1, BASE)]


def freeze_record(sources) -> dict:
    """#177's record: the look-half cause split for C95, h448 and the teacher under both
    scorings, C95's G5 list, and the inputs, pins and harness that produced them. The confirm
    half is never read."""
    from training.aug_readout_gate import digests, sha256_file
    from training.aug_readout_spec import PREREG_PATH, check_pins
    from training.external_corpus import app_scorings, word_grades

    paths = freeze_input_paths(sources)
    data = sources.tarteel(LOOK, FREEZE_MODELS)
    if any(half(item.audio_filename) != LOOK for item in data.items):
        raise SystemExit("the freeze read a word outside the look half")
    _, grades = word_grades(data.items, data.decodes, sources.harness(), FREEZE_MODELS,
                            app_scorings())
    harness = sources.harness_identity()
    prereg = sha256_file(PREREG_PATH)
    check_pins(CONSONANT1, harness, prereg)
    return {
        "issue": 177,
        "round": CONSONANT1.name,
        "half": LOOK,
        "command": FREEZE_COMMAND,
        "rule": "C95's Tarteel v1 look-half false flags per 1,000 graded words by "
                "word_flags cause, under the shipped default and with allowances off, graded "
                "through the Muraja harness at the round's pin; G5 is the ten consonant "
                "substitutions behind the most C95 false-flagged words at the shipped default, "
                "the madd alif (ا→∅) aside, ties by substitution (Unicode order)",
        "pins": {"muraja_commit": CONSONANT1.pins.muraja_commit,
                 "harness_sha256": CONSONANT1.pins.harness_sha256,
                 "prereg_sha256": CONSONANT1.pins.prereg_sha256},
        "harness": harness,
        "inputs": digests(paths),
        "items": len(data.items),
        **freeze_tables(grades),
    }


def write_freeze(sources, half: str, path: Path = FREEZE_PATH,
                 doc_path: Path = FREEZE_DOC_PATH) -> dict:
    """Write #177's committed record and its generated doc. Refuses the confirm half before
    reading anything, and refuses to move a freeze that already exists."""
    _require_look(half)
    if path.exists():
        raise SystemExit(f"{path.name} exists: the freeze is fixed")
    from training.consonant_freeze_doc import render

    record = freeze_record(sources)
    write_text_atomically(path, json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    write_text_atomically(doc_path, render(record))
    return record


def frozen_substitutions(spec: RoundSpec, path: Path = FREEZE_PATH) -> tuple[str, ...]:
    """The substitution list #177 froze before training; empty until it lands, so G5 cannot
    certify."""
    if spec.name != CONSONANT1.name or not path.exists():
        return ()
    record = json.loads(path.read_text(encoding="utf-8"))
    return tuple(row["substitution"] for row in record["substitutions"])


# --- the catch arm (G6) -------------------------------------------------------------------------


def _held_out_mistakes(marks: Sequence[str] = G6_MARKS) -> list:
    """Ship A's held-out verified sites heard as the partner letter at ``marks``, before the
    shared exclusions (an existing definition, applied as :mod:`training.truth_baseline` does)."""
    from tadabur.ship_a import HELD_OUT
    from tadabur.ship_a_verified import SHIP_A, load_verified
    from training.site_outcomes import MISTAKE_SIDE, side

    sites, designs = load_verified()
    return [site for site in sites
            if site.mark in marks and side(site) == MISTAKE_SIDE
            and site.site_id in designs
            and (designs[site.site_id].frame, designs[site.site_id].half) == (SHIP_A, HELD_OUT)]


def g6_sites() -> list:
    """The held-out mistakes less the shared exclusions, as the readout froze them."""
    from tadabur.exclusions import current

    excluded = current()
    return [site for site in _held_out_mistakes() if site.site_id not in excluded]


def load_g6_sites(path: Path = G6_SITES_PATH) -> list[str]:
    """The frozen G6 site ids, in the order they were frozen."""
    return json.loads(path.read_text(encoding="utf-8"))["sites"]


def g6_scope(path: Path = G6_SITES_PATH) -> dict:
    """G6's sites **by their frozen ids**, refused when the live recomputation differs, so a
    verdict edit cannot move the denominator between C95's baseline and the readout. Carries the
    held-out mistakes before the shared exclusions and the ids it removed."""
    from tadabur.exclusions import current

    sites = _held_out_mistakes()
    excluded = current()
    live = {site.site_id: site for site in sites if site.site_id not in excluded}
    frozen = load_g6_sites(path)
    if sorted(live) != sorted(frozen):
        raise SystemExit(f"{path.name} and the live G6 list differ: G6's denominator moved")
    return {"sites": [live[key] for key in frozen], "candidates": len(sites),
            "excluded": sorted(site.site_id for site in sites if site.site_id in excluded)}


def write_g6_sites(path: Path = G6_SITES_PATH) -> list[str]:
    """Freeze G6's ids, so the denominator cannot drift between C95 and the readout, with the
    held-out mistakes the shared exclusions removed."""
    from tadabur.exclusions import current

    if path.exists():
        raise SystemExit(f"{path.name} exists: G6's sites are fixed")
    sites = _held_out_mistakes()
    excluded = current()
    kept = [site.site_id for site in sites if site.site_id not in excluded]
    write_text_atomically(path, json.dumps(
        {"issue": CONSONANT1.issue, "rule": "Ship A's held-out verified real mistakes at ط↔ق "
         "and س↔ش, heard as the partner letter, less the shared exclusions",
         "candidates": len(sites),
         "excluded": sorted(site.site_id for site in sites if site.site_id in excluded),
         "sites": kept}, ensure_ascii=False, indent=1) + "\n")
    return kept


def catch_difference(candidate: int, baseline: int) -> int:
    """The paired count of flagged sites, candidate − baseline (G6's bar reads it)."""
    return candidate - baseline


def catch_table(sites: Sequence, outcomes: Mapping[str, Mapping[str, str]],
                baseline: str) -> dict:
    """G6's table from each model's kept outcome per site: the flag counts, and the paired
    count difference against ``baseline``, read from the models actually there (the first is the
    candidate). ``None`` difference when the baseline is not among them."""
    from training.muraja_policy import WORD_FLAGGED

    table = {"sites": [site.site_id for site in sites], "models": {}}
    for name, per_site in outcomes.items():
        table["models"][name] = {
            "graded": len(sites),
            "flagged": sum(outcome in WORD_FLAGGED for outcome in per_site.values()),
            "outcomes": dict(per_site)}
    first = next(iter(outcomes), None)
    table["difference"] = (catch_difference(table["models"][first]["flagged"],
                                            table["models"][baseline]["flagged"])
                           if first is not None and baseline in table["models"] else None)
    return table


def binds_cache(recorded: Mapping, expected: Mapping) -> bool:
    """Whether a truth-baseline cache's identity is the round's: the same checkpoint's bytes,
    the same weight selector, and no recorded field (step, recipe, where a cache keeps them)
    that disagrees. One checkpoint holds live and averaged weights, so its hash names neither:
    a selector-free identity means the live weights, never the pin's EMA (#175 review R2.1,
    R3.1)."""
    from training.truth_baseline import LIVE

    if recorded.get("model_ref") != expected["model_ref"] \
            or recorded.get("checkpoint_sha256") != expected["checkpoint_sha256"] \
            or recorded.get("weights", LIVE) != expected["weights"]:
        return False
    return all(recorded[key] == value for key, value in expected.items() if key in recorded)


def _expected_identity(spec: RoundSpec, name: str,
                       identities: Mapping[str, Mapping]) -> dict | None:
    """The identity ``name``'s caches must be: the sitting's validated one for the candidate,
    the round's pin for C95. The truth baseline's own caches are already bound to the frozen
    protocol and the database's checkpoint, so they need no round identity; any other model
    cannot be bound and G6 cannot read it."""
    from training.aug_readout_gate import identity_record
    from training.truth_baseline import DEFAULT_MODELS

    if name in identities:
        return dict(identities[name])
    if (pinned := spec.models.get(name)) is not None:
        return identity_record(pinned)
    if name in DEFAULT_MODELS:
        return None
    raise ValueError(f"{name}: no identity to bind its caches to; G6 cannot read it")


def catch_arm(sites: Sequence, models: Sequence[str], spec: RoundSpec | None = None,
              cache_dir: Path | None = None, scope: Mapping | None = None,
              identities: Mapping[str, Mapping] | None = None) -> dict | None:
    """Each site's word graded through Muraja on every model's recorded check sequence, reusing
    :func:`training.truth_baseline.app_outcomes`; ``models[0]`` is the candidate and the count
    difference is read against ``spec.baseline``. ``None`` when a model has no cache: a missing
    input, which reads "cannot certify", never an error. ``scope`` (from :func:`g6_scope`) records
    what the shared exclusions removed. ``identities`` are the sitting's validated identities
    (the candidate's, from its manifest): each model's decodes **and** check caches must be that
    identity, so G6 cannot decide against other weights (#175 review R1.2)."""
    from tadabur.staged_audio import REGISTRY_PATH, load_staged_clips
    from training.aug_readout_gate import model_spec
    from training.truth_baseline import (
        CHECK_PROTOCOLS, CHECK_STORE, CHECKS, DECODE_STORE, DEFAULT_MODELS, app_outcomes,
        check_cache, committed_text, items_of, model_caches, read_cache,
    )
    from training.muraja_policy import require_harness

    spec = spec or CONSONANT1
    identities = identities or {}
    directory = Path(cache_dir) if cache_dir is not None else CHECK_STORE.directory
    reading_caches = directory == CHECK_STORE.directory
    missing = [name for name in models
               if not (directory / f"{name}.json").exists()
               or (reading_caches and not DECODE_STORE.path(name).exists())]
    if missing:
        return None
    registry = load_staged_clips(REGISTRY_PATH)
    items = items_of(sites, registry)
    sequences = {}
    for name in models:
        ref = DEFAULT_MODELS.get(name) or model_spec(name, spec=spec).ref
        expected = _expected_identity(spec, name, identities)
        if reading_caches:
            cache, recorded = model_caches(name, ref, items, registry, None)
        else:
            cache = None
            path = directory / f"{name}.json"
            recorded = read_cache(path)
            check_cache(path, recorded, ref, CHECK_PROTOCOLS)
        for kind, held in (("decodes", cache), ("checks", recorded)):
            if expected is not None and held is not None \
                    and not binds_cache(held["identity"] or {}, expected):
                raise ValueError(f"{name}: its {kind} cache is {held['identity']}, not the "
                                 f"round's {expected}")
        sequences[name] = {key: row[CHECKS] for key, row in recorded["items"].items()
                           if key in items}
        if lost := set(items) - set(sequences[name]):
            raise SystemExit(f"{name}: its check cache lacks {len(lost)} of G6's items")
    decodes = {name: {key: committed_text(checks) for key, checks in sequences[name].items()}
               for name in models}
    _, kept = app_outcomes(sites, [], decodes, sequences, require_harness())
    scoring = SHIPPED_DEFAULT.name
    outcomes = {name: {site.site_id: kept.grades[scoring][name][site.site_id].outcome
                       for site in sites} for name in models}
    table = catch_table(sites, outcomes, spec.baseline)
    if scope is not None:
        table["scope"] = {"candidates": scope["candidates"], "excluded": scope["excluded"]}
    return table


def descriptive_sites() -> dict[str, list]:
    """#174's report-only real-mistake marks, less the shared exclusions, grouped by mark."""
    from tadabur.exclusions import current

    excluded = current()
    groups: dict[str, list] = {mark: [] for mark in DESCRIPTIVE_MARKS}
    for site in _held_out_mistakes(DESCRIPTIVE_MARKS):
        if site.site_id not in excluded:
            groups[site.mark].append(site)
    return groups


def held_sites(sites: Sequence, models: Sequence[str], cache_dir: Path | None = None) -> list:
    """The sites every model's check cache holds, from the caches' keys (no replay)."""
    from training.truth_baseline import CHECK_STORE, item_key, read_cache

    directory = Path(cache_dir) if cache_dir is not None else CHECK_STORE.directory
    held: set[str] | None = None
    for name in models:
        path = directory / f"{name}.json"
        if not path.exists():
            return []
        keys = set(read_cache(path)["items"])
        held = keys if held is None else held & keys
    return [site for site in sites if item_key(site) in (held or set())]


def descriptive_catch(models: Sequence[str], spec: RoundSpec | None = None,
                      cache_dir: Path | None = None,
                      identities: Mapping[str, Mapping] | None = None) -> dict:
    """The noisier mistake marks #174 reports and never gates: per mark, how many sites survive
    the shared exclusions, how many both caches hold, and each model's flagged count there. It
    never enters G6's frozen denominator; a mark with no surviving site is disclosed."""
    spec = spec or CONSONANT1
    out = {}
    for mark, sites in descriptive_sites().items():
        readable = held_sites(sites, models, cache_dir)
        table = (catch_arm(readable, models, spec=spec, cache_dir=cache_dir,
                           identities=identities) if readable else None)
        out[mark] = {"sites": len(sites), "read": len(readable),
                     "models": {} if table is None else {
                         name: {"graded": row["graded"], "flagged": row["flagged"]}
                         for name, row in table["models"].items()}}
    return out


# --- C95 against C95: the tracer ----------------------------------------------------------------


def tracer_text(views: Mapping[str, Mapping],
                comparisons: Sequence[tuple[str, str]] = ((CANDIDATE, C95),),
                reproduction_totals: Mapping | None = None) -> str:
    """The tracer's summary: the #95 reproduction check's totals, then one line per table with
    each paired difference's point, its interval and the fact that nothing is certified."""
    from training.aug_readout_doc import interval_text

    names = [comparison_name(*pair) for pair in comparisons]
    lines = ["C95 against C95: no candidate exists yet, so every read is plumbing.", ""]
    if reproduction_totals is not None:
        totals = ", ".join(f"{name} {row['flagged']}/{row['graded']}"
                           for name, row in reproduction_totals.items())
        lines += [f"the #95 reproduction check: {totals}", ""]
    for table, view in views.items():
        for name in names:
            row = view["comparisons"].get(name)
            if row is not None:
                lines.append(f"{table} ({name}): point difference {row['point']:.2f} / "
                             f"{interval_text(row['interval'])} / cannot certify")
    return "\n".join(lines)


def tracer_spec(spec: RoundSpec = CONSONANT1) -> RoundSpec:
    """The round's spec with both the candidate and C95 reading C95's committed decodes. Only
    the tracer does this: the production round reads its candidate's own files."""
    from training.aug_readout_spec import AUG95_ROOT, Decodes

    shadow = {name: Decodes(AUG95_ROOT, C95_DECODE) for name in
              ("tarteel_v1.look", "qdat", "tadabur_dev_unseen")}
    return dataclasses.replace(
        spec, decode_sources={**spec.decode_sources,
                              **{(name, CANDIDATE): source
                                 for name, source in shadow.items()}})


def tracer_views(sources, workers: int = 1) -> dict:
    """The look half, qdat and Tadabur dev, C95 against itself (:func:`tracer_text`'s views)."""
    from training import aug_readout as ar

    models = (CANDIDATE, C95)
    views = {}
    for name, read in (("tarteel_v1.look",
                        lambda: sources.tarteel(LOOK, (*models, BASE))),
                       ("qdat", lambda: sources.qdat((*models, BASE)))):
        data = read()
        grades = sources.grade(data.items, data.decodes, models)
        views[f"{name}.false_flags"] = compare(
            false_flag_terms(grades, models, SHIPPED_DEFAULT.name), PER_1000)
        views[f"{name}.word_agreement"] = compare(
            word_agreement_terms(data.items, ar.checks_text(data, (*models, BASE)), models,
                                 workers=workers), POINTS)
    tadabur = sources.tadabur(models)
    views["tadabur_dev_unseen.teacher_agreement"] = compare(
        teacher_agreement_terms(tadabur.clips, tadabur.decodes, models), POINTS)
    return views


def tracer(sources, workers: int = 1) -> dict:
    """The #95 reproduction check (its totals only), then C95 against itself: every point
    difference 0.00 and every interval unbounded, so nothing certifies."""
    from training import aug_readout as ar

    totals = ar.reproduction(ar.Sources())
    ar.require_reproduction(totals)
    return {"reproduction": totals, "views": tracer_views(sources, workers)}


# --- C95's identity -----------------------------------------------------------------------------


def write_models(path: Path = MODELS_PATH) -> dict:
    """C95's identity per set, from #95's committed identity records."""
    from training.aug_readout_gate import READOUT_DIR

    if path.exists():
        raise SystemExit(f"{path.name} exists: C95's identity is frozen")
    committed = json.loads((READOUT_DIR / "models.json").read_text(encoding="utf-8"))
    records = {}
    for set_name in ("tarteel_v1.look", "tarteel_v1.confirm", "qdat", "qud_half",
                     "tadabur_dev_unseen"):
        records[f"{set_name}/{C95}"] = {**committed[f"{set_name}/{C95_DECODE}"], "model": C95}
    write_text_atomically(path, json.dumps(records, ensure_ascii=False, indent=1) + "\n")
    return records


# --- the round's confirm sitting ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RoundData:
    """Every set the round's confirm sitting reads, with the roles bound. A set whose decodes
    are not committed is ``None`` and named in ``absent``: it is reported as an absent input and
    every rule that reads it cannot certify, never a crash (#174, review R1.3)."""

    tarteel: object | None
    qdat: object | None
    qud_other: object | None
    qud_half: object | None
    tadabur: object | None
    bench: Mapping[str, str]
    catch: Mapping | None
    #: The report-only mistake marks (#174); never G6's frozen denominator.
    report_only_catch: Mapping | None = None
    absent: tuple[str, ...] = ()


ROUND_ROLES = (CANDIDATE, C95, H448, BASE)
STUDENTS = (CANDIDATE, C95, H448)


def bind_round(decodes: Mapping[str, Mapping], weights: str) -> dict:
    """Role names to decodes: the candidate's chosen weights, then C95, ``h448`` and the teacher."""
    from training.aug_readout_gate import arm_model

    roles = {arm_model(CANDIDATE, weights): CANDIDATE, C95: C95, H448: H448, BASE: BASE}
    return {roles.get(name, name): rows for name, rows in decodes.items()}


def confirm_report(data: RoundData, spec: RoundSpec = CONSONANT1, workers: int = 1) -> dict:
    """The round's confirm tables and verdict, the bars read as #174 states them, and every
    "reported, not gating" item the pre-registration lists."""
    from training import aug_readout as ar
    from training.aug_readout_metrics import (
        abstention_terms, bench_terms, blank_terms, ungraded_terms,
    )
    from training.aug_readout_verdict import verdict
    from training.nonpro_eval import bench_departures

    shipped = SHIPPED_DEFAULT.name
    comparisons = spec.comparisons
    student = (CANDIDATE, C95, H448)

    def flags(d, cause=None) -> dict | None:
        return None if d is None else compare(
            false_flag_terms(d.grades, ROUND_ROLES, shipped, cause=cause), PER_1000,
            comparisons)

    def agreement(d, subset=None) -> dict | None:
        return None if d is None else compare(
            word_agreement_terms(d.items, ar.checks_text(d, ROUND_ROLES), student,
                                 subset=subset, workers=workers), POINTS, comparisons)

    bench = [] if data.qdat is None else ar.bench_items(data.qdat, data.bench)
    bench_keys = {i.key for i in bench}
    departures = {i.key: bench_departures(i, data.bench[i.audio_filename]) for i in bench}
    departed = [grade for grade in (data.qdat.grades if data.qdat is not None else ())
                if grade.item.key in bench_keys
                and grade.word_index in departures[grade.item.key]]
    bench_views = {} if data.qdat is None else {
        kind: compare(terms, POINTS, comparisons) for kind, terms in bench_terms(
            bench, ar.stream_text(ar.SetData(bench, data.qdat.decodes), ROUND_ROLES),
            data.bench, ROUND_ROLES).items()}
    frozen = frozen_substitutions(spec)
    # Every substitution any model carries is reported; only the freeze's list gates G5 (#174).
    substitutions = {} if data.tarteel is None else substitution_views(
        substitution_terms(data.tarteel.grades, ROUND_ROLES, shipped,
                           observed_substitutions(data.tarteel.grades, ROUND_ROLES, shipped)),
        PER_1000, comparisons)
    tadabur = data.tadabur
    report = {
        "issue": spec.issue,
        "main": {"all_cause": flags(data.tarteel),
                 "consonant_cause": flags(data.tarteel, CONSONANT),
                 "tashkeel_cause": flags(data.tarteel, TASHKEEL_CAUSE),
                 # G4's companion (#174, reported beside it).
                 "qud_other_tashkeel_cause": flags(data.qud_other, TASHKEEL_CAUSE)},
        "mechanism": {"tarteel_v1_confirm": agreement(data.tarteel),
                      "qdat": agreement(data.qdat),
                      "tarteel_v1_confirm_departs": agreement(data.tarteel, DEPARTS),
                      "qdat_departs": agreement(data.qdat, DEPARTS),
                      # Reported beside bar 4b (#174).
                      "tarteel_v1_confirm_equals": agreement(data.tarteel, EQUALS),
                      "qdat_equals": agreement(data.qdat, EQUALS)},
        "consistency": {"qdat": flags(data.qdat)},
        "guards": {
            "qud_other": flags(data.qud_other),
            # Reported, not gating: the half #95 read.
            "qud_half": flags(data.qud_half),
            "tadabur_dev_unseen": None if tadabur is None else compare(
                teacher_agreement_terms(tadabur.clips, tadabur.decodes, ROUND_ROLES),
                POINTS, comparisons),
            "qdat_bench": bench_views or {kind: None
                                          for kind in ("consonant", "haraka", "shaddah")},
        },
        "substitutions": substitutions,
        "g5_frozen": list(frozen),
        "catch": {"ship_a_verified": data.catch,
                  "ship_a_verified_report_only": data.report_only_catch,
                  # The catch side as #95 read it (RetaSy is #171's).
                  # ``departed`` is already each word's grade: grade it, never read it as a set.
                  "qdat_bench_departures": None if not departed else compare(
                      false_flag_terms(departed, ROUND_ROLES, shipped), PER_1000, comparisons)},
        "leniency": {
            "blank_ratio": None if tadabur is None else compare(
                blank_terms(tadabur.clips, tadabur.blanks, ROUND_ROLES), POINTS, comparisons),
            "abstentions": {name: None if d is None else compare(
                abstention_terms(d.items, ar.checks_text(d, ROUND_ROLES), ROUND_ROLES, workers),
                POINTS, comparisons)
                for name, d in (("tarteel_v1_confirm", data.tarteel), ("qdat", data.qdat))},
            "ungraded": {name: None if d is None else compare(
                ungraded_terms(d.grades, ROUND_ROLES, shipped), PER_1000, comparisons)
                for name, d in (("tarteel_v1_confirm", data.tarteel), ("qdat", data.qdat),
                                ("qud_other", data.qud_other))},
        },
        "sizes": {name: None if d is None else len(d.items)
                  for name, d in (("tarteel_v1_confirm", data.tarteel), ("qdat", data.qdat),
                                  ("qud_other", data.qud_other), ("qud_half", data.qud_half))}
        | {"tadabur_dev_unseen": None if tadabur is None else len(tadabur.clips),
           "qdat_bench": None if data.qdat is None else len(bench),
           "ship_a_verified": len(data.catch["sites"]) if data.catch else 0},
    }
    if data.absent:
        report["absent"] = list(data.absent)
    report["verdict"] = verdict(report, spec=spec)
    return report


def load_confirm(sources, weights: str, spec: RoundSpec = CONSONANT1,
                 identities: Mapping[str, Mapping] | None = None,
                 absent: Sequence[str] = ()) -> RoundData:
    """Every set the round's confirm sitting reads, graded, with the roles bound. A set named
    ``absent`` (its decodes are not committed) is not read. ``identities`` are the sitting's
    validated per-model identities: G6 binds both models' truth-baseline caches to them (and to
    the round's pin) before replay."""
    from training import aug_readout as ar
    from training.aug_readout_gate import (
        QDAT_SET, QUD_OTHER_SET, QUD_SET, TADABUR_SET, TARTEEL_CONFIRM, arm_model,
    )

    absent = tuple(sorted(set(absent)))
    candidate = arm_model(CANDIDATE, weights)
    models = (candidate, C95, H448, BASE)

    def graded(data) -> object:
        decodes = bind_round(data.decodes, weights)
        return ar.SetData(data.items, decodes,
                          sources.grade(data.items, decodes, ROUND_ROLES))

    def read(name, fetch):
        return None if name in absent else graded(fetch())

    tadabur = None if TADABUR_SET in absent else sources.tadabur(models)
    scope = g6_scope()
    catch_identities = ({candidate: identities[candidate]}
                        if identities and candidate in identities else {})
    return RoundData(
        tarteel=read(TARTEEL_CONFIRM, lambda: sources.tarteel(CONFIRM, models)),
        qdat=read(QDAT_SET, lambda: sources.qdat(models)),
        qud_other=read(QUD_OTHER_SET, lambda: sources.qud_other(models)),
        qud_half=read(QUD_SET, lambda: sources.qud(models)),
        tadabur=None if tadabur is None else ar.TadaburData(
            tadabur.clips, bind_round(tadabur.decodes, weights),
            bind_round(tadabur.blanks, weights), tadabur.redecoded_identical),
        bench=sources.bench() if "qdat" not in absent else {},
        catch=catch_arm(scope["sites"], (candidate, C95), spec=spec, scope=scope,
                        identities=catch_identities),
        report_only_catch=descriptive_catch((candidate, C95), spec=spec,
                                            identities=catch_identities),
        absent=absent)


def coverage() -> dict:
    """The round's frozen populations, decoded and reference-evaluable, with the reasons."""
    from training.aug_readout import COVERAGE_STATEMENT, coverage as aug95_coverage

    base = aug95_coverage()["populations"]
    return {"issue": CONSONANT1.issue, "statement": COVERAGE_STATEMENT,
            "populations": {name: base[name]
                            for name in ("tarteel_v1.look", "tarteel_v1.confirm")}
            | {"qud_other": _qud_coverage(load_qud_complement())}}


def _qud_coverage(frozen: set[str]) -> dict:
    """The QUD other half's decoded, evaluated and excluded segments."""
    from tadabur.qud_overlap import PILOT
    from training.qud_pilot import REPORT_PATH, read_decodes

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    unsupported = {key for slug in PILOT
                   for key in report["items"][slug]["phonetizer_unsupported_list"]}
    excluded = sorted(frozen & unsupported)
    lines = {}
    for slug in PILOT:
        lines.update(read_decodes(slug, BASE))
    return {
        "decoded": len(frozen),
        "evaluated": len(frozen) - len(excluded),
        "excluded": len(excluded),
        "excluded_words": sum(lines[key]["segment"][1] - lines[key]["segment"][0] + 1
                              for key in excluded),
        "excluded_keys": excluded,
        "reason": "the QUD pilot left these segments out as phonetizer-unsupported (its "
                  "waqf/leen reference limitation, `qud_eval/pilot/report.json`); every model "
                  "shares the exclusion",
    }


# --- running it ---------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("select-qud", help="freeze the QUD other half")
    commands.add_parser("g6", help="freeze G6's held-out site ids")
    commands.add_parser("models", help="record C95's identity per set")
    freeze = commands.add_parser("freeze", help="write the round's look-half freeze and its doc")
    freeze.add_argument("--half", required=True, choices=(LOOK, CONFIRM))
    freeze.add_argument("--dry-run", action="store_true",
                        help="print one model's look-half summary; write nothing")
    freeze.add_argument("--model", default=None,
                        help="with --dry-run: the model to summarise (C95 by default)")
    commands.add_parser("doc", help=f"render {FREEZE_DOC_PATH.name} from freeze.json")
    tracer_command = commands.add_parser("tracer", help="C95 against C95 on committed decodes")
    tracer_command.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.command == "select-qud":
        record = write_qud_complement()
        print(f"wrote {QUD_SELECTION_PATH}: {record['kept_counts']}")
    elif args.command == "g6":
        print(f"wrote {G6_SITES_PATH}: {write_g6_sites()}")
    elif args.command == "models":
        print(f"wrote {MODELS_PATH}: {sorted(write_models())}")
    elif args.command == "freeze":
        from training import aug_readout as ar

        sources = ar.Sources(CONSONANT1)
        if args.dry_run:
            print(json.dumps(run_freeze(CONSONANT1, sources, args.half, args.model),
                             ensure_ascii=False, indent=1))
        else:
            if args.model:
                raise SystemExit("--model belongs to --dry-run: the freeze reads "
                                 f"{', '.join(FREEZE_LABELS[m] for m in FREEZE_MODELS)}")
            record = write_freeze(sources, args.half)
            print(f"wrote {FREEZE_PATH} and {FREEZE_DOC_PATH}: "
                  f"{len(record['substitutions'])} G5 substitutions, {record['items']} clips")
    elif args.command == "doc":
        from training.consonant_freeze_doc import render

        write_text_atomically(FREEZE_DOC_PATH,
                              render(json.loads(FREEZE_PATH.read_text(encoding="utf-8"))))
        print(f"wrote {FREEZE_DOC_PATH}")
    else:
        from training import aug_readout as ar

        record = tracer(ar.Sources(tracer_spec()), args.workers)
        print(tracer_text(record["views"],
                          reproduction_totals=record["reproduction"]))


if __name__ == "__main__":
    main()
