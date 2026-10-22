"""The readout of #94's augmentation experiment (#95), under the discipline #95 and #128 fix.

#94 continued ``h448`` for 10,000 steps with augmented student input and KL to the clean teacher
(the **candidate**, ``runs/aug94_candidate_v2``, recipe ``aug94-mixture-v2``), and bit-identically
without augmentation (the **control**, ``runs/aug94_control``). Each run keeps two weights at its final step: live and EMA.
This module reads them against ``h448`` and the base teacher (whose decodes are committed and
reused) on the sets #95 names, through the pre-registered bars of
:mod:`training.aug_readout_verdict`, in a fixed order :mod:`training.aug_readout_gate` enforces:

1. ``look``: Tarteel v1's **look** half (:mod:`tadabur.tarteel_split`), checks protocol, read for
   the candidate's live and EMA weights only (the control is never decoded on it): false flags
   per 1,000 graded words (shipped default) and word agreement with the teacher. It chooses one
   thing only, live or EMA (:func:`choose`), and writes ``look.json`` and ``choice.json``; the
   choice binds both arms and holds the candidate's identities and the look inputs' SHA-256.
2. ``confirm``: refuses unless ``choice.json``, ``look.json`` and the look's claim are committed
   (:func:`training.aug_readout_gate.committed_choice`). It reads the **confirm** half and every
   guard once, for the chosen weights, and writes ``report.json`` (with the choice's commit, the
   checkpoint SHA-256s and every input file's SHA-256) and ``verdict.json``: pass, fail or
   "cannot certify", never reinterpreted. A "cannot certify" is resolved by a new
   pre-registration on #128, never by a re-run.

Each sitting first requires a complete identity manifest for every model it reads, then claims
itself exclusively (:func:`training.aug_readout_gate.claim`): it runs once, a concurrent run
fails, and after an interruption only the same sitting may run again.

**Sets** (:data:`~training.aug_readout_gate.SETS`): Tarteel v1's halves; qdat (and qdat_bench, its
clips, stream arm); RetaSy (catch side, restricted: aggregates only); QUD's two professional
mushafs on a **salted 50% of segments** (:func:`qud_selection_record`, frozen in
``qud_selection.json`` before any candidate decode); Tadabur's ``decode_evalset`` dev clips
``h448`` holds no copy of (#159's 671), at ``stream_b0``, where ``h448`` and the base teacher are
decoded again for the blank share only.

**Identity.** Every decode line carries its set's whole provenance (audio name, checksum, span),
each field equal to the committed base decode's of the same item (:func:`read_readout`).

Files (``tadabur/aug_readout/``): ``qud_selection.json``, ``decodes/<set>.<model>.jsonl.gz``
(RetaSy's under the ignored ``restricted/``), ``models.json``, ``look.json``,
``look.claim.json``, ``choice.json``, ``confirm.claim.json``, ``report.json``, ``verdict.json``,
``coverage.json`` (each frozen population's decoded and reference-evaluable sizes, with the keys
the phonetizer could not build a reference for); the doc ``docs/augmentation-readout.md``.

Usage (from ``tools/``; torch-free; ``look`` and ``confirm`` need the Muraja harness)::

  python -m training.aug_readout select-qud                 # once, before any candidate decode
  python -m training.aug_readout merge --decodes-dir <the GPU box's decodes, copied>
  python -m training.aug_readout look                       # then commit look*.json, choice.json
  python -m training.aug_readout confirm
  python -m training.aug_readout coverage                   # then commit coverage.json, doc
  python -m training.aug_readout doc
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tadabur.tarteel_split import CONFIRM, LOOK, half
from training.aug_readout_gate import (
    ARMS,
    BASELINES,
    BLANKS,
    LOOK_MODELS,
    QDAT_SET,
    QUD_SET,
    READOUT_DIR,
    RETASY_SET,
    SETS,
    STREAM,
    TADABUR_SET,
    TARTEEL_CONFIRM,
    TARTEEL_LOOK,
    WEIGHTS,
    CommittedChoice,
    Layout,
    arm_model,
    baseline_identities,
    canonical,
    claim,
    committed_choice,
    decodes_file,
    digests,
    harness_identity,
    make_choice,
    needs,
    read_manifest,
    require_manifest,
    validate_identity,
)
from training.aug_readout_metrics import (
    BASE,
    CANDIDATE,
    COMPARISONS,
    CONTROL,
    H448,
    PER_1000,
    POINTS,
    abstention_terms,
    bench_terms,
    blank_terms,
    clips_flagged_terms,
    compare,
    false_flag_terms,
    teacher_agreement_terms,
    ungraded_terms,
    word_agreement_terms,
)
from training.aug_readout_verdict import verdict
from training.tashkeel_eval import write_text_atomically
from training.truth_baseline import CHECKS

QUD_SELECTION_PATH = READOUT_DIR / "qud_selection.json"
COVERAGE_PATH = READOUT_DIR / "coverage.json"
DOC_PATH = Path(__file__).parent.parent.parent / "docs" / "augmentation-readout.md"
#: What the coverage record states of the verdict's scope (review R1.1, 2026-10-22).
COVERAGE_STATEMENT = (
    "The verdict and every table above apply to the reference-evaluable population recorded "
    "here. The excluded segments were not graded, so this readout does not measure them and "
    "does not bound their possible grades; the exclusions are reference-dependent, common to "
    "every model and not a random additional subsample. In particular the QUD guard's pass "
    "describes the evaluated share of the frozen half, not the whole pre-registered half."
)
#: What ``look.json`` states of the control (#95, settled 2026-10-16).
CONTROL_ON_LOOK = "the control was not decoded on the look half; the choice binds both arms"


# --- QUD: the salted half ---------------------------------------------------------------------

QUD_SALT = "aug95-qud-half-2026-10"
QUD_CUTOFF = 2**63


def qud_kept(key: str) -> bool:
    """Whether a QUD segment (its item key, ``<mp3>#<segment>``) is in the half: the first 8
    bytes, big-endian, of SHA-256 over ``"<salt>:<key>"`` below 2**63."""
    digest = hashlib.sha256(f"{QUD_SALT}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") < QUD_CUTOFF


def qud_selection_record(keys_by_mushaf: Mapping[str, Sequence[str]]) -> dict:
    """The rule over every decoded segment of both mushafs, and the kept keys per mushaf."""
    kept = {slug: sorted(k for k in keys if qud_kept(k)) for slug, keys in
            sorted(keys_by_mushaf.items())}
    return {
        "issue": 95,
        "rule": f'kept when the first 8 bytes (big-endian) of sha256("{QUD_SALT}:" + key) are '
                "below 2**63; key is the segment's item key, <mp3>#<segment index>",
        "salt": QUD_SALT,
        "cutoff": QUD_CUTOFF,
        "segments": {slug: len(keys) for slug, keys in sorted(keys_by_mushaf.items())},
        "kept_counts": {slug: len(k) for slug, k in kept.items()},
        "kept": kept,
    }


def load_qud_selection(path: Path = QUD_SELECTION_PATH) -> set[str]:
    """The frozen half, each key re-checked against the rule."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["salt"] != QUD_SALT or record["cutoff"] != QUD_CUTOFF:
        raise ValueError(f"{path} was drawn under another rule")
    keys = {k for keys in record["kept"].values() for k in keys}
    if any(not qud_kept(k) for k in keys):
        raise ValueError(f"{path} holds a key the rule does not keep")
    return keys


# --- the choice ---------------------------------------------------------------------------------


def choose(inputs: Mapping[str, dict]) -> str:
    """Live or EMA: the candidate's weights with fewer look-half false flags per 1,000; on a tie,
    higher word agreement with the teacher; on a tie again, live (#94's checkpoint rule is the
    final step's live weights unless the look half says otherwise)."""
    def key(weights: str) -> tuple:
        row = inputs[arm_model(CANDIDATE, weights)]
        return (row["false_flags_per_1000"], -row["word_agreement_points"], WEIGHTS.index(weights))

    return min(WEIGHTS, key=key)


CHOICE_RULE = ("the candidate's weights with fewer Tarteel v1 look-half false flags per 1,000 "
               "graded words (shipped default); a tie goes to higher word agreement with the "
               "teacher, then to live; both arms are read at the chosen weights")


# --- the data -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SetData:
    """One set's items (external-corpus items), their decode lines per model by item key, and
    (graded sets) every word's grade per model."""

    items: list
    decodes: Mapping[str, Mapping[str, dict]]
    grades: list = field(default_factory=list)


@dataclass(frozen=True)
class TadaburData:
    """Tadabur's dev clips ``h448`` holds no copy of (:class:`training.nonpro_gap.DevClip`), each
    model's ``stream_b0`` decode per clip (the committed #159 decodes for the baselines) and
    each model's blank count per clip."""

    clips: list
    decodes: Mapping[str, Mapping[str, str]]
    blanks: Mapping[str, Mapping[str, Sequence[int]]]
    redecoded_identical: Mapping[str, int] = field(default_factory=dict)


class Sources:
    """Where the readout reads its inputs: the committed decodes, the datasets' metadata and the
    harness. Tests substitute it; each method reads only the set it names, and
    :meth:`input_files` names every file a sitting's methods read."""

    def input_files(self, phase: str, weights: str | None) -> list[Path]:
        """Every file the sitting reads: the readout decodes of each model it needs, the
        committed baseline decodes it reuses and the frozen selections."""
        from tadabur.exposure import EVALSET_MANIFEST_PATH
        from tadabur.nonpro_sources import QDAT, RETASY, TARTEEL_V1
        from tadabur.qud_overlap import PILOT
        from training.nonpro_eval import decodes_file as committed
        from training.nonpro_gap import DECODES_DIR as GAP_DECODES, GRID_PATH
        from training.qud_pilot import decodes_file as qud_committed

        readout = [decodes_file(s, m) for s, models in needs(phase, weights).items()
                   for m in models]
        if phase == LOOK:
            return [*readout, committed(TARTEEL_V1, BASE)]
        return [*readout,
                *(committed(d, m) for d in (TARTEEL_V1, QDAT, RETASY) for m in BASELINES),
                *(qud_committed(slug, m) for slug in PILOT for m in BASELINES),
                *(GAP_DECODES / f"clean.{m}.jsonl.gz" for m in BASELINES),
                QUD_SELECTION_PATH, EVALSET_MANIFEST_PATH, GRID_PATH]

    def tarteel(self, which_half: str, models: Sequence[str]) -> SetData:
        from tadabur.nonpro_sources import TARTEEL_V1

        return self._nonpro(TARTEEL_V1, f"tarteel_v1.{which_half}", models,
                            lambda item: half(item.audio_filename) == which_half)

    def qdat(self, models: Sequence[str]) -> SetData:
        from tadabur.nonpro_sources import QDAT

        return self._nonpro(QDAT, QDAT_SET, models)

    def retasy(self, models: Sequence[str]) -> SetData:
        from tadabur.nonpro_sources import IN_CORRECT, RETASY

        return self._nonpro(RETASY, RETASY_SET, models,
                            lambda item: item.labels["final_label"] == IN_CORRECT)

    def _nonpro(self, dataset, set_name: str, models: Sequence[str],
                keep: Callable | None = None) -> SetData:
        from tadabur.waqf_segments import hafs_segment_reference
        from training.external_corpus import read_decode_file
        from training.nonpro_eval import build_items, decodes_file as committed

        baselines = {m: read_decode_file(committed(dataset, m)) for m in BASELINES
                     if m == BASE or m in models}
        items, _ = build_items(dataset, list(baselines[BASE].values()), hafs_segment_reference())
        items = [i for i in items if keep is None or keep(i)]
        decodes = {**{m: baselines[m] for m in models if m in BASELINES},
                   **{m: read_readout(set_name, m, baselines[BASE], {i.key for i in items})
                      for m in models if m not in BASELINES}}
        return SetData(items, decodes)

    def qud(self, models: Sequence[str]) -> SetData:
        from training.qud_pilot import load_pilot

        pilot = load_pilot()
        kept = load_qud_selection()
        items = [i for i in pilot.items if i.key in kept]
        keys = {i.key for i in items}
        decodes = {**{m: pilot.decodes[m] for m in models if m in BASELINES},
                   **{m: read_readout(QUD_SET, m, pilot.decodes[BASE], keys)
                      for m in models if m not in BASELINES}}
        return SetData(items, decodes)

    def tadabur(self, models: Sequence[str]) -> TadaburData:
        from training.nonpro_gap import committed_grid, dev_clips, gap_decodes

        seen = set(committed_grid()["sample"]["h448_seen"])
        clips = [c for c in dev_clips() if c.filename not in seen]
        keys = {c.filename for c in clips}
        committed = gap_decodes("clean")
        readout = {m: read_readout(TADABUR_SET, m, committed[BASE], keys)
                   for m in (*models, *BASELINES)}
        decodes = {m: {k: (committed[m] if m in BASELINES else readout[m])[k][STREAM]
                       for k in keys} for m in (*models, *BASELINES)}
        blanks = {m: {k: readout[m][k][BLANKS] for k in keys} for m in readout}
        identical = {m: sum(readout[m][k][STREAM] == committed[m][k][STREAM] for k in keys)
                     for m in BASELINES}
        return TadaburData(clips, decodes, blanks, identical)

    def bench(self) -> dict[str, str]:
        from training.nonpro_eval import bench_transcripts

        return bench_transcripts()

    def harness(self):
        """The one harness every grade of this run uses (located once)."""
        from training.muraja_policy import require_harness

        if not hasattr(self, "_harness"):
            self._harness = require_harness()
        return self._harness

    def harness_identity(self) -> dict:
        return harness_identity(self.harness())

    def grade(self, items: Sequence, decodes: Mapping[str, Mapping[str, dict]],
              models: Sequence[str]) -> list:
        """Every placed word of every item, per model, under the shipped default."""
        from training.external_corpus import word_grades
        from training.muraja_policy import SHIPPED_DEFAULT

        _, grades = word_grades(items, decodes, self.harness(), models, [SHIPPED_DEFAULT])
        return grades


def same_harness(sources: Sources, sitting: Mapping) -> None:
    """Refuse to write a sitting's result if the harness it graded with is not the one it
    claimed."""
    if sources.harness_identity() != sitting["harness"]:
        raise SystemExit("the Muraja harness changed during the sitting; nothing is written")


def read_readout(set_name: str, model: str, base: Mapping[str, dict], keys: set[str]
                 ) -> dict[str, dict]:
    """A model's merged readout decodes of ``set_name`` (``merge`` checked their count), holding
    every item of ``keys`` and only items the base decoded, each line carrying every field of
    its set's provenance, equal to the committed base decode's of its item."""
    from training.external_corpus import read_decode_file

    fields = SETS[set_name].provenance
    lines = read_decode_file(decodes_file(set_name, model))
    if missing := keys - set(lines):
        raise ValueError(f"{set_name}.{model}: {len(missing)} items not decoded")
    if stray := set(lines) - set(base):
        raise ValueError(f"{set_name}.{model}: {len(stray)} items the base never decoded")
    for key, line in lines.items():
        theirs = base[key]
        if absent := [f for f in fields if f not in line or f not in theirs]:
            raise ValueError(f"{set_name}.{model} {key}: no {absent} to check its audio by")
        if any(line[f] != theirs[f] for f in fields):
            raise ValueError(f"{set_name}.{model} {key}: not the audio the base decode read")
    return lines


def bind(decodes: Mapping[str, Mapping], weights: str) -> dict:
    """Model names to roles: each arm's chosen weights as the arm, the baselines as themselves."""
    roles = {arm_model(arm, weights): arm for arm in ARMS}
    return {roles.get(m, m): d for m, d in decodes.items()}


def checks_text(data: SetData, models: Sequence[str]) -> dict[str, dict[str, str]]:
    """Each item's committed transcript of its recorded checks, per model."""
    from training.decoding import committed_text

    return {m: {i.key: committed_text(data.decodes[m][i.key][CHECKS]) for i in data.items}
            for m in models}


def stream_text(data: SetData, models: Sequence[str]) -> dict[str, dict[str, str]]:
    return {m: {i.key: data.decodes[m][i.key][STREAM] for i in data.items} for m in models}


# --- look ---------------------------------------------------------------------------------------


def look_inputs(data: SetData, workers: int = 1) -> dict:
    """The choice's inputs on the look half, per candidate weights: false flags per 1,000 graded
    words (shipped default) and word agreement with the teacher (points). Nothing else."""
    from training.muraja_policy import SHIPPED_DEFAULT

    flags = compare(false_flag_terms(data.grades, LOOK_MODELS, SHIPPED_DEFAULT.name), PER_1000,
                    comparisons=())
    agreement = compare(word_agreement_terms(data.items, checks_text(data, (*LOOK_MODELS, BASE)),
                                             LOOK_MODELS, workers=workers), POINTS, comparisons=())
    return {m: {"graded_words": flags["models"][m]["den"],
                "flagged_words": flags["models"][m]["num"],
                "false_flags_per_1000": flags["models"][m]["point"],
                "word_agreement_points": agreement["models"][m]["point"]}
            for m in LOOK_MODELS}


def run_look(sources: Sources, layout: Layout = Layout(), workers: int = 1) -> dict:
    """Read the look half once and write the choice. Refused once a choice exists, without a
    complete identity manifest for the candidate's two weights, and while (or after) another
    look sitting holds the claim."""
    if layout.choice.exists():
        raise SystemExit(f"the look half was read: {layout.choice.name} exists; it is read once")
    identities = require_manifest(read_manifest(layout), needs(LOOK, None), None)
    sitting = {"phase": LOOK, "identities": identities,
               "inputs": digests(sources.input_files(LOOK, None)),
               "harness": sources.harness_identity()}
    with claim(layout.claim(LOOK), sitting):
        data = sources.tarteel(LOOK, (*LOOK_MODELS, BASE))
        if any(half(i.audio_filename) != LOOK for i in data.items):
            raise ValueError("an item outside the look half reached the look")
        data = SetData(data.items, data.decodes,
                       sources.grade(data.items, data.decodes, LOOK_MODELS))
        inputs = look_inputs(data, workers)
        weights = choose(inputs)
        same_harness(sources, sitting)
        look_text = canonical({"issue": 95, "half": LOOK, "items": len(data.items),
                               "inputs": inputs, "rule": CHOICE_RULE, "choice": weights,
                               "control": CONTROL_ON_LOOK, "identities": identities,
                               "input_sha256": sitting["inputs"],
                               "harness": sitting["harness"]})
        write_text_atomically(layout.look, look_text)
        choice = make_choice(weights, look_text, sitting)
        write_text_atomically(layout.choice, canonical(choice))
    return choice


# --- confirm ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfirmData:
    """Every set the confirm sitting reads, with roles bound (candidate, control, h448, base)."""

    tarteel: SetData
    qdat: SetData
    qud: SetData
    retasy: SetData
    tadabur: TadaburData
    bench: Mapping[str, str]


ROLES = (CANDIDATE, CONTROL, H448, BASE)
STUDENTS = (CANDIDATE, CONTROL, H448)


def bench_items(qdat: SetData, bench: Mapping[str, str]) -> list:
    by_clip = {i.audio_filename: i for i in qdat.items}
    return [by_clip[clip] for clip in sorted(bench)]


def confirm_report(data: ConfirmData, workers: int = 1) -> dict:
    """Every table of #95's confirm sitting, then the verdict (:mod:`training.aug_readout_verdict`)."""
    from training.muraja_policy import SHIPPED_DEFAULT
    from training.nonpro_eval import bench_departures

    shipped = SHIPPED_DEFAULT.name

    def flags(grades: Sequence, models: Sequence[str] = ROLES,
              comparisons: Sequence[tuple[str, str]] = COMPARISONS) -> dict:
        return compare(false_flag_terms(grades, models, shipped), PER_1000, comparisons)

    def agreement(d: SetData) -> dict:
        return compare(word_agreement_terms(d.items, checks_text(d, ROLES), STUDENTS,
                                            workers=workers), POINTS)

    bench = bench_items(data.qdat, data.bench)
    bench_keys = {i.key for i in bench}
    departures = {i.key: bench_departures(i, data.bench[i.audio_filename]) for i in bench}
    departed = [g for g in data.qdat.grades if g.item.key in bench_keys
                and g.word_index in departures[g.item.key]]
    tadabur = data.tadabur
    report = {
        # control − h448 beside the two differences, for the attribution's description only.
        "main": {"tarteel_v1_confirm": flags(data.tarteel.grades,
                                             comparisons=(*COMPARISONS, (CONTROL, H448)))},
        "mechanism": {"tarteel_v1_confirm": agreement(data.tarteel), "qdat": agreement(data.qdat)},
        "consistency": {"qdat": flags(data.qdat.grades)},
        "guards": {
            "qud_half": flags(data.qud.grades),
            "tadabur_dev_unseen": compare(teacher_agreement_terms(tadabur.clips, tadabur.decodes,
                                                                  ROLES), POINTS),
            "qdat_bench": {kind: compare(terms, POINTS) for kind, terms in bench_terms(
                bench, stream_text(SetData(bench, data.qdat.decodes), ROLES), data.bench,
                ROLES).items()},
        },
        "catch": {
            "retasy_in_correct": {"words": flags(data.retasy.grades),
                                  "clips": compare(clips_flagged_terms(data.retasy.grades, ROLES,
                                                                       shipped), POINTS)},
            "qdat_bench_departures": flags(departed) if departed else None,
        },
        "leniency": {
            "blank_ratio": compare(blank_terms(tadabur.clips, tadabur.blanks, ROLES), POINTS),
            "baselines_redecoded_identical": {**dict(tadabur.redecoded_identical),
                                              "clips": len(tadabur.clips)},
            "abstentions": {name: compare(abstention_terms(d.items, checks_text(d, ROLES), ROLES,
                                                           workers), POINTS)
                            for name, d in (("tarteel_v1_confirm", data.tarteel),
                                            ("qdat", data.qdat))},
            "ungraded": {name: compare(ungraded_terms(d.grades, ROLES, shipped), PER_1000)
                         for name, d in (("tarteel_v1_confirm", data.tarteel), ("qdat", data.qdat),
                                         ("qud_half", data.qud))},
        },
        "sizes": {"tarteel_v1_confirm": len(data.tarteel.items), "qdat": len(data.qdat.items),
                  "qud_half": len(data.qud.items), "retasy_in_correct": len(data.retasy.items),
                  "tadabur_dev_unseen": len(tadabur.clips), "qdat_bench": len(bench)},
    }
    report["verdict"] = verdict(report)
    return report


def load_confirm(sources: Sources, weights: str) -> ConfirmData:
    """Every confirm set, graded, with roles bound."""
    chosen = tuple(arm_model(arm, weights) for arm in ARMS)
    models = (*chosen, *BASELINES)

    def graded(data: SetData) -> SetData:
        decodes = bind(data.decodes, weights)
        return SetData(data.items, decodes, sources.grade(data.items, decodes, ROLES))

    tadabur = sources.tadabur(chosen)
    return ConfirmData(
        tarteel=graded(sources.tarteel(CONFIRM, models)),
        qdat=graded(sources.qdat(models)),
        qud=graded(sources.qud(models)),
        retasy=graded(sources.retasy(models)),
        tadabur=TadaburData(tadabur.clips, bind(tadabur.decodes, weights),
                            bind(tadabur.blanks, weights), tadabur.redecoded_identical),
        bench=sources.bench())


def run_confirm(sources: Sources, layout: Layout = Layout(), workers: int = 1,
                gate: Callable[[Layout], CommittedChoice] = committed_choice) -> dict:
    """Read the confirm half and the guards once, for the committed choice, and write the
    report and the verdict. Refused without a committed choice, once a verdict exists, without
    a complete identity manifest for every model read, and while (or after) another confirm
    sitting holds the claim."""
    committed = gate(layout)
    if layout.verdict.exists():
        raise SystemExit(f"{layout.verdict.name} exists: the confirm half is read once")
    weights = committed.choice["weights"]
    identities = require_manifest(read_manifest(layout), needs(CONFIRM, weights),
                                  committed.choice)
    sitting = {"phase": CONFIRM, "choice_commit": committed.commit,
               "choice_sha256": committed.sha256, "identities": identities,
               "inputs": digests(sources.input_files(CONFIRM, weights)),
               "harness": sources.harness_identity()}
    with claim(layout.claim(CONFIRM), sitting):
        report = confirm_report(load_confirm(sources, weights), workers)
        same_harness(sources, sitting)
        report = {"issue": 95, "choice": committed.choice, "audit": sitting,
                  "baselines": baseline_identities(), **report}
        write_text_atomically(layout.report, canonical(report))
        write_text_atomically(layout.verdict, canonical(
            {"issue": 95, "choice": weights, "choice_commit": committed.commit,
             **report["verdict"]}))
    return report


# --- merging the GPU box's decodes -----------------------------------------------------------


def expected_counts() -> dict[str, int]:
    """Items per set, from the committed base decodes and the frozen QUD half."""
    from tadabur.nonpro_sources import QDAT, RETASY, TARTEEL_V1
    from training.external_corpus import read_decode_file
    from training.nonpro_eval import decodes_file as committed
    from training.nonpro_gap import committed_grid, dev_clips

    tarteel = [line["audio_filename"] for line in
               read_decode_file(committed(TARTEEL_V1, BASE)).values()]
    seen = set(committed_grid()["sample"]["h448_seen"])
    return {
        TARTEEL_LOOK: sum(half(c) == LOOK for c in tarteel),
        TARTEEL_CONFIRM: sum(half(c) == CONFIRM for c in tarteel),
        QDAT_SET: len(read_decode_file(committed(QDAT, BASE))),
        RETASY_SET: len(read_decode_file(committed(RETASY, BASE))),
        QUD_SET: len(load_qud_selection()),
        TADABUR_SET: sum(c.filename not in seen for c in dev_clips()),
    }


def coverage() -> dict:
    """Each frozen population's decoded and reference-evaluable sizes, the keys the phonetizer
    could not build a reference for, and the canonical word occurrences they carry.

    Reads the committed base decodes, the frozen QUD selection and the QUD pilot's committed
    left-out lists: the disclosure of the reading's scope, not a new reading."""
    from tadabur.nonpro_sources import TARTEEL_V1
    from tadabur.qud_overlap import PILOT
    from tadabur.waqf_segments import hafs_segment_reference
    from training.nonpro_eval import build_items, decodes_file as committed, read_decode_file
    from training.qud_pilot import REPORT_PATH as QUD_REPORT, read_decodes

    base = read_decode_file(committed(TARTEEL_V1, BASE))
    _, unsupported = build_items(TARTEEL_V1, list(base.values()), hafs_segment_reference())
    unsupported = set(unsupported)
    populations = {}
    for which in (LOOK, CONFIRM):
        decoded = {k for k, line in base.items() if half(line["audio_filename"]) == which}
        excluded = sorted(decoded & unsupported)
        populations[f"tarteel_v1.{which}"] = {
            "decoded": len(decoded),
            "evaluated": len(decoded) - len(excluded),
            "excluded": len(excluded),
            "excluded_words": sum(base[k]["word_to"] - base[k]["word_from"] + 1
                                  for k in excluded),
            "excluded_keys": excluded,
            "reason": "the phonetizer could not build a reference for these clips "
                      "(`training.nonpro_eval.build_items` returns them unsupported); every "
                      "model shares the exclusion",
        }
    frozen = load_qud_selection()
    pilot = json.loads(QUD_REPORT.read_text(encoding="utf-8"))
    unsupported = {key for slug in PILOT
                   for key in pilot["items"][slug]["phonetizer_unsupported_list"]}
    excluded = sorted(frozen & unsupported)
    lines = {}
    for slug in PILOT:
        lines.update(read_decodes(slug, BASE))
    populations[QUD_SET] = {
        "decoded": len(frozen),
        "evaluated": len(frozen) - len(excluded),
        "excluded": len(excluded),
        "excluded_words": sum(lines[k]["segment"][1] - lines[k]["segment"][0] + 1
                              for k in excluded),
        "excluded_keys": excluded,
        "reason": "the QUD pilot left these segments out as phonetizer-unsupported (its "
                  "waqf/leen reference limitation), recorded in `qud_eval/pilot/report.json`; "
                  "every model shares the exclusion",
    }
    return {"issue": 95, "statement": COVERAGE_STATEMENT, "populations": populations}


def merge(decodes_dir: Path, layout: Layout = Layout(),
          gate: Callable[[Layout], CommittedChoice] = committed_choice) -> dict:
    """Every set's per-shard decodes on the GPU box as one gzipped file per set and model, each
    checked against its set's count, with the identity records into ``models.json``. Every
    identity is validated first, and a gated set's decodes are not read before the choice is
    committed and names their weights."""
    from training.external_corpus import merge_decode_files

    found = {}
    for path in sorted(decodes_dir.glob("*.identity.json")):
        set_name, model = path.name.removesuffix(".identity.json").rsplit(".", 1)
        found[(set_name, model)] = path
    choice = gate(layout).choice if any(SETS[s].gated for s, _ in found) else None
    records = read_manifest(layout)
    required: dict[str, list[str]] = {}
    for (set_name, model), path in found.items():
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_identity(set_name, model, record)
        records[f"{set_name}/{model}"] = record
        required.setdefault(set_name, []).append(model)
    require_manifest(records, required, choice)
    counts = expected_counts()
    for (set_name, model), path in found.items():
        merge_decode_files(sorted(decodes_dir.glob(f"{set_name}.*.{model}.jsonl")), path,
                           decodes_file(set_name, model), counts[set_name])
    write_text_atomically(layout.models, canonical(records))
    return records


# --- running it ---------------------------------------------------------------------------------


def write_qud_selection(path: Path = QUD_SELECTION_PATH) -> dict:
    """Freeze the QUD half from every segment the pilot decoded (its committed base decodes)."""
    from tadabur.qud_overlap import PILOT
    from training.qud_pilot import read_decodes

    if path.exists():
        raise SystemExit(f"{path.name} exists: the QUD half is fixed")
    record = qud_selection_record({slug: sorted(read_decodes(slug, BASE)) for slug in PILOT})
    write_text_atomically(path, json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("select-qud", help="freeze QUD's salted half (before any decode)")
    merged = commands.add_parser("merge", help="gather the GPU box's decodes")
    merged.add_argument("--decodes-dir", type=Path, required=True)
    for name, text in (("look", "read the look half once and write the choice"),
                       ("confirm", "read the confirm half and the guards once: the verdict")):
        command = commands.add_parser(name, help=text)
        command.add_argument("--workers", type=int, default=8)
    commands.add_parser("coverage", help="record each frozen population's evaluated share")
    commands.add_parser("doc", help="render docs/augmentation-readout.md")
    args = parser.parse_args()
    layout = Layout()

    if args.command == "select-qud":
        record = write_qud_selection()
        print(f"wrote {QUD_SELECTION_PATH}: {record['kept_counts']} of {record['segments']}")
    elif args.command == "merge":
        print(f"wrote {layout.models}: {sorted(merge(args.decodes_dir, layout))}")
    elif args.command == "look":
        print(f"wrote {layout.choice}: {run_look(Sources(), layout, args.workers)['weights']}")
    elif args.command == "confirm":
        report = run_confirm(Sources(), layout, args.workers)
        print(f"wrote {layout.verdict}: {report['verdict']['verdict']}")
    elif args.command == "coverage":
        record = coverage()
        write_text_atomically(COVERAGE_PATH, canonical(record))
        print(f"wrote {COVERAGE_PATH}: " + ", ".join(
            f"{name} {row['evaluated']} of {row['decoded']}"
            for name, row in record["populations"].items()))
    else:
        from training.aug_readout_doc import render

        look = json.loads(layout.look.read_text(encoding="utf-8"))
        report = json.loads(layout.report.read_text(encoding="utf-8"))
        disclosed = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
        write_text_atomically(DOC_PATH, render(look, report, disclosed))
        print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
