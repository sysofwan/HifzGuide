"""The catch-loss audit (#225, #257): blind owner listening on the look phase's gating.

A candidate can lower its flag count by no longer catching real mistakes, so the round's bar 2
is pre-claim-gated on a direct measurement of what the candidate stops flagging
(``docs/consonant-round2-preregistration.md`` and, for round 3,
``docs/consonant-round3-preregistration.md``, "Catch-loss audit"). This module draws the
audit's blind worklist and scores the owner's verdicts; :mod:`tadabur.tashkeel_audit_ui`
serves it through the #61 listening UI's whole-word question.

**Draw** (:func:`draw`). The population is the look sitting's gating words the audit's models
graded at the shipped default (:mod:`training.aug_readout`'s ``look_grades.jsonl.gz`` for C95
and the committed live/EMA choice, and, on a round whose spec names a reference model, its
grades of the same words from ``look_grades_<reference>.jsonl.gz`` beside it). With two models
(round 2), *removed* is a word C95 flags and the candidate does not, *added* the reverse. With
a reference (round 3's ``r2``), *removed* is a word C95 and the reference flag and the
candidate does not — round 2's removals are already audited — and *added* a word the candidate
flags and neither C95 nor the reference does. Each stratum is sampled without replacement, 100
removed and 50 added (all of a smaller stratum), seeded by the spec's pre-registered seed and
ranked by a salted hash of the word, so the draw depends only on the seed and the inputs.
Three files land in ``<root>/catch_loss/``, committed before any listening:

* ``worklist.jsonl`` — all the page may know: an opaque id, the clip's provenance (split,
  shard, row, audio file name and SHA-256, the recorded span), the ayah, the 0-based Uthmani
  word index and the word's text, sorted by id so the file's order says nothing of the
  stratum;
* ``key.jsonl`` — never served: each item's stratum and sampling probability, and each
  model's grade of the word (``c95``, the reference on a three-model round, and ``candidate``;
  outcome, causes and the consonant substitutions behind them);
* ``summary.json`` — the seed, the choice's commit, each stratum's population (words and
  causes) and the sample size actually taken.

Draw refuses while the pre-registration is not committed and pinned to the round's spec,
while ``choice.json`` or the export is not committed, when an export row is not a look
gating row of the tranche manifest, when a three-model round's reference grades are not
committed or do not repeat the export's baseline grades word for word or do not reproduce the
pre-registration's disclosed counts, and once a worklist exists.

**Score** (:func:`score`). Joins ``verdicts.jsonl`` (the UI's ``correct`` / ``mistake`` /
``unclear``) to ``key.jsonl`` and writes ``result.json``: the share of ``mistake`` among the
removed stratum's decisive verdicts is at most the pre-registered 20% (exactly 20% passes),
and fewer than 80 decisive removed verdicts cannot certify whatever the share. Both strata
are reported by cause, with unclear counts and the words whose flag is one of the owner's
harder pairs (both letters of a substitution in :data:`HARD_LETTERS`) marked as noisier
verdicts.

Usage (from ``tools/``)::

  python -m tadabur.catch_loss_audit draw [--spec NAME] [--root DIR] [--tranche PATH]
  python -m tadabur.catch_loss_audit score [--spec NAME] [--directory DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from training.aug_readout import LOOK_GRADES, look_grade_parity, look_reference_path
from training.aug_readout_gate import Layout, committed_choice, is_committed, sha256_file
from training.aug_readout_metrics import CANDIDATE
from training.aug_readout_spec import (CONSONANT2, CONSONANT2_CATCH_LOSS_SEED, SPECS,
                                       RoundSpec)
from training.counterfactual_eval import wilson_interval
from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE
from training.tashkeel_eval import write_text_atomically

from .listening_session import read_verdicts
from .tlog_index import read_jsonl
from .tlog_pass1 import read_jsonl_gz
from .truth_sites import UNCLEAR
from .word_sample import WORD_VERDICTS

#: Round 2's pre-registered seed: the draw is reproducible from it and the committed inputs.
#: Each round's draw uses its own spec's ``catch_loss_seed`` (round 3's is in
#: :data:`training.aug_readout_spec.CONSONANT3`).
SEED = CONSONANT2_CATCH_LOSS_SEED
REMOVED, ADDED = "removed", "added"
#: The words drawn per stratum (all of a stratum smaller than its sample).
SAMPLE = {REMOVED: 100, ADDED: 50}
#: The owner's harder pairs: a flag is noisier when one substitution has both letters here.
HARD_LETTERS = frozenset("سصث")

WORKLIST = "worklist.jsonl"
KEY = "key.jsonl"
SUMMARY = "summary.json"
VERDICTS = "verdicts.jsonl"
RESULT = "result.json"

CAVEATS = (
    "A word's position is the canonical word the grade's Muraja word starts on: a Muraja word "
    "spanning two canonical words highlights only the first.",
    "Each clip is played whole; a row's start/end samples are provenance, not a cut.",
    "Fewer than 80 decisive removed verdicts certifies nothing, whatever the share is.",
)


# --- the draw -----------------------------------------------------------------------------------


def _rank(seed: str, stratum: str, word: Mapping) -> str:
    """A word's place in its stratum's draw: a salted hash, so the order is the seed's alone."""
    row = word["c95"]
    return hashlib.sha256(
        f"{seed}:{stratum}:{row['key']}:{row['position']}".encode("utf-8")).hexdigest()


def _identifier(seed: str, row: Mapping) -> str:
    """The page's opaque id for a word: the seed and its provenance, and nothing of a model."""
    return hashlib.sha256(
        f"{seed}:{row['split']}:{row['shard_file']}:{row['row']}:{row['word_index']}"
        .encode("utf-8")).hexdigest()[:16]


def graded_words(rows: Sequence[Mapping], *, baseline: str, candidate: str,
                 reference: str | None = None) -> list[dict]:
    """Each word the audit's models graded on a look gating clip, in one population or neither.

    Two models (no ``reference``): *removed* when ``baseline`` flags the word and ``candidate``
    does not, *added* the reverse. Three (``reference`` names round 3's ``r2``): *removed* when
    ``baseline`` and ``reference`` flag and ``candidate`` does not — round 2's removals, the
    words ``baseline`` flags and ``reference`` does not, are already audited — *added* when
    ``candidate`` flags and neither of the other two does. A word graded by one model only is in
    neither population (the audit compares the same word's grades), and a word every model flags
    or none flags is in neither stratum.

    A flag is the export's ``flagged`` boolean, not its ``outcome``: a word grade's outcome is
    ``flagged_unattributed`` (the carrier's group is unknown) or ``flagged_elsewhere``, never
    ``flagged``, and every one is still a flag (:data:`training.muraja_policy.WORD_FLAGGED`).
    """
    names = (baseline, candidate) if reference is None else (baseline, reference, candidate)
    by_word: dict[tuple[str, int], dict[str, Mapping]] = {}
    for row in rows:
        if row["model"] not in names:
            raise ValueError(f"the export carries {row['model']!r}, not the audit's models "
                             f"{names}")
        models = by_word.setdefault((row["key"], row["position"]), {})
        if row["model"] in models:
            raise ValueError(f"{row['key']}/{row['position']}: {row['model']} graded it twice")
        models[row["model"]] = row
    words = []
    for key, position in sorted(by_word):
        graded = by_word[key, position]
        grades = tuple(graded.get(name) for name in names)
        if any(row is None or not row["graded"] for row in grades):
            continue
        if len({(row["word_index"], row["audio_sha256"]) for row in grades}) != 1:
            raise ValueError(f"{key}/{position}: the models graded different words")
        flags = [bool(row["flagged"]) for row in grades]
        c95, cand = grades[0], grades[-1]
        removed = flags[0] and all(flags[1:-1]) and not flags[-1]
        added = flags[-1] and not flags[0] and not any(flags[1:-1])
        if removed == added:
            continue
        word = {"c95": c95, "candidate": cand,
                "stratum": REMOVED if removed else ADDED,
                "flag": c95 if removed else cand}
        if reference is not None:
            word["reference"] = grades[1]
        words.append(word)
    return words


def _population(words: Sequence[Mapping], stratum: str) -> tuple[list[dict], dict]:
    """One stratum's words and its population: the words, and each cause once per word."""
    members = [word for word in words if word["stratum"] == stratum]
    causes = Counter(cause for word in members for cause in set(word["flag"]["causes"]))
    return members, {"words": len(members), "causes": dict(sorted(causes.items()))}


def _sample(members: Sequence[Mapping], size: int, seed: str) -> list[dict]:
    return sorted(members, key=lambda word: _rank(seed, word["stratum"], word))[:size]


def _key_row(word: Mapping, probability: float, seed: str) -> dict:
    def grade(row: Mapping) -> dict:
        return {"outcome": row["outcome"], "causes": row["causes"],
                "consonants": row["consonants"]}

    row = {"id": _identifier(seed, word["c95"]), "stratum": word["stratum"],
           "probability": probability, "c95": grade(word["c95"])}
    if "reference" in word:
        row["reference"] = grade(word["reference"])
    row["candidate"] = grade(word["candidate"])
    return row


def _worklist_row(word: Mapping, uthmani_words: Callable[[str], list[str]],
                  seed: str) -> dict:
    row = word["c95"]
    words = uthmani_words(row["surah_ayah"])
    if not 0 <= row["word_index"] < len(words):
        raise ValueError(f"{row['key']}: word {row['word_index']} is outside "
                         f"{row['surah_ayah']}")
    return {"id": _identifier(seed, row), "split": row["split"],
            "shard_file": row["shard_file"], "row": row["row"],
            "audio_filename": row["audio_filename"], "audio_sha256": row["audio_sha256"],
            "start_sample": row["start_sample"], "end_sample": row["end_sample"],
            "surah_ayah": row["surah_ayah"], "word_index": row["word_index"],
            "word": words[row["word_index"]]}


def _tranche_rows(path: Path) -> dict[tuple[str, str, int], Mapping]:
    """The tranche manifest's rows by (split, shard file, row), refusing a duplicate row."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["split"], row["shard_file"], row["row"])
        if key in rows:
            raise ValueError(f"{path}: {key} is in the manifest twice")
        rows[key] = row
    return rows


def _require_gating(rows: Sequence[Mapping], tranche: Path) -> None:
    manifest = _tranche_rows(tranche)
    keys = {(row["split"], row["shard_file"], row["row"]) for row in rows}
    for key in sorted(keys):
        row = manifest.get(key)
        if row is None:
            raise SystemExit(f"{key} is not a row of {tranche}: the audit reads the look "
                             "tranche's manifest")
        if row["stratum"] != "gating":
            raise SystemExit(f"{key} is {row['stratum']!r}, not a look gating row: only gating "
                             "is audited")
    for row in rows:
        if row["audio_sha256"] != manifest[(row["split"], row["shard_file"],
                                            row["row"])]["audio_sha256"]:
            raise ValueError(f"{row['key']}: the export and {tranche} disagree on its "
                             "audio_sha256")


def _require_preregistration(spec: RoundSpec) -> None:
    path = spec.prereg_path
    if path is None or not spec.pins.prereg_sha256:
        raise SystemExit("the pre-registration's SHA-256 is not pinned in the round's spec: "
                         "the audit is drawn only after the freeze")
    if not is_committed(path):
        raise SystemExit(f"{path} is not committed and unedited: the pre-registration is the "
                         "audit's premise")
    if sha256_file(path) != spec.pins.prereg_sha256:
        raise SystemExit(f"{path} is not the pinned pre-registration: it is edited since its "
                         "commit")


def _reference_rows(export: Path, export_rows: Sequence[Mapping], spec: RoundSpec,
                    reference: str) -> list[Mapping]:
    """The reference model's rows beside the export, after the file repeats the export's
    baseline grades word for word: the reference file is a second grade of the same words,
    never a rewrite of the export."""
    path = look_reference_path(export.parent, reference)
    if not is_committed(path):
        raise SystemExit(f"{path.name} is not committed: a three-model round reads the "
                         f"{reference!r} model's grades of the export's words")
    rows = read_jsonl_gz(path)
    repeated = [row for row in rows if row["model"] == spec.baseline]
    graded = [row for row in export_rows if row["model"] == spec.baseline]
    if repeated != graded:
        raise SystemExit(f"{path.name} does not repeat {export.name}'s C95 grades word for "
                         "word: the reference file is a second grade, not a rewrite")
    reference_rows = [row for row in rows if row["model"] == reference]
    if not reference_rows:
        raise SystemExit(f"{path.name} holds no {reference!r} grades: the draw compares the "
                         f"{reference!r} model beside {spec.baseline} and the candidate")
    if {(row["key"], row["position"]) for row in reference_rows} != {
            (row["key"], row["position"]) for row in graded}:
        raise SystemExit(f"{path.name} grades other words than {export.name}: the two files "
                         "must carry the same words")
    return reference_rows


def _require_disclosure(spec: RoundSpec, rows: Sequence[Mapping],
                        reference: str | None) -> None:
    """Refuse unless the grades reproduce the pre-registration's disclosed counts.

    The counts are the pre-registration's facts about the round's own committed look sitting
    (``docs/consonant-round3-preregistration.md``), so a spec that merely inherits them while
    reading another root — a fixture, a staging copy — is held only to counts it declares
    itself. A round that overrides the declaration, or reads its own root, is held to it.
    """
    declared = spec.catch_loss_disclosure
    if not declared:
        return
    if reference is None:
        raise SystemExit(f"{spec.name} declares disclosed catch-loss counts without a "
                         "reference model: nothing can reproduce them")
    registered = SPECS.get(spec.name)
    if (registered is not None and spec.root != registered.root
            and dict(declared) == dict(registered.catch_loss_disclosure)):
        return
    found = look_grade_parity(rows, baseline=spec.baseline, reference=reference)
    if found != dict(declared):
        raise SystemExit(f"the look grades do not reproduce the pre-registration's disclosed "
                         f"counts: {found} is not {dict(declared)}")


def draw(spec: RoundSpec, *, uthmani_words: Callable[[str], list[str]],
         tranche: Path) -> dict:
    """Draw and write the audit's worklist, key and summary; return the summary."""
    out = spec.root / "catch_loss"
    worklist = out / WORKLIST
    if worklist.exists():
        raise SystemExit(f"{worklist} exists: a worklist is drawn once, never redrawn")
    seed = spec.catch_loss_seed
    if seed is None:
        raise SystemExit(f"{spec.name} declares no catch-loss seed: the audit draws only on a "
                         "round whose pre-registration names one")
    _require_preregistration(spec)
    layout = Layout(spec.root)
    export = layout.root / LOOK_GRADES
    if not is_committed(export):
        raise SystemExit(f"{export.name} is not committed: the audit reads the look sitting's "
                         "committed word grades")
    choice = committed_choice(layout, spec=spec)
    rows = read_jsonl_gz(export)
    _require_gating(rows, tranche)
    reference = spec.catch_loss_reference
    reference_rows = ([] if reference is None
                      else _reference_rows(export, rows, spec, reference))
    _require_disclosure(spec, [*rows, *reference_rows], reference)
    words = graded_words([*rows, *reference_rows], baseline=spec.baseline,
                         candidate=choice.choice["models"][CANDIDATE], reference=reference)
    sampled: dict[str, list[dict]] = {}
    summary = {"seed": seed, "choice_commit": choice.commit, "populations": {},
               "sampled": {}}
    for stratum, size in SAMPLE.items():
        members, population = _population(words, stratum)
        sampled[stratum] = _sample(members, size, seed)
        summary["populations"][stratum] = population
        summary["sampled"][stratum] = len(sampled[stratum])
    if not sampled[REMOVED]:
        raise SystemExit("the look sitting removed no word: nothing can be audited, so the "
                         "audit cannot certify (no worklist is drawn)")
    items = [word for stratum in SAMPLE for word in sampled[stratum]]
    worklist_rows = sorted((_worklist_row(word, uthmani_words, seed) for word in items),
                           key=lambda row: row["id"])
    if len({row["id"] for row in worklist_rows}) != len(worklist_rows):
        raise ValueError("two worklist items share an id")
    probabilities = {stratum: summary["sampled"][stratum] / population["words"]
                     for stratum, population in summary["populations"].items()
                     if population["words"]}
    key = sorted((_key_row(word, probabilities[word["stratum"]], seed) for word in items),
                 key=lambda row: row["id"])
    write_text_atomically(worklist, _jsonl(worklist_rows))
    write_text_atomically(out / KEY, _jsonl(key))
    write_text_atomically(out / SUMMARY, _json(summary))
    return summary


def _json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def _jsonl(rows: Sequence[Mapping]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


# --- the scorer ---------------------------------------------------------------------------------


def _counts(heard: Sequence[str]) -> dict:
    """One set of verdicts' counts, the real-mistake share and its Wilson 95% interval."""
    real = heard.count(MISTAKE_SIDE)
    correct = heard.count(CORRECT_SIDE)
    unclear = len(heard) - real - correct
    decisive = real + correct
    return {"decisive": decisive, "real": real, "correct": correct, "unclear": unclear,
            "real_share": real / decisive if decisive else None,
            "wilson": list(wilson_interval(real, decisive)) if decisive else None}


def _hard_pair(consonants: Sequence[Sequence[str]]) -> bool:
    """Whether one substitution of the flag has both letters in :data:`HARD_LETTERS`."""
    for _, substitution in consonants:
        expected, _, heard = substitution.partition("→")
        if expected in HARD_LETTERS and heard in HARD_LETTERS:
            return True
    return False


def _flagging(stratum: str) -> str:
    return "c95" if stratum == REMOVED else "candidate"


def score(directory: Path, *, spec: RoundSpec = CONSONANT2) -> dict:
    """Score the sitting in ``directory`` and write its ``result.json``; return it."""
    ids = [row["id"] for row in read_jsonl(directory / WORKLIST)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{directory / WORKLIST} holds a duplicate id")
    key_rows = read_jsonl(directory / KEY)
    key = {row["id"]: row for row in key_rows}
    if len(key) != len(key_rows) or set(key) != set(ids):
        raise SystemExit(f"{directory / KEY} does not hold exactly the worklist's items")
    verdicts = read_verdicts(directory / VERDICTS)
    silent = sorted(set(ids) - set(verdicts))
    if silent:
        raise SystemExit(f"{len(silent)} of {len(ids)} worklist items have no verdict; the "
                         f"audit scores only a complete sitting (first: {silent[0]})")
    stray = sorted(set(verdicts) - set(ids))
    if stray:
        raise SystemExit(f"{len(stray)} verdicts are not worklist items (first: {stray[0]})")
    heard: dict[str, str] = {}
    for ident in ids:
        answer = verdicts[ident].heard
        if answer not in WORD_VERDICTS:
            raise ValueError(f"{ident}: {answer!r} is not one of {WORD_VERDICTS}")
        heard[ident] = answer
    at_most = spec.bars["catch_loss_real_at_most"]
    at_least = spec.bars["catch_loss_decisive_at_least"]
    unknown = sorted({row["stratum"] for row in key.values()} - {REMOVED, ADDED})
    if unknown:
        raise ValueError(f"{directory / KEY} names unknown strata: {unknown}")
    counts = {stratum: _counts([heard[row["id"]] for row in key.values()
                                if row["stratum"] == stratum])
              for stratum in (REMOVED, ADDED)}
    if counts[REMOVED]["decisive"] < at_least:
        verdict = "cannot_certify"
    elif counts[REMOVED]["real_share"] <= at_most:
        verdict = "pass"
    else:
        verdict = "fail"
    by_cause = {stratum: _by_cause(key.values(), heard, stratum)
                for stratum in (REMOVED, ADDED)}
    noisy = {stratum: _noisy(key.values(), heard, stratum) for stratum in (REMOVED, ADDED)}
    result = {"verdict": verdict,
              "bars": {"real_at_most": at_most, "decisive_at_least": at_least},
              "removed": counts[REMOVED], "added": counts[ADDED],
              "by_cause": by_cause, "noisy": noisy, "caveats": list(CAVEATS)}
    write_text_atomically(directory / RESULT, _json(result))
    return result


def _by_cause(key: Sequence[Mapping], heard: Mapping[str, str], stratum: str) -> dict:
    """The stratum's verdicts per cause of the flagging model; a word counts in each cause."""
    groups: dict[str, list[str]] = {}
    for row in key:
        if row["stratum"] != stratum:
            continue
        for cause in set(row[_flagging(stratum)]["causes"]):
            groups.setdefault(cause, []).append(heard[row["id"]])
    return {cause: _counts(answers) for cause, answers in sorted(groups.items())}


def _noisy(key: Sequence[Mapping], heard: Mapping[str, str], stratum: str) -> dict:
    """The stratum's items whose flag is a harder pair, and their verdicts."""
    counts = {"items": 0, "decisive": 0, "real": 0}
    for row in key:
        if row["stratum"] != stratum:
            continue
        if not _hard_pair(row[_flagging(stratum)]["consonants"]):
            continue
        counts["items"] += 1
        answer = heard[row["id"]]
        counts["decisive"] += answer != UNCLEAR
        counts["real"] += answer == MISTAKE_SIDE
    return counts


# --- the CLI ------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    #: The round whose spec, seed and root the audit reads; the fields say the round's facts.
    rounds = dict(choices=tuple(SPECS), default=CONSONANT2.name)
    drawing = commands.add_parser("draw", help="draw the blind worklist from the look sitting")
    drawing.add_argument("--spec", **rounds,
                         help="the round's spec (its seed, reference and root)")
    drawing.add_argument("--root", type=Path,
                         help="the round's readout root, overriding its spec's (its choice "
                              "and look export)")
    drawing.add_argument("--tranche", type=Path, required=True,
                         help="the look tranche manifest (tlog/tranches/look.jsonl)")
    scoring = commands.add_parser("score", help="score the owner's verdicts")
    scoring.add_argument("--spec", **rounds,
                         help="the round's spec (its root and bars)")
    scoring.add_argument("--directory", type=Path,
                         help="the audit's directory, overriding the spec's "
                              "catch_loss (its worklist, key and verdicts)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = SPECS[args.spec]
    if args.command == "draw":
        from .waqf_segments import _uthmani_words

        if args.root is not None:
            spec = replace(spec, root=args.root)
        summary = draw(spec, uthmani_words=_uthmani_words, tranche=args.tranche)
        print(json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True))
    else:
        result = score(args.directory or spec.root / "catch_loss", spec=spec)
        print(f"{result['verdict']}: {result['removed']['real']} of "
              f"{result['removed']['decisive']} decisive removed verdicts are real mistakes")


if __name__ == "__main__":
    main()
