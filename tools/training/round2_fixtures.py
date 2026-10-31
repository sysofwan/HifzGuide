"""Synthetic inputs shared by the round-2 readout tests (#227): no harness, no GPU, no corpus.

``filled_spec`` stands in for what the run issues will commit: the round-2 candidate's and
control's checkpoints, which the real ``CONSONANT2`` refuses until they exist. ``TlogSources`` is
a :class:`training.aug_readout.Sources` that reads only the tlog look set, from items, flags and
decode texts the test hands it, and fails on any other set.
"""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path

from tadabur.qud_text import ItemReference
from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import aug_readout_spec as asp
from training.external_corpus import WordGrade
from training.muraja_policy import SHIPPED_DEFAULT, SiteGrade
from training.nonpro_eval import ClipItem
from training.truth_baseline import DEFAULT_MODELS
from training.word_flags import Flag

SHIPPED = SHIPPED_DEFAULT.name
REFERENCE = "قَالَ سَططَ"
CLEAN = REFERENCE.replace(" ", "")
SWAPPED = "قَالَشَططَ"  # the second word's س heard as ش
OFFSETS = (0, 6)
PREREG_NAME = "consonant-round2-preregistration.md"
R2_MODELS = ("r2_candidate_live", "r2_candidate_ema", "r2_control_live", "r2_control_ema")
RECIPE = "aug94-mixture-v2"


def consonant_flag(substitution: str, cause: str = "other_consonant") -> Flag:
    return Flag((cause,), (), "minor", (), consonants=((cause, substitution),))


TASHKEEL_FLAG = Flag(("tashkeel",), ("fatha dropped",), "tashkeel", ())


# --- the round-2 identities the run issues will fill in ------------------------------------------


def r2_identities() -> dict[str, asp.ModelIdentity]:
    """One checkpoint per arm, read at its live and its averaged weights."""
    return {
        f"r2_{arm}_{weights}": asp.ModelIdentity(
            f"/root/scratch/round2/{arm}/checkpoint.pt", weights, 10_000, RECIPE,
            hashlib.sha256(arm.encode()).hexdigest())
        for arm in ("candidate", "control") for weights in ("live", "ema")}


def filled_spec(*, root: Path | None = None, prereg: tuple[Path, str] | None = None
                ) -> asp.RoundSpec:
    """``CONSONANT2`` with the arms' identities filled and, optionally, its pre-registration
    pointed at a test repository's committed document and pinned to that document's SHA-256."""
    spec = dataclasses.replace(asp.CONSONANT2,
                               models={**asp.CONSONANT2.models, **r2_identities()})
    if root is not None:
        spec = dataclasses.replace(spec, root=root)
    if prereg is not None:
        path, sha = prereg
        spec = dataclasses.replace(
            spec, prereg_path=path,
            pins=dataclasses.replace(spec.pins, prereg_sha256=sha))
    return spec


#: The roles ``CONSONANT2`` pinned before the run issues filled its arms: its baseline and
#: #178's context column.
UNFILLED_ROLES = (asp.C95, asp.C178)


def unfilled_spec() -> asp.RoundSpec:
    """``CONSONANT2`` as it stood before the run issues filled the arms: every ``r2_*`` role is
    refused by name, never resolved to #94's checkpoints."""
    return dataclasses.replace(
        asp.CONSONANT2, models={name: identity
                                for name, identity in asp.CONSONANT2.models.items()
                                if name in UNFILLED_ROLES})


def model_record(spec: asp.RoundSpec, model: str) -> dict:
    """The identity record a decode run writes for one pinned model."""
    return {"model": model,
            "fingerprints": ag.expected_fingerprints(ag.model_spec(model, spec=spec).ref),
            "identity": ag.identity_record(spec.models[model])}


def look_manifest(spec: asp.RoundSpec) -> dict:
    """``models.json`` for the look sitting: the four arms' weights, C95 and the teacher."""
    records = {f"tlog.look/{m}": model_record(spec, m) for m in (*R2_MODELS, asp.C95)}
    records["tlog.look/base"] = {
        "model": "base", "fingerprints": ag.expected_fingerprints(DEFAULT_MODELS["base"]),
        "identity": ag.baseline_identities()["base"]}
    return records


def confirm_manifest(spec: asp.RoundSpec, weights: str = "ema") -> dict:
    """``models.json`` as a round's merges leave it: the look's records plus every confirm-phase
    set's identities, the teacher and the committed baselines included (what a sitting
    validates before it grades)."""
    from tadabur.tarteel_split import CONFIRM

    records = look_manifest(spec)
    for set_name, models in ag.needs(CONFIRM, weights, spec).items():
        for model in models:
            records[f"{set_name}/{model}"] = _identity_record(spec, model)
    return records


def _identity_record(spec: asp.RoundSpec, model: str) -> dict:
    """One decode run's identity record: the round's pin for an arm or a pinned baseline, the
    committed truth-baseline identity for a model the round does not pin."""
    if model in spec.models:
        return model_record(spec, model)
    return {"model": model,
            "fingerprints": ag.expected_fingerprints(DEFAULT_MODELS[model]),
            "identity": ag.baseline_identities()[model]}


# --- a git repository holding the pre-registration -----------------------------------------------


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    (path / "README").write_text("x")
    git(path, "add", "README")
    git(path, "commit", "-qm", "init")
    return path


def write_prereg(repo: Path, text: str = "the pre-registration\n") -> tuple[Path, str]:
    """The round-2 pre-registration in ``repo``'s ``docs/``, written but not committed."""
    path = repo / "docs" / PREREG_NAME
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path, hashlib.sha256(text.encode("utf-8")).hexdigest()


def commit_prereg(repo: Path, text: str = "the pre-registration\n") -> tuple[Path, str]:
    path, sha = write_prereg(repo, text)
    git(repo, "add", f"docs/{PREREG_NAME}")
    git(repo, "commit", "-qm", "freeze the pre-registration")
    return path, sha


# --- tlog items, decodes and word grades ---------------------------------------------------------


def tlog_item(n: int, surah: int) -> ClipItem:
    name = f"{surah}_{n}_{n:09d}.flac"
    return ClipItem(f"{name}#0", "tlog", name, surah, 1, 0, 16000, "e" * 64,
                    ItemReference(("قَالَ", "سَططَ"), (0, 1), REFERENCE, OFFSETS),
                    f"r{n}", surah, {})


def decode_line(item: ClipItem, text: str) -> dict:
    return {"key": item.key, "audio_filename": item.audio_filename, "audio_sha256": "e" * 64,
            "start_sample": 0, "end_sample": 16000, "split": "clean", "shard_file": "clean-0",
            "row": 0, "checks": [{"hop": text, "overlap": "", "flush": True}]}


def word_grade(model: str, item: ClipItem, position: int, flag: Flag | None,
               graded: bool = True) -> WordGrade:
    outcome = "flagged" if flag is not None else "not_flagged" if graded else "not_graded"
    return WordGrade(model, SHIPPED, item, position, 2, position + 1, 7 + position,
                     SiteGrade(outcome), flag)


class TlogSources(ar.Sources):
    """The tlog look set, synthetic: any other set, or any other tranche, fails the test.

    ``flags(model, item_index, position)`` is the word's flag (or ``None``), ``graded(...)``
    whether it is graded, ``texts[model]`` the model's decode of every item (the teacher,
    ``base``, hears the clean reading)."""

    harness_record = {"build": {"muraja_commit": "test"}, "muraja_commit": asp.MURAJA_COMMIT,
                      "binary_sha256": asp.HARNESS_SHA256}

    def __init__(self, files: Path, items: list[ClipItem],
                 flags: Callable[[str, int, int], Flag | None],
                 texts: dict[str, str] | None = None,
                 graded: Callable[[int, int], bool] = lambda i, p: True,
                 spec: asp.RoundSpec | None = None):
        self.spec = spec or filled_spec()
        files.mkdir(parents=True, exist_ok=True)
        (files / "look.input").write_text("x", encoding="utf-8")
        self.files, self.items, self.flags = files, items, flags
        self.texts, self.graded = texts or {}, graded
        self.read: list[str] = []
        self.grade_calls = 0

    def input_files(self, phase, weights):
        return [self.files / "look.input"]

    def decode_path(self, set_name, model):
        path = self.files / f"{set_name}.{model}.jsonl.gz"
        path.write_text(f"{set_name} {model}", encoding="utf-8")
        return path

    def tlog(self, role, models):
        assert role == "look", f"{role} was read"
        self.read.append(role)
        return ar.SetData(self.items, {
            m: {i.key: decode_line(i, CLEAN if m == "base" else self.texts.get(m, CLEAN))
                for i in self.items} for m in models})

    def grade(self, items, decodes, models):
        self.grade_calls += 1
        index = {i.key: n for n, i in enumerate(items)}
        return [word_grade(m, i, p, self.flags(m, index[i.key], p),
                           self.graded(index[i.key], p))
                for m in models for i in items for p in (0, 1)]

    def harness_identity(self):
        return dict(self.harness_record)

    def _refused(self, *args, **kwargs):
        raise AssertionError("a set other than tlog.look was read")

    tarteel = qdat = qud = qud_other = retasy = tadabur = bench = _refused
