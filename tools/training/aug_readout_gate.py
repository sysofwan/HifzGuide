"""The augmentation readout's phase transitions (#95): what may be decoded, merged and read, when.

The readout (:mod:`training.aug_readout`) has two sittings, each read once. Every step that can
see a held-out decode passes through this module, so the order is enforced in one place:

* **Models** (:func:`model_spec`): #94's two arms at their final step, live or EMA weights, and
  the truth baseline's ``h448`` and base teacher, whose decodes are committed and reused.
* **Sets** (:data:`SETS`): Tarteel v1's look half is the only set read before the choice; every
  other set is **gated** (:func:`allowed_models`).
* **Identity manifests** (:func:`require_manifest`): before any decode is read, every model whose
  decodes a sitting consumes must have a complete identity record (its checkpoint's SHA-256, its
  weights, step :data:`FINAL_STEP`, the truth baseline's frozen fingerprints) for every set it
  reads; a baseline's must be the one every reused corpus recorded (:func:`baseline_identities`).
* **The committed choice** (:func:`committed_choice`): ``choice.json`` and ``look.json`` tracked by
  git and unchanged from ``HEAD``, the choice bound to the look's SHA-256, to the SHA-256 of the
  look's inputs and to the candidate's validated identities. Nothing gated is decoded, merged or
  read without it.
* **A sitting's claim** (:func:`claim`): an exclusive-create marker binding the sitting to its
  choice, identities, input files' SHA-256 and the Muraja harness that grades it
  (:func:`harness_identity`), held under an exclusive lock while it reads. A
  second invocation while one runs fails; after an interruption only the same sitting (the same
  claim) may run again; another sitting is refused.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from tadabur.tarteel_split import CONFIRM, LOOK
from training.aug_readout_metrics import BASE, CANDIDATE, CONTROL, H448
from training.aug_readout_spec import (AUG95, AUG95_ROOT, ModelIdentity, RoundSpec,
                                       TLOG_CONFIRM, TLOG_LOOK, check_pins)
from training.aug_readout_spec import model_spec as pinned_model_spec
from training.external_decode import DEFAULT_SPECS, NO_AUGMENTATION, ModelSpec
from training.truth_baseline import (
    ARM_PROTOCOLS, CHECKS, DEFAULT_MODELS, EMA, LIVE, expected_fingerprint,
)

READOUT_DIR = AUG95_ROOT


@dataclass(frozen=True)
class Layout:
    """Where one readout keeps its manifests, choice, claims and results."""

    root: Path = READOUT_DIR

    @property
    def models(self) -> Path:
        return self.root / "models.json"

    @property
    def look(self) -> Path:
        return self.root / "look.json"

    @property
    def choice(self) -> Path:
        return self.root / "choice.json"

    @property
    def preclaim(self) -> Path:
        return self.root / "preclaim.json"

    @property
    def report(self) -> Path:
        return self.root / "report.json"

    @property
    def verdict(self) -> Path:
        return self.root / "verdict.json"

    def claim(self, phase: str) -> Path:
        return self.root / f"{phase}.claim.json"


# --- the models ---------------------------------------------------------------------------------

#: #94's two runs on the GPU box (its run note, ``docs/augmentation-experiment.md``); each saves
#: one ``checkpoint.pt``, overwritten at every save, so the file holds the last step saved. The
#: candidate is ``aug94_candidate_v2``, re-run from its own code directory under the silent-segment
#: fix (PR #166) after the first candidate crashed at step ~1,913 and saved nothing.
ARMS = {
    CANDIDATE: "/root/scratch/issue-94/repo-fix/tools/runs/aug94_candidate_v2/checkpoint.pt",
    CONTROL: "/root/scratch/issue-94/repo/tools/runs/aug94_control/checkpoint.pt",
}
#: The augmentation recipe each arm's checkpoint config must record: the candidate's
#: ``aug94-mixture-v2`` (a silent noise segment re-drawn), the control's none.
RECIPES = {CANDIDATE: "aug94-mixture-v2", CONTROL: NO_AUGMENTATION}
#: #94's checkpoint rule, fixed before training: the final step.
FINAL_STEP = 10_000
#: The selector vocabulary is the cache producer's (:data:`training.truth_baseline.LIVE`).
WEIGHTS = (LIVE, EMA)
BASELINES = (H448, BASE)


def arm_model(arm: str, weights: str, spec: RoundSpec | None = None) -> str:
    """One arm's decode name at ``weights``: ``<arm>_<weights>`` for #94's checkpoints, and the
    round's prefixed form (``r2_<arm>_<weights>``) for a round whose arms it pins itself."""
    return f"{(spec or AUG95).arm_prefix}{arm}_{weights}"


def look_models(spec: RoundSpec | None = None) -> tuple[str, ...]:
    """The models the look set is decoded and read for: each of the round's look roles at both
    weight selectors. #95 and the consonant round decode the candidate only; round 2 decodes
    the candidate and the control, so each chooses its own weights."""
    specification = spec or AUG95
    return tuple(arm_model(role, weights, specification)
                 for role in specification.look_roles for weights in WEIGHTS)


#: The models the look half is decoded and read for in #95: the candidate's two weights. The
#: control is never decoded on the look half.
LOOK_MODELS = look_models(AUG95)


def model_spec(name: str, spec: RoundSpec | None = None) -> ModelSpec:
    """The decode spec of a readout model: a round's pinned model, an arm's weights at
    :data:`FINAL_STEP`, or a truth-baseline model. A round that names its own arms pins all of
    them, so an arm the spec does not pin is refused by name, never resolved to #94's
    checkpoints (#227)."""
    specification = spec or AUG95
    if name in DEFAULT_SPECS:
        return DEFAULT_SPECS[name]
    if (pinned := pinned_model_spec(specification, name)) is not None:
        return pinned
    if specification.arm_prefix:
        raise ValueError(f"{name!r} is not a {specification.name} model: the round's arms are "
                         f"pinned in its spec")
    arm, _, weights = name.rpartition("_")
    if arm not in ARMS or weights not in WEIGHTS:
        raise ValueError(f"{name!r} is no readout model")
    return ModelSpec(ARMS[arm], use_ema=weights == EMA, step=FINAL_STEP, recipe=RECIPES[arm])


# --- the sets -----------------------------------------------------------------------------------

TARTEEL_LOOK, TARTEEL_CONFIRM = f"tarteel_v1.{LOOK}", f"tarteel_v1.{CONFIRM}"
QDAT_SET, RETASY_SET, QUD_SET = "qdat", "retasy", "qud_half"
QUD_OTHER_SET, TADABUR_SET = "qud_other", "tadabur_dev_unseen"
STREAM = "stream_b0"
#: The argmax frames' blank count beside a ``stream_b0`` decode: ``[blank, frames]``.
BLANKS = "stream_b0_blanks"
#: What a decode line of an external corpus's clip records of its audio (#156's decoder).
CLIP_PROVENANCE = ("audio_filename", "audio_sha256", "start_sample", "end_sample")
#: What a tlog decode line records of the row it was decoded from, each matched against the
#: committed base decode of the same item: the S7 manifest's own row key and the clip.
TLOG_PROVENANCE = ("split", "shard_file", "row", "audio_sha256", "start_sample", "end_sample")


@dataclass(frozen=True)
class ReadoutSet:
    """One set the readout decodes: the protocols each item is decoded under, the provenance
    every decode line must carry and match against the committed base decode of its item,
    whether it is read only after the choice (``gated``), and whether its per-item decodes may
    be committed."""

    name: str
    protocols: tuple[str, ...]
    provenance: tuple[str, ...]
    gated: bool = True
    open: bool = True


SETS = {s.name: s for s in (
    ReadoutSet(TARTEEL_LOOK, (CHECKS,), CLIP_PROVENANCE, gated=False),
    ReadoutSet(TARTEEL_CONFIRM, (CHECKS,), CLIP_PROVENANCE),
    ReadoutSet(QDAT_SET, (CHECKS, STREAM), CLIP_PROVENANCE),
    ReadoutSet(RETASY_SET, (CHECKS,), CLIP_PROVENANCE, open=False),
    ReadoutSet(QUD_SET, (CHECKS,), ("audio_filename", "mp3_sha256", "segment_index",
                                    "start_sample", "end_sample")),
    ReadoutSet(QUD_OTHER_SET, (CHECKS,), ("audio_filename", "mp3_sha256", "segment_index",
                                          "start_sample", "end_sample")),
    ReadoutSet(TADABUR_SET, (STREAM,), ("reciter_id", "clean_sha256", "start_sample",
                                        "end_sample")),
    # Round 2's own sets (#227): the look tranche is read before the choice, the pooled
    # confirm.1+2 tranches only after it, on the same committed decodes.
    ReadoutSet(TLOG_LOOK, (CHECKS,), TLOG_PROVENANCE, gated=False),
    ReadoutSet(TLOG_CONFIRM, (CHECKS,), TLOG_PROVENANCE),
)}
GATED = tuple(name for name, s in SETS.items() if s.gated)


def decodes_file(set_name: str, model: str) -> Path:
    """#95's decode file of ``set_name`` for ``model`` (the default round's)."""
    return round_decodes_file(AUG95, set_name, model)


def round_decodes_file(spec: RoundSpec, set_name: str, model: str) -> Path:
    """A round's decode file of ``set_name`` for ``model``: in the other round's directory when
    the round reuses its committed decodes (C95's are #95's), else in the round's own root."""
    source = spec.decode_sources.get((set_name, model))
    root, name = (source.root, source.model) if source is not None else (spec.root, model)
    directory = root / "decodes" if SETS[set_name].open else root / "restricted"
    return directory / f"{set_name}.{name}.jsonl.gz"


def chosen_weights(spec: RoundSpec, choice: Mapping) -> dict[str, str]:
    """Each of the round's arms and the weights it was chosen at. A choice's ``weights`` is one
    selector for every arm (#95, the consonant round) or each arm's own (round 2, whose arms
    choose independently on the look set) — the one place either form is read, and validated:
    a partial per-arm mapping names no model."""
    weights = choice["weights"]
    arms = [role for role in spec.students if role in ARMS]
    if not _valid_weights(spec, weights):
        raise SystemExit(f"the choice's weights {weights!r} are not {list(WEIGHTS)}"
                         f"{f', one per arm {arms}' if isinstance(weights, Mapping) else ''}")
    return {role: weights[role] if isinstance(weights, Mapping) else weights for role in arms}


def _valid_weights(spec: RoundSpec, weights) -> bool:
    """A choice's ``weights``: one selector for every arm, or each of the round's arms named
    with its own (round 2)."""
    arms = [role for role in spec.students if role in ARMS]
    if isinstance(weights, Mapping):
        return set(weights) == set(arms) and all(w in WEIGHTS for w in weights.values())
    return weights in WEIGHTS


def allowed_models(set_name: str, choice: Mapping | None,
                   spec: RoundSpec | None = None) -> tuple[str, ...]:
    """The models a set may be decoded (and read) for: a look set, its round's look roles at
    both weights; any gated set only with a committed choice (``choice``, from
    :func:`committed_choice`), for the round's students at the chosen weights (and, on the
    tlog confirm tranches, the teacher, whose checks the bars read beside the candidate's; on
    Tadabur, the two baselines, for the blank share). A round with no control (``consonant1``)
    reads no control decodes."""
    specification = spec or AUG95
    if not SETS[set_name].gated:
        if set_name != specification.look_set:
            raise SystemExit(f"{set_name} is read only after the look half, with a committed "
                             f"choice: {specification.look_set} is this round's look set")
        return look_models(specification)
    if choice is None:
        raise SystemExit(f"{set_name} is read only after the look half, with a committed choice")
    chosen = chosen_weights(specification, choice)
    students = tuple(arm_model(role, chosen[role], specification) if role in chosen else role
                     for role in specification.students)
    baselines = ((BASE,) if set_name == TLOG_CONFIRM
                 else BASELINES if set_name == TADABUR_SET else ())
    return tuple(dict.fromkeys((*students, *baselines)))


def needs(phase: str, weights: str | None,
          spec: RoundSpec | None = None) -> dict[str, tuple[str, ...]]:
    """Every set a sitting reads readout decodes of, and the models it reads there."""
    specification = spec or AUG95
    declared = specification.sets
    if phase == LOOK:
        return {specification.look_set: look_models(specification)}
    return {name: allowed_models(name, {"weights": weights}, specification) for name in declared
            if name in SETS and SETS[name].gated}


# --- identity -----------------------------------------------------------------------------------

_SHA256 = re.compile(r"[0-9a-f]{64}")


def expected_fingerprints(ref: str) -> dict:
    return {arm: expected_fingerprint(ref, mode) for arm, mode in ARM_PROTOCOLS.items()}


def baseline_identities() -> dict[str, dict]:
    """The one identity of each baseline across every committed corpus the readout reuses
    (Tarteel v1, qdat and RetaSy, #156; QUD, #144; Tadabur dev, #159), each fingerprint checked
    against the truth baseline's protocol."""
    from training.nonpro_eval import REPORT_PATH as NONPRO_REPORT
    from training.nonpro_gap import REPORT_PATH as GAP_REPORT
    from training.qud_pilot import REPORT_PATH as QUD_REPORT

    found: dict[str, dict] = {}
    for path in (NONPRO_REPORT, QUD_REPORT, GAP_REPORT):
        for name, record in json.loads(path.read_text(encoding="utf-8"))["models"].items():
            model = name.rsplit("/", 1)[1]
            for arm, fingerprint in record["fingerprints"].items():
                if fingerprint != expected_fingerprint(DEFAULT_MODELS[model], ARM_PROTOCOLS[arm]):
                    raise ValueError(f"{path.name} {name}: {arm} is not the frozen protocol")
            if found.setdefault(model, record["identity"]) != record["identity"]:
                raise ValueError(f"{path.name} {name}: another {model} than the other corpora's")
    return found


def identity_record(identity: ModelIdentity) -> dict:
    """A round's pinned model as the identity a decode run records and the readout checks."""
    return {"model_ref": identity.ref, "weights": identity.weights, "step": identity.step,
            "recipe": identity.recipe, "checkpoint_sha256": identity.checkpoint_sha256}


def validate_identity(set_name: str, model: str, record: Mapping,
                      spec: RoundSpec | None = None) -> dict:
    """One decode run's identity record, complete and the model the readout names: the frozen
    fingerprints; a baseline's committed identity; an arm's checkpoint path, its SHA-256, its
    weights, step :data:`FINAL_STEP` and its arm's recipe (:data:`RECIPES`), and nothing else.
    Returns the identity."""
    specification = spec or AUG95
    decoded = model_spec(model, spec=specification)
    pinned = specification.models.get(model)
    name = f"{set_name}/{model}"
    if record.get("model") != model or record.get("fingerprints") != expected_fingerprints(
            decoded.ref):
        raise ValueError(f"{name}: not {model} under the frozen protocol's fingerprints")
    identity = record.get("identity")
    if pinned is not None:  # a round's pinned baseline, e.g. C95 for the consonant round
        expected = identity_record(pinned)
        if identity != expected:
            raise ValueError(f"{name}: {identity} is not {model}'s pinned checkpoint")
        return dict(identity)
    if model in BASELINES:
        if identity != baseline_identities()[model]:
            raise ValueError(f"{name}: not the committed {model}")
        return dict(identity)
    expected = {"model_ref": decoded.ref, "weights": EMA if decoded.use_ema else LIVE,
                "step": FINAL_STEP, "recipe": decoded.recipe}
    if not isinstance(identity, Mapping) or set(identity) != {*expected, "checkpoint_sha256"} \
            or any(identity[k] != v for k, v in expected.items()) \
            or not _SHA256.fullmatch(str(identity["checkpoint_sha256"])):
        raise ValueError(f"{name}: {identity} is not {model} at step {FINAL_STEP} under "
                         f"{decoded.recipe!r} with a checkpoint SHA-256")
    return dict(identity)


def require_manifest(records: Mapping[str, Mapping], required: Mapping[str, Iterable[str]],
                     choice: Mapping | None, spec: RoundSpec | None = None) -> dict[str, dict]:
    """Every ``(set, model)`` a sitting reads has a valid identity record, each model one
    identity across its sets, each gated set only the chosen weights, and the candidate the
    identity the committed choice holds. Returns the identity per model, including the choice's
    for each model it selected even when no corpus file of this sitting arrived."""
    per_model: dict[str, dict] = {}
    for set_name, models in required.items():
        if SETS[set_name].gated:
            allowed = allowed_models(set_name, choice, spec)
        for model in models:
            if SETS[set_name].gated and model not in allowed:
                raise ValueError(f"{set_name}/{model}: weights the choice did not pick")
            record = records.get(f"{set_name}/{model}")
            if record is None:
                raise ValueError(f"no identity record for {set_name}/{model}")
            identity = validate_identity(set_name, model, record, spec=spec)
            if per_model.setdefault(model, identity) != identity:
                raise ValueError(f"{model} is not one checkpoint across sets")
    if choice is not None:
        selected = set(choice.get("models", {}).values())
        for model, identity in choice["identities"].items():
            if per_model.get(model, identity) != identity:
                raise ValueError(f"{model} is not the checkpoint the committed choice read")
            # The choice validated its selected models on the look half. Carry that identity
            # even when none of this sitting's corpus files arrived, so a reader that binds
            # the truth-baseline caches (G6) is not left without one (#175 review R2.3).
            if model in selected:
                per_model.setdefault(model, identity)
    return per_model


def read_manifest(layout: Layout) -> dict:
    return json.loads(layout.models.read_text(encoding="utf-8")) if layout.models.exists() else {}


# --- the committed choice ----------------------------------------------------------------------


def _git(directory: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True)


def is_committed(path: Path) -> bool:
    """Whether ``path`` is tracked by git and unchanged from ``HEAD``."""
    return path.exists() and \
        _git(path.parent, "ls-files", "--error-unmatch", path.name).returncode == 0 and \
        _git(path.parent, "diff", "--quiet", "HEAD", "--", path.name).returncode == 0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def choice_models(spec: RoundSpec, weights) -> dict[str, str]:
    """The arms a round's choice binds: each of its students that is an arm, at the weights
    ``weights`` names — one selector for every arm, or each arm's own (round 2). A round
    without a control (``consonant1``) names its candidate only."""
    chosen = chosen_weights(spec, {"weights": weights})
    return {role: arm_model(role, chosen[role], spec) for role in spec.students if role in ARMS}


def make_choice(weights: str | Mapping[str, str], look_text: str, sitting: Mapping,
                spec: RoundSpec = AUG95) -> dict:
    """``choice.json``: the round's issue, the weights, bound to the look's SHA-256, to the look
    sitting (its input files' SHA-256 and the candidate's identities)."""
    return {"issue": spec.issue, "weights": weights,
            "models": choice_models(spec, weights),
            "identities": dict(sitting["identities"]),
            "look_sha256": sha256_bytes(look_text.encode("utf-8")),
            "look_inputs_sha256": sha256_bytes(canonical(sitting["inputs"]).encode("utf-8"))}


@dataclass(frozen=True)
class CommittedChoice:
    """A validated ``choice.json``: its content, its SHA-256 and the commit that holds it."""

    choice: dict
    sha256: str
    commit: str


def committed_choice(layout: Layout = Layout(),
                     spec: RoundSpec = AUG95) -> CommittedChoice:
    """The choice, refused unless it, ``look.json`` and the look's claim are tracked by git and
    unchanged from ``HEAD``, and bound to them by content as well as by hash: its weights are the
    ones the look chose, its candidate identities (checkpoint SHA-256, weights, step, recipe) the
    ones the look and its claim recorded and the round's pins, and its inputs and harness the
    claim's."""
    for path in (layout.choice, layout.look, layout.claim(LOOK)):
        if not is_committed(path):
            raise SystemExit(f"{path.name} is not committed (tracked and unchanged from HEAD)")
    raw = layout.choice.read_bytes()
    choice = json.loads(raw)
    look_raw = layout.look.read_bytes()
    look = json.loads(look_raw)
    look_claim = json.loads(layout.claim(LOOK).read_text(encoding="utf-8"))
    if choice.get("look_sha256") != sha256_bytes(look_raw):
        raise SystemExit(f"{layout.choice.name} is not bound to the committed {layout.look.name}")
    # The semantic bindings: the weights the look chose, and the checkpoints the look read.
    weights = choice.get("weights")
    if not _valid_weights(spec, weights) or weights != look.get("choice"):
        raise SystemExit(f"{layout.choice.name} names other weights than {layout.look.name} chose")
    if choice.get("models") != choice_models(spec, weights):
        raise SystemExit(f"{layout.choice.name} names other models than its weights")
    identities = choice.get("identities") or {}
    if set(identities) != set(look_models(spec)):
        raise SystemExit(f"{layout.choice.name} holds no identity for every look arm's weights")
    for model, identity in identities.items():
        validate_identity(spec.look_set, model, {"model": model, "identity": identity,
                                                 "fingerprints": expected_fingerprints(
                                                     model_spec(model, spec=spec).ref)},
                          spec=spec)
    if not (identities == look.get("identities") == look_claim.get("identities")):
        raise SystemExit(f"{layout.choice.name} holds other checkpoints than the look read")
    if look_claim.get("phase") != LOOK or choice.get("look_inputs_sha256") != sha256_bytes(
            canonical(look_claim["inputs"]).encode("utf-8")) \
            or look.get("input_sha256") != look_claim["inputs"] \
            or look.get("harness") != look_claim.get("harness"):
        raise SystemExit(f"{layout.choice.name} is not bound to the look sitting's inputs")
    commit = _git(layout.choice.parent, "log", "-1", "--format=%H", "--",
                  layout.choice.name).stdout.strip()
    return CommittedChoice(choice, sha256_bytes(raw), commit)


def require_preregistration_text(spec: RoundSpec) -> str:
    """The round's pre-registration's SHA-256, refused unless it is committed in HEAD and
    equal to the round's pin: its text is never edited after it, and a claim opens a tranche on
    the text the round froze even where no harness is at hand."""
    if not is_committed(spec.prereg_path):
        raise SystemExit(f"{spec.prereg_path.name} is not committed: nothing grades a tlog "
                         "decode before the pre-registration's commit")
    prereg = sha256_file(spec.prereg_path)
    pins = spec.pins
    if pins is not None and prereg != pins.prereg_sha256:
        raise SystemExit(f"the pre-registration pin differs: {prereg!r} is not the pinned "
                         f"{pins.prereg_sha256!r}")
    return prereg


def require_preregistration(spec: RoundSpec, harness: Mapping) -> str:
    """The round's pre-registration's SHA-256, refused unless it is committed in HEAD and the
    round's pins hold: no tlog decode is graded before the pre-registration's commit, and its
    text is never edited after it (round 2's data discipline 1, #227)."""
    prereg = require_preregistration_text(spec)
    check_pins(spec, harness, prereg)
    return prereg


# --- a sitting's claim --------------------------------------------------------------------------


def harness_identity(harness) -> dict:
    """What grades a sitting's words: the Muraja harness's build record, and the SHA-256 of its
    binary and of the Quran and layout databases it reads (:class:`training.muraja_policy.Harness`)."""
    return {"build": harness.build, "muraja_commit": harness.muraja_commit,
            "compiler": harness.compiler,
            "binary_sha256": sha256_file(harness.binary),
            "quran_db_sha256": sha256_file(harness.quran_db),
            "layout_db_sha256": sha256_file(harness.layout_db)}


TOOLS_DIR = Path(__file__).parent.parent


def input_name(path: Path) -> str:
    """A sitting's name for one input file: relative to ``tools/`` when inside it."""
    path = path.resolve()
    return str(path.relative_to(TOOLS_DIR.resolve())) \
        if path.is_relative_to(TOOLS_DIR.resolve()) else str(path)


def digests(paths: Iterable[Path]) -> dict[str, str]:
    """Each input file's SHA-256, by its name (:func:`input_name`)."""
    return {input_name(p): sha256_file(p) for p in sorted(set(paths))}


@contextmanager
def claim(path: Path, sitting: Mapping) -> Iterator[None]:
    """Hold the sitting ``sitting`` describes: create its claim exclusively (or, after an
    interruption, find the same claim) and keep an exclusive lock on it while the sitting reads.
    Another sitting's claim, or a sitting already running, is refused."""
    text = canonical(sitting).encode("utf-8")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        _lock(fd, path)
        os.write(fd, text)
        os.fsync(fd)
    except FileExistsError:
        fd = os.open(path, os.O_RDONLY)
        _lock(fd, path)
        if path.read_bytes() != text:
            os.close(fd)
            raise SystemExit(f"{path.name} claims another sitting: the readout reads once")
    try:
        yield
    finally:
        os.close(fd)


def _lock(fd: int, path: Path) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise SystemExit(f"{path.name}: the sitting is being read now") from None
