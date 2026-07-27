"""Frozen fixtures for the P7.H conditional-reference integration eval (#35, the product gate).

ADR-0004 (``docs/adr/0004-waqf-head-and-joint-whole-clip-fine-tune.md``) makes the *product*
goal — not frame-F1, not event-F1 — a `.strict` scoring gate: does consuming the phoneme
decode **and** the predicted waqf events together (snap → per-run realized-reference selection →
strict scoring) actually regain the wasl-sensitive **i'raab** (final desinence) and **cross-word
idgham** discrimination that today's *ignore-end-word-tashkeel* hack throws away? A boundary the
reciter **stops** at (waqf) realizes its terminal word differently from one they **continue**
through (wasl): the pre-pause word takes its pausal form (``quran_phonetizer``'s CleanEnd — a
tanwin becomes a madd, the final haraka drops) and loses the cross-word gemination/idgham it would
carry in continuation. So the *same* phoneme realization is **correct** after a genuine stop but a
dropped-i'raab / missed-idgham **error** mid-continuation — only the waqf head tells them apart.

Each line is one :class:`IntegrationCase` as JSON. A case fixes one **boundary** (a word edge in
one ayah) and one **reciter realization** of it, and carries the two *pre-resolved, normalized*
realized references so the eval is a pure function of the scorer — no ``quran_transcript`` or model
at eval/test time (the phoneme forms are frozen here, exactly as :mod:`tadabur.waqf_freeze` freezes
the F0 event ground truth). The generator that resolves the forms from the phonetizer is
:mod:`tadabur.waqf_integration_gen`; it re-derives and re-validates every field, so this file is a
deterministic, auditable artifact, never hand-edited.

Per case:

* ``waqf_reference`` — the realized reference if the reciter **paused** here (terminal CleanEnd:
  pausal madd present, no cross-word idgham). Normalized (the scorer's cache form).
* ``wasl_reference`` — the realized reference if the reciter **continued** (terminal in
  continuation: pausal madd absent, the tanwin's noon/ghunna carried onto the next word).
* ``true_class`` — what the reciter actually did (``waqf`` / ``wasl``), the adjudicated ground
  truth the waqf head must recover.
* ``recitation`` — how they realized it: ``correct`` (matches the realized form of ``true_class``),
  ``dropped`` (a true **wasl** rendered with the pausal ending — the i'raab/idgham **error** a
  false waqf would forgive), or ``interior`` (a genuine mistake *inside* the word, not at its edge —
  a control the end-word hack must still reject).
* ``decode`` — the reciter's raw phoneme realization (the model-decode stand-in); the scorer
  normalizes it, so it is stored un-normalized (the scorer's normalization is not idempotent).
* ``expected_strict`` — the correct `.strict` verdict given the true realization: ``accept`` for
  ``correct``, ``reject`` for ``dropped`` / ``interior``.
* ``phenomenon`` — ``iraab`` or ``cross_word_idgham``, the ADR discrimination the boundary exercises.

Keyed by ``case_id`` (``"surah:ayah#word_index/recitation"``), stable and unique per line.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# The two realized-reference forms a boundary can take, and the classes a waqf head predicts.
WAQF = "waqf"
WASL = "wasl"
BOUNDARY_CLASSES: tuple[str, ...] = (WAQF, WASL)

# How the reciter realized the boundary word. ``correct`` matches ``true_class``'s realized form;
# ``dropped`` is a true wasl said with the pausal ending (the i'raab/idgham error); ``interior`` is
# a genuine mistake inside the word (a control that is not an end-word tashkeel difference).
CORRECT = "correct"
DROPPED = "dropped"
INTERIOR = "interior"
RECITATIONS: tuple[str, ...] = (CORRECT, DROPPED, INTERIOR)

# The correct `.strict` verdicts, one per case (the ground truth the scenarios are scored against).
ACCEPT = "accept"
REJECT = "reject"
STRICT_VERDICTS: tuple[str, ...] = (ACCEPT, REJECT)

# The ADR-named discriminations a boundary exercises (final desinence vs cross-word idgham/ghunna).
IRAAB = "iraab"
CROSS_WORD_IDGHAM = "cross_word_idgham"
PHENOMENA: tuple[str, ...] = (IRAAB, CROSS_WORD_IDGHAM)

_BOUNDARY = frozenset(BOUNDARY_CLASSES)
_RECITATIONS = frozenset(RECITATIONS)
_VERDICTS = frozenset(STRICT_VERDICTS)
_PHENOMENA = frozenset(PHENOMENA)

_FIXTURE_DIR = Path(__file__).parent / "waqf_integration_fixtures"
WAQF_INTEGRATION_PATH = _FIXTURE_DIR / "waqf_integration_cases.jsonl"


@dataclass(frozen=True)
class IntegrationCase:
    """One adjudicated boundary realization for the conditional-reference product gate.

    ``surah_ayah`` is ``"surah:ayah"`` and ``boundary_word_index`` the Uthmani word the boundary
    falls after; ``word`` / ``next_word`` are those two Uthmani words (audit only). ``phenomenon``
    is the ADR discrimination exercised. ``waqf_reference`` / ``wasl_reference`` are the two
    *normalized* realized references (paused vs continued). ``true_class`` is the adjudicated
    ground-truth boundary class, ``recitation`` how the reciter realized it, ``decode`` their
    raw phoneme realization (the scorer normalizes it), and ``expected_strict`` the correct
    `.strict` verdict. ``note`` is optional free text.
    """

    case_id: str
    surah_ayah: str
    boundary_word_index: int
    word: str
    next_word: str
    phenomenon: str
    waqf_reference: str
    wasl_reference: str
    true_class: str
    recitation: str
    decode: str
    expected_strict: str
    note: str = ""


# The JSON field names one fixture line carries, for documentation/validation.
SCHEMA_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(IntegrationCase))


def _parse_entry(data: dict, source: str) -> IntegrationCase:
    unknown = set(data) - set(SCHEMA_FIELDS)
    if unknown:
        raise ValueError(f"{source}: unknown fixture field(s) {sorted(unknown)}")
    entry = IntegrationCase(**data)
    for name, value, allowed in (
        ("true_class", entry.true_class, _BOUNDARY),
        ("recitation", entry.recitation, _RECITATIONS),
        ("expected_strict", entry.expected_strict, _VERDICTS),
        ("phenomenon", entry.phenomenon, _PHENOMENA),
    ):
        if value not in allowed:
            raise ValueError(
                f"{source}: case {entry.case_id!r} has {name} {value!r}, "
                f"expected one of {sorted(allowed)}"
            )
    # A ``correct`` realization must be an accept; ``dropped`` / ``interior`` a reject — the
    # expected verdict is a property of the realization, not a free label, so a mismatch is a
    # corrupt fixture, not a graded outcome.
    expected = ACCEPT if entry.recitation == CORRECT else REJECT
    if entry.expected_strict != expected:
        raise ValueError(
            f"{source}: case {entry.case_id!r} recitation {entry.recitation!r} implies "
            f"expected_strict {expected!r}, got {entry.expected_strict!r}"
        )
    # A ``dropped`` error is only defined at a true wasl (the pausal ending is only wrong when the
    # reciter did *not* pause); a true waqf realized pausally is ``correct``.
    if entry.recitation == DROPPED and entry.true_class != WASL:
        raise ValueError(
            f"{source}: case {entry.case_id!r} recitation 'dropped' requires true_class 'wasl', "
            f"got {entry.true_class!r}"
        )
    return entry


def load_integration_cases(path: Path = WAQF_INTEGRATION_PATH) -> list[IntegrationCase]:
    """Load and validate the integration cases, in file order.

    Every line must carry exactly the schema fields, known enum values, and the
    recitation↔verdict / dropped↔wasl invariants; a violation fails loudly so a corrupt or
    mislabelled fixture never silently distorts the product gate. A missing file yields no cases
    (the set has not been generated yet). Blank and ``#``-prefixed lines are ignored.
    """
    entries: list[IntegrationCase] = []
    if not path.exists():
        return entries
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(_parse_entry(json.loads(line), f"{path}:{lineno}"))
    return entries


def write_integration_cases(
    entries: list[IntegrationCase], path: Path = WAQF_INTEGRATION_PATH
) -> None:
    """Atomically (over)write the integration cases, validating every entry first.

    Each entry round-trips through :func:`_parse_entry` so an invalid enum or a broken
    recitation↔verdict invariant fails loudly *before* anything touches disk — the file is never
    left partially rewritten. Entries are written in the given order as one JSON object per line
    (sorted keys, UTF-8), then ``os.replace``-swapped in.
    """
    for entry in entries:
        _parse_entry(asdict(entry), "write_integration_cases")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
