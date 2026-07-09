"""Curated eval fixture sets for the P3.5 poison audit (#6) and eval harness (#7).

Defines the on-disk schema, canonical paths, and a validating loader for the two
hand-labelled sets ADR-0001 requires — built *before* fine-tuning, two-sided and
targeted (not aggregate PER):

* ``should_accept.jsonl`` — acceptable-imperfect amateur clips the fine-tuned
  model should ADMIT (measures recall *gain* vs the base model).
* ``should_reject.jsonl`` — genuinely-wrong substitutions the model must still
  REJECT (measures that discrimination is *retained*, not collapsed).

Each line is one :class:`EvalFixtureEntry` as JSON (see :data:`SCHEMA_FIELDS`).
This module ships the empty files, their schema, and the loader — **not** the
labelled data: the P3.5 audit (#6) fills the sets in, and #7's eval harness reads
them back through :func:`load_should_accept` / :func:`load_should_reject`. Both
slices depend on this module for the shared contract, so neither invents its own
layout.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .contrast_attribution import contrast_vocabulary

_FIXTURE_DIR = Path(__file__).parent / "eval_fixtures"
SHOULD_ACCEPT_PATH = _FIXTURE_DIR / "should_accept.jsonl"
SHOULD_REJECT_PATH = _FIXTURE_DIR / "should_reject.jsonl"

# The two verdicts, one per set. Stored explicitly on every entry so a mislabelled
# line (a reject in the accept set) is caught by the loader rather than silently
# skewing recall/discrimination metrics.
ACCEPT = "accept"
REJECT = "reject"
_VERDICTS = frozenset({ACCEPT, REJECT})


@dataclass(frozen=True)
class EvalFixtureEntry:
    """One human-labelled eval clip.

    ``clip_id`` / ``audio_ref`` identify the clip (``audio_ref`` is Tadabur's
    ``audio_filename``, matching the sampler worklist and filter manifest).
    ``surah_ayah`` is ``"surah:ayah"``. ``contrast`` is the bucket it exercises
    (a soft pair, ``shadda``, or ``marginal`` — see ``contrast_vocabulary``).
    ``verdict`` is ``"accept"`` or ``"reject"`` and must match the set it lives
    in. ``note`` is an optional free-text rationale from the human labeller.
    """

    clip_id: str
    audio_ref: str
    surah_ayah: str
    contrast: str
    verdict: str
    note: str = ""


# The JSON field names one fixture line carries, for documentation/validation.
SCHEMA_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(EvalFixtureEntry))


def _parse_entry(data: dict, expected_verdict: str, source: str) -> EvalFixtureEntry:
    unknown = set(data) - set(SCHEMA_FIELDS)
    if unknown:
        raise ValueError(f"{source}: unknown fixture field(s) {sorted(unknown)}")
    entry = EvalFixtureEntry(**data)
    if entry.verdict != expected_verdict:
        raise ValueError(
            f"{source}: entry {entry.clip_id!r} has verdict {entry.verdict!r}, "
            f"expected {expected_verdict!r} for this set"
        )
    if entry.contrast not in contrast_vocabulary():
        raise ValueError(
            f"{source}: entry {entry.clip_id!r} has unknown contrast {entry.contrast!r}"
        )
    return entry


def load_eval_fixtures(path: Path, expected_verdict: str) -> list[EvalFixtureEntry]:
    """Load and validate one fixture set, in file order.

    Every entry must carry exactly the schema fields, the set's ``expected_verdict``,
    and a known ``contrast``; a violation fails loudly so a corrupt or mislabelled
    fixture never silently distorts the eval. A missing file yields no entries (the
    set has not been populated yet). Blank and ``#``-prefixed lines are ignored.
    """
    if expected_verdict not in _VERDICTS:
        raise ValueError(f"expected_verdict must be one of {sorted(_VERDICTS)}")
    entries: list[EvalFixtureEntry] = []
    if not path.exists():
        return entries
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(_parse_entry(json.loads(line), expected_verdict, f"{path}:{lineno}"))
    return entries


def load_should_accept() -> list[EvalFixtureEntry]:
    """The should-accept set (acceptable-imperfect clips the model should admit)."""
    return load_eval_fixtures(SHOULD_ACCEPT_PATH, ACCEPT)


def load_should_reject() -> list[EvalFixtureEntry]:
    """The should-reject set (genuinely-wrong clips the model must still reject)."""
    return load_eval_fixtures(SHOULD_REJECT_PATH, REJECT)


def write_eval_fixtures(
    entries: list[EvalFixtureEntry], path: Path, expected_verdict: str
) -> None:
    """Atomically (over)write one fixture set, validating every entry first.

    Each entry is round-tripped through :func:`_parse_entry` so a wrong verdict or
    unknown contrast fails loudly *before* anything touches disk — the file is
    never left partially rewritten or holding an invalid line. Entries are written
    in the given order as one JSON object per line, then ``os.replace``-swapped in.
    The P3.5 audit UI (#6) uses this to persist labels through the same schema #7
    reads back.
    """
    for entry in entries:
        _parse_entry(asdict(entry), expected_verdict, "write_eval_fixtures")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
