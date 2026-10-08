"""Canonical waqf event-fixture set for the P7.F0 HITL gate (helper #27).

ADR-0004 makes the product goal *event-level*: after the scorer's boundary snap and
300/700 ms post-processing, does each candidate boundary come out as the right kind
of event? A silence VAD only detects *silence*, so the eval must be graded against
**human** verdicts, not the teacher — measuring false-waqf at true-wasl boundaries,
false-wasl at genuine stops, and a mid-word-closure rejection set (qalqala on ق/ط,
the hamza in شَيء). This module owns the on-disk schema those human verdicts are
persisted to — the waqf analogue of :mod:`tadabur.eval_fixtures`. The adjudication
UI and event-level eval that wrote and read it were retired with the waqf head
(ADR-0011); the committed verdicts remain human truth, and this loader is how they are
read.

Each line is one :class:`WaqfEventEntry` as JSON. A boundary is adjudicated into
exactly one of three **classes** — ``waqf`` (a true stop), ``wasl`` (continuation,
no pause), or ``mid_word_closure`` (a stop-consonant/hamza silence that is *not* a
waqf) — carried as both the detector's ``predicted`` class and the human
``verdict``, so the confusion between them is the metric. Verdicts are keyed by
``(clip_id, boundary_index)`` — one candidate boundary, one fixture line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

# The three mutually-exclusive kinds a candidate boundary can be. Used for both the
# detector's ``predicted`` class and the human ``verdict`` (their disagreement is the
# false-waqf / false-wasl / mid-word-closure signal ADR-0004's eval measures).
WAQF = "waqf"
WASL = "wasl"
MID_WORD_CLOSURE = "mid_word_closure"
WAQF_EVENT_CLASSES: tuple[str, ...] = (WAQF, WASL, MID_WORD_CLOSURE)
_CLASSES = frozenset(WAQF_EVENT_CLASSES)

_FIXTURE_DIR = Path(__file__).parent / "waqf_event_fixtures"
WAQF_EVENTS_PATH = _FIXTURE_DIR / "waqf_events.jsonl"


@dataclass(frozen=True)
class WaqfEventEntry:
    """One human-adjudicated candidate waqf boundary.

    ``clip_id`` / ``audio_ref`` identify the **whole** clip the boundary lives in
    (``audio_ref`` is Tadabur's ``audio_filename``). ``surah_ayah`` is
    ``"surah:ayah"``. ``boundary_index`` orders the candidate boundaries within the
    clip and, with ``clip_id``, is the fixture's stable key. ``word_index`` is the
    Uthmani word position the boundary falls after, and ``(start_s, end_s)`` its
    time span within the clip (the candidate silence, or the word edge for a wasl
    candidate) so the reviewer can seek to it. ``predicted`` is the detector's class
    and ``verdict`` the human's — each one of :data:`WAQF_EVENT_CLASSES`. ``note``
    is an optional free-text rationale.
    """

    clip_id: str
    audio_ref: str
    surah_ayah: str
    boundary_index: int
    word_index: int
    start_s: float
    end_s: float
    predicted: str
    verdict: str
    note: str = ""


# The JSON field names one fixture line carries, for documentation/validation.
SCHEMA_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(WaqfEventEntry))


def _parse_entry(data: dict, source: str) -> WaqfEventEntry:
    unknown = set(data) - set(SCHEMA_FIELDS)
    if unknown:
        raise ValueError(f"{source}: unknown fixture field(s) {sorted(unknown)}")
    entry = WaqfEventEntry(**data)
    for field_name, value in (("predicted", entry.predicted), ("verdict", entry.verdict)):
        if value not in _CLASSES:
            raise ValueError(
                f"{source}: entry {entry.clip_id!r}#{entry.boundary_index} has "
                f"{field_name} {value!r}, expected one of {list(WAQF_EVENT_CLASSES)}"
            )
    return entry


def load_waqf_events(path: Path = WAQF_EVENTS_PATH) -> list[WaqfEventEntry]:
    """Load and validate the waqf event fixtures, in file order.

    Every entry must carry exactly the schema fields and a known ``predicted`` /
    ``verdict`` class; a violation fails loudly so a corrupt or mislabelled fixture
    never silently distorts F0's event-level eval. A missing file yields no entries
    (the set has not been populated yet). Blank and ``#``-prefixed lines are ignored.
    """
    entries: list[WaqfEventEntry] = []
    if not path.exists():
        return entries
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(_parse_entry(json.loads(line), f"{path}:{lineno}"))
    return entries
