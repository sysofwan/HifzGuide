"""The one exclusion every evaluation path applies: items that are no site of their stratum.

An item leaves every evaluation denominator for a **reason**, each decided by its guard at
the current verdicts:

* ``waqf_realized`` (:mod:`tadabur.waqf_guard`, #134): its prescribed mark only exists in wasl
  and the reciter paused there. Its verdict stays a training target (what was heard);
* ``onset_trimmed`` (:mod:`tadabur.onset_guard`, #138): its clip's opening sound is cut, so
  the excerpt, not the model, decides it. It is no training target either: an export of
  verified targets (none exists yet) must leave this reason's items out.

The population builders (:func:`tadabur.eval_harness.build`,
:func:`training.ship_a_sizing.run`) and the scorers (``truth_scorer.score(excluded_items=…)``,
``eval_report.fixture_report(excluded_items=…)``) take one :class:`Exclusions` and remove its
items after the design weights are computed over every site; every report lists each item
taken out under its reason (the first of :data:`REASONS` that applies, so it is counted
once).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .listening_session import VERDICTS_PATH
from .onset_guard import GUARD_PATH as ONSET_GUARD_PATH
from .onset_guard import ONSET_TRIMMED, trimmed_ids
from .waqf_guard import GUARD_PATH as WAQF_GUARD_PATH
from .waqf_guard import WAQF_REALIZED, realized_ids

#: The reasons, in the order every report lists them.
REASONS = (WAQF_REALIZED, ONSET_TRIMMED)
#: The sidecar each reason's guard decides from.
GUARD_PATHS = {WAQF_REALIZED: WAQF_GUARD_PATH, ONSET_TRIMMED: ONSET_GUARD_PATH}


@dataclass(frozen=True)
class Exclusions:
    """The item ids each reason takes out (an id may have several reasons)."""

    by_reason: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        if tuple(self.by_reason) != REASONS:
            raise ValueError(f"an exclusion names every reason, in order: {REASONS}")

    @classmethod
    def of(cls, **ids: Collection[str]) -> "Exclusions":
        """Exclusions from the ids given per reason (the others none)."""
        unknown = set(ids) - set(REASONS)
        if unknown:
            raise ValueError(f"unknown exclusion reasons: {sorted(unknown)}")
        return cls({reason: frozenset(ids.get(reason, ())) for reason in REASONS})

    def reason(self, item_id: str) -> str | None:
        """The first reason that takes the item out, or ``None``."""
        return next((r for r, ids in self.by_reason.items() if item_id in ids), None)

    def __contains__(self, item_id: str) -> bool:
        return self.reason(item_id) is not None

    def listed(self, item_ids: Iterable[str]) -> dict[str, list[str]]:
        """Per reason, the sorted ids among ``item_ids`` taken out under it (the report's
        lists; each id under its first reason)."""
        lists: dict[str, list[str]] = {reason: [] for reason in REASONS}
        for item_id in sorted(set(item_ids)):
            reason = self.reason(item_id)
            if reason is not None:
                lists[reason].append(item_id)
        return lists


NONE = Exclusions.of()


def current(verdicts_path: Path = VERDICTS_PATH) -> Exclusions:
    """What each guard takes out at the verdicts in ``verdicts_path`` (read at every use)."""
    return Exclusions({WAQF_REALIZED: realized_ids(verdicts_path=verdicts_path),
                       ONSET_TRIMMED: trimmed_ids(verdicts_path=verdicts_path)})
