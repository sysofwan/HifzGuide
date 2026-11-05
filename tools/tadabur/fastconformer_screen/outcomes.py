"""#267's letter-level outcomes: what each column carries at a site, and unique catches.

Every column (the teacher, C95, the round-2 EMA and FastConformer) is read the same way: what
does the whole-item decode carry at the site's letter, or its shadda? A site's outcome is

* **heard**: it carries the letter the human heard (on a mistake site, the caught mistake);
* **prescribed**: it carries the mushaf's letter instead (the text prior wrote over the site);
* **other**: it carries a third letter;
* **none**: nothing is aligned at the carrier (an empty slot, a gap or an unaligned carrier).

#267's rule counts a site **flagged** when it does not carry the prescribed letter, so
``none`` is flagged. A **unique catch** is a decisive mistake where FastConformer carries the
heard letter and none of the three other columns does. The halves are never pooled.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

HEARD, PRESCRIBED, OTHER, NONE = "heard", "prescribed", "other", "none"
OUTCOMES = (HEARD, PRESCRIBED, OTHER, NONE)
#: FastConformer first; the three columns it is compared against, in the report's order.
COLUMNS = ("fastconformer", "base", "c95", "r2_candidate_ema")


def outcome(carried: str | None, prescribed: str, heard: str) -> str:
    """One site's outcome from what the column carried there."""
    if carried is None:
        return NONE
    if carried == heard:
        return HEARD
    if carried == prescribed:
        return PRESCRIBED
    return OTHER


def is_flagged(carried: str | None, prescribed: str) -> bool:
    """#267's flag: the column did not carry the prescribed letter (``none`` counts)."""
    return carried != prescribed


def edit_heard(carried: str | None, label: str) -> bool:
    """An edit was heard when the column carries the letter the edit made audible."""
    return carried is not None and carried == label


def decoy_unchanged(carried: str | None, prescribed: str) -> bool:
    """A decoy is unchanged when the column still carries the letter the edit left alone."""
    return carried is not None and carried == prescribed


def breakdown(outcomes: Iterable[str]) -> dict[str, int]:
    """How many of each outcome, zeros included, in :data:`OUTCOMES` order."""
    counts = Counter(outcomes)
    return {name: counts[name] for name in OUTCOMES}


def unique_catches(outcomes: dict[str, dict[str, str]]) -> list[str]:
    """The sites FastConformer heard and none of the other three columns did, sorted."""
    caught = []
    for site_id, row in outcomes.items():
        missing = [name for name in COLUMNS if name not in row]
        if missing:
            raise ValueError(f"{site_id}: no outcome for {', '.join(missing)}")
        if row["fastconformer"] == HEARD and all(
                row[name] != HEARD for name in COLUMNS if name != "fastconformer"):
            caught.append(site_id)
    return sorted(caught)


def student_carried(model: str, sites: Sequence) -> dict[str, str | None]:
    """Each site's carried value under one student's committed whole-item decode."""
    from collections import defaultdict

    from training.site_outcomes import item_outcomes
    from training.truth_baseline import DECODE_STORE, read_cache
    from training.truth_scorer import item_key
    from training.zipformer_readout import WHOLE_ITEM

    items = read_cache(DECODE_STORE.path(model))["items"]
    by_item: dict[str, list] = defaultdict(list)
    for site in sites:
        by_item[item_key(site)].append(site)
    missing = sorted(set(by_item) - set(items))
    if missing:
        raise ValueError(
            f"{model}: {len(missing)} items have no decode, e.g. {missing[:3]}")
    carried: dict[str, str | None] = {}
    for key, item_sites in by_item.items():
        outcomes = item_outcomes(item_sites, items[key][WHOLE_ITEM[model]])
        carried.update({site_id: value.committed for site_id, value in outcomes.items()})
    return carried
