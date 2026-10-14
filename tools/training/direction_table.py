"""Per target-pair direction and in-word geminate, by verdict side: what one arm decoded (#90).

The decode-level diagnostic of goals 1-2 (ADR-0012 keeps per-decode fidelity as the
explanation of an app outcome). This is the **single-model core**: one arm (a model under a
decode protocol) over verified sites. #57's diff compares two arms through the same rows and
:func:`difference`; nothing here is duplicated there.

**Rows.** Each verified pair site has a **direction**: the mushaf's letter and its partner in
the pair (``ط→ق`` is a ``ط`` the mushaf prescribes, whose partner is ``ق``), so the nine target
pairs (acceptance rules §7) give eighteen directions. An in-word geminate (``shaddah`` site,
prescribed ``held``) is the direction ``geminate``. Each direction is split by **verdict side**
(``training.site_outcomes.side``), never pooled across sides (ADR-0008):

* **correct** recitation: the mushaf's letter, or a held geminate, was heard;
* **mistake**: the partner, or a single consonant, was heard.

Every direction also pools into its **family**, as Muraja and §7 group them: the six soft pairs
together (Muraja has one switch for them), and ``ذ↔ظ``, ``ط↔ق``, ``س↔ش`` and the geminates each
on their own; the non-soft pairs are never pooled with the soft ones. A direction too small for
a claim (fewer than 10 reciters or 20 sites, ``acceptance_stats.is_sparse``) is marked
``insufficient_evidence``; its family row is where a claim can be read.

**Rates**, weighted ratios over a row's sites (``truth_scorer.SiteRow``: its design weight,
reciter and outcome per arm, exactly as the report scores them):

* ``swap``: the decode committed the partner (a geminate: a single consonant). On the correct
  side a mishearing (the partner-swap and geminate-collapse rates); on the mistake side the
  model hearing the mistake;
* ``mushaf``: it committed the mushaf's letter (a held geminate). On the mistake side a silent
  correction;
* ``other``: neither: another letter, or nothing aligned at the carrier.

Each rate carries its §1 interval (``truth_scorer.estimate``: the reciter-clustered bootstrap,
or a Wilson bound where a sparse or degenerate row is independent and equally weighted, else
none), its point and sums on the sites' own weights and its interval on §1's unit scale
(``truth_scorer.unit_scale``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tadabur.truth_sites import HELD, NOT_HELD, SHADDAH, SOFT_PAIRS, TARGET_PAIRS
from training.acceptance_stats import Resample, difference_interval, is_sparse, ratio
from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE, side
from training.truth_scorer import SiteRow, estimate, unit_scale

GEMINATE = "geminate"
SOFT_PAIRS_FAMILY = "soft pairs"
#: A family row pools every direction of its family.
ALL_DIRECTIONS = "all"
SWAP = "swap"
MUSHAF = "mushaf"
OTHER = "other"
RATES = (SWAP, MUSHAF, OTHER)
SIDES = (CORRECT_SIDE, MISTAKE_SIDE)


def _pair_directions(pair: str) -> list[str]:
    a, b = pair.split("↔")
    return [f"{a}→{b}", f"{b}→{a}"]


def family(pair: str) -> str:
    """A pair's family: the soft pairs pool together, every other target pair is its own."""
    return SOFT_PAIRS_FAMILY if pair in SOFT_PAIRS else pair


#: Every family and its directions, in report order: the soft pairs, the three target pairs
#: Muraja does not forgive, then the geminates.
FAMILIES: dict[str, tuple[str, ...]] = {
    SOFT_PAIRS_FAMILY: tuple(d for p in sorted(SOFT_PAIRS) for d in _pair_directions(p)),
    **{p: tuple(_pair_directions(p)) for p in sorted(TARGET_PAIRS - SOFT_PAIRS)},
    GEMINATE: (GEMINATE,),
}


def partner(site) -> str:
    """What a swap commits at the site: the pair's other letter, or a single consonant."""
    if site.mark == SHADDAH:
        return NOT_HELD
    a, b = site.mark.split("↔")
    return b if site.prescribed == a else a


def direction(site) -> str | None:
    """The site's direction, or ``None`` for a site no row holds (a tashkeel site, an added
    shaddah's)."""
    if site.mark == SHADDAH:
        return GEMINATE if site.prescribed == HELD else None
    if site.mark not in TARGET_PAIRS:
        return None
    return f"{site.prescribed}→{partner(site)}"


@dataclass(frozen=True, order=True)
class Key:
    """One row: a family, one of its directions (or :data:`ALL_DIRECTIONS`), one side."""

    family: str
    direction: str
    side: str

    def holds(self, row: SiteRow) -> bool:
        found = direction(row.site)
        return (found is not None and side(row.site) == self.side
                and found in FAMILIES[self.family]
                and self.direction in (ALL_DIRECTIONS, found))


#: Every row in report order: per family, each direction then the family pooled, per side.
KEYS: tuple[Key, ...] = tuple(
    Key(name, d, s)
    for name, directions in FAMILIES.items()
    for d in (*directions, ALL_DIRECTIONS)
    if not (d == ALL_DIRECTIONS and len(directions) == 1)
    for s in SIDES
)


def _terms(rows: Sequence[SiteRow], arm: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each rate's weighted per-site (numerator, denominator) for one arm."""
    w = np.array([r.weight for r in rows], dtype=float)
    committed = [r.outcomes[arm].committed for r in rows]
    swap = np.array([c == partner(r.site) for c, r in zip(committed, rows)], dtype=float)
    mushaf = np.array([c == r.site.prescribed for c, r in zip(committed, rows)], dtype=float)
    return {SWAP: (w * swap, w), MUSHAF: (w * mushaf, w), OTHER: (w * (1 - swap - mushaf), w)}


def rates(rows: Sequence[SiteRow], arm: str) -> dict[str, dict]:
    """Each rate of one arm over a row's sites, with its §1 interval (module docstring)."""
    clusters = [r.reciter_id for r in rows]
    unit_rows = unit_scale(rows)
    w = np.array([r.weight for r in unit_rows], dtype=float)
    terms, unit_terms = _terms(rows, arm), _terms(unit_rows, arm)
    resample = Resample.by_cluster(clusters)
    return {name: estimate(clusters, terms[name], w, resample, unit_terms[name]) for name in RATES}


def difference(rows: Sequence[SiteRow], arm: str, other: str, rate: str = SWAP) -> dict:
    """``arm`` − ``other`` on one rate over the same sites, paired (one reciter draw serves
    both, :func:`training.acceptance_stats.difference_interval`)."""
    clusters = [r.reciter_id for r in rows]
    unit_rows = unit_scale(rows)
    w = np.array([r.weight for r in unit_rows], dtype=float)
    first, second = _terms(unit_rows, arm)[rate], _terms(unit_rows, other)[rate]
    point = ratio(*_terms(rows, arm)[rate]) - ratio(*_terms(rows, other)[rate])
    interval = difference_interval(clusters, first, second, w, Resample.by_cluster(clusters))
    return {"point": point if math.isfinite(point) else None, **interval.as_dict()}


def members(rows: Sequence[SiteRow], key: Key) -> list[SiteRow]:
    return [row for row in rows if key.holds(row)]


def table(rows: Sequence[SiteRow], arms: Sequence[str]) -> list[dict]:
    """Every row of :data:`KEYS` over ``rows``: its sites, reciters, whether it is
    insufficient evidence, and each arm's rates (absent when the row has no site)."""
    out = []
    for key in KEYS:
        held = members(rows, key)
        reciters = len({r.reciter_id for r in held})
        entry = {"family": key.family, "direction": key.direction, "side": key.side,
                 "sites": len(held), "reciters": reciters,
                 "insufficient_evidence": is_sparse(len(held), reciters)}
        if held:
            entry["arms"] = {arm: rates(held, arm) for arm in arms}
        out.append(entry)
    return out
