"""The two-sided eval at decode level, split by fixture side (ADR-0008, #55): the pure core.

ADR-0001 wants the fine-tune judged two-sided: aggregate accuracy can improve while the model's
ability to tell a soft-pair consonant from its partner collapses. ADR-0008 locates where that
signal lives: in the **decode**, read separately on the two fixture sides, because one
confusion cell means opposite things on them. Reference ``ز`` decoded ``ذ`` is a mishearing
where the reciter said ``ز``, and the model correctly hearing the mistake where the reciter
said ``ذ``. Summed, a model that improved in both directions and one that got worse in both
can post the same number. This module builds the per-side matrices and nothing that pools
them. Torch-free: sites, decodes and fingerprints in, a JSON-ready report out.

What is read
------------
The **re-located P3.5 fixtures** (#83, ``tadabur/truth_sites/p35_fixtures.jsonl``): one truth
site per labelled contrast occurrence, pinned to its carrier, rather than every aligned column
of the clip (ADR-0008: a clip label says nothing about its other positions). Each site's
outcome under each decode is #84's (:func:`training.truth_scorer.outcomes_by_arm`, the
consonant-commitment and gemination rules frozen in :mod:`training.site_outcomes`), so there
is one attribution path. The decodes are #84's caches (:mod:`tadabur.eval_harness` reads them).

Fixture side and recitation side
--------------------------------
A should-accept fixture says the mushaf's letter (or gemination) was said at the site:
``heard = prescribed``, **correct recitation**. A should-reject fixture is a clip-level
verdict that never says what was said at a given site (acceptance rules §8), so its sites stay
``pending`` until the listening session (#61) hears each one. A site heard ≠ prescribed is a
**real mistake**; one heard = prescribed joins the correct side whatever its fixture said;
``unclear`` leaves every denominator. So the matrices are split by this site-level
recitation side (acceptance rules §1, *Sides*), each row records which fixture side its sites
came from, and the real-mistake side fills in by itself once verdicts land. Until then it
reports its sites as pending adjudication, with no numbers.

Rows, roles and the sign convention
-----------------------------------
A row is one :class:`training.truth_scorer.Cell` of the targeted-safeguard population: one
side, one family (a target pair, ``ذ↔ظ`` included, or shaddah) and one heard value, so pair
rows are **directional** (§7). Every pair and gemination state has a row on both sides, with
or without sites. What the decode committed at a site is one **role**:

* :data:`AS_HEARD`: what the reciter said;
* :data:`AS_PARTNER`: the pair's other letter, or the other gemination state;
* :data:`OTHER_CONSONANT`: a letter outside the pair (pair rows only);
* :data:`NO_COMMIT`: nothing at the carrier (C = 0).

:data:`SIGN_CONVENTIONS` names each role's meaning per side. On the correct side
:data:`AS_PARTNER` is a mishearing; on the mistake side it is the mushaf's value, a collapse
onto the reference (a silent correction).

Rates, intervals and support
----------------------------
Each role's rate is Σw·[role] / Σw over the row's sites (the roles of a row sum to 1), with
its §1 interval from :mod:`training.acceptance_stats`: the reciter-clustered bootstrap, a
Wilson bound where a sparse or degenerate row is independent and equally weighted, otherwise
none. ADR-0008 asked for clip-clustered intervals; every clip has one reciter, so reciter
clusters nest the clips and are at least as conservative. A row with fewer than 10 reciters
or 20 sites is **too small** to support a claim. No row carries a verdict, and a family
without sufficient support on both sides is "in scope, insufficient evidence" (§7). Shaddah
rows are provisional until #92.

Each row also lists the ``pending`` and ``unclear`` sites that could belong to it once heard
(:meth:`training.truth_scorer.Cell.could_hold`: same family, and a prescribed value that fits
the row's side), and every role rate carries §1's best and worst case with their weight
counted in (:func:`training.truth_scorer.exclusion_range`, the rule #84's cells use).

Never pooled
------------
:func:`confusion_row` takes only a directional cell (one side, one family, one of its heard
values; never a pooled ``all`` cell) and refuses a site it does not hold, and
:func:`fixture_report` builds every row through it, so no public path puts both sides (or two
directions, or two families) in one row, and there is no cross-side total.

For the paired diff (#57)
-------------------------
Every site is recorded with its item's fingerprint (audio checksum, span, realized reference)
and, per arm, what was committed and its role. The report carries the fixture fingerprint
(the scored sites, their fixture sides and reciters), the schema fingerprint and, per arm,
the model identity, decode fingerprint and a hash of its decodes, so a diff can refuse two
reports of different cohorts.

``strict_accept`` (ADR-0001's data-hygiene gate at the ``.strict`` threshold) is not read
here. It lives with the gate in :mod:`tadabur.scorer`.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from tadabur.truth_sites import (
    P35_FIXTURE,
    SCHEMA_FIELDS,
    SHADDAH,
    TARGET_PAIRS,
    TruthSite,
    label_states,
)
from training.acceptance_stats import REPLICATES, SEED, Resample, is_sparse
from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE, SiteOutcome, family, side
from training.truth_scorer import (
    SAFEGUARD,
    Cell,
    estimate,
    exclusion_range,
    item_key,
    outcomes_by_arm,
    site_weights,
)

#: The version of this report's shape; part of its schema fingerprint.
REPORT_SCHEMA = "tadabur.eval_report/fixture-sides-v1"

SHOULD_ACCEPT = "should_accept"
SHOULD_REJECT = "should_reject"
FIXTURE_SIDES = (SHOULD_ACCEPT, SHOULD_REJECT)

SIDES = (CORRECT_SIDE, MISTAKE_SIDE)
#: Every family with a row on both sides: the seven target pairs (§7), then shaddah.
FAMILIES = (*sorted(TARGET_PAIRS), SHADDAH)

AS_HEARD = "as_heard"
AS_PARTNER = "as_partner"
OTHER_CONSONANT = "other_consonant"
NO_COMMIT = "no_commit"

SIGN_CONVENTIONS: dict[str, dict[str, str]] = {
    CORRECT_SIDE: {
        "side": "heard = prescribed: the mushaf's letter or gemination was said",
        AS_HEARD: "committed what was said",
        AS_PARTNER: "misheard as the partner (the confusable letter, or the other gemination state)",
        OTHER_CONSONANT: "misheard as a letter outside the pair",
        NO_COMMIT: "nothing committed at the carrier",
    },
    MISTAKE_SIDE: {
        "side": "heard ≠ prescribed: the reciter said the partner of the mushaf's letter or state",
        AS_HEARD: "the mistake heard",
        AS_PARTNER: "collapsed onto the reference: the mushaf's value, a silent correction",
        OTHER_CONSONANT: "a third letter: flagged, but not what was said",
        NO_COMMIT: "nothing committed at the carrier: the mistake missed",
    },
}

SUFFICIENT = "sufficient"
TOO_SMALL = "too_small"
NO_SITE = "none"
SUPPORTED = "supported"
INSUFFICIENT_EVIDENCE = "in scope, insufficient evidence"


def members(family_: str) -> list[str]:
    """The values a family's sites prescribe and are heard as: a pair's two letters, or the
    two gemination states."""
    return sorted(label_states(family_)[0])


def roles(family_: str) -> tuple[str, ...]:
    if family_ == SHADDAH:
        return (AS_HEARD, AS_PARTNER, NO_COMMIT)
    return (AS_HEARD, AS_PARTNER, OTHER_CONSONANT, NO_COMMIT)


def role(site: TruthSite, outcome: SiteOutcome) -> str:
    """What the decode committed at a site with a verdict, relative to what was heard."""
    if not outcome.commits:
        return NO_COMMIT
    if outcome.committed == site.heard:
        return AS_HEARD
    if outcome.committed in members(site.mark):
        return AS_PARTNER
    return OTHER_CONSONANT


def fingerprint(value) -> str:
    """SHA-256 of a value's canonical JSON."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def item_fingerprint(site: TruthSite) -> str:
    """The decoded item's identity: its audio's checksum, its span and its realized reference."""
    return fingerprint([site.audio_sha256, site.start_sample, site.end_sample, site.reference])


def schema_fingerprint() -> str:
    """Changes with this report's shape, the truth-site schema, the families or the roles."""
    return fingerprint({
        "report": REPORT_SCHEMA,
        "truth_site_fields": list(SCHEMA_FIELDS),
        "families": list(FAMILIES),
        "roles": {f: list(roles(f)) for f in FAMILIES},
    })


@dataclass(frozen=True)
class FixtureSite:
    """One P3.5 site with the fixture side it came from, its reciter and its stratum weight."""

    site: TruthSite
    fixture_side: str
    reciter_id: int
    weight: float


def _support(sites: int, reciters: int) -> str:
    if not sites:
        return NO_SITE
    return TOO_SMALL if is_sparse(sites, reciters) else SUFFICIENT


def _check_cell(cell: Cell) -> None:
    """A row is one direction: a safeguard cell on one side, of one family, heard as one of
    that family's values. A pooled ``all`` cell (or any other) is refused."""
    if not (
        cell.population == SAFEGUARD
        and cell.side in SIDES
        and cell.family in FAMILIES
        and cell.heard in members(cell.family)
    ):
        raise ValueError(
            f"{cell} is not one direction of one family on one side: "
            "correct recitation and real mistakes are never pooled"
        )


def confusion_row(
    cell: Cell,
    sites: Sequence[FixtureSite],
    excluded: Sequence[FixtureSite],
    outcomes: Mapping[str, Mapping[str, SiteOutcome]],
) -> dict:
    """One directional row: each arm's rate of every role over the cell's sites, with its
    best and worst case once the ``excluded`` sites (no verdict yet) are counted in.

    Raises on a cell that is not one direction (:func:`_check_cell`), on any site the cell
    does not hold (the other side, another direction, or one with no verdict), and on an
    excluded site that has a verdict or could not belong here. Those refusals are what keep
    the sides apart. ``outcomes[arm][site_id]`` is each site's outcome under each arm.
    """
    _check_cell(cell)
    strays = sorted(s.site.site_id for s in sites if not cell.holds(s.site))
    strays += sorted(
        s.site.site_id for s in excluded if side(s.site) is not None or not cell.could_hold(s.site)
    )
    if strays:
        raise ValueError(
            f"{cell.label} on the {cell.side} side does not hold {strays[:3]}: "
            "correct recitation and real mistakes are never pooled"
        )
    heard = cell.heard
    prescribed = heard if cell.side == CORRECT_SIDE else next(m for m in members(cell.family) if m != heard)
    clusters = [s.reciter_id for s in sites]
    row = {
        "side": cell.side,
        "family": cell.family,
        "prescribed": prescribed,
        "heard": heard,
        "sites": len(sites),
        "reciters": len(set(clusters)),
        "support": _support(len(sites), len(set(clusters))),
        "provisional": cell.family == SHADDAH,
        "fixture_sides": {f: sum(s.fixture_side == f for s in sites) for f in FIXTURE_SIDES},
        "excluded": sorted(s.site.site_id for s in excluded),
        "arms": {},
    }
    if not sites:
        return row
    resample = Resample.by_cluster(clusters)
    weights = np.array([s.weight for s in sites], dtype=float)
    excluded_weight = sum(s.weight for s in excluded)
    for arm in sorted(outcomes):
        committed = [role(s.site, outcomes[arm][s.site.site_id]) for s in sites]
        rates = {}
        for r in roles(cell.family):
            num = weights * np.array([c == r for c in committed], dtype=float)
            rates[r] = {
                "sites": committed.count(r),
                **estimate(clusters, (num, weights), weights, resample),
                "sensitivity": (
                    exclusion_range(float(num.sum()), float(weights.sum()), excluded_weight)
                    if excluded else None
                ),
            }
        row["arms"][arm] = rates
    return row


def _family_support(rows: Sequence[dict], pending: Counter) -> list[dict]:
    """Per family, the best direction's support on each side, the sites still without a
    verdict, and the §7 status."""
    rank = {NO_SITE: 0, TOO_SMALL: 1, SUFFICIENT: 2}
    table = []
    for family_ in FAMILIES:
        best = {
            side_: max(
                (r["support"] for r in rows if (r["family"], r["side"]) == (family_, side_)),
                key=rank.__getitem__,
            )
            for side_ in SIDES
        }
        supported = all(best[s] == SUFFICIENT for s in SIDES)
        table.append({
            "family": family_,
            **best,
            "without_verdict": sum(n for key, n in pending.items() if key[0] == family_),
            "status": SUPPORTED if supported else INSUFFICIENT_EVIDENCE,
            "provisional": family_ == SHADDAH,
        })
    return table


def _check(sites: Sequence[TruthSite], fixture_side_of: Mapping[str, str]) -> None:
    if len({s.site_id for s in sites}) != len(sites):
        raise ValueError("duplicate site ids")
    for site in sites:
        if site.source != P35_FIXTURE or family(site.mark) not in FAMILIES:
            raise ValueError(f"{site.site_id}: not a P3.5 pair or shaddah site")
        fixture = fixture_side_of.get(site.site_id)
        if fixture not in FIXTURE_SIDES:
            raise ValueError(f"{site.site_id}: no fixture side ({fixture!r})")
        if fixture == SHOULD_ACCEPT and side(site) != CORRECT_SIDE:
            raise ValueError(f"{site.site_id}: a should-accept site is heard as prescribed")


def fixture_report(
    sites: Sequence[TruthSite],
    fixture_side_of: Mapping[str, str],
    reciter_of: Mapping[str, int],
    decodes: Mapping[str, Mapping[str, str]],
    models: Mapping[str, Mapping],
) -> dict:
    """The fixture-side report (module docstring).

    ``sites`` are the P3.5 truth sites with any site-level verdict applied;
    ``fixture_side_of[site_id]`` is :data:`SHOULD_ACCEPT` or :data:`SHOULD_REJECT`;
    ``reciter_of[audio_filename]`` a canonical reciter id; ``decodes[arm][item_key]`` a decode;
    ``models[arm]`` the arm's model identity and decode fingerprint, recorded as given.
    """
    _check(sites, fixture_side_of)
    if set(models) != set(decodes):
        raise ValueError("every arm needs its model fingerprint, and only its arms")
    weights = site_weights(sites)
    fixture_sites = [
        FixtureSite(s, fixture_side_of[s.site_id], reciter_of[s.audio_filename], weights[s.site_id])
        for s in sorted(sites, key=lambda s: s.site_id)
    ]
    outcomes = outcomes_by_arm(sites, decodes)
    arms = sorted(decodes)

    rows = []
    for side_ in SIDES:
        for family_ in FAMILIES:
            for heard in members(family_):
                cell = Cell(SAFEGUARD, side_, family_, heard)
                held = [s for s in fixture_sites if cell.holds(s.site)]
                could = [s for s in fixture_sites if side(s.site) is None and cell.could_hold(s.site)]
                rows.append(confusion_row(cell, held, could, outcomes))

    pending = Counter(
        (family(s.site.mark), s.site.prescribed, s.site.heard, s.fixture_side)
        for s in fixture_sites
        if side(s.site) is None
    )
    items = sorted({item_key(s) for s in sites})
    return {
        "schema": REPORT_SCHEMA,
        "fingerprints": {
            "fixtures": fingerprint([
                {**asdict(s.site), "fixture_side": s.fixture_side, "reciter_id": s.reciter_id}
                for s in fixture_sites
            ]),
            "schema": schema_fingerprint(),
            "arms": {
                arm: {**models[arm], "decodes_sha256": fingerprint({k: decodes[arm][k] for k in items})}
                for arm in arms
            },
        },
        "bootstrap": {"replicates": REPLICATES, "seed": SEED, "cluster": "canonical reciter id"},
        "arms": arms,
        "sign_conventions": SIGN_CONVENTIONS,
        "families": _family_support(rows, pending),
        "rows": rows,
        "pending": [
            dict(zip(("family", "prescribed", "heard", "fixture_side"), key), sites=n)
            for key, n in sorted(pending.items())
        ],
        "sites": [_site_record(s, outcomes, arms) for s in fixture_sites],
    }


def _site_record(
    s: FixtureSite, outcomes: Mapping[str, Mapping[str, SiteOutcome]], arms: Sequence[str]
) -> dict:
    """One site for the paired diff: its item's fingerprint and, per arm, the commit and role."""
    site = s.site
    has_verdict = side(site) is not None
    return {
        "site_id": site.site_id,
        "item": item_key(site),
        "item_fingerprint": item_fingerprint(site),
        "reciter_id": s.reciter_id,
        "fixture_side": s.fixture_side,
        "side": side(site),
        "family": family(site.mark),
        "prescribed": site.prescribed,
        "heard": site.heard,
        "weight": s.weight,
        "arms": {
            arm: {
                "committed": outcomes[arm][site.site_id].committed,
                "role": role(site, outcomes[arm][site.site_id]) if has_verdict else None,
            }
            for arm in arms
        },
    }
