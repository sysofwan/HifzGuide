"""Ship A's sites and its reciter split (#126): what the owner verifies for goals 1-2.

Ship A (ADR-0012) trains on sites the owner verifies by ear: where the shipped ``h448``
(streamed at b = 0, no bias) commits a wrong letter at a target-pair carrier, or collapses
an in-word geminate. Training labels may be mined from a model's flags (ADR-0012,
acceptance rules §1); evaluation on them is stratified by that outcome, with inclusion
probability 1 inside the pool (a census), and read on held-out reciters only.

**Sites.** From the scored pool (:func:`tadabur.power_frame.score_pool`, the #84 scorer's
outcomes of both frozen decodes):

* **pair**: every target-pair carrier (the six soft pairs and ``ذ↔ظ``) where ``h448``
  commits a consonant other than the mushaf letter (F at decode level). One site per
  carrier: a ``ذ`` or ``ظ`` carrier sits in two pairs, and is asked as the pair whose partner
  ``h448`` committed, else as the first of its pairs in label order. Stratum
  ``ship_a:<prescribed>→<partner>:h448_partner`` when ``h448`` committed the pair partner (the
  swap the soft-pair allowance forgives), ``ship_a:<prescribed>:h448_other`` for any other
  letter (no allowance forgives it).
* **geminate**: every reference geminate inside one word that ``h448`` decodes as one
  consonant. Cross-word geminates are left out: the app never scores them (#86).

**Base already right** (ADR-0011 §3). Where the base teacher commits the mushaf value at a
flagged site, the training target is the teacher's own decode, so the label needs no
listening. Geminates the teacher holds therefore leave the worklist
(``ship_a:shaddah:held:h448_single:base_held``); the rest are listened to
(``…:base_not_held``). Pair sites stay on the worklist whatever the teacher did (D6).

**Reciter split** (acceptance rules §9). A canonical reciter is in the ``train`` half when
:func:`training.acceptance_stats.reciter_hash` under :data:`SPLIT_SALT` is below
``round((1 − held-out share) · 2**64)``, else ``held_out``; at a share of ½ the cutoff is 2**63,
the bias split's. The 50/50 split is a **proposal**: the owner confirms it before the first
sitting. ``split.json`` lists every pool reciter's half under it and under
:data:`ALTERNATIVE_HELD_OUT_SHARE`.

Files (``tools/tadabur/ship_a/``): ``sites.jsonl`` (one site per line: the worklist row of
:mod:`tadabur.listening_session`, with its provenance, plus the reciter, its half and what
each decode committed), ``sites.summary.json`` (the population sidecar: per stratum and per
family, sites and reciters by half and the base teacher's outcome; the frame's carrier
populations; the inputs' checksums) and ``split.json``.

Usage (from ``tools/``; torch-free, needs ``quran-transcript``, ~3 min)::

  python -m tadabur.ship_a mine
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from training.acceptance_stats import reciter_hash
from training.site_outcomes import family
from training.tashkeel_eval import write_text_atomically

from .listening_session import POOL_CLIPS_PATH, SessionSite, parse_row, row_json
from .mining_pool import load_manifest
from .power_frame import (
    FRAME_PATH,
    SHIP_A_DIR,
    ScoredSite,
    direction,
    load_scored_pool,
    partner,
)
from .truth_sites import HELD, NOT_HELD, SHADDAH, TARGET_PAIRS, TruthSite, _check_file

SITES_PATH = SHIP_A_DIR / "sites.jsonl"
SITES_SUMMARY_PATH = SHIP_A_DIR / "sites.summary.json"
SPLIT_PATH = SHIP_A_DIR / "split.json"

SPLIT_SALT = "ship-a-verified-split-2026-10"
TRAIN = "train"
HELD_OUT = "held_out"
HALVES = (TRAIN, HELD_OUT)
#: The proposed split, 50/50 (the owner confirms it before the first sitting).
HELD_OUT_SHARE = 0.5
#: The alternative the sizing reports beside it: two thirds held out, for the held-out
#: half's power, since labels to train on are needed in far smaller numbers than claims.
ALTERNATIVE_HELD_OUT_SHARE = 2 / 3

PAIR = "pair"
GEMINATE = "geminate"
#: The frame stratum Ship A's geminates come from (``tadabur.power_frame.frame_stratum``).
IN_WORD_GEMINATES = "shaddah:held:in_word"


def reciter_half(reciter_id: int, held_out_share: float = HELD_OUT_SHARE) -> str:
    """``train`` or ``held_out`` for one canonical reciter id (module docstring)."""
    cutoff = round((1 - held_out_share) * 2**64)
    return TRAIN if reciter_hash(reciter_id, SPLIT_SALT) < cutoff else HELD_OUT


@dataclass(frozen=True)
class ShipASite:
    """One Ship A site: its worklist row (Ship A's stratum and population, inclusion
    probability its clip's in the pool), its clip's reciter and half, and what ``h448`` and
    the base teacher committed at the carrier (``None``: nothing aligned)."""

    row: SessionSite
    reciter_id: int
    half: str
    h448_committed: str | None
    base_committed: str | None

    @property
    def site(self) -> TruthSite:
        return self.row.site

    @property
    def kind(self) -> str:
        return GEMINATE if self.site.mark == SHADDAH else PAIR

    @property
    def family(self) -> str:
        """The pair label, or ``shaddah``."""
        return family(self.site.mark)

    @property
    def base_already_right(self) -> bool:
        """The base teacher commits the mushaf value: ADR-0011 §3 takes its decode as target."""
        return self.base_committed == self.site.prescribed

    @property
    def listened(self) -> bool:
        """On the worklist: every pair site, and the geminates the teacher does not hold."""
        return self.kind == PAIR or not self.base_already_right

    @property
    def partner_swap(self) -> bool:
        return self.kind == PAIR and self.h448_committed == partner(self.site)


def ship_a_stratum(scored: ScoredSite) -> str:
    """The stratum a flagged site is mined into (module docstring)."""
    site = scored.site
    if site.mark == SHADDAH:
        teacher = "base_held" if scored.base_mushaf else "base_not_held"
        return f"ship_a:shaddah:held:h448_single:{teacher}"
    if scored.h448.committed == partner(site):
        return f"ship_a:{direction(site)}:h448_partner"
    return f"ship_a:{site.prescribed}:h448_other"


def flagged_sites(scored: Iterable[ScoredSite]) -> list[ScoredSite]:
    """Every pair carrier ``h448`` flags, once, and every in-word geminate it collapses."""
    carriers: dict[tuple, list[ScoredSite]] = defaultdict(list)
    geminates = []
    for one in scored:
        if one.site.mark in TARGET_PAIRS and one.h448_flagged:
            site = one.site
            carriers[(site.audio_filename, site.start_sample, site.reference_index)].append(one)
        elif one.stratum == IN_WORD_GEMINATES and one.h448.committed == NOT_HELD:
            geminates.append(one)
    chosen = []
    for key in sorted(carriers):
        options = sorted(carriers[key], key=lambda one: one.site.mark)
        chosen.append(next((o for o in options if o.h448.committed == partner(o.site)),
                           options[0]))
    return chosen + geminates


def mine(scored: Iterable[ScoredSite]) -> list[ShipASite]:
    """Ship A's sites, each in its stratum with that stratum's population, sorted by id."""
    flagged = flagged_sites(scored)
    strata = [ship_a_stratum(one) for one in flagged]
    population: dict[str, int] = defaultdict(int)
    for stratum in strata:
        population[stratum] += 1
    sites = [
        ShipASite(
            row=replace(one.row, site=replace(one.site, stratum=stratum,
                                              stratum_population=population[stratum])),
            reciter_id=one.reciter_id,
            half=reciter_half(one.reciter_id),
            h448_committed=one.h448.committed,
            base_committed=one.base.committed,
        )
        for one, stratum in zip(flagged, strata)
    ]
    return sorted(sites, key=lambda s: s.site.site_id)


# --- the files -------------------------------------------------------------------------------

_SITE_FIELDS = ("reciter_id", "half", "h448_committed", "base_committed")


def site_json(site: ShipASite) -> dict:
    return {**row_json(site.row), **{name: getattr(site, name) for name in _SITE_FIELDS}}


def parse_ship_a_site(data: dict, where: str) -> ShipASite:
    """One validated row: a worklist row plus the four Ship A fields, its half recomputed."""
    extra = {name: data.get(name) for name in _SITE_FIELDS}
    row = parse_row({k: v for k, v in data.items() if k not in _SITE_FIELDS}, where)
    site = ShipASite(row=row, **extra)
    if type(site.reciter_id) is not int or site.half != reciter_half(site.reciter_id):
        raise ValueError(f"{where}: half {site.half!r} is not reciter {site.reciter_id}'s")
    if site.kind == GEMINATE and (site.site.prescribed, site.h448_committed) != (HELD, NOT_HELD):
        raise ValueError(f"{where}: a Ship A geminate is a held one h448 decoded single")
    if site.kind == PAIR and site.h448_committed in (None, site.site.prescribed):
        raise ValueError(f"{where}: a Ship A pair site is one h448 committed another letter at")
    return site


def write_sites(sites: list[ShipASite], path: Path = SITES_PATH) -> None:
    """Validate every row and the file, then write atomically, one JSON object per line."""
    for number, site in enumerate(sites, 1):
        parse_ship_a_site(site_json(site), f"write_sites[{number}]")
    _check_file([site.site for site in sites], "write_sites")
    write_text_atomically(path, "".join(
        json.dumps(site_json(site), ensure_ascii=False, sort_keys=True) + "\n" for site in sites
    ))


def load_sites(path: Path = SITES_PATH) -> list[ShipASite]:
    """Every Ship A site in file order, each row and the truth-site file invariants checked."""
    with open(path, encoding="utf-8") as f:
        sites = [parse_ship_a_site(json.loads(raw), f"{path}:{n}")
                 for n, raw in enumerate(f, 1) if raw.strip()]
    _check_file([site.site for site in sites], str(path))
    return sites


# --- the sidecars ----------------------------------------------------------------------------


def _counts(sites: list[ShipASite]) -> dict:
    """Sites and reciters, in all and per half, and how many the base teacher already gets right."""
    def reciters(group) -> int:
        return len({s.reciter_id for s in group})

    return {
        "sites": len(sites),
        "reciters": reciters(sites),
        "base_already_right": sum(s.base_already_right for s in sites),
        "listened": sum(s.listened for s in sites),
        **{half: {"sites": len(group), "reciters": reciters(group),
                  "base_already_right": sum(s.base_already_right for s in group),
                  "listened": sum(s.listened for s in group)}
           for half in HALVES for group in [[s for s in sites if s.half == half]]},
    }


def summarize(sites: list[ShipASite], frame_strata: dict, provenance: dict) -> dict:
    """The population sidecar: per Ship A stratum and per family, the counts of
    :func:`_counts`; the frame's carrier populations per stratum; the inputs."""
    def grouped(key) -> dict:
        groups: dict[str, list[ShipASite]] = defaultdict(list)
        for site in sites:
            groups[key(site)].append(site)
        return {name: _counts(group) for name, group in sorted(groups.items())}

    return {
        **provenance,
        "split": {"salt": SPLIT_SALT, "held_out_share": HELD_OUT_SHARE, "status": "proposal"},
        "strata": grouped(lambda s: s.site.stratum),
        "families": grouped(lambda s: s.family),
        "pair_commits": {
            "partner": _counts([s for s in sites if s.partner_swap]),
            "other_letter": _counts([s for s in sites if s.kind == PAIR and not s.partner_swap]),
            "by_letter": dict(Counter(
                f"{s.site.prescribed}→{s.h448_committed}" for s in sites if s.kind == PAIR
            ).most_common()),
        },
        "frame_populations": {
            name: {"carriers": stratum["population"], "reciters": len(stratum["per_reciter"])}
            for name, stratum in frame_strata.items()
            if name.startswith("pair:") or name.startswith("shaddah:held:")
        },
    }


def split_record(reciters: Iterable[int]) -> dict:
    """Every pool reciter's half under the proposed split and the alternative, with counts."""
    reciters = sorted(set(reciters))
    shares = {"proposed": HELD_OUT_SHARE, "alternative": ALTERNATIVE_HELD_OUT_SHARE}
    record = {
        "status": "proposal: the owner confirms the split before the first sitting",
        "salt": SPLIT_SALT,
        "algorithm": "train when the first 8 bytes (big-endian) of SHA-256 over the UTF-8 "
                     "'<salt>:<reciter id in decimal>' are below round((1 - held-out share) "
                     "* 2**64), else held_out (acceptance rules §9)",
    }
    for name, share in shares.items():
        halves = {str(r): reciter_half(r, share) for r in reciters}
        record[name] = {
            "held_out_share": share,
            "cutoff": round((1 - share) * 2**64),
            "reciters": {half: sum(h == half for h in halves.values()) for half in HALVES},
            "halves": halves,
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("mine", help="mine Ship A's sites and write them with their sidecars")
    parser.parse_args()
    provenance, scored = load_scored_pool()
    frame = json.loads(FRAME_PATH.read_text(encoding="utf-8"))
    if frame["inputs"] != provenance["inputs"]:
        raise ValueError(f"{FRAME_PATH} was built from other inputs: rebuild it first")
    sites = mine(scored)
    write_sites(sites)
    summary = summarize(sites, frame["strata"], provenance)
    write_text_atomically(SITES_SUMMARY_PATH, json.dumps(summary, indent=1, ensure_ascii=False,
                                                         sort_keys=True) + "\n")
    split = split_record(clip.reciter_id for clip in load_manifest(POOL_CLIPS_PATH))
    write_text_atomically(SPLIT_PATH, json.dumps(split, indent=1, sort_keys=True) + "\n")
    for name, counts in summary["strata"].items():
        print(f"{name:46s} {counts['sites']:4d} sites {counts['reciters']:4d} reciters  "
              f"base right {counts['base_already_right']:4d}  held out "
              f"{counts[HELD_OUT]['sites']:4d}")


if __name__ == "__main__":
    main()
