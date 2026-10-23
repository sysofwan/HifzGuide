"""Fold #61's site verdicts into Ship A's truth sites (#90).

The owner answered every row of Ship A's worklist (#61). This writes the answered sites as
truth sites, so the #84 scorer (:mod:`training.truth_baseline`) and #55's per-side report read
one truth:

* ``truth_sites/ship_a_verified.jsonl``: every ``new_audit`` row of ``ship_a/worklist.jsonl``
  in the truth-site schema (:mod:`tadabur.truth_sites`), ``heard`` set to the owner's verdict
  (:func:`tadabur.listening_session.adjudicated`). That is Ship A's listened sites
  (:mod:`tadabur.ship_a`) and the reject-pile consonant sites #87 drew that are no Ship A site;
* ``ship_a/verified_design.jsonl``: per site, its **frame** (``ship_a``: a site
  ``h448`` flagged; ``reject_pile``: one the base teacher heard as the partner, #87), its
  canonical reciter's **half** of the reciter split (:func:`tadabur.ship_a.reciter_half`, D11),
  and its **chance of being on the worklist**, whose inverse is its design weight (acceptance
  rules §1 as amended 2026-10-10: a Ship A site its ``listen_probability``, a reject-pile site
  the within-stratum draw #87 made, so it keeps its stratum weight).

The 23 P3.5 re-listens keep their own ids and file (``truth_sites/p35_fixtures.jsonl``); every
reader applies the verdicts to them at load, as :mod:`tadabur.eval_harness` does. The word
sample's verdicts are read by :func:`tadabur.word_sample.verified_words`.

**Every answered site is folded**, whatever its note. What leaves evaluation is decided at use
by the shared exclusion (:mod:`tadabur.exclusions`), so a site whose note denies the recitation
is Hafs keeps its verdict here and is taken out as ``not_hafs`` wherever it is scored.

Usage (from ``tools/``; torch-free, a second)::

  python -m tadabur.ship_a_verified build
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from training.tashkeel_eval import write_text_atomically

from .listening_session import VERDICTS_PATH, Verdict, adjudicated, load_worklist, read_verdicts
from .ship_a import HALVES, SHIP_A_DIR, assert_verdict_sites_kept, load_sites, reciter_half
from .staged_audio import load_staged_clips
from .truth_sites import NEW_AUDIT, TRUTH_SITES_DIR, TruthSite, load_truth_sites, write_truth_sites

SITES_PATH = TRUTH_SITES_DIR / "ship_a_verified.jsonl"
DESIGN_PATH = SHIP_A_DIR / "verified_design.jsonl"
WORKLIST_PATH = SHIP_A_DIR / "worklist.jsonl"

SHIP_A = "ship_a"
REJECT_PILE = "reject_pile"
FRAMES = (SHIP_A, REJECT_PILE)
#: The stratum suffix of #87's reject-pile consonant sites (``listening_session``).
REJECT_PILE_STRATUM = ":base_partner"


@dataclass(frozen=True)
class Design:
    """How one verified site came to be listened to (module docstring)."""

    site_id: str
    frame: str
    half: str
    listen_probability: float

    def __post_init__(self) -> None:
        if self.frame not in FRAMES or self.half not in HALVES:
            raise ValueError(f"{self.site_id}: frame {self.frame!r} or half {self.half!r}")
        if type(self.listen_probability) is not float or not 0 < self.listen_probability <= 1:
            raise ValueError(f"{self.site_id}: listen_probability is not a float in (0, 1]")

    @property
    def weight(self) -> float:
        return 1 / self.listen_probability


def fold(
    rows: Iterable[TruthSite], ship_a: Mapping[str, float], draws: Mapping[str, float],
    reciter_of: Mapping[str, int], verdicts: Mapping[str, Verdict],
) -> tuple[list[TruthSite], list[Design]]:
    """The answered ``new_audit`` worklist rows as truth sites, sorted by id, and each one's
    design. ``ship_a`` gives each listened Ship A site's listening probability, ``draws`` each
    row's within-stratum draw probability; every row must have a verdict."""
    sites = sorted((row for row in rows if row.source == NEW_AUDIT), key=lambda s: s.site_id)
    missing = [s.site_id for s in sites if s.site_id not in verdicts]
    if missing:
        raise ValueError(f"{len(missing)} worklist site(s) have no verdict, e.g. {missing[0]}")
    designs = []
    for site in sites:
        if site.site_id in ship_a:
            frame, probability = SHIP_A, ship_a[site.site_id]
        elif site.stratum.endswith(REJECT_PILE_STRATUM):
            frame, probability = REJECT_PILE, draws[site.site_id]
        else:
            raise ValueError(f"{site.site_id}: neither a listened Ship A site nor a reject-pile one")
        half = reciter_half(reciter_of[site.audio_filename])
        designs.append(Design(site.site_id, frame, half, probability))
    return adjudicated(sites, verdicts), designs


def write_designs(designs: list[Design], path: Path = DESIGN_PATH) -> None:
    write_text_atomically(path, "".join(
        json.dumps(asdict(d), ensure_ascii=False, sort_keys=True) + "\n" for d in designs))


def load_designs(path: Path = DESIGN_PATH) -> dict[str, Design]:
    """Every verified site's design, by site id."""
    with open(path, encoding="utf-8") as f:
        designs = [Design(**json.loads(raw)) for raw in f if raw.strip()]
    by_id = {d.site_id: d for d in designs}
    if len(by_id) != len(designs):
        raise ValueError(f"{path}: two designs share a site id")
    return by_id


def load_verified(sites_path: Path = SITES_PATH, design_path: Path = DESIGN_PATH
                  ) -> tuple[list[TruthSite], dict[str, Design]]:
    """The verified sites and their designs, one design per site."""
    sites, designs = load_truth_sites(sites_path), load_designs(design_path)
    if {s.site_id for s in sites} != set(designs):
        raise ValueError(f"{sites_path} and {design_path} name different sites")
    return sites, designs


def build(verdicts_path: Path = VERDICTS_PATH) -> tuple[list[TruthSite], list[Design]]:
    """The fold of the committed worklist, Ship A sites and verdicts (module docstring).

    A worklist row whose site a paused ta-marbuta correction removed (#130) keeps the
    listening probability the frozen worklist recorded for it: its verdict is still an
    answer, and the guard excludes it through :func:`tadabur.waqf_guard.realized_ids`."""
    from .ship_a_worklist import GEMINATE, PAIR_PARTNER, worklist_part

    rows = load_worklist(WORKLIST_PATH)
    sites = load_sites()
    ship_a = {s.site.site_id: s.listen_probability for s in sites if s.listened}
    for row in rows:
        if worklist_part(row) in (PAIR_PARTNER, GEMINATE):
            ship_a.setdefault(row.site.site_id, row.draw_probability)
    verdicts = read_verdicts(verdicts_path)
    assert_verdict_sites_kept(
        ((s.site.site_id, s.site.reference_index) for s in sites),
        ((r.site_id, r.reference_index) for r in load_truth_sites(SITES_PATH)
         if not r.stratum.endswith(REJECT_PILE_STRATUM)),
        verdicts)
    reciter_of = {name: clip.reciter_id for name, clip in load_staged_clips().items()}
    return fold([r.site for r in rows], ship_a, {r.site.site_id: r.draw_probability for r in rows},
                reciter_of, verdicts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="write the verified truth sites and their designs")
    parser.parse_args()
    sites, designs = build()
    write_truth_sites(sites, SITES_PATH)
    write_designs(designs)
    frames = {f: sum(d.frame == f for d in designs) for f in FRAMES}
    print(f"{len(sites)} verified sites ({frames}) -> {SITES_PATH.name}, {DESIGN_PATH.name}")


if __name__ == "__main__":
    main()
