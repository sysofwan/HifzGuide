"""#267's sets and the run's first artifact: G6 removal, the round-3 refusal, the erratum.

* Ship A is the verified population minus the eight held-out G6 sites, and the decode plan
  leaves out every item that holds only a G6 site. Edits are all 300 items; qdat is decoded
  whole. Nothing else may be decoded: tlog look, every confirm tranche, qdat_bench, QUD's other
  half and Tadabur dev are round-3 inputs and are refused before any file is opened.
* The teacher's letter-level bars are computed from the committed whole-item decodes, before
  FastConformer produces anything. The dated erratum announces them; the decode refuses to run
  without it, because the frozen rule's quoted Ship A figures were Muraja word grades, not the
  letter-level measure the rule defines.
"""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from collections.abc import Sequence

from tadabur.fastconformer_screen.gates import ScreenStop
from tadabur.fastconformer_screen.verdict import Rates

#: The only sets this screen may decode, and the measure each one feeds.
DECODABLE = {"ship-a": "Ship A's decisive mistake sites and its correct sites",
             "edits": "the 150 synthetic edit/decoy pairs",
             "qdat": "the reported word disagreement per 1,000 words"}
#: The run's first artifact: the dated erratum announcing the teacher's letter-level bars.
ERRATUM_PATH = Path(__file__).resolve().parents[3] / "docs" / "fastconformer-screen-erratum.md"


def require_decodable(name: str) -> None:
    """Refuse anything that is not ship-a, edits or qdat, naming it (round-3 inputs)."""
    if name not in DECODABLE:
        raise ScreenStop(
            f"{name!r} is not a decodable set: only {', '.join(sorted(DECODABLE))} may be "
            f"decoded; {name!r} is a round-3 input (#267)")


def decode_plan(name: str) -> list[str]:
    """The item keys one set decodes, in a deterministic order.

    qdat's keys are the committed base decode's keys (:func:`training.nonpro_eval.decodes_file`),
    the same file the report reads its passages from: the decode needs no scoring or training
    import (and so no ``edlib`` or ``quran-transcript``) just to list what it decodes.
    """
    require_decodable(name)
    if name == "ship-a":
        from training.truth_scorer import item_key

        return sorted({item_key(site) for site in ship_a_sites()})
    if name == "edits":
        return sorted(item["item_id"] for item in edit_items())
    from tadabur.nonpro_sources import QDAT
    from training.external_corpus import read_decode_file
    from training.nonpro_eval import decodes_file

    return sorted(read_decode_file(decodes_file(QDAT, "base")))


@lru_cache(maxsize=1)
def ship_a_sites() -> tuple:
    """The verified sites minus G6, in a deterministic order."""
    from tadabur.ship_a_verified import load_verified
    from training.consonant_readout import load_g6_sites

    held_out = set(load_g6_sites())
    return tuple(sorted((site for site in load_verified()[0] if site.site_id not in held_out),
                        key=lambda site: site.site_id))


def split_sides(sites: Sequence) -> tuple[list, list]:
    """The sites by the human verdict: mistakes and correct recitation, never pooled."""
    from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE, side

    mistakes = [site for site in sites if side(site) == MISTAKE_SIDE]
    correct = [site for site in sites if side(site) == CORRECT_SIDE]
    return mistakes, correct


@lru_cache(maxsize=1)
def edit_items() -> tuple:
    """The 300 committed edit items (150 edits and their 150 decoys)."""
    from tadabur.synthetic_edits import read_items

    return tuple(read_items())


def _percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator


@lru_cache(maxsize=1)
def located_ship_a_sites() -> tuple:
    """The Ship A sites step 2's text-only mapping locates, minus G6: the comparison set.

    Every column is scored on these sites, so an unlocated site cannot be a denominator
    difference between FastConformer and the models it is compared with.
    """
    from tadabur.fastconformer_screen.mapping import locate_site

    return tuple(site for site in ship_a_sites() if locate_site(site) is not None)


@lru_cache(maxsize=1)
def teacher_bars() -> Rates:
    """The teacher's four rates with #267's letter-level rule, on the located, non-G6 set.

    The Ship A rates read the teacher's committed ``stream_b0`` whole-item text through the
    same carrier rules as every other column; the edit rates are ``teacher_agrees``, which
    already reads the carrier. Computed from committed decodes, so it is pre-outcome.
    """
    from tadabur.fastconformer_screen.outcomes import is_flagged, student_carried
    from training.zipformer_readout import TEACHER, whole_item_texts

    sites = located_ship_a_sites()
    mistakes, correct = split_sides(sites)
    carried = student_carried(TEACHER, sites)
    texts = whole_item_texts("edits")[TEACHER]
    from tadabur.synthetic_edits import teacher_agrees

    items = edit_items()
    edits = [item for item in items if item["role"] == "edit"]
    decoys = [item for item in items if item["role"] == "decoy"]
    return Rates(
        mistakes_flagged=_percent(
            sum(is_flagged(carried[site.site_id], site.prescribed) for site in mistakes),
            len(mistakes)),
        correct_flagged=_percent(
            sum(is_flagged(carried[site.site_id], site.prescribed) for site in correct),
            len(correct)),
        edits_heard=_percent(
            sum(teacher_agrees(item, texts[item["item_id"]]) for item in edits), len(edits)),
        decoys_unchanged=_percent(
            sum(teacher_agrees(item, texts[item["item_id"]]) for item in decoys), len(decoys)),
    )


def require_erratum(rates: Rates, path: Path = ERRATUM_PATH) -> None:
    """Refuse to decode unless the dated erratum announces these bars verbatim."""
    path = Path(path)
    if not path.is_file():
        raise ScreenStop(
            f"no erratum at {path}: the teacher's letter-level bars are the run's first "
            f"artifact, and the decode refuses without them (#267)")
    text = path.read_text(encoding="utf-8")
    missing = [f"{getattr(rates, field.name):.1f}%" for field in fields(rates)
               if f"{getattr(rates, field.name):.1f}%" not in text]
    if missing:
        raise ScreenStop(
            f"erratum {path} does not announce the teacher's bars: missing {', '.join(missing)}")
