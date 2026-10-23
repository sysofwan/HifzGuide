"""Ship A's sites and its reciter split (#126): what the owner verifies for goals 1-2.

Ship A (ADR-0012) trains on sites the owner verifies by ear: where the shipped ``h448``
(streamed at b = 0, no bias) swaps a target pair's letters, or collapses an in-word
geminate. Training labels may be mined from a model's flags (ADR-0012, acceptance rules
§1); evaluation on them is stratified by that outcome, with known inclusion probabilities,
and read on held-out reciters only.

**Sites.** From the scored mining pool (:func:`tadabur.power_frame.score_pool`, the #84
scorer's outcomes of both frozen decodes) and from the clips Ship A's mining extension
found (:mod:`tadabur.ship_a_mining`):

* **pair**: every carrier of a target pair (:data:`tadabur.truth_sites.TARGET_PAIRS`:
  the six soft pairs, ``ذ↔ظ``, and the owner's ``س↔ش`` and ``ط↔ق``) where ``h448`` commits a
  consonant other than the mushaf letter, once per carrier. A carrier in two pairs (``ذ``,
  ``ظ``, ``س``, ``ط``, ``ق``) is asked as the pair whose partner ``h448`` committed, else as the
  first of its pairs in label order. Stratum ``ship_a:<prescribed>→<partner>:h448_partner``
  for a **partner swap**, ``ship_a:<prescribed>:h448_other`` for any other letter. Only the
  partner swaps are listened to (D9); the other letters are mined and kept but no target
  pair's question asks about them (``ت``→``ه`` at a paused ta marbuta is a reference
  problem, #130). The extension's clips were selected for a partner swap, so only those are
  taken from it (``ship_a_mining:…`` strata).
* **geminate**: every reference geminate inside one word that the pool's ``h448`` decodes
  as one consonant. Cross-word geminates are left out: the app never scores them (#86).

**Base already right** (ADR-0011 §3). Where the base teacher commits the mushaf value at a
flagged site, the training target is the teacher's own decode. Geminates the teacher holds
are not listened to (D10, ``ship_a:shaddah:held:h448_single:base_held``); the 115 it does
not hold are (``…:base_not_held``). Partner swaps are listened to whatever the teacher did.

**Reciter split** (acceptance rules §9; owner-confirmed 2026-10-10, D11). A canonical
reciter is in the ``train`` half when :func:`training.acceptance_stats.reciter_hash` under
:data:`SPLIT_SALT` is below 2**63 (the bias split's cutoff), else ``held_out``. ``split.json``
lists every reciter of the pool and the extension.

Files (``tools/tadabur/ship_a/``): ``sites.jsonl`` (one site per line: the worklist row of
:mod:`tadabur.listening_session`, with its provenance, plus the reciter, its half and what
each decode committed), ``sites.summary.json`` (the population sidecar: per stratum and per
family, sites and reciters by half and the base teacher's outcome; the frame's carrier
populations; the inputs' checksums) and ``split.json``.

Usage (from ``tools/``; torch-free, needs ``quran-transcript``, ~4 min)::

  python -m tadabur.ship_a mine
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from training.acceptance_stats import reciter_hash
from training.site_outcomes import family
from training.tashkeel_eval import write_text_atomically

from .listening_session import SessionSite, parse_row, row_json
from .mining_pool import load_manifest
from .power_frame import (
    FRAME_PATH,
    INPUTS,
    MINING_POOL_FILES,
    SHIP_A_DIR,
    PoolFiles,
    ScoredSite,
    direction,
    input_checksums,
    load_scored_pool,
    partner,
)
from .truth_sites import (
    CONSONANTS,
    HELD,
    NOT_HELD,
    SHADDAH,
    TARGET_PAIRS,
    TruthSite,
    _check_file,
)

SITES_PATH = SHIP_A_DIR / "sites.jsonl"
SITES_SUMMARY_PATH = SHIP_A_DIR / "sites.summary.json"
SPLIT_PATH = SHIP_A_DIR / "split.json"

SPLIT_SALT = "ship-a-verified-split-2026-10"
TRAIN = "train"
HELD_OUT = "held_out"
HALVES = (TRAIN, HELD_OUT)
#: The train half's cutoff: 50/50, confirmed by the owner (D11).
SPLIT_CUTOFF = 2**63
#: How the listened sites are chosen (:func:`select_listening`).
SELECTION_SALT = "ship-a-listening-2026-10"
#: The ``كسب`` swaps asked (owner decision 2026-10-10 B: about 20); the rest take the
#: sample's verdict (``docs/ship-a-sizing.md``).
KASB_SAMPLE = 20
KASB_DIRECTION = "س→ش"
#: The ``كسب`` swaps' stratum (after the source's prefix): one sample is drawn across both.
KASB_STRATUM = "kasb:س→ش:h448_partner"
#: The verdict-bearing sites the owner authorized removing from the Ship A population
#: (#130 amendment, 2026-10-23: the owner re-listened to
#: ``tadabur_spk0167_S17_A15_6d312abb_000008.wav`` 18:15 «ءَالِهَةً لَّوْلَا» and heard the
#: waqf, so the pause before its shaddah makes that site waqf_realized):
#: https://github.com/sysofwan/HifzGuide/issues/130#issuecomment-6358608933. Every other
#: verdict-bearing site must keep its id and reference index through a rebuild.
OWNER_AUTHORIZED_REMOVALS = frozenset({
    "new_audit:tadabur_spk0167_S17_A15_6d312abb_000008.wav#0@63:shaddah",
})

#: The strata prefixes of the pool's sites and of the mining extension's.
POOL_PREFIX = "ship_a"
MINING_PREFIX = "ship_a_mining"

PAIR = "pair"
GEMINATE = "geminate"
#: The frame stratum Ship A's geminates come from (``tadabur.power_frame.frame_stratum``).
IN_WORD_GEMINATES = "shaddah:held:in_word"


def reciter_half(reciter_id: int) -> str:
    """``train`` or ``held_out`` for one canonical reciter id (module docstring)."""
    return TRAIN if reciter_hash(reciter_id, SPLIT_SALT) < SPLIT_CUTOFF else HELD_OUT


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
    #: The chance the site is on the worklist (:func:`select_listening`); 0 if it never is.
    listen_probability: float = 0.0

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
        """On the worklist (:func:`select_listening`)."""
        return self.listen_probability > 0

    @property
    def kasb(self) -> bool:
        """A ``س``→``ش`` swap on a ``كسب``-form carrier (D9 B: a sample of them is asked)."""
        return self.site.stratum.endswith(f":{KASB_STRATUM}")

    @property
    def partner_swap(self) -> bool:
        return self.kind == PAIR and self.h448_committed == partner(self.site)


def kasb_form(row: SessionSite) -> bool:
    """Whether a ``س`` carrier is the ``س`` of the consonants ``كسب`` in its word (كَسَبَت,
    يَكسِبُون, …): the root ``h448`` and the base teacher both hear as ``كشب`` (D9 B)."""
    site, offsets = row.site, row.word_offsets
    start = max(o for o in offsets if o <= site.reference_index)
    end = min(o for o in offsets if o > site.reference_index)
    letters = [i for i in range(start, end) if site.reference[i] in CONSONANTS]
    k = letters.index(site.reference_index)
    return "".join(site.reference[i] for i in letters[max(0, k - 1):k + 2]) == "كسب"


def ship_a_stratum(scored: ScoredSite, prefix: str) -> str:
    """The stratum a flagged site is mined into (module docstring): a partner swap's by the
    base teacher's outcome, the ``كسب`` swaps in their own."""
    site = scored.site
    if site.mark == SHADDAH:
        teacher = "base_held" if scored.base_mushaf else "base_not_held"
        return f"{prefix}:shaddah:held:h448_single:{teacher}"
    if scored.h448.committed != partner(site):
        return f"{prefix}:{site.prescribed}:h448_other"
    if direction(site) == KASB_DIRECTION and kasb_form(scored.row):
        return f"{prefix}:{KASB_STRATUM}"
    teacher = "base_right" if scored.base_mushaf else "base_wrong"
    return f"{prefix}:{direction(site)}:h448_partner:{teacher}"


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


def _rank(*parts) -> str:
    return hashlib.sha256(":".join((SELECTION_SALT, *map(str, parts))).encode()).hexdigest()


def select_listening(sites: list[ShipASite]) -> dict[str, float]:
    """Each listened site's chance of being on the worklist (owner decisions D10 and
    2026-10-10 A and B); a site absent from the result is not listened to.

    * a geminate the base teacher does not hold: always (D10);
    * a partner swap the base teacher gets wrong: always (its label needs the ear);
    * a partner swap the teacher gets right, on a **held-out** reciter: one per reciter and
      family, the first by salted rank, chance ``1 / k`` among that reciter's ``k`` (for
      evaluation). On the train half none: their target is the teacher's decode;
    * a ``كسب`` swap: :data:`KASB_SAMPLE` of them, reciters in salted order taking one site
      each per round (their sites by salted rank), so the sample spreads over reciters. A
      site's chance is its reciter's expected share over the salted orders
      (:func:`round_robin_shares`), divided by the reciter's sites.
    """
    chosen: dict[str, float] = {}
    teacher_right: dict[tuple, list[ShipASite]] = defaultdict(list)
    kasb: dict[int, list[ShipASite]] = defaultdict(list)
    for site in sites:
        if site.kasb:
            kasb[site.reciter_id].append(site)
        elif site.kind == GEMINATE or site.partner_swap:
            if not site.base_already_right:
                chosen[site.site.site_id] = 1.0
            elif site.partner_swap and site.half == HELD_OUT:
                teacher_right[(site.reciter_id, site.family)].append(site)
    for group in teacher_right.values():
        first = min(group, key=lambda s: _rank(s.site.site_id))
        chosen[first.site.site_id] = 1 / len(group)
    queues = {r: sorted(group, key=lambda s: _rank(s.site.site_id)) for r, group in kasb.items()}
    order = sorted(queues, key=lambda r: _rank("reciter", r))
    taken: dict[int, int] = defaultdict(int)
    for round_ in range(max(map(len, queues.values()), default=0)):
        for reciter in order:
            if sum(taken.values()) < KASB_SAMPLE and round_ < len(queues[reciter]):
                taken[reciter] += 1
    shares = round_robin_shares({r: len(q) for r, q in queues.items()}, KASB_SAMPLE)
    for reciter, m in taken.items():
        for site in queues[reciter][:m]:
            chosen[site.site.site_id] = shares[reciter] / len(queues[reciter])
    return chosen


def listening_weights(sites: Iterable[ShipASite]) -> dict[str, float]:
    """Each listened site's weight in any Ship A estimate: the inverse of its listening
    probability (1 where its stratum is listened to whole). The estimand is the frozen
    verified site set (owner decision, 2026-10-10), so no weight expands the pool or the
    extension to a wider frame."""
    return {s.site.site_id: 1 / s.listen_probability for s in sites if s.listened}


def round_robin_shares(sizes: Mapping[int, int], k: int) -> dict[int, float]:
    """Each group's expected number of picks when ``k`` are taken one per group per round,
    groups in a uniformly random order (a salted hash): every full round gives each group
    with a site left one more; the ``R`` picks of the last, partial round go to ``R`` of its
    ``E`` eligible groups, each with chance ``R / E``. Exact; ``k`` at or above the total
    takes everything."""
    full = 0
    while full < max(sizes.values(), default=0) and \
            sum(min(full + 1, n) for n in sizes.values()) <= k:
        full += 1
    left = k - sum(min(full, n) for n in sizes.values())
    eligible = [g for g, n in sizes.items() if n > full]
    extra = min(1.0, left / len(eligible)) if eligible else 0.0
    return {g: min(full, n) + (extra if n > full else 0.0) for g, n in sizes.items()}


def mine_all(pool: Iterable[ScoredSite], extension: Iterable[ScoredSite] = ()) -> list[ShipASite]:
    """The pool's flagged sites and the extension's partner swaps (module docstring), each in
    its stratum with that stratum's population and its chance of being listened to, by id."""
    flagged = [*((one, POOL_PREFIX) for one in flagged_sites(pool)),
               *((one, MINING_PREFIX) for one in flagged_sites(extension)
                 if one.site.mark in TARGET_PAIRS and one.h448.committed == partner(one.site))]
    strata = [ship_a_stratum(one, prefix) for one, prefix in flagged]
    population = Counter(strata)
    sites = [
        ShipASite(
            row=replace(one.row, site=replace(one.site, stratum=stratum,
                                              stratum_population=population[stratum])),
            reciter_id=one.reciter_id,
            half=reciter_half(one.reciter_id),
            h448_committed=one.h448.committed,
            base_committed=one.base.committed,
        )
        for (one, _), stratum in zip(flagged, strata)
    ]
    chances = select_listening(sites)
    return sorted((replace(s, listen_probability=chances.get(s.site.site_id, 0.0)) for s in sites),
                  key=lambda s: s.site.site_id)



# --- the files -------------------------------------------------------------------------------

_SITE_FIELDS = ("reciter_id", "half", "h448_committed", "base_committed", "listen_probability")


def site_json(site: ShipASite) -> dict:
    return {**row_json(site.row), **{name: getattr(site, name) for name in _SITE_FIELDS}}


def parse_ship_a_site(data: dict, where: str) -> ShipASite:
    """One validated row: a worklist row plus the Ship A fields, its half recomputed."""
    extra = {name: data.get(name) for name in _SITE_FIELDS}
    row = parse_row({k: v for k, v in data.items() if k not in _SITE_FIELDS}, where)
    site = ShipASite(row=row, **extra)
    if type(site.reciter_id) is not int or site.half != reciter_half(site.reciter_id):
        raise ValueError(f"{where}: half {site.half!r} is not reciter {site.reciter_id}'s")
    if site.kind == GEMINATE and (site.site.prescribed, site.h448_committed) != (HELD, NOT_HELD):
        raise ValueError(f"{where}: a Ship A geminate is a held one h448 decoded single")
    if site.kind == PAIR and site.h448_committed in (None, site.site.prescribed):
        raise ValueError(f"{where}: a Ship A pair site is one h448 committed another letter at")
    if type(site.listen_probability) is not float or not 0 <= site.listen_probability <= 1:
        raise ValueError(f"{where}: listen_probability is not a float in [0, 1]")
    return site


def write_sites(sites: list[ShipASite], path: Path = SITES_PATH) -> None:
    """Validate every row and the file, then write atomically, one JSON object per line."""
    for number, site in enumerate(sites, 1):
        parse_ship_a_site(site_json(site), f"write_sites[{number}]")
    _check_file([site.site for site in sites], "write_sites")
    write_text_atomically(path, "".join(
        json.dumps(site_json(site), ensure_ascii=False, sort_keys=True) + "\n" for site in sites
    ))


def assert_verdict_sites_kept(current: Iterable[tuple[str, int]],
                              before: Iterable[tuple[str, int]],
                              verdict_ids: Iterable[str]) -> None:
    """The rebuild preflight: fail before artifacts are written if a verdict-bearing site
    ``before`` (id and reference index) is dropped or moved, less
    :data:`OWNER_AUTHORIZED_REMOVALS`. ``current`` is the rebuilt set's (id, index)."""
    now = dict(current)
    verdicts = set(verdict_ids)
    moved = sorted(i for i, index in before
                   if i in verdicts and i not in OWNER_AUTHORIZED_REMOVALS
                   and now.get(i) != index)
    if moved:
        raise ValueError(f"{len(moved)} verdict-bearing site(s) would move or disappear, "
                         f"e.g. {moved[0]}")


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


def carrier_word(site: ShipASite) -> str:
    """The realized word holding the site's carrier."""
    offsets, index = site.row.word_offsets, site.site.reference_index
    start = max(o for o in offsets if o <= index)
    end = min(o for o in offsets if o > index)
    return site.site.reference[start:end].strip()


def swap_checks(sites: list[ShipASite]) -> dict:
    """Per partner-swap direction, what could make its count misleading: how many reciters
    and ayat hold it, the most any one reciter holds, how often the base teacher commits the
    mushaf letter there, and the commonest carrier word with its share."""
    checks = {}
    for name, group in sorted(_by([s for s in sites if s.partner_swap],
                                  lambda s: f"{s.site.prescribed}→{s.h448_committed}").items()):
        word, count = Counter(map(carrier_word, group)).most_common(1)[0]
        checks[name] = {
            "sites": len(group),
            "reciters": len({s.reciter_id for s in group}),
            "max_per_reciter": Counter(s.reciter_id for s in group).most_common(1)[0][1],
            "ayahs": len({s.site.surah_ayah for s in group}),
            "base_already_right": sum(s.base_already_right for s in group),
            "top_word": word,
            "top_word_sites": count,
        }
    return checks


def listening_counts(sites: list[ShipASite]) -> dict[str, dict[str, int]]:
    """How :func:`select_listening` treats each kind of site: how many there are and how
    many are listened to."""
    def kind(site: ShipASite) -> str:
        if site.kind == GEMINATE:
            return "geminate, teacher holds" if site.base_already_right else "geminate"
        if not site.partner_swap:
            return "other letter"
        if site.kasb:
            return "كسب swap"
        if not site.base_already_right:
            return "swap, teacher wrong"
        return f"swap, teacher right, {site.half}"

    return {name: {"sites": len(group), "listened": sum(s.listened for s in group)}
            for name, group in sorted(_by(sites, kind).items())}


def _by(sites: list[ShipASite], key) -> dict[str, list[ShipASite]]:
    groups: dict[str, list[ShipASite]] = defaultdict(list)
    for site in sites:
        groups[key(site)].append(site)
    return groups


def summarize(sites: list[ShipASite], frame_strata: dict, provenance: dict) -> dict:
    """The population sidecar: per Ship A stratum and per family, the counts of
    :func:`_counts`; the pool's pair commits (partner swaps, other letters, by letter) and
    the extension's families; the frame's carrier populations per stratum; the inputs."""
    def grouped(key) -> dict:
        return {name: _counts(group) for name, group in sorted(_by(sites, key).items())}

    pool = [s for s in sites if s.site.stratum.startswith(f"{POOL_PREFIX}:")]
    return {
        **provenance,
        "split": {"salt": SPLIT_SALT, "cutoff": SPLIT_CUTOFF, "status": "confirmed (D11)"},
        "strata": grouped(lambda s: s.site.stratum),
        "families": grouped(lambda s: s.family),
        "pool_pair_commits": {
            "partner": _counts([s for s in pool if s.partner_swap]),
            "other_letter": _counts([s for s in pool if s.kind == PAIR and not s.partner_swap]),
            "by_letter": dict(Counter(
                f"{s.site.prescribed}→{s.h448_committed}" for s in pool if s.kind == PAIR
            ).most_common()),
        },
        "partner_swap_checks": swap_checks(sites),
        "listening": listening_counts(sites),
        "extension_families": {
            name: _counts(group) for name, group in sorted(_by(
                [s for s in sites if s.site.stratum.startswith(f"{MINING_PREFIX}:")],
                lambda s: s.family).items())
        },
        "frame_populations": {
            name: {"carriers": stratum["population"], "reciters": len(stratum["per_reciter"])}
            for name, stratum in frame_strata.items()
            if name.startswith("pair:") or name.startswith("shaddah:held:")
        },
    }


def split_record(reciters: Iterable[int]) -> dict:
    """Every reciter's half under the confirmed split, with counts."""
    halves = {str(r): reciter_half(r) for r in sorted(set(reciters))}
    return {
        "status": "confirmed by the owner, 2026-10-10 (D11)",
        "salt": SPLIT_SALT,
        "cutoff": SPLIT_CUTOFF,
        "algorithm": "train when the first 8 bytes (big-endian) of SHA-256 over the UTF-8 "
                     "'<salt>:<reciter id in decimal>' are below the cutoff, else held_out "
                     "(acceptance rules §9)",
        "reciters": {half: sum(h == half for h in halves.values()) for half in HALVES},
        "halves": halves,
    }


def inputs(extension: PoolFiles) -> dict[str, str]:
    """The checksums of the pool's and the extension's committed files."""
    return {**input_checksums(), **input_checksums(extension.inputs("ship_a/mining"))}


def main() -> None:
    from .ship_a_mining import FILES as EXTENSION_FILES

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("mine", help="mine Ship A's sites and write them with their sidecars")
    parser.parse_args()
    frame = json.loads(FRAME_PATH.read_text(encoding="utf-8"))
    if frame["inputs"] != input_checksums(INPUTS):
        raise ValueError(f"{FRAME_PATH} was built from other inputs: rebuild it first")
    pool_provenance, pool = load_scored_pool(MINING_POOL_FILES)
    extension_provenance, extension = load_scored_pool(EXTENSION_FILES)
    sites = mine_all(pool, extension)
    write_sites(sites)
    provenance = {"inputs": inputs(EXTENSION_FILES), "pool": pool_provenance,
                  "extension": extension_provenance}
    summary = summarize(sites, frame["strata"], provenance)
    write_text_atomically(SITES_SUMMARY_PATH, json.dumps(summary, indent=1, ensure_ascii=False,
                                                         sort_keys=True) + "\n")
    clips = [*load_manifest(MINING_POOL_FILES.clips),
             *load_manifest(EXTENSION_FILES.clips, None, EXTENSION_FILES.use,
                            EXTENSION_FILES.strata)]
    split = split_record(clip.reciter_id for clip in clips)
    write_text_atomically(SPLIT_PATH, json.dumps(split, indent=1, sort_keys=True) + "\n")
    for name, counts in summary["strata"].items():
        print(f"{name:50s} {counts['sites']:4d} sites {counts['reciters']:4d} reciters  "
              f"base right {counts['base_already_right']:4d}  held out "
              f"{counts[HELD_OUT]['sites']:4d}")


if __name__ == "__main__":
    main()
