"""Ship A's planning frame: what the two frozen pool decodes commit, per site (#126).

Ship A (ADR-0012, goals 1-2) verifies by ear the sites where the shipped ``h448`` commits the
wrong letter at a target-pair carrier or collapses an in-word geminate, and trains on the
verified labels. Its sizing (:mod:`training.ship_a_sizing`) needs, for every stratum of the
mining pool, **who** holds its sites and **what each decode commits there**. This module
reads both off the committed pool decodes, torch-free and without any app model: every
field is a **decode-level** fact about one site, read by the #84 scorer's own rules
(:func:`training.site_outcomes.item_outcomes`), never a Muraja grade.

* The eligible sites are #87's (:func:`tadabur.listening_session.pool_candidates`): every
  mid-word haraka and prescribed sukun, every reference geminate and single consonant, and
  every carrier of a target-pair letter, per pair. Eligibility reads the realized reference
  alone.
* A frame **stratum** depends on the reference alone too: ``pair:<prescribed>→<partner>``
  per direction (a ``ذ`` or ``ظ`` carrier sits in two pairs, so in two directions),
  ``shaddah:held:in_word`` / ``shaddah:held:cross_word`` / ``shaddah:not_held``, and
  ``tashkeel:<mark>``. Cross-word geminates are counted apart because the app never scores
  them (#86).
* The two decodes are ``h448`` streamed at b = 0 with no bias (today's model,
  ``mining_pool/h448_stream_decodes.json``) and the base teacher's whole-span decode
  (``mining_pool/base_decodes.json``). No verdict exists, so outcomes are read against the
  **mushaf** (what the site prescribes), never against what was heard.

Per stratum the frame keeps its population and, per canonical reciter, the sums of
:data:`FIELDS`: sites; ``h448`` commits, commits of the mushaf value, commits of the pair
partner (pair sites), one consonant at a held geminate (shaddah held sites), an empty slot
at a prescribed sukun or haraka (tashkeel sites; goal 3, counted only); the base teacher's
commits and mushaf commits; sites where the base committed what ``h448`` committed; sites
``h448`` flags (commits something other than the mushaf value) where the base commits the
mushaf value (those need no listening, ADR-0011 §3); and the listening seconds of #87's
model (:func:`tadabur.listening_session.listening_seconds`).

Usage (from ``tools/``; needs ``quran-transcript`` for the Uthmani words, ~3 min)::

  python -m tadabur.power_frame
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from training.site_outcomes import SiteOutcome, family, item_outcomes
from training.tashkeel_eval import write_text_atomically

from .listening_session import (
    H448_DECODES_PATH,
    POOL_CLIPS_PATH,
    POOL_DECODES_PATH,
    SessionSite,
    census,
    crosses_words,
    listening_seconds,
    mode_of,
    pool_candidates,
    uthmani_tanween,
)
from .mining_pool import PoolClip, load_base_decodes, load_manifest, segment_key
from .pool_stream import load_stream_decodes
from .staged_audio import StagedClip, load_staged_clips
from .truth_sites import HARAKA_CHARS, HELD, NOT_HELD, SHADDAH, SUKUN, TARGET_PAIRS, TruthSite

SHIP_A_DIR = Path(__file__).parent / "ship_a"
FRAME_PATH = SHIP_A_DIR / "frame.json"

FIELDS = (
    "sites",
    "h448_commit",
    "h448_mushaf",
    "h448_partner",
    "h448_single_on_held",
    "h448_empty_on_sukun",
    "h448_empty_on_haraka",
    "base_commit",
    "base_mushaf",
    "base_agrees_with_h448",
    "base_mushaf_where_h448_flagged",
    "listen_seconds",
)

#: The committed inputs the frame (and everything mined beside it) is a function of.
INPUTS = {
    "mining_pool/clips.jsonl": POOL_CLIPS_PATH,
    "mining_pool/base_decodes.json": POOL_DECODES_PATH,
    "mining_pool/h448_stream_decodes.json": H448_DECODES_PATH,
}


def input_checksums(paths: Mapping[str, Path] = INPUTS) -> dict[str, str]:
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def partner(site: TruthSite) -> str:
    """The other letter of a pair site's pair."""
    (other,) = set(site.mark.split("↔")) - {site.prescribed}
    return other


def direction(site: TruthSite) -> str:
    """``<prescribed>→<partner>`` of a pair site."""
    return f"{site.prescribed}→{partner(site)}"


@dataclass(frozen=True)
class ScoredSite:
    """One eligible pool site: its census row (truth-site skeleton, design, excerpt), its
    clip's canonical reciter, its frame stratum, and what each frozen decode committed."""

    row: SessionSite
    reciter_id: int
    stratum: str
    h448: SiteOutcome
    base: SiteOutcome

    @property
    def site(self) -> TruthSite:
        return self.row.site

    @property
    def h448_flagged(self) -> bool:
        """``h448`` committed something other than the mushaf value (F, decode level)."""
        return self.h448.flagged(self.site)

    @property
    def base_mushaf(self) -> bool:
        return self.base.committed == self.site.prescribed


def frame_stratum(row: SessionSite) -> str:
    """The site's stratum in the frame, a function of its reference alone (module doc)."""
    site = row.site
    if site.mark in TARGET_PAIRS:
        return f"pair:{direction(site)}"
    if site.mark == SHADDAH and site.prescribed == HELD:
        where = crosses_words(site.reference, list(row.word_offsets), site.reference_index)
        return f"shaddah:held:{'cross_word' if where else 'in_word'}"
    if site.mark == SHADDAH:
        return "shaddah:not_held"
    return f"tashkeel:{site.mark}"


def score_pool(
    clips: list[PoolClip],
    base_decodes: Mapping[str, str],
    h448_decodes: Mapping[str, str],
    registry: Mapping[str, StagedClip],
    tanween_of: Callable[[str], list[bool]],
) -> list[ScoredSite]:
    """Every eligible site of the pool's kept segments with both decodes' outcomes, in
    segment-key order."""
    segment_of = {
        (clip.audio_filename, segment.start_sample): segment_key(clip.audio_filename,
                                                                 segment.segment_index)
        for clip in clips
        for segment in clip.segments
    }
    reciter_of = {clip.audio_filename: clip.reciter_id for clip in clips}
    rows_by_segment: dict[str, list[SessionSite]] = defaultdict(list)
    for row in census(pool_candidates(clips, base_decodes, h448_decodes, registry, tanween_of)):
        rows_by_segment[segment_of[(row.site.audio_filename, row.site.start_sample)]].append(row)
    scored = []
    for key in sorted(rows_by_segment):
        rows = rows_by_segment[key]
        sites = [row.site for row in rows]
        h448 = item_outcomes(sites, h448_decodes[key])
        base = item_outcomes(sites, base_decodes[key])
        scored += [
            ScoredSite(row, reciter_of[row.site.audio_filename], frame_stratum(row),
                       h448[row.site.site_id], base[row.site.site_id])
            for row in rows
        ]
    return scored


def site_fields(scored: ScoredSite) -> tuple[float, ...]:
    """One site's contribution to each of :data:`FIELDS`."""
    site, h448, base = scored.site, scored.h448, scored.base
    empty = h448.carrier_aligned and not h448.commits
    return (
        1,
        h448.commits,
        h448.committed == site.prescribed,
        site.mark in TARGET_PAIRS and h448.committed == partner(site),
        site.mark == SHADDAH and site.prescribed == HELD and h448.committed == NOT_HELD,
        site.mark == SUKUN and empty,
        site.mark in HARAKA_CHARS and empty,
        base.commits,
        scored.base_mushaf,
        h448.commits and base.committed == h448.committed,
        scored.h448_flagged and scored.base_mushaf,
        listening_seconds(scored.row),
    )


def build_frame(scored: list[ScoredSite]) -> dict[str, dict]:
    """Per frame stratum: its mode, family, prescribed value, population and per-reciter
    sums of :data:`FIELDS` (counts as ints, seconds to 0.01 s)."""
    strata: dict[str, dict] = {}
    for one in scored:
        site = one.site
        stratum = strata.setdefault(one.stratum, {
            "mode": mode_of(site.mark),
            "family": family(site.mark),
            "prescribed": site.prescribed,
            "population": 0,
            "per_reciter": {},
        })
        stratum["population"] += 1
        sums = stratum["per_reciter"].setdefault(one.reciter_id, [0] * len(FIELDS))
        for i, value in enumerate(site_fields(one)):
            sums[i] += value
    for stratum in strata.values():
        stratum["per_reciter"] = {
            str(reciter): [*map(int, sums[:-1]), round(sums[-1], 2)]
            for reciter, sums in sorted(stratum["per_reciter"].items())
        }
    return dict(sorted(strata.items()))


def load_scored_pool() -> tuple[dict, list[ScoredSite]]:
    """The committed pool scored by both decodes, with the inputs' provenance."""
    registry = load_staged_clips()
    clips = load_manifest(POOL_CLIPS_PATH, registry)
    base_fingerprint, base_decodes = load_base_decodes(POOL_DECODES_PATH)
    h448_provenance, h448_decodes = load_stream_decodes(H448_DECODES_PATH)
    provenance = {
        "inputs": input_checksums(),
        "base_decode": base_fingerprint,
        "h448_decode": h448_provenance,
    }
    return provenance, score_pool(clips, base_decodes, h448_decodes, registry,
                                  uthmani_tanween())


def main() -> None:
    provenance, scored = load_scored_pool()
    frame = {
        **provenance,
        "scored_against": "the mushaf (prescribed) value at every site; no verdict exists",
        "fields": list(FIELDS),
        "strata": build_frame(scored),
    }
    write_text_atomically(FRAME_PATH, json.dumps(frame, ensure_ascii=False, sort_keys=True) + "\n")
    for name, stratum in frame["strata"].items():
        totals = [sum(v[i] for v in stratum["per_reciter"].values()) for i in range(len(FIELDS))]
        flagged = totals[FIELDS.index("h448_commit")] - totals[FIELDS.index("h448_mushaf")]
        print(f"{name:28s} N={stratum['population']:6d} reciters={len(stratum['per_reciter']):4d} "
              f"h448_flagged={flagged:5d}")


if __name__ == "__main__":
    main()
