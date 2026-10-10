"""The owner's Ship A listening for #61 (#126): one worklist, frozen by checksum.

The blind UI (:mod:`tadabur.tashkeel_audit_ui`) asks, in one queue:

* every Ship A **partner swap** (:mod:`tadabur.ship_a`): a target-pair carrier where
  ``h448`` commits the pair's other letter, from the mining pool and the mining extension,
  both halves of the reciter split (other letters are not asked, D9);
* every Ship A **geminate** the base teacher does not hold (the ones it holds take the
  teacher's decode as target and need no listening, ADR-0011 §3, D10);
* the **23 P3.5 re-listens** (acceptance rules §8: the nominal rejects, re-adjudicated at
  site level), as #87 drew them;
* the **reject-pile** consonant sites #87 drew (acceptance rules §8 includes them, and they
  are the evidence for the mix of real mistakes), less any that are Ship A sites already;
* Ship A's **word sample** (:mod:`tadabur.word_sample`), the model-independent denominator;
* and, unless left out at launch, the synthetic-edit check (#107), which the UI always
  queues with the session.

The P3.5 and reject-pile rows are #87's own (``listening_session/worklist.jsonl``), so their
ids, strata, inclusion probabilities and excerpts carry over unchanged. A sampled word may
be a word a site asks about in another recitation; the page shows it whole, so the words
are asked after every other row (:func:`tadabur.tashkeel_audit_ui.queue`).

**Sittings.** The UI asks in a fixed order (:func:`tadabur.tashkeel_audit_ui.queue`: the
sites and edit-check items in one seeded order, then the words) and resumes at the first
unanswered row, so a sitting is a run of the queue:
``worklist.summary.json`` cuts it into the fewest runs of about equal minutes, none longer
than :data:`SITTING_MINUTES`, by #87's
listening model (each excerpt twice plus 4 s; a word twice plus
:data:`tadabur.word_sample.ANSWER_SECONDS`).

**Frozen.** The summary records the SHA-256 of ``worklist.jsonl`` and of
``word_sample.jsonl``; a test checks both still match.

**Exposure.** The build writes the reciter split's two uses to the exposure registry
(acceptance rules §6): ``ship_a.verified_train`` holds the recordings of every train-half
Ship A site (listened, or the teacher's target), ``ship_a.verified_held_out`` those of every
held-out site and sampled word.

Usage (from ``tools/``; torch-free)::

  python -m tadabur.ship_a_worklist build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path

from training.tashkeel_eval import write_text_atomically

from .edit_check import BLIND_CHECK_PATH, load_edit_check
from .listening_session import (
    WORKLIST_PATH as SESSION_WORKLIST_PATH,
    SessionSite,
    listening_seconds,
    load_worklist,
    write_worklist,
)
from .exposure import SHIP_A_HELD_OUT, SHIP_A_TRAIN, Exposure, whole_recording, write_use
from .ship_a import (
    HALVES,
    HELD_OUT,
    SHIP_A_DIR,
    SITES_PATH,
    TRAIN,
    ShipASite,
    load_sites,
    reciter_half,
)
from .staged_audio import StagedClip, load_staged_clips
from .tashkeel_audit_ui import queue, target_of
from .word_sample import WORD_SAMPLE_PATH, WordSite, load_words

WORKLIST_PATH = SHIP_A_DIR / "worklist.jsonl"
SUMMARY_PATH = SHIP_A_DIR / "worklist.summary.json"

#: The longest sitting the queue is cut into, in listening minutes.
SITTING_MINUTES = 75

PAIR_PARTNER = "pair_partner"
GEMINATE = "geminate"
P35 = "p35_relisten"
REJECT_PILE = "reject_pile"
WORDS = "word_sample"
EDIT_CHECK = "edit_check"
PARTS = (PAIR_PARTNER, GEMINATE, P35, REJECT_PILE, WORDS, EDIT_CHECK)

#: #87's strata the session carries over: the P3.5 re-listens and the reject-pile sites.
P35_PREFIX = "p35_fixture:"
REJECT_PILE_SUFFIX = ":base_partner"


def ship_a_part(site: ShipASite) -> str:
    """A listened Ship A site is a geminate or a partner swap."""
    return GEMINATE if site.kind == GEMINATE else PAIR_PARTNER


def carried_part(row: SessionSite) -> str | None:
    """Which carried-over part an #87 row is in, if any."""
    if row.site.stratum.startswith(P35_PREFIX):
        return P35
    return REJECT_PILE if row.site.stratum.endswith(REJECT_PILE_SUFFIX) else None


def assemble(
    sites: Iterable[ShipASite], session_rows: Iterable[SessionSite]
) -> tuple[list[SessionSite], dict[str, str], int]:
    """The worklist's site rows (Ship A's first, then #87's carried over, each id once), the
    part of each row by id, and how many carried rows were Ship A sites already."""
    parts = {s.site.site_id: ship_a_part(s) for s in sites if s.listened}
    rows = [s.row for s in sites if s.listened]
    duplicates = 0
    for row in session_rows:
        part = carried_part(row)
        if part is None:
            continue
        if row.site.site_id in parts:
            duplicates += 1
            continue
        parts[row.site.site_id] = part
        rows.append(row)
    return rows, parts, duplicates


def shared_words(rows: Iterable, words: Iterable[WordSite]) -> int:
    """How many sampled words another row asks about too (the same ayah and word, any
    recitation): the page shows a sampled word whole, so the words are asked last."""
    asked = {target_of(row).word for row in rows}
    return sum((w.site.surah_ayah, w.site.word_index) in asked for w in words)


def lean_swaps(sites: Iterable[ShipASite]) -> list[ShipASite]:
    """A smaller set of partner swaps to listen to, for the owner to weigh: every swap the
    base teacher gets wrong (its training label needs the ear), plus, for evaluation, one
    swap the teacher gets right per held-out reciter and family (the first by site id).
    The train half's teacher-right swaps would take the teacher's decode (ADR-0011 §3)."""
    chosen, seen = [], set()
    for site in sorted((s for s in sites if s.partner_swap), key=lambda s: s.site.site_id):
        if not site.base_already_right:
            chosen.append(site)
        elif site.half == HELD_OUT and (site.family, site.reciter_id) not in seen:
            seen.add((site.family, site.reciter_id))
            chosen.append(site)
    return chosen


def split_exposures(
    sites: Iterable[ShipASite], words: Iterable[WordSite], registry: dict[str, StagedClip],
) -> dict[str, list[Exposure]]:
    """Each half's recordings (module docstring), whole, with their staged checksums."""
    names = {TRAIN: set(), HELD_OUT: set()}
    for site in sites:
        names[site.half].add(site.site.audio_filename)
    for word in words:
        names[reciter_half(word.site.reciter_id)].add(word.site.audio_filename)
    return {use: sorted(whole_recording(registry[n], registry[n].audio_sha256) for n in names[half])
            for use, half in ((SHIP_A_TRAIN, TRAIN), (SHIP_A_HELD_OUT, HELD_OUT))}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seconds(row) -> float:
    """#87's listening model for a site or an edit-check item; a word's own."""
    return row.listening_seconds() if isinstance(row, WordSite) else listening_seconds(row)


def summarize(
    rows: list[SessionSite], words: list[WordSite], items: list, parts: dict[str, str],
    reciter_of: dict[str, int],
) -> dict:
    """Per part: rows, minutes, reciters and rows per half of the split (an edit-check item
    has no reciter); the queue cut into sittings."""
    order = queue(rows, items, words)
    part_of = {**parts, **{w.site.site_id: WORDS for w in words},
               **{i.site.site_id: EDIT_CHECK for i in items}}

    def describe(members: list) -> dict:
        ids = [m.site.site_id for m in members]
        counts = {part: sum(part_of[i] == part for i in ids) for part in PARTS}
        return {"rows": len(members), "minutes": round(sum(map(seconds, members)) / 60, 1),
                "by_part": {p: n for p, n in counts.items() if n}}

    by_part = {}
    for part in PARTS:
        members = [row for row in order if part_of[row.site.site_id] == part]
        reciters = [reciter_of[m.site.audio_filename] for m in members
                    if m.site.audio_filename in reciter_of]
        halves = [reciter_half(r) for r in reciters]
        by_part[part] = {**describe(members), "reciters": len(set(reciters)),
                         "rows_by_half": {half: halves.count(half) for half in HALVES}}
    total = sum(map(seconds, order))
    count = math.ceil(total / 60 / SITTING_MINUTES)
    cuts, sittings, start, spent = [k * total / count for k in range(1, count)], [], 0, 0.0
    for position, row in enumerate(order):
        spent += seconds(row)
        if cuts and spent >= cuts[0]:
            sittings.append({"queue_positions": [start + 1, position + 1],
                             **describe(order[start:position + 1])})
            start, cuts = position + 1, cuts[1:]
    sittings.append({"queue_positions": [start + 1, len(order)], **describe(order[start:])})
    return {
        "parts": by_part,
        "rows": len(order),
        "listening_minutes": round(total / 60, 1),
        "listening_minutes_without_edit_check": round(
            sum(seconds(r) for r in order if part_of[r.site.site_id] != EDIT_CHECK) / 60, 1),
        "sittings": sittings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="assemble the worklist and write it with its summary")
    parser.parse_args()
    sites = load_sites()
    rows, parts, duplicates = assemble(sites, load_worklist(SESSION_WORKLIST_PATH))
    words = load_words()
    write_worklist(rows, WORKLIST_PATH)
    items = load_edit_check()
    registry = load_staged_clips()
    for use, recordings in split_exposures(sites, words, registry).items():
        write_use(use, recordings)
    reciter_of = {name: clip.reciter_id for name, clip in registry.items()}
    summary = {
        "inputs": {
            "ship_a/sites.jsonl": sha256_of(SITES_PATH),
            "listening_session/worklist.jsonl": sha256_of(SESSION_WORKLIST_PATH),
            "synthetic_edits/blind_check.jsonl": sha256_of(BLIND_CHECK_PATH),
        },
        "frozen": {
            "ship_a/worklist.jsonl": sha256_of(WORKLIST_PATH),
            "ship_a/word_sample.jsonl": sha256_of(WORD_SAMPLE_PATH),
        },
        "carried_rows_already_ship_a_sites": duplicates,
        "sampled_words_also_asked_as_sites": shared_words([*rows, *items], words),
        "lean_partner_swaps": {"rows": len(lean := lean_swaps(sites)),
                               "minutes": round(sum(seconds(s.row) for s in lean) / 60, 1)},
        **summarize(rows, words, items, parts, reciter_of),
    }
    write_text_atomically(SUMMARY_PATH, json.dumps(summary, indent=1, ensure_ascii=False,
                                                   sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("rows", "listening_minutes",
                                              "listening_minutes_without_edit_check")}))
    for part, entry in summary["parts"].items():
        print(f"{part:20s} {entry['rows']:4d} rows {entry['minutes']:6.1f} min")
    for k, sitting in enumerate(summary["sittings"], 1):
        print(f"sitting {k}: {sitting['rows']} rows, {sitting['minutes']} min")


if __name__ == "__main__":
    main()
