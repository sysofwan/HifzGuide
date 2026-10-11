"""Guard Ship A's sites against a missed waqf at a wasl-dependent boundary (#134).

Segments are cut at the recitation VAD's pauses (:mod:`tadabur.waqf_detect`), and the
segmenter misses about one real pause in five. A missed pause leaves the reference in wasl,
so a correct reciter who stopped is compared with a mark that only exists in wasl: a geminate
made by joining two words (tanween or noon/meem idgham into the next word, a letter
assimilated into the next word's first: ``إِمْلَاقٍ نَّحْنُ``, ``خَيْرٌ لَّوْ``, ``هَمَّت طَّائِفَتَانِ``),
a ta marbuta read ``ت`` (``ه`` at a pause, #130), or a word-final haraka. A sun-letter ``لام``
(``غُلِبَتِ الرُّومُ``) keeps its shaddah after a pause and is not affected. The segmentation is not
changed (re-cutting moves every reference); the guard works downstream, as this sidecar.

**Which items depend on wasl** (:func:`classify`). Every Ship A site, every row of #61's
frozen worklist (geminates, partner swaps, P3.5 re-listens, the reject pile) and every word
of the word sample is realized with a waqf at each boundary of its word inside its segment
(the segment's words split there and each side phonetized on its own through
:func:`tadabur.waqf_segments.hafs_segment_reference`, the way every segment reference is
built), on both sides: the word after the boundary (a leading geminate, an assimilated
letter) and the word before it (a ta marbuta, a final haraka, a tanween's geminate). An item
**depends on wasl** at a boundary when the characters its prescribed mark is read from (a
geminate's two halves, a letter's carrier, a tashkeel mark's carrier and mark) do not
reappear unchanged in the waqf realization of its word (:func:`survives`); a sampled word,
judged whole, when its word changes at all. Nothing is hand-listed: idgham and cross-word
assimilation come out dependent, a sun letter does not.

**Was there a pause** (:mod:`tadabur.local_pause`). At each boundary an item depends on, the
local pause check reads the VAD more sensitively than the segmenter, at the operating point
its calibration on the owner's waqf fixtures picks. A boundary the clip's word times cannot
place (a re-read clip, or one the segmenter skipped) is ``unplaced`` and counts as no pause.

**The owner decides first.** A verdict whose note says ``waqf`` (or that the reciter paused
or stopped) marks a pause, one that says ``wasl`` (or negates a pause: "no waqf", "not
paused") marks none, overriding the detector for that item in both directions
(:func:`owner_says`); an ambiguous note leaves the detector's decision. Notes are
read from the verdicts file at every use (:func:`realized_ids`), so the guard stays right
while verdicts accumulate; ``build`` refreshes the counts.

**Treatment.** An item that depends on wasl at a boundary judged a pause is
``waqf_realized``: its prescribed mark is not what a correct waqf realizes, so it is no site
of its stratum and leaves every evaluation denominator (the population builders,
:mod:`training.ship_a_sizing`, and the scorer, ``truth_scorer.score(waqf_realized=…)``).
Its verdict is kept as a training target: what was heard.

Files (``tools/tadabur/ship_a/``):

* ``waqf_guard.boundaries.jsonl``: every boundary the local check was asked about, Ship A's
  and the fixtures', with its window and scores (``local_pause score``); the fixtures' rows
  also carry the clip's canonical reciter, the owner's verdict and the segmenter's decisions
  (today's, from re-segmenting the fixture clips, and the one the fixture recorded). The
  fixtures' split predates canonical reciter ids, so a reciter with boundaries in both splits
  keeps only its calibration ones (:func:`disjoint_fixtures`): the splits are
  reciter-disjoint;
* ``waqf_guard.vad.json``: the VAD pass's provenance;
* ``waqf_guard.segmentations.jsonl``, ``waqf_guard.segmentations.json``: the segmentations
  the windows are read on (``record``): per clip of the fixtures' re-segmentation and of the
  P3.5 re-location that a guarded P3.5 row sits on, its canonical reciter, skip reason,
  re-reads, word times and the segmenter's cuts; and each run's record and files' checksums;
* ``waqf_guard.jsonl``: one row per item, its dependence, each dependent boundary's window,
  score and detector decision;
* ``waqf_guard.summary.json``: counts per part and per stratum at the current verdicts, the
  calibration (curve, operating point, calibration and test numbers next to the
  segmenter's) and the inputs' checksums.

Usage (from ``tools/``; ``windows`` and ``build`` are torch-free and need quran-transcript)::

  python -m tadabur.resegment --registry tadabur/staged_audio/clips.jsonl \\
      --use waqf_boundary --audio-dir stage/clips --out-dir seg_fixtures   # GPU, word times
  python -m tadabur.waqf_guard record --fixture-segmentation seg_fixtures \\
      --p35-segmentation stage/seg_p35
  python -m tadabur.waqf_guard windows --out windows.jsonl
  python -m tadabur.local_pause score --windows windows.jsonl --audio-dir … \\
      --out tadabur/ship_a/waqf_guard.boundaries.jsonl \\
      --provenance tadabur/ship_a/waqf_guard.vad.json      # GPU, a minute
  python -m tadabur.waqf_guard build
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path

from training.tashkeel_eval import write_text_atomically

from .clip_status import read_clip_status
from .listening_session import (
    VERDICTS_PATH,
    SessionSite,
    Verdict,
    load_worklist,
    read_verdicts,
)
from .local_pause import SPLITS, TEST, boundary_window, calibrate, fires, point
from .mining_pool import CLIPS_PATH as POOL_CLIPS_PATH
from .ship_a import SHIP_A_DIR, SITES_PATH, load_sites
from .ship_a_worklist import GEMINATE, PAIR_PARTNER, WORKLIST_PATH, carried_part
from .staged_audio import load_staged_clips
from .truth_sites import NOT_HELD, P35_FIXTURE, SHADDAH, TARGET_PAIRS
from .waqf_event_fixtures import WAQF, load_waqf_events
from .word_sample import WORD_SAMPLE_PATH, load_words

GUARD_PATH = SHIP_A_DIR / "waqf_guard.jsonl"
SUMMARY_PATH = SHIP_A_DIR / "waqf_guard.summary.json"
BOUNDARIES_PATH = SHIP_A_DIR / "waqf_guard.boundaries.jsonl"
VAD_RUN_PATH = SHIP_A_DIR / "waqf_guard.vad.json"
SEGMENTATIONS_PATH = SHIP_A_DIR / "waqf_guard.segmentations.jsonl"
SEGMENTATIONS_RUN_PATH = SHIP_A_DIR / "waqf_guard.segmentations.json"
EXTENSION_CLIPS_PATH = SHIP_A_DIR / "mining" / "clips.jsonl"
FIXTURE_DIR = Path(__file__).parent / "waqf_event_fixtures"

#: Where an item appears: Ship A's sites, a worklist part, the word sample.
SHIP_A_SITES = "ship_a_sites"
WORD_SAMPLE = "word_sample"
#: A sampled word's mark: judged whole.
WORD = "word"

#: The item's word follows the boundary (its start changes), or precedes it (its end does).
AFTER = "after"
BEFORE = "before"

DEPENDENT = "wasl_dependent"
NOT_DEPENDENT = "not_dependent"
#: quran-transcript cannot realize a waqf at one of the item's boundaries (a leen madd on a
#: final sukun); the item is kept as a site and counted.
UNSUPPORTED = "phonetizer_unsupported"

#: A measured boundary is a site's or a waqf fixture's.
FIXTURE = "fixture"
SITE = "site"
WAQF_REALIZED = "waqf_realized"
#: Who decided a dependent item's pause.
OWNER_NOTE = "owner_note"
LOCAL_CHECK = "local_check"
UNPLACED = "unplaced"

#: Words of a note that name a pause, or continuation.
_PAUSE_WORDS = frozenset({"waqf", "pause", "paused", "pauses", "pausing", "stop", "stopped",
                          "stops", "stopping"})
_WASL_WORDS = frozenset({"wasl"})
#: Words that negate a pause or wasl word among the :data:`_NEGATION_REACH` words before it.
_NEGATORS = frozenset({"no", "not", "never", "without", "nor", "cannot", "didnt", "doesnt",
                       "dont", "hasnt", "havent", "hadnt", "isnt", "wasnt", "wont"})
_NEGATION_REACH = 3


# --- which items depend on wasl --------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """Anything the guard classifies: a site of Ship A or the worklist, or a sampled word,
    with the segment it is read in (its realized reference and words) and where it appears."""

    item_id: str
    parts: tuple[str, ...]
    source: str
    stratum: str
    mark: str
    prescribed: str
    audio_filename: str
    surah_ayah: str
    word_start: int
    word_offsets: tuple[int, ...]
    reference: str
    reference_index: int


def site_item(row: SessionSite, part: str) -> Item:
    site = row.site
    return Item(site.site_id, (part,), site.source, site.stratum, site.mark, site.prescribed,
                site.audio_filename, site.surah_ayah, row.word_start, row.word_offsets,
                site.reference, site.reference_index)


def worklist_part(row: SessionSite) -> str:
    """A worklist row's part (``ship_a_worklist.PARTS``)."""
    return carried_part(row) or (GEMINATE if row.site.mark == SHADDAH else PAIR_PARTNER)


def merge(items: Iterable[Item]) -> list[Item]:
    """One item per id, appearing in every part any copy does; the copies must agree."""
    merged: dict[str, Item] = {}
    for item in items:
        known = merged.get(item.item_id)
        if known is not None:
            if replace(known, parts=()) != replace(item, parts=()):
                raise ValueError(f"{item.item_id}: two rows disagree on the site")
            item = replace(item, parts=tuple(sorted({*known.parts, *item.parts})))
        merged[item.item_id] = item
    return [merged[k] for k in sorted(merged)]


def all_items() -> list[Item]:
    """Ship A's sites, the frozen worklist's site rows and the word sample, merged by id."""
    sites = (site_item(s.row, SHIP_A_SITES) for s in load_sites())
    rows = (site_item(r, worklist_part(r)) for r in load_worklist(WORKLIST_PATH))
    words = (Item(w.site.site_id, (WORD_SAMPLE,), WORD_SAMPLE, w.site.stratum, WORD, WORD,
                  w.site.audio_filename, w.site.surah_ayah, w.word_start, w.word_offsets,
                  w.site.reference, w.site.reference_index) for w in load_words())
    return merge([*sites, *rows, *words])


@cache
def _ayah_words(surah_ayah: str) -> tuple[str, ...]:
    from .waqf_segments import _uthmani_words

    return tuple(_uthmani_words(surah_ayah))


@cache
def realize(surah_ayah: str, start: int, end: int) -> tuple[str, tuple[int, ...]] | None:
    """Ayah words ``[start, end)`` phonetized as one run, its last word in waqf form (how
    every segment reference is built), with each word's offset; ``None`` where
    quran-transcript cannot."""
    from .waqf_segments import hafs_segment_reference

    try:
        reference, offsets = hafs_segment_reference()(list(_ayah_words(surah_ayah)[start:end]))
    except (KeyError, IndexError):
        return None
    return reference, tuple(offsets)


def mark_positions(item: Item) -> tuple[int, ...] | None:
    """The reference positions an item's prescribed mark is read from: a letter's carrier
    (and a single consonant's), a geminate's two halves, a tashkeel mark's carrier and mark;
    ``None`` for a sampled word, which is judged whole."""
    i = item.reference_index
    if item.mark == WORD:
        return None
    if item.mark in TARGET_PAIRS or (item.mark == SHADDAH and item.prescribed == NOT_HELD):
        return (i,)
    return (i, i + 1)


def survives(wasl: str, waqf: str, positions: Iterable[int]) -> bool:
    """Whether the characters of ``wasl`` at ``positions`` reappear in ``waqf`` unchanged and
    in the same arrangement (difflib's matching blocks: a geminate whose first half is gone,
    or a letter replaced by another, does not)."""
    positions = list(positions)
    mapped = {}
    for block in SequenceMatcher(None, wasl, waqf, autojunk=False).get_matching_blocks():
        mapped.update({block.a + k: block.b + k for k in range(block.size)})
    return all(p in mapped and mapped[p] - mapped[positions[0]] == p - positions[0]
               for p in positions)


@dataclass(frozen=True)
class Dependence:
    """A boundary an item depends on: the ayah word it precedes, which side of it the item's
    word is, and that word as its segment realizes it (wasl) and with a waqf there."""

    word: int
    side: str
    wasl: str
    waqf: str


@dataclass(frozen=True)
class Classification:
    dependences: tuple[Dependence, ...]
    #: Boundaries (the ayah word each precedes) where no waqf could be realized.
    unsupported: tuple[int, ...]

    @property
    def dependence(self) -> str:
        if self.dependences:
            return DEPENDENT
        return UNSUPPORTED if self.unsupported else NOT_DEPENDENT


def classify(item: Item) -> Classification:
    """The boundaries of the item's word, inside its segment, where a waqf changes its
    prescribed mark (module docstring)."""
    count, start = len(item.word_offsets) - 1, item.word_start
    if realize(item.surah_ayah, start, start + count) != (item.reference, item.word_offsets):
        raise ValueError(f"{item.item_id}: its reference is not the realization of its words")
    word = bisect.bisect_right(item.word_offsets, item.reference_index) - 1
    lo, hi = item.word_offsets[word], item.word_offsets[word + 1]
    wasl = item.reference[lo:hi].rstrip(" ")
    positions = mark_positions(item)
    if positions is not None and max(positions) >= lo + len(wasl):
        raise ValueError(f"{item.item_id}: its mark is not read inside one word")
    dependences, unsupported = [], []
    for cut, side in ((word, AFTER), (word + 1, BEFORE)):
        if not 0 < cut < count:
            continue  # a segment edge: the reference already realizes a waqf there
        part = (realize(item.surah_ayah, start + cut, start + count) if side == AFTER
                else realize(item.surah_ayah, start, start + cut))
        if part is None:
            unsupported.append(start + cut)
            continue
        reference, offsets = part
        span = (offsets[0], offsets[1]) if side == AFTER else (offsets[-2], offsets[-1])
        waqf = reference[span[0]:span[1]].rstrip(" ")
        changed = (wasl != waqf if positions is None
                   else not survives(wasl, waqf, (p - lo for p in positions)))
        if changed:
            dependences.append(Dependence(start + cut, side, wasl, waqf))
    return Classification(tuple(dependences), tuple(unsupported))


# --- the windows the local check reads -------------------------------------------------------


def site_key(audio_filename: str, word: int) -> str:
    return f"site:{audio_filename}@{word}"


def fixture_key(audio_filename: str, word: int) -> str:
    return f"fixture:{audio_filename}@{word}"


def placeable(word_times: Iterable[float], re_reads: int) -> tuple[float, ...] | None:
    """A clip's word times when they place its words: not for a re-read clip (a word's time
    names one pass) or one the segmenter skipped (none)."""
    times = tuple(word_times)
    return times if times and not re_reads else None


def _manifest_times(path: Path) -> dict[str, tuple[float, ...] | None]:
    with open(path, encoding="utf-8") as f:
        clips = [json.loads(line) for line in f if line.strip()]
    return {c["audio_filename"]: placeable(c["word_times"], c["re_reads"]) for c in clips}


# --- the segmentations the windows are read on -------------------------------------------------

#: The re-segmentation of the waqf fixtures' clips (:mod:`tadabur.resegment`, #134), and the
#: P3.5 re-location's (#83), each recorded in :data:`SEGMENTATIONS_PATH` for the clips it places.
FIXTURE_RESEGMENT = "waqf_fixture_resegment"
P35_RELOCATION = "p35_relocation"


def segmentation_records(source: str, segmentation_dir: Path, names: Iterable[str]) -> list[dict]:
    """What the windows and the fixture exclusions read off one ``tadabur.resegment`` output,
    for the clips ``names``: each clip's canonical reciter, skip reason, re-reads, word times
    and the words its segments start at after the first (the segmenter's cuts)."""
    wanted = set(names)
    with open(segmentation_dir / "segmentation.jsonl", encoding="utf-8") as f:
        cuts = {row["audio_filename"]: sorted(s["word_start"] for s in row["segments"][1:])
                for row in map(json.loads, f)}
    return [{"source": source, "audio_filename": c.audio_filename, "reciter_id": c.reciter_id,
             "skip_reason": c.skip_reason, "re_reads": c.re_reads,
             "word_times": list(c.word_times), "cuts": cuts.get(c.audio_filename, [])}
            for c in read_clip_status(segmentation_dir / "clip_status.jsonl")
            if c.audio_filename in wanted]


def segmentation_provenance(segmentation_dir: Path) -> dict:
    """One ``tadabur.resegment`` output's run record and its files' checksums."""
    return {"run": json.loads((segmentation_dir / "run.json").read_text(encoding="utf-8")),
            **{f"{name}_sha256": sha256_of(segmentation_dir / name)
               for name in ("clip_status.jsonl", "segmentation.jsonl")}}


def read_segmentations(path: Path = SEGMENTATIONS_PATH) -> dict[str, dict[str, dict]]:
    """The committed segmentation records, per source and clip."""
    records: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for row in map(json.loads, f):
            records[row["source"]][row["audio_filename"]] = row
    return records


def _times(records: Mapping[str, dict]) -> dict[str, tuple[float, ...] | None]:
    return {name: placeable(r["word_times"], r["re_reads"]) for name, r in records.items()}


def site_windows(items: Iterable[Item]) -> list[dict]:
    """One row per boundary any item depends on: its key, clip, word and window (``null``
    where the clip's word times cannot place it). A P3.5 row is read on the P3.5
    re-location's segmentation, every other item on the pool's or the extension's."""
    pool = {**_manifest_times(POOL_CLIPS_PATH), **_manifest_times(EXTENSION_CLIPS_PATH)}
    p35 = _times(read_segmentations()[P35_RELOCATION])
    windows: dict[tuple[str, int], list[float] | None] = {}
    for item in items:
        times = (p35 if item.source == P35_FIXTURE else pool)[item.audio_filename]
        for dependence in classify(item).dependences:
            boundary = (item.audio_filename, dependence.word)
            window = None if times is None else boundary_window(times, dependence.word)
            if windows.setdefault(boundary, window) != window:
                raise ValueError(f"{boundary}: two segmentations place the boundary apart")
    return [{"key": site_key(name, word), "kind": SITE, "audio_filename": name,
             "word": word, "window": window}
            for (name, word), window in sorted(windows.items())]


def fixture_boundaries() -> list[dict]:
    """One row per adjudicated fixture boundary (the ayah word it precedes), with its split
    and the clip's canonical reciter: the owner's verdict (a pause when any of its rows says
    ``waqf``) and the segmenter's decision as the fixture recorded it."""
    reciter_of = {name: clip.reciter_id for name, clip in load_staged_clips().items()}
    seen: dict[tuple, dict] = {}
    for split in SPLITS:
        for entry in load_waqf_events(FIXTURE_DIR / f"waqf_events.{split}.jsonl"):
            key = (split, entry.audio_ref, entry.word_index + 1)
            row = seen.setdefault(key, {"split": split, "audio_filename": entry.audio_ref,
                                        "word": entry.word_index + 1,
                                        "reciter_id": reciter_of.get(entry.audio_ref),
                                        "waqf": False, "segmenter_fixture": False})
            row["waqf"] |= entry.verdict == WAQF
            row["segmenter_fixture"] |= entry.predicted == WAQF
    return [seen[k] for k in sorted(seen)]


def reciter_overlap(rows: Iterable[Mapping]) -> set[int]:
    """The canonical reciters with fixture boundaries in both splits."""
    rows = list(rows)
    return set.intersection(*({r["reciter_id"] for r in rows if r["split"] == split
                               and r["reciter_id"] is not None} for split in SPLITS))


def disjoint_fixtures(rows: list[dict]) -> tuple[list[dict], dict[int, int]]:
    """The fixture boundaries with every test boundary of a reciter who also has calibration
    boundaries left out (the calibration side chose the operating point, so it keeps them),
    and how many each such reciter lost. The fixtures' split predates the canonical reciter
    ids; a clip whose reciter the registry does not know is never placed."""
    shared = reciter_overlap(rows)
    dropped = [r for r in rows if r["split"] == TEST and r["reciter_id"] in shared]
    kept = [r for r in rows if not (r["split"] == TEST and r["reciter_id"] in shared)]
    if reciter_overlap(kept):
        raise ValueError("calibration and test fixtures share a canonical reciter")
    return kept, dict(sorted(Counter(r["reciter_id"] for r in dropped).items()))


def fixture_windows() -> list[dict]:
    """Every reciter-disjoint fixture boundary (:func:`disjoint_fixtures`) with today's
    segmenter decision (a cut there when re-segmenting the clips with
    :mod:`tadabur.resegment`) and its window on today's word times."""
    records = read_segmentations()[FIXTURE_RESEGMENT]
    times = _times(records)
    rows = []
    for row in disjoint_fixtures(fixture_boundaries())[0]:
        name, word = row["audio_filename"], row["word"]
        clip_times = times.get(name)
        placed = clip_times is not None and word < len(clip_times) - 1
        rows.append({"key": fixture_key(name, word), "kind": FIXTURE, **row,
                     "segmenter": name in records and word in records[name]["cuts"],
                     "window": boundary_window(clip_times, word) if placed else None})
    return rows


# --- the decisions ---------------------------------------------------------------------------


def owner_says(note: str) -> bool | None:
    """What a verdict's note says about a pause: ``True`` for a waqf, ``False`` for wasl,
    ``None`` when it says neither or is ambiguous, which leaves the detector's decision.

    The note is lower-cased with its whitespace and apostrophes normalized, then read word by
    word: a pause word (``waqf``, ``pause(d)``, ``stop(ped)``) or ``wasl`` is negated when
    ``no``, ``not``, ``never``, ``without`` or a word ending in ``n't`` is among the three
    words before it ("no pause", "not paused", "has not stopped", "didn't stop"). A negated
    pause says wasl; a negated wasl says a pause. Two claims that disagree are ambiguous."""
    words = re.findall(r"[a-z']+", " ".join(note.lower().replace("\u2019", "'").split()))
    claims = set()
    for k, word in enumerate(words):
        if word not in _PAUSE_WORDS | _WASL_WORDS:
            continue
        negated = any(w in _NEGATORS or w.endswith("n't")
                      for w in words[max(0, k - _NEGATION_REACH):k])
        claims.add((word in _PAUSE_WORDS) != negated)
    return claims.pop() if len(claims) == 1 else None


def read_boundaries(path: Path = BOUNDARIES_PATH) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {row["key"]: row for row in map(json.loads, f)}


def guard_row(item: Item, classification: Classification, boundaries: Mapping[str, dict],
              operating_point: Mapping[str, float]) -> dict:
    """The item's sidecar row: its dependence and, per dependent boundary, the window, the
    scores and the local check's decision."""
    measured = []
    for d in classification.dependences:
        key = site_key(item.audio_filename, d.word)
        if key not in boundaries:
            raise ValueError(f"{key} was never measured: re-run windows and local_pause score")
        row = boundaries[key]
        paused = None if row["window"] is None else fires(
            row["scores"], operating_point["threshold"], operating_point["duration"])
        measured.append({**asdict(d), "key": key, "window": row["window"],
                         "scores": row["scores"], "paused": paused,
                         "decided_by": UNPLACED if paused is None else LOCAL_CHECK})
    return {"item_id": item.item_id, "parts": list(item.parts), "stratum": item.stratum,
            "mark": item.mark, "prescribed": item.prescribed,
            "dependence": classification.dependence,
            "unsupported_boundaries": list(classification.unsupported),
            "boundaries": measured,
            "detector_paused": any(b["paused"] for b in measured)}


def resolve(row: Mapping, verdict: Verdict | None) -> tuple[str, str | None]:
    """An item's status and who decided it: an owner note first, else the local check (an
    unplaced boundary is no pause). Only a dependent item can be realized under waqf."""
    if row["dependence"] != DEPENDENT:
        return SITE, None
    owner = owner_says(verdict.note) if verdict is not None else None
    if owner is not None:
        return (WAQF_REALIZED if owner else SITE), OWNER_NOTE
    if row["detector_paused"]:
        return WAQF_REALIZED, LOCAL_CHECK
    unplaced = any(b["paused"] is None for b in row["boundaries"])
    return SITE, UNPLACED if unplaced else LOCAL_CHECK


def read_guard(path: Path = GUARD_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def realized_ids(guard_path: Path = GUARD_PATH,
                 verdicts_path: Path = VERDICTS_PATH) -> frozenset[str]:
    """Every item ``waqf_realized`` at the current verdicts: what the population builders and
    the scorer leave out of every denominator."""
    verdicts = read_verdicts(verdicts_path)
    return frozenset(row["item_id"] for row in read_guard(guard_path)
                     if resolve(row, verdicts.get(row["item_id"]))[0] == WAQF_REALIZED)


# --- the summary -----------------------------------------------------------------------------


def _tally(rows: list[dict], verdicts: Mapping[str, Verdict]) -> dict:
    counts = Counter()
    for row in rows:
        status, decided_by = resolve(row, verdicts.get(row["item_id"]))
        counts["items"] += 1
        counts[row["dependence"]] += 1
        if row["dependence"] == DEPENDENT:
            counts["detector_paused"] += row["detector_paused"]
            counts["unplaced"] += decided_by == UNPLACED
            counts["decided_by_owner_note"] += decided_by == OWNER_NOTE
        counts[status] += 1
    return {k: counts[k] for k in ("items", DEPENDENT, NOT_DEPENDENT, UNSUPPORTED,
                                   "detector_paused", "unplaced", "decided_by_owner_note",
                                   WAQF_REALIZED, SITE)}


def summarize(rows: list[dict], verdicts: Mapping[str, Verdict], calibration: dict,
              provenance: dict) -> dict:
    """Counts per part and per stratum (the word sample's per-reciter strata as one) at the
    current verdicts; the owner's notes; the calibration; the inputs."""
    by_part, by_stratum = defaultdict(list), defaultdict(list)
    for row in rows:
        for part in row["parts"]:
            by_part[part].append(row)
        by_stratum[WORD_SAMPLE if row["mark"] == WORD else row["stratum"]].append(row)
    notes = Counter()
    for row in rows:
        verdict = verdicts.get(row["item_id"])
        says = owner_says(verdict.note) if verdict is not None else None
        if says is not None:
            notes[f"{'waqf' if says else 'wasl'} note, {row['dependence']}"] += 1
    return {
        **provenance,
        "treatment": "a wasl-dependent item at a boundary judged a pause is waqf_realized: "
                     "no site of its stratum, out of every evaluation denominator; its "
                     "verdict stays a training target (heard)",
        "parts": {p: _tally(r, verdicts) for p, r in sorted(by_part.items())},
        "strata": {s: _tally(r, verdicts) for s, r in sorted(by_stratum.items())},
        "all": _tally(rows, verdicts),
        "owner_notes": dict(sorted(notes.items())),
        "calibration": calibration,
    }


def segmenter_on_fixture_rows() -> dict:
    """The segmenter's precision and recall per split as the fixtures recorded it, row by
    row (the issue's table: a positive is a row the owner called ``waqf``)."""
    out = {}
    for split in SPLITS:
        entries = load_waqf_events(FIXTURE_DIR / f"waqf_events.{split}.jsonl")
        out[split] = point((e.predicted == WAQF for e in entries),
                           (e.verdict == WAQF for e in entries)).as_dict()
    return out


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    """Classify every item, apply the local check at its operating point, and write the
    sidecar and its summary."""
    boundaries = read_boundaries()
    measured = [b for b in boundaries.values() if b["kind"] == FIXTURE]
    if [{k: v for k, v in b.items() if k != "scores"} for b in measured] != fixture_windows():
        raise ValueError("the measured fixture boundaries are not the recorded segmentations' "
                         "reciter-disjoint ones: re-run windows and local_pause score")
    fixtures = [b for b in measured if b["window"]]
    calibration = calibrate(fixtures)
    operating_point = calibration["operating_point"]
    if operating_point is None:
        raise ValueError("no point of the calibration curve meets the precision floor")
    rows = [guard_row(item, classify(item), boundaries, operating_point)
            for item in all_items()]
    write_text_atomically(GUARD_PATH, "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    unplaced = Counter(b["split"] for b in boundaries.values()
                       if b["kind"] == FIXTURE and not b["window"])
    provenance = {
        "inputs": {name: sha256_of(path) for name, path in (
            ("ship_a/sites.jsonl", SITES_PATH), ("ship_a/worklist.jsonl", WORKLIST_PATH),
            ("ship_a/word_sample.jsonl", WORD_SAMPLE_PATH),
            ("ship_a/waqf_guard.boundaries.jsonl", BOUNDARIES_PATH),
            ("ship_a/waqf_guard.segmentations.jsonl", SEGMENTATIONS_PATH),
            ("listening_session/verdicts.jsonl", VERDICTS_PATH))},
        "detector": {**json.loads(VAD_RUN_PATH.read_text(encoding="utf-8")),
                     "operating_point": operating_point},
        "segmentations": json.loads(SEGMENTATIONS_RUN_PATH.read_text(encoding="utf-8")),
        "fixture_test_boundaries_dropped_for_a_reciter_in_calibration": {
            str(r): n for r, n in disjoint_fixtures(fixture_boundaries())[1].items()},
        "fixture_boundaries_unplaced": dict(sorted(unplaced.items())),
        "segmenter_on_fixture_rows": segmenter_on_fixture_rows(),
    }
    summary = summarize(rows, read_verdicts(), calibration, provenance)
    write_text_atomically(SUMMARY_PATH, json.dumps(summary, indent=1, ensure_ascii=False,
                                                   sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser(
        "record", help="record the segmentations the windows are read on (where they live)")
    record.add_argument("--fixture-segmentation", type=Path, required=True,
                        help="tadabur.resegment's output for the waqf_boundary clips")
    record.add_argument("--p35-segmentation", type=Path, required=True,
                        help="the P3.5 re-location's tadabur.resegment output")
    windows = commands.add_parser("windows", help="list every boundary the check must read")
    windows.add_argument("--out", type=Path, required=True)
    commands.add_parser("build", help="write the sidecar and its summary")
    args = parser.parse_args()
    if args.command == "record":
        p35_clips = {i.audio_filename for i in all_items() if i.source == P35_FIXTURE}
        fixture_clips = {r["audio_filename"] for r in fixture_boundaries()}
        records = [*segmentation_records(FIXTURE_RESEGMENT, args.fixture_segmentation,
                                         fixture_clips),
                   *segmentation_records(P35_RELOCATION, args.p35_segmentation, p35_clips)]
        write_text_atomically(SEGMENTATIONS_PATH, "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records))
        provenance = {FIXTURE_RESEGMENT: segmentation_provenance(args.fixture_segmentation),
                      P35_RELOCATION: segmentation_provenance(args.p35_segmentation)}
        write_text_atomically(SEGMENTATIONS_RUN_PATH,
                              json.dumps(provenance, indent=1, sort_keys=True) + "\n")
        print(f"{len(records)} clips recorded")
        return
    if args.command == "windows":
        rows = [*site_windows(all_items()), *fixture_windows()]
        write_text_atomically(args.out, "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        print(f"{len(rows)} windows ({sum(r['window'] is None for r in rows)} unplaced)")
        return
    summary = build()
    print(json.dumps({"all": summary["all"],
                      "operating_point": summary["calibration"]["operating_point"]},
                     ensure_ascii=False))
    for split, report in summary["calibration"]["splits"].items():
        print(split, json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
