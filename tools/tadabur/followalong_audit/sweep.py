"""The S2 sweeps (#308): the engine on a perfect decode, clean and planted.

The Oracle arm (:mod:`tadabur.followalong_audit.oracle`) is the reference placed on the clock,
so an engine fed it grades a perfect decode: every word it does not grade correct is a
follow-along or reference fault. This module runs that arm over two populations —

* the **tlog** clips, every confirm.1/confirm.2 gating clip #248 leaves (5,742), at the clip's
  own phase A word times, in strict and balanced; and
* the **mushaf**, every ayah (6,236), at the median tlog speaking pace in Characters per
  second, because tlog never covers most ayat. Ayat the phonetizer cannot read are tallied,
  never dropped.

— and over the same clips **planted** with one seeded mistake per kind (``plant``), which
measures what the follow-along catches when the model hears the mistake. A planted word is
**caught** when its kept grade is below correct; caught on the inserted copy or its neighbours
for an added word; and **not_graded** when the engine kept no row for it, reported apart because
neither a catch nor a miss says anything about the engine. Every finding carries the per-check
statuses, the word's position, and whether a window boundary splits it.
"""

from __future__ import annotations

import gzip
import json
import math
import sqlite3
import statistics
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tadabur.followalong_audit.app_arm import word_grade_rows
from tadabur.followalong_audit.oracle import oracle_checks
from tadabur.followalong_audit.clips import gating_rows, oracle_eligible
from tadabur.followalong_audit.patterns import group_findings, issue_body
from tadabur.followalong_audit.plant import KINDS, Planted, plant, plantable_words, planting_seed
from tadabur.followalong_audit.session import session_request
from tadabur.followalong_audit.tracer import load_word_times
from tadabur.waqf_segments import _uthmani_words, hafs_word_reference
from training.decoding import SAMPLE_RATE, WINDOW, check_schedule, clusters
from training.muraja_policy import AUDIT_SCORINGS, CORRECT, CURRENT_COMMIT, word_traces
from training.nonpro_decode import item_key

#: The sweep's one seed: the clip/kind draw and the run identity both hang from it.
SWEEP_SEED = 308

#: The run's committed tables, beside the sweep's other provenance.
ORACLE_SWEEP_DIR = Path(__file__).parent / "oracle_sweep"

#: The committed phase A decode rows, the only source of a clip's length (``end_sample``).
DECODES_PATH = (Path(__file__).resolve().parents[1] / "tlog_mining" / "decodes"
                / "tlog.gating.r3_candidate_ema.jsonl.gz")

#: The waqf (pause) signs a printed word may carry (``PhonemeIndex.swift:88-91``): U+06D6-U+06DB.
WAQF_SIGNS = frozenset(chr(codepoint) for codepoint in range(0x06D6, 0x06DC))

#: How many requests one engine call is handed: enough to amortise the harness's workers, few
#: enough that a batch's results stay small.
GRADE_BATCH = 512

#: The groupings the report pools findings by: one cause each, then boundaries/positions, then
#: the reference form (a candidate issue needs ten distinct words).
PATTERN_DIMENSIONS = (
    ("error_kind",),
    ("error_kind", "position"),
    ("error_kind", "boundary"),
    ("form",),
)

#: The word-length cells a catch rate is reported in (Characters, both ends inclusive).
LENGTH_BINS = ((1, 2, "1-2"), (3, 4, "3-4"), (5, 6, "5-6"), (7, 10, "7-10"), (11, None, "11+"))


@dataclass(frozen=True)
class SweepClip:
    """One clip's reference placement: its ayah, phoneme words and the clock they sit on."""

    item: str
    surah: int
    ayah: int
    words: tuple[str, ...]
    #: One ``(start_s, end_s)`` per word, or ``None`` for a clip whose alignment failed.
    word_times: tuple[tuple[float, float], ...] | None = None
    n_samples: int = SAMPLE_RATE
    #: The 0-based word indices a pause sign marks.
    waqf: frozenset[int] = frozenset()


@dataclass(frozen=True)
class CleanSweep:
    """A clean sweep's findings, denominators, and the word table the App arm's strata read."""

    clips: int
    findings: list[dict]
    graded_words: dict[str, int]
    not_graded: dict[str, int]
    words_by_position: dict[str, dict[str, int]]
    word_rows: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class PlantedSweep:
    """One planted sweep: a row per clip, kind and mode, and the kinds no clip could carry."""

    rows: list[dict]
    unplantable: dict[str, int]


def word_position(index: int, count: int, *, waqf: bool) -> str:
    """A word's position in its ayah: ``last``, ``waqf``, ``first`` or ``middle``.

    An ayah's end is always a stop, so the last word reads ``last`` even when a pause sign marks
    it; a marked word inside the ayah reads ``waqf``.
    """
    if index == count - 1:
        return "last"
    if waqf:
        return "waqf"
    if index == 0:
        return "first"
    return "middle"


def splits_window(start_s: float, end_s: float, n_samples: int) -> bool:
    """Whether a transcriber window edge falls strictly inside a word's ``[start, end)``.

    The edges are the full windows' starts and ends (:func:`training.decoding.check_schedule`);
    a word that begins or ends on one is not split, and a clip too short for a full window has
    no edge to split anything.
    """
    edges = {
        edge / SAMPLE_RATE
        for span in check_schedule(n_samples) if span.kind == WINDOW
        for edge in (span.start, span.end)
    }
    return any(start_s < edge < end_s for edge in edges)


def median_speech_pace(clips: Sequence[tuple[int, Sequence[tuple[float, float]] | None]]) -> float:
    """The median Characters per second over the speech span of the clips with word times.

    Each clip's span is its first word's start to its last word's end — the clock the oracle
    spreads Characters over — so a clip's silence before and after does not slow the pace.
    """
    rates = []
    for characters, times in clips:
        if not times:
            continue
        span = times[-1][1] - times[0][0]
        if span > 0:
            rates.append(characters / span)
    if not rates:
        raise ValueError("no aligned clip to take a speech pace from")
    return statistics.median(rates)


def paced_word_times(lengths: Sequence[int],
                     pace: float) -> tuple[list[tuple[float, float]], int]:
    """Words of ``lengths`` Characters laid end to end at ``pace`` Characters per second.

    Returns each word's ``(start_s, end_s)`` and the clip's sample count, rounded up so the
    clip never ends inside its last Character.
    """
    times: list[tuple[float, float]] = []
    at = 0.0
    for length in lengths:
        end = at + length / pace
        times.append((at, end))
        at = end
    return times, math.ceil(at * SAMPLE_RATE)


def length_bin(length: int) -> str:
    """The catch-rate cell a word of ``length`` Characters falls in."""
    for low, high, label in LENGTH_BINS:
        if length >= low and (high is None or length <= high):
            return label
    return LENGTH_BINS[0][2]


def _offsets(words: Sequence[str]) -> list[int]:
    offsets = [0]
    for word in words:
        offsets.append(offsets[-1] + len(word))
    return offsets


def _request(item: str, clip: SweepClip, words: Sequence[str],
             word_times: Sequence[tuple[float, float]] | None) -> dict:
    """One oracle session over ``words`` placed at ``word_times``, on the clip's ayah.

    The reported words are the clip's own ayah words (1..N), whatever the planting changed:
    Muraja grades its own phoneme words, and the planted reference is what its checks hear.
    """
    reference = "".join(words)
    arm = oracle_checks(reference, _offsets(words), word_times, clip.n_samples)
    return session_request(item, clip.surah, clip.ayah, len(clip.words), arm.checks)


def _batches(items: Sequence, size: int = GRADE_BATCH):
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


def _error_type(trace: Mapping) -> str:
    """A word's error type: the first error's ``type`` (``mismatch``/``tashkeel``/...), or its
    quality when the engine kept none (a skipped word)."""
    if not trace["errors"]:
        return trace["quality"]
    kind = trace["errors"][0]["kind"]
    return kind["type"] if isinstance(kind, Mapping) else str(kind)


def _oracle_causes(trace: Mapping) -> list[str]:
    """Why a word is flagged in the oracle table: its errors' types, or its quality."""
    return sorted({_error_type(trace)})


def _statuses(trace: Mapping | None, request: Mapping) -> list:
    """The word's status at every check of its session (``None`` where it was not graded)."""
    if trace is None:
        return [None for _ in request["checks"]]
    return [check["status"] for check in trace["checks"]]


def _boundary(clip: SweepClip, index: int) -> bool | None:
    if clip.word_times is None or index >= len(clip.word_times):
        return None
    start, end = clip.word_times[index]
    return splits_window(start, end, clip.n_samples)


def clean_sweep(clips: Sequence[SweepClip], engine) -> CleanSweep:
    """Grade every clip's reference on its own clock and keep every word not graded correct.

    ``engine`` shares ``Harness.run``'s ``run(requests) -> (build, results)`` shape. Each
    non-correct word is a finding with its per-check statuses, position and window boundary;
    a word with no row in ``final`` is counted as ``not_graded`` instead.
    """
    ordered = sorted(clips, key=lambda clip: clip.item)
    findings: list[dict] = []
    graded = {scoring.name: 0 for scoring in AUDIT_SCORINGS}
    not_graded = {scoring.name: 0 for scoring in AUDIT_SCORINGS}
    positions = {scoring.name: {} for scoring in AUDIT_SCORINGS}
    word_rows: list[dict] = []

    for batch in _batches(ordered):
        requests = [_request(f"{clip.item}/oracle", clip, clip.words, clip.word_times)
                    for clip in batch]
        _, results = engine.run(requests)
        for clip, request in zip(batch, requests):
            traces = word_traces(request, results, AUDIT_SCORINGS)
            by_word = {(trace["scoring"], trace["word"]): trace for trace in traces}
            word_rows.extend({**row, "item": clip.item} for row in word_grade_rows(
                request["item"], len(clip.words), traces, _oracle_causes))
            for scoring in AUDIT_SCORINGS:
                mode = scoring.name
                for word in range(1, len(clip.words) + 1):
                    trace = by_word.get((mode, word))
                    index = word - 1
                    if trace is None:
                        not_graded[mode] += 1
                        continue
                    graded[mode] += 1
                    position = word_position(index, len(clip.words), waqf=index in clip.waqf)
                    positions[mode][position] = positions[mode].get(position, 0) + 1
                    if trace["quality"] == CORRECT:
                        continue
                    findings.append({
                        "item": clip.item,
                        "surah": clip.surah,
                        "ayah": clip.ayah,
                        "word": word,
                        "mode": mode,
                        "quality": trace["quality"],
                        "error_kind": _error_type(trace),
                        "form": clip.words[index],
                        "position": position,
                        "boundary": _boundary(clip, index),
                        "statuses": _statuses(trace, request),
                        "marked": trace["marked"],
                        "setting_checks": trace["setting_checks"],
                    })

    return CleanSweep(clips=len(ordered), findings=findings, graded_words=graded,
                      not_graded=not_graded, words_by_position=positions,
                      word_rows=word_rows)


def _planted_times(clip: SweepClip, kind: str,
                   planted: Planted) -> tuple[tuple[float, float], ...] | None:
    """The clip's word times for the planted words: the copy shares its word's span, a skipped
    word takes its span with it, and every other edit keeps the clock."""
    if clip.word_times is None:
        return None
    times = list(clip.word_times)
    if kind == "skip_word":
        del times[planted.word]
    elif kind == "add_word":
        start, end = times[planted.word]
        middle = (start + end) / 2
        times[planted.word] = (start, middle)
        times.insert(planted.inserted, (middle, end))
    return tuple(times)


def _planted_boundary(clip: SweepClip, kind: str, planted: Planted) -> bool | None:
    if clip.word_times is None or kind == "skip_word":
        return None
    times = _planted_times(clip, kind, planted)
    index = planted.inserted if kind == "add_word" else planted.word
    if times is None or index >= len(times):
        return None
    start, end = times[index]
    return splits_window(start, end, clip.n_samples)


def _planted_row(clip: SweepClip, kind: str, planted: Planted, mode: str,
                 by_word: Mapping[tuple[str, int], Mapping], request: Mapping) -> dict:
    """One row: the planted word's kept grade, and whether the engine caught the mistake."""
    form = clip.words[planted.word] if kind == "skip_word" else planted.words[planted.word]
    row = {
        "item": clip.item,
        "surah": clip.surah,
        "ayah": clip.ayah,
        "kind": kind,
        "mode": mode,
        "word": planted.word + 1,
        "label": planted.label,
        "protected": planted.protected,
        "position": word_position(planted.word, len(clip.words),
                                  waqf=planted.word in clip.waqf),
        "boundary": _planted_boundary(clip, kind, planted),
        "length": len(clusters(form)),
        "form": form,
        "caught_on_inserted": None,
        "caught_on_neighbours": None,
    }
    if kind == "add_word":
        inserted = planted.word + 1
        twins = [inserted - 1, inserted + 1]
        neighbours = [word for word in twins if 1 <= word <= len(clip.words)]
        on_inserted = by_word.get((mode, inserted))
        row["caught_on_inserted"] = (on_inserted is not None
                                     and on_inserted["quality"] != CORRECT)
        row["caught_on_neighbours"] = any(
            by_word.get((mode, word)) is not None
            and by_word[(mode, word)]["quality"] != CORRECT for word in neighbours)
        row["quality"] = on_inserted["quality"] if on_inserted is not None else None
        row["not_graded"] = on_inserted is None
        caught = row["caught_on_inserted"] or row["caught_on_neighbours"]
        row["caught"] = caught if caught or on_inserted is not None else None
        row["statuses"] = (_statuses(on_inserted, request)
                           if row["caught"] is False or row["caught"] is None else [])
    else:
        trace = by_word.get((mode, planted.word + 1))
        row["quality"] = trace["quality"] if trace is not None else None
        row["not_graded"] = trace is None
        row["caught"] = None if trace is None else trace["quality"] != CORRECT
        row["statuses"] = (_statuses(trace, request)
                           if row["caught"] is False or row["caught"] is None else [])
    row["marked_is_planted"] = bool(
        planted.character is not None and by_word.get((mode, planted.word + 1)) is not None
        and planted.character in by_word[(mode, planted.word + 1)]["marked"])
    return row


def planted_sweep(clips: Sequence[SweepClip], engine, *, seed: int,
                  kinds: Sequence[str] = KINDS) -> PlantedSweep:
    """Plant one seeded mistake per kind and clip and grade both modes over the planted reference."""
    ordered = sorted(clips, key=lambda clip: clip.item)
    specs: list[tuple[SweepClip, str, Planted]] = []
    unplantable: dict[str, int] = {}
    for clip in ordered:
        for kind in kinds:
            if not plantable_words(kind, clip.words):
                unplantable[kind] = unplantable.get(kind, 0) + 1
                continue
            planted = plant(kind, clip.words,
                            random.Random(planting_seed(seed, clip.item, kind)))
            specs.append((clip, kind, planted))

    rows: list[dict] = []
    for batch in _batches(specs):
        requests = [_request(f"{clip.item}/{kind}", clip, planted.words,
                             _planted_times(clip, kind, planted))
                    for clip, kind, planted in batch]
        _, results = engine.run(requests)
        for (clip, kind, planted), request in zip(batch, requests):
            traces = word_traces(request, results, AUDIT_SCORINGS)
            by_word = {(trace["scoring"], trace["word"]): trace for trace in traces}
            for scoring in AUDIT_SCORINGS:
                rows.append(_planted_row(clip, kind, planted, scoring.name, by_word, request))
    return PlantedSweep(rows=rows, unplantable=dict(sorted(unplantable.items())))


def false_flags_per_1000(result: CleanSweep) -> dict[str, dict]:
    """False flags per 1,000 graded words, by mode, overall, by cause and by position.

    A position's denominator is that position's graded words, not the clip's: a pattern at
    waqf says nothing about words in the middle.
    """
    rates: dict[str, dict] = {}
    for mode, total in result.graded_words.items():
        findings = [finding for finding in result.findings if finding["mode"] == mode]
        kinds: dict[str, int] = {}
        by_position: dict[str, int] = {}
        for finding in findings:
            kinds[finding["error_kind"]] = kinds.get(finding["error_kind"], 0) + 1
            by_position[finding["position"]] = by_position.get(finding["position"], 0) + 1
        rates[mode] = {
            "all": 1000 * len(findings) / total if total else 0.0,
            "error_kind": {kind: 1000 * count / total for kind, count in sorted(kinds.items())},
            "position": {
                position: 1000 * by_position.get(position, 0) / count
                for position, count in sorted(result.words_by_position[mode].items()) if count
            },
        }
    return rates


def catch_rates(rows: Sequence[Mapping]) -> list[dict]:
    """Catch rate per kind, mode, position and word-length cell.

    ``n`` counts the graded planted words (a ``not_graded`` word is apart, neither caught nor
    missed), ``caught`` those below correct, and ``rate`` is ``caught / n`` (``None`` when
    nothing was graded).
    """
    cells: dict[tuple, dict] = {}
    for row in rows:
        key = (row["kind"], row["mode"], row["position"], length_bin(row["length"]))
        cell = cells.setdefault(key, {
            "kind": key[0], "mode": key[1], "position": key[2], "length_bin": key[3],
            "n": 0, "caught": 0, "not_graded": 0,
        })
        if row["caught"] is None:
            cell["not_graded"] += 1
        else:
            cell["n"] += 1
            cell["caught"] += bool(row["caught"])
    for cell in cells.values():
        cell["rate"] = cell["caught"] / cell["n"] if cell["n"] else None
    return [cells[key] for key in sorted(cells)]


# --- the committed run -----------------------------------------------------------------------

class _References:
    """The phonetizer's reference for an ayah, cached: 5,742 clips share 6,236 ayat."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[tuple[str, ...], str, list[int]]] = {}

    def __call__(self, label: str) -> tuple[tuple[str, ...], str, list[int]]:
        if label not in self._cache:
            uthmani = _uthmani_words(label)
            reference, offsets = hafs_word_reference()(uthmani)
            words = tuple(reference[offsets[i]:offsets[i + 1]]
                          for i in range(len(offsets) - 1))
            self._cache[label] = (words, reference, offsets)
        return self._cache[label]


def _end_samples(path: Path = DECODES_PATH) -> dict[str, int]:
    """Each gating clip's sample count, from its committed decode row (``end_sample``)."""
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return {record["key"]: int(record["end_sample"]) for record in map(json.loads, source)}


def _waqf_indices(words: Sequence[str]) -> frozenset[int]:
    """The words a waqf sign marks, by their position in the phonetizer's word list."""
    return frozenset(index for index, word in enumerate(words)
                     if any(char in WAQF_SIGNS for char in word))


def _tlog_clips(rows: Sequence[Mapping], word_times: Mapping[str, Mapping],
                samples: Mapping[str, int], references: _References) -> list[SweepClip]:
    clips = []
    for row in sorted(rows, key=lambda row: item_key(row["audio_path"])):
        key = item_key(row["audio_path"])
        record = word_times[key]
        if key not in samples:
            raise ValueError(f"{key} has word times but no committed decode row for its length")
        words, _, _ = references(row["filename_label"])
        times = (tuple(tuple(pair) for pair in record["word_times"])
                 if record["aligned"] else None)
        surah, ayah = (int(part) for part in row["filename_label"].split(":"))
        clips.append(SweepClip(key, surah, ayah, words, times, int(samples[key]),
                              _waqf_indices(_uthmani_words(row["filename_label"]))))
    return clips


def _mushaf_clips(harness, pace: float,
                  references: _References) -> tuple[list[SweepClip], list[dict]]:
    """Every ayah of the mushaf on synthetic word times at the tlog pace, and the unreadable."""
    connection = sqlite3.connect(f"file:{harness.quran_db}?mode=ro", uri=True)
    ayat = connection.execute("SELECT surah, ayah FROM ayahs ORDER BY surah, ayah").fetchall()
    clips: list[SweepClip] = []
    unreadable: list[dict] = []
    for surah, ayah in ayat:
        label = f"{surah}:{ayah}"
        try:
            words, reference, offsets = references(label)
        except (KeyError, IndexError) as error:
            unreadable.append({"ayah": label, "reason": f"{type(error).__name__}: {error}"})
            continue
        lengths = [len(clusters(reference[offsets[i]:offsets[i + 1]]))
                   for i in range(len(offsets) - 1)]
        times, n_samples = paced_word_times(lengths, pace)
        clips.append(SweepClip(label, surah, ayah, words, tuple(times), n_samples,
                              _waqf_indices(_uthmani_words(label))))
    return clips, unreadable


def _protected_split(rows: Sequence[Mapping]) -> list[dict]:
    """A haraka drop's catch rate split by whether the letter is a protected exemption."""
    cells: dict[tuple[str, bool], dict] = {}
    for row in rows:
        if row["kind"] != "haraka_drop":
            continue
        key = (row["mode"], bool(row["protected"]))
        cell = cells.setdefault(key, {"kind": "haraka_drop", "mode": key[0],
                                     "protected": key[1], "n": 0, "caught": 0,
                                     "not_graded": 0})
        if row["caught"] is None:
            cell["not_graded"] += 1
        else:
            cell["n"] += 1
            cell["caught"] += bool(row["caught"])
    for cell in cells.values():
        cell["rate"] = cell["caught"] / cell["n"] if cell["n"] else None
    return [cells[key] for key in sorted(cells)]


def _inserted_split(rows: Sequence[Mapping]) -> list[dict]:
    """An added word's catches split between the inserted copy and its neighbours."""
    cells: dict[str, dict] = {}
    for row in rows:
        if row["kind"] != "add_word":
            continue
        cell = cells.setdefault(row["mode"], {
            "kind": "add_word", "mode": row["mode"], "n": 0, "caught": 0,
            "not_graded": 0, "caught_on_inserted": 0, "caught_on_neighbours": 0,
        })
        if row["caught"] is None:
            cell["not_graded"] += 1
        else:
            cell["n"] += 1
            cell["caught"] += bool(row["caught"])
        cell["caught_on_inserted"] += bool(row["caught_on_inserted"])
        cell["caught_on_neighbours"] += bool(row["caught_on_neighbours"])
    for cell in cells.values():
        cell["rate"] = cell["caught"] / cell["n"] if cell["n"] else None
    return [cells[mode] for mode in sorted(cells)]


def _observed_exemptions(rows: Sequence[Mapping]) -> list[dict]:
    """Unprotected letters the engine never flagged a haraka drop on, with their plant counts."""
    planted: dict[str, int] = {}
    caught: dict[str, int] = {}
    for row in rows:
        if row["kind"] != "haraka_drop" or row["protected"]:
            continue
        letter = row["form"][0] if row["form"] else ""
        planted[letter] = planted.get(letter, 0) + 1
        caught[letter] = caught.get(letter, 0) + bool(row["caught"])
    return [{"letter": letter, "planted": planted[letter], "caught": caught.get(letter, 0)}
            for letter in sorted(planted) if caught.get(letter, 0) == 0]


def _planted_findings(rows: Sequence[Mapping]) -> list[dict]:
    """The planted mistakes the engine missed, in the shape :func:`group_findings` pools."""
    return [{
        "item": row["item"], "surah": row["surah"], "ayah": row["ayah"],
        "word": row["word"], "mode": row["mode"], "quality": row["quality"],
        "error_kind": row["kind"], "form": row["form"], "position": row["position"],
        "boundary": row["boundary"], "statuses": row["statuses"] or [None],
    } for row in rows if row["caught"] is False]


def _clean_section(result: CleanSweep) -> dict:
    return {
        "clips": result.clips,
        "graded_words": result.graded_words,
        "not_graded": result.not_graded,
        "words_by_position": result.words_by_position,
        "false_flags_per_1000": false_flags_per_1000(result),
    }


@dataclass(frozen=True)
class SweepResult:
    """A finished run: the committed report, the oracle word table, and the run note."""

    report: dict
    word_rows: list[dict]
    run_note: str
    unreadable: list[dict]


def run_sweep(harness, seed: int = SWEEP_SEED, log=None) -> SweepResult:
    """Both clean sweeps and the planted one over the committed populations.

    ``log`` is called once per stage with a one-line progress message; a full run is long
    enough that a silent process is indistinguishable from a stuck one.
    """
    def stage(message: str) -> None:
        if log is not None:
            log(message)

    word_times = load_word_times()
    rows = oracle_eligible(gating_rows(), word_times)
    references = _References()
    tlog_clips = _tlog_clips(rows, word_times, _end_samples(), references)
    aligned = [clip for clip in tlog_clips if clip.word_times]
    pace = median_speech_pace(
        [(len(clusters("".join(clip.words))), clip.word_times) for clip in aligned])
    stage(f"tlog: {len(tlog_clips):,} clips, pace {pace:.3f} Characters/s")
    clean_tlog = clean_sweep(tlog_clips, harness)
    stage(f"tlog clean: {len(clean_tlog.findings):,} findings")
    mushaf_clips, unreadable = _mushaf_clips(harness, pace, references)
    stage(f"mushaf: {len(mushaf_clips):,} clips, {len(unreadable)} unreadable")
    clean_mushaf = clean_sweep(mushaf_clips, harness)
    stage(f"mushaf clean: {len(clean_mushaf.findings):,} findings")
    planted = planted_sweep(tlog_clips, harness, seed=seed)
    stage(f"planted: {len(planted.rows):,} rows")
    findings = clean_tlog.findings + clean_mushaf.findings + _planted_findings(planted.rows)
    patterns: list[dict] = []
    for dimensions in PATTERN_DIMENSIONS:
        patterns.extend(group_findings(findings, dimensions))

    report = {
        "seed": seed,
        "muraja_commit": CURRENT_COMMIT,
        "modes": [scoring.name for scoring in AUDIT_SCORINGS],
        "pace": {"basis": "speech_span_median", "clips": len(aligned),
                 "characters_per_second": pace},
        "clean": {
            "tlog": _clean_section(clean_tlog),
            "mushaf": {
                **_clean_section(clean_mushaf),
                "ayat": len(mushaf_clips) + len(unreadable),
                "graded": len(mushaf_clips),
                "unreadable": len(unreadable),
                "characters_per_second": pace,
            },
        },
        "planted": {
            "unplantable": planted.unplantable,
            "catch_rates": catch_rates(planted.rows),
            "protected_split": _protected_split(planted.rows),
            "inserted_split": _inserted_split(planted.rows),
            "observed_exemptions": _observed_exemptions(planted.rows),
        },
        "patterns": patterns,
    }
    note = sweep_note(report, planted, unreadable)
    return SweepResult(report=report, word_rows=clean_tlog.word_rows, run_note=note,
                       unreadable=unreadable)


def oracle_summary(word_rows: Sequence[Mapping],
                   app_rows: Sequence[Mapping]) -> dict:
    """The App-arm doc's oracle section: the App flags (strict) the oracle shares.

    Both tables are keyed ``(item, word)``; a word the App flagged under strict is
    oracle-explained when the perfect decode flags it too. Words outside the oracle table
    (an App clip no longer in the population) are not counted as shared.
    """
    oracle = {(row["item"], row["word"]): row["strict"]["flagged"] for row in word_rows}
    app_flagged = [row for row in app_rows if row["strict"]["flagged"]]
    shared = sum(1 for row in app_flagged if oracle.get((row["item"], row["word"])))
    return {
        "scoring": "strict",
        "flagged": shared,
        "app_flagged": len(app_flagged),
        "share": shared / len(app_flagged) if app_flagged else 0.0,
        "oracle_table": "oracle_sweep/word_grades.jsonl.gz",
    }


def build_report(harness, seed: int = SWEEP_SEED) -> dict:
    """The committed report alone (the slow test re-derives it from the harness)."""
    return run_sweep(harness, seed=seed).report


def sweep_note(report: Mapping, planted: PlantedSweep, unreadable: Sequence[Mapping]) -> str:
    """The run note: provenance, the reading rules, and a draft issue for each candidate."""
    pace = report["pace"]
    tlog, mushaf = report["clean"]["tlog"], report["clean"]["mushaf"]
    lines = [
        "# S2 oracle sweep run (#308)",
        "",
        "Generated by `python -m tadabur.followalong_audit oracle-sweep`; never edited by hand.",
        "",
        f"- Seed {report['seed']}, Muraja `{report['muraja_commit']}`, modes "
        + ", ".join(report["modes"]) + ".",
        f"- tlog: {tlog['clips']:,} clips; {mushaf['ayat']:,} mushaf ayat "
        f"({mushaf['graded']:,} graded, {mushaf['unreadable']} unreadable).",
        f"- Pace: {pace['characters_per_second']:.3f} Characters/s over the speech span of "
        f"{pace['clips']:,} aligned clips (first word start to last word end).",
        f"- Planted rows: {len(planted.rows):,}; unplantable kinds: "
        + (", ".join(f"{kind} ({count})"
                     for kind, count in sorted(planted.unplantable.items()))
           or "none") + ".",
        "",
        "## Reading the rules",
        "",
        "- `not_graded`: the engine kept no row for the planted word. It is reported apart and"
        " counted as neither caught nor missed.",
        "- `protected`: a haraka drop on و ا ء ي is labelled protected (the app's exemption);"
        " every other letter is `other`.",
        "- `inserted` / `neighbour`: an added word is a copy of its neighbour inserted right"
        " after it; caught means a non-correct grade on the inserted word or its neighbours,"
        " and the two are reported apart.",
        "- `speech span`: the pace is the median Characters per second from a clip's first"
        " word start to its last word end, recomputed by this run.",
        "- Every word the engine does not grade correct under a perfect decode is a finding:"
        " no model and no ear is involved.",
        "",
        "## Unreadable mushaf ayat",
        "",
    ]
    lines += [f"- {entry['ayah']}: {entry['reason']}" for entry in unreadable] or ["- none"]
    lines += ["", "## Muraja issue drafts", "",
              "These drafts are **not filed** here: the owner files them in the private Muraja"
              " repo. Each covers ten or more distinct words.", ""]
    candidates = [pattern for pattern in report["patterns"] if pattern["candidate"]]
    if not candidates:
        lines.append("No pattern covers ten words.")
    for number, pattern in enumerate(candidates, start=1):
        lines += [f"### Draft {number}", "", issue_body(pattern)]
    return "\n".join(lines) + "\n"

