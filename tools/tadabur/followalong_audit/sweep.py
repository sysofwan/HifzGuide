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
**caught** when the kept grade of the engine word it is recited in is below correct; for an
added word, on the copied engine word or the word after it (the harness reports the ayah's own
phoneme words, so a copy inserted into the reference has no word of its own); and
**not_graded** when the engine kept no row for it, reported apart because neither a catch nor a
miss says anything about the engine. Every finding carries the per-check statuses, the word's
position, and whether a window boundary splits it, and every finding and every planted row is
committed as evidence beside the report. The grade is read in Muraja's own word numbering,
which a wasl merge makes fewer than the phonetizer's canonical words the reference is placed and
planted in; the planted rows keep both. Only the clean sweeps' findings are pooled into
patterns: a planted miss is catch capacity, not a fault of the follow-along or the reference.
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
from training.muraja_policy import (
    AUDIT_SCORINGS,
    CORRECT,
    CURRENT_COMMIT,
    PENDING,
    MurajaText,
    counterparts_in,
    word_traces,
)
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

#: The cause groupings the report pools every finding by.
PATTERN_DIMENSIONS = (
    ("error_kind",),
    ("error_kind", "position"),
    ("error_kind", "boundary"),
)

#: The reference-form grouping: a form seen once is no pattern, so the report keeps only the
#: forms that reach :data:`~tadabur.followalong_audit.patterns.CANDIDATE_WORDS`.
WORD_DIMENSIONS = ("form",)

#: The word-length cells a catch rate is reported in (Characters, both ends inclusive).
LENGTH_BINS = ((1, 2, "1-2"), (3, 4, "3-4"), (5, 6, "5-6"), (7, 10, "7-10"), (11, None, "11+"))


@dataclass(frozen=True)
class SweepClip:
    """One clip's reference in both coordinate systems: the phonetizer's canonical words, which
    the reference is placed and planted in, and Muraja's own phoneme words, which it reports and
    grades. A wasl merge recites two canonical words in one engine word.
    """

    item: str
    surah: int
    ayah: int
    #: The phonetizer's canonical phoneme words.
    words: tuple[str, ...]
    #: Muraja's own phoneme words, in its own numbering (``1..len(engine_words)``).
    engine_words: tuple[str, ...]
    #: The 1-based engine word each canonical word is recited in, ascending.
    word_of: tuple[int, ...]
    #: One ``(start_s, end_s)`` per canonical word, or ``None`` for a failed alignment.
    word_times: tuple[tuple[float, float], ...] | None = None
    n_samples: int = SAMPLE_RATE
    #: The 0-based engine words a printed pause sign marks.
    waqf: frozenset[int] = frozenset()
    #: Per canonical cluster, in ayah order: the ``(engine word, engine group)`` it lands in, or
    #: ``None`` where the phonetizer's reference and Muraja's do not line up.
    cluster_groups: tuple[tuple[int, int] | None, ...] = ()


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

    The reported words are Muraja's own (``1..len(engine_words)``), whatever the planting
    changed: its checks hear the planted reference, but it numbers its own phoneme words, and a
    wasl merge makes that fewer than the canonical words planted here.
    """
    reference = "".join(words)
    arm = oracle_checks(reference, _offsets(words), word_times, clip.n_samples)
    return session_request(item, clip.surah, clip.ayah, len(clip.engine_words), arm.checks)


def oracle_request(clip: SweepClip) -> dict:
    """The Oracle arm's harness request for one clip: its canonical words at its own word times,
    reported in Muraja's own word numbering (:func:`_request`)."""
    return _request(clip.item, clip, clip.words, clip.word_times)


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


def _engine_span(clip: SweepClip, word: int) -> tuple[float, float] | None:
    """The placed span of a 1-based engine word: the times of the canonical words it recites."""
    if clip.word_times is None:
        return None
    spans = [clip.word_times[index] for index, engine in enumerate(clip.word_of)
             if engine == word]
    if not spans:
        return None
    return spans[0][0], spans[-1][1]


def _engine_boundary(clip: SweepClip, word: int) -> bool | None:
    """Whether a transcriber window edge falls strictly inside engine word ``word``."""
    span = _engine_span(clip, word)
    return None if span is None else splits_window(*span, clip.n_samples)


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
                request["item"], len(clip.engine_words), traces, _oracle_causes))
            for scoring in AUDIT_SCORINGS:
                mode = scoring.name
                for word in range(1, len(clip.engine_words) + 1):
                    trace = by_word.get((mode, word))
                    index = word - 1
                    if trace is None or trace["quality"] == PENDING:
                        not_graded[mode] += 1
                        continue
                    graded[mode] += 1
                    position = word_position(index, len(clip.engine_words),
                                             waqf=index in clip.waqf)
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
                        "form": clip.engine_words[index],
                        "position": position,
                        "boundary": _engine_boundary(clip, word),
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
    """Whether a window edge splits the planted word's placed span.

    For an added word, the copy's own span (the insert takes half its source's); for a skipped
    word, the span the skip takes with it, which is known even though the word is gone; for
    every other kind, the placed span of the engine word the plant is recited in.
    """
    if clip.word_times is None:
        return None
    if kind == "add_word":
        times = _planted_times(clip, kind, planted)
        start, end = times[planted.inserted]
        return splits_window(start, end, clip.n_samples)
    if kind == "skip_word":
        start, end = clip.word_times[planted.word]
        return splits_window(start, end, clip.n_samples)
    return _engine_boundary(clip, clip.word_of[planted.word])


def _planted_group(clip: SweepClip, planted: Planted) -> tuple[int, int] | None:
    """The engine ``(word, group)`` the planted Character lands in, or ``None`` when the plant
    has no single Character (skip, add) or the two references do not line up there.

    The app marks the printed graphemes of the group it flags (:func:`highlights`), so a kept
    error's ``group_index`` equal to this one is an error whose marked letter is the planted one.
    """
    if planted.character is None:
        return None
    before = sum(len(clusters(word)) for word in clip.words[:planted.word])
    index = before + planted.character
    return clip.cluster_groups[index] if index < len(clip.cluster_groups) else None


def _below_correct(trace: Mapping | None) -> bool:
    """Whether a kept grade is a verdict below correct (a pending word is no verdict)."""
    return trace is not None and trace["quality"] not in (CORRECT, PENDING)


def _planted_row(clip: SweepClip, kind: str, planted: Planted, mode: str,
                 by_word: Mapping[tuple[str, int], Mapping], request: Mapping) -> dict:
    """One row: the planted word's kept grade, and whether the engine caught the mistake.

    The grade is read at the engine word the plant is recited in, not at the canonical word's
    ordinal: a wasl merge recites two canonical words as one engine word.
    """
    form = clip.words[planted.word] if kind == "skip_word" else planted.words[planted.word]
    engine = clip.word_of[planted.word]
    row = {
        "item": clip.item,
        "surah": clip.surah,
        "ayah": clip.ayah,
        "kind": kind,
        "mode": mode,
        "word": engine,
        "reference_word": planted.word + 1,
        "label": planted.label,
        "protected": planted.protected,
        "position": word_position(engine - 1, len(clip.engine_words),
                                  waqf=(engine - 1) in clip.waqf),
        "boundary": _planted_boundary(clip, kind, planted),
        "length": len(clusters(clip.words[planted.word])),
        "form": form,
        "caught_on_copied": None,
        "caught_on_following": None,
    }
    if kind == "add_word":
        # The harness reports the ayah's own phoneme words, so the copy has no word number of
        # its own: the insertion can displace the grade of the copied engine word and of the
        # word after it, and those are the two the split reports.
        affected = [engine, engine + 1] if engine < len(clip.engine_words) else [engine]
        traces = {word: by_word.get((mode, word)) for word in affected}
        verdicts = [trace["quality"] for trace in traces.values()
                    if trace is not None and trace["quality"] != PENDING]
        row["caught_on_copied"] = _below_correct(traces[engine])
        row["caught_on_following"] = _below_correct(traces.get(engine + 1))
        row["quality"] = traces[engine]["quality"] if traces[engine] is not None else None
        row["not_graded"] = not verdicts
        row["caught"] = None if not verdicts else any(q != CORRECT for q in verdicts)
        row["statuses"] = (_statuses(traces[engine], request)
                           if row["caught"] is not True else [])
    else:
        trace = by_word.get((mode, engine))
        pending = trace is not None and trace["quality"] == PENDING
        row["quality"] = trace["quality"] if trace is not None else None
        row["not_graded"] = trace is None or pending
        row["caught"] = (None if row["not_graded"]
                         else trace["quality"] != CORRECT)
        row["statuses"] = (_statuses(trace, request)
                           if row["caught"] is False or row["caught"] is None else [])
    group = _planted_group(clip, planted)
    trace = by_word.get((mode, engine))
    row["marked_is_planted"] = (
        None if group is None or trace is None else
        group[0] == engine and any(error["group_index"] == group[1]
                                   for error in trace["errors"]))
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


def _tally(cell: dict, row: Mapping) -> None:
    """Count one planted row into ``cell``: a ``not_graded`` word is apart from caught/missed."""
    if row["caught"] is None:
        cell["not_graded"] += 1
    else:
        cell["n"] += 1
        cell["caught"] += bool(row["caught"])


def _cells(cells: Mapping) -> list[dict]:
    """The cells in key order, each finished with its rate: caught over graded, ``None`` if none."""
    for cell in cells.values():
        cell["rate"] = cell["caught"] / cell["n"] if cell["n"] else None
    return [cells[key] for key in sorted(cells)]


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
        _tally(cell, row)
    return _cells(cells)


# --- the committed run -----------------------------------------------------------------------

@dataclass(frozen=True)
class AyahReference:
    """One ayah's two references: the phonetizer's canonical words, which the reference is
    placed and planted in, and Muraja's own phoneme words, which it reports and grades."""

    words: tuple[str, ...]
    engine_words: tuple[str, ...]
    #: The 1-based engine word each canonical word is recited in, ascending.
    word_of: tuple[int, ...]
    #: Per canonical cluster, in ayah order: ``(engine word, engine group)`` or ``None``.
    cluster_groups: tuple[tuple[int, int] | None, ...]
    #: The 0-based engine words a printed pause sign marks.
    waqf: frozenset[int]


def _cluster_starts(offset: int, word: str) -> list[int]:
    """Each cluster's first code-point index in the ayah's spaceless canonical reference."""
    starts, at = [], offset
    for character in clusters(word):
        starts.append(at)
        at += len(character)
    return starts


def _canonical_words(label: str) -> tuple[tuple[str, ...], str, tuple[int, ...]]:
    """The phonetizer's canonical phoneme words of ``"surah:ayah"``, their spaceless reference
    and word offsets: the coordinates the reference is placed and the mistakes planted in."""
    uthmani = _uthmani_words(label)
    reference, offsets = hafs_word_reference()(uthmani)
    words = tuple(reference[offsets[index]:offsets[index + 1]]
                  for index in range(len(offsets) - 1))
    return words, reference, tuple(offsets)


def ayah_references(text) -> AyahReferences:
    """The ayah references the oracle arm reads, cached, over ``text`` (a
    :class:`training.muraja_policy.MurajaText`). One accessor, so F1-align's reference is the
    oracle's own by construction."""
    return AyahReferences(text, _canonical_words)


class AyahReferences:
    """An ayah's two references, cached: 5,742 clips share 6,236 ayat.

    ``canonical`` gives the phonetizer's words, ``text`` gives Muraja's own reference. The two
    are the same recitation (a pausal taa marbuta aside, :func:`counterparts_in` maps across),
    Muraja's separated by the words it reports, so a canonical word's first mapped Character
    places it in an engine word, and each canonical cluster in an engine group.
    """

    def __init__(self, text, canonical) -> None:
        self._text = text
        self._canonical = canonical
        self._cache: dict[str, AyahReference] = {}

    def __call__(self, label: str) -> AyahReference:
        if label not in self._cache:
            self._cache[label] = self._build(label)
        return self._cache[label]

    def _build(self, label: str) -> AyahReference:
        words, reference, offsets = self._canonical(label)
        surah, ayah = (int(part) for part in label.split(":"))
        phonemes = self._text.phonemes(surah, ayah)
        # counterparts_in keys Muraja's Characters; the plants are laid on the phonetizer's.
        placed = {canonical: at for at, canonical
                  in counterparts_in(reference, phonemes).items()}
        engine_words = tuple(phonemes.split(" "))
        groups = [tuple(self._text.word_groups(surah, ayah, number))
                  for number in range(1, len(engine_words) + 1)]

        def engine_of(at: int) -> int:
            return phonemes[:at].count(" ") + 1

        word_of: list[int] = []
        cluster_groups: list[tuple[int, int] | None] = []
        for index, word in enumerate(words):
            starts = _cluster_starts(offsets[index], word)
            mapped = [placed[position] for position in starts if position in placed]
            # A word whose every Character is spelled differently is recited in the engine word
            # before it: the phonemes are monotone, and the merge keeps them together.
            word_of.append(engine_of(mapped[0]) if mapped else (word_of[-1] if word_of else 1))
            for position in starts:
                at = placed.get(position)
                group = None if at is None else next(
                    (number for number, (start, end)
                     in enumerate(groups[engine_of(at) - 1]) if start <= at < end), None)
                cluster_groups.append(None if at is None or group is None
                                      else (engine_of(at), group))
        waqf = frozenset(index for index in range(len(engine_words))
                         if any(sign in self._text.printed(surah, ayah, index + 1)
                                for sign in WAQF_SIGNS))
        return AyahReference(words, engine_words, tuple(word_of),
                              tuple(cluster_groups), waqf)


def end_samples(path: Path = DECODES_PATH) -> dict[str, int]:
    """Each gating clip's sample count, from its committed decode row (``end_sample``)."""
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return {record["key"]: int(record["end_sample"]) for record in map(json.loads, source)}


def _clip(item: str, label: str, ayah: AyahReference,
          word_times: tuple[tuple[float, float], ...] | None,
          n_samples: int) -> SweepClip:
    """A sweep clip over one ayah's record, either side of its canon."""
    surah, ayah_number = (int(part) for part in label.split(":"))
    return SweepClip(item, surah, ayah_number, ayah.words, ayah.engine_words, ayah.word_of,
                     word_times, n_samples, ayah.waqf, ayah.cluster_groups)


def tlog_clips(rows: Sequence[Mapping], word_times: Mapping[str, Mapping],
                samples: Mapping[str, int], references: AyahReferences) -> list[SweepClip]:
    clips = []
    for row in sorted(rows, key=lambda row: item_key(row["audio_path"])):
        key = item_key(row["audio_path"])
        record = word_times[key]
        if key not in samples:
            raise ValueError(f"{key} has word times but no committed decode row for its length")
        ayah = references(row["filename_label"])
        times = (tuple(tuple(pair) for pair in record["word_times"])
                 if record["aligned"] else None)
        if times is not None and len(times) != len(ayah.words):
            raise ValueError(f"{key}: {len(times)} word times for {len(ayah.words)} words")
        clips.append(_clip(key, row["filename_label"], ayah, times, int(samples[key])))
    return clips


def _mushaf_clips(harness, pace: float,
                  references: AyahReferences) -> tuple[list[SweepClip], list[dict]]:
    """Every ayah of the mushaf on synthetic word times at the tlog pace, and the unreadable."""
    connection = sqlite3.connect(f"file:{harness.quran_db}?mode=ro", uri=True)
    ayat = connection.execute("SELECT surah, ayah FROM ayahs ORDER BY surah, ayah").fetchall()
    clips: list[SweepClip] = []
    unreadable: list[dict] = []
    for surah, ayah in ayat:
        label = f"{surah}:{ayah}"
        try:
            record = references(label)
        except (KeyError, IndexError) as error:
            unreadable.append({"ayah": label, "reason": f"{type(error).__name__}: {error}"})
            continue
        lengths = [len(clusters(word)) for word in record.words]
        times, n_samples = paced_word_times(lengths, pace)
        clips.append(_clip(label, label, record, tuple(times), n_samples))
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
        _tally(cell, row)
    return _cells(cells)


def _inserted_split(rows: Sequence[Mapping]) -> list[dict]:
    """An added word's catches split between the copied word and the word after it."""
    cells: dict[str, dict] = {}
    for row in rows:
        if row["kind"] != "add_word":
            continue
        cell = cells.setdefault(row["mode"], {
            "kind": "add_word", "mode": row["mode"], "n": 0, "caught": 0,
            "not_graded": 0, "caught_on_copied": 0, "caught_on_following": 0,
        })
        _tally(cell, row)
        cell["caught_on_copied"] += bool(row["caught_on_copied"])
        cell["caught_on_following"] += bool(row["caught_on_following"])
    return _cells(cells)


def _observed_exemptions(rows: Sequence[Mapping]) -> list[dict]:
    """Unprotected letters the engine never flagged a haraka drop on, with their plant counts."""
    planted: dict[str, int] = {}
    caught: dict[str, int] = {}
    for row in rows:
        if row["kind"] != "haraka_drop" or row["protected"]:
            continue
        letter = row["label"][1:]  # a drop's label is ``-<letter>``
        planted[letter] = planted.get(letter, 0) + 1
        caught[letter] = caught.get(letter, 0) + bool(row["caught"])
    return [{"letter": letter, "planted": planted[letter], "caught": caught.get(letter, 0)}
            for letter in sorted(planted) if caught.get(letter, 0) == 0]


def _clean_section(result: CleanSweep) -> dict:
    return {
        "clips": result.clips,
        "findings": len(result.findings),
        "graded_words": result.graded_words,
        "not_graded": result.not_graded,
        "words_by_position": result.words_by_position,
        "false_flags_per_1000": false_flags_per_1000(result),
    }


#: The two committed evidence tables, beside the report they explain.
CLEAN_FINDINGS_FILE = "clean_findings.jsonl.gz"
PLANTED_ROWS_FILE = "planted_rows.jsonl.gz"


@dataclass(frozen=True)
class SweepResult:
    """A finished run: the committed report, the oracle word table, the two evidence tables, and
    the run note."""

    report: dict
    word_rows: list[dict]
    clean_findings: list[dict]
    planted_rows: list[dict]
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
    references = ayah_references(MurajaText(harness.quran_db))
    clips = tlog_clips(rows, word_times, end_samples(), references)
    aligned = [clip for clip in clips if clip.word_times]
    pace = median_speech_pace(
        [(len(clusters("".join(clip.words))), clip.word_times) for clip in aligned])
    stage(f"tlog: {len(clips):,} clips, pace {pace:.3f} Characters/s")
    clean_tlog = clean_sweep(clips, harness)
    stage(f"tlog clean: {len(clean_tlog.findings):,} findings")
    mushaf_clips, unreadable = _mushaf_clips(harness, pace, references)
    stage(f"mushaf: {len(mushaf_clips):,} clips, {len(unreadable)} unreadable")
    clean_mushaf = clean_sweep(mushaf_clips, harness)
    stage(f"mushaf clean: {len(clean_mushaf.findings):,} findings")
    planted = planted_sweep(clips, harness, seed=seed)
    stage(f"planted: {len(planted.rows):,} rows")
    findings = clean_tlog.findings + clean_mushaf.findings
    patterns: list[dict] = []
    for dimensions in PATTERN_DIMENSIONS:
        patterns.extend(group_findings(findings, dimensions))
    patterns.extend(pattern for pattern in group_findings(findings, WORD_DIMENSIONS)
                    if pattern["candidate"])
    evidence_findings = ([{**finding, "sweep": "tlog"} for finding in clean_tlog.findings]
                         + [{**finding, "sweep": "mushaf"} for finding in clean_mushaf.findings])

    report = {
        "seed": seed,
        "muraja_commit": CURRENT_COMMIT,
        "modes": [scoring.name for scoring in AUDIT_SCORINGS],
        "pace": {"basis": "speech_span_median", "clips": len(aligned),
                 "characters_per_second": pace},
        "evidence": {"clean_findings": CLEAN_FINDINGS_FILE, "planted_rows": PLANTED_ROWS_FILE},
        "clean": {
            "tlog": _clean_section(clean_tlog),
            "mushaf": {
                **_clean_section(clean_mushaf),
                "ayat": len(mushaf_clips) + len(unreadable),
                "graded": len(mushaf_clips),
                "unreadable": len(unreadable),
                "unreadable_ayat": list(unreadable),
                "characters_per_second": pace,
            },
        },
        "planted": {
            "rows": len(planted.rows),
            "unplantable": planted.unplantable,
            "catch_rates": catch_rates(planted.rows),
            "protected_split": _protected_split(planted.rows),
            "inserted_split": _inserted_split(planted.rows),
            "observed_exemptions": _observed_exemptions(planted.rows),
        },
        "patterns": patterns,
    }
    note = sweep_note(report)
    return SweepResult(report=report, word_rows=clean_tlog.word_rows,
                       clean_findings=evidence_findings, planted_rows=planted.rows,
                       run_note=note, unreadable=unreadable)


def oracle_summary(word_rows: Sequence[Mapping],
                   app_rows: Sequence[Mapping]) -> dict:
    """The App-arm doc's oracle section: the oracle's own flags and the App flags it shares.

    Both tables are keyed ``(item, word)``; a word the App flagged under strict is
    oracle-explained when the perfect decode flags it too. Words outside the oracle table
    (an App clip no longer in the population) are not counted as shared.
    """
    oracle = {(row["item"], row["word"]): row["strict"]["flagged"] for row in word_rows}
    app_flagged = [row for row in app_rows if row["strict"]["flagged"]]
    shared = sum(1 for row in app_flagged if oracle.get((row["item"], row["word"])))
    return {
        "scoring": "strict",
        "flagged": sum(1 for flagged in oracle.values() if flagged),
        "app_flagged": len(app_flagged),
        "shared": shared,
        "share": shared / len(app_flagged) if app_flagged else 0.0,
        "oracle_table": "oracle_sweep/word_grades.jsonl.gz",
    }


def build_report(harness, seed: int = SWEEP_SEED) -> dict:
    """The committed report alone (the slow test re-derives it from the harness)."""
    return run_sweep(harness, seed=seed).report


def sweep_note(report: Mapping) -> str:
    """The run note: provenance, the reading rules, and a draft issue for each candidate.

    A pure function of the committed report, so the note can be regenerated from the JSON.
    """
    pace = report["pace"]
    tlog, mushaf = report["clean"]["tlog"], report["clean"]["mushaf"]
    planted = report["planted"]
    unreadable = mushaf["unreadable_ayat"]
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
        f"- Planted rows: {planted['rows']:,}; unplantable kinds: "
        + (", ".join(f"{kind} ({count})"
                     for kind, count in sorted(planted["unplantable"].items()))
           or "none") + ".",
        f"- Evidence: `{report['evidence']['clean_findings']}` "
        f"({tlog['findings'] + mushaf['findings']:,} false flags) and "
        f"`{report['evidence']['planted_rows']}` ({planted['rows']:,} planted rows), each with"
        " its per-check statuses, position and window boundary.",
        "",
        "## Reading the rules",
        "",
        "- `not_graded`: the engine kept no row for the planted word. It is reported apart and"
        " counted as neither caught nor missed.",
        "- `protected`: a haraka drop on و ا ء ي is labelled protected (the app's exemption);"
        " every other letter is `other`.",
        "- `word` / `reference_word`: the planted word's engine word (the one Muraja reports and"
        " grades) and the canonical word it was planted in. A wasl merge recites two canonical"
        " words in one engine word.",
        "- `copied` / `following`: an added word is a copy of its neighbour inserted right"
        " after it. The harness reports the ayah's own phoneme words, so the copy has no word"
        " number of its own: caught means a non-correct grade on the copied word or the word"
        " after it, and the two are reported apart.",
        "- `marked_is_planted`: whether an error's marked phoneme group is the group the plant"
        " landed in, compared in Muraja's own group numbering; `None` for a skipped or added"
        " word, which has no single Character.",
        "- `speech span`: the pace is the median Characters per second from a clip's first"
        " word start to its last word end, recomputed by this run.",
        "- Every word the engine does not grade correct under a perfect decode is a finding:"
        " no model and no ear is involved. Only those findings are pooled into patterns; a"
        " planted mistake the engine misses is catch capacity, read from the catch rates.",
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

