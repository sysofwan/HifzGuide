"""False flags in Muraja's own device sessions, replayed through its engine (#151).

Every app-outcome number before this one is simulated (#90, ``docs/truth-baseline.md``): Tadabur
audio, PyTorch decodes, an idealised check schedule. Muraja's ``tools/test_logs`` hold real
sessions from the device: every check the app ran (its ``Update:`` lines,
``FollowAlong/FollowAlongManager+iOS.swift:744``), decoded by the CoreML model on the phone, with
real timing, the speech detector and the silence flush. The owner (2026-10-14): the logs hold no
intentional pronunciation or substitution mistake, and each has docs on what went wrong or what it
tests. So a word the app shows as not correct is a **false flag** unless the log's docs say what
else it is.

**The inventory** (``device_logs/inventory.json``, written by hand from Muraja's commits, issues,
PRs, ADRs and baseline intents) records per log the model it ran and why, what it tests, whether
it is excluded (:data:`EXCLUSIONS`), and its **explained** words: the words its docs, or its own
decode where the docs name the session's purpose but not the words, show were skipped, re-read
or never recited (each with its basis, :data:`BASES`, and its source). What a log file itself
says is read here only as positions and counts (:class:`LogFacts`); no log text is written out.

**Replay.** ``muraja_harness/log_replay.swift`` runs Muraja's own ``tools/replay_log.swift``
(``executeReplay``, the replay its ``run_log_tests.sh`` checks) once per log and engine
configuration of :data:`training.muraja_policy.SCORINGS`, at the pin (:data:`MURAJA_COMMIT`) and
at Muraja ``main`` (:data:`MURAJA_MAIN`), each on its own checkout's logs. **Parity comes first**:
the shipped default must reproduce each log's committed ``baseline.json`` (every word's quality
and the final position) at both commits, or nothing is reported.

**Attribution.** Each word a scoring shows as not correct (:func:`muraja_policy.displayed`) is
attributed by :func:`training.word_flags.flag` to the errors of the grade the engine kept: the
tashkeel mark and whether it was dropped or changed, a soft pair, a non-soft target pair
(``ذ↔ظ``, ``س↔ش``, ``ط↔ق``), another consonant, a collapsed geminate, the engine's skip marking, or
no recorded error. A word may carry several causes.

**Rates.** Per 1,000 graded words: each word a session kept a grade for stands for a word recited.
Pooled over a model's included logs, with an interval from a bootstrap that resamples logs
(``acceptance_stats.ratio_interval``), since the words of one session are not independent; with
too few logs there is none. Two views: every graded word, and the explained words left out of
both the flags and the words.

**The logged overlap.** The app logs each update's overlap cut to 30 Characters
(``unconfirmed.prefix(30)``, ``FollowAlongManager+iOS.swift:744``), so a replay of a session on the
1 s hop feeds the engine a shorter preview than it saw. Two tests bound the effect: the device
logs replayed with each preview lengthened to the 40 Characters the transcriber's own line keeps
(``RealtimeTranscriber+iOS.swift:642``), and #90's simulated word sample regraded with its
overlaps cut to 30 Characters.

Usage (from ``tools/``, on a Mac, with the harness built at the pin and at Muraja ``main``)::

    bash muraja_harness/build.sh
    bash muraja_harness/build.sh 3a49a1c16d3f9391d8b43e51a5270524965237a2
    python -m training.device_logs
"""

from __future__ import annotations

import json
import re
import statistics
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from tadabur.exclusions import current
from tadabur.listening_session import read_verdicts
from tadabur.staged_audio import REGISTRY_PATH, load_staged_clips
from tadabur.truth_sites import SOFT_PAIRS, UNFORGIVEN_PAIRS
from tadabur.word_sample import load_words, verified_words
from training import truth_baseline
from training.acceptance_stats import ratio_interval
from training.decoding import STOP, clusters, make_check
from training.muraja_policy import (
    ALLOWANCES_ON,
    CORRECT,
    HARNESS_ROOT,
    LOG_REPLAY_BINARY,
    MURAJA_COMMIT,
    SCORINGS,
    SHIPPED_DEFAULT,
    WORD_FLAGGED,
    Harness,
    MurajaText,
    Scoring,
    SiteGrade,
    displayed,
    engine_runs,
    require_harness,
)
from training.site_outcomes import CORRECT_SIDE
from training.tashkeel_eval import write_text_atomically
from training.word_flags import CAUSES, flag

#: Muraja ``main`` when this report was made; it differs from the pin only outside FollowAlong.
MURAJA_MAIN = "3a49a1c16d3f9391d8b43e51a5270524965237a2"
DEVICE_LOGS_DIR = Path(__file__).parent.parent / "device_logs"
INVENTORY_PATH = DEVICE_LOGS_DIR / "inventory.json"
REPORT_PATH = DEVICE_LOGS_DIR / "report.json"
DOC_PATH = Path(__file__).parent.parent.parent / "docs" / "device-log-false-flags.md"

H448 = "h448"
TEACHER = "teacher"
MODELS = (H448, TEACHER)
#: Why a log is left out: built from reference phonemes, no device session (a mined Tadabur clip
#: transcribed offline), the same session as another log, or a session whose docs record a real
#: recitation mistake (a slip into another passage's wording).
EXCLUSIONS = ("synthetic", "not_a_device_session", "duplicate_session", "documented_mistake")
DOCUMENTED_MISTAKE = "documented_mistake"
#: What an explained word was: skipped by the reader, inside a deliberate re-read, or never
#: recited (before the session's start, after its end, or placed on a twin).
KINDS = ("skip", "re-read", "not_recited")
#: Where the explanation comes from: Muraja's docs, or the log's own decode (and Start line).
BASES = ("docs", "decode")
#: A slip region (kind ``slip``) is a sensitivity's only: the words in and around a documented
#: slip into another passage (:class:`Slip`), by one mechanical rule in two widths (:func:`regions`).
SLIP = "slip"
SLIP_WIDTHS = ("narrow", "wide")
#: The sensitivity beside each headline model row: every session of the model, a slip excluded
#: or not, with its slip regions left out as explained words are.
SLIPS_EXCLUDED = "all sessions, slip regions excluded"

#: The two streaming protocols the logs were recorded under: 1 s hops from Muraja 9f81f0a
#: (2026-04-29, the shipped one, the simulation's), 3 s hops before it.
ONE_SECOND = "1 s hop"
THREE_SECOND = "3 s hop"
#: Characters of each update's overlap the app logs (``FollowAlongManager+iOS.swift:744``).
OVERLAP_CAP = 30
#: A median inference time below this marks the h448 student, one CoreML model; the teacher runs
#: six chunks in turn and every one of its sessions is far slower (the doc gives both ranges).
STUDENT_INFERENCE_MS = 60

#: The groups rates are pooled over: each model, and the teacher's 1 s-hop sessions alone (the
#: protocol the simulation models).
TEACHER_ONE_SECOND = "teacher, 1 s hop"
GROUPS = (H448, TEACHER, TEACHER_ONE_SECOND)
ALL_WORDS = "all"
UNEXPLAINED = "unexplained"
#: One soft pair not forgiven at a time, every other allowance on: the harness's one-pair switch
#: (``muraja_harness/scoring.swift``) over Muraja's own pair set.
PAIR_SCORINGS = tuple(replace(ALLOWANCES_ON, name=f"not_forgiven:{pair}", soft_pairs_disabled=(pair,))
                      for pair in sorted(SOFT_PAIRS))
#: Each allowance and the scoring with it alone off, against ``allowances_on``; then the target
#: pairs no Muraja allowance forgives (``None``).
ALLOWANCE_ROWS = (
    *((f"soft pair {s.soft_pairs_disabled[0]}", s.name) for s in PAIR_SCORINGS),
    ("all six soft pairs (Muraja's one switch)", "soft_pair_forgiveness_off"),
    ("shaddah suppression", "shaddah_suppression_off"),
    ("the Tashkeel toggle", "shipped_default"),
    *((f"target pair {pair}", None) for pair in sorted(UNFORGIVEN_PAIRS)),
)


def sensitivity(model: str, width: str) -> str:
    """The name of a model's slip sensitivity at one width."""
    return f"{model}, {SLIPS_EXCLUDED}, {width}"


#: The populations the allowance table reads: the headline groups and each model's sensitivity.
ALLOWANCE_GROUPS = (TEACHER_ONE_SECOND, TEACHER, *(sensitivity(TEACHER, w) for w in SLIP_WIDTHS),
                    H448, *(sensitivity(H448, w) for w in SLIP_WIDTHS))
#: The simulation's arm for each model (#90's word sample, the recorded check sequence).
SIMULATION_ARMS = {H448: "h448/checks", TEACHER: "base/checks"}


def position(word: str) -> tuple[int, int, int]:
    surah, ayah, index = (int(part) for part in word.split(":"))
    return surah, ayah, index


# --- the inventory ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Explained:
    """A run of a log's words (``first`` to ``last``, inclusive) that its docs or decode explain."""

    first: str
    last: str
    kind: str
    basis: str
    source: str

    def covers(self, word: str) -> bool:
        return position(self.first) <= position(word) <= position(self.last)


@dataclass(frozen=True)
class Slip:
    """A documented slip: the reader said another passage's (``twin``'s) wording from word
    ``start`` to word ``end`` (the first and last word it replaced, or, for added words, the
    words the detector placed them between), as its docs record it."""

    start: str
    end: str
    twin: str | None
    basis: str
    source: str


def regions(slip: Slip, ayah_length: Callable[[int, int], int]) -> dict[str, tuple[Explained, ...]]:
    """A slip's region at each width. **Narrow:** from ``start`` to the end of ``end``'s ayah.
    **Wide:** from the first word of ``start``'s ayah to the end of the ayah after ``end``'s,
    where the correction and the catch-up land. Both add the twin ayah in full."""
    (s, a, _), (_, b, _) = position(slip.start), position(slip.end)
    twin = ()
    if slip.twin is not None:
        ts, ta = (int(x) for x in slip.twin.split(":"))
        twin = (Explained(f"{ts}:{ta}:1", f"{ts}:{ta}:{ayah_length(ts, ta)}", SLIP, slip.basis, slip.source),)
    narrow = Explained(slip.start, f"{s}:{b}:{ayah_length(s, b)}", SLIP, slip.basis, slip.source)
    wide = Explained(f"{s}:{a}:1", f"{s}:{b + 1}:{ayah_length(s, b + 1)}", SLIP, slip.basis, slip.source)
    return {"narrow": (narrow, *twin), "wide": (wide, *twin)}


@dataclass(frozen=True)
class LogEntry:
    """One log of Muraja's ``tools/test_logs``, as the inventory records it."""

    log: str
    commit: str
    committed: str
    model: str | None
    model_basis: str | None
    app_version: str | None
    purpose: str
    sources: tuple[str, ...]
    exclusion: str | None
    exclusion_detail: str | None
    explained: tuple[Explained, ...]
    #: Per width of :data:`SLIP_WIDTHS`, the regions of :attr:`slips` (:func:`with_regions`).
    slip_regions: Mapping[str, tuple[Explained, ...]] = field(default_factory=dict)
    slips: tuple[Slip, ...] = ()

    @property
    def included(self) -> bool:
        return self.exclusion is None

    def explanation(self, word: str) -> Explained | None:
        return next((e for e in self.explained if e.covers(word)), None)

    def in_slip(self, word: str, width: str) -> bool:
        return any(region.covers(word) for region in self.slip_regions.get(width, ()))


def load_inventory(path: Path = INVENTORY_PATH) -> tuple[LogEntry, ...]:
    entries = []
    for record in json.loads(path.read_text(encoding="utf-8")):
        excluded = record["excluded"] or {}
        entry = LogEntry(
            log=record["log"], commit=record["commit"], committed=record["committed"],
            model=record["model"], model_basis=record["model_basis"],
            app_version=record["app_version"], purpose=record["purpose"],
            sources=tuple(record["sources"]), exclusion=excluded.get("reason"),
            exclusion_detail=excluded.get("detail"),
            explained=tuple(Explained(**e) for e in record["explained"]),
            slips=tuple(Slip(slip["from"], slip["to"], slip["twin"], slip["basis"], slip["source"])
                        for slip in record["slips"]),
        )
        problems = [
            entry.model not in (*MODELS, None) and f"model {entry.model}",
            entry.exclusion not in (*EXCLUSIONS, None) and f"exclusion {entry.exclusion}",
            entry.included and entry.model is None and "an included log has no model",
            *(f"explained {e.kind}/{e.basis}" for e in entry.explained
              if e.kind not in KINDS or e.basis not in BASES or position(e.first) > position(e.last)),
            (entry.exclusion == DOCUMENTED_MISTAKE) != bool(entry.slips) and "a documented mistake needs its slips",
            *(f"slip {x.start}-{x.end}/{x.basis}" for x in entry.slips
              if x.basis not in BASES or position(x.start) > position(x.end)),
        ]
        if any(problems):
            raise ValueError(f"{entry.log}: {[p for p in problems if p]}")
        entries.append(entry)
    if [e.log for e in entries] != sorted({e.log for e in entries}):
        raise ValueError("the inventory must list each log once, in name order")
    return tuple(entries)


def with_regions(entry: LogEntry, ayah_length: Callable[[int, int], int]) -> LogEntry:
    """The entry with its slips' regions, each width the union over its slips."""
    by_slip = [regions(slip, ayah_length) for slip in entry.slips]
    return replace(entry, slip_regions={w: tuple(r for slip in by_slip for r in slip[w]) for w in SLIP_WIDTHS})


# --- what a log says, as positions and counts ------------------------------------------------------
_UPDATE = re.compile(r"Update: type=(\w+), (hop|confirmed)=(.*?), (?:overlap|unconfirmed)=(.*), didFlush=")
_START = re.compile(r"Start: position=(\d+):(\d+)(?::(\d+))?, page=(\d+)")
_LEGACY_START = re.compile(r"expected=(\d+):(\d+)")
_PAGE = re.compile(r"(?:^|\] )Page: (\d+)\s*$")
_INFERENCE = re.compile(r"\b(?:Preview|Window)(?: inference)? (\d+)ms")
_TRANSCRIBED_PREVIEW = re.compile(r"\] Preview \d+ms — (.*)$")


@dataclass(frozen=True)
class LogFacts:
    """What a log file says about its session, without its text."""

    start: str
    pages: tuple[int, ...]
    protocol: str
    updates: int
    #: 1 s-hop updates whose logged overlap is exactly :data:`OVERLAP_CAP` Characters (cut).
    capped_overlaps: int
    median_inference_ms: float | None


def read_facts(text: str) -> LogFacts:
    lines = text.splitlines()
    start = next((m for line in lines if (m := _START.search(line))), None)
    if start is not None:
        start_word = f"{start[1]}:{start[2]}:{start[3] or 0}"
        pages = [int(start[4])]
    else:
        legacy = next(m for line in lines if (m := _LEGACY_START.search(line)))
        start_word, pages = f"{legacy[1]}:{legacy[2]}:0", []
    pages += [int(m[1]) for line in lines if (m := _PAGE.search(line))]
    updates = [m for line in lines if (m := _UPDATE.search(line))]
    one_second = any(m[2] == "hop" for m in updates)
    times = [int(m[1]) for line in lines if (m := _INFERENCE.search(line))]
    return LogFacts(
        start=start_word,
        pages=tuple(dict.fromkeys(pages)),
        protocol=ONE_SECOND if one_second else THREE_SECOND,
        updates=len(updates) or sum(1 for line in lines if "inference " in line and "ms — " in line),
        capped_overlaps=sum(len(clusters(m[4])) == OVERLAP_CAP for m in updates) if one_second else 0,
        median_inference_ms=statistics.median(times) if times else None,
    )


_CONSOLE_PREFIX = re.compile(r"^(?:\[[^\]]*\] ?)+")
_LEGACY_WINDOW = re.compile(r"^Window inference (\d+)ms \u2014 confirmed: (.*?), unconfirmed: (.*)$")
_LEGACY_PREVIEW = re.compile(r"^Preview inference (\d+)ms \u2014 (.*)$")


def _update_check(line: str) -> dict | None:
    """``Update: type=…, hop=…, overlap=…, didFlush=…`` (or ``confirmed=`` / ``unconfirmed=``) as
    ``parseUpdateLine`` reads it: each field up to the next field's first separator."""
    hop_key = "hop=" if "hop=" in line else "confirmed="
    overlap_key = ", overlap=" if "overlap=" in line else ", unconfirmed="
    if "type=" not in line or hop_key not in line or overlap_key not in line:
        return None
    after_hop = line[line.index(hop_key) + len(hop_key):]
    after_overlap = line[line.index(overlap_key) + len(overlap_key):]
    if overlap_key not in after_hop or ", didFlush=" not in after_overlap:
        return None
    flush = line[line.index("didFlush=") + len("didFlush="):].strip() == "true"
    return {"hop": after_hop[:after_hop.index(overlap_key)],
            "overlap": after_overlap[:after_overlap.index(", didFlush=")], "flush": flush}


def session_checks(text: str) -> list[dict]:
    """The checks a session's log hands the engine, in order, read as the pinned Muraja replay
    parses a log (``tools/replay_log.swift``, ``parseLogEvents``): each ``Update`` line and the
    ``Stop`` line's final phonemes (as the session's stop); in a log with no structured line, each
    legacy ``Window inference`` (confirmed, unconfirmed) and non-empty ``Preview inference``.
    Withheld hops are never aligned and are left out. The console prefix (``[time] [level]
    [category]``) is stripped first."""
    structured = "Update: type=" in text or "Withheld update: " in text
    checks: list[dict] = []
    for raw in text.splitlines():
        line = _CONSOLE_PREFIX.sub("", raw.strip()).strip()
        if structured:
            if line.startswith("Update: type=") and (check := _update_check(line)) is not None:
                checks.append(check)
            elif line.startswith("Stop: ") and "finalPhonemes=" in line:
                checks.append(make_check(STOP, line[line.index("finalPhonemes=") + len("finalPhonemes="):], ""))
        elif m := _LEGACY_WINDOW.match(line):
            checks.append({"hop": m[2], "overlap": m[3].strip(), "flush": False})
        elif (m := _LEGACY_PREVIEW.match(line)) and m[2].strip():
            checks.append({"hop": "", "overlap": m[2].strip(), "flush": False})
    return checks


def lengthened_previews(text: str) -> tuple[str, int]:
    """The log with each preview update's overlap replaced by the longer text the transcriber's
    own line logged just before it (40 Characters, ``RealtimeTranscriber+iOS.swift:642``, where
    the update line keeps 30), and how many previews were lengthened."""
    out, lengthened, transcribed = [], 0, None
    for line in text.splitlines(keepends=True):
        if (m := _TRANSCRIBED_PREVIEW.search(line.rstrip("\n"))) is not None:
            transcribed = m[1]
        elif (m := _UPDATE.search(line)) is not None:
            longer = transcribed or ""
            if m[1] == "preview" and m[2] == "hop" and len(longer) > len(m[4]) and longer.startswith(m[4]):
                line = line[: m.start(4)] + longer + line[m.end(4):]
                lengthened += 1
            transcribed = None
        out.append(line)
    return "".join(out), lengthened


# --- replay ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Replay:
    """One log replayed under one engine configuration: the final position, every kept grade,
    and every grade decision the word scorer logged (``Grade end word …``), as a device log
    keeps them."""

    final_position: str
    words: Mapping[str, dict]
    grade_lines: tuple[str, ...] = field(default=(), compare=False)


def log_dir(commit: str, root: Path = HARNESS_ROOT) -> Path:
    """Muraja's ``tools/test_logs`` in the checkout ``build.sh`` made for ``commit``."""
    return root / commit / "muraja" / "tools" / "test_logs"


def replay_logs(
    harness: Harness, paths: Mapping[str, Path], scorings: Sequence[Scoring]
) -> tuple[dict, dict[str, dict[str, Replay]]]:
    """The build record, and ``[log][engine name]`` the replay of each log in ``paths``: one
    harness run per scoring mode, since the replay reads its mode from ``REPLAY_MODE``."""
    build, replays = {}, {}
    for mode in sorted({s.mode for s in scorings}):
        requests = [{"item": log, "log": str(path), "scorings": engine_runs([s for s in scorings if s.mode == mode])}
                    for log, path in sorted(paths.items())]
        build, results = harness.run(requests, env={"REPLAY_MODE": mode})
        for result in results:
            replays.setdefault(result["item"], {})[result["scoring"]] = Replay(
                result["final_position"], {w["position"]: w["status"] for w in result["words"]},
                tuple(result["grade_lines"]))
    return build, replays


def baseline_differences(replay: Replay, baseline: dict) -> list[str]:
    """How a shipped-default replay differs from a log's committed ``baseline.json``."""
    qualities = {word: status["quality"] for word, status in replay.words.items()}
    expected = baseline["word_qualities"]
    differences = [f"final position {replay.final_position}, baseline {baseline['final_position']}"
                   ] if replay.final_position != baseline["final_position"] else []
    return differences + [f"{word}: {qualities.get(word)}, baseline {expected.get(word)}"
                          for word in sorted(set(qualities) | set(expected), key=position)
                          if qualities.get(word) != expected.get(word)]


# --- attribution -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WordGrade:
    """A graded word of a log under one scoring: shown as correct or not, and why not."""

    log: str
    word: str
    scoring: str
    quality: str
    causes: tuple[str, ...]
    marks: tuple[str, ...]
    pairs: tuple[str, ...]
    explained: Explained | None
    #: The consonant errors behind its consonant causes, ``(cause, substitution)``.
    consonants: tuple[tuple[str, str], ...] = ()
    #: The kept grade's errors as the engine reports them (with their group indices).
    errors: tuple[dict, ...] = ()

    @property
    def flagged(self) -> bool:
        return self.quality != CORRECT


def word_grades(
    entry: LogEntry, replays: Mapping[str, Replay], scorings: Sequence[Scoring], text: MurajaText
) -> list[WordGrade]:
    """Every graded word of a log under every scoring, attributed when shown as not correct."""
    out = []
    for scoring in scorings:
        for word, status in sorted(replays[scoring.engine_name].words.items(), key=lambda w: position(w[0])):
            shown = displayed(status["quality"], scoring)
            surah, ayah, index = position(word)
            why = None if shown == CORRECT else flag(
                SiteGrade("flagged", shown, tuple(status["errors"])),
                text.phonemes(surah, ayah).split(" ")[index - 1])
            out.append(WordGrade(
                entry.log, word, scoring.name, shown,
                why.causes if why else (), why.marks if why else (), why.pairs if why else (),
                entry.explanation(word), why.consonants if why else (), tuple(status["errors"])))
    return out


# --- rates -------------------------------------------------------------------------------------------
def per_1000(share: float | str | None) -> float | None:
    """A share as a rate per 1,000; an infinite bootstrap bound (``"inf"``) has none."""
    return None if share is None or isinstance(share, str) else 1000 * share


def rate(hits: Sequence[bool], logs: Sequence[str]) -> dict:
    """Flags per 1,000 words and their log-clustered interval (``ratio_interval``)."""
    cluster_of = {log: k for k, log in enumerate(sorted(set(logs)))}
    hit = np.array(hits, dtype=float)
    ones = np.ones_like(hit)
    interval = ratio_interval([cluster_of[log] for log in logs], hit, ones, ones).as_dict()
    return {
        "flags": int(hit.sum()), "words": len(hits), "logs": len(cluster_of),
        "per_1000": per_1000(hit.sum() / len(hits)) if len(hits) else None,
        "lower": per_1000(interval["lower"]), "upper": per_1000(interval["upper"]),
        "method": interval["method"],
    }


def view_of(grades: Sequence[WordGrade]) -> dict:
    """One scoring over a set of graded words: the false-flag rate, each cause's, and the
    tashkeel marks and target pairs behind them (counts)."""
    logs = [g.log for g in grades]
    flagged = [g for g in grades if g.flagged]
    return {
        "false_flags": rate([g.flagged for g in grades], logs),
        "causes": {c: rate([g.flagged and c in g.causes for g in grades], logs) for c in CAUSES},
        "marks": dict(sorted(Counter(m for g in flagged for m in set(g.marks)).items())),
        "pairs": dict(sorted(Counter(p for g in flagged for p in g.pairs).items())),
    }


def group_of(entry: LogEntry, facts: LogFacts) -> tuple[str, ...]:
    if entry.model == H448:
        return (H448,)
    return (TEACHER, TEACHER_ONE_SECOND) if facts.protocol == ONE_SECOND else (TEACHER,)


def views(grades: Sequence[WordGrade], scorings: Sequence[Scoring]) -> dict:
    """``[scoring][view]`` :func:`view_of` the words, every one and the unexplained ones."""
    return {s.name: {ALL_WORDS: view_of([g for g in grades if g.scoring == s.name]),
                     UNEXPLAINED: view_of([g for g in grades if g.scoring == s.name and not g.explained])}
            for s in scorings}


def device_sessions(entries: Sequence[LogEntry], model: str) -> list[LogEntry]:
    """A model's device sessions, those excluded for a documented slip included."""
    return [e for e in entries if e.model == model and (e.included or e.exclusion == DOCUMENTED_MISTAKE)]


def slip_sensitivity(
    entries: Sequence[LogEntry], grades: Mapping[str, Sequence[WordGrade]], scorings: Sequence[Scoring]
) -> dict:
    """``[model][width][scoring]`` :func:`view_of` every device session of the model, an
    explained word or a word in the session's slip region (of that width) left out."""
    return {model: {width: {s.name: view_of([g for e in device_sessions(entries, model) for g in grades[e.log]
                                             if g.scoring == s.name and not g.explained
                                             and not e.in_slip(g.word, width)])
                            for s in scorings}
                    for width in SLIP_WIDTHS}
            for model in MODELS}


def allowance_effect(
    sessions: Sequence[LogEntry], grades: Mapping[str, Sequence[WordGrade]], row: tuple[str, str | None],
    width: str | None,
) -> dict:
    """One allowance over some sessions (explained words, and slip-region words of ``width``,
    left out): the words flagged with it alone off that ``allowances_on`` does not flag, as a
    rate over the words graded with it off, the words ``allowances_on`` flags that it does not
    (``removed``), and each added word with the sessions it recurs in. A target pair no allowance
    forgives gets the words ``allowances_on`` flags for it instead (``present``)."""
    label, off = row
    hits, logs, removed = [], [], 0
    carriers: dict[str, set[str]] = {}
    for e in sessions:
        kept = [g for g in grades[e.log] if not g.explained and (width is None or not e.in_slip(g.word, width))]
        on = {g.word for g in kept if g.scoring == ALLOWANCES_ON.name and g.flagged}
        if off is None:
            pair = label.rsplit(" ", 1)[1]
            graded = [(g, g.flagged and pair in g.pairs) for g in kept if g.scoring == ALLOWANCES_ON.name]
        else:
            graded = [(g, g.flagged and g.word not in on) for g in kept if g.scoring == off]
            removed += len(on - {g.word for g, _ in graded if g.flagged})
        for g, hit in graded:
            hits.append(hit)
            logs.append(e.log)
            if hit:
                carriers.setdefault(g.word, set()).add(e.log)
    return {"allowance": off is not None, **rate(hits, logs), "removed": removed,
            "carriers": sorted(([w, len(s)] for w, s in carriers.items()), key=lambda ws: position(ws[0]))}


def allowance_table(
    entries: Sequence[LogEntry], facts: Mapping[str, LogFacts], grades: Mapping[str, Sequence[WordGrade]]
) -> dict:
    """``[allowance][group]`` :func:`allowance_effect`, for :data:`ALLOWANCE_ROWS` and
    :data:`ALLOWANCE_GROUPS`."""
    included = [e for e in entries if e.included]
    populations = {group: ([e for e in included if group in group_of(e, facts[e.log])], None)
                   for group in (TEACHER_ONE_SECOND, TEACHER, H448)}
    populations |= {sensitivity(m, w): (device_sessions(entries, m), w) for m in MODELS for w in SLIP_WIDTHS}
    out: dict = {}
    for row in ALLOWANCE_ROWS:
        out[row[0]] = {}
        for group in ALLOWANCE_GROUPS:
            sessions, width = populations[group]
            out[row[0]][group] = allowance_effect(sessions, grades, row, width)
    return out


def log_counts(grades: Sequence[WordGrade], scorings: Sequence[Scoring]) -> dict:
    """Per scoring: one log's graded words and flags, every one and the unexplained ones."""
    out = {}
    for s in scorings:
        mine = [g for g in grades if g.scoring == s.name]
        out[s.name] = {"words": len(mine), "flags": sum(g.flagged for g in mine),
                       "unexplained_words": sum(not g.explained for g in mine),
                       "unexplained_flags": sum(g.flagged and not g.explained for g in mine)}
    return out


def covered_ayat(words: Iterable[str]) -> str:
    """The ayat a session graded, as runs: ``2:17-2:25, 3:1``."""
    ayat = sorted({position(w)[:2] for w in words})
    runs: list[list[tuple[int, int]]] = []
    for ayah in ayat:
        if runs and runs[-1][-1][0] == ayah[0] and runs[-1][-1][1] + 1 == ayah[1]:
            runs[-1].append(ayah)
        else:
            runs.append([ayah])
    return ", ".join(f"{r[0][0]}:{r[0][1]}" + (f"-{r[-1][0]}:{r[-1][1]}" if len(r) > 1 else "") for r in runs)


def word_row(g: WordGrade) -> dict:
    return {"log": g.log, "word": g.word, "scoring": g.scoring, "quality": g.quality,
            "causes": list(g.causes), "marks": list(g.marks), "pairs": list(g.pairs),
            "explained": g.explained.kind if g.explained else None,
            "basis": g.explained.basis if g.explained else None}


# --- the simulation (#90) ------------------------------------------------------------------------------
def simulation_rates(report: Mapping) -> dict:
    """#90's word sample, per model and scoring: false flags and causes over the correctly
    recited words (count, and the weighted rate per 1,000 with its interval)."""
    def scaled(cell: Mapping) -> dict:
        return {"flags": cell["words"], "per_1000": per_1000(cell["point"]),
                "lower": per_1000(cell["lower"]), "upper": per_1000(cell["upper"])}

    out = {}
    for model, arm in SIMULATION_ARMS.items():
        out[model] = {}
        for scoring, cell in report["word_sample"]["arms"][arm].items():
            out[model][scoring] = {"words": cell["correct_words"], **scaled(cell["false_flags"]),
                                   "causes": {c: scaled(v) for c, v in cell["causes"].items()}}
    return out


def simulated_truncation(harness: Harness) -> dict:
    """#90's word sample regraded with every check's overlap cut to :data:`OVERLAP_CAP`
    Characters: per model and scoring, the false flags before and after and the words whose
    shown grade changed. Torch-free: it reads the committed check sequences."""
    tb = truth_baseline
    words = verified_words(load_words(), read_verdicts())
    registry = load_staged_clips(REGISTRY_PATH)
    items = tb.items_of([w.site for w in words], registry)
    excluded = current()
    correct = [w.site.site_id for w in words if w.side == CORRECT_SIDE and w.site.site_id not in excluded]

    def kept(cap: int | None):
        sequences = {}
        for name, ref in tb.DEFAULT_MODELS.items():
            _, recorded = tb.model_caches(name, ref, items, registry, None)
            sequences[tb.arm(name, tb.CHECKS)] = {
                key: [{**c, "overlap": "".join(clusters(c["overlap"])[:cap])} if cap else c
                      for c in value[tb.CHECKS]]
                for key, value in recorded["items"].items() if key in items}
        decodes = {arm_: {key: "" for key in seq} for arm_, seq in sequences.items()}
        return tb.app_outcomes([], words, decodes, sequences, harness)[1].grades

    full, cut = kept(None), kept(OVERLAP_CAP)
    out: dict = {}
    for model, arm in SIMULATION_ARMS.items():
        out[model] = {}
        for scoring in full:
            before, after = full[scoring][arm], cut[scoring][arm]
            out[model][scoring] = {
                "words": len(correct),
                "flags_full": sum(before[w].outcome in WORD_FLAGGED for w in correct),
                "flags_cut": sum(after[w].outcome in WORD_FLAGGED for w in correct),
                "changed": sum(before[w] != after[w] for w in correct),
            }
    return out


def device_truncation(
    harness: Harness, entries: Sequence[LogEntry], texts: Mapping[str, str],
    replays: Mapping[str, Mapping[str, Replay]], scorings: Sequence[Scoring],
) -> dict:
    """The included 1 s-hop logs replayed with each preview lengthened (:func:`lengthened_previews`):
    previews lengthened, and per scoring the flags before and after and the words whose shown
    grade changed (a word graded in only one replay counts as changed)."""
    with tempfile.TemporaryDirectory() as tmp:
        paths, lengthened = {}, {}
        for entry in entries:
            text, n = lengthened_previews(texts[entry.log])
            if n:
                path = Path(tmp) / f"{entry.log}.log"
                path.write_text(text, encoding="utf-8")
                paths[entry.log], lengthened[entry.log] = path, n
        _, longer = replay_logs(harness, paths, scorings)
    out = {"logs": len(paths), "previews_lengthened": sum(lengthened.values()), "scorings": {}}
    for s in scorings:
        flags_before = flags_after = changed = 0
        for log in paths:
            before = {w: displayed(st["quality"], s) for w, st in replays[log][s.engine_name].words.items()}
            after = {w: displayed(st["quality"], s) for w, st in longer[log][s.engine_name].words.items()}
            flags_before += sum(q != CORRECT for q in before.values())
            flags_after += sum(q != CORRECT for q in after.values())
            changed += sum(before.get(w) != after.get(w) for w in set(before) | set(after))
        out["scorings"][s.name] = {"flags_logged": flags_before, "flags_lengthened": flags_after,
                                   "changed": changed}
    return out


# --- the report --------------------------------------------------------------------------------------
def require(commit: str) -> Harness:
    harness = Harness.locate(commit=commit, binary=LOG_REPLAY_BINARY)
    if harness is None:
        raise SystemExit(f"no Muraja log replay for {commit} under {HARNESS_ROOT}; "
                         f"build it with `bash muraja_harness/build.sh {commit}`")
    return harness


def build_report(scorings: Sequence[Scoring] = SCORINGS) -> dict:
    replayed = (*scorings, *PAIR_SCORINGS)
    pin, main = require(MURAJA_COMMIT), require(MURAJA_MAIN)
    text = MurajaText(pin.quran_db)
    entries = [with_regions(e, lambda surah, ayah: len(text.phonemes(surah, ayah).split(" ")))
               for e in load_inventory()]
    pin_dir, main_dir = log_dir(MURAJA_COMMIT), log_dir(MURAJA_MAIN)
    on_disk = sorted(p.stem for p in pin_dir.glob("*.log") if not p.stem.startswith("generated_"))
    if on_disk != [e.log for e in entries]:
        raise SystemExit(f"the inventory does not list the pin's non-generated logs: {on_disk}")
    texts = {e.log: (pin_dir / f"{e.log}.log").read_text(encoding="utf-8") for e in entries}
    facts = {log: read_facts(text) for log, text in texts.items()}
    for e in entries:
        ms = facts[e.log].median_inference_ms
        if e.model and ms is not None and (ms < STUDENT_INFERENCE_MS) != (e.model == H448):
            raise SystemExit(f"{e.log}: a median inference of {ms} ms contradicts the model {e.model}")

    builds, replays, parity = {}, {}, {}
    for name, harness, directory in (("pin", pin, pin_dir), ("main", main, main_dir)):
        paths = {e.log: directory / f"{e.log}.log" for e in entries}
        builds[name], replays[name] = replay_logs(harness, paths, replayed)
        parity[name] = {e.log: baseline_differences(
            replays[name][e.log][SHIPPED_DEFAULT.engine_name],
            json.loads((directory / f"{e.log}.baseline.json").read_text(encoding="utf-8")))
            for e in entries}
    broken = {name: [log for log, diff in by_log.items() if diff] for name, by_log in parity.items()}
    if any(broken.values()):
        raise SystemExit(f"replay does not reproduce the committed baselines: {broken}")
    main_differs = sorted(log for log in replays["pin"] if replays["pin"][log] != replays["main"][log])

    reported = [e for e in entries if e.included or e.exclusion == DOCUMENTED_MISTAKE]
    grades = {e.log: word_grades(e, replays["pin"][e.log], replayed, text) for e in reported}
    included = [e for e in entries if e.included]
    groups = {group: views([g for e in included if group in group_of(e, facts[e.log]) for g in grades[e.log]],
                           scorings) for group in GROUPS}
    mistakes = {model: views([g for e in reported if not e.included and e.model == model
                              for g in grades[e.log]], scorings) for model in MODELS}
    return {
        "muraja": {"pin": MURAJA_COMMIT, "main": MURAJA_MAIN, "builds": builds,
                   "main_differs_from_pin": main_differs},
        "scorings": [s.as_dict() for s in scorings],
        "logs": [{
            "log": e.log, "commit": e.commit, "committed": e.committed, "model": e.model,
            "model_basis": e.model_basis, "app_version": e.app_version, "purpose": e.purpose,
            "sources": list(e.sources), "exclusion": e.exclusion, "exclusion_detail": e.exclusion_detail,
            "explained": [vars(x) for x in e.explained],
            "slips": [vars(x) for x in e.slips],
            "slip_regions": {w: [vars(x) for x in r] for w, r in e.slip_regions.items()},
            "start": facts[e.log].start, "pages": list(facts[e.log].pages),
            "protocol": facts[e.log].protocol, "updates": facts[e.log].updates,
            "capped_overlaps": facts[e.log].capped_overlaps,
            "median_inference_ms": facts[e.log].median_inference_ms,
            "ayat": covered_ayat(replays["pin"][e.log][SHIPPED_DEFAULT.engine_name].words),
            "parity": {name: not parity[name][e.log] for name in parity},
            "counts": log_counts(grades[e.log], scorings) if e.log in grades else None,
        } for e in entries],
        "groups": groups,
        "documented_mistakes": mistakes,
        "slips_excluded": slip_sensitivity(entries, grades, scorings),
        "allowances": allowance_table(entries, facts, grades),
        "words": [word_row(g) for e in reported for g in grades[e.log]
                  if g.flagged and g.scoring in {s.name for s in scorings}],
        "simulation": simulation_rates(
            json.loads(truth_baseline.REPORT_PATH.read_text(encoding="utf-8"))),
        "truncation": {
            "device": device_truncation(pin, [e for e in included if facts[e.log].protocol == ONE_SECOND],
                                        texts, replays["pin"], scorings),
            "simulation": simulated_truncation(require_harness()),
        },
    }


def main() -> None:
    from training.device_logs_doc import render

    report = build_report()
    write_text_atomically(REPORT_PATH, json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    write_text_atomically(DOC_PATH, render(report))
    print(f"Wrote {REPORT_PATH} and {DOC_PATH}")


if __name__ == "__main__":
    main()
