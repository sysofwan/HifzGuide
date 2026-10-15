"""The owner's ``h448`` sessions, from the app's own bug-report exports, replayed (#153).

#151 (:mod:`training.device_logs`) found no included device session on ``h448``. The owner then
recited pages 4, 9 and 42 on the app as shipped (1.0.27, ``MuaalemStudent_h448_6BIT``) and
exported each as a bug report, to measure false flags. The recitation is taken as correct except
where the owner says otherwise (``owner_sessions.json``): the owner's mistakes are the catch side,
kept out of every false-flag rate. A word the owner names (2:60:11) and the added phrases are
confirmed mistakes; an ayah the owner says holds mistakes without naming its words (2:256) is
left out unlocalised, its flags reported with no catch-rate denominator. Every word the
phone showed as not correct carries the owner's verdict (a mistake, or a model error with what
explains it, read from the decode). A word flagged only under another scoring was never shown on
the phone and stays a false flag on the rule that the recitation is correct.

**The exports are the owner's private data.** They are read from the paths given on the command
line, each checked against its pinned name and SHA-256. Only aggregates, positions, counts, the
owner's verdicts and the per-word table are written; no log text, device info or settings dump
(``test_device_exports`` enforces it).

**Sessions.** An export holds the tail of the app's log: whole or truncated app runs (``===
Previous Session ===``, ``=== Current Session ===``), each with the model it loaded, and in them
the reader's sessions, each opened by ScoreCollector's ``… session pending on page N``
(:func:`split_sessions`). The app no longer logs a ``Start:`` line (Muraja ``ba91ac3``), so a
session is replayed from that line on and Muraja's replay seats the engine on the ayah of its
first ``expected=`` (``replay_log.swift:extractStartPosition``). A session whose opening was
truncated away is not replayed; it is linked to the whole session in another export it is the
tail of (:func:`link`), which also gives a truncated app run its version and model.

**Parity and the mode.** No log line records the scoring mode. The word scorer logs every
end-word grade it decides (``Grade end word …``, ``QuranFollowAlong+WordScoring.swift:750``), and
the harness keeps the same lines from the replay (``muraja_harness/log_replay.swift``). Each
session is replayed in every mode with the shipped flags, at the pin and at Muraja ``main``, as
logged and with each preview lengthened to the transcriber's own line
(:func:`device_logs.lengthened_previews`); its mode is the one whose replay reproduces the
device's lines (:func:`parity`). Every mismatch is reported, and the kept grades are checked
against ScoreCollector's committed ones (:func:`commit_parity`).

**Scoring.** As #151: every scoring of :data:`training.muraja_policy.SCORINGS` (Balanced), each
soft pair alone (:data:`device_logs.PAIR_SCORINGS`), and the session as recorded (its own mode,
the shipped flags: :data:`AS_RECORDED`). Attribution, rates and the allowance table are #151's
(:func:`device_logs.word_grades`, :func:`device_logs.view_of`, :func:`device_logs.allowance_effect`).
The consonant flags are tallied by substitution (:func:`word_flags.substitution_tally`) here, in
#151's device logs (replayed at the pin, each population's flag count checked against #151's
report) and in #90's word sample.

Usage (from ``tools/``, on a Mac, with the harness built at the pin and at Muraja ``main``)::

    python -m training.device_exports ~/Downloads/muraja-bug-report\\ {69,70,71}.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path

from tadabur.listening_session import read_verdicts
from tadabur.word_sample import load_words, verified_words
from training import device_logs as dl
from training import truth_baseline
from training.device_logs import (
    ALLOWANCE_ROWS,
    H448,
    PAIR_SCORINGS,
    SLIP_WIDTHS,
    UNEXPLAINED,
    Explained,
    LogEntry,
    Replay,
    Slip,
    WordGrade,
    position,
)
from training.muraja_policy import (
    CORRECT,
    MURAJA_COMMIT,
    SCORINGS,
    SHIPPED_DEFAULT,
    TASHKEEL_ERROR,
    Harness,
    MurajaText,
    Scoring,
)
from training.tashkeel_eval import write_text_atomically
from training.word_flags import CAUSES, TASHKEEL, consonant_errors, error_of, substitution, substitution_tally

SESSIONS_PATH = dl.DEVICE_LOGS_DIR / "owner_sessions.json"
REPORT_PATH = dl.DEVICE_LOGS_DIR / "owner_report.json"
DOC_PATH = dl.DOC_PATH.parent / "owner-device-sessions.md"

#: The modes a session may have run in, as Muraja's replay names them (``REPLAY_MODE``).
MODES = ("strict", "balanced", "lenient")
#: The session as the device ran it: its own mode, the shipped flags, tashkeel shown.
AS_RECORDED = "as_recorded"
#: The basis of what the owner said about these sessions (2026-10-14 and 2026-10-15).
OWNER = "owner"
#: The owner's mistakes, two kinds of explained words: a word the owner names as a mistake, and
#: an ayah the owner says holds mistakes without naming its words. Both are out of every
#: false-flag rate; only the first is catch-side evidence word by word.
MISTAKE = "mistake"
MISTAKE_AYAH = "mistake_ayah"
#: A non-correct word's status: a mistake the owner made, a word of an ayah the owner says holds
#: mistakes (words not identified), a model error the owner confirmed, or (never shown on the
#: phone, so given no verdict) a false flag by the rule that the recitation is correct.
IN_MISTAKE_AYAH = "in_mistake_ayah"
MODEL_ERROR = "model_error"
ASSUMED = "assumed"
STATUSES = (MISTAKE, IN_MISTAKE_AYAH, MODEL_ERROR, ASSUMED)
#: The model-load line of an app run, and the opening of a reader's session.
_LOADED = re.compile(r"\[MuaalemInference\] \[([^\]]+)\] ✅ Loaded")
_PENDING = re.compile(r"\[ScoreCollector\] (Tilawah|Hifz) session pending on page (\d+)")
_APP_RUN = re.compile(r"^=== (Previous|Current) Session ===$")
_APP_VERSION = re.compile(r"^App Version: (\S+)")
_TRUNCATED = "[… truncated …]"
#: How many of a truncated session's opening lines must appear in a whole session (each line
#: carries its millisecond timestamp) for it to be that session's tail.
_OVERLAP_LINES = 20
_GRADE_LINE = re.compile(r"\] (Grade end word (\d+:\d+:\d+): .* → (\w+))$")
_SHADOW = re.compile(r"Shadow speech: #(\d+) (\w+)(?:\([^)]*\))? cycle=(\d+)(?: (\d+:\d+:\d+)→(\d+:\d+:\d+))?")
_ADDED_SAVED = re.compile(r"\[ScoreCollector\] Added words saved (\d+:\d+:\d+)→(\d+:\d+:\d+)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- an export, split ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Session:
    """One reader's session in an export: from its ``session pending`` line (or, when the export
    truncated its opening, from the export's first line) to the next session or app run."""

    export: str
    #: ``<export number>#<n>``: the n-th session of the export, from 1.
    name: str
    #: Its lines, 1-based, inclusive, in the export.
    first_line: int
    last_line: int
    #: ``None`` for a session whose opening was truncated away.
    page: int | None
    kind: str | None
    #: The app run's: ``None`` where the export truncated the run's header or load line.
    app_version: str | None
    model: str | None
    #: The session's text, never written out.
    text: str
    #: The app run it is in: the export's n-th, from 0 (0 is the tail the export truncated).
    run: int = 0
    #: For a truncated session, the whole session in another export it is the tail of.
    copy_of: str | None = None
    #: Where :attr:`app_version` and :attr:`model` were read: the session's own app run, or the
    #: app run of the session it shares a run with in another export.
    run_basis: str | None = None

    @property
    def whole(self) -> bool:
        return self.page is not None


def split_sessions(export: str, text: str) -> list[Session]:
    """The reader's sessions of an export, in order (module docstring)."""
    lines = text.splitlines()
    number = re.search(r"(\d+)", export)
    tag = number[1] if number else export
    body = lines.index(_TRUNCATED) + 1 if _TRUNCATED in lines else 0
    # each app run: [first line, its version, its model]; the first may be a truncated tail
    runs: list[list] = [[body, None, None]]
    for i in range(body, len(lines)):
        if _APP_RUN.match(lines[i]):
            runs.append([i, None, None])
        elif (m := _APP_VERSION.match(lines[i])) and runs[-1][1] is None and i - runs[-1][0] < 6:
            runs[-1][1] = m[1]
        elif (m := _LOADED.search(lines[i])) and runs[-1][2] is None:
            runs[-1][2] = m[1]
    sessions = []
    for k, (start, version, model) in enumerate(runs):
        end = runs[k + 1][0] if k + 1 < len(runs) else len(lines)
        opens = [i for i in range(start, end) if _PENDING.search(lines[i])]
        # updates before a run's first opening belong to a session whose opening was truncated
        lead = lines[start:opens[0] if opens else end]
        heads = [(start, None)] if any("Update: " in line for line in lead) else []
        heads += [(i, _PENDING.search(lines[i])) for i in opens]
        for j, (first, opening) in enumerate(heads):
            last = heads[j + 1][0] if j + 1 < len(heads) else end
            sessions.append(Session(
                export=export, name=f"{tag}#{len(sessions) + 1}", first_line=first + 1, last_line=last,
                page=int(opening[2]) if opening else None, kind=opening[1] if opening else None,
                app_version=version, model=model, text="\n".join(lines[first:last]) + "\n", run=k,
                run_basis="its app run's header and model-load line" if model else None))
    return sessions


def link(sessions: Sequence[Session]) -> list[Session]:
    """The sessions with each truncated one tied to the whole session it is the tail of, and
    each session of a truncated app run given that run's version and model, read where the run
    is whole (through as many exports as it takes)."""
    by_name = {s.name: s for s in sessions}
    copies: dict[str, str] = {}
    for t in sessions:
        if not t.whole:
            # its opening lines (the export may cut the first mid-way) in a whole session's text
            head = "\n".join(t.text.split("\n")[1:1 + _OVERLAP_LINES])
            original = next((o for o in sessions if o.whole and o.export != t.export and head in o.text), None)
            if original is not None:
                copies[t.name] = original.name

    def run_of(s: Session, seen: frozenset[str]) -> tuple[str | None, str | None, str | None]:
        if s.model is not None:
            return s.app_version, s.model, s.run_basis
        for t in sessions:
            if t.export == s.export and t.run == s.run and copies.get(t.name) not in (None, *seen):
                original = by_name[copies[t.name]]
                version, model, _ = run_of(original, seen | {original.name})
                if model is not None:
                    return version, model, f"the app run of {original.name} ({t.name} is its tail)"
        return None, None, None

    return [replace(s, copy_of=copies.get(s.name), **dict(zip(("app_version", "model", "run_basis"),
                                                              run_of(s, frozenset({s.name})))))
            for s in sessions]


# --- what the device logged, as positions and counts ----------------------------------------------
def grade_lines(text: str) -> list[str]:
    """The word scorer's grade decisions a log holds, in order (``Grade end word …``)."""
    return [m[1] for line in text.splitlines() if (m := _GRADE_LINE.search(line))]


def _word_of(line: str) -> str:
    return line.split(" ", 4)[3].rstrip(":")


def _quality_of(line: str) -> str:
    return line.rsplit("→ ", 1)[1]


def parity(device: Sequence[str], replay: Sequence[str]) -> dict:
    """The device's grade decisions against a replay's: how many of each, how many the longest
    common subsequence matches, and every word a decision of either side differs on."""
    matcher = SequenceMatcher(None, list(device), list(replay), autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    differing = {_word_of(line) for op, a1, a2, b1, b2 in matcher.get_opcodes() if op != "equal"
                 for line in (*device[a1:a2], *replay[b1:b2])}
    return {"device": len(device), "replay": len(replay), "matched": matched,
            "exact": matched == len(device) == len(replay),
            "words": len({_word_of(line) for line in device}),
            "differing_words": sorted(differing, key=position)}


_COMMIT = re.compile(r"\[ScoreCollector\] Commit (\d+:\d+:\d+) quality=(\w+)")
_UPGRADED = re.compile(r"\[FollowAlong\] Upgraded: (\d+:\d+:\d+) ")


def commit_parity(text: str, replay: Replay) -> dict:
    """Each word's grade ScoreCollector committed on the device (its first ``Commit`` line)
    against the grade the replay kept. A word the device regraded after its commit (an
    ``Upgraded:`` line for it later in the log) may differ; any other difference is unexplained."""
    commits: dict[str, str] = {}
    regraded: set[str] = set()
    for line in text.splitlines():
        if (m := _COMMIT.search(line)) and m[1] not in commits:
            commits[m[1]] = m[2]
        elif (m := _UPGRADED.search(line)) and m[1] in commits:
            regraded.add(m[1])
    kept = {word: status["quality"] for word, status in replay.words.items()}
    differ = sorted((w for w in set(commits) | set(kept) if commits.get(w) != kept.get(w)), key=position)
    return {"committed": len(commits), "kept": len(kept),
            "equal": sum(commits.get(w) == kept.get(w) for w in commits),
            "regraded_after_commit": [{"word": w, "committed": commits.get(w), "kept": kept.get(w)}
                                      for w in differ if w in regraded],
            "unexplained": [{"word": w, "committed": commits.get(w), "kept": kept.get(w)}
                            for w in differ if w not in regraded]}


_COMMITTED_ERRORS = re.compile(r"\[ScoreCollector\] Commit (\d+:\d+:\d+) quality=(\w+) .*errors=\d+ \[(.*)\]$")


def committed_flags(text: str) -> list[dict]:
    """The words ScoreCollector committed as not correct on the device (each word's first
    commit), with their errors as substitutions (``ل→ء``, ``ه→∅``) or ``tashkeel``: the grades
    the owner saw and gave verdicts on."""
    out, seen = [], set()
    for line in text.splitlines():
        if (m := _COMMITTED_ERRORS.search(line)) and m[1] not in seen:
            seen.add(m[1])
            if m[2] != CORRECT:
                errors = [error_of(d) for d in m[3].split(",") if d]
                out.append({"word": m[1], "quality": m[2],
                            "errors": [substitution(e) if e else TASHKEEL for e in errors]})
    return sorted(out, key=lambda r: position(r["word"]))


@dataclass(frozen=True)
class Added:
    """An added-phrase classification of Muraja's shadow-speech detector in a session: its id,
    every classification the detection logged (in order), the words it was placed between,
    and whether ScoreCollector saved added words there (it does so when the session ends)."""

    detection: int
    kinds: tuple[str, ...]
    start: str
    end: str
    saved: bool


def added_phrases(text: str) -> list[Added]:
    lines = text.splitlines()
    saved = {(m[1], m[2]) for line in lines if (m := _ADDED_SAVED.search(line))}
    detections = [m for line in lines if (m := _SHADOW.search(line))]
    kinds: dict[int, list[str]] = {}
    for m in detections:
        if kinds.setdefault(int(m[1]), [])[-1:] != [m[2]]:
            kinds[int(m[1])].append(m[2])
    added = {(int(m[1]), m[4], m[5]) for m in detections if m[2] == "added"}
    return [Added(d, tuple(kinds[d]), start, end, (start, end) in saved) for d, start, end in sorted(added)]


def decision_counts(text: str, word: str) -> dict[str, int]:
    """How many of the device's grade decisions for a word fell on each quality."""
    return dict(sorted(Counter(_quality_of(line) for line in grade_lines(text) if _word_of(line) == word).items()))


# --- the owner's annotations ------------------------------------------------------------------------
#: How a model error the owner confirmed is explained: by the decode (the model heard another
#: letter or mark there), or by a window boundary inside the word (the engine graded it on a
#: decode that lost part of it).
EXPLAINED_BY = ("decode", "boundary")


@dataclass(frozen=True)
class Annotations:
    """``owner_sessions.json``, written by hand: the exports the report is built from (name and
    SHA-256); the owner's statements; per session the words they make real mistakes, the words
    they confirm as model errors (each with what explains it, read from the decode), the added
    phrases (with the passage whose wording they are, read from the decode) and any words the
    decode shows were never recited (#151's rule: decode evidence only)."""

    exports: Mapping[str, str]
    statements: Mapping[str, dict]
    mistakes: Mapping[str, tuple[Explained, ...]]
    model_errors: Mapping[str, Mapping[str, dict]]
    added: Mapping[str, tuple[Slip, ...]]
    explained: Mapping[str, tuple[Explained, ...]]


def load_annotations(text: MurajaText, path: Path = SESSIONS_PATH) -> Annotations:
    record = json.loads(path.read_text(encoding="utf-8"))
    statements = {st["id"]: st for st in record["statements"]}

    def said(note: dict) -> str:
        st = statements[note["statement"]]
        return f"{OWNER} {st['date']}: {st['text']}"

    mistakes: dict[str, list[Explained]] = {}
    for m in record["mistakes"]:
        if "ayah" in m:
            surah, ayah = (int(x) for x in m["ayah"].split(":"))
            first, last = f"{surah}:{ayah}:1", f"{surah}:{ayah}:{len(text.phonemes(surah, ayah).split(' '))}"
            kind, source = MISTAKE_AYAH, f"{said(m)} (the ayah holds mistakes; its words are not identified)"
        else:
            first = last = m["word"]
            kind, source = MISTAKE, said(m)
        mistakes.setdefault(m["session"], []).append(Explained(first, last, kind, OWNER, source))
    model_errors: dict[str, dict[str, dict]] = {}
    for e in record["model_errors"]:
        if e["explained_by"] not in EXPLAINED_BY:
            raise ValueError(f"a model error is explained by one of {EXPLAINED_BY}: {e}")
        model_errors.setdefault(e["session"], {})[e["word"]] = {
            "basis": OWNER, "date": statements[e["statement"]]["date"], "statement": e["statement"],
            "explained_by": e["explained_by"], "evidence": e["evidence"]}
    added: dict[str, list[Slip]] = {}
    for a in record["added"]:
        added.setdefault(a["session"], []).append(Slip(a["from"], a["to"], a["twin"], OWNER, f"{said(a)} ({a['source']})"))
    explained: dict[str, list[Explained]] = {}
    for e in record["explained"]:
        if e["kind"] not in dl.KINDS or e["basis"] != "decode":
            raise ValueError(f"explained words must be decode evidence of {dl.KINDS}: {e}")
        explained.setdefault(e["session"], []).append(Explained(e["first"], e["last"], e["kind"], e["basis"], e["source"]))
    return Annotations({e["name"]: e["sha256"] for e in record["exports"]}, statements,
                       {k: tuple(v) for k, v in mistakes.items()}, model_errors,
                       {k: tuple(v) for k, v in added.items()}, {k: tuple(v) for k, v in explained.items()})


def entry_of(session: Session, notes: Annotations, text: MurajaText) -> LogEntry:
    """A target session as #151's attribution reads a log: its explained words are the decode's
    and the owner's mistakes (both out of the false-flag rates), its slips the added phrases."""
    entry = LogEntry(
        log=session.name, commit="", committed="", model=MODEL_OF[session.model], model_basis=session.run_basis,
        app_version=session.app_version, purpose=f"page {session.page}", sources=(session.export,),
        exclusion=None, exclusion_detail=None,
        explained=(*notes.explained.get(session.name, ()), *notes.mistakes.get(session.name, ())),
        slips=notes.added.get(session.name, ()))
    return dl.with_regions(entry, lambda surah, ayah: len(text.phonemes(surah, ayah).split(" ")))


#: The model names an app run loads, and the model each is.
MODEL_OF = {"MuaalemStudent_h448_6BIT": H448}
#: The app version the target sessions must have run.
APP_VERSION = "1.0.27"
#: The words #151 found carrying the allowance-sensitive flags in the teacher era (#153).
RECURRING = ("2:19:2", "2:19:15", "2:22:2", "2:22:3", "2:23:14", "2:60:7", "2:60:8", "2:255:20")
_SETTING = re.compile(r"^\s+Pronunciation: (\w+)$")


def scorings_for(mode: str) -> tuple[Scoring, ...]:
    """Every scoring a session is graded under: #151's, each soft pair alone, and as recorded."""
    return (*SCORINGS, *PAIR_SCORINGS, replace(SHIPPED_DEFAULT, name=AS_RECORDED, mode=mode))


#: The shipped flags in each mode: the runs a session's mode is identified from.
MODE_SCORINGS = tuple(replace(SHIPPED_DEFAULT, name=f"mode:{m}", mode=m) for m in MODES)
#: The scorings the rates, causes and per-word table report (an as-recorded stand-in: only the
#: name is read).
REPORTED_SCORINGS = (*SCORINGS, Scoring(AS_RECORDED))


# --- the report --------------------------------------------------------------------------------------
def read_exports(paths: Sequence[Path], notes: Annotations) -> tuple[list[dict], list[Session]]:
    """Each export (name, SHA-256, the scoring mode set when it was exported) and the linked
    sessions of all of them; an export not pinned in the annotations is refused."""
    if sorted(p.name for p in paths) != sorted(notes.exports):
        raise SystemExit(f"give exactly the pinned exports: {sorted(notes.exports)}")
    exports, sessions = [], []
    for path in sorted(paths, key=lambda p: p.name):
        if sha256(path) != notes.exports[path.name]:
            raise SystemExit(f"{path.name} is not the pinned export (SHA-256 differs)")
        text = path.read_text(encoding="utf-8")
        mine = split_sessions(path.name, text)
        header = text.split(_TRUNCATED, 1)[0].splitlines()
        exports.append({"name": path.name, "sha256": notes.exports[path.name],
                        "mode_at_export": next(m[1] for line in header if (m := _SETTING.match(line))).lower(),
                        "sessions": [s.name for s in mine]})
        sessions += mine
    return exports, link(sessions)


def _cause_counts(grades: Sequence[WordGrade]) -> dict[str, int]:
    return {c: sum(c in g.causes for g in grades) for c in CAUSES if any(c in g.causes for g in grades)}


def _owned(g: WordGrade, kind: str) -> bool:
    return g.explained is not None and g.explained.basis == OWNER and g.explained.kind == kind


def _flag_row(g: WordGrade) -> dict:
    return {"session": g.log, "word": g.word, "quality": g.quality, "causes": list(g.causes),
            "marks": list(g.marks), "pairs": list(g.pairs)}


def catch_side(entries: Sequence[LogEntry], grades: Mapping[str, Sequence[WordGrade]]) -> dict:
    """``[scoring]`` the owner's mistakes, apart from the false flags. **Words** the owner names
    as mistakes: graded, shown as not correct, the causes and each flagged word. **Ayat** the
    owner says hold mistakes without naming the words: the flags observed in them and their
    causes only, with no denominator, since which of their words were wrong is not known."""
    out = {}
    for s in REPORTED_SCORINGS:
        mine = [g for e in entries for g in grades[e.log] if g.scoring == s.name]
        words = [g for g in mine if _owned(g, MISTAKE)]
        in_ayat = [g for g in mine if _owned(g, MISTAKE_AYAH) and g.flagged]
        flagged = [g for g in words if g.flagged]
        out[s.name] = {
            "words": {"graded": len(words), "flagged": len(flagged), "causes": _cause_counts(flagged),
                      "flagged_words": [_flag_row(g) for g in flagged]},
            "ayat": {"flagged": len(in_ayat), "causes": _cause_counts(in_ayat),
                     "flagged_words": [_flag_row(g) for g in in_ayat]},
        }
    return out


def _covers(ayat: str, word: str) -> bool:
    """Whether a run list (``2:17-2:25, 3:1``, :func:`device_logs.covered_ayat`) holds the word's ayah."""
    surah, ayah, _ = position(word)
    for run in ayat.split(", "):
        first, _, last = run.partition("-")
        (s1, a1), (s2, a2) = (tuple(int(x) for x in p.split(":")) for p in (first, last or first))
        if s1 == surah == s2 and a1 <= ayah <= a2:
            return True
    return False


def era(report151: Mapping, word: str) -> dict:
    """A word in #151's sessions, per group (the teacher's 1 s-hop sessions, all its included
    sessions, and every ``h448`` session): how many sessions graded its ayah and, per scoring,
    how many showed it as not correct. Explained words, and slip-region words (narrow), are left
    out, as #151's rates leave them out."""
    narrow = {e["log"]: [Explained(**r) for r in e["slip_regions"].get("narrow", [])] for e in report151["logs"]}
    flagged: dict[str, set[str]] = {}
    for row in report151["words"]:
        if row["word"] == word and row["explained"] is None and not any(r.covers(word) for r in narrow[row["log"]]):
            flagged.setdefault(row["scoring"], set()).add(row["log"])
    teacher = [e for e in report151["logs"] if e["model"] == dl.TEACHER and e["exclusion"] is None]
    groups = {dl.TEACHER_ONE_SECOND: [e for e in teacher if e["protocol"] == dl.ONE_SECOND], dl.TEACHER: teacher,
              H448: [e for e in report151["logs"] if e["model"] == H448 and e["exclusion"] in (None, dl.DOCUMENTED_MISTAKE)]}
    out = {}
    for group, sessions in groups.items():
        reached = {e["log"] for e in sessions if _covers(e["ayat"], word)}
        out[group] = {"sessions": len(reached), "flagged": {s.name: len(reached & flagged.get(s.name, set())) for s in SCORINGS}}
    return out


def recurring(entries: Sequence[LogEntry], grades: Mapping[str, Sequence[WordGrade]], texts: Mapping[str, str],
              report151: Mapping) -> list[dict]:
    """Each of :data:`RECURRING`: in each target session that graded it, its grade per scoring
    (soft pairs alone included) and the device's grade decisions for it; and #151's era."""
    out = []
    for word in RECURRING:
        now = []
        for e in entries:
            mine = {g.scoring: g for g in grades[e.log] if g.word == word}
            if mine:
                now.append({"session": e.log, "decisions": decision_counts(texts[e.log], word),
                            "flagged": {name: {"quality": g.quality, "causes": list(g.causes), "pairs": list(g.pairs),
                                               "marks": list(g.marks)}
                                        for name, g in mine.items() if g.flagged}})
        out.append({"word": word, "now": now, "era": era(report151, word)})
    return out


def comparison(report151: Mapping) -> dict:
    """#151's rows the doc compares with: the teacher (1 s hop) and ``h448`` sensitivity rates
    per scoring, #90's ``h448`` word sample, and the allowance rows of both."""
    groups = (dl.TEACHER_ONE_SECOND, *(dl.sensitivity(H448, w) for w in SLIP_WIDTHS))
    return {
        "rates": {s.name: {dl.TEACHER_ONE_SECOND: report151["groups"][dl.TEACHER_ONE_SECOND][s.name][UNEXPLAINED]["false_flags"],
                           **{dl.sensitivity(H448, w): report151["slips_excluded"][H448][w][s.name]["false_flags"]
                              for w in SLIP_WIDTHS},
                           "simulation": report151["simulation"][H448][s.name]}
                  for s in SCORINGS},
        "allowances": {label: {g: by[g] for g in groups} for label, by in report151["allowances"].items()},
    }


def simulation_sample(ayat: Sequence[str]) -> list[dict]:
    """#90's word sample on the sessions' ayat: each sampled word's ayah, its reference word
    (1-based) and its listening verdict."""
    words = verified_words(load_words(), read_verdicts())
    covered = [w for w in words if any(_covers(a, f"{w.site.surah_ayah}:1") for a in ayat)]
    return sorted(({"ayah": w.site.surah_ayah, "reference_word": w.site.word_index + 1, "verdict": w.side}
                   for w in covered), key=lambda r: position(f"{r['ayah']}:{r['reference_word']}"))


def verdict(entry: LogEntry, word: str, notes: Annotations) -> dict:
    """What a non-correct word of a target session is (:data:`STATUSES`), and on what basis."""
    explanation = entry.explanation(word)
    if explanation is not None and explanation.basis == OWNER:
        status = MISTAKE if explanation.kind == MISTAKE else IN_MISTAKE_AYAH
        return {"status": status, "basis": OWNER, "source": explanation.source}
    if word in notes.model_errors.get(entry.log, {}):
        return {"status": MODEL_ERROR, **notes.model_errors[entry.log][word]}
    return {"status": ASSUMED, "basis": "the recitation is taken as correct; the phone never showed the word as not correct"}


def _false_flag_tally(entries: Sequence[LogEntry], grades: Mapping[str, Sequence[WordGrade]], scoring: str,
                      width: str | None = None) -> dict:
    """The consonant substitutions behind a scoring's false flags (explained words, and slip-region
    words of ``width``, left out): :func:`word_flags.substitution_tally`, and the flags."""
    flagged = [g for e in entries for g in grades[e.log] if g.scoring == scoring and g.flagged and not g.explained
               and not (width and e.in_slip(g.word, width))]
    return _tallies([g.consonants for g in flagged])


def _tallies(consonants: Sequence[Sequence[tuple[str, str]]]) -> dict:
    """The flags, and :func:`word_flags.substitution_tally` of their consonant errors by
    substitution and by expected letter."""
    return {"flags": len(consonants), "substitutions": substitution_tally(consonants),
            "expected": substitution_tally(consonants, by_expected=True)}


def device_log_tallies(pin: Harness, text: MurajaText, report151: Mapping) -> dict:
    """#151's device logs, replayed at the pin under its scorings: ``[population][scoring]`` the
    substitutions behind the false flags of the teacher's included sessions (1 s hop and all
    eras) and of every ``h448`` session, slip regions (narrow) left out. Each population's flag
    count must equal #151's committed one."""
    entries = [dl.with_regions(e, lambda surah, ayah: len(text.phonemes(surah, ayah).split(" ")))
               for e in dl.load_inventory() if e.included or e.exclusion == dl.DOCUMENTED_MISTAKE]
    directory = dl.log_dir(MURAJA_COMMIT)
    paths = {e.log: directory / f"{e.log}.log" for e in entries}
    _, replays = dl.replay_logs(pin, paths, SCORINGS)
    facts = {log: dl.read_facts(path.read_text(encoding="utf-8")) for log, path in paths.items()}
    grades = {e.log: dl.word_grades(e, replays[e.log], SCORINGS, text) for e in entries}
    teacher = [e for e in entries if e.included and e.model == dl.TEACHER]
    populations = {dl.TEACHER_ONE_SECOND: ([e for e in teacher if facts[e.log].protocol == dl.ONE_SECOND], None),
                   dl.TEACHER: (teacher, None),
                   dl.sensitivity(H448, "narrow"): (dl.device_sessions(entries, H448), "narrow")}
    out = {}
    for population, (sessions, width) in populations.items():
        out[population] = {s.name: _false_flag_tally(sessions, grades, s.name, width) for s in SCORINGS}
        committed = (report151["slips_excluded"][H448]["narrow"] if width else report151["groups"][population])
        for s in SCORINGS:
            theirs = (committed[s.name] if width else committed[s.name][UNEXPLAINED])["false_flags"]["flags"]
            if out[population][s.name]["flags"] != theirs:
                raise SystemExit(f"#151's {population} replays to {out[population][s.name]['flags']} flags under "
                                 f"{s.name}, its report has {theirs}")
    return out


def simulation_tallies(text: MurajaText) -> dict:
    """#90's word sample (its committed report): ``[model][scoring]`` the substitutions behind
    the false flags of each model's recorded-check arm, each word read on Muraja's reference."""
    report = json.loads(truth_baseline.REPORT_PATH.read_text(encoding="utf-8"))
    sites = {w.site.site_id: w.site for w in load_words()}
    out: dict = {}
    for model, arm in dl.SIMULATION_ARMS.items():
        out[model] = {}
        for scoring, cell in report["word_sample"]["arms"][arm].items():
            consonants = [consonant_errors([e for d in f["errors"] if (e := error_of(d))],
                                           text.word_phonemes(text.locate(sites[word_id])))
                          for word_id, f in cell["flags"].items() if f["quality"] not in (TASHKEEL_ERROR, "skipped")]
            out[model][scoring] = _tallies(consonants) | {"flags": len(cell["flags"])}
    return out


def build_report(paths: Sequence[Path]) -> dict:
    pin, main = dl.require(MURAJA_COMMIT), dl.require(dl.MURAJA_MAIN)
    text = MurajaText(pin.quran_db)
    notes = load_annotations(text)
    exports, sessions = read_exports(paths, notes)
    targets = [s for s in sessions if s.whole and MODEL_OF.get(s.model) == H448 and s.app_version == APP_VERSION]
    if [s for s in sessions if s.whole and s not in targets]:
        raise SystemExit("a whole session on another app or model: report it separately before going on")
    lengthened = {s.name: dl.lengthened_previews(s.text) for s in targets}
    with tempfile.TemporaryDirectory() as tmp:
        logged_paths, long_paths = {}, {}
        for s in targets:
            logged_paths[s.name] = Path(tmp) / f"{s.name}.log"
            long_paths[s.name] = Path(tmp) / f"{s.name}.lengthened.log"
            logged_paths[s.name].write_text(s.text, encoding="utf-8")
            long_paths[s.name].write_text(lengthened[s.name][0], encoding="utf-8")
        builds, parity_runs = {}, {}
        for commit, harness in (("pin", pin), ("main", main)):
            for variant, paths_ in (("logged", logged_paths), ("lengthened", long_paths)):
                builds[commit], parity_runs[(commit, variant)] = dl.replay_logs(harness, paths_, MODE_SCORINGS)
        _, replays = dl.replay_logs(pin, long_paths, (*SCORINGS, *PAIR_SCORINGS, *MODE_SCORINGS))

    device = {s.name: grade_lines(s.text) for s in targets}

    def lines(run: Mapping[str, Mapping[str, Replay]], name: str, scoring: Scoring) -> list[str]:
        return [line for line in run[name][scoring.engine_name].grade_lines if line.startswith("Grade end word")]

    modes = {s.name: max(MODE_SCORINGS, key=lambda m: parity(device[s.name], lines(parity_runs[("pin", "lengthened")], s.name, m))["matched"]).mode
             for s in targets}
    entries = [entry_of(s, notes, text) for s in targets]
    grades = {e.log: dl.word_grades(e, replays[e.log], scorings_for(modes[e.log]), text) for e in entries}
    texts = {s.name: s.text for s in targets}
    report151 = json.loads(dl.REPORT_PATH.read_text(encoding="utf-8"))

    session_rows = []
    for s in sessions:
        row = {"session": s.name, "export": s.export, "whole": s.whole, "copy_of": s.copy_of, "page": s.page,
               "kind": s.kind, "app_version": s.app_version, "model": s.model, "run_basis": s.run_basis,
               "target": s in targets}
        if s in targets:
            facts = dl.read_facts(s.text)
            entry = entries[targets.index(s)]
            if MODEL_OF[s.model] == H448 and not facts.median_inference_ms < dl.STUDENT_INFERENCE_MS:
                raise SystemExit(f"{s.name}: a median inference of {facts.median_inference_ms} ms contradicts h448")
            as_recorded = replays[s.name][replace(SHIPPED_DEFAULT, mode=modes[s.name]).engine_name]
            noted = [a for a in notes.added.get(s.name, ())]
            found = added_phrases(s.text)
            if sorted((a.start, a.end) for a in noted) != sorted((a.start, a.end) for a in found):
                raise SystemExit(f"{s.name}: the annotated added phrases {noted} are not the detector's {found}")
            row |= {
                "mode": modes[s.name], "start": facts.start, "updates": facts.updates,
                "previews_lengthened": lengthened[s.name][1], "capped_overlaps": facts.capped_overlaps,
                "median_inference_ms": facts.median_inference_ms,
                "ayat": dl.covered_ayat(as_recorded.words), "words": len(as_recorded.words),
                "ayat_words": sum(len(text.phonemes(surah, ayah).split(" "))
                                  for surah, ayah in {position(w)[:2] for w in as_recorded.words}),
                "skipped": sum(st["quality"] == "skipped" for st in as_recorded.words.values()),
                "parity": {f"{commit}/{variant}": {m.mode: parity(device[s.name], lines(run, s.name, m)) for m in MODE_SCORINGS}
                           for (commit, variant), run in parity_runs.items()},
                "commit_parity": commit_parity(s.text, as_recorded),
                "added": [vars(a) | {"twin": note.twin, "source": note.source,
                                     "kept": {w: as_recorded.words[w]["quality"] for w in (a.start, a.end)}}
                          for a in found for note in noted if (note.start, note.end) == (a.start, a.end)],
                "explained": [vars(x) for x in notes.explained.get(s.name, ())],
                "mistakes": [vars(x) for x in notes.mistakes.get(s.name, ())],
                "slip_regions": {w: [vars(x) for x in r] for w, r in entry.slip_regions.items()},
                "counts": dl.log_counts(grades[s.name], REPORTED_SCORINGS),
                "committed_flags": [c | {"verdict": verdict(entry, c["word"], notes)} for c in committed_flags(s.text)],
            }
            shown = {c["word"] for c in row["committed_flags"]} | {g.word for g in grades[s.name]
                                                                   if g.scoring == AS_RECORDED and g.flagged}
            unjudged = sorted(w for w in shown if verdict(entry, w, notes)["status"] == ASSUMED)
            unfounded = sorted(set(notes.model_errors.get(s.name, {})) - shown)
            if unjudged or unfounded:
                raise SystemExit(f"{s.name}: words the phone showed as not correct with no owner verdict {unjudged}; "
                                 f"model errors the phone never showed {unfounded}")
        session_rows.append(row)

    reported = {s.name for s in REPORTED_SCORINGS}
    catch = catch_side(entries, grades)
    return {
        "muraja": {"pin": MURAJA_COMMIT, "main": dl.MURAJA_MAIN, "builds": builds},
        "statements": list(notes.statements.values()),
        "exports": exports,
        "sessions": session_rows,
        "scorings": [s.as_dict() for s in SCORINGS] + [{"name": AS_RECORDED, "mode": "the session's own",
                                                        "soft_pairs_enabled": None, "shaddah_suppression": None,
                                                        "tashkeel_shown": True}],
        "rates": dl.views([g for e in entries for g in grades[e.log]], REPORTED_SCORINGS),
        "slips_excluded": dl.slip_sensitivity(entries, grades, REPORTED_SCORINGS)[H448],
        "allowances": {row[0]: {str(width): dl.allowance_effect(entries, grades, row, width)
                                for width in (None, *SLIP_WIDTHS)} for row in ALLOWANCE_ROWS},
        "catch": catch,
        "words": [dl.word_row(g) | {"substitutions": [label for _, label in g.consonants],
                                    "verdict": verdict(e, g.word, notes)}
                  for e in entries for g in grades[e.log] if g.flagged and g.scoring in reported],
        "substitutions": {
            "sessions": {s.name: _false_flag_tally(entries, grades, s.name) for s in REPORTED_SCORINGS},
            **{key: {s.name: _tallies([g.consonants for e in entries for g in grades[e.log]
                                       if g.scoring == s.name and g.flagged and _owned(g, kind)])
                     for s in REPORTED_SCORINGS}
               for key, kind in (("owner_mistakes", MISTAKE), ("mistake_ayat", MISTAKE_AYAH))},
            "device_logs": device_log_tallies(pin, text, report151),
            "simulation": simulation_tallies(text),
        },
        "recurring": recurring(entries, grades, texts, report151),
        "comparison": comparison(report151),
        "simulation_sample": simulation_sample([r["ayat"] for r in session_rows if r["target"]]),
    }


def main() -> None:
    from training.device_exports_doc import render

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("exports", nargs="+", type=Path, help="the owner's bug-report exports")
    report = build_report(parser.parse_args().exports)
    write_text_atomically(REPORT_PATH, json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    write_text_atomically(DOC_PATH, render(report))
    print(f"Wrote {REPORT_PATH} and {DOC_PATH}")


if __name__ == "__main__":
    main()
