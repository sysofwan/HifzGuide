"""The drawn words' four arms, once, as one committed table (#311 PR2; prereg §5, amendment §7).

The scorer reads ``s5_draw/arms.jsonl.gz``: one row per **drawn** word (the 540 originals of
``key.jsonl``; a repeat is the same word and needs none), holding every arm's kept trace under
both audit scorings, the App arm's flag causes, and the reference F1-align matches against.

The **App** arm is the gate's own regeneration — ``reproduction.json`` pins its traces by digest,
and the build here refuses any other directory. It is never re-run: the CoreML decoder is not
bit-stable (``payload_probe.json``), so the regen the gate certified is the identity. The
**Oracle**, **Teacher** and **Student-PT** arms replay committed inputs through the same pinned
engine: the Oracle's canonical reference at the clip's phase A word times, and phase A's ``base``
and round 3's ``r3_candidate_ema`` decodes. No device runs here — the replay is CPU only.

The Oracle, Teacher and Student cells are compact (``quality, score, withheld, errors``): the
scorer reads a kept outcome from them and nothing else. The App's cells keep their whole check
list, which F1-align and the miss labels read.

Usage (from ``tools/``; the harness is the Mac's pinned build)::

  python -m tadabur.followalong_audit s5-arms --traces ~/hifzguide-scratch/fa-apparm/regen40/traces
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from tadabur.followalong_audit.atomic import write_bytes_atomically
from tadabur.followalong_audit.clips import gating_rows, oracle_eligible
from tadabur.followalong_audit.session import session_request
from tadabur.followalong_audit.sweep import (
    AyahReference, ayah_references, end_samples, oracle_request, tlog_clips,
)
from tadabur.followalong_audit.tracer import load_word_times
from tadabur.tlog_mining.candidate_inventory import BASE, GATING, R3, load_decodes
from training.muraja_policy import (
    AUDIT_SCORINGS, CURRENT_COMMIT, MurajaText, require_harness, word_traces,
)
from training.nonpro_decode import item_key

AUDIT = Path(__file__).resolve().parent
DRAW_DIR = AUDIT / "s5_draw"
APP_ARM = AUDIT / "app_arm"
ORACLE_SWEEP = AUDIT / "oracle_sweep"
#: The two scorings every arm's row holds.
SCORINGS = tuple(scoring.name for scoring in AUDIT_SCORINGS)
#: The App cell's fields: the whole trace row, less its item, word and scoring.
APP_FIELDS = ("quality", "score", "withheld", "errors", "checks", "setting_checks")
#: The compact cell the replayed arms keep: what a kept outcome needs.
COMPACT_FIELDS = ("quality", "score", "withheld", "errors")
#: A harness call's size (the sweep's own, so one clip's results stay small).
GRADE_BATCH = 512


def oracle_reference(reference: AyahReference, word: int) -> dict | None:
    """F1-align's reference for engine ``word``: its canonical text and its neighbours'.

    ``previous``, ``word`` and ``next`` are the canonical words mapped to engine words
    ``word - 1, word, word + 1`` concatenated as the oracle text joins them (prereg §5, l.284-288):
    a wasl-merged word is its canonical words with no separator. ``None`` when no canonical word is
    mapped to ``word`` (F1-align cannot fire there); each neighbour is ``None`` at the ayah's ends
    and where no canonical word is mapped to it.
    """
    if not 1 <= word <= len(reference.engine_words):
        raise ValueError(f"engine word {word} is outside the ayah's "
                         f"{len(reference.engine_words)} words")
    text = engine_text(reference)
    if word not in text:
        return None
    return {"previous": text.get(word - 1), "word": text[word], "next": text.get(word + 1)}


def engine_text(reference: AyahReference) -> dict[int, str]:
    """Each engine word's canonical words, concatenated in order."""
    mapped: dict[int, list[str]] = defaultdict(list)
    for word, engine in zip(reference.words, reference.word_of):
        mapped[engine].append(word)
    return {engine: "".join(words) for engine, words in mapped.items()}


def compact(trace: Mapping | None) -> dict | None:
    """One replayed arm's cell as committed: the kept grade alone."""
    return None if trace is None else {name: trace[name] for name in COMPACT_FIELDS}


def build_arms(traces: Path, *, log: Callable[[str], None] = lambda _message: None,
               harness=None) -> list[dict]:
    """Every drawn word's arms row, sorted by ``(item, word)``.

    ``traces`` is the gate's per-clip regeneration: its identity must equal
    ``reproduction.json``'s, or the App arm would be another run's.
    """
    from tadabur.followalong_audit.build import traces_identity

    drawn = sorted({(row["item"], row["word"]) for row in _read_jsonl(DRAW_DIR / "key.jsonl")
                    if row["repeat_of"] is None})
    pinned = json.loads((DRAW_DIR / "reproduction.json").read_text(encoding="utf-8"))["traces"]
    found = traces_identity(Path(traces))
    if found != pinned:
        raise ValueError(f"{traces} holds the traces {found}, not the gate's {pinned}: the App "
                         f"arm would be another run's")
    log(f"arms: {len(drawn)} drawn words over {len({item for item, _ in drawn})} clips")

    wanted = {item for item, _ in drawn}
    rows = [row for row in gating_rows() if item_key(row["audio_path"]) in wanted]
    population = {item_key(row["audio_path"]): row for row in rows}
    app = _app_cells(Path(traces), population, drawn)
    causes = _causes(drawn)
    log("arms: App cells read")

    engine = require_harness(commit=CURRENT_COMMIT) if harness is None else harness
    text = MurajaText(engine.quran_db)
    word_times = load_word_times()
    clips = tlog_clips(oracle_eligible(rows, word_times), word_times, end_samples(),
                       ayah_references(text))
    ayahs = ayah_references(text)
    by_item = {clip.item: clip for clip in clips}
    without = sorted(wanted - set(by_item))
    if without:
        log(f"arms: {len(without)} drawn clips have no oracle reference ({without[:3]})")

    decodes = load_decodes()[GATING]
    oracle = _grade(clips, oracle_request, engine, log, "Oracle")
    teacher = _grade(clips, _replay(decodes[BASE]), engine, log, "Teacher")
    student = _grade(clips, _replay(decodes[R3]), engine, log, "Student-PT")

    _require_outcomes(app, drawn, APP_ARM / "word_grades.jsonl.gz", "App")
    _require_outcomes(oracle, drawn, ORACLE_SWEEP / "word_grades.jsonl.gz", "Oracle")

    out = []
    for item, word in drawn:
        clip = by_item.get(item)
        out.append({
            "item": item, "word": word,
            "app": app[(item, word)], "causes": causes[(item, word)],
            "oracle": _by_word(oracle, item, word),
            "teacher": _by_word(teacher, item, word),
            "student": _by_word(student, item, word),
            "reference": None if clip is None else oracle_reference(
                ayahs(f"{clip.surah}:{clip.ayah}"), word),
        })
    return out


def _require_outcomes(cells: Mapping, drawn: Sequence[tuple[str, int]], table_path: Path,
                      name: str) -> None:
    """Refuse a replay whose kept outcomes are not the committed table's.

    The arms are the committed tables' evidence, word for word: a row that disagrees means the
    wrong decodes, weights or word times were replayed.
    """
    from tadabur.followalong_audit.attribution import kept_outcome

    table = {(row["item"], row["word"]): row for row in _read_jsonl_gz(table_path)}
    for cell in drawn:
        committed = table[cell]
        for scoring in AUDIT_SCORINGS:
            outcome = kept_outcome(cells.get(cell, {}).get(scoring.name), scoring)
            expected = _table_outcome(committed[scoring.name])
            if outcome != expected:
                raise ValueError(f"{name} {cell[0]} word {cell[1]} reads {outcome} under "
                                 f"{scoring.name}, not the committed table's {expected}")


def _table_outcome(cell: Mapping) -> str:
    """A committed word table's line as a kept outcome."""
    if not cell["graded"]:
        return "ungraded"
    return "flagged" if cell["flagged"] else "passed"


def write_arms(rows: Sequence[Mapping], path: Path) -> list[str]:
    """Write the committed table, refusing to replace a different one; ``[]`` when unchanged."""
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    data = gzip.compress(body.encode("utf-8"), mtime=0)
    target = Path(path)
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError(f"{target} holds another arms table: rebuilding it needs its own "
                             f"approved step")
        return []
    write_bytes_atomically(target, data)
    return [target.name]


# --- the arms ------------------------------------------------------------------------------------

def _app_cells(traces: Path, population: Mapping[str, Mapping],
               drawn: Sequence[tuple[str, int]]) -> dict:
    """Each drawn word's App trace under both scorings, from the gate's regeneration.

    A word the engine never graded has no trace row: its cell is ``None`` for that scoring, which
    is the table's own ``graded: false`` (a pending word is ungraded, never a pass).
    """
    wanted = set(drawn)
    cells: dict[tuple[str, int], dict] = {cell: dict.fromkeys(SCORINGS) for cell in drawn}
    for item in sorted({item for item, _ in drawn}):
        path = traces / f"{population[item]['audio_sha256']}.jsonl.gz"
        for trace in _read_jsonl_gz(path):
            cell = (trace["item"], trace["word"])
            if cell in wanted:
                cells[cell][trace["scoring"]] = {name: trace[name] for name in APP_FIELDS}
    return cells


def _causes(drawn: Sequence[tuple[str, int]]) -> dict:
    """Each drawn word's flag causes per scoring, from the committed App table."""
    rows = {(row["item"], row["word"]): row for row in _read_jsonl_gz(
        APP_ARM / "word_grades.jsonl.gz")}
    return {(item, word): {scoring: list(rows[(item, word)][scoring]["causes"])
                           for scoring in SCORINGS}
            for item, word in drawn}


def _replay(checks: Mapping[str, Sequence[dict]]) -> Callable[[object], dict]:
    """The request that replays one detector decode's committed checks."""
    def request_of(clip) -> dict:
        if clip.item not in checks:
            raise ValueError(f"{clip.item} has no committed checks in the detector decode")
        return session_request(clip.item, clip.surah, clip.ayah, len(clip.engine_words),
                               checks[clip.item])
    return request_of


def _grade(clips: Sequence, request_of: Callable[[object], dict], engine,
           log: Callable[[str], None], name: str) -> dict[tuple[str, int], dict]:
    """Replay each clip's request through the engine and keep every word's compact cells."""
    cells: dict[tuple[str, int], dict] = {}
    requests = [request_of(clip) for clip in clips]
    for start in range(0, len(requests), GRADE_BATCH):
        batch = requests[start:start + GRADE_BATCH]
        _, results = engine.run(batch)
        for request in batch:
            for trace in word_traces(request, results, AUDIT_SCORINGS):
                cells.setdefault((request["item"], trace["word"]), {})[trace["scoring"]] = compact(
                    trace)
        log(f"arms: {name} {min(start + len(batch), len(requests))}/{len(requests)} clips")
    return cells


def _by_word(cells: Mapping, item: str, word: int) -> dict:
    return {scoring: cells.get((item, word), {}).get(scoring) for scoring in SCORINGS}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]
