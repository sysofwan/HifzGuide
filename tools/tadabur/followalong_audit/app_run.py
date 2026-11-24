"""The App-arm run's per-word table and its resumable, parallel clip driver (#309).

``word_grade_rows`` is the committed ``word_grades.jsonl.gz`` line: one row per word of the ayah,
under each of the audit's two scorings, with that scoring's outcome (``graded``/``flagged``) and
the causes of a flagged word. A word the engine's grade store never kept is ``graded: false`` —
it is the S5 stratum, not a gap in the run.

``run_clips`` runs ``run_one(clip) -> rows`` over the clips and persists each **successful**
clip's result under ``out_dir``, so a killed or partly failed run resumes: finished clips are not
run again and a failed clip is retried. Failures are results too (``failure`` with the reason,
empty rows), never an aborted run: with 5,742 clips some will fail and the table must say which.
More than one worker runs the clips in a process pool, so the CoreML sleeps of one clip overlap
another's; ``run_one`` must then be picklable.
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tadabur.followalong_audit.atomic import write_text_atomically
from tadabur.panel_seal import SealedPanelError
from training.muraja_policy import AUDIT_SCORINGS, CORRECT, PENDING, Scoring, displayed


def word_grade_rows(item: str, word_count: int, traces: Sequence[Mapping],
                    causes_of: Callable[[Mapping], Sequence[str]]) -> list[dict]:
    """One row per word of the ayah (``1..word_count``), each scoring's cell filled from its trace.

    ``traces`` are :func:`training.muraja_policy.word_traces` rows for this item; ``causes_of``
    names why a flagged word is flagged and is only consulted for a flagged word.
    """
    by_word: dict[tuple[str, int], Mapping] = {}
    for trace in traces:
        if trace["item"] != item:
            raise ValueError(f"{trace['item']} is not {item}: a trace stays with its item")
        word = trace["word"]
        if not 1 <= word <= word_count:
            raise ValueError(f"{item} word {word} is outside the ayah's 1..{word_count}")
        key = (trace["scoring"], word)
        if key in by_word:
            raise ValueError(f"{item} word {word} has two {trace['scoring']} traces")
        by_word[key] = trace
    rows = []
    for word in range(1, word_count + 1):
        row = {"item": item, "word": word}
        for scoring in AUDIT_SCORINGS:
            row[scoring.name] = _cell(by_word.get((scoring.name, word)), scoring, causes_of)
        rows.append(row)
    return rows


def _cell(trace: Mapping | None, scoring: Scoring,
          causes_of: Callable[[Mapping], Sequence[str]]) -> dict:
    """One scoring's outcome for one word: ungraded, graded and passed, or graded and flagged.

    "Flagged" is read the engine's way, through :func:`training.muraja_policy.displayed`: the
    quality as this scoring shows it, so a scoring that hides tashkeel errors or runs lenient is
    graded by its own rule instead of by ``quality != correct``.
    """
    if trace is None or trace["quality"] == PENDING:
        return {"graded": False, "flagged": False, "causes": []}
    if displayed(trace["quality"], scoring) == CORRECT:
        return {"graded": True, "flagged": False, "causes": []}
    return {"graded": True, "flagged": True, "causes": list(causes_of(trace))}


# --- the resumable run ----------------------------------------------------------------------------

_RESULT_FIELDS = ("item", "failure", "rows")

#: How many times one clip is attempted before its failure is recorded. The CoreML driver
#: occasionally exits 0 having written nothing (#309): retrying costs one clip and keeps a
#: transient from entering the table as an exclusion, while a clip that fails every attempt is
#: still recorded as failed, never as a silent pass.
ATTEMPTS = 3


def _cache_path(out_dir: Path, clip: Mapping) -> Path:
    """A clip's result file: its source checksum, so a resumed run finds it without the WAVs."""
    return Path(out_dir) / f"{clip['audio_sha256']}.json"


def _write_result(path: Path, result: Mapping) -> None:
    write_text_atomically(path, json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")


def _run_cached(job: tuple[dict, Callable, str]) -> dict:
    """One clip's result: its cached one when a previous run finished it, else ``run_one``'s.

    A failed clip is not cached, so a rerun retries exactly the clips that failed. Each attempt
    before the last is announced on stderr, so a run's log records the transient that a retry
    absorbed.
    """
    from training.nonpro_decode import item_key

    clip, run_one, out_dir = job
    path = _cache_path(Path(out_dir), clip)
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if set(cached) == set(_RESULT_FIELDS):
            return cached
    item = item_key(clip["audio_path"])
    reason = ""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            rows = list(run_one(clip))
        except SealedPanelError:  # a seal violation is never a flaky clip
            raise
        except Exception as error:  # noqa: BLE001 - the clip's reason, never the run's death
            reason = f"{type(error).__name__}: {error}"
            if attempt < ATTEMPTS:
                print(f"retrying {item} after attempt {attempt}/{ATTEMPTS}: {reason}",
                      file=sys.stderr, flush=True)
            continue
        result = {"item": item, "failure": None, "rows": rows}
        _write_result(path, result)
        return result
    return {"item": item, "failure": reason, "rows": []}


#: The run-identity record's name: what a cache directory was produced by.
IDENTITY_NAME = "run_identity.json"


def check_identity(out_dir: Path, identity: Mapping | None) -> None:
    """Refuse a cache directory that a run with another identity already filled.

    A cache keyed by the clip alone would silently answer a 130 ms run from a 40 ms one, a new
    harness or new assets; the identity record makes the directory belong to exactly one
    configuration, so the P3 comparison cannot read identical rows from both sides.
    """
    if identity is None:
        return
    path = Path(out_dir) / IDENTITY_NAME
    identity = dict(identity)
    if path.is_file():
        found = json.loads(path.read_text(encoding="utf-8"))
        if found != identity:
            differing = sorted(key for key in set(found) | set(identity)
                               if found.get(key) != identity.get(key))
            raise ValueError(f"{out_dir} holds another run's results ({', '.join(differing)}): "
                             "give each configuration its own --out directory")
        return
    write_text_atomically(path, json.dumps(identity, ensure_ascii=False, indent=2,
                                           sort_keys=True) + "\n")


def run_clips(clips: Sequence[Mapping], run_one: Callable[[Mapping], list[dict]],
              out_dir: Path, *, workers: int = 1,
              identity: Mapping | None = None) -> list[dict]:
    """Run every clip, in the input's order, keeping each finished result under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_identity(out_dir, identity)
    jobs = [(clip, run_one, str(out_dir)) for clip in clips]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_run_cached, jobs))
    return [_run_cached(job) for job in jobs]


def require_accounted(clips: Sequence[Mapping], results: Sequence[Mapping]) -> None:
    """Refuse a run where a clip has no result, or neither rows nor a failure reason."""
    from training.nonpro_decode import item_key

    by_item = {result["item"]: result for result in results}
    for clip in clips:
        item = item_key(clip["audio_path"])
        result = by_item.get(item)
        if result is None:
            raise ValueError(f"{clip['audio_path']} has no result")
        if result["failure"] is None and not result["rows"]:
            raise ValueError(f"{clip['audio_path']} has neither rows nor a failure reason")


def determinism_keys(clips: Sequence[Mapping], size: int, *, seed: int) -> set[str]:
    """``size`` clip keys to re-run for determinism, drawn under ``seed`` from the population.

    The draw is sorted and shuffled, so it depends on the set of clips and the seed only: a
    reversed input picks the same keys, the same seed picks them again.
    """
    from training.nonpro_decode import item_key

    keys = sorted({item_key(clip["audio_path"]) for clip in clips})
    if size > len(keys):
        raise ValueError(f"cannot draw {size} determinism clips from {len(keys)}")
    random.Random(seed).shuffle(keys)
    return set(keys[:size])
