"""The App arm over the whole population (#309): the app's CoreML decoder, then Muraja's engine.

One clip is graded in three steps, all of them the app's own path:

1. the WAV is the fetch's (verified against its digest manifest), and the driver
   (``tools/muraja_harness/transcribe_checks.swift``) runs the app's transcriber over it at the
   audit latency, writing every update's full hop and overlap (:func:`tracer.driver_checks`);
2. those checks are replayed through the Muraja harness at the clip's own ayah, word 1, under
   the audit's strict and balanced scorings (:func:`session.session_request`), giving one
   :func:`training.muraja_policy.word_traces` row per graded word per scoring;
3. the traces become :func:`app_run.word_grade_rows`, the committed table's line, with each
   flag's causes read by :func:`training.word_flags.flag` from the kept grade.

Everything expensive inside a worker is cached per process: the harness build probe, the Quran
text database and the WAV digest manifest are each read once, so a pool of 5,742 clip jobs does
not re-derive them.
"""

from __future__ import annotations

import gzip
import json
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tadabur.followalong_audit import tracer
from tadabur.followalong_audit.app_run import (
    ClipOutput,
    determinism_keys,
    run_clips,
    word_grade_rows,
)
from tadabur.followalong_audit.atomic import write_bytes_atomically, write_text_atomically
from tadabur.followalong_audit.clips import verify_wav
from tadabur.followalong_audit.fetch import read_manifest
from tadabur.followalong_audit.session import session_request
from training.muraja_policy import (
    AUDIT_SCORINGS,
    CURRENT_COMMIT,
    FLAGGED,
    Harness,
    MurajaText,
    SiteGrade,
    displayed,
    word_traces,
)
from training.nonpro_decode import item_key
from training.word_flags import flag

#: The per-inference latency the audit's App arm runs at (S1's h448-era median, decision 5).
DEVICE_LATENCY_MS = tracer.DEVICE_LATENCY_MS


@dataclass(frozen=True)
class AppArm:
    """One clip's App arm, as ``run_clips``' ``run_one``: checks, traces, word rows.

    ``wav_manifest_path`` rather than the manifest itself: a pool job pickles this object once
    per clip, and the 5,742-entry manifest is read once per worker instead. ``traces_dir`` keeps
    the full per-check traces only for the seeded P2 and P3 subsets; the whole-population run
    commits the graded rows alone.
    """

    driver: Path
    assets_dir: Path
    wav_dir: Path
    wav_manifest_path: Path
    quran_db: Path
    latency_ms: int = DEVICE_LATENCY_MS
    traces_dir: Path | None = None

    def __call__(self, clip: Mapping) -> ClipOutput:
        key = item_key(clip["audio_path"])
        surah, ayah = (int(part) for part in clip["filename_label"].split(":"))
        wav = Path(self.wav_dir) / f"{clip['audio_sha256']}.wav"
        verify_wav(wav, _manifest(str(self.wav_manifest_path)).get(clip["audio_sha256"]))
        text = _text(str(self.quran_db))
        word_count = len(text.phonemes(surah, ayah).split(" "))
        timed = tracer.driver_checks(Path(self.driver), wav, self.latency_ms,
                                     Path(self.assets_dir))
        request = session_request(key, surah, ayah, word_count, timed["checks"])
        _, results = _harness().run([request])
        traces = word_traces(request, results, AUDIT_SCORINGS)
        if self.traces_dir is not None:
            _write_traces(Path(self.traces_dir), clip["audio_sha256"], traces)
        return ClipOutput(word_grade_rows(key, word_count, traces, causes_of(text, surah, ayah)),
                          timed["payload_sha256"])


def causes_of(text: MurajaText, surah: int, ayah: int) -> Callable[[Mapping], Sequence[str]]:
    """Read a trace row's flag causes with :func:`training.word_flags.flag`'s rule.

    The grade the rule sees is the quality **as that scoring displays it** — the same
    :func:`training.muraja_policy.displayed` the engine's own outcome uses — so a scoring that
    hides tashkeel errors cannot be attributed causes from a grade it never showed.
    """
    phonemes = text.phonemes(surah, ayah).split(" ")
    scorings = {scoring.name: scoring for scoring in AUDIT_SCORINGS}

    def causes(trace: Mapping) -> Sequence[str]:
        shown = displayed(trace["quality"], scorings[trace["scoring"]])
        grade = SiteGrade(FLAGGED, shown, tuple(trace["errors"]))
        return flag(grade, phonemes[trace["word"] - 1]).causes

    return causes


@lru_cache(maxsize=None)
def _manifest(path: str) -> dict[str, dict]:
    return read_manifest(Path(path))


@lru_cache(maxsize=None)
def _harness() -> Harness:
    harness = Harness.locate(commit=CURRENT_COMMIT)
    if harness is None:
        raise SystemExit(f"no Muraja harness for {CURRENT_COMMIT}: build it with "
                         "`bash tools/muraja_harness/build.sh`")
    return harness


@lru_cache(maxsize=None)
def _text(quran_db: str) -> MurajaText:
    return MurajaText(Path(quran_db))


def _write_traces(traces_dir: Path, audio_sha256: str, traces: Sequence[Mapping]) -> None:
    """A subset clip's full per-check traces, one gzipped JSON line per word per scoring.

    Each row repeats the clip's whole check list, so the text is large and compresses ~25×: 1,200
    subset clips are 261 MB raw and 10 MB gzipped. Fixed mtime, so the file is byte-identical
    across runs. A session that graded no word yields no trace rows and so an empty file: the
    clip's checks are not kept (they are the S5 clips, and keeping them would need their own
    record shape).
    """
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in traces)
    write_bytes_atomically(Path(traces_dir) / f"{audio_sha256}.jsonl.gz",
                           gzip.compress(body.encode("utf-8"), mtime=0))


# --- the run's tables ---------------------------------------------------------------------------


def grades_from_results(results: Sequence[Mapping]) -> list[dict]:
    """The flat ``word_grades`` rows of a run, in the run's clip order."""
    return [row for result in results for row in result["rows"]]


def write_grades(rows: Sequence[Mapping], path: Path) -> None:
    """``word_grades.jsonl.gz``, in #300's style: sorted keys, fixed mtime, byte-identical."""
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    write_bytes_atomically(Path(path), gzip.compress(body.encode("utf-8"), mtime=0))


def read_grades(path: Path) -> list[dict]:
    """A committed word table's rows."""
    with gzip.open(Path(path), "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def clip_run_rows(clips: Sequence[Mapping], results: Sequence[Mapping]) -> list[dict]:
    """One line per clip: its provenance, how many words it produced, and any failure."""
    by_item = {result["item"]: result for result in results}
    rows = []
    for clip in clips:
        item = item_key(clip["audio_path"])
        result = by_item[item]
        strict = [row["strict"] for row in result["rows"]]
        rows.append({
            "item": item,
            "audio_path": clip["audio_path"],
            "split": clip["split"],
            "shard_file": clip["shard_file"],
            "row": clip["row"],
            "filename_label": clip["filename_label"],
            "words": len(strict),
            "graded": sum(1 for cell in strict if cell["graded"]),
            "flagged": sum(1 for cell in strict if cell["flagged"]),
            "failure": result["failure"],
        })
    return rows


def write_jsonl(rows: Sequence[Mapping], path: Path) -> None:
    """One sorted-key JSON object per line."""
    write_text_atomically(Path(path), "".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                                              + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict]:
    """A table ``write_jsonl`` wrote."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


# --- the determinism subset ---------------------------------------------------------------------


def determinism_report(clips: Sequence[Mapping], arm: AppArm, out_root: Path, *,
                       size: int, seed: int, workers: int = 1) -> dict:
    """Run the seeded subset twice in separate directories and compare the decode and the table.

    Each run's per-clip result holds the word table **and** the digest of the driver's own stdout
    (``ClipOutput.payload_sha256``): comparing the tables alone would report two different
    check schedules as identical whenever they reduce to the same flags, and two identical
    failures as identical runs. A clip that failed in either run is listed in ``failed`` and
    makes the check unsuccessful; a clip whose payload or rows differ is listed in
    ``differences``. Both directories are cleared first: a resumable cache would otherwise answer
    the second check from the first invocation's results and "agree" with itself, so a
    determinism rerun after a failure would prove nothing.
    """
    keys = determinism_keys(clips, size, seed=seed)
    subset = [clip for clip in clips if item_key(clip["audio_path"]) in keys]
    out_root = Path(out_root)
    for name in ("run1", "run2"):
        shutil.rmtree(out_root / name, ignore_errors=True)
    first = run_clips(subset, arm, out_root / "run1", workers=workers)
    second = run_clips(subset, arm, out_root / "run2", workers=workers)
    failed: list[str] = []
    differing: list[str] = []
    rows_differ: list[str] = []
    for left, right in zip(first, second):
        if left["failure"] is not None or right["failure"] is not None:
            failed.append(left["item"])
        elif left != right:
            differing.append(left["item"])
            if left["rows"] != right["rows"]:
                rows_differ.append(left["item"])
    return {
        "seed": seed,
        "size": len(subset),
        "clips": [result["item"] for result in first],
        "identical": not differing and not failed,
        "differences": differing,
        # Which of the differing clips also changed a word's graded outcome: a decode that varies
        # without moving a decision is a different finding from one that moves it.
        "rows_differ": rows_differ,
        "failed": failed,
        "payloads": {left["item"]: {"run1": left["payload_sha256"],
                                    "run2": right["payload_sha256"]}
                     for left, right in zip(first, second)},
    }


def payload_variation(clip: Mapping, arm: AppArm, out_root: Path, *, runs: int) -> dict:
    """Run one clip ``runs`` times and count the decode digests and the word tables they made.

    The seeded pair is a sample, and a pair that agrees does not show the decoder is bit-stable:
    the committed pair came back identical while one clip had already differed in an earlier
    pass. Repeating a single clip measures that instead of inferring it. Each repetition gets its
    own cleared directory — the run's cache would otherwise answer every later repetition from
    the first, and the probe would find perfect stability by construction. A repetition that
    failed is listed in ``failed_runs`` and its (empty) result is left out of both tallies.
    """
    if runs < 2:
        raise ValueError("a probe compares at least two runs of the clip")
    runs_root = Path(out_root) / "runs"
    shutil.rmtree(runs_root, ignore_errors=True)
    results = [run_clips([clip], arm, runs_root / f"{index:02d}")[0] for index in range(runs)]
    succeeded = [result for result in results if result["failure"] is None]
    payloads = Counter(result["payload_sha256"] for result in succeeded)
    return {
        "item": item_key(clip["audio_path"]),
        "latency_ms": arm.latency_ms,
        "runs": runs,
        "failed_runs": [index for index, result in enumerate(results)
                        if result["failure"] is not None],
        "payloads": dict(sorted(payloads.items())),
        "changed_words": _changed_words(succeeded),
    }


def _changed_words(results: Sequence[Mapping]) -> list[dict]:
    """The words whose row left the first run's, over the repetitions that produced a table."""
    if not results:
        return []
    first = {(row["item"], row["word"]): row for row in results[0]["rows"]}
    changed: dict[tuple[str, int], dict] = {}
    for result in results[1:]:
        for row in result["rows"]:
            key = (row["item"], row["word"])
            if first.get(key) != row:
                changed[key] = {"item": row["item"], "word": row["word"]}
    return [changed[key] for key in sorted(changed)]
