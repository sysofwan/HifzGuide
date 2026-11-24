"""The S1 tracer (#307): 20 clips through the app's decoder and the oracle, then gate P1b.

For every drawn clip the tracer builds two check sequences and replays each through S0's
harness in strict and balanced:

* the **App** arm, from ``tools/muraja_harness/transcribe_checks.swift`` running Muraja's own
  offline transcriber over the app's CoreML model. Only its full hops and overlaps are graded;
  the device ``.log``'s 30-Character clip is used for gate P1b and never reported.
* the **Oracle** arm, the clip's reference phonemes on its phase A teacher word times
  (:mod:`tadabur.followalong_audit.oracle`). The text is the **spaceless** phonetization
  (:func:`tadabur.waqf_segments.hafs_word_reference`) — exactly the labels those word times
  were aligned on, and the form the model emits. It runs at zero latency (decision 2: the
  Python schedule is the app's only at zero latency).

Gate P1b then replays the App checks with every overlap clipped to 30 Characters, as the
``.log`` renders them (``TranscribeLog.swift:43-46``), and reports every word whose kept
quality differs. Determinism is two driver runs over the same WAV: byte-identical check JSON
or the diff is reported, never worked around (decision 5).
"""

from __future__ import annotations

import difflib
import gzip
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tadabur.followalong_audit.assets import RELEASE, stage_assets
from tadabur.followalong_audit.clips import draw_record_path, verify_wav, wav_samples
from tadabur.followalong_audit.oracle import OracleArm, oracle_checks
from tadabur.followalong_audit.p1b import (
    LOG_OVERLAP_CHARACTERS,
    changed_decisions,
    clip_overlaps,
)
from tadabur.followalong_audit.session import preview_window_ratio, session_request
from tadabur.tlog_index import sha256_file
from tadabur.waqf_segments import hafs_word_reference
from training.muraja_policy import (
    AUDIT_SCORINGS,
    CURRENT_COMMIT,
    HARNESS_ROOT,
    Harness,
    MurajaText,
    word_traces,
)
from training.nonpro_decode import item_key
from training.tlog_eval import _canonical_words

#: The per-inference wall time the app arm runs at: the h448-era device medians were 37-42 ms
#: (decision 5, #307). The Oracle arm runs at 0 (decision 2).
DEVICE_LATENCY_MS = 40

#: The committed phase A word times (``tadabur/tlog_mining/word_times``), one shard per tranche.
WORD_TIMES_DIR = Path(__file__).resolve().parents[1] / "tlog_mining" / "word_times"

#: Where the verified release is expanded for the driver. Outside the repo, and never
#: committed: the model is tens of megabytes.
DEFAULT_STAGING = Path.home() / ".cache/hifzguide/followalong-audit" / f"assets-{RELEASE}"


@dataclass
class ClipRun:
    """One clip's two arms, their timing, and why it failed if it did."""

    clip: dict
    app: list[dict] | None = None
    oracle: OracleArm | None = None
    app_request: dict | None = None
    oracle_request: dict | None = None
    samples: int = 0
    word_count: int = 0
    surah: int = 0
    ayah: int = 0
    wav_sha256: str | None = None
    wall_seconds: float | None = None
    second_wall_seconds: float | None = None
    deterministic: bool | None = None
    determinism_diff: list[str] | None = None
    inference_count: int | None = None
    failure_count: int | None = None
    alignment_reason: str | None = None
    failure: str | None = None
    #: The words any App check's own statuses graded, and the words the engine's ``final``
    #: store kept (S0's per-word trace's population). A clip can differ: the stop's alignment
    #: shows grades the store never kept when no window or hop advanced the reader.
    check_graded_words: list[int] = field(default_factory=list)
    final_words: int | None = None
    extra: dict = field(default_factory=dict)


def driver_path(root: Path = HARNESS_ROOT, commit: str = CURRENT_COMMIT) -> Path:
    """The built app-arm driver (``build.sh``'s ``transcribe_checks``), or a refusal."""
    manifest = Path(root) / commit / "harness.json"
    if not manifest.exists():
        raise SystemExit(f"no harness build at {manifest}: run `bash tools/muraja_harness/build.sh "
                         f"{commit}`")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    binary = record.get("transcribe_checks_binary")
    if not binary:
        raise SystemExit(f"{manifest} has no transcribe_checks_binary: rebuild with "
                         f"`bash tools/muraja_harness/build.sh {commit}`")
    if not Path(binary).is_file():
        raise SystemExit(f"{binary} is missing: rebuild with "
                         f"`bash tools/muraja_harness/build.sh {commit}`")
    return Path(binary)


def harness_record(commit: str = CURRENT_COMMIT, root: Path = HARNESS_ROOT) -> dict:
    """The build record the harnesses were compiled under."""
    return json.loads((Path(root) / commit / "harness.json").read_text(encoding="utf-8"))


def load_word_times(directory: Path = WORD_TIMES_DIR,
                    tranches: Sequence[str] = ("confirm.1", "confirm.2")) -> dict[str, dict]:
    """The committed phase A word times, keyed by item."""
    keyed: dict[str, dict] = {}
    for tranche in tranches:
        with gzip.open(Path(directory) / f"{tranche}.jsonl.gz", "rt", encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                keyed[record["key"]] = record
    return keyed


def _driver_once(driver: Path, wav: Path, latency_ms: int,
                 assets_dir: Path) -> tuple[bytes, bytes, float]:
    """One driver process over one WAV: its stdout, stderr and wall time, or its own failure."""
    env = {
        **os.environ,
        "HIFZGUIDE_ASSETS_DIR": str(assets_dir),
        "TRANSCRIBE_INFERENCE_LATENCY_MS": str(latency_ms),
    }
    started = time.perf_counter()
    done = subprocess.run([str(driver), str(wav)], capture_output=True, env=env, check=False)
    elapsed = time.perf_counter() - started
    if done.returncode != 0:
        raise RuntimeError(done.stderr.decode("utf-8", "replace").strip())
    return done.stdout, done.stderr, elapsed


def _driver_payload(stdout: bytes, stderr: bytes) -> dict:
    """The driver's JSON payload, or an error naming what it wrote instead.

    The driver exits 0 only after printing its payload, so a zero exit whose stdout is not JSON
    is the one failure whose cause is invisible from the exception alone: the reason carries the
    byte count, the bytes it wrote and the stderr the run would otherwise drop (#309).
    """
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"exited 0 with {len(stdout)} bytes of stdout that is not JSON ({error}): "
            f"stdout starts {stdout[:120]!r}, stderr "
            f"{stderr.decode('utf-8', 'replace').strip()[:200]!r}") from error


def driver_checks(driver: Path, wav: Path, latency_ms: int, assets_dir: Path) -> dict:
    """One clip through the driver: its full checks and timing, without the determinism rerun.

    The S3 App-arm run is 5,742 clips, so it pays for one driver run per clip and re-runs only
    its seeded determinism subset (:func:`tadabur.followalong_audit.app_arm.determinism_report`).
    """
    stdout, stderr, wall = _driver_once(driver, wav, latency_ms, assets_dir)
    payload = _driver_payload(stdout, stderr)
    return {
        "checks": payload["checks"],
        "wall_seconds": wall,
        "inference_count": payload.get("inference_count"),
        "failure_count": payload.get("failure_count"),
        "inference_latency_ms": payload.get("inference_latency_ms"),
    }


def run_driver(driver: Path, wav: Path, latency_ms: int, assets_dir: Path) -> dict:
    """One clip through the driver: its full checks, wall times, and two-run determinism.

    Raises ``RuntimeError`` with the driver's own reason when Muraja cannot publish the clip.
    """
    first, first_stderr, wall = _driver_once(driver, wav, latency_ms, assets_dir)
    second, _, second_wall = _driver_once(driver, wav, latency_ms, assets_dir)
    payload = _driver_payload(first, first_stderr)
    deterministic = first == second
    return {
        "checks": payload["checks"],
        "wall_seconds": wall,
        "second_wall_seconds": second_wall,
        "deterministic": deterministic,
        "determinism_diff": None if deterministic else _json_diff(first, second),
        "inference_count": payload.get("inference_count"),
        "failure_count": payload.get("failure_count"),
        "inference_latency_ms": payload.get("inference_latency_ms"),
    }


def _json_diff(first: bytes, second: bytes) -> list[str]:
    """The first lines that differ between two driver outputs, for the run note."""
    left = json.dumps(json.loads(first), ensure_ascii=False, indent=2, sort_keys=True).splitlines()
    right = json.dumps(json.loads(second), ensure_ascii=False, indent=2, sort_keys=True).splitlines()
    return list(difflib.unified_diff(left, right, "run 1", "run 2", lineterm=""))[:40]


def prepare_clip(clip: Mapping, driver: Path, wav_dir: Path, assets_dir: Path,
                 word_times: Mapping[str, dict], text: MurajaText,
                 wav_manifest: Mapping[str, Mapping], latency_ms: int) -> ClipRun:
    """Read one clip's WAV, run both arms, and keep whatever failed on the record.

    The WAV must be the file the fetch wrote (:func:`tadabur.followalong_audit.clips.verify_wav`):
    its bytes are what the App arm grades, so a truncated or substituted copy fails the clip
    rather than entering the table.
    """
    run = ClipRun(clip=dict(clip))
    try:
        key = item_key(clip["audio_path"])
        run.surah, run.ayah = (int(part) for part in clip["filename_label"].split(":"))
        wav = Path(wav_dir) / f"{clip['audio_sha256']}.wav"
        run.samples = wav_samples(wav)
        run.wav_sha256 = verify_wav(wav, wav_manifest.get(clip["audio_sha256"]))
        run.word_count = len(text.phonemes(run.surah, run.ayah).split(" "))
    except (OSError, ValueError, KeyError) as error:
        run.failure = f"clip setup: {error}"
        return run

    record = word_times.get(key)
    if record is None:
        run.failure = f"no committed word times for {key}"
    elif record["surah_ayah"] != clip["filename_label"] or \
            record["audio_sha256"] != clip["audio_sha256"]:
        run.failure = (f"the word times for {key} name {record['surah_ayah']} / "
                       f"{record['audio_sha256']}")
    if run.failure:
        return run

    try:
        timed = run_driver(driver, wav, latency_ms, assets_dir)
    except RuntimeError as error:
        run.failure = f"driver: {error}"
        return run
    run.app = timed.pop("checks")
    run.wall_seconds = timed.pop("wall_seconds")
    run.second_wall_seconds = timed.pop("second_wall_seconds")
    run.deterministic = timed.pop("deterministic")
    run.determinism_diff = timed.pop("determinism_diff")
    run.inference_count = timed.pop("inference_count")
    run.failure_count = timed.pop("failure_count")
    run.extra["inference_latency_ms"] = timed.pop("inference_latency_ms")

    try:
        words = _canonical_words(clip["filename_label"])
        reference, offsets = hafs_word_reference()(words)
        times = [tuple(pair) for pair in record["word_times"]] if record["aligned"] else None
        run.oracle = oracle_checks(reference, offsets, times, run.samples)
        run.alignment_reason = record.get("reason")
    except (KeyError, IndexError, ValueError) as error:
        run.failure = f"oracle: {error}"
    return run


def arm_requests(run: ClipRun) -> list[dict]:
    """The engine requests for the arms ``run`` actually produced."""
    key = item_key(run.clip["audio_path"])
    requests = []
    if run.app is not None:
        run.app_request = session_request(f"{key}/app", run.surah, run.ayah, run.word_count,
                                          run.app)
        requests.append(run.app_request)
    if run.oracle is not None:
        run.oracle_request = session_request(f"{key}/oracle", run.surah, run.ayah,
                                             run.word_count, run.oracle.checks)
        requests.append(run.oracle_request)
    return requests


def run_tracer(clips: Sequence[Mapping], *, clip_list: Path, wav_dir: Path, assets_dir: Path,
               manifest: Mapping, wav_manifest: Mapping[str, Mapping], out: Path,
               latency_ms: int = DEVICE_LATENCY_MS, staging_dir: Path | None = None,
               draw: Mapping | None = None) -> dict:
    """Run both arms and gate P1b over ``clips``; write the tables under ``out``.

    ``assets_dir`` is the release cache holding the verified files; the driver is handed a
    staging directory built from them (:func:`tadabur.followalong_audit.assets.stage_assets`),
    so the model it loads is the manifest's bytes. ``wav_manifest`` is the fetch's digest
    record of the WAVs; every clip is verified against it before it is run and the record is
    written beside the tables. ``draw`` is the clip list's selection record, written into the
    run's metadata and note.
    """
    record = harness_record()
    if record.get("hifzguide_release") != RELEASE:
        raise SystemExit(f"the harness was built against {record.get('hifzguide_release')}, not "
                         f"{RELEASE}: the App arm would run weights the app does not ship")
    staged = stage_assets(Path(assets_dir), manifest, staging_dir or DEFAULT_STAGING)
    driver = driver_path()
    harness = Harness.locate(commit=CURRENT_COMMIT)
    if harness is None:
        raise SystemExit(f"no Muraja harness for {CURRENT_COMMIT} under {HARNESS_ROOT}")
    text = MurajaText(harness.quran_db)
    word_times = load_word_times()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    runs = [prepare_clip(clip, driver, Path(wav_dir), staged.directory, word_times, text,
                         wav_manifest, latency_ms) for clip in clips]
    requests = [request for run in runs for request in arm_requests(run)]
    _, results = harness.run(requests)

    rows: list[dict] = []
    for run in runs:
        for arm, request in (("app", run.app_request), ("oracle", run.oracle_request)):
            if request is not None:
                rows.extend({**row, "arm": arm}
                            for row in word_traces(request, results, AUDIT_SCORINGS))
    for run in runs:
        if run.app_request is None:
            continue
        written = {row["word"] for row in rows
                   if row["arm"] == "app" and row["item"] == run.app_request["item"]}
        run.final_words = len(written)
        run.check_graded_words = sorted({
            status["word"] for result in results if result["item"] == run.app_request["item"]
            for check in result["checks"] for status in check["statuses"]})

    # Gate P1b: the App checks with each overlap cut to the 30 Characters a `.log` renders.
    clipped = [run for run in runs if run.app_request is not None]
    clipped_requests = [
        session_request(run.app_request["item"], run.surah, run.ayah, run.word_count,
                        clip_overlaps(run.app))
        for run in clipped
    ]
    _, clipped_results = harness.run(clipped_requests)
    clipped_rows = [row for request in clipped_requests
                    for row in word_traces(request, clipped_results, AUDIT_SCORINGS)]
    app_rows = [row for row in rows if row["arm"] == "app"]
    changed = changed_decisions(app_rows, clipped_rows)
    coverage = {"drawn": len(clips), "replayed": len(clipped)}
    if coverage["replayed"] != coverage["drawn"]:
        verdict = "incomplete"
    elif changed:
        verdict = "pass"
    elif coverage["drawn"] < 100:
        verdict = "extend-to-100"
    else:
        verdict = "fail-no-difference"

    p1b = {
        "gate": "P1b",
        "clips": coverage,
        "verdict": verdict,
        "log_overlap_characters": LOG_OVERLAP_CHARACTERS,
        "changed": changed,
    }
    both_arms = sum(1 for run in runs if run.app is not None and run.oracle is not None)
    # P1b's verdict is the gate's own App-only coverage; a run is complete only when every
    # drawn clip produced both arms, so an Oracle failure cannot pass as a finished run.
    complete = bool(clips) and both_arms == len(clips) and not any(
        run.failure is not None for run in runs)
    meta = {
        "issue": 307,
        "clips": len(clips),
        "both_arms": both_arms,
        "failed": [{"item": item_key(run.clip["audio_path"]), "reason": run.failure}
                   for run in runs if run.failure is not None],
        "clip_list": clip_list.name,
        "clip_list_sha256": sha256_file(clip_list),
        "draw": dict(draw) if draw is not None else None,
        "latency_ms": latency_ms,
        "scorings": [scoring.name for scoring in AUDIT_SCORINGS],
        "assets": {
            "release": RELEASE,
            "staging_dir": str(staged.directory),
            "model": staged.model,
            "model_sha256": staged.model_sha256,
            "checksums": staged.checksums,
        },
        "harness": record,
        "driver": str(driver),
    }
    write_outputs(out, rows, runs, p1b, meta, latency_ms, draw, wav_manifest, clip_list)
    return {"words": len(rows), "changed": changed, "verdict": verdict,
            "complete": complete, "failed": len(meta["failed"])}


def write_outputs(out: Path, rows: Sequence[Mapping], runs: Sequence[ClipRun],
                  p1b: Mapping, meta: Mapping, latency_ms: int,
                  draw: Mapping | None, wav_manifest: Mapping[str, Mapping],
                  clip_list: Path) -> None:
    """Write the tables and everything the committed output must state about itself.

    The clip list is copied byte for byte (its digest in ``meta`` is of the drawn input), and the
    selection record beside it, so the output directory holds the population and provenance the
    tables were produced from. A list that already lives in the output directory is left as is.
    """
    out = Path(out)
    if Path(clip_list).resolve() != (out / clip_list.name).resolve():
        shutil.copyfile(clip_list, out / clip_list.name)
    if draw is not None:
        write_json(draw_record_path(out / clip_list.name), draw)
    write_jsonl(Path(out) / "words.jsonl", rows)
    write_jsonl(Path(out) / "checks.jsonl",
                [{"item": item_key(run.clip["audio_path"]), "arm": "app", "checks": run.app}
                 for run in runs if run.app is not None]
                + [{"item": item_key(run.clip["audio_path"]), "arm": "oracle",
                    "checks": run.oracle.checks} for run in runs if run.oracle is not None])
    run_rows = [clip_run_row(run) for run in runs]
    write_jsonl(Path(out) / "clip_runs.jsonl", run_rows)
    write_json(Path(out) / "p1b.json", p1b)
    wav_manifest_path = Path(out) / "wav_sha256.json"
    write_json(wav_manifest_path, wav_manifest)
    write_json(Path(out) / "meta.json",
               {**meta, "wav_manifest_sha256": sha256_file(wav_manifest_path)})
    (Path(out) / "run_note.md").write_text(run_note(run_rows, p1b, latency_ms, draw),
                                           encoding="utf-8")


def clip_run_row(run: ClipRun) -> dict:
    """One clip's provenance, cadence and timings (no audio, no absolute WAV path)."""
    return {
        **{name: run.clip.get(name) for name in (
            "audio_path", "audio_sha256", "split", "shard_file", "row", "filename_label")},
        "item": item_key(run.clip["audio_path"]),
        "surah": run.surah,
        "ayah": run.ayah,
        "wav_sha256": run.wav_sha256,
        "samples": run.samples,
        "word_count": run.word_count,
        "app_preview_window": preview_window_ratio(run.app) if run.app is not None else None,
        "oracle_preview_window": (preview_window_ratio(run.oracle.checks)
                                  if run.oracle is not None else None),
        "oracle_aligned": run.oracle.aligned if run.oracle is not None else None,
        "alignment_reason": run.alignment_reason,
        "app_checks": len(run.app) if run.app is not None else None,
        "app_final_words": run.final_words,
        "app_check_graded_words": run.check_graded_words,
        "oracle_checks": len(run.oracle.checks) if run.oracle is not None else None,
        "wall_seconds": run.wall_seconds,
        "second_wall_seconds": run.second_wall_seconds,
        "deterministic": run.deterministic,
        "determinism_diff": run.determinism_diff,
        "inference_count": run.inference_count,
        "failure_count": run.failure_count,
        "inference_latency_ms": run.extra.get("inference_latency_ms"),
        "failure": run.failure,
    }


def run_note(rows: Sequence[Mapping], p1b: Mapping, latency_ms: int,
             draw: Mapping | None = None) -> str:
    """The short run note: per-clip wall time (S3 sizing), cadence, selection, and failures.

    ``rows`` are :func:`clip_run_row`'s records, the same ones ``clip_runs.jsonl`` holds, so the
    note can be regenerated from the committed run without the WAVs or the driver.
    """
    both_arms = sum(1 for row in rows if row["failure"] is None
                    and row["app_checks"] is not None and row["oracle_checks"] is not None)
    lines = [
        "# S1 tracer run (#307)",
        "",
        f"{both_arms}/{len(rows)} clips produced both arms.",
        "",
        _selection_line(draw),
        "",
        f"The App arm ran Muraja's transcriber at {latency_ms} ms per inference; the Oracle arm "
        "fills the clip's reference phonemes onto the phase A teacher word times and is cut on "
        "the zero-latency Python schedule (decision 2). The arms also differ in their own audio "
        "and output gates and silence handling, so the preview:window ratios below are not a "
        "latency-only comparison and neither bounds the other.",
        "",
        "App rows are the words Muraja's grade store kept (S0's `word_traces` population); the "
        "Oracle arm covers every word of the ayah.",
        "",
        "| clip | ayah | wall s | run 2 s | app preview:window | oracle preview:window | aligned |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['item']} | {row['filename_label']} | {_seconds(row['wall_seconds'])} | "
            f"{_seconds(row['second_wall_seconds'])} | "
            f"{_ratio(row['app_preview_window'])} | {_ratio(row['oracle_preview_window'])} | "
            f"{row['oracle_aligned']} |")
    coverage = p1b["clips"]
    lines += ["", f"P1b: {p1b['verdict']} — {len(p1b['changed'])} changed word decision(s); "
                  f"{coverage['replayed']}/{coverage['drawn']} drawn clips replayed."]
    unkept = [row for row in rows if row["app_final_words"] == 0 and row["app_check_graded_words"]]
    lines += ([f"App tables hold the words the engine's grade store kept; {len(unkept)} clip(s) "
               "graded words at a check but kept none (no window or hop advanced the reader):"]
              if unkept else [])
    lines += [f"- {row['audio_path']}: checks graded words "
              f"{row['app_check_graded_words']}, `final` kept none" for row in unkept]
    failures = [row for row in rows if row["failure"] is not None]
    lines += ["", "Failures:"]
    lines += ([f"- {row['audio_path']}: {row['failure']}" for row in failures]
              if failures else ["- none"])
    flaky = [row for row in rows if row["deterministic"] is False]
    if flaky:
        lines += ["", "Non-deterministic driver runs (CoreML; Muraja not patched — decision 5):"]
        for row in flaky[:3]:
            lines += [f"- {row['audio_path']}", "```", *(row["determinism_diff"] or []), "```"]
    return "\n".join(lines) + "\n"


def _selection_line(draw: Mapping | None) -> str:
    """What the draw selected from, or that the run cannot say."""
    if draw is None:
        return ("Selection: not recorded (run the `draw` command beside the clip list so it "
                "writes `clips.draw.json`).")
    return (f"Selection: seed {draw['seed']}, {draw['size']} clips from "
            f"{draw['eligible_rows']:,} of {draw['phase_a_rows']:,} phase A gating rows; "
            f"#248's exclusion left out {draw['excluded_rows']} clips of "
            f"{len(draw['excluded_ayat'])} ayat ({', '.join(draw['excluded_ayat'])}).")


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8")


def write_json(path: Path, payload: Mapping) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
