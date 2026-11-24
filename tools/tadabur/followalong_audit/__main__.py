"""The follow-along audit's CLI (#307, #309).

S1's tracer:

    # the seeded draw (claim-gated phase A rows of confirm.1 and confirm.2, minus #248's
    # unreadable ayat) and the selection record it writes beside the list
    python -m tadabur.followalong_audit draw --size 20 --seed 7 \
        --out tadabur/followalong_audit/tracer/clips.jsonl

    # both arms over the fetched WAVs, then gate P1b
    python -m tadabur.followalong_audit run --clips tadabur/followalong_audit/tracer/clips.jsonl \
        --wav-dir ~/.cache/hifzguide/followalong-audit/wav \
        --assets-dir <release v1.0.7 cache> --manifest <manifest.json> \
        --out tadabur/followalong_audit/tracer --latency-ms 40

S3's App arm over the whole population:

    # fetch every gating clip's WAV (resumable; one shard per worker, each deleted once read)
    python -m tadabur.followalong_audit fetch --wav-dir <scratch>/wav \
        --shard-cache <scratch>/shards --workers 4

    # the App arm at 40 ms, resumable and parallel, then the committed tables
    python -m tadabur.followalong_audit app-run --out <scratch>/app40 --wav-dir <scratch>/wav \
        --assets-dir <release v1.0.7 cache> --manifest <manifest.json> --workers 4

    # P3 at 0 ms and 130 ms on one seeded 500-clip subset (the same --size/--seed both times)
    python -m tadabur.followalong_audit app-run --out <scratch>/app0 --size 500 --seed 309 \
        --latency-ms 0 ... --grades-out tadabur/followalong_audit/app_arm/word_grades.0ms.jsonl.gz \
        --traces-dir tadabur/followalong_audit/app_arm/traces/p3_0
    python -m tadabur.followalong_audit app-run --out <scratch>/app130 --size 500 --seed 309 \
        --latency-ms 130 ... --grades-out .../word_grades.130ms.jsonl.gz --traces-dir ...

    # determinism on the seeded subset, one clip repeated (the pair is a sample), the detectors'
    # strata, then the summary and doc.
    # Each cache directory records its identity (latency, weights, harness, driver, audio and
    # clip set) and refuses a run that would reuse another configuration's results.
    python -m tadabur.followalong_audit determinism --out-root <scratch>/determinism ...
    python -m tadabur.followalong_audit determinism-probe --item <clip item key> \
        --out-root <scratch>/probe --runs 20 ...
    python -m tadabur.followalong_audit strata            # S2: add --oracle <word table>
    python -m tadabur.followalong_audit report            # S2: add --oracle <summary>

S2's oracle and planted-mistake sweeps (CPU only, no audio):

    python -m tadabur.followalong_audit oracle-sweep

The WAVs are named ``<audio_sha256>.wav`` and live outside the repo; the run writes only decode
text and provenance. The release cache holds the verified ``models.zip``, ``mel_filters.bin``
and ``window.bin``; the run expands them into a staging directory (never the repo) and hands that
to the driver. ``run``'s exit status is non-zero unless every drawn clip produced both arms with
no preparation failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tadabur.followalong_audit.assets import RELEASE, stage_assets
from tadabur.followalong_audit.clips import (
    app_arm_rows,
    draw_clips,
    draw_record_path,
    gating_rows,
    oracle_eligible,
    read_clips,
    selected_clips,
    selection_record,
    write_clips,
)
from tadabur.followalong_audit.fetch import fetch_clips
from tadabur.followalong_audit.provenance import (
    PROVENANCE_PATH,
    provenance_rows,
    write_provenance,
)
from tadabur.followalong_audit.tracer import (
    DEFAULT_STAGING,
    DEVICE_LATENCY_MS,
    driver_path,
    harness_record,
    load_word_times,
    run_tracer,
    write_json,
)


def main() -> int:
    from tadabur.followalong_audit.sweep import CLEAN_FINDINGS_FILE, PLANTED_ROWS_FILE

    parser = argparse.ArgumentParser(prog="python -m tadabur.followalong_audit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    draw = commands.add_parser("draw", help="draw the seeded clip list")
    draw.add_argument("--size", type=int, default=20)
    draw.add_argument("--seed", type=int, default=7)
    draw.add_argument("--out", type=Path, required=True)

    run = commands.add_parser("run", help="run both arms and gate P1b")
    run.add_argument("--clips", type=Path, required=True)
    _assets_arguments(run)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--draw", type=Path,
                     help="the clip list's selection record (default: <clips>.draw.json)")
    run.add_argument("--latency-ms", type=int, default=DEVICE_LATENCY_MS)

    fetch = commands.add_parser("fetch", help="fetch the population's WAVs and provenance")
    fetch.add_argument("--wav-dir", type=Path, required=True)
    fetch.add_argument("--shard-cache", type=Path, required=True)
    fetch.add_argument("--workers", type=int, default=1)
    fetch.add_argument("--manifest-out", type=Path, default=PROVENANCE_PATH)

    app_run = commands.add_parser("app-run", help="grade the population with the App arm")
    _assets_arguments(app_run)
    app_run.add_argument("--out", type=Path, required=True,
                         help="the resumable run's per-clip result cache")
    app_run.add_argument("--workers", type=int, default=1)
    app_run.add_argument("--latency-ms", type=int, default=DEVICE_LATENCY_MS)
    app_run.add_argument("--clips", type=Path,
                         help="grade this clip list instead of the whole population")
    app_run.add_argument("--size", type=int,
                         help="grade a seeded draw of this many clips (P3's 500-clip subset)")
    app_run.add_argument("--seed", type=int, default=_p3_seed(),
                         help="the draw's seed (default: P3's)")
    app_run.add_argument("--grades-out", type=Path,
                         help="the committed word table (default: the run directory's)")
    app_run.add_argument("--clip-runs-out", type=Path,
                         help="one line per clip (default: the run directory's clip_runs.jsonl)")
    app_run.add_argument("--traces-dir", type=Path,
                         help="keep full per-check traces here (the P2 and P3 subsets only)")

    determinism = commands.add_parser("determinism", help="re-run the seeded subset twice")
    _assets_arguments(determinism)
    determinism.add_argument("--out-root", type=Path, required=True)
    determinism.add_argument("--workers", type=int, default=1)
    determinism.add_argument("--size", type=int, default=25)
    determinism.add_argument("--seed", type=int, default=319)
    determinism.add_argument("--latency-ms", type=int, default=DEVICE_LATENCY_MS)

    probe = commands.add_parser("determinism-probe",
                                help="repeat one clip to measure the decode's variation")
    _assets_arguments(probe)
    probe.add_argument("--item", required=True,
                       help="the clip's item key, as the committed tables name it")
    probe.add_argument("--out-root", type=Path, required=True)
    probe.add_argument("--runs", type=int, default=20)
    probe.add_argument("--latency-ms", type=int, default=DEVICE_LATENCY_MS)
    probe.add_argument("--out", type=Path, default=_app_arm("payload_probe.json"))

    strata = commands.add_parser("strata", help="re-grade the detectors and size the pools")
    strata.add_argument("--grades", type=Path, default=_app_arm("word_grades.jsonl.gz"))
    strata.add_argument("--clip-runs", type=Path, default=_app_arm("clip_runs.jsonl"),
                        help="the run's per-clip table: its failures are excluded, not S5")
    strata.add_argument("--oracle", type=Path,
                        help="S2's word table, when #308 has run (same shape as --grades)")
    strata.add_argument("--pools-out", type=Path, default=_app_arm("pools.json"))
    strata.add_argument("--cells-out", type=Path, default=_app_arm("cells.json"),
                        help="S1's primary-cause cells and S4's ungraded split")

    report = commands.add_parser("report", help="write the summary and the generated doc")
    report.add_argument("--grades", type=Path, default=_app_arm("word_grades.jsonl.gz"))
    report.add_argument("--clip-runs", type=Path, default=_app_arm("clip_runs.jsonl"))
    report.add_argument("--pools", type=Path, default=_app_arm("pools.json"))
    report.add_argument("--p3-zero", type=Path, default=_app_arm("word_grades.0ms.jsonl.gz"))
    report.add_argument("--p3-130", type=Path, default=_app_arm("word_grades.130ms.jsonl.gz"))
    report.add_argument("--traces-root", type=Path, default=_app_arm("traces"),
                        help="the committed subset traces, read for the App arm's own cadence")
    report.add_argument("--determinism", type=Path, default=_app_arm("determinism.json"),
                        help="the seeded two-run check, quoted in the doc's run block")
    report.add_argument("--payload-probe", type=Path, default=_app_arm("payload_probe.json"),
                        help="one clip repeated, quoted beside the seeded pair when it varies")
    report.add_argument("--exports-dir", type=Path, default=Path.home() / "Downloads")
    report.add_argument("--oracle", type=Path, help="S2's oracle summary, when #308 has run")
    report.add_argument("--reuse-p2-p3", type=Path,
                        help="an existing summary whose P2/P3 to keep (an unchanged App-arm"
                             " run, when only S2's oracle sections are being filled)")
    report.add_argument("--p2-clips", type=int, default=_p2_clips())
    report.add_argument("--p2-seed", type=int, default=_p2_seed())
    report.add_argument("--p3-seed", type=int, default=_p3_seed())
    report.add_argument("--latency-ms", type=int, default=DEVICE_LATENCY_MS)
    report.add_argument("--no-all-clips", action="store_true",
                        help="skip the whole-population P2 agreement (default: report it)")
    report.add_argument("--summary-out", type=Path, default=_app_arm("summary.json"))
    report.add_argument("--doc-out", type=Path, default=_doc_path())

    oracle = commands.add_parser("oracle-sweep",
                                 help="run the clean and planted oracle sweeps")
    oracle.add_argument("--seed", type=int, default=_sweep_seed())
    oracle.add_argument("--report-out", type=Path, default=_sweep_report())
    oracle.add_argument("--doc-out", type=Path, default=_sweep_doc())
    oracle.add_argument("--word-grades-out", type=Path,
                        default=_oracle_sweep_dir() / "word_grades.jsonl.gz")
    oracle.add_argument("--clean-findings-out", type=Path,
                        default=_oracle_sweep_dir() / CLEAN_FINDINGS_FILE)
    oracle.add_argument("--planted-rows-out", type=Path,
                        default=_oracle_sweep_dir() / PLANTED_ROWS_FILE)
    oracle.add_argument("--summary-out", type=Path,
                        default=_oracle_sweep_dir() / "summary.json")
    oracle.add_argument("--run-note-out", type=Path,
                        default=_oracle_sweep_dir() / "run_note.md")

    args = parser.parse_args()
    if args.command == "draw":
        return _draw(args)
    if args.command == "fetch":
        return _fetch(args)
    if args.command == "app-run":
        return _app_run(args)
    if args.command == "determinism":
        return _determinism(args)
    if args.command == "determinism-probe":
        return _determinism_probe(args)
    if args.command == "strata":
        return _strata(args)
    if args.command == "report":
        return _report(args)
    if args.command == "oracle-sweep":
        return _oracle_sweep(args)
    return _run(args)


def _assets_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--wav-dir", type=Path, required=True)
    command.add_argument("--wav-manifest", type=Path,
                         help="the fetch's WAV digest manifest (default: <wav-dir>/wav_sha256.json)")
    command.add_argument("--assets-dir", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--staging-dir", type=Path,
                         help="where to expand the verified release (default: ~/.cache)")


def _p2_clips() -> int:
    from tadabur.followalong_audit.bridge import P2_CLIPS

    return P2_CLIPS


def _p2_seed() -> int:
    from tadabur.followalong_audit.bridge import P2_SEED

    return P2_SEED


def _p3_seed() -> int:
    from tadabur.followalong_audit.cadence import P3_SEED

    return P3_SEED


def _app_arm(name: str) -> Path:
    from tadabur.followalong_audit.app_doc import APP_ARM_DIR

    return APP_ARM_DIR / name


def _doc_path() -> Path:
    from tadabur.followalong_audit.app_doc import DOC_PATH

    return DOC_PATH


def _sweep_seed() -> int:
    from tadabur.followalong_audit.sweep import SWEEP_SEED

    return SWEEP_SEED


def _oracle_sweep_dir() -> Path:
    from tadabur.followalong_audit.sweep import ORACLE_SWEEP_DIR

    return ORACLE_SWEEP_DIR


def _sweep_report() -> Path:
    from tadabur.followalong_audit.sweep_doc import REPORT_PATH

    return REPORT_PATH


def _sweep_doc() -> Path:
    from tadabur.followalong_audit.sweep_doc import DOC_PATH

    return DOC_PATH


def _wav_manifest_path(args) -> Path:
    return args.wav_manifest or (args.wav_dir / "wav_sha256.json")


def _selected_clips(args) -> list[dict]:
    """The clips an App-arm run grades: the whole population, a named list, or a seeded draw."""
    clips = app_arm_rows()
    if getattr(args, "clips", None) is not None:
        return selected_clips(clips, read_clips(args.clips))
    if getattr(args, "size", None) is not None:
        return draw_clips(clips, args.size, seed=args.seed)
    return clips


def _clips_sha256(clips) -> str:
    """A digest of the clip set a cache directory was filled from, order-independent."""
    import hashlib

    from training.nonpro_decode import item_key

    body = "\n".join(sorted(item_key(clip["audio_path"]) for clip in clips)) + "\n"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _run_identity(args, staged, driver, harness, clips) -> dict:
    """What a run cache directory belongs to: weights, engine, driver, audio and clips."""
    from tadabur.tlog_index import sha256_file
    from training.muraja_policy import CURRENT_COMMIT

    return {
        "version": 1,
        "latency_ms": args.latency_ms,
        "release": RELEASE,
        "model_sha256": staged.model_sha256,
        "muraja_commit": CURRENT_COMMIT,
        "harness_build": json.dumps(harness.build, sort_keys=True),
        "driver_sha256": sha256_file(Path(driver)),
        "wav_manifest_sha256": sha256_file(_wav_manifest_path(args)),
        "clips_sha256": _clips_sha256(clips),
    }


def _failed_items(path: Path) -> set[str]:
    """The items an App-arm run recorded a failure for (its per-clip table)."""
    from tadabur.followalong_audit.app_arm import read_jsonl

    return {row["item"] for row in read_jsonl(path) if row["failure"] is not None}


def _report_run(args, harness) -> dict:
    """The summary's run block: the identity every number in the doc hangs from."""
    from tadabur.followalong_audit.provenance import PROVENANCE_PATH
    from tadabur.tlog_index import sha256_file
    from training.aug_readout_gate import harness_identity
    from training.muraja_policy import CURRENT_COMMIT

    determinism = (json.loads(Path(args.determinism).read_text(encoding="utf-8"))
                   if Path(args.determinism).is_file() else None)
    probe = (json.loads(Path(args.payload_probe).read_text(encoding="utf-8"))
             if Path(args.payload_probe).is_file() else None)
    return {
        "muraja_commit": CURRENT_COMMIT,
        "harness": harness_identity(harness),
        "determinism": determinism,
        "payload_probe": probe,
        "release": RELEASE,
        "latency_ms": args.latency_ms,
        "seeds": {"p2": args.p2_seed, "p3": args.p3_seed},
        "driver_sha256": sha256_file(driver_path()),
        "audio_manifest_sha256": sha256_file(PROVENANCE_PATH),
        "word_grades_sha256": sha256_file(args.grades),
    }


def _app_cadence(traces_root: Path, latency_ms: int) -> list[dict]:
    """The App arm's previews per window, from the committed subset traces of each latency run.

    P3's two runs are the subsets drawn at :data:`P3_LATENCIES`; the P2 subset ran at the audit's
    own latency. A run whose traces are not present is left out rather than reported as zero.
    """
    from tadabur.followalong_audit.cadence import P3_LATENCIES, app_ratios

    runs = [(f"p3_{latency}", latency) for latency in P3_LATENCIES] + [("p2", latency_ms)]
    return [app_ratios(Path(traces_root) / name, latency_ms=latency)
            for name, latency in runs if (Path(traces_root) / name).is_dir()]


def _staged(args):
    """The verified release expanded for the driver, refusing a harness on other weights."""
    from training.muraja_policy import CURRENT_COMMIT, HARNESS_ROOT, Harness

    record = harness_record()
    if record.get("hifzguide_release") != RELEASE:
        raise SystemExit(f"the harness was built against {record.get('hifzguide_release')}, not "
                         f"{RELEASE}: the App arm would run weights the app does not ship")
    staged = stage_assets(Path(args.assets_dir),
                          json.loads(args.manifest.read_text(encoding="utf-8")),
                          args.staging_dir or DEFAULT_STAGING)
    driver = driver_path()
    harness = Harness.locate(commit=CURRENT_COMMIT)
    if harness is None:
        raise SystemExit(f"no Muraja harness for {CURRENT_COMMIT} under {HARNESS_ROOT}")
    return staged, driver, harness


def _draw(args) -> int:
    rows = gating_rows()
    eligible = oracle_eligible(rows, load_word_times())
    clips = draw_clips(eligible, args.size, seed=args.seed)
    write_clips(clips, args.out)
    write_json(draw_record_path(args.out),
               selection_record(rows, eligible, size=args.size, seed=args.seed))
    print(f"{len(clips)} clips from {len(eligible)} oracle-eligible gating rows -> {args.out}")
    return 0


def _run(args) -> int:
    if args.latency_ms < 0:
        raise SystemExit("--latency-ms must be >= 0")
    draw_record = args.draw or draw_record_path(args.clips)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = run_tracer(
        read_clips(args.clips), clip_list=args.clips, wav_dir=args.wav_dir,
        assets_dir=args.assets_dir, manifest=manifest,
        wav_manifest=json.loads(_wav_manifest_path(args).read_text(encoding="utf-8")),
        out=args.out, latency_ms=args.latency_ms, staging_dir=args.staging_dir,
        draw=json.loads(draw_record.read_text(encoding="utf-8"))
        if draw_record.is_file() else None)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["complete"] else 1


def _fetch(args) -> int:
    rows = app_arm_rows()
    manifest = fetch_clips(rows, args.wav_dir, workers=args.workers,
                           shard_cache=args.shard_cache)
    lines = provenance_rows(rows, manifest)
    write_provenance(lines, args.manifest_out)
    print(json.dumps({"rows": len(rows), "wavs": len(manifest),
                      "manifest": str(args.manifest_out)}, sort_keys=True))
    return 0


def _app_run(args) -> int:
    if args.latency_ms < 0:
        raise SystemExit("--latency-ms must be >= 0")
    from tadabur.followalong_audit.app_arm import (
        AppArm,
        clip_run_rows,
        grades_from_results,
        write_grades,
        write_jsonl,
    )
    from tadabur.followalong_audit.app_run import require_accounted, run_clips

    staged, driver, harness = _staged(args)
    clips = _selected_clips(args)
    arm = AppArm(driver=driver, assets_dir=staged.directory, wav_dir=args.wav_dir,
                 wav_manifest_path=_wav_manifest_path(args), quran_db=harness.quran_db,
                 latency_ms=args.latency_ms, traces_dir=args.traces_dir)
    results = run_clips(clips, arm, args.out, workers=args.workers,
                        identity=_run_identity(args, staged, driver, harness, clips))
    require_accounted(clips, results)
    rows = grades_from_results(results)
    grades_out = args.grades_out or Path(args.out) / f"word_grades.{args.latency_ms}ms.jsonl.gz"
    write_grades(rows, grades_out)
    clip_runs_out = args.clip_runs_out or Path(args.out) / "clip_runs.jsonl"
    write_jsonl(clip_run_rows(clips, results), clip_runs_out)
    failed = sum(1 for result in results if result["failure"] is not None)
    print(json.dumps({"clips": len(clips), "size": args.size, "seed": args.seed,
                      "failed": failed, "words": len(rows),
                      "grades": str(grades_out), "clip_runs": str(clip_runs_out)},
                     sort_keys=True))
    return 0 if not failed else 1


def _determinism(args) -> int:
    from tadabur.followalong_audit.app_arm import AppArm, determinism_report

    staged, driver, harness = _staged(args)
    report = determinism_report(
        app_arm_rows(),
        AppArm(driver=driver, assets_dir=staged.directory, wav_dir=args.wav_dir,
               wav_manifest_path=_wav_manifest_path(args), quran_db=harness.quran_db,
               latency_ms=args.latency_ms),
        args.out_root, size=args.size, seed=args.seed, workers=args.workers)
    write_json(Path(args.out_root) / "determinism.json", report)
    print(json.dumps({key: report[key]
                      for key in ("seed", "size", "identical", "differences", "failed")},
                     sort_keys=True))
    return 0 if report["identical"] else 1


def _determinism_probe(args) -> int:
    from tadabur.followalong_audit.app_arm import AppArm, payload_variation
    from training.nonpro_decode import item_key

    population = app_arm_rows()
    clip = next((row for row in population if item_key(row["audio_path"]) == args.item), None)
    if clip is None:
        raise SystemExit(
            f"{args.item} is not one of the population's {len(population)} claim-gated"
            " confirm.1/confirm.2 clips: the probe repeats a clip the audit grades, it does not"
            " add one")
    staged, driver, harness = _staged(args)
    probe = payload_variation(
        clip, AppArm(driver=driver, assets_dir=staged.directory, wav_dir=args.wav_dir,
                     wav_manifest_path=_wav_manifest_path(args), quran_db=harness.quran_db,
                     latency_ms=args.latency_ms),
        args.out_root, runs=args.runs)
    write_json(Path(args.out), probe)
    print(json.dumps({"item": probe["item"], "runs": probe["runs"],
                      "payloads": probe["payloads"],
                      "changed_words": len(probe["changed_words"]),
                      "failed_runs": probe["failed_runs"]}, sort_keys=True))
    return 0 if not probe["failed_runs"] else 1


def _strata(args) -> int:
    from tadabur.followalong_audit.app_arm import read_grades, read_jsonl
    from tadabur.followalong_audit.strata_run import (
        cells_record,
        pools_from_run,
        stratified_run,
        write_cells,
        write_pools,
    )

    oracle = read_grades(args.oracle) if args.oracle is not None else None
    stratified, meta, marks = stratified_run(app_arm_rows(), read_grades(args.grades),
                                             oracle=oracle,
                                             failed=_failed_items(args.clip_runs))
    pools = pools_from_run(stratified, meta, marks)
    write_pools(pools, args.pools_out)
    write_cells(cells_record(stratified, meta["detector_ungraded"]), args.cells_out)
    print(json.dumps({"clips": pools["clips"], "words": pools["words"],
                      "pools": pools["pools"], "catch_loss": pools["catch_loss"]},
                     sort_keys=True))
    return 0


def _report(args) -> int:
    from tadabur.followalong_audit.app_arm import read_grades, read_jsonl
    from tadabur.followalong_audit.app_doc import write_doc, write_summary
    from tadabur.followalong_audit.audit_report import build_summary
    from tadabur.followalong_audit.bridge import p2_section
    from tadabur.followalong_audit.cadence import h448_ratios, owner_ratios, p3_section
    from training.device_exports import load_annotations
    from training.muraja_policy import CURRENT_COMMIT, MurajaText, require_harness

    population = app_arm_rows()
    clip_runs = read_jsonl(args.clip_runs)
    failed = [{"item": row["item"], "reason": row["failure"]}
              for row in clip_runs if row["failure"] is not None]
    failed_items = {row["item"] for row in clip_runs if row["failure"] is not None}
    grades = read_grades(args.grades)
    harness = require_harness(commit=CURRENT_COMMIT)
    oracle = (json.loads(args.oracle.read_text(encoding="utf-8"))
              if args.oracle is not None else None)
    reused = (json.loads(args.reuse_p2_p3.read_text(encoding="utf-8"))
              if args.reuse_p2_p3 is not None else None)
    if reused is not None:
        p2, p3, run = reused["p2"], reused["p3"], reused.get("run")
    else:
        p2 = p2_section(population, grades, harness=harness, subset_size=args.p2_clips,
                        seed=args.p2_seed, all_clips=not args.no_all_clips,
                        failed=failed_items)
        zero, other = read_grades(args.p3_zero), read_grades(args.p3_130)
        notes = load_annotations(MurajaText(harness.quran_db))
        exports = [Path(args.exports_dir) / name for name in sorted(notes.exports)]
        ratios = {**h448_ratios(), **owner_ratios(exports, notes)}
        p3 = p3_section(zero, other, device_ratios=ratios, seed=args.p3_seed,
                        app_cadence=_app_cadence(args.traces_root, args.latency_ms))
        run = _report_run(args, harness)
    pools = json.loads(args.pools.read_text(encoding="utf-8"))
    summary = build_summary(grades, clips=len(population), failed=failed, p2=p2, p3=p3,
                            pools=pools, oracle=oracle, run=run)
    write_summary(summary, args.summary_out)
    write_doc(json.loads(args.summary_out.read_text(encoding="utf-8")), args.doc_out)
    print(json.dumps({"clips": len(population), "failed": len(failed),
                      "p2": p2["verdict"], "doc": str(args.doc_out)}, sort_keys=True))
    return 0


def _oracle_sweep(args) -> int:
    from tadabur.followalong_audit.app_arm import read_grades, write_grades
    from tadabur.followalong_audit.sweep import oracle_summary, run_sweep
    from tadabur.followalong_audit.sweep_doc import render
    from tadabur.followalong_audit.atomic import write_text_atomically
    from training.muraja_policy import CURRENT_COMMIT, require_harness

    harness = require_harness(commit=CURRENT_COMMIT)
    result = run_sweep(harness, seed=args.seed,
                       log=lambda message: print(message, file=sys.stderr, flush=True))
    write_json(Path(args.report_out), result.report)
    write_text_atomically(Path(args.doc_out), render(result.report))
    write_grades(result.word_rows, args.word_grades_out)
    write_grades(result.clean_findings, args.clean_findings_out)
    write_grades(result.planted_rows, args.planted_rows_out)
    summary = oracle_summary(result.word_rows, read_grades(_app_arm("word_grades.jsonl.gz")))
    write_json(Path(args.summary_out), summary)
    write_text_atomically(Path(args.run_note_out), result.run_note)
    print(json.dumps({
        "seed": args.seed,
        "tlog_clips": result.report["clean"]["tlog"]["clips"],
        "mushaf_ayat": result.report["clean"]["mushaf"]["ayat"],
        "patterns": len(result.report["patterns"]),
        "candidates": sum(1 for p in result.report["patterns"] if p["candidate"]),
        "clean_findings": len(result.clean_findings),
        "planted_rows": len(result.planted_rows),
        "oracle_share": summary["share"],
        "doc": str(args.doc_out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
