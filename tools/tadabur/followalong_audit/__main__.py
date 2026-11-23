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
    python -m tadabur.followalong_audit app-run --out <scratch>/app0 --latency-ms 0 ... \
        --grades-out tadabur/followalong_audit/app_arm/word_grades.0ms.jsonl.gz

    # determinism on the seeded subset, the detectors' strata, then the summary and doc
    python -m tadabur.followalong_audit determinism --out-root <scratch>/determinism ...
    python -m tadabur.followalong_audit strata
    python -m tadabur.followalong_audit report

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

    strata = commands.add_parser("strata", help="re-grade the detectors and size the pools")
    strata.add_argument("--grades", type=Path, default=_app_arm("word_grades.jsonl.gz"))
    strata.add_argument("--pools-out", type=Path, default=_app_arm("pools.json"))

    report = commands.add_parser("report", help="write the summary and the generated doc")
    report.add_argument("--grades", type=Path, default=_app_arm("word_grades.jsonl.gz"))
    report.add_argument("--clip-runs", type=Path, default=_app_arm("clip_runs.jsonl"))
    report.add_argument("--pools", type=Path, default=_app_arm("pools.json"))
    report.add_argument("--p3-zero", type=Path, default=_app_arm("word_grades.0ms.jsonl.gz"))
    report.add_argument("--p3-130", type=Path, default=_app_arm("word_grades.130ms.jsonl.gz"))
    report.add_argument("--exports-dir", type=Path, default=Path.home() / "Downloads")
    report.add_argument("--oracle", type=Path, help="S2's oracle summary, when #308 has run")
    report.add_argument("--p2-clips", type=int, default=200)
    report.add_argument("--p2-seed", type=int, default=319)
    report.add_argument("--no-all-clips", action="store_true",
                        help="skip the whole-population P2 agreement (default: report it)")
    report.add_argument("--summary-out", type=Path, default=_app_arm("summary.json"))
    report.add_argument("--doc-out", type=Path, default=_doc_path())

    args = parser.parse_args()
    if args.command == "draw":
        return _draw(args)
    if args.command == "fetch":
        return _fetch(args)
    if args.command == "app-run":
        return _app_run(args)
    if args.command == "determinism":
        return _determinism(args)
    if args.command == "strata":
        return _strata(args)
    if args.command == "report":
        return _report(args)
    return _run(args)


def _assets_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--wav-dir", type=Path, required=True)
    command.add_argument("--wav-manifest", type=Path,
                         help="the fetch's WAV digest manifest (default: <wav-dir>/wav_sha256.json)")
    command.add_argument("--assets-dir", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--staging-dir", type=Path,
                         help="where to expand the verified release (default: ~/.cache)")


def _app_arm(name: str) -> Path:
    from tadabur.followalong_audit.app_doc import APP_ARM_DIR

    return APP_ARM_DIR / name


def _doc_path() -> Path:
    from tadabur.followalong_audit.app_doc import DOC_PATH

    return DOC_PATH


def _wav_manifest_path(args) -> Path:
    return args.wav_manifest or (args.wav_dir / "wav_sha256.json")


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
    clips = app_arm_rows()
    arm = AppArm(driver=driver, assets_dir=staged.directory, wav_dir=args.wav_dir,
                 wav_manifest_path=_wav_manifest_path(args), quran_db=harness.quran_db,
                 latency_ms=args.latency_ms, traces_dir=args.traces_dir)
    results = run_clips(clips, arm, args.out, workers=args.workers)
    require_accounted(clips, results)
    rows = grades_from_results(results)
    grades_out = args.grades_out or Path(args.out) / f"word_grades.{args.latency_ms}ms.jsonl.gz"
    write_grades(rows, grades_out)
    clip_runs_out = args.clip_runs_out or Path(args.out) / "clip_runs.jsonl"
    write_jsonl(clip_run_rows(clips, results), clip_runs_out)
    failed = sum(1 for result in results if result["failure"] is not None)
    print(json.dumps({"clips": len(clips), "failed": failed, "words": len(rows),
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
    print(json.dumps({key: report[key] for key in ("seed", "size", "identical", "differences")},
                     sort_keys=True))
    return 0 if report["identical"] else 1


def _strata(args) -> int:
    from tadabur.followalong_audit.app_arm import read_grades
    from tadabur.followalong_audit.strata_run import candidate_pools, write_pools

    pools = candidate_pools(app_arm_rows(), read_grades(args.grades))
    write_pools(pools, args.pools_out)
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
    failed = [{"item": row["item"], "reason": row["failure"]}
              for row in read_jsonl(args.clip_runs) if row["failure"] is not None]
    grades = read_grades(args.grades)
    harness = require_harness(commit=CURRENT_COMMIT)
    oracle = (json.loads(args.oracle.read_text(encoding="utf-8"))
              if args.oracle is not None else None)
    p2 = p2_section(population, grades, harness=harness, subset_size=args.p2_clips,
                    seed=args.p2_seed, all_clips=not args.no_all_clips)
    zero, other = read_grades(args.p3_zero), read_grades(args.p3_130)
    notes = load_annotations(MurajaText(harness.quran_db))
    exports = [Path(args.exports_dir) / name for name in sorted(notes.exports)]
    ratios = {**h448_ratios(), **owner_ratios(exports, notes)}
    p3 = p3_section(zero, other, device_ratios=ratios,
                    clips=len({row["item"] for row in zero}))
    summary = build_summary(grades, clips=len(population), failed=failed, p2=p2, p3=p3,
                            pools=json.loads(args.pools.read_text(encoding="utf-8")),
                            oracle=oracle)
    write_summary(summary, args.summary_out)
    write_doc(json.loads(args.summary_out.read_text(encoding="utf-8")), args.doc_out)
    print(json.dumps({"clips": len(population), "failed": len(failed),
                      "p2": p2["verdict"], "doc": str(args.doc_out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
