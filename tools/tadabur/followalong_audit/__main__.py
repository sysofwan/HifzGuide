"""The S1 tracer's CLI (#307).

    # the seeded draw (claim-gated phase A rows of confirm.1 and confirm.2)
    python -m tadabur.followalong_audit draw --size 20 --seed 7 \
        --out tadabur/followalong_audit/tracer/clips.jsonl

    # both arms over the fetched WAVs, then gate P1b
    python -m tadabur.followalong_audit run --clips tadabur/followalong_audit/tracer/clips.jsonl \
        --wav-dir ~/.cache/hifzguide/followalong-audit/wav \
        --assets-dir <release v1.0.7 unpacked> --manifest <manifest.json> \
        --out tadabur/followalong_audit/tracer --latency-ms 40

The WAVs are named ``<audio_sha256>.wav`` and live outside the repo; the run writes only
decode text and provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tadabur.followalong_audit.clips import (
    draw_clips,
    gating_rows,
    oracle_eligible,
    read_clips,
    write_clips,
)
from tadabur.followalong_audit.tracer import DEVICE_LATENCY_MS, load_word_times, run_tracer


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
    run.add_argument("--wav-dir", type=Path, required=True)
    run.add_argument("--assets-dir", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--latency-ms", type=int, default=DEVICE_LATENCY_MS)

    args = parser.parse_args()
    if args.command == "draw":
        rows = oracle_eligible(gating_rows(), load_word_times())
        clips = draw_clips(rows, args.size, seed=args.seed)
        write_clips(clips, args.out)
        print(f"{len(clips)} clips from {len(rows)} oracle-eligible gating rows -> {args.out}")
        return 0

    if args.latency_ms < 0:
        raise SystemExit("--latency-ms must be >= 0")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = run_tracer(
        read_clips(args.clips), clip_list=args.clips, wav_dir=args.wav_dir,
        assets_dir=args.assets_dir, manifest=manifest, out=args.out,
        latency_ms=args.latency_ms)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
