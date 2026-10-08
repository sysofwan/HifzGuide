"""Diagnose the haraka gap: whole waqf segments vs 5 s windows vs the stream (#85).

The base teacher matches 0.9617 of reference harakat on whole waqf segments (ADR-0005) but
0.841 on ADR-0007's <=5 s word-snapped windows. This decodes the same mining-pool audio as
whole segments, as those windows, and streamed at commit blocks b=0 and b=1, with the base
teacher and ``h448``, and reports per haraka where the difference comes from. The arms,
sites and classes are defined in :mod:`training.haraka_gap_arms`; the statistics in
:mod:`training.haraka_gap_report`. Every number is **agreement with the mushaf**, not truth.

Two steps, each idempotent. ``decode`` needs the GPU and the staged pool audio (verified
against the staged-clip registry); ``report`` is CPU-only and reads the decode caches::

  cd tools
  flock /root/scratch/gpu.lock python -m training.haraka_gap decode \
      --audio-dir /root/scratch/issue-83/stage/clips --cache-dir CACHE \
      --model base=obadx/muaalem-model-v3_2 \
      --model h448=/root/repos/HifzGuide/tools/runs/h448_stream/checkpoint.pt \
  && python -m training.haraka_gap report --cache-dir CACHE \
      --out-json ../docs/haraka-gap.json --out-md ../docs/haraka-gap-tables.md

Every model is decoded at one precision and one batch size (:data:`WEIGHTS_DTYPE`,
:data:`DECODE_BATCH_SIZE`), so the fingerprints of two models differ only in the model,
which ``report`` checks.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from tadabur.mining_pool import load_base_decodes, segment_key
from training.haraka_gap_arms import (
    BLOCKS,
    SEAM_SECONDS,
    SEGMENT_ALL,
    STREAM,
    ClipPlan,
    Observation,
    build_arms,
    haraka_sites,
    model_observations,
    plan_pool,
    score_model,
    site_times,
    stream_unit_key,
    window_key,
    windowed_population,
)
from training.haraka_gap_report import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    Ledger,
    build_report,
    render_markdown,
)
from training.tashkeel_eval import write_text_atomically
from training.windowing import TARGET_SAMPLE_RATE

#: One precision and one batch size for every arm and both models: a span-vs-stream
#: difference is then windowing, and a difference between models is weights.
WEIGHTS_DTYPE = "bf16"
DECODE_BATCH_SIZE = 1

# --- Decoding (Linux + CUDA) -----------------------------------------------------------


def cache_path(cache_dir: Path, model: str) -> Path:
    return cache_dir / f"{model}.json"


def decode_model(model: str, model_ref: str, plans: list[ClipPlan], audio_dir: Path,
                 cache_dir: Path, device: str) -> None:
    """Decode every unit of every arm with one model and write its cache.

    Each clip's WAV is verified against the staged-clip registry first. Segments are
    decoded whole, one per forward pass, keeping each token's (doubled) midpoint step for
    :func:`site_times`; windows are decoded whole; the stream's per-window argmax rows are
    computed once and committed at every block in :data:`BLOCKS`
    (:func:`training.decoding.stream_emissions`), and kept in an ``.npz`` beside the cache.
    """
    import soundfile as sf

    from tadabur.staged_audio import load_staged_clips, verify_staged
    from training.decoding import (
        SPANS,
        Decoder,
        scan_ctc,
        stream_emissions,
        stream_protocol,
        tokens_to_phonemes,
        window_audio,
    )

    registry = load_staged_clips()
    decoder = Decoder.load(model_ref, device, weights_dtype=WEIGHTS_DTYPE,
                           batch_size=DECODE_BATCH_SIZE)
    segments: dict[str, dict] = {}
    windows: dict[str, str] = {}
    streams: dict[str, str] = {}
    rows_by_clip: dict[str, np.ndarray] = {}
    decoded = [plan for plan in plans if plan.segments]
    for done, plan in enumerate(decoded, 1):
        name = plan.clip.audio_filename
        verify_staged(registry[name], audio_dir)
        waveform, rate_hz = sf.read(audio_dir / name, dtype="float32")
        if rate_hz != TARGET_SAMPLE_RATE:
            raise ValueError(f"{name} is {rate_hz} Hz, expected {TARGET_SAMPLE_RATE}")
        rows = decoder.span_class_ids(
            waveform[seg.start_sample:seg.end_sample] for seg in plan.segments)
        for seg, row in zip(plan.segments, rows):
            runs = scan_ctc(row)
            segments[segment_key(name, seg.segment_index)] = {
                "decode": tokens_to_phonemes(run.token_id for run in runs),
                "mid2": [run.start_step + run.end_step for run in runs],
            }
        if plan.eligible:
            texts = decoder.decode_spans(
                waveform[w.start_sample:w.start_sample + w.num_samples] for w in plan.windows)
            for window, text in zip(plan.windows, texts):
                windows[window_key(name, window.window_index)] = text
            start = plan.recitation_start_sample
            window_rows = decoder.window_rows(
                window_audio(waveform[start:start + plan.recitation_num_samples]))
            rows_by_clip[name] = np.stack(window_rows).astype(np.uint8)
            for block in BLOCKS:
                streams[stream_unit_key(name, block)] = tokens_to_phonemes(
                    e.token_id for e in stream_emissions(window_rows, block))
        if done % 100 == 0 or done == len(decoded):
            print(f"[{model}] {done}/{len(decoded)} clips", flush=True)
    fingerprints = {
        "spans": decoder.fingerprint(SPANS).as_dict(),
        **{STREAM[b]: decoder.fingerprint(stream_protocol(b)).as_dict() for b in BLOCKS},
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"{model}_stream_rows.npz", **{
        f"clip{i:05d}": rows for i, (_, rows) in enumerate(sorted(rows_by_clip.items()))})
    write_text_atomically(cache_path(cache_dir, model), json.dumps({
        "model": model,
        "model_ref": str(model_ref),
        "fingerprints": fingerprints,
        "stream_rows_clips": sorted(rows_by_clip),
        "segments": segments,
        "windows": windows,
        "streams": streams,
    }, ensure_ascii=False, sort_keys=True))


def load_cache(cache_dir: Path, model: str) -> dict:
    return json.loads(cache_path(cache_dir, model).read_text(encoding="utf-8"))


def check_fingerprints(caches: Mapping[str, dict]) -> None:
    """Refuse models decoded under different settings (only the model may differ)."""
    from training.decoding import DecodeFingerprint

    models = sorted(caches)
    for mode in caches[models[0]]["fingerprints"]:
        reference = DecodeFingerprint.from_dict(caches[models[0]]["fingerprints"][mode], models[0])
        for other in models[1:]:
            reference.check_comparable(
                DecodeFingerprint.from_dict(caches[other]["fingerprints"].get(mode), other),
                f"{models[0]} and {other} ({mode})",
            )


# --- CLI --------------------------------------------------------------------------------


def _decode(args) -> None:
    plans = plan_pool()
    for spec in args.model:
        model, _, model_ref = spec.partition("=")
        if not model_ref:
            raise SystemExit(f"--model takes NAME=REF, got {spec!r}")
        decode_model(model, model_ref, plans, args.audio_dir, args.cache_dir, args.device)


def _report(args) -> None:
    plans = plan_pool()
    models = args.models
    caches = {model: load_cache(args.cache_dir, model) for model in models}
    check_fingerprints(caches)
    base = caches[args.timing_model]
    population = windowed_population(plans)
    times, time_sources = site_times(plans, population, base["segments"])
    arms = build_arms(plans, times)

    observations: list[Observation] = []
    for model in models:
        cache = caches[model]
        decodes = {
            **{key: value["decode"] for key, value in cache["segments"].items()},
            **cache["windows"],
            **cache["streams"],
        }
        print(f"scoring {model}", flush=True)
        observations += model_observations(model, arms, score_model(arms, decodes, args.workers))

    _, cached_base = load_base_decodes()
    same = sum(base["segments"][key]["decode"] == text for key, text in cached_base.items())
    eligible = [plan for plan in plans if plan.eligible]
    meta = {
        "pool_clips": len(plans),
        "eligible_clips": len(eligible),
        "reciters": len({plan.clip.reciter_id for plan in eligible}),
        "exclusions": dict(sorted(Counter(p.exclusion for p in plans if p.exclusion).items())),
        "windows": sum(len(plan.windows) for plan in eligible),
        "segments_all": len(arms.units[SEGMENT_ALL]),
        "segments_eligible": sum(len(plan.segments) for plan in eligible),
        "stream_seconds": round(sum(p.recitation_num_samples for p in eligible)
                                / TARGET_SAMPLE_RATE, 1),
        "population_sites": len(population),
        "eligible_sites_in_no_window": sum(
            len(haraka_sites(seg)) for plan in eligible for seg in plan.segments
        ) - len(population),
        "site_time_sources": dict(sorted(time_sources.items())),
        "timing_model": args.timing_model,
        "base_segments_identical_to_pool_cache": [same, len(cached_base)],
        "fingerprints": {model: caches[model]["fingerprints"] for model in models},
        "model_refs": {model: caches[model]["model_ref"] for model in models},
        "weights_dtype": WEIGHTS_DTYPE,
        "decode_batch_size": DECODE_BATCH_SIZE,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "cluster": "reciter"},
        "seam_seconds": SEAM_SECONDS,
    }
    print("bootstrapping", flush=True)
    report = build_report(Ledger(observations), models, meta, gap_model=args.timing_model)
    write_text_atomically(args.out_json, json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    write_text_atomically(args.out_md, render_markdown(report))
    print(f"wrote {args.out_json} and {args.out_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    decode = sub.add_parser("decode", help="decode every arm with each model (GPU)")
    decode.add_argument("--audio-dir", type=Path, required=True,
                        help="staged pool clips (tadabur.staged_audio), verified by checksum")
    decode.add_argument("--cache-dir", type=Path, required=True)
    decode.add_argument("--model", action="append", required=True,
                        help="NAME=REF: a hub id or a distillation checkpoint (repeatable)")
    decode.add_argument("--device", default="cuda")
    decode.set_defaults(run=_decode)
    report = sub.add_parser("report", help="score the cached decodes and write the report")
    report.add_argument("--cache-dir", type=Path, required=True)
    report.add_argument("--models", nargs="+", default=["base", "h448"])
    report.add_argument("--timing-model", default="base",
                        help="the model whose whole-segment decode times the stream sites, "
                             "and whose numbers the gap arithmetic explains")
    report.add_argument("--workers", type=int, default=8)
    report.add_argument("--out-json", type=Path, required=True)
    report.add_argument("--out-md", type=Path, required=True)
    report.set_defaults(run=_report)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
