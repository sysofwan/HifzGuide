"""Diagnose the haraka gap: whole waqf segments vs 5 s windows vs the stream (#85).

The base teacher matches 0.9617 of reference harakat on whole waqf segments (ADR-0005) but
0.841 on ADR-0007's <=5 s word-snapped windows. This decodes the same mining-pool audio as
whole segments, as those windows, and streamed at commit blocks b=0 and b=1, with the base
teacher and ``h448``, and reports per haraka where the difference comes from. The arms,
sites and classes are defined in :mod:`training.haraka_gap_arms`; the statistics in
:mod:`training.haraka_gap_report`. Every number is **agreement with the mushaf**, not truth.

Two steps. ``decode`` needs the GPU and the staged pool audio (verified against the
staged-clip registry) and writes each model's cache plus ``provenance.json`` (cache
checksums, the checkpoint's sha256 or the hub commit, and the source manifests' hashes).
The caches the committed report was built from are tracked in ``haraka_gap_cache/``, so
``report`` is CPU-only and runs anywhere; it refuses caches whose checksums or source
manifests no longer match::

  cd tools
  python -m training.haraka_gap report \
      --out-json ../docs/haraka-gap.json --out-md ../docs/haraka-gap-tables.md

To decode afresh (GPU box)::

  flock /root/scratch/gpu.lock python -m training.haraka_gap decode \
      --audio-dir /root/scratch/issue-83/stage/clips --cache-dir training/haraka_gap_cache \
      --model base=obadx/muaalem-model-v3_2 \
      --model h448=/root/repos/HifzGuide/tools/runs/h448_stream/checkpoint.pt

Every model is decoded at one precision and one batch size (:data:`WEIGHTS_DTYPE`,
:data:`DECODE_BATCH_SIZE`); ``report`` checks that every arm of every model fingerprints
those settings and states them from the fingerprints (:func:`validated_settings`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from tadabur.mining_pool import load_base_decodes, segment_key
from training.haraka_gap_arms import (
    BLOCKS,
    SEAM_STEPS,
    SEGMENT_ALL,
    STREAM,
    ClipPlan,
    Observation,
    build_arms,
    haraka_sites,
    model_observations,
    plan_pool,
    score_model,
    site_positions,
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
    :func:`~training.haraka_gap_arms.site_positions`; windows are decoded whole; the stream's per-window argmax rows are
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


def expected_modes() -> dict[str, str]:
    """Each cached arm's fingerprint key and the decode mode it must name."""
    from training.decoding import SPANS, stream_protocol

    return {"spans": SPANS, **{STREAM[b]: stream_protocol(b) for b in BLOCKS}}


def validated_settings(caches: Mapping[str, dict]) -> dict:
    """The decode settings every arm of every model shares, or an error.

    Each model must fingerprint exactly the expected arms, each naming its own protocol and
    the model's own reference; within a model only the mode may differ between arms, and
    between models only the model. The returned settings (precision, batch size, device,
    autocast, inference policy) are what the report states, so it cannot advertise
    settings the decodes were not made under.
    """
    from dataclasses import replace

    from training.decoding import DecodeFingerprint

    modes = expected_modes()
    shared: dict | None = None
    for model, cache in sorted(caches.items()):
        stored = cache.get("fingerprints") or {}
        if set(stored) != set(modes):
            raise ValueError(f"{model}: fingerprints for {sorted(stored)}, expected {sorted(modes)}")
        parsed = {key: DecodeFingerprint.from_dict(stored[key], f"{model} {key}") for key in modes}
        for key, fingerprint in parsed.items():
            if fingerprint.mode != modes[key]:
                raise ValueError(f"{model} {key}: mode {fingerprint.mode!r}, expected {modes[key]!r}")
            if fingerprint.model != cache["model_ref"]:
                raise ValueError(f"{model} {key}: decoded {fingerprint.model!r}, "
                                 f"not {cache['model_ref']!r}")
        spans = parsed["spans"]
        for key, fingerprint in parsed.items():
            spans.check_comparable(replace(fingerprint, mode=spans.mode), f"{model}: spans and {key}")
        settings = {k: v for k, v in spans.as_dict().items() if k not in ("model", "mode")}
        if shared is not None and settings != shared:
            raise ValueError(f"{model} was decoded under {settings}, the others under {shared}")
        shared = settings
    if shared is None:
        raise ValueError("no model caches to compare")
    return shared


# --- Provenance of the committed caches ---------------------------------------------------

#: The tracked home of the decode caches the committed report was built from.
CACHE_DIR = Path(__file__).parent / "haraka_gap_cache"
PROVENANCE = "provenance.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifests() -> dict[str, Path]:
    """The committed inputs a cache is only valid against."""
    from tadabur.mining_pool import CLIPS_PATH, DECODES_PATH
    from tadabur.staged_audio import REGISTRY_PATH

    return {"mining_pool/clips.jsonl": CLIPS_PATH, "mining_pool/base_decodes.json": DECODES_PATH,
            "staged_audio/clips.jsonl": REGISTRY_PATH}


def model_identity(model_ref: str) -> dict:
    """An immutable name for the weights: a checkpoint file's sha256, or the hub commit of
    the locally cached snapshot ``from_pretrained`` loaded."""
    path = Path(model_ref)
    if path.is_file():
        return {"checkpoint_sha256": _sha256(path)}
    from huggingface_hub import snapshot_download

    return {"hub_revision": Path(snapshot_download(model_ref, local_files_only=True)).name}


def cache_files(models: list[str]) -> list[str]:
    return [name for model in models for name in (f"{model}.json", f"{model}_stream_rows.npz")]


def write_provenance(cache_dir: Path, models: Mapping[str, str]) -> None:
    """Record each cache file's checksum, each model's immutable identity and the source
    manifests the decodes were made against."""
    record = {
        "files": {name: _sha256(cache_dir / name) for name in cache_files(sorted(models))},
        "models": {model: {"ref": ref, **model_identity(ref)} for model, ref in sorted(models.items())},
        "sources": {name: _sha256(path) for name, path in source_manifests().items()},
    }
    write_text_atomically(cache_dir / PROVENANCE, json.dumps(record, indent=1, sort_keys=True) + "\n")


def verified_provenance(cache_dir: Path, models: list[str]) -> dict:
    """The cache directory's provenance, after checking every cache file and source
    manifest against it: a cache decoded from another pool or edited since is refused."""
    record = json.loads((cache_dir / PROVENANCE).read_text(encoding="utf-8"))
    for name in cache_files(models):
        if _sha256(cache_dir / name) != record["files"].get(name):
            raise ValueError(f"{cache_dir / name} does not match its recorded checksum")
    for name, path in source_manifests().items():
        if _sha256(path) != record["sources"][name]:
            raise ValueError(f"{path} changed since the caches were decoded against it")
    return record


# --- CLI --------------------------------------------------------------------------------


def _model_refs(specs: list[str]) -> dict[str, str]:
    refs = {}
    for spec in specs:
        model, _, model_ref = spec.partition("=")
        if not model_ref:
            raise SystemExit(f"--model takes NAME=REF, got {spec!r}")
        refs[model] = model_ref
    return refs


def _decode(args) -> None:
    plans = plan_pool()
    refs = _model_refs(args.model)
    for model, model_ref in refs.items():
        decode_model(model, model_ref, plans, args.audio_dir, args.cache_dir, args.device)
    write_provenance(args.cache_dir, refs)


def _provenance(args) -> None:
    write_provenance(args.cache_dir, _model_refs(args.model))


def _report(args) -> None:
    plans = plan_pool()
    models = args.models
    provenance = verified_provenance(args.cache_dir, models)
    caches = {model: load_cache(args.cache_dir, model) for model in models}
    settings = validated_settings(caches)
    base = caches[args.timing_model]
    population = windowed_population(plans)
    positions, position_sources = site_positions(plans, population, base["segments"])
    arms = build_arms(plans, positions)

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
        "site_position_sources": dict(sorted(position_sources.items())),
        "timing_model": args.timing_model,
        "base_segments_identical_to_pool_cache": [same, len(cached_base)],
        "fingerprints": {model: caches[model]["fingerprints"] for model in models},
        "models": {model: provenance["models"][model] for model in models},
        "cache_files": {name: provenance["files"][name] for name in cache_files(models)},
        "source_manifests": provenance["sources"],
        "decode_settings": settings,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "cluster": "reciter"},
        "seam_steps": SEAM_STEPS,
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
    provenance = sub.add_parser(
        "provenance", help="(re)write provenance.json for caches already decoded (GPU box)")
    provenance.add_argument("--cache-dir", type=Path, required=True)
    provenance.add_argument("--model", action="append", required=True, help="NAME=REF")
    provenance.set_defaults(run=_provenance)
    report = sub.add_parser("report", help="score the cached decodes and write the report")
    report.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                        help=f"decode caches with their provenance (default {CACHE_DIR.name}/)")
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
