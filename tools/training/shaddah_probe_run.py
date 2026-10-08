"""Run the shaddah probe (#86) on the mining pool, for the base teacher and ``h448``.

Four stages, each cached in ``--work-dir`` so a re-run only does what is missing:

1. ``posteriors`` (GPU) -- every kept pool segment decoded whole by each model through
   :meth:`training.decoding.Decoder.span_log_posteriors`, bf16 weights, batch size 1: the
   fingerprint of the base teacher's committed decodes (``mining_pool/base_decodes.json``),
   whose strings the base pass must reproduce exactly.
2. ``measure`` (CPU) -- every consonant site of every segment
   (:func:`training.shaddah_probe.measure_segment`), as weighted observations.
3. ``stretch`` (GPU) -- each collapsed geminate's interval, and that of its matched single
   control, stretched by every factor (:func:`tadabur.time_stretch.stretch_span`) and paired
   with an equal-length decoy, then re-decoded and re-measured.
4. ``report`` -- :func:`training.shaddah_probe.model_report` per model, into one JSON.

Every staged clip's checksum is verified against the registry before its audio is read.

Usage (from ``tools/`` on the GPU box; the whole probe is one command)::

  flock /root/scratch/gpu.lock python -m training.shaddah_probe_run \\
      --audio-dir /root/scratch/issue-83/stage/clips \\
      --h448 /root/repos/HifzGuide/tools/runs/h448_stream/checkpoint.pt \\
      --work-dir /root/scratch/issue-86/work --out ../docs/shaddah-probe.json
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from training import shaddah_probe as sp

BASE = "base"
H448 = "h448"
WEIGHTS_DTYPE = "bf16"
BATCH_SIZE = 1


@dataclass(frozen=True)
class PoolSegmentRef:
    """One kept pool segment: where its audio is and what it should say."""

    key: str
    audio_filename: str
    start_sample: int
    end_sample: int
    reference: str
    reciter_id: int
    inclusion_probability: float


def pool_segments() -> list[PoolSegmentRef]:
    from tadabur.mining_pool import load_manifest, segment_key

    return [
        PoolSegmentRef(
            key=segment_key(clip.audio_filename, seg.segment_index),
            audio_filename=clip.audio_filename,
            start_sample=seg.start_sample,
            end_sample=seg.end_sample,
            reference=seg.reference,
            reciter_id=clip.reciter_id,
            inclusion_probability=clip.inclusion_probability,
        )
        for clip in load_manifest()
        for seg in clip.segments
        if seg.kept
    ]


def read_segments(segments: list[PoolSegmentRef], audio_dir: Path):
    """Yield ``(segment, samples)`` clip by clip, each clip checksum-verified first."""
    import soundfile as sf

    from tadabur.staged_audio import load_staged_clips, verify_staged

    registry = load_staged_clips()
    current, waveform = None, None
    for segment in segments:
        if segment.audio_filename != current:
            verify_staged(registry[segment.audio_filename], audio_dir)
            waveform, rate = sf.read(audio_dir / segment.audio_filename, dtype="float32")
            if rate != 16000 or waveform.ndim != 1:
                raise ValueError(f"{segment.audio_filename} is not 16 kHz mono")
            current = segment.audio_filename
        yield segment, np.ascontiguousarray(waveform[segment.start_sample : segment.end_sample])


def load_decoder(model: str, h448: Path):
    from training.decoding import Decoder
    from training.distill_student import TEACHER_MODEL_ID

    ref = TEACHER_MODEL_ID if model == BASE else str(h448)
    return Decoder.load(ref, "cuda", weights_dtype=WEIGHTS_DTYPE, batch_size=BATCH_SIZE)


# --- Stage 1: posteriors --------------------------------------------------------------


def posteriors_stage(model, segments, audio_dir, h448, work_dir) -> None:
    from training.decoding import SPANS

    out = work_dir / f"posteriors_{model}.npz"
    if out.exists():
        return
    decoder = load_decoder(model, h448)
    arrays = {}
    for i, (segment, samples) in enumerate(read_segments(segments, audio_dir)):
        (arrays[segment.key],) = decoder.span_log_posteriors([samples])
        if i % 500 == 0:
            print(f"[{model}] posteriors {i}/{len(segments)}", flush=True)
    np.savez(work_dir / f"posteriors_{model}.tmp.npz", **arrays)
    (work_dir / f"posteriors_{model}.fingerprint.json").write_text(
        json.dumps(decoder.fingerprint(SPANS).as_dict(), indent=2, sort_keys=True) + "\n"
    )
    os.replace(work_dir / f"posteriors_{model}.tmp.npz", out)
    del decoder
    _free_gpu()


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


# --- Stage 2: measure -----------------------------------------------------------------


def _measure_one(args):
    segment, log_posteriors = args
    return segment, sp.measure_segment(segment.reference, log_posteriors)


def census_dropped(decode: str, reference: str) -> frozenset[int]:
    """The #83 census's dropped gemination sites of one decode (raw reference indices)."""
    from tadabur.contrast_attribution import DROPPED, SHADDA_CONTRAST, contrast_sites

    return frozenset(
        site.reference_index
        for site in contrast_sites(decode, reference, SHADDA_CONTRAST)
        if site.change == DROPPED
    )


def measure_stage(model, segments, work_dir, workers) -> None:
    out = work_dir / f"observations_{model}.json"
    if out.exists():
        return
    from tadabur.mining_pool import load_base_decodes

    _, cached = load_base_decodes()
    posteriors = np.load(work_dir / f"posteriors_{model}.npz")
    observations, mismatches = [], []
    with ProcessPoolExecutor(workers) as pool:
        jobs = ((segment, posteriors[segment.key]) for segment in segments)
        for segment, measure in pool.map(_measure_one, jobs, chunksize=8):
            if model == BASE and measure.decode != cached[segment.key]:
                mismatches.append(segment.key)
            observations.extend(
                sp.segment_observations(
                    segment.key,
                    segment.reciter_id,
                    segment.inclusion_probability,
                    measure,
                    census_dropped(measure.decode, segment.reference),
                )
            )
    record = {
        "decode_fingerprint": json.loads(
            (work_dir / f"posteriors_{model}.fingerprint.json").read_text()
        ),
        "segments": len(segments),
        "base_decode_mismatches": mismatches if model == BASE else None,
        "observations": [asdict(o) for o in observations],
    }
    out.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def load_observations(work_dir: Path, model: str) -> tuple[dict, list[sp.Observation]]:
    record = json.loads((work_dir / f"observations_{model}.json").read_text(encoding="utf-8"))
    observations = [
        sp.Observation(**{**o, "site": sp.SiteMeasure(**o["site"])})
        for o in record.pop("observations")
    ]
    return record, observations


# --- Stage 3: stretch -----------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    from training.edit_decomposition import align

    return sum(op != "match" for op, _, _ in align(list(a), list(b)))


def stretch_stage(model, segments, audio_dir, h448, work_dir) -> None:
    from tadabur.time_stretch import added_samples, stretch_span

    out = work_dir / f"stretch_{model}.json"
    if out.exists():
        return
    _, observations = load_observations(work_dir, model)
    pairs = sp.match_controls(observations)
    wanted: dict[str, list[sp.Observation]] = {}
    for collapsed, control in pairs:
        for o in (collapsed, control):
            if o is not None:
                wanted.setdefault(o.segment, []).append(o)
    by_key = {s.key: s for s in segments}
    todo = [by_key[key] for key in sorted(wanted, key=lambda k: (by_key[k].audio_filename, k))]
    posteriors = np.load(work_dir / f"posteriors_{model}.npz")
    decoder = load_decoder(model, h448)
    population = {id(c): sp.COLLAPSED for c, _ in pairs} | {
        id(s): sp.SINGLE for _, s in pairs if s is not None
    }
    trials = []
    for segment, samples in read_segments(todo, audio_dir):
        original = posteriors[segment.key]
        original_decode = "".join(t.char for t in sp.decode_tokens(original))
        for o in wanted[segment.key]:
            index = o.site.reference_index
            frames = sp.site_interval(segment.reference, original, index)
            start = frames[0] * sp.FRAME_SAMPLES
            end = min(frames[1] * sp.FRAME_SAMPLES, len(samples))
            for factor in sp.STRETCH_FACTORS:
                edited = stretch_span(samples, start, end, factor)
                decoy = np.concatenate(
                    [samples, np.zeros(added_samples(start, end, factor), dtype=np.float32)]
                )
                measured = [
                    sp.measure_segment(segment.reference, lp, only=frozenset({index}))
                    for lp in decoder.span_log_posteriors([edited, decoy])
                ]
                trials.append(
                    sp.StretchTrial(
                        segment=segment.key,
                        reciter_id=o.reciter_id,
                        weight=o.weight,
                        population=population[id(o)],
                        reference_index=index,
                        factor=factor,
                        edited=measured[0].sites[0],
                        decoy=measured[1].sites[0],
                        edited_changes=_levenshtein(measured[0].decode, original_decode),
                        decoy_changes=_levenshtein(measured[1].decode, original_decode),
                    )
                )
        print(f"[{model}] stretch trials {len(trials)}", flush=True)
    record = {
        "pairs": len(pairs),
        "unmatched_controls": sum(c is None for _, c in pairs),
        "trials": [asdict(t) for t in trials],
    }
    out.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    del decoder
    _free_gpu()


def load_trials(work_dir: Path, model: str) -> tuple[dict, list[sp.StretchTrial]]:
    record = json.loads((work_dir / f"stretch_{model}.json").read_text(encoding="utf-8"))
    trials = [
        sp.StretchTrial(
            **{**t, "edited": sp.SiteMeasure(**t["edited"]), "decoy": sp.SiteMeasure(**t["decoy"])}
        )
        for t in record.pop("trials")
    ]
    return record, trials


# --- Stage 4: report ------------------------------------------------------------------


def report_stage(models, work_dir, out: Path) -> None:
    import hafs_phonetizer

    report = {
        "issue": 86,
        "preregistration": "docs/shaddah-probe-preregistration.md",
        "pool": "tools/tadabur/mining_pool (kept segments)",
        "phonetizer_revision": hafs_phonetizer.REVISION,
        "constants": {
            "frame_ms": sp.FRAME_MS,
            "present_log_ratio": sp.PRESENT_LOG_RATIO,
            "absent_log_ratio": sp.ABSENT_LOG_RATIO,
            "present_second_peak": sp.PRESENT_SECOND_PEAK,
            "stretch_factors": list(sp.STRETCH_FACTORS),
            "bootstrap": {"resamples": sp.BOOTSTRAP_RESAMPLES, "seed": sp.BOOTSTRAP_SEED},
        },
        "models": {},
    }
    for model in models:
        measured, observations = load_observations(work_dir, model)
        stretched, trials = load_trials(work_dir, model)
        report["models"][model] = {
            "model": measured["decode_fingerprint"]["model"],
            "decode_fingerprint": measured["decode_fingerprint"],
            "segments": measured["segments"],
            "base_decode_mismatches": measured["base_decode_mismatches"],
            "stretch_pairs": stretched["pairs"],
            "stretch_unmatched_controls": stretched["unmatched_controls"],
            **sp.model_report(observations, trials),
        }
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--h448", type=Path, required=True, help="the h448 checkpoint file")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=["posteriors", "measure", "stretch", "report", "all"],
                        default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="first N segments only (smoke)")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    segments = pool_segments()
    if args.limit:
        segments = segments[: args.limit]
    models = (BASE, H448)
    stages = ["posteriors", "measure", "stretch", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        for model in models:
            if stage == "posteriors":
                posteriors_stage(model, segments, args.audio_dir, args.h448, args.work_dir)
            elif stage == "measure":
                measure_stage(model, segments, args.work_dir, args.workers)
            elif stage == "stretch":
                stretch_stage(model, segments, args.audio_dir, args.h448, args.work_dir)
        if stage == "report":
            report_stage(models, args.work_dir, args.out)


if __name__ == "__main__":
    main()
