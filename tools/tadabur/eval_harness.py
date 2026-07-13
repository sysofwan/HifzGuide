"""Two-sided eval harness CLI (ADR-0001, #7) — decode the fixtures, score, record.

Runs an arbitrary Muaalem checkpoint over the curated should-accept / should-reject
fixtures (:mod:`tadabur.eval_fixtures`) and writes the full two-sided report
(:mod:`tadabur.eval_report`): the soft-pair + shadda confusion matrix and the
should-accept recall / should-reject discrimination against Muraja's ``.strict`` gate.
Run once on the base model to establish the baseline this slice ships; run again on a
fine-tuned checkpoint (``--model-id``) to compare.

The fixtures name a clip by its per-segment ``clip_id`` but carry no reference or audio.
Both are joined from the P3.5 segment run (:mod:`tadabur.segment_score`): the realized,
already-normalized reference from the segment manifest (via
:func:`tadabur.audit_ui.segment_display_index`), and the clip's audio from
``--audio-dir`` under its :func:`tadabur.audit_sampler.local_audio_path` name — the same
files the audit UI served. A fixture clip missing from either **fails loudly**: a silent
skip would bias recall/discrimination.

Decoding reuses the exact filter engine (:class:`tadabur.inference.MuaalemPhonemeModel`,
bf16 on CUDA) and 16 kHz-mono preprocessing (:func:`tadabur.audio.decode_to_mono_16k`),
so train/inference parity holds. Greedy CTC decode is deterministic; the report is a
pure function of the decodes, so a rerun on the same checkpoint reproduces it.

Usage:
  python -m tadabur.eval_harness \\
      --segment-manifest audit_run/segment_manifest_v2.jsonl \\
      --audio-dir audit_run/segment_audio_v2 \\
      --out eval_fixtures/base_model_baseline.json \\
      [--model-id obadx/muaalem-model-v3_2] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio import TARGET_SAMPLE_RATE, decode_to_mono_16k
from .audit_sampler import local_audio_path
from .audit_ui import segment_display_index
from .eval_fixtures import EvalFixtureEntry, load_should_accept, load_should_reject
from .eval_report import ClipDecode, EvalReport, evaluate

DEFAULT_MODEL_ID = "obadx/muaalem-model-v3_2"
DEFAULT_BATCH_SIZE = 16


def _resolve_audio_path(audio_dir: Path, clip_id: str) -> Path:
    return audio_dir / local_audio_path(clip_id)


def _prepare_clips(
    fixtures: list[EvalFixtureEntry],
    references: dict[str, str],
    audio_dir: Path,
) -> list[tuple[EvalFixtureEntry, Path]]:
    """Pair each fixture with its on-disk audio path, failing loudly on any gap.

    A fixture clip with no realized reference in the segment manifest, or no exported
    audio under ``audio_dir``, cannot be scored; skipping it would bias the metrics, so
    every miss is collected and raised together (pointing at the wrong manifest/audio-dir).
    """
    missing_ref: list[str] = []
    missing_audio: list[str] = []
    prepared: list[tuple[EvalFixtureEntry, Path]] = []
    for entry in fixtures:
        if not references.get(entry.clip_id):
            missing_ref.append(entry.clip_id)
            continue
        audio_path = _resolve_audio_path(audio_dir, entry.clip_id)
        if not audio_path.is_file():
            missing_audio.append(entry.clip_id)
            continue
        prepared.append((entry, audio_path))

    problems = []
    if missing_ref:
        problems.append(f"{len(missing_ref)} without a segment-manifest reference (e.g. {missing_ref[:3]})")
    if missing_audio:
        problems.append(f"{len(missing_audio)} without exported audio (e.g. {missing_audio[:3]})")
    if problems:
        raise FileNotFoundError(
            "Cannot score every fixture clip: "
            + "; ".join(problems)
            + ". Check --segment-manifest and --audio-dir match the run that curated "
            "the fixtures."
        )
    return prepared


def _decode_clips(
    prepared: list[tuple[EvalFixtureEntry, Path]],
    references: dict[str, str],
    model_id: str,
    batch_size: int,
) -> list[ClipDecode]:
    """Batch-decode every prepared clip and pair each decode with its fixture label."""
    from .inference import MuaalemPhonemeModel  # lazy: torch/transformers import

    model = MuaalemPhonemeModel.load(model_id)
    clips: list[ClipDecode] = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        waveforms = [decode_to_mono_16k(path.read_bytes()) for _, path in batch]
        decodes = model.decode_batch(waveforms, TARGET_SAMPLE_RATE)
        for (entry, _), decode in zip(batch, decodes):
            clips.append(
                ClipDecode(
                    clip_id=entry.clip_id,
                    contrast=entry.contrast,
                    verdict=entry.verdict,
                    predicted=decode.phonemes,
                    reference=references[entry.clip_id],
                )
            )
    return clips


def run_eval(
    segment_manifest: Path,
    audio_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EvalReport:
    """Decode the curated fixtures with ``model_id`` and score them into a report."""
    fixtures = load_should_accept() + load_should_reject()
    references = segment_display_index(segment_manifest)["reference"]
    prepared = _prepare_clips(fixtures, references, audio_dir)
    clips = _decode_clips(prepared, references, model_id, batch_size)
    return evaluate(clips, model_id)


def _print_summary(report: EvalReport) -> None:
    accept, reject = report.should_accept, report.should_reject
    print(f"Model: {report.model_id}  (strict threshold {report.strict_threshold})")
    print(
        f"should-accept recall:        {accept.recall}  "
        f"({accept.accepted}/{accept.total} admitted)"
    )
    print(
        f"should-reject discrimination: {reject.discrimination}  "
        f"({reject.rejected}/{reject.total} still rejected)"
    )
    shadda = report.shadda_confusion
    print(f"shadda occurrences: added={shadda.added} dropped={shadda.dropped}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment-manifest", type=Path, required=True,
                        help="Segment manifest (JSONL) carrying each clip's realized reference_phonemes.")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="Directory of exported segment audio (local_audio_path names).")
    parser.add_argument("--out", type=Path, required=True, help="Report output path (JSON).")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help=f"Muaalem checkpoint to evaluate (default: {DEFAULT_MODEL_ID}).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Decode batch size (default: {DEFAULT_BATCH_SIZE}).")
    args = parser.parse_args()

    report = run_eval(args.segment_manifest, args.audio_dir, args.model_id, args.batch_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report.to_json_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    _print_summary(report)
    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
