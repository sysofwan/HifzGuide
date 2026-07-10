"""The Tadabur filtering pipeline: stream → decode → score → passing-subset manifest.

This is Phase 3 of PRD #1 (ADR-0001). It streams ``FaisaI/tadabur`` once, resamples
each clip to 16 kHz mono (``tadabur.audio``), runs **batched** bf16 GPU inference in
one variable-length full-ayah pass (``tadabur.inference.MuaalemPhonemeModel.decode_batch``,
no 250-frame windowing), and greedy-CTC-decodes the phoneme head. Each decoded string
is scored against the cached ``quran-transcript`` reference for its ``surah:ayah``
(``tadabur.reference_phonemes``) with the ported ``.balanced`` gate
(``tadabur.scorer``). Passers are written to a resumable manifest
(``tadabur.manifest``) of the quality-filtered training subset.

Filtering is light on VRAM (~1.5 GB), so throughput comes from a large inference
batch over the 365k+ clips. The stream order and greedy (argmax) decode are
deterministic, and the manifest is resumable and idempotent, so the whole run
reproduces identically and can restart after a crash without re-scoring or
duplicating work.

Usage:
  python -m tadabur.filter --manifest passing_subset.jsonl [--batch-size 64]
    [--config-name preview] [--limit N] [--device cuda]
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from datasets import Audio, load_dataset

from .audio import TARGET_SAMPLE_RATE, decode_to_mono_16k
from .dataset_source import AUDIO_COLUMN, DATASET_ID, resolve_audio_filename
from .inference import MODEL_ID, MuaalemPhonemeModel
from .manifest import FilterManifest, ManifestRecord
from .reference_phonemes import load_reference_phonemes
from .scorer import BALANCED_SCORER, Scorer

DEFAULT_BATCH_SIZE = 64


@dataclass(frozen=True)
class Clip:
    """The audio bytes and metadata the filter needs from one streamed Tadabur row."""

    audio_filename: str
    surah_ayah: str
    reciter_id: int
    audio_bytes: bytes


def canonical_surah_ayah(surah_id: int, ayah_id: int) -> str:
    """Map a Tadabur ``(surah_id, ayah_id)`` to a canonical ``"surah:ayah"`` key.

    Tadabur numbers ``surah_id`` **0-indexed** (0–113, a surah *array index*, and
    the same 0-based number embedded in the audio filename, e.g. ``S77`` for
    Al-Naba, the 78th surah) while ``ayah_id`` is the natural **1-indexed** ayah
    number. Our reference cache is keyed by the canonical 1-indexed
    ``surah:ayah`` (``quran-transcript``), so we shift the surah by one here.
    Without this shift every clip gates against the wrong ayah and *nothing*
    passes the filter.
    """
    return f"{surah_id + 1}:{ayah_id}"


def parse_clip(row: dict) -> Clip:
    """Extract the audio bytes and required metadata from a streamed Tadabur row.

    Fails loudly if a required field is missing rather than silently emitting a
    manifest row that cannot be traced back to its audio or reference ayah.
    """
    audio = row.get(AUDIO_COLUMN)
    if not audio or audio.get("bytes") is None:
        raise ValueError(f"Tadabur row has no decodable audio bytes: {row.get('audio_filename')!r}")
    for field in ("surah_id", "ayah_id", "reciter_id"):
        if row.get(field) is None:
            raise ValueError(f"Tadabur row missing required field {field!r}: {row!r}")
    return Clip(
        audio_filename=resolve_audio_filename(row),
        surah_ayah=canonical_surah_ayah(int(row["surah_id"]), int(row["ayah_id"])),
        reciter_id=int(row["reciter_id"]),
        audio_bytes=row[AUDIO_COLUMN]["bytes"],
    )


def score_batch(
    clips: list[Clip],
    model: MuaalemPhonemeModel,
    references: dict[str, str],
    scorer: Scorer,
    skip_unknown_refs: bool = False,
) -> list[ManifestRecord]:
    """Decode and score a batch of clips, returning a record for each passer.

    Each clip's decoded phonemes are gated against its cached reference. By
    default the reference must exist (all 6236 canonical ayat are cached) or the
    row is bad data and we fail loudly. ``skip_unknown_refs`` relaxes that to
    *skip* a clip whose ``surah:ayah`` is not canonical instead of raising — used
    for the ``preview`` config, which mixes in non-canonical rows the strict full
    run never sees. ``ayah_duration_s`` is the duration of the 16 kHz waveform
    actually scored.
    """
    waveforms = [decode_to_mono_16k(clip.audio_bytes) for clip in clips]
    decodes = model.decode_batch(waveforms, TARGET_SAMPLE_RATE)

    records: list[ManifestRecord] = []
    for clip, waveform, decode in zip(clips, waveforms, decodes):
        reference = references.get(clip.surah_ayah)
        if reference is None:
            if skip_unknown_refs:
                continue
            raise ValueError(
                f"No cached reference for {clip.surah_ayah} "
                f"(clip {clip.audio_filename}); outside the canonical 6236 ayat."
            )
        result = scorer.gate(decode.phonemes, reference)
        if result.passed:
            records.append(
                ManifestRecord(
                    audio_filename=clip.audio_filename,
                    surah_ayah=clip.surah_ayah,
                    match_ratio=result.match_ratio,
                    ayah_duration_s=len(waveform) / TARGET_SAMPLE_RATE,
                    reciter_id=clip.reciter_id,
                    contrasts=scorer.attribute(decode.phonemes, reference),
                    predicted_phonemes=decode.phonemes,
                )
            )
    return records


def _batched(iterable: Iterable, size: int) -> Iterator[list]:
    """Yield successive ``size``-length lists from ``iterable`` (final may be shorter)."""
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def stream_clips(
    dataset_id: str,
    config_name: str | None,
    split: str,
    start: int,
    limit: int | None,
) -> Iterator[Clip]:
    """Stream parsed clips from ``dataset_id``, resuming after ``start`` clips.

    Reads audio with ``decode=False`` (raw WAV bytes, no ``torchcodec`` dependency),
    skips the ``start`` clips already scored on a prior run, and stops after
    ``limit`` clips this run when given.
    """
    dataset = load_dataset(dataset_id, name=config_name, split=split, streaming=True)
    dataset = dataset.cast_column(AUDIO_COLUMN, Audio(decode=False))
    rows: Iterator[dict] = iter(dataset)
    if start:
        rows = itertools.islice(rows, start, None)
    if limit is not None:
        rows = itertools.islice(rows, limit)
    for row in rows:
        yield parse_clip(row)


def run_filter(
    manifest: FilterManifest,
    model: MuaalemPhonemeModel,
    references: dict[str, str],
    scorer: Scorer,
    dataset_id: str = DATASET_ID,
    config_name: str | None = None,
    split: str = "train",
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    skip_unknown_refs: bool = False,
) -> None:
    """Filter the stream in batches, committing passers to ``manifest`` as it goes.

    Resumes from ``manifest.clips_processed`` and commits after every batch so a
    crash loses at most the last in-flight batch. ``skip_unknown_refs`` is passed
    through to :func:`score_batch` (see there).
    """
    clips = stream_clips(
        dataset_id, config_name, split, start=manifest.clips_processed, limit=limit
    )
    for batch in _batched(clips, batch_size):
        records = score_batch(batch, model, references, scorer, skip_unknown_refs)
        manifest.commit_batch(records, num_clips=len(batch))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Output JSONL manifest of the passing subset (appended to if it exists).",
    )
    parser.add_argument("--dataset", default=DATASET_ID, help="HF dataset id.")
    parser.add_argument(
        "--config-name",
        default=None,
        help="Dataset config name (e.g. 'preview' for fast small-row-group streaming).",
    )
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Clips per bf16 inference batch (filtering is ~1.5 GB VRAM; go large).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many clips this run (after resume skip).",
    )
    parser.add_argument("--model-id", default=MODEL_ID, help="HF model id.")
    parser.add_argument(
        "--device", default="cuda", help="Torch device (default: cuda)."
    )
    parser.add_argument(
        "--skip-unknown-refs",
        action="store_true",
        help="Skip (rather than fail on) clips whose surah:ayah is not a canonical "
             "ayah. Needed for the 'preview' config, which mixes in non-canonical rows.",
    )
    args = parser.parse_args()

    print(f"Loading references and {args.model_id} (bf16) on {args.device} ...")
    references = load_reference_phonemes()
    model = MuaalemPhonemeModel.load(args.model_id, device=args.device)

    with FilterManifest.open(args.manifest) as manifest:
        if manifest.clips_processed:
            print(f"Resuming after {manifest.clips_processed} clips already scored.")
        run_filter(
            manifest,
            model,
            references,
            BALANCED_SCORER,
            dataset_id=args.dataset,
            config_name=args.config_name,
            split=args.split,
            batch_size=args.batch_size,
            limit=args.limit,
            skip_unknown_refs=args.skip_unknown_refs,
        )
        print(
            f"Done. {manifest.clips_processed} clips scored; "
            f"{manifest.passers_written} passers in {args.manifest}."
        )


if __name__ == "__main__":
    main()
