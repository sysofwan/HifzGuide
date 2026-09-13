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

  The 'preview' config is a fixed 300-row sample; to filter the full 385-shard corpus
  (the P3.5 audit needs it — see ``tadabur.shard_reader``) read parquet shards directly:
  python -m tadabur.filter --manifest passing_subset_full.jsonl --shards 0-19
    [--delete-shards] [--batch-size 4]

  ``--rejects`` additionally keeps what the gate turned away (``tadabur.rejects``), and
  ``--reject-audio-out`` stages the 16 kHz WAV of every reject matching the clean-re-read
  predicate — the mining pass Muraja ADR-0016 builds its follow-along corpus from. Both
  are off by default, and without them this writes exactly the bytes it always did.
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
from .rejects import RejectRecord, build_reject_record, write_clip_wav
from .scorer import BALANCED_SCORER, Scorer

DEFAULT_BATCH_SIZE = 64

# Skip clips longer than this before the single-pass GPU decode: the model runs one
# variable-length full-ayah pass (no windowing), so a pathologically long clip (a
# mis-segmented whole-page recording, not a real ayah) blows up the Wav2Vec2-BERT
# activations and OOMs the whole batch. The longest *legitimately-scored* ayah is
# ~82 s and p99 is ~44 s, so a 50 s cap keeps 99.5% of clips while dropping only the
# monsters that cannot be decoded on a 16 GB GPU anyway.
MAX_AYAH_DURATION_S = 50.0


@dataclass(frozen=True)
class Clip:
    """The audio bytes and metadata the filter needs from one streamed Tadabur row."""

    audio_filename: str
    surah_ayah: str
    reciter_id: int
    audio_bytes: bytes


@dataclass(frozen=True)
class ScoredBatch:
    """What one decoded batch yielded: the passers, the rejects, and staged audio.

    ``passers`` is the manifest's input, exactly as before. ``rejects`` is the sink's,
    and is built unconditionally — every field on it is a ``GateResult`` the gate
    already computed, so producing it costs nothing a run can notice, and making it
    conditional would mean the reject path only ever runs when it is also being
    written, which is the path least likely to be exercised by a test.
    ``clean_re_read_audio`` pairs a clip's stable filename with the 16 kHz waveform the
    gate actually scored, for the subset matching the clean-re-read predicate; the
    caller stages it (or, with no ``--reject-audio-out``, does not).
    """

    passers: list[ManifestRecord]
    rejects: list[RejectRecord]
    clean_re_read_audio: list[tuple[str, "object"]]


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
) -> ScoredBatch:
    """Decode and score a batch of clips into a :class:`ScoredBatch`.

    Each clip's decoded phonemes are gated against its cached reference. By
    default the reference must exist (all 6236 canonical ayat are cached) or the
    row is bad data and we fail loudly. ``skip_unknown_refs`` relaxes that to
    *skip* a clip whose ``surah:ayah`` is not canonical instead of raising — used
    for the ``preview`` config, which mixes in non-canonical rows the strict full
    run never sees. ``ayah_duration_s`` is the duration of the 16 kHz waveform
    actually scored.

    A clip dropped before the decode — over-long, or (under ``skip_unknown_refs``)
    without a reference — is in neither list: it was never gated, so calling it a
    reject would put a verdict in the sink that no gate ever reached.
    """
    waveforms = [decode_to_mono_16k(clip.audio_bytes) for clip in clips]
    scorable: list[tuple[Clip, "object"]] = []
    for clip, waveform in zip(clips, waveforms):
        if len(waveform) / TARGET_SAMPLE_RATE > MAX_AYAH_DURATION_S:
            print(
                f"Skipping over-long clip {clip.audio_filename} "
                f"({len(waveform) / TARGET_SAMPLE_RATE:.1f}s > {MAX_AYAH_DURATION_S:.0f}s cap)."
            )
            continue
        scorable.append((clip, waveform))
    if not scorable:
        return []
    decodes = model.decode_batch([w for _, w in scorable], TARGET_SAMPLE_RATE)

    records: list[ManifestRecord] = []
    rejects: list[RejectRecord] = []
    clean_re_read_audio: list[tuple[str, "object"]] = []
    for (clip, waveform), decode in zip(scorable, decodes):
        reference = references.get(clip.surah_ayah)
        if reference is None:
            if skip_unknown_refs:
                continue
            raise ValueError(
                f"No cached reference for {clip.surah_ayah} "
                f"(clip {clip.audio_filename}); outside the canonical 6236 ayat."
            )
        result = scorer.gate(decode.phonemes, reference)
        duration_s = len(waveform) / TARGET_SAMPLE_RATE
        if result.passed:
            records.append(
                ManifestRecord(
                    audio_filename=clip.audio_filename,
                    surah_ayah=clip.surah_ayah,
                    match_ratio=result.match_ratio,
                    ayah_duration_s=duration_s,
                    reciter_id=clip.reciter_id,
                    contrasts=scorer.attribute(decode.phonemes, reference),
                    predicted_phonemes=decode.phonemes,
                )
            )
            continue
        reject = build_reject_record(
            audio_filename=clip.audio_filename,
            surah_ayah=clip.surah_ayah,
            reciter_id=clip.reciter_id,
            ayah_duration_s=duration_s,
            predicted=decode.phonemes,
            result=result,
            scorer=scorer,
        )
        rejects.append(reject)
        if reject.is_clean_re_read:
            clean_re_read_audio.append((clip.audio_filename, waveform))
    return ScoredBatch(records, rejects, clean_re_read_audio)


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
    clip_source: Iterable[Clip] | None = None,
    reject_audio_dir: Path | None = None,
) -> None:
    """Filter the stream in batches, committing passers to ``manifest`` as it goes.

    Resumes from ``manifest.clips_processed`` and commits after every batch so a
    crash loses at most the last in-flight batch. ``skip_unknown_refs`` is passed
    through to :func:`score_batch` (see there). ``clip_source`` overrides the default
    ``datasets`` stream with a caller-supplied iterable of :class:`Clip` (the full-config
    parquet-shard reader; see :func:`main`) — the caller then owns resume-skipping and
    ``limit``, since a shard source is positioned by shard, not stream offset.

    ``reject_audio_dir`` stages the clean-re-read WAVs. They are written *before* the
    commit, so the checkpoint never advances past a clip whose audio is missing; a
    replayed batch rewrites identical bytes. Whether the manifest actually keeps the
    rejects is the manifest's business (it opened the sink, or did not), which is why
    they are handed over unconditionally.
    """
    clips = clip_source if clip_source is not None else stream_clips(
        dataset_id, config_name, split, start=manifest.clips_processed, limit=limit
    )
    for batch in _batched(clips, batch_size):
        scored = score_batch(batch, model, references, scorer, skip_unknown_refs)
        if reject_audio_dir is not None:
            for audio_filename, waveform in scored.clean_re_read_audio:
                write_clip_wav(reject_audio_dir, audio_filename, waveform)
        manifest.commit_batch(
            scored.passers, num_clips=len(batch), rejects=scored.rejects
        )


def _shard_clip_source(
    spec: str,
    clips_processed: int,
    dataset_id: str,
    shard_cache: Path | None,
    delete_shards: bool,
    limit: int | None,
) -> Iterator[Clip]:
    """Build a :class:`Clip` iterator over the full-config parquet shards named by ``spec``.

    Resume is by whole shard: each shard is exactly :data:`~tadabur.shard_reader.ROWS_PER_SHARD`
    rows, so the ``clips_processed`` checkpoint already reflects an integer number of
    finished shards and the run skips those. A ``--limit`` (partial shard) is honoured for
    this run but breaks the shard-boundary invariant, so it is meant for smoke probes into
    a throwaway manifest, not resumable full runs.
    """
    from .shard_reader import ROWS_PER_SHARD, iter_shard_rows, parse_shard_spec

    indices = parse_shard_spec(spec)
    done_shards = clips_processed // ROWS_PER_SHARD
    remaining = indices[done_shards:]
    if done_shards:
        print(f"Shard resume: skipping {done_shards} finished shard(s).")
    rows = iter_shard_rows(
        remaining,
        dataset_id=dataset_id,
        cache_dir=shard_cache,
        delete_after=delete_shards,
    )
    clips = (parse_clip(row) for row in rows)
    return itertools.islice(clips, limit) if limit is not None else clips


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
    parser.add_argument(
        "--shards",
        default=None,
        help="Read the full 'default' config directly from parquet shards instead of "
             "the (300-row) 'preview' stream. A spec like '0-9' or '0,5,20-21' "
             "(see tadabur.shard_reader); dodges the datasets nested-chunk bug.",
    )
    parser.add_argument(
        "--shard-cache",
        type=Path,
        default=None,
        help="Directory for downloaded shards (default: the shared HF cache).",
    )
    parser.add_argument(
        "--delete-shards",
        action="store_true",
        help="Delete each 2.4 GB shard after scoring it, to bound disk on long runs.",
    )
    parser.add_argument(
        "--rejects",
        type=Path,
        default=None,
        help="Also write a JSONL of every clip the gate rejected, with the GateResult "
             "behind the verdict (tadabur.rejects). Omitted, no reject file is opened "
             "and the passing manifest is byte-identical to a run without this flag.",
    )
    parser.add_argument(
        "--reject-audio-out",
        type=Path,
        default=None,
        help="Directory to stage the 16 kHz mono WAV of each reject matching the "
             "clean-re-read predicate (Muraja ADR-0016). Independent of --rejects, "
             "though a run normally wants both.",
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

    with FilterManifest.open(args.manifest, rejects_path=args.rejects) as manifest:
        if manifest.clips_processed:
            print(f"Resuming after {manifest.clips_processed} clips already scored.")
        clip_source = None
        if args.shards:
            clip_source = _shard_clip_source(
                args.shards, manifest.clips_processed, args.dataset,
                args.shard_cache, args.delete_shards, args.limit,
            )
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
            clip_source=clip_source,
            reject_audio_dir=args.reject_audio_out,
        )
        print(
            f"Done. {manifest.clips_processed} clips scored; "
            f"{manifest.passers_written} passers in {args.manifest}."
        )
        if args.rejects:
            print(f"{manifest.rejects_written} rejects in {args.rejects}.")


if __name__ == "__main__":
    main()
