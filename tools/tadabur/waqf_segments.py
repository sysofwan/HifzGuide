"""Waqf-aware segmentation of passing Tadabur clips into realized reference labels.

The Tadabur filter (``tadabur.filter``) admits a clip on a *single* full-ayah
reference — one canonical guess at what was recited. But a reciter who **stops**
(makes waqf) partway through an ayah realizes the words differently from one who
recites it continuously (wasl): the word before the stop drops its final haraka
and loses the cross-word gemination/idgham it would carry in continuation. Labelling
every clip with the full-ayah wasl reference therefore injects *phantom* pre-waqf
gemination mismatches into the fine-tune data. This module removes them the way
upstream Muaalem does — by making each training example's label match what was
*actually* recited.

It needs no model and no GPU. Tadabur already ships a forced alignment per clip
(``metadata.word_alignments``: per-word ``start``/``end``). An intra-ayah **waqf
pause** shows up as an inter-word *gap* — ``word[i+1].start - word[i].end`` above a
threshold — while continuous recitation shows overlapping/near-zero gaps. We split
each clip's words at those pauses into contiguous **waqf segments** and phonetize
each segment's Uthmani text on its own: ``quran_phonetizer``'s CleanEnd op puts the
segment's terminal word in **waqf** form and leaves the interior words in **wasl**,
which is exactly the realized reference.

The output is an **offsets manifest** (:class:`SegmentRecord` JSONL), not new audio:
each segment is a lightweight ``(start_s, end_s)`` view into the whole clip, sliced
at collate time by P4 (#8). Whole passing clips are kept locally as 16 kHz mono WAV;
the full Tadabur source is streamed, never landed. The manifest is written by a
deterministic full sort-then-rewrite, so re-running reproduces it byte-for-byte.

Two per-clip data-quality cases are **skipped and tallied**, never silently
mislabeled: a clip whose alignment word count differs from its Uthmani word count
(the vocative ``يا`` is a separate simple-text word but merged in Uthmani, so the
positional word map is unsafe), and one of the 8 ayat ``quran_phonetizer`` cannot
handle (leen madd on a final sukoon — see ``generate_phonemes.FALLBACK_PHONEMES``).

Usage:
  python -m tadabur.waqf_segments --passing passing_subset.jsonl \
      --out segments.jsonl --audio-dir clips/ [--config-name preview] \
      [--pause-threshold 0.25] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

_TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import generate_phonemes  # noqa: E402  (tools/ sibling module)

from .audio import TARGET_SAMPLE_RATE, decode_to_mono_16k
from .manifest import ManifestRecord, read_records
from .reference_phonemes import load_reference_phonemes

# Tadabur source identifiers, redefined locally (as in tadabur.smoke_decode) so this
# no-model labelling stage need not import the GPU inference path from tadabur.filter.
DATASET_ID = "FaisaI/tadabur"
AUDIO_COLUMN = "audio"

# Inter-word gap (seconds) at or above which a boundary is a waqf pause. Validated
# against 300 preview clips: inter-word gaps are overwhelmingly negative (words
# overlap in continuous recitation; median ≈ -0.12 s, p95 ≈ 0.10 s), and genuine
# pauses form a clear tail beyond ~0.15 s. 0.25 s sits in the stable middle of the
# 0.15–0.30 s band, so it catches real stops without splitting on alignment slack.
DEFAULT_PAUSE_THRESHOLD_S = 0.25

# A callable that turns Uthmani text into its phoneme string (a seam so the pure
# segmentation logic is testable without quran-transcript).
Phonetizer = Callable[[str], str]


@dataclass(frozen=True)
class WordAlignment:
    """One force-aligned word from a clip's ``metadata.word_alignments``."""

    word: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class SegmentRecord:
    """One waqf segment: a realized-reference view into a whole passing clip.

    ``audio_filename`` names the whole 16 kHz-mono clip kept on local disk;
    ``(start_s, end_s)`` is the segment's span within it (sliced at collate time,
    never re-materialized). ``segment_index`` orders the segments within the clip
    and, with ``audio_filename``, is the manifest's stable key. ``word_start`` /
    ``word_end`` are the half-open alignment/Uthmani word range the segment covers.
    ``realized_reference_phonemes`` is ``quran_phonetizer`` over the segment's
    Uthmani words — terminal word in waqf form, interior words in wasl.
    """

    audio_filename: str
    surah_ayah: str
    reciter_id: int
    segment_index: int
    word_start: int
    word_end: int
    start_s: float
    end_s: float
    realized_reference_phonemes: str


def parse_word_alignments(metadata_json: str) -> list[WordAlignment]:
    """Parse the ``word_alignments`` list out of a row's ``metadata`` JSON string.

    Tadabur stores ``metadata`` as a JSON-encoded string; its ``word_alignments``
    are the forced-alignment words in recitation order. Fails loudly if the field
    is absent, since a clip with no alignment cannot be segmented.
    """
    metadata = json.loads(metadata_json)
    alignments = metadata.get("word_alignments")
    if not alignments:
        raise ValueError("metadata has no word_alignments")
    return [
        WordAlignment(w["word"], float(w["start"]), float(w["end"]))
        for w in alignments
    ]


def split_at_pauses(
    alignments: list[WordAlignment], pause_threshold_s: float
) -> list[tuple[int, int]]:
    """Split words into contiguous ``[start, end)`` ranges at waqf pauses.

    A pause falls before word ``i+1`` when the inter-word gap
    ``alignments[i+1].start_s - alignments[i].end_s`` is at least
    ``pause_threshold_s``. Continuous (wasl) recitation, whose words overlap or
    abut, yields a single range spanning the whole clip.
    """
    if not alignments:
        return []
    boundaries = [0]
    for i in range(len(alignments) - 1):
        gap = alignments[i + 1].start_s - alignments[i].end_s
        if gap >= pause_threshold_s:
            boundaries.append(i + 1)
    boundaries.append(len(alignments))
    return list(zip(boundaries, boundaries[1:]))


def build_clip_segments(
    manifest_record: ManifestRecord,
    alignments: list[WordAlignment],
    uthmani_words: list[str],
    phonetize: Phonetizer,
    pause_threshold_s: float,
) -> list[SegmentRecord]:
    """Build the waqf-segment records for one passing clip.

    The alignment words and ``uthmani_words`` must correspond one-to-one and in
    order (the caller guarantees equal length; see :func:`_segment_passing_rows`),
    so word range ``[a, b)`` selects both the segment's time span (from the
    alignment) and its Uthmani text (for the realized reference). Each segment's
    reference is ``phonetize`` over its space-joined Uthmani words, which lands the
    terminal word in waqf form and the interior words in wasl.
    """
    assert len(alignments) == len(uthmani_words)
    records: list[SegmentRecord] = []
    for index, (start, end) in enumerate(
        split_at_pauses(alignments, pause_threshold_s)
    ):
        reference = phonetize(" ".join(uthmani_words[start:end]))
        records.append(
            SegmentRecord(
                audio_filename=manifest_record.audio_filename,
                surah_ayah=manifest_record.surah_ayah,
                reciter_id=manifest_record.reciter_id,
                segment_index=index,
                word_start=start,
                word_end=end,
                start_s=alignments[start].start_s,
                end_s=alignments[end - 1].end_s,
                realized_reference_phonemes=reference,
            )
        )
    return records


def hafs_phonetizer() -> Phonetizer:
    """A :data:`Phonetizer` that phonetizes Uthmani text with the Hafs moshaf.

    Reuses ``generate_phonemes.HAFS_MOSHAF`` — the same recitation configuration
    the full-ayah reference cache is built with — so a segment's realized reference
    differs from the full-ayah reference only where waqf vs wasl actually differs.
    """
    from quran_transcript import quran_phonetizer
    from quran_transcript.phonetics.moshaf_attributes import MoshafAttributes

    moshaf = MoshafAttributes(**generate_phonemes.HAFS_MOSHAF)
    return lambda text: quran_phonetizer(text, moshaf).phonemes


def write_segment_manifest(path: Path, records: list[SegmentRecord]) -> None:
    """Write ``records`` as a deterministic, idempotent JSONL offsets manifest.

    Records are sorted by ``(audio_filename, segment_index)`` and written with
    sorted JSON keys, then the file is replaced atomically, so re-running the build
    over the same passing subset reproduces the manifest byte-for-byte.
    """
    ordered = sorted(records, key=lambda r: (r.audio_filename, r.segment_index))
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in ordered:
            f.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def read_segment_manifest(path: Path) -> list[SegmentRecord]:
    """Load every :class:`SegmentRecord` from an offsets manifest in file order."""
    records: list[SegmentRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(SegmentRecord(**json.loads(line)))
    return records


def _uthmani_words(surah_ayah: str) -> list[str]:
    """The Uthmani words of ``"surah:ayah"`` (1-indexed), via quran-transcript."""
    from quran_transcript import Aya

    surah, ayah = (int(part) for part in surah_ayah.split(":"))
    return list(Aya(surah, ayah).get().uthmani_words)


def _save_local_clip(audio_dir: Path, audio_filename: str, audio_bytes: bytes) -> None:
    """Resample a streamed clip to 16 kHz mono and write it under ``audio_dir``."""
    waveform = decode_to_mono_16k(audio_bytes)
    sf.write(
        audio_dir / audio_filename, waveform, TARGET_SAMPLE_RATE, subtype="PCM_16"
    )


def _stream_passing_rows(
    passing: dict[str, ManifestRecord],
    dataset_id: str,
    config_name: str | None,
    split: str,
    limit: int | None,
) -> Iterator[tuple[dict, ManifestRecord]]:
    """Stream Tadabur, yielding only rows whose clip is in the passing subset.

    Reads audio undecoded (raw WAV bytes, no ``torchcodec``) like the filter, pairs
    each kept row with its passing :class:`ManifestRecord`, and stops as soon as
    every passing clip has been seen so the whole corpus need not be streamed.
    """
    from .filter import resolve_audio_filename  # lazy: keeps this stage torch-free

    dataset = load_dataset(dataset_id, name=config_name, split=split, streaming=True)
    dataset = dataset.cast_column(AUDIO_COLUMN, Audio(decode=False))
    remaining = set(passing)
    consumed = 0
    for row in dataset:
        if limit is not None and consumed >= limit:
            break
        consumed += 1
        audio_filename = resolve_audio_filename(row)
        record = passing.get(audio_filename)
        if record is None:
            continue
        yield row, record
        remaining.discard(audio_filename)
        if not remaining:
            break


def build_segments(
    passing_records: list[ManifestRecord],
    phonetize: Phonetizer,
    audio_dir: Path,
    dataset_id: str = DATASET_ID,
    config_name: str | None = None,
    split: str = "train",
    limit: int | None = None,
    pause_threshold_s: float = DEFAULT_PAUSE_THRESHOLD_S,
) -> tuple[list[SegmentRecord], Counter]:
    """Stream the passing subset, save local audio, and build segment records.

    For each passing clip found in the stream, saves its whole 16 kHz-mono waveform
    under ``audio_dir`` and splits it into waqf segments. Clips whose alignment word
    count disagrees with their Uthmani word count, or that hit the phonetizer's
    8-ayah gap, are skipped and tallied (returned :class:`~collections.Counter`)
    rather than silently mislabeled. Returns the collected records and the skip
    tally.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    passing = {record.audio_filename: record for record in passing_records}

    records: list[SegmentRecord] = []
    skips: Counter = Counter()
    for row, record in _stream_passing_rows(
        passing, dataset_id, config_name, split, limit
    ):
        alignments = parse_word_alignments(row["metadata"])
        uthmani_words = _uthmani_words(record.surah_ayah)
        if len(alignments) != len(uthmani_words):
            skips["word_count_mismatch"] += 1
            continue
        try:
            clip_records = build_clip_segments(
                record, alignments, uthmani_words, phonetize, pause_threshold_s
            )
        except KeyError:
            # quran_phonetizer raises KeyError on the 8 leen-madd-on-sukoon ayat
            # (generate_phonemes.FALLBACK_PHONEMES); those cannot be re-phonetized
            # per segment, so skip the clip rather than emit a bad reference.
            skips["phonetizer_unsupported"] += 1
            continue
        _save_local_clip(audio_dir, record.audio_filename, row[AUDIO_COLUMN]["bytes"])
        records.extend(clip_records)
    return records, skips


def _normalized_realized_reference(records: list[SegmentRecord]) -> str:
    """A clip's realized reference: its segments' phonemes joined and normalized.

    Segments are ordered by ``segment_index`` and space-joined (the same word
    separator the full-ayah reference uses), then run through the ``.balanced``
    normalization so it can be compared on the same footing as the cached full-ayah
    reference by :func:`shadda_contrast_report`.
    """
    from .normalization import normalize_phonemes

    ordered = sorted(records, key=lambda r: r.segment_index)
    joined = " ".join(r.realized_reference_phonemes for r in ordered)
    return normalize_phonemes(joined).normalized


def shadda_contrast_report(
    passing_records: list[ManifestRecord],
    segment_records: list[SegmentRecord],
    references: dict[str, str],
) -> dict[str, int]:
    """Count phantom pre-waqf shadda contrasts the realized labels remove.

    For every clip that split into more than one segment (i.e. it contained an
    intra-ayah waqf), re-attributes the ``.balanced`` shadda-contrast bucket for the
    model's stored decode against (a) the cached full-ayah reference — the label the
    filter used — and (b) the realized (segmented) reference. A clip that showed a
    shadda contrast under the full-ayah label but not under the realized one is a
    phantom pre-waqf gemination mismatch that this stage removes. No model inference
    — it reuses the decode already stored in the passing manifest.
    """
    from .contrast_attribution import SHADDA_CONTRAST, attribute_contrasts

    by_clip: dict[str, list[SegmentRecord]] = {}
    for record in segment_records:
        by_clip.setdefault(record.audio_filename, []).append(record)
    passing = {record.audio_filename: record for record in passing_records}

    report = Counter()
    for audio_filename, clip_segments in by_clip.items():
        if len(clip_segments) < 2:
            continue
        clip = passing.get(audio_filename)
        if clip is None or clip.surah_ayah not in references:
            continue
        report["clips_with_waqf"] += 1
        predicted = clip.predicted_phonemes
        before = SHADDA_CONTRAST in attribute_contrasts(
            predicted, references[clip.surah_ayah]
        )
        after = SHADDA_CONTRAST in attribute_contrasts(
            predicted, _normalized_realized_reference(clip_segments)
        )
        report["shadda_before"] += int(before)
        report["shadda_after"] += int(after)
        report["phantom_removed"] += int(before and not after)
    return dict(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passing",
        type=Path,
        required=True,
        help="Passing-subset manifest from tadabur.filter (JSONL).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output offsets manifest (JSONL) of waqf segments.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory to write whole passing clips as 16 kHz mono WAV.",
    )
    parser.add_argument("--dataset", default=DATASET_ID, help="HF dataset id.")
    parser.add_argument(
        "--config-name",
        default=None,
        help="Dataset config name (e.g. 'preview' for fast small-row-group streaming).",
    )
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=DEFAULT_PAUSE_THRESHOLD_S,
        help=f"Inter-word gap (s) that marks a waqf pause (default: {DEFAULT_PAUSE_THRESHOLD_S}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Scan at most this many streamed rows (before the passing-subset filter).",
    )
    args = parser.parse_args()

    passing_records = read_records(args.passing)
    print(f"Loaded {len(passing_records)} passing clips from {args.passing}.")

    records, skips = build_segments(
        passing_records,
        hafs_phonetizer(),
        audio_dir=args.audio_dir,
        dataset_id=args.dataset,
        config_name=args.config_name,
        split=args.split,
        limit=args.limit,
        pause_threshold_s=args.pause_threshold,
    )
    write_segment_manifest(args.out, records)

    clips = len({r.audio_filename for r in records})
    print(
        f"Wrote {len(records)} segments over {clips} clips to {args.out} "
        f"(audio in {args.audio_dir}). Skipped: {dict(skips)}"
    )

    references = load_reference_phonemes()
    report = shadda_contrast_report(passing_records, records, references)
    print(f"Pre-waqf shadda-contrast report: {report}")


if __name__ == "__main__":
    main()
