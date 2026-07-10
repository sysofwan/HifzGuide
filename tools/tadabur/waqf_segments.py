"""Stage passing Tadabur clips locally, and the realized-reference label vocabulary.

The Tadabur filter (``tadabur.filter``) admits a clip on a *single* full-ayah
reference — one canonical guess at what was recited. But a reciter who **stops**
(makes waqf) partway through an ayah realizes the words differently from one who
recites it continuously (wasl): the word before the stop drops its final haraka and
loses the cross-word gemination/idgham it would carry in continuation. Labelling
every clip with the full-ayah wasl reference therefore injects *phantom* pre-waqf
gemination mismatches into the fine-tune data. Splitting each clip at its intra-ayah
pauses and phonetizing each segment on its own removes them — the segment's terminal
word lands in waqf form (``quran_phonetizer``'s CleanEnd op) and the interior words
in wasl, which is exactly the realized reference.

*Where* those pauses are is decided by the model, not by timestamps: the forced
alignment Tadabur ships absorbs waqf silence into adjacent word spans, so the
inter-word gap is ~0 even at a real stop (see :mod:`tadabur.waqf_detect`). The
segmentation + scoring therefore lives in :mod:`tadabur.segment_score`, which needs a
GPU. This module is the torch-free half: it **stages** each passing clip as a whole
16 kHz mono WAV on local disk (the full Tadabur source is streamed, never landed) so
the model pass can decode it, and it owns the shared realized-reference label
vocabulary — :class:`SegmentRecord`, :func:`hafs_phonetizer`, :func:`_uthmani_words`.

A full build must locate every passing clip in the stream: one it cannot find raises
rather than stage a partial clip set (see :func:`_require_all_streamed`), while a
``--limit`` smoke run — which may legitimately stop early — records the shortfall as
the ``missing_due_to_limit`` tally instead.

Usage:
  python -m tadabur.waqf_segments --passing passing_subset.jsonl \
      --audio-dir clips/ [--config-name preview] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

_TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import generate_phonemes  # noqa: E402  (tools/ sibling module)

from .audio import TARGET_SAMPLE_RATE, decode_to_mono_16k
from .dataset_source import AUDIO_COLUMN, DATASET_ID, resolve_audio_filename
from .manifest import ManifestRecord, read_records

# A callable that turns Uthmani text into its phoneme string (a seam so the
# segmentation logic in tadabur.segment_score is testable without quran-transcript).
Phonetizer = Callable[[str], str]
# A callable giving an ayah's spaceless phoneme reference + per-word phoneme offsets
# (len == n_words + 1) — the alignment reference tadabur.waqf_detect.segment_clip uses.
WordReference = Callable[[list[str]], "tuple[str, list[int]]"]


@dataclass(frozen=True)
class SegmentRecord:
    """One waqf segment: a realized-reference view into a whole passing clip.

    ``audio_filename`` names the whole 16 kHz-mono clip kept on local disk;
    ``(start_s, end_s)`` is the segment's span within it (sliced at collate time,
    never re-materialized). ``segment_index`` orders the segments within the clip
    and, with ``audio_filename``, is the manifest's stable key. ``word_start`` /
    ``word_end`` are the half-open Uthmani word range the segment covers.
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


def _spaceless_word_offsets(
    phonemes: str, mappings: list, uthmani_words: list[str]
) -> tuple[str, list[int]]:
    """Spaceless phoneme reference + per-word phoneme offsets from a phonetizer run.

    ``mappings[i].pos[0]`` is the offset in ``phonemes`` the i-th input character maps
    to; using the input-char offset of each Uthmani word start recovers each word's
    phoneme boundary even where the phonetizer **merges words at a wasl** (so there is
    no separating space to split on). The spaces the phonetizer keeps at un-merged
    boundaries are then stripped, and the offsets remapped, to the spaceless string
    the (space-free) model decode is aligned against. ``boundaries`` has
    ``len(uthmani_words) + 1`` entries (word ``j`` starts at ``boundaries[j]``).
    """
    input_starts: list[int] = []
    cursor = 0
    for word in uthmani_words:
        input_starts.append(cursor)
        cursor += len(word) + 1  # + the joining space
    spaced_offsets = [mappings[s].pos[0] for s in input_starts] + [len(phonemes)]

    spaceless: list[str] = []
    remap: list[int] = []
    kept = 0
    for char in phonemes:
        remap.append(kept)
        if not char.isspace():
            spaceless.append(char)
            kept += 1
    remap.append(kept)  # sentinel for the len(phonemes) end offset
    return "".join(spaceless), [remap[offset] for offset in spaced_offsets]


def hafs_word_reference() -> WordReference:
    """A callable giving an ayah's spaceless phoneme reference + per-word offsets.

    Phonetizes the whole ayah once with the Hafs moshaf (single words fail the
    phonetizer standalone) and derives per-word phoneme boundaries from the
    phonetizer's char-level ``mappings`` (:func:`_spaceless_word_offsets`) — robust to
    wasl word-merges, unlike splitting the phonetic output on spaces. Injected into
    :func:`tadabur.waqf_detect.segment_clip` as its alignment ``reference`` /
    ``boundaries``. Raises ``KeyError`` / ``IndexError`` on the ayat quran_phonetizer
    cannot handle (leen madd on a final sukoon), which the caller tallies as
    ``phonetizer_unsupported``.
    """
    from quran_transcript import quran_phonetizer
    from quran_transcript.phonetics.moshaf_attributes import MoshafAttributes

    moshaf = MoshafAttributes(**generate_phonemes.HAFS_MOSHAF)

    def compute(uthmani_words: list[str]) -> tuple[str, list[int]]:
        out = quran_phonetizer(" ".join(uthmani_words), moshaf)
        return _spaceless_word_offsets(out.phonemes, out.mappings, uthmani_words)

    return compute


def _uthmani_words(surah_ayah: str) -> list[str]:
    """The Uthmani words of ``"surah:ayah"`` (1-indexed), via quran-transcript."""
    from quran_transcript import Aya

    surah, ayah = (int(part) for part in surah_ayah.split(":"))
    return list(Aya(surah, ayah).get().uthmani_words)


def _save_local_clip(audio_dir: Path, audio_filename: str, waveform: np.ndarray) -> None:
    """Write an already-decoded 16 kHz mono clip under ``audio_dir`` as PCM_16 WAV."""
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


def _require_all_streamed(
    requested: set[str], found: set[str], dataset_id: str, config_name: str | None, split: str
) -> None:
    """Raise if any passing clip never turned up in a full (unlimited) stream.

    A silently-partial clip set is a data-integrity failure for the label source: a
    stale passing subset or the wrong ``--config-name``/``--split`` would drop clips
    (and their labels) with no error. Listing the misses points the operator at the
    cause. Only called for full builds; ``--limit`` runs surface their shortfall as
    ``missing_due_to_limit`` instead (see :func:`stage_clips`).
    """
    missing = sorted(requested - found)
    if missing:
        preview = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise ValueError(
            f"{len(missing)} passing clip(s) were not found in dataset "
            f"{dataset_id!r} config {config_name!r} split {split!r}; the staged clip "
            f"set would be partial: {preview}{more}. Check "
            f"--dataset/--config-name/--split match the passing subset, or that the "
            f"passing manifest is not stale."
        )


def stage_clips(
    passing_records: list[ManifestRecord],
    audio_dir: Path,
    dataset_id: str = DATASET_ID,
    config_name: str | None = None,
    split: str = "train",
    limit: int | None = None,
) -> Counter:
    """Stream the passing subset and save each clip's whole 16 kHz-mono waveform.

    For each passing clip found in the stream, decodes its audio to 16 kHz mono and
    writes it under ``audio_dir`` as a PCM_16 WAV — the input the model pass
    (:mod:`tadabur.segment_score`) reads. Returns a :class:`~collections.Counter` of
    any clips missed; a full build (``limit`` is ``None``) that cannot find every
    passing clip raises rather than stage a partial set (see
    :func:`_require_all_streamed`), while a ``--limit`` smoke run records its
    shortfall as ``missing_due_to_limit``.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    passing = {record.audio_filename: record for record in passing_records}

    skips: Counter = Counter()
    found: set[str] = set()
    for row, record in _stream_passing_rows(
        passing, dataset_id, config_name, split, limit
    ):
        found.add(record.audio_filename)
        waveform = decode_to_mono_16k(row[AUDIO_COLUMN]["bytes"])
        _save_local_clip(audio_dir, record.audio_filename, waveform)

    missing = set(passing) - found
    if missing:
        if limit is None:
            _require_all_streamed(set(passing), found, dataset_id, config_name, split)
        skips["missing_due_to_limit"] = len(missing)
    return skips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passing",
        type=Path,
        required=True,
        help="Passing-subset manifest from tadabur.filter (JSONL).",
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
        "--limit",
        type=int,
        default=None,
        help="Scan at most this many streamed rows (before the passing-subset filter).",
    )
    args = parser.parse_args()

    passing_records = read_records(args.passing)
    print(f"Loaded {len(passing_records)} passing clips from {args.passing}.")

    skips = stage_clips(
        passing_records,
        audio_dir=args.audio_dir,
        dataset_id=args.dataset,
        config_name=args.config_name,
        split=args.split,
        limit=args.limit,
    )
    staged = len(passing_records) - sum(skips.values())
    print(f"Staged {staged} clips to {args.audio_dir}. Missed: {dict(skips)}")


if __name__ == "__main__":
    main()
