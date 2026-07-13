"""Model-driven waqf segmentation + per-segment scoring for the P3.5 audit (#6).

Each admitted Tadabur clip is split into **waqf segments** — contiguous word ranges
bounded by the reciter's intra-ayah pauses — and each is labelled with its *realized*
reference (terminal word in waqf form, interior words in wasl). Those segments, not
the whole ayah, are the units that go to the fine-tune (#8), so the human poison
audit (#6) must grade *them*.

This module owns the model pass end to end. For each passing clip staged on local
disk (by :mod:`tadabur.waqf_segments`) it: (1) runs the dedicated recitation VAD
(:mod:`tadabur.vad`) over all clips to find the reciter's waqf pauses, decodes the
*whole* clip once to per-frame phoneme ids, and hands both to
:func:`tadabur.waqf_detect.segment_clip`, which places each VAD pause on a word
boundary (see :mod:`tadabur.waqf_detect` for why the phoneme head's own blank runs
over-split); then (2) for each resulting segment, slices the clip waveform to the
segment span, decodes *that* span, and scores its decode against the segment's realized
reference with the ported ``.balanced`` gate — the same normalization / Smith-Waterman
/ contrast attribution the full-ayah filter uses, only per segment.

Unlike the whole-clip filter it does **not** apply the ``match_ratio`` pass bar — every
segment of an admitted clip is already in the training set, so low-scoring segments are
still emitted (``match_ratio`` and ``contrasts`` are observational, driving the audit's
per-contrast and marginal-band sampling). It does drop five kinds of mislabelled or
unusable segment: a **short segment** — an interior-split piece under ``MIN_SEGMENT_WORDS``
words (a false-pause / lead-in sliver; a whole-ayah single segment is exempt) — a
**degenerate span** (audio under ``MIN_DECODE_SAMPLES`` ≈ 25 ms, too short to feature-
extract), the gate's **repeated-phrase poison** (a decode with an interior insertion run of
``scorer.MAX_INSERTION_RUN`` phonemes), a **boundary mismatch** — a segment whose audio
overruns its assigned reference at an *interior* waqf split by ``MAX_BOUNDARY_TRIM``+
phonemes (a localized repeat straddling the pause corrupts the whole-clip decode→word map,
so the reference is cut on the wrong word) — and an **edge re-cut failure**, where a re-cut
clip edge (:mod:`tadabur.waqf_detect`) still overruns the aligned span by the same margin.
The short-segment and degenerate-span drops are structural and applied before the decode;
the rest need each segment's decode. All are mislabelled or unusable examples, not marginal
ones to audit, and are tallied by reason.

The output is a single segment manifest (JSONL). Its :class:`ManifestRecord` fields
(``audio_filename`` = the per-segment id, ``surah_ayah``, ``match_ratio``,
``ayah_duration_s`` = the segment's own duration, ``reciter_id``, ``contrasts``,
``predicted_phonemes``) let :mod:`tadabur.audit_sampler` draw the worklist directly
via :func:`~tadabur.manifest.read_records`, which ignores the extra per-segment
display fields (``uthmani``, ``raw_reference_phonemes``, ``reference_phonemes``,
``start_s`` / ``end_s`` / ``segment_index``) that :mod:`tadabur.audit_ui` reads to
show the reviewer the segment's realized reference rather than the full ayah's.
Each segment's sliced audio is written to ``--audio-out`` under the same
:func:`~tadabur.audit_sampler.local_audio_path` name the sampler/UI expect, so no
separate HF export step is needed.

Usage:
  python -m tadabur.segment_score \
      --passing passing_subset.jsonl --clips-dir clips/ \
      --out-manifest segment_manifest.jsonl --audio-out segment_audio/ \
      [--min-silence-ms 300] [--min-speech-ms 700] [--boundary-tol 3] \
      [--min-segment-words 3] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import TARGET_SAMPLE_RATE
from .audit_sampler import local_audio_path
from .clip_status import ClipStatus, write_clip_status
from .contrast_attribution import all_contrasts
from .manifest import ManifestRecord, read_records
from .normalization import normalize_phonemes
from .scorer import BALANCED_SCORER, MAX_INSERTION_RUN

# Segmentation boundary QC (a Tadabur segmentation policy, not a Muraja parameter): the
# segment's audio must be covered by its assigned reference. At an *interior* waqf split, a
# leading (non-first segment) or trailing (non-last segment) local-alignment trim of this
# many or more query phonemes means the audio overran the reference boundary — a mis-placed
# or repeat-straddled split (a localized repeat across the pause corrupts the whole-clip
# decode→word map, so the reference is cut a word or two too early). At a **clip edge** the
# outer extent was re-cut to the aligned matched span (:mod:`tadabur.waqf_detect`), so a
# first-segment leading / last-segment trailing trim should be ~0; a survivor means the
# re-cut failed. Either way the segment is mislabelled and dropped (by distinct reasons).
# Threshold matches ``MAX_INSERTION_RUN``: the manifest shows a clean gap (edge trims are 0
# for ~95% of splits, then jump to 7+).
MAX_BOUNDARY_TRIM = 5

# Minimum words an *interior-split* segment must span to be kept (a Tadabur segmentation
# policy, not a Muraja parameter). A waqf pause that peels off a one- or two-word sliver
# yields a micro-segment with too little context to be a useful fine-tune unit or to audit
# reliably — often a false pause (breath) or a short lead-in the edge re-cut could not
# fully trim. Such a piece is dropped (``short_segment``). A clip with **no** interior
# split (one segment spanning the whole ayah) is exempt: a genuinely short ayah is a
# legitimate whole-ayah unit, so it is kept whatever its word count.
MIN_SEGMENT_WORDS = 3

# The SeamlessM4T feature extractor needs at least this many samples to produce a mel
# frame; a shorter slice throws (negative FFT dimensions). A degenerate span this short
# (≈25 ms) is never a real waqf segment — two VAD pauses mapped onto near-adjacent word
# edges, or an edge re-cut that collapsed the span — so it is dropped (``degenerate_span``)
# before decoding rather than allowed to crash the batch.
MIN_DECODE_SAMPLES = 400
from . import vad, waqf_detect
from .waqf_detect import WaqfSpan
from .waqf_segments import (
    SegmentRecord,
    _uthmani_words,
    hafs_phonetizer,
    hafs_word_reference,
)

DEFAULT_BATCH_SIZE = 16


def segment_id(record: SegmentRecord) -> str:
    """A stable, unique per-segment audio id derived from the clip + index.

    ``audio_filename`` alone repeats across a clip's segments; suffixing the
    ``segment_index`` makes the id unique so the sampler can key, and the UI can
    play, each segment independently. Kept ``.wav``-suffixed and human-legible;
    :func:`~tadabur.audit_sampler.local_audio_path` hashes it for the on-disk name.
    """
    stem = Path(record.audio_filename).stem
    return f"{stem}__seg{record.segment_index}.wav"


def slice_segment(
    clip_waveform: np.ndarray, start_s: float, end_s: float
) -> np.ndarray:
    """The ``[start_s, end_s)`` span of a 16 kHz mono clip waveform.

    Sample bounds are clamped to the waveform so a segment whose alignment end
    slightly overshoots the decoded audio still yields a valid (non-empty) slice.
    """
    n = len(clip_waveform)
    start = max(0, min(n, int(round(start_s * TARGET_SAMPLE_RATE))))
    end = max(start, min(n, int(round(end_s * TARGET_SAMPLE_RATE))))
    return np.ascontiguousarray(clip_waveform[start:end], dtype=np.float32)


def _load_clip(clips_dir: Path, audio_filename: str) -> np.ndarray:
    """Read a whole 16 kHz mono clip WAV (written by ``waqf_segments``) as float32."""
    waveform, sample_rate = sf.read(clips_dir / audio_filename, dtype="float32")
    if waveform.ndim > 1:  # defensive: collapse any stray channel dim to mono
        waveform = waveform.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"{audio_filename} is {sample_rate} Hz, expected {TARGET_SAMPLE_RATE} Hz; "
            "clips must be the 16 kHz mono WAVs written by tadabur.waqf_segments."
        )
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _uthmani_segment_text(surah_ayah: str, word_start: int, word_end: int) -> str:
    """The space-joined Uthmani words a segment covers (for the UI display)."""
    words = _uthmani_words(surah_ayah)
    return " ".join(words[word_start:word_end])


def _records_for_spans(
    passing_record: ManifestRecord,
    uthmani_words: list[str],
    spans: tuple[WaqfSpan, ...],
    phonetize,
) -> list[SegmentRecord]:
    """Turn a clip's waqf spans into realized-reference :class:`SegmentRecord`s.

    Each span's realized reference is ``phonetize`` over its space-joined Uthmani
    words — terminal word in waqf form, interior words in wasl. May raise
    ``KeyError`` on the 8 leen-madd-on-sukoon ayat the phonetizer cannot handle
    (``generate_phonemes.FALLBACK_PHONEMES``); the caller skips such a clip.
    """
    return [
        SegmentRecord(
            audio_filename=passing_record.audio_filename,
            surah_ayah=passing_record.surah_ayah,
            reciter_id=passing_record.reciter_id,
            segment_index=index,
            word_start=span.word_start,
            word_end=span.word_end,
            start_s=span.start_s,
            end_s=span.end_s,
            realized_reference_phonemes=phonetize(
                " ".join(uthmani_words[span.word_start : span.word_end])
            ),
        )
        for index, span in enumerate(spans)
    ]


def segment_clips(
    passing_records: list[ManifestRecord],
    clips_dir: Path,
    model,
    phonetize,
    word_reference,
    pauses_by_clip: dict[str, list[tuple[float, float]]],
    *,
    boundary_tol: int = waqf_detect.DEFAULT_BOUNDARY_TOL,
    max_decode_ratio: float = waqf_detect.DEFAULT_MAX_DECODE_RATIO,
    min_align_ratio: float = waqf_detect.DEFAULT_MIN_ALIGN_RATIO,
) -> tuple[list[SegmentRecord], Counter, list[ClipStatus]]:
    """Split every passing clip at its VAD-detected waqf pauses.

    ``pauses_by_clip`` maps each clip's ``audio_filename`` to its ``(start_s, end_s)``
    waqf silence gaps (from :func:`tadabur.vad.compute_clip_pauses`). For each passing
    clip staged under ``clips_dir`` (see ``tadabur.waqf_segments``), decodes the whole
    clip **one at a time** (a batched decode of full clips OOMs — attention is quadratic
    in length) to per-frame phoneme ids, phonetizes the ayah once via ``word_reference``
    into its spaceless phoneme reference + per-word boundaries, and passes those plus the
    clip's pauses to :func:`tadabur.waqf_detect.segment_clip` to place each pause on a
    word boundary. Each resulting word range becomes a
    :class:`~tadabur.waqf_segments.SegmentRecord` with its realized reference.

    A clip :func:`~tadabur.waqf_detect.segment_clip` cannot segment safely
    (``repeated_recitation`` / ``low_alignment``) is tallied and kept whole — one
    segment spanning the ayah — rather than dropped, so it still reaches the audit. A
    clip whose reference cannot be phonetized (the 8-ayah leen-madd gap) is skipped and
    tallied (``phonetizer_unsupported``); a passing clip missing from ``clips_dir`` is
    tallied ``clip_missing``. Returns the records, the skip tally, and a
    :class:`~tadabur.clip_status.ClipStatus` per passing clip — the whole-clip
    eligibility sidecar the windowed-label builder (#25) needs to exclude a clip whose
    segments are unusable, since a kept-whole skip is indistinguishable from a genuine
    no-pause clip in the segment manifest and a missing/unphonetizable clip leaves no
    rows at all.
    """
    ordered = sorted(passing_records, key=lambda r: r.audio_filename)
    records: list[SegmentRecord] = []
    skips: Counter = Counter()
    statuses: list[ClipStatus] = []
    for passing in ordered:
        uthmani_words = _uthmani_words(passing.surah_ayah)
        if not (clips_dir / passing.audio_filename).exists():
            skips["clip_missing"] += 1
            statuses.append(_clip_status(passing, len(uthmani_words), 0.0, "clip_missing"))
            continue
        waveform = _load_clip(clips_dir, passing.audio_filename)
        duration_s = len(waveform) / TARGET_SAMPLE_RATE
        class_ids = list(model.decode(waveform, TARGET_SAMPLE_RATE).class_ids)
        pauses = pauses_by_clip.get(passing.audio_filename, [])
        try:
            reference, boundaries = word_reference(uthmani_words)
            result = waqf_detect.segment_clip(
                class_ids, duration_s, reference, boundaries, pauses,
                boundary_tol=boundary_tol,
                max_decode_ratio=max_decode_ratio, min_align_ratio=min_align_ratio,
            )
            if result.skip is not None:
                skips[result.skip] += 1
                spans = (WaqfSpan(0, len(uthmani_words), 0.0, duration_s),)
            else:
                spans = result.spans
            clip_records = _records_for_spans(
                passing, uthmani_words, spans, phonetize
            )
        except (KeyError, IndexError):
            skips["phonetizer_unsupported"] += 1
            statuses.append(
                _clip_status(passing, len(uthmani_words), duration_s, "phonetizer_unsupported")
            )
            continue
        records.extend(clip_records)
        statuses.append(
            _clip_status(passing, len(uthmani_words), duration_s, result.skip)
        )
    return records, skips, statuses


def _clip_status(
    passing: ManifestRecord, n_words: int, duration_s: float, skip_reason: str | None
) -> ClipStatus:
    """One :class:`ClipStatus` for a passing clip, carrying its eligibility inputs."""
    return ClipStatus(
        audio_filename=passing.audio_filename,
        surah_ayah=passing.surah_ayah,
        reciter_id=passing.reciter_id,
        n_words=n_words,
        duration_s=round(duration_s, 3),
        skip_reason=skip_reason,
    )


def _decode_all(
    model, waveforms: list[np.ndarray], batch_size: int
) -> list[str]:
    """Greedy-CTC-decode every segment waveform, in fixed-size batches, in order."""
    phonemes: list[str] = []
    for start in range(0, len(waveforms), batch_size):
        batch = waveforms[start : start + batch_size]
        for decode in model.decode_batch(batch, TARGET_SAMPLE_RATE):
            phonemes.append(decode.phonemes)
    return phonemes


def score_segments(
    segments: list[SegmentRecord],
    clips_dir: Path,
    model,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_segment_words: int = MIN_SEGMENT_WORDS,
) -> tuple[list[dict], list[SegmentRecord], Counter]:
    """Decode and score every segment; return ``(rows, kept, drops)``.

    Segments are processed in a stable ``(audio_filename, segment_index)`` order so
    the manifest is deterministic. Each surviving segment yields one manifest row (dict)
    carrying the :class:`ManifestRecord` fields the sampler needs plus the per-segment
    display fields the UI reads; the realized reference is normalized **once** here (the
    gate/attribution require a pre-normalized reference — normalization is not
    idempotent). Four kinds of mislabelled or too-thin segment are dropped (not emitted)
    and tallied in the ``drops`` counter, keyed by reason:

    * ``short_segment`` — an *interior-split* segment spanning fewer than
      :data:`MIN_SEGMENT_WORDS` words (a false-pause / lead-in sliver). A whole-ayah
      single segment is exempt (see :data:`MIN_SEGMENT_WORDS`).
    * ``degenerate_span`` — the segment's audio slice is shorter than
      :data:`MIN_DECODE_SAMPLES` (≈25 ms), too short for the feature extractor: two VAD
      pauses landed on near-adjacent word edges, or an edge re-cut collapsed the span.
    * ``repeated_phrase`` — the decode has an interior insertion run of
      :data:`~tadabur.scorer.MAX_INSERTION_RUN` phonemes (the reciter repeated words).
    * ``boundary_mismatch`` — at an *interior* waqf boundary (a leading trim on a non-first
      segment or a trailing trim on a non-last segment) the audio overran its assigned
      reference by :data:`MAX_BOUNDARY_TRIM`+ phonemes, i.e. the split landed on the wrong
      word (typically a localized repeat straddling the pause).
    * ``edge_recut_failed`` — at a re-cut **clip edge** (a first-segment leading trim or a
      last-segment trailing trim) the audio still overran the aligned matched span by
      :data:`MAX_BOUNDARY_TRIM`+ phonemes, i.e. the outer-edge re-cut
      (:func:`~tadabur.waqf_detect.segment_clip`) failed (e.g. a repeat past the decode-ratio
      guard left the edge misaligned).

    The ``short_segment`` and ``degenerate_span`` drops are structural (no decode needed)
    and applied **before** the batched decode, so a droppable segment is neither decoded
    (wasted GPU) nor — for a degenerate slice — allowed to crash the batch; the remaining
    gate-based drops need each segment's decode. A low ``match_ratio`` alone is **not** a
    drop — such a segment stays as an observational passer the audit samples the marginal
    band from. ``kept`` is the surviving segments in row order, so the caller stages audio
    for exactly the emitted rows.
    """
    ordered = sorted(segments, key=lambda s: (s.audio_filename, s.segment_index))

    # The last segment index per clip: a trailing trim on it is a true clip end (benign),
    # not an interior split. A single whole-clip span is both first and last, so neither
    # of its edges is ever boundary-checked.
    last_index: dict[str, int] = {}
    for seg in ordered:
        last_index[seg.audio_filename] = max(
            last_index.get(seg.audio_filename, -1), seg.segment_index
        )

    # A clip's segment count (indices are contiguous 0-based): >1 means the clip was cut at
    # an interior waqf pause, so its short slivers are droppable; ==1 is a whole-ayah span.
    segment_count: dict[str, int] = {af: idx + 1 for af, idx in last_index.items()}

    drops: Counter = Counter()

    # Structural pre-filter (no decode): drop word-count slivers and audio-degenerate spans
    # before the batched decode, so neither wastes GPU nor crashes the batch on a too-short
    # slice. Only the survivors are decoded and gate-scored below.
    clip_cache: dict[str, np.ndarray] = {}
    to_score: list[SegmentRecord] = []
    waveforms: list[np.ndarray] = []
    for seg in ordered:
        if (
            segment_count[seg.audio_filename] > 1
            and seg.word_end - seg.word_start < min_segment_words
        ):
            drops["short_segment"] += 1
            continue
        if seg.audio_filename not in clip_cache:
            clip_cache[seg.audio_filename] = _load_clip(clips_dir, seg.audio_filename)
        waveform = slice_segment(clip_cache[seg.audio_filename], seg.start_s, seg.end_s)
        if len(waveform) < MIN_DECODE_SAMPLES:
            drops["degenerate_span"] += 1
            continue
        to_score.append(seg)
        waveforms.append(waveform)

    predicted = _decode_all(model, waveforms, batch_size)

    rows: list[dict] = []
    kept: list[SegmentRecord] = []
    for seg, decode in zip(to_score, predicted):
        reference = normalize_phonemes(seg.realized_reference_phonemes).normalized
        result = BALANCED_SCORER.gate(decode, reference)
        # A long interior insertion run is a repeated-phrase poison label: reject it
        # from the training manifest (a low match_ratio alone stays — it is still an
        # observational passer the audit samples the marginal band from).
        if result.max_insertion_run >= MAX_INSERTION_RUN:
            drops["repeated_phrase"] += 1
            continue
        # A large trim at a *interior* waqf boundary means the split landed on the wrong
        # word (the segment's audio extends past, or before, its reference): a leading trim
        # on a non-first segment, a trailing trim on a non-last segment.
        is_last = seg.segment_index == last_index[seg.audio_filename]
        interior_trim = max(
            0 if seg.segment_index == 0 else result.leading_trim,
            0 if is_last else result.trailing_trim,
        )
        if interior_trim >= MAX_BOUNDARY_TRIM:
            drops["boundary_mismatch"] += 1
            continue
        # The clip's outer edges were re-cut to the aligned matched span
        # (:func:`~tadabur.waqf_detect.segment_clip`), so a first-segment leading trim or a
        # last-segment trailing trim should now be ~0. A surviving large edge trim means the
        # re-cut failed (e.g. a repeat past the 1.6x decode-ratio guard) — drop it too.
        edge_trim = max(
            result.leading_trim if seg.segment_index == 0 else 0,
            result.trailing_trim if is_last else 0,
        )
        if edge_trim >= MAX_BOUNDARY_TRIM:
            drops["edge_recut_failed"] += 1
            continue
        record = ManifestRecord(
            audio_filename=segment_id(seg),
            surah_ayah=seg.surah_ayah,
            match_ratio=result.match_ratio,
            ayah_duration_s=round(seg.end_s - seg.start_s, 3),
            reciter_id=seg.reciter_id,
            contrasts=BALANCED_SCORER.attribute(decode, reference),
            predicted_phonemes=decode,
        )
        row = asdict(record)
        row["contrasts"] = list(record.contrasts)
        row["clip_audio_filename"] = seg.audio_filename
        row["word_start"] = seg.word_start
        row["word_end"] = seg.word_end
        row["uthmani"] = _uthmani_segment_text(
            seg.surah_ayah, seg.word_start, seg.word_end
        )
        row["raw_reference_phonemes"] = seg.realized_reference_phonemes
        row["reference_phonemes"] = reference
        row["start_s"] = seg.start_s
        row["end_s"] = seg.end_s
        row["segment_index"] = seg.segment_index
        rows.append(row)
        kept.append(seg)
    return rows, kept, drops


def write_segment_manifest(path: Path, rows: list[dict]) -> None:
    """Write scored segment rows as a deterministic, key-sorted JSONL manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stage_segment_audio(
    segments: list[SegmentRecord],
    clips_dir: Path,
    audio_out: Path,
) -> None:
    """Slice each segment's audio and write it under its ``local_audio_path`` name.

    Pre-stages the exact files the sampler's worklist and the UI's ``/audio``
    route reference (keyed on the per-segment id), so the audit needs no separate
    HF audio-export step — the whole clips are already on local disk.
    """
    audio_out.mkdir(parents=True, exist_ok=True)
    clip_cache: dict[str, np.ndarray] = {}
    for seg in segments:
        if seg.audio_filename not in clip_cache:
            clip_cache[seg.audio_filename] = _load_clip(clips_dir, seg.audio_filename)
        waveform = slice_segment(
            clip_cache[seg.audio_filename], seg.start_s, seg.end_s
        )
        out_name = local_audio_path(segment_id(seg))
        sf.write(audio_out / out_name, waveform, TARGET_SAMPLE_RATE, subtype="PCM_16")


def _load_or_compute_pauses(passing, args, torch) -> dict[str, list[tuple[float, float]]]:
    """The VAD waqf pauses per clip, from ``--pauses-cache`` if present else computed.

    The VAD pass is the run's slow stage (~1 clip/s); caching its output as JSON lets a
    re-run (e.g. after a later-stage fix) skip it. The cache is keyed by ``audio_filename``
    with ``[start_s, end_s]`` pause lists; a cache missing any passing clip is treated as
    stale and recomputed so a partial file is never silently trusted.
    """
    cache = args.pauses_cache
    if cache is not None and cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        needed = {p.audio_filename for p in passing}
        if needed <= set(raw):
            print(f"Loaded VAD pauses for {len(raw)} clips from cache {cache}.")
            return {af: [tuple(p) for p in spans] for af, spans in raw.items()}
        print(f"Cache {cache} is missing {len(needed - set(raw))} clips; recomputing VAD.")

    pauses_by_clip = vad.compute_clip_pauses(
        passing, args.clips_dir,
        device=torch.device(args.device), dtype=getattr(torch, args.vad_dtype),
        min_silence_ms=args.min_silence_ms, min_speech_ms=args.min_speech_ms,
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(
                {af: [list(p) for p in spans] for af, spans in pauses_by_clip.items()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Wrote VAD pauses for {len(pauses_by_clip)} clips to cache {cache}.")
    return pauses_by_clip


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--passing", type=Path, required=True,
        help="Passing-subset manifest from tadabur.filter (JSONL).",
    )
    parser.add_argument(
        "--clips-dir", type=Path, required=True,
        help="Directory of whole 16 kHz mono clip WAVs (tadabur.waqf_segments --audio-dir).",
    )
    parser.add_argument(
        "--out-manifest", type=Path, required=True,
        help="Output scored segment manifest (JSONL) for the sampler + UI.",
    )
    parser.add_argument(
        "--out-clip-status", type=Path, default=None,
        help=(
            "Output per-clip status sidecar (JSONL) for the windowed-label builder "
            "(#25); defaults to <out-manifest>.clip_status.jsonl."
        ),
    )
    parser.add_argument(
        "--audio-out", type=Path, required=True,
        help="Directory to write each segment's sliced audio for listening.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Segments per scoring decode batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--min-silence-ms", type=float, default=vad.DEFAULT_MIN_SILENCE_MS,
        help=(
            "Min silence (ms) the recitation VAD counts as a waqf "
            f"(default: {vad.DEFAULT_MIN_SILENCE_MS:g})."
        ),
    )
    parser.add_argument(
        "--min-speech-ms", type=float, default=vad.DEFAULT_MIN_SPEECH_MS,
        help=(
            "Min speech (ms) between pauses the VAD keeps as a segment "
            f"(default: {vad.DEFAULT_MIN_SPEECH_MS:g})."
        ),
    )
    parser.add_argument(
        "--boundary-tol", type=int, default=waqf_detect.DEFAULT_BOUNDARY_TOL,
        help=(
            "Max phonemes a pause may sit from a word edge to split there; farther "
            f"is a mid-word stop (default: {waqf_detect.DEFAULT_BOUNDARY_TOL})."
        ),
    )
    parser.add_argument(
        "--min-segment-words", type=int, default=MIN_SEGMENT_WORDS,
        help=(
            "Drop an interior-split segment spanning fewer than this many words "
            f"(a whole-ayah single segment is exempt; default: {MIN_SEGMENT_WORDS})."
        ),
    )
    parser.add_argument(
        "--device", default="cuda", help="Torch device for the model (default: cuda).",
    )
    parser.add_argument(
        "--vad-dtype", default="bfloat16",
        help="Torch dtype for the VAD forward (default: bfloat16).",
    )
    parser.add_argument(
        "--pauses-cache", type=Path, default=None,
        help=(
            "Optional JSON cache of the VAD waqf pauses (audio_filename -> [[start,end],"
            " ...]). Loaded if it exists (skipping the ~1 clip/s VAD pass), else computed "
            "and written. Lets a re-run reuse the slow VAD stage."
        ),
    )
    args = parser.parse_args()

    passing = read_records(args.passing)
    print(f"Loaded {len(passing)} passing clips from {args.passing}.")

    import torch

    from .inference import MuaalemPhonemeModel

    pauses_by_clip = _load_or_compute_pauses(passing, args, torch)
    total_pauses = sum(len(p) for p in pauses_by_clip.values())
    print(
        f"VAD found {total_pauses} interior waqf pauses across "
        f"{len(pauses_by_clip)} clips."
    )

    model = MuaalemPhonemeModel.load(device=args.device)
    segments, skips, clip_statuses = segment_clips(
        passing, args.clips_dir, model, hafs_phonetizer(), hafs_word_reference(),
        pauses_by_clip, boundary_tol=args.boundary_tol,
    )
    clips = len({s.audio_filename for s in segments})
    print(
        f"Segmented {clips} clips into {len(segments)} waqf segments. "
        f"Skips/fallbacks: {dict(skips)}"
    )

    clip_status_path = args.out_clip_status or args.out_manifest.with_suffix(
        args.out_manifest.suffix + ".clip_status.jsonl"
    )
    write_clip_status(clip_status_path, clip_statuses)
    print(f"Wrote {len(clip_statuses)} per-clip statuses to {clip_status_path}.")

    rows, kept, drops = score_segments(
        segments, args.clips_dir, model, args.batch_size,
        min_segment_words=args.min_segment_words,
    )
    write_segment_manifest(args.out_manifest, rows)
    stage_segment_audio(kept, args.clips_dir, args.audio_out)

    contrasted = sum(1 for r in rows if r["contrasts"])
    buckets = {c: sum(1 for r in rows if c in r["contrasts"]) for c in all_contrasts()}
    print(
        f"Scored {len(rows)} segments ({contrasted} with contrasts; "
        f"rejected {drops['short_segment']} as short segment, "
        f"{drops['degenerate_span']} as degenerate span, "
        f"{drops['repeated_phrase']} as repeated-phrase poison, "
        f"{drops['boundary_mismatch']} as boundary mismatch, "
        f"{drops['edge_recut_failed']} as edge re-cut failure). Wrote manifest "
        f"to {args.out_manifest} and segment audio to {args.audio_out}."
    )
    print("Per-contrast segment counts: " + ", ".join(
        f"{c}={n}" for c, n in buckets.items() if n
    ))


if __name__ == "__main__":
    main()
