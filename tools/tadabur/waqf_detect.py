"""Model-driven waqf detection: split a clip at pauses the *model* hears.

The forced alignment Tadabur ships (``metadata.word_alignments``) misses many real
waqf pauses: the aligner absorbs the silence into an adjacent word's span, so the
inter-word *gap* is ~0 even though the reciter clearly stopped (verified on
3:159, 40:20, …). Detecting waqf from those gaps therefore both over-splits on madd
elongations (voiced, not a stop) and under-splits on absorbed pauses.

This module detects pauses from the Muaalem phoneme head instead. Greedy CTC emits
the **blank** token (``PHONEME_PAD_ID``) on silence, so a run of blank frames is a
pause the model itself heard — independent of the aligner, and not fooled by a madd
(which is voiced and decodes to real phonemes). To split at a *word* boundary (so
each segment can be phonetized into its waqf/wasl realized form), each blank-run is
mapped to a word edge: the model's decoded phonemes are Smith-Waterman-aligned to the
ayah's per-word ``reference``, and the run is placed at the reference position of the
last phoneme before it. A run that lands within a few phonemes of a word edge is a
waqf; one that lands mid-word is a stop-consonant closure (qalqala on ق/ط, the hamza
in شَيء) and is **not** split. This is all pure logic over a frame-id sequence plus
the injected ``reference`` / per-word ``boundaries`` — the model forward and the
phonetizer both live in :mod:`tadabur.segment_score` /
:mod:`tadabur.waqf_segments`, so the segmentation is unit-testable with synthetic
frame ids and a synthetic reference.

Two whole-clip cases are skipped rather than mis-segmented: a clip whose decode is
far longer than the ayah reference (``repeated_recitation`` — the reciter repeated
the ayah, breaking the one-pass word map), and one whose best alignment barely
matches the ayah at all (``low_alignment``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .phoneme_vocab import PHONEME_ID_TO_CHAR, PHONEME_PAD_ID
from .smith_waterman import smith_waterman

# A blank-run at least this long (seconds) is a waqf candidate. Shorter blank runs
# are stop-consonant closures / inter-phoneme silence, not stops.
DEFAULT_MIN_PAUSE_S = 0.35
# A mapped pause must land within this many reference phonemes of a word edge to be a
# waqf; farther in is mid-word (a stop closure), so it is not split.
DEFAULT_BOUNDARY_TOL = 3
# Skip a clip whose decode is more than this multiple of the reference length: the
# reciter repeated the ayah, so the one-pass decode↔word map is invalid.
DEFAULT_MAX_DECODE_RATIO = 1.6
# Skip a clip whose best local alignment score is below this fraction of the
# reference length: the decode does not match this ayah (bad clip / wrong label).
DEFAULT_MIN_ALIGN_RATIO = 0.45


@dataclass(frozen=True)
class WaqfSpan:
    """One waqf segment: a half-open Uthmani word range and its clip time span."""

    word_start: int
    word_end: int
    start_s: float
    end_s: float


@dataclass(frozen=True)
class SegmentationResult:
    """A clip's waqf spans, or a ``skip`` reason when it cannot be segmented safely."""

    spans: tuple[WaqfSpan, ...]
    skip: str | None = None


def find_blank_runs(
    class_ids: list[int],
    seconds_per_frame: float,
    min_pause_s: float = DEFAULT_MIN_PAUSE_S,
) -> list[tuple[float, float]]:
    """Runs of blank frames (in seconds) that are at least ``min_pause_s`` long.

    Each maximal run of consecutive blank (``PHONEME_PAD_ID``) frames reaching
    ``min_pause_s`` is returned as a ``(start_s, end_s)`` pair. Shorter blank runs —
    the inter-phoneme / stop-closure silences — are dropped. Two long runs from one
    breath (a decode blip between them) need no merging here: they map to the same
    word edge and :func:`segment_clip` keeps one split per word edge.
    """
    min_frames = min_pause_s / seconds_per_frame if seconds_per_frame else 0
    runs: list[tuple[float, float]] = []
    start: int | None = None
    for k, cid in enumerate(list(class_ids) + [1]):  # sentinel non-blank closes a run
        if cid == PHONEME_PAD_ID and start is None:
            start = k
        elif cid != PHONEME_PAD_ID and start is not None:
            if (k - start) >= min_frames:
                runs.append((start * seconds_per_frame, k * seconds_per_frame))
            start = None
    return runs


def collapse_with_times(
    class_ids: list[int], seconds_per_frame: float
) -> tuple[str, list[float]]:
    """Greedy-CTC-collapse ids to a phoneme string plus each phoneme's onset time.

    Mirrors :func:`tadabur.phoneme_vocab.greedy_ctc_decode` (collapse repeats, drop
    blanks) but also returns, for every emitted phoneme, the time of the frame it
    first appears on — so a blank-run's position can be located in the decode.
    """
    chars: list[str] = []
    times: list[float] = []
    previous: int | None = None
    for k, cid in enumerate(class_ids):
        if cid != previous:
            previous = cid
            if cid != PHONEME_PAD_ID:
                chars.append(PHONEME_ID_TO_CHAR[cid])
                times.append(k * seconds_per_frame)
    return "".join(chars), times


def _map_run_to_word(
    run_start_s: float,
    decode_times: list[float],
    query_to_ref: list[tuple[int, int]],
    boundaries: list[int],
    boundary_tol: int,
) -> int | None:
    """The interior word index a blank-run splits at, or ``None`` if it is mid-word.

    Counts the decoded phonemes emitted before the run, maps that query position to
    its reference position through the alignment, and returns the nearest word edge
    only if it is within ``boundary_tol`` phonemes (else the run is a mid-word stop).
    Word ``0`` and the final edge are not interior splits, so they map to ``None``.
    """
    if not query_to_ref:
        return None
    q_star = sum(1 for t in decode_times if t < run_start_s)
    ref_pos = min(query_to_ref, key=lambda qr: abs(qr[0] - q_star))[1]
    j = min(range(len(boundaries)), key=lambda k: abs(boundaries[k] - ref_pos))
    if abs(boundaries[j] - ref_pos) <= boundary_tol and 0 < j < len(boundaries) - 1:
        return j
    return None


def segment_clip(
    class_ids: list[int],
    clip_duration_s: float,
    reference: str,
    boundaries: list[int],
    *,
    min_pause_s: float = DEFAULT_MIN_PAUSE_S,
    boundary_tol: int = DEFAULT_BOUNDARY_TOL,
    max_decode_ratio: float = DEFAULT_MAX_DECODE_RATIO,
    min_align_ratio: float = DEFAULT_MIN_ALIGN_RATIO,
) -> SegmentationResult:
    """Split one clip into waqf spans from its per-frame phoneme ids.

    ``reference`` is the ayah's spaceless phoneme string and ``boundaries`` its per-
    word phoneme offsets (``len == n_words + 1``, computed phonetizer-side so wasl
    word-merges are handled — see ``tadabur.waqf_segments.hafs_word_reference``).
    Returns a single whole-ayah span when the model heard no interior waqf, one span
    per word-range between the confirmed pauses otherwise, or a
    :class:`SegmentationResult` with a ``skip`` reason for a clip that cannot be
    segmented safely (``repeated_recitation`` / ``low_alignment``). Word ranges and
    time spans both come from the pauses: a split at word ``j`` cuts the words there
    and the clip time at the pause itself.
    """
    n_words = len(boundaries) - 1
    whole = SegmentationResult((WaqfSpan(0, n_words, 0.0, clip_duration_s),))
    if not class_ids or n_words <= 0 or not reference:
        return whole

    seconds_per_frame = clip_duration_s / len(class_ids)
    query, decode_times = collapse_with_times(class_ids, seconds_per_frame)
    if len(query) > max_decode_ratio * len(reference):
        return SegmentationResult((), skip="repeated_recitation")

    alignment = smith_waterman(query, reference)
    if alignment.score < min_align_ratio * len(reference):
        return SegmentationResult((), skip="low_alignment")
    query_to_ref = sorted(
        (q, alignment.ref_start + i)
        for i, q in enumerate(alignment.ref_to_query)
        if q >= 0
    )

    runs = find_blank_runs(class_ids, seconds_per_frame, min_pause_s)
    cuts: dict[int, tuple[float, float]] = {}
    for run_start_s, run_end_s in runs:
        word = _map_run_to_word(
            run_start_s, decode_times, query_to_ref, boundaries, boundary_tol
        )
        if word is not None:
            cuts[word] = (run_start_s, run_end_s)
    if not cuts:
        return whole

    spans: list[WaqfSpan] = []
    prev_word, prev_time = 0, 0.0
    for word in sorted(cuts):
        run_start_s, run_end_s = cuts[word]
        spans.append(WaqfSpan(prev_word, word, prev_time, run_start_s))
        prev_word, prev_time = word, run_end_s
    spans.append(WaqfSpan(prev_word, n_words, prev_time, clip_duration_s))
    return SegmentationResult(tuple(spans))
