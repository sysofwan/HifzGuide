"""Map a clip's waqf pauses to word boundaries, so each segment is phonetizable.

The reciter's pauses come from the dedicated recitation VAD (:mod:`tadabur.vad`) as
``(start_s, end_s)`` silence gaps. Detecting pauses from the Muaalem phoneme head's CTC
blank runs (the earlier approach) was **over-eager** — a blank threshold fired on every
inter-word micro-silence and cut tiny one-word segments; the phoneme head transcribes,
it does not tell a genuine waqf from a breath. The VAD is fine-tuned for exactly that
distinction (see :mod:`tadabur.vad`).

This module turns a pause *time* into a *word* split (so each segment can be phonetized
into its waqf/wasl realized form). The clip's decoded phonemes are Smith-Waterman-
aligned to the ayah's per-word ``reference``, and each pause is placed at the reference
position of the last phoneme before it. A pause that lands within a few phonemes of a
word edge is a waqf; one that lands mid-word is dropped (a real waqf only ever falls on
a word end). This is pure logic over a frame-id sequence, the injected pauses, and the
``reference`` / per-word ``boundaries`` — the VAD, the model forward, and the phonetizer
all live in :mod:`tadabur.vad` / :mod:`tadabur.segment_score` /
:mod:`tadabur.waqf_segments`, so the segmentation is unit-testable with synthetic frame
ids, pauses, and reference.

A **re-read** (the reciter recites some words, stops, then repeats them and carries on) is
segmented, not skipped. The clip's decode is split at the VAD pauses into *chunks*, each
chunk is locally aligned to the reference on its own, and a chunk that aligns *backward*
(re-covers words an earlier chunk already recited) marks a re-read seam. The clip is then
cut there into two time-consecutive segments — "read until the waqf" and "re-read point
until the end" — that are disjoint in clip time but overlap in Uthmani words (the seam is
a genuine waqf: the reciter really did stop before repeating). This pause-anchored
per-chunk alignment also fixes the plain wrong-word / timing-offset failures a single
whole-clip alignment produced, because each pause is mapped to a word by where its own
neighbouring chunks align rather than by a global running phoneme count a repeat inflates.

Only two whole-clip cases are still skipped rather than segmented: a clip whose best
alignment barely matches the ayah at all (``low_alignment``), and a gross repeat with no
usable pause seam to split on — a segment whose decode still overruns its reference by more
than :data:`DEFAULT_MAX_DECODE_RATIO` (``repeated_recitation``).

The clip's **outer** edges are re-cut to the whole-clip alignment's matched span. Tadabur
clips carry the previous ayah's tail as lead-in (and sometimes a trailing next word or
takbir) that no *interior* waqf pause ever trims — a single-segment clip would otherwise
keep it whole and mislabel it (see #20). The matched span's onset/offset locate the first
and last decoded phonemes that align to this ayah's reference, so the first segment's
``start_s`` and the last segment's ``end_s`` are moved there; interior VAD-defined
boundaries are untouched. This is a segmentation-*extent* change only — the parity-locked
Muraja gate is not touched (ADR-0001).
"""

from __future__ import annotations

from dataclasses import dataclass

from .phoneme_vocab import PHONEME_ID_TO_CHAR, PHONEME_PAD_ID
from .smith_waterman import smith_waterman

# A mapped pause must land within this many reference phonemes of a word edge to be a
# waqf split; farther in is mid-word (a stop closure), so the chunks it separates are
# merged rather than cut.
DEFAULT_BOUNDARY_TOL = 3
# Skip a *segment* whose decode is more than this multiple of its reference word span: a
# repeat with no pause seam to split it on, so it cannot be cut into two single-pass clips.
DEFAULT_MAX_DECODE_RATIO = 1.6
# Skip a clip whose best local alignment score is below this fraction of the
# reference length: the decode does not match this ayah (bad clip / wrong label).
DEFAULT_MIN_ALIGN_RATIO = 0.45
# Outward safety pad (seconds) applied to a re-cut outer edge. The CTC spikes locating
# the matched span fire late, so cutting exactly at the aligned onset frame would chop
# the onset consonant's attack — self-defeating for the fine-tune. Pad both edges out by
# this much and clamp back into the clip. ~50ms ≈ one Muaalem logit frame.
EDGE_RECUT_PAD_S = 0.05


# Pause classes, mirrored by the audit candidate layer (``tadabur.waqf_candidates``).
WAQF = "waqf"
RE_READ = "re_read"
MID_WORD_CLOSURE = "mid_word_closure"
# A pause in a clip that was kept whole / could not be segmented: its alignment is
# untrusted, so the pause is emitted with ``word_index=None`` (the candidate layer then
# interpolates rather than trusting a phoneme-aligned word).
UNPLACED = "unplaced"


@dataclass(frozen=True)
class WaqfSpan:
    """One waqf segment: a half-open Uthmani word range and its clip time span."""

    word_start: int
    word_end: int
    start_s: float
    end_s: float


@dataclass(frozen=True)
class PauseAttribution:
    """One VAD pause placed on a word by the **phoneme alignment**, not by time.

    ``(start_s, end_s)`` is the VAD silence gap. ``kind`` is how
    :func:`segment_clip` classified it — :data:`WAQF` / :data:`RE_READ` (a confirmed
    stop that became a segment split), :data:`MID_WORD_CLOSURE` (a stop the decode
    placed inside a word, so the chunks it separates were merged), or :data:`UNPLACED`
    (a pause in a clip kept whole / not segmented, always with ``word_index=None``).
    ``word_index`` is the
    Uthmani word the pause falls **after**: the last word the decode had fully covered
    when it reached the pause (from the aligning chunk's matched-span end), or ``None``
    when neither neighbouring chunk aligned reliably enough to place it. This is the same
    phoneme-alignment signal the runtime model has (it never sees forced-alignment
    timing), so the audit label matches what inference can reproduce.
    """

    start_s: float
    end_s: float
    kind: str
    word_index: int | None


@dataclass(frozen=True)
class SegmentationResult:
    """A clip's waqf spans, or a ``skip`` reason when it cannot be segmented safely.

    ``re_reads`` counts the re-read seams the segmentation cut the clip at (a chunk that
    aligned backward over words an earlier chunk already covered). It is 0 for an ordinary
    clip; a positive value flags a clip whose segments have overlapping Uthmani word ranges
    and is surfaced for manual review.

    ``pauses`` carries every VAD pause's phoneme-aligned attribution
    (:class:`PauseAttribution`), in clip-time order, so the audit candidate layer can place
    a mid-word closure on the word the decode actually reached rather than by interpolating
    time across the segment. ``segment_clip`` emits one attribution per input pause even
    when it keeps the clip whole or skips it (as :data:`UNPLACED`, ``word_index=None``), so
    the sidecar lists every processed clip; ``pauses`` is empty only when the clip had no
    VAD pauses at all.
    """

    spans: tuple[WaqfSpan, ...]
    skip: str | None = None
    re_reads: int = 0
    pauses: tuple[PauseAttribution, ...] = ()


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


# A chunk whose best local-alignment score falls below this fraction of its own length is
# unreliable (mostly bleed/noise); it never anchors a split and is merged with a neighbour.
_MIN_CHUNK_ALIGN_RATIO = 0.3
# Sentinel distance for an unreliable chunk edge, so it can never satisfy ``boundary_tol``.
_FAR = 1 << 30


@dataclass(frozen=True)
class _ChunkAlign:
    """One inter-pause chunk of the decode, aligned to the reference on its own.

    ``q_lo``/``q_hi`` are the chunk's half-open query (decoded-phoneme) index range and
    ``t_lo``/``t_hi`` its clip-time span. ``start_word``/``end_word`` are the reference
    word edges its matched span snaps to, with ``start_dist``/``end_dist`` the phoneme
    distance of each snap (``_FAR`` when unreliable). ``ref_start``/``ref_end`` are the raw
    (un-snapped) reference phoneme offsets of the matched span, used to attribute pauses
    to the last fully-covered word and to reject a resume that lands left of its word edge.
    ``reliable`` is False for an empty,
    barely-aligning, or sub-word chunk, which then anchors no split (it is merged with a
    neighbour and never contributes a fabricated word range).

    ``ref_kinds[i]`` is the Smith-Waterman outcome (``"match"`` / ``"mismatch"`` / ``"gap"``)
    for reference position ``ref_start + i``, retained so :func:`_word_supported` can ask
    whether this chunk's own decode independently covered a given word — the signal that
    tells a genuine re-read apart from a phantom over-read.
    """

    q_lo: int
    q_hi: int
    t_lo: float
    t_hi: float
    start_word: int
    start_dist: int
    end_word: int
    end_dist: int
    reliable: bool
    ref_start: int = 0
    ref_end: int = 0
    ref_kinds: tuple[str, ...] = ()


def _nearest_boundary(boundaries: list[int], pos: int) -> tuple[int, int]:
    """The word-edge index nearest reference phoneme ``pos``, and its phoneme distance."""
    j = min(range(len(boundaries)), key=lambda k: abs(boundaries[k] - pos))
    return j, abs(boundaries[j] - pos)


def _unplaced_attributions(
    pauses: list[tuple[float, float]]
) -> tuple[PauseAttribution, ...]:
    """Explicit ``None`` attributions for a clip kept whole / not segmented.

    A skipped or un-decodable clip is still emitted as one whole-clip segment and may
    contain interior VAD pauses, but its alignment is untrusted, so each pause is marked
    :data:`UNPLACED` with ``word_index=None``. Emitting these upholds the invariant that
    ``segment_clip`` returns one attribution per input pause, so the audit sidecar lists
    every processed clip: a clip *absent* from it then unambiguously means a stale /
    mismatched artifact, while these explicit ``None`` pauses fall back to interpolation
    rather than raising.
    """
    return tuple(PauseAttribution(start_s, end_s, UNPLACED, None) for start_s, end_s in pauses)


def _completed_word(boundaries: list[int], ref_pos: int) -> int:
    """The last Uthmani word fully covered by reference phoneme ``ref_pos``.

    Counts how many word ends (``boundaries[1:]``) sit at or before ``ref_pos``; that
    many words are complete, so the last one's index is the count minus one. Floored at
    ``-1`` for a position before the first word end (caller clamps into the word range).
    """
    return sum(1 for b in boundaries[1:] if b <= ref_pos) - 1


# A word counts as covered by a chunk when at least this fraction of its reference
# phonemes aligned diagonally (matched or substituted) in that chunk's own local
# alignment. Prefix-anchored, so a chunk that only over-snaps into a word's first
# phoneme (a phantom over-read) does not clear the bar, while a real utterance that
# tail-dropped its last phoneme still does.
_MIN_WORD_SUPPORT_COVERAGE = 0.55


def _word_supported(chunk: _ChunkAlign, boundaries: list[int], word: int) -> bool:
    """True if ``chunk``'s own decode independently recited Uthmani ``word``.

    Looks only at this chunk's local alignment (not the whole-clip pass): over the word's
    reference span it requires the aligned-diagonal phonemes to be prefix-anchored (start
    within the first phoneme), include at least one exact match, and cover at least
    :data:`_MIN_WORD_SUPPORT_COVERAGE` of the word (and ``min(2, len)`` phonemes). This is
    the runtime-faithful signal that separates a genuine re-read — the earlier chunk truly
    re-uttered the overlap word — from a phantom over-read, where the earlier chunk's
    matched span merely over-snapped its end past where the reciter actually stopped.
    """
    lo, hi = boundaries[word], boundaries[word + 1]
    length = hi - lo
    if length <= 0:
        return False
    diagonal = exact = 0
    first: int | None = None
    for pos in range(lo, hi):
        local = pos - chunk.ref_start
        if 0 <= local < len(chunk.ref_kinds) and chunk.ref_kinds[local] in ("match", "mismatch"):
            diagonal += 1
            if first is None:
                first = pos - lo
            if chunk.ref_kinds[local] == "match":
                exact += 1
    return (
        first is not None
        and first <= 1
        and exact >= 1
        and diagonal >= min(2, length)
        and diagonal / length >= _MIN_WORD_SUPPORT_COVERAGE
    )


# The largest run of consecutive unsupported reference words `_supported_end` will bridge
# while still treating the recitation as ongoing. A single dropped word is a routine CTC
# omission mid-recitation (the reciter kept going); a run of two or more skipped reference
# words means the decode genuinely left the contiguous recitation — the signature of a
# phantom over-read that snapped its end onto a far, isolated duplicate word.
_MAX_SUPPORT_GAP = 1


def _supported_end(chunk: _ChunkAlign, boundaries: list[int], n_words: int) -> int:
    """Half-open word end of the words ``chunk``'s own decode actually recited.

    Scans forward from the chunk's snapped start through the words it keeps supporting, up
    to its snapped end (inclusive, so a tail-dropped final phoneme is still credited),
    bridging interior gaps of up to :data:`_MAX_SUPPORT_GAP` unsupported words (a routine
    single-word CTC dropout) but stopping once a longer unsupported run appears. A phantom
    over-read — whose matched span over-snapped past a multi-word gap onto an isolated
    later duplicate — is therefore truncated at that gap, reporting the word the reciter
    actually reached rather than the inflated snap. If the chunk supports none of its words
    (a fully unreliable decode) the snapped ``end_word`` is trusted rather than collapsing
    the segment to nothing.
    """
    last_supported: int | None = None
    gap = 0
    upper = min(n_words - 1, chunk.end_word)
    for word in range(chunk.start_word, upper + 1):
        if _word_supported(chunk, boundaries, word):
            last_supported = word
            gap = 0
        else:
            gap += 1
            if gap > _MAX_SUPPORT_GAP:
                break
    # Nothing word-supported: a fully unreliable decode over the chunk's claimed words. Fall
    # back to the snapped ``end_word`` (the pre-support-check baseline) rather than collapsing
    # the segment to nothing. This only reaches here for an on-edge chunk that is otherwise
    # reliable, and empirically never fires on the audit worklist — it is a defensive floor.
    return last_supported + 1 if last_supported is not None else chunk.end_word


def _mid_word_attribution(
    earlier: _ChunkAlign,
    later: _ChunkAlign,
    boundaries: list[int],
    n_words: int,
    boundary_tol: int,
) -> int | None:
    """Word a mid-word closure falls after — the furthest of two phoneme-aligned estimates.

    ``reached`` is the last word the pre-pause chunk's matched span fully covered; ``resume``
    is the word before the post-pause chunk cleanly restarts on a word edge. A forward stop
    with a CTC tail-drop under-counts ``reached``, so ``resume`` rescues it; a re-read that
    over-ran and backed up makes ``resume`` earlier, so ``reached`` wins. ``resume`` counts
    only when the post-pause chunk begins on or right of its snapped edge (``ref_start`` not
    left of the boundary) — a resume that snaps within tolerance but sits left of the edge is
    the same unfinished word continuing (a mid-word breath), which ``reached`` handles. Either
    may be ``None``; the result is ``None`` only when both are.
    """
    reached = _completed_word(boundaries, earlier.ref_end) if earlier.reliable else None
    resume = (
        later.start_word - 1
        if later.reliable
        and later.start_dist <= boundary_tol
        and later.ref_start >= boundaries[later.start_word]
        else None
    )
    candidates = [w for w in (reached, resume) if w is not None]
    if not candidates:
        return None
    return max(0, min(max(candidates), n_words - 1))


def _align_chunk(
    query: str,
    q_lo: int,
    q_hi: int,
    t_lo: float,
    t_hi: float,
    reference: str,
    boundaries: list[int],
) -> _ChunkAlign:
    """Locally align one inter-pause chunk to the reference and snap its edges to words."""
    chunk = query[q_lo:q_hi]
    unreliable = _ChunkAlign(
        q_lo, q_hi, t_lo, t_hi, 0, _FAR, 0, _FAR, reliable=False
    )
    if not chunk:
        return unreliable
    alignment = smith_waterman(chunk, reference)
    if alignment.score < _MIN_CHUNK_ALIGN_RATIO * len(chunk):
        return unreliable

    start_word, start_dist = _nearest_boundary(boundaries, alignment.ref_start)
    end_word, end_dist = _nearest_boundary(boundaries, alignment.ref_end)
    # A chunk whose matched span snaps to less than a whole word (both edges on the same
    # boundary) is a sliver, not a trustworthy anchor: treat it as unreliable so it can
    # never split the clip nor fabricate a one-word range downstream.
    if end_word <= start_word:
        return unreliable
    return _ChunkAlign(
        q_lo=q_lo,
        q_hi=q_hi,
        t_lo=t_lo,
        t_hi=t_hi,
        start_word=start_word,
        start_dist=start_dist,
        end_word=end_word,
        end_dist=end_dist,
        reliable=True,
        ref_start=alignment.ref_start,
        ref_end=alignment.ref_end,
        ref_kinds=tuple(m.kind for m in alignment.ref_matches),
    )


def segment_clip(
    class_ids: list[int],
    clip_duration_s: float,
    reference: str,
    boundaries: list[int],
    pauses: list[tuple[float, float]],
    *,
    boundary_tol: int = DEFAULT_BOUNDARY_TOL,
    max_decode_ratio: float = DEFAULT_MAX_DECODE_RATIO,
    min_align_ratio: float = DEFAULT_MIN_ALIGN_RATIO,
) -> SegmentationResult:
    """Split one clip into waqf spans at the reciter's pauses, re-read-aware.

    ``pauses`` are the clip's ``(start_s, end_s)`` waqf silences (the interior gaps the
    recitation VAD found — see :mod:`tadabur.vad`). ``reference`` is the ayah's spaceless
    phoneme string and ``boundaries`` its per-word phoneme offsets (``len == n_words + 1``,
    computed phonetizer-side so wasl word-merges are handled — see
    ``tadabur.waqf_segments.hafs_word_reference``).

    The decode is split at the pauses into **chunks**, and each chunk is locally aligned to
    the reference on its own, snapping to the word edges it starts/ends on. A pause is then:

    * a **waqf** split when both neighbouring chunks land on a word edge and the later
      chunk continues forward (its start word ``>=`` the earlier chunk's end word);
    * a **re-read** split when the later chunk lands on a word edge *earlier* than where the
      previous chunk ended (it re-recites already-said words) — the seam is still a genuine
      stop, so the clip is cut there into two time-consecutive segments whose Uthmani word
      ranges overlap ("read until the waqf" / "re-read point until the end");
    * a **mid-word** closure otherwise (a chunk edge off any word boundary) — the chunks it
      separates are merged, not split (a real waqf only lands on a word end).

    The first segment's ``start_s`` and the last segment's ``end_s`` are re-cut to the
    whole-clip alignment's matched-span onset/offset (± :data:`EDGE_RECUT_PAD_S`) to trim
    neighbour-ayah lead-in / trailing bleed; interior segment edges sit exactly at their
    split pause.

    Returns the spans plus a ``re_reads`` count, or a :class:`SegmentationResult` with a
    ``skip`` reason for a clip that cannot be segmented safely: ``low_alignment`` (the decode
    does not match this ayah) or ``repeated_recitation`` (a segment's decode still overruns
    its reference by ``max_decode_ratio`` — a gross repeat with no pause seam to split on).
    """
    n_words = len(boundaries) - 1
    whole = SegmentationResult(
        (WaqfSpan(0, n_words, 0.0, clip_duration_s),),
        pauses=_unplaced_attributions(pauses),
    )
    if not class_ids or n_words <= 0 or not reference:
        return whole

    seconds_per_frame = clip_duration_s / len(class_ids)
    query, decode_times = collapse_with_times(class_ids, seconds_per_frame)

    alignment = smith_waterman(query, reference)
    if alignment.score < min_align_ratio * len(reference):
        return SegmentationResult(
            (), skip="low_alignment", pauses=_unplaced_attributions(pauses)
        )

    # Re-cut the clip's outer edges to the whole-clip matched span's onset/offset (raw 1:1
    # decode, padded outward and clamped): trims the neighbour-ayah lead-in / trailing bleed
    # no interior pause covers (see #20). Interior segment edges stay at their split pause.
    # An empty (all-blank) decode scores 0 and is rejected as ``low_alignment`` above, so
    # ``query`` is non-empty and these indices are valid here.
    recut_start = max(0.0, decode_times[alignment.query_start] - EDGE_RECUT_PAD_S)
    recut_end = min(clip_duration_s, decode_times[alignment.query_end - 1] + EDGE_RECUT_PAD_S)

    # Split the query (and clip time) at each pause into chunks: chunk i spans the decoded
    # phonemes whose onset falls between pause i-1's end and pause i's start.
    pauses = sorted(pauses)
    cut_q = [sum(1 for t in decode_times if t < pause_start) for pause_start, _ in pauses]
    q_bounds = [0, *cut_q, len(query)]
    t_los = [0.0, *(pause_end for _, pause_end in pauses)]
    t_his = [*(pause_start for pause_start, _ in pauses), clip_duration_s]
    chunks = [
        _align_chunk(
            query, q_bounds[i], q_bounds[i + 1], t_los[i], t_his[i],
            reference, boundaries,
        )
        for i in range(len(q_bounds) - 1)
    ]

    # Classify and attribute each pause between an adjacent chunk pair. A split (waqf or
    # re-read) only when both chunk edges land on a word boundary; a non-split pause merges
    # the chunks into one segment. ``supported_end`` is the word the *earlier* chunk's own
    # decode actually reached — the phoneme-alignment signal runtime has, with no forced-
    # alignment timing. It classifies the seam and, un-inflating a phantom over-read whose
    # matched span over-snapped past where the reciter stopped, sets the segment's true extent:
    # the seam is a re-read when the later chunk resumes inside that real coverage
    # (``later.start_word < supported_end`` — true for an adjacent re-read and a gross restart
    # alike); otherwise the later chunk merely continues forward, an ordinary waqf. The MARKER
    # is dispatched on that classification: a re-read takes the stop frontier ``supported_end-1``
    # (its resume word points backward, so the reached/resume estimate would mislocate it),
    # while a forward waqf keeps the reached/resume estimate (its resume word pins the stop even
    # when the decode dropped a short final word). A mid-word closure falls off any word edge and
    # also uses the reached/resume estimate.
    split_after: list[bool] = []
    supported_ends: list[int | None] = []  # supported end per split pause; None for a merge
    pause_attrib: list[PauseAttribution] = []
    re_reads = 0
    for i, (pause_start, pause_end) in enumerate(pauses):
        earlier, later = chunks[i], chunks[i + 1]
        on_edge = (
            earlier.reliable and later.reliable
            and earlier.end_dist <= boundary_tol
            and later.start_dist <= boundary_tol
        )
        split_after.append(on_edge)
        if on_edge:
            supported_end = _supported_end(earlier, boundaries, n_words)
            genuine = later.start_word < supported_end
            kind = RE_READ if genuine else WAQF
            re_reads += genuine
            supported_ends.append(supported_end)
        else:
            kind = MID_WORD_CLOSURE
            supported_ends.append(None)
        if kind == RE_READ:
            # A re-read resumes behind the stop, so the stop is the last word the earlier
            # chunk reached — not the reached/resume estimate, whose resume anchor points back
            # at the (earlier) re-read word.
            word_index: int | None = max(0, min(supported_end - 1, n_words - 1))
        else:
            # An ordinary forward waqf or a mid-word closure: the reciter stopped and moved on.
            # The resume word pins the stop even when the decode dropped a short final word's
            # phonemes, so the reached/resume estimate beats the support frontier here.
            word_index = _mid_word_attribution(earlier, later, boundaries, n_words, boundary_tol)
        pause_attrib.append(PauseAttribution(pause_start, pause_end, kind, word_index))

    groups: list[list[int]] = [[0]]
    for i, do_split in enumerate(split_after):
        if do_split:
            groups.append([i + 1])
        else:
            groups[-1].append(i + 1)

    spans: list[WaqfSpan] = []
    for gi, group in enumerate(groups):
        first, last = chunks[group[0]], chunks[group[-1]]
        reliable = [chunks[c] for c in group if chunks[c].reliable]
        starts = [c.start_word for c in reliable]
        # The clip is admitted as this whole ayah, so its first segment starts at word 0 and
        # interior split words come from the chunk aligns.
        word_start = 0 if gi == 0 else (starts[0] if starts else 0)
        if gi == len(groups) - 1:
            # Final segment: bound the extent by where the last *reliable* chunk's own decode
            # actually reached (``_supported_end``), not a blind snap to the whole ayah. This
            # un-inflates an early stop — the reciter ended mid-ayah, so trusting ``n_words``
            # invents word markers for words never recited. The last reliable chunk (not simply
            # ``last``) is used so a trailing unreliable fragment — e.g. an elongated final-word
            # tail or a post-ayah artifact chunk — cannot collapse a completed recitation.
            word_end = (
                _supported_end(reliable[-1], boundaries, n_words) if reliable else n_words
            )
        else:
            # The segment ends at its terminating split pause, so its extent is the word the
            # pre-pause chunk's decode actually reached (``_supported_end``): this un-inflates
            # a phantom over-read into a contiguous waqf while leaving a genuine re-read's
            # overlap and a full-restart's whole-ayah span intact.
            term_end = supported_ends[group[-1]]
            word_end = term_end if term_end is not None else (
                reliable[-1].end_word if reliable else n_words
            )
        if word_end <= word_start:
            word_end = min(n_words, word_start + 1)
            word_start = max(0, word_end - 1)

        start_s = recut_start if gi == 0 else pauses[group[0] - 1][1]
        end_s = recut_end if gi == len(groups) - 1 else pauses[group[-1]][0]
        spans.append(WaqfSpan(word_start, word_end, start_s, end_s))

        # A segment whose decode still runs well past its reference words is an unsplit
        # repeat (no pause seam fell between the passes) — keep the whole clip out.
        ref_span = boundaries[word_end] - boundaries[word_start]
        if ref_span > 0 and last.q_hi - first.q_lo > max_decode_ratio * ref_span:
            return SegmentationResult(
                (), skip="repeated_recitation",
                pauses=_unplaced_attributions(pauses),
            )

    return SegmentationResult(
        tuple(spans), re_reads=re_reads, pauses=tuple(pause_attrib)
    )
