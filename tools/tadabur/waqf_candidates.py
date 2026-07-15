"""Derive the waqf candidate-boundary manifest the F0 adjudication UI consumes (#30).

F0a (#27) built the sampler / UI / fixture schema but deliberately left their **input**
— the candidate-boundary manifest — to be produced here: "the segmentation/VAD pass, the
waqf analogue of the poison audit's filter manifest". This module is that producer, and
it is **torch-free**: every candidate is read off artifacts the P3.5 segmentation pass
already wrote (:mod:`tadabur.segment_score`), so no model or GPU re-run is needed.

The three classes ADR-0004's event-level eval must measure fall straight out of the
segmentation:

* **waqf** — a confirmed intra-ayah stop. It *is* an interior segment boundary: the gap
  between two adjacent segments (``seg_i.end_s`` → ``seg_{i+1}.start_s``), which
  :mod:`tadabur.waqf_detect` only creates when a VAD pause mapped onto a word edge.
* **mid_word_closure** — a VAD pause that fell **inside** a segment, i.e. one
  :func:`tadabur.waqf_detect.segment_clip` treated as mid-word (a qalqala/hamza stop
  silence, not a waqf). Read from the raw VAD pause list, minus the boundary pauses.
* **wasl** — an interior Uthmani word edge with **no** pause, where the detector implicitly
  said "continuation". Every word edge strictly inside a segment is one.

Per-word edge times (for wasl edges and for a mid-word pause's word attribution) are
interpolated within each segment's ``[start_s, end_s]`` span, phoneme-proportionally from
:func:`tadabur.waqf_segments.hafs_word_reference` when available, else uniformly by word
count — approximate on purpose: the adjudication UI plays the whole clip and seeks near the
boundary, and the human calls it by ear.

Inputs are the segment manifest (:mod:`tadabur.segment_score`, a JSONL whose per-segment
``audio_filename`` is ``<clip>__seg<index>.wav``) and the VAD pause map
(``{clip_audio_filename: [[start_s, end_s], ...]}``) that :mod:`tadabur.segment_score`
stages alongside it. Output is a JSONL of
:class:`~tadabur.waqf_event_sampler.WaqfCandidate` rows, the sampler's input unit.

Usage:
  python -m tadabur.waqf_candidates --segment-manifest segment_manifest.jsonl \\
    --vad-pauses vad_pauses.json --out candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .waqf_event_fixtures import MID_WORD_CLOSURE, WAQF, WASL
from .waqf_event_sampler import WaqfCandidate

# A pause is a *boundary* pause (already a waqf segment split) when its onset sits within
# this many seconds of a segment's end. The segmentation sets ``seg_i.end_s`` exactly to a
# confirming pause's start, so the match is essentially exact; the tolerance only absorbs
# float round-trips through JSON.
BOUNDARY_MATCH_TOL_S = 0.02

# Per-word-edge phoneme boundaries for an ayah, or ``None`` when the phonetizer cannot
# handle it (interpolation then falls back to uniform-by-word).
WordBoundaries = Callable[[str], list[int] | None]

_SEG_SUFFIX = "__seg"


@dataclass(frozen=True)
class Segment:
    """One waqf segment as read from the segment manifest, keyed to its whole clip.

    ``word_start`` / ``word_end`` are the half-open Uthmani word range this segment
    covers, derived cumulatively from the per-segment word counts (the manifest's
    ``uthmani`` is the segment's own words). ``start_s`` / ``end_s`` are its clip time
    span (outer edges already re-cut by :mod:`tadabur.waqf_detect`).
    """

    clip_id: str
    surah_ayah: str
    segment_index: int
    start_s: float
    end_s: float
    n_words: int
    word_start: int
    word_end: int


def clip_base(audio_filename: str) -> str:
    """The whole-clip filename a per-segment ``<clip>__seg<n>.wav`` belongs to."""
    stem = audio_filename[: -len(".wav")] if audio_filename.endswith(".wav") else audio_filename
    base = stem.split(_SEG_SUFFIX)[0]
    return f"{base}.wav"


def read_segments(path: Path) -> dict[str, list[Segment]]:
    """Group the segment manifest into per-clip, word-range-annotated segment lists.

    Each clip's segments are ordered by ``segment_index`` and carry the explicit half-open
    Uthmani ``word_start`` / ``word_end`` range the manifest records. These are trusted
    verbatim rather than re-derived cumulatively from word counts, because a re-read clip's
    segments overlap in words (two passes over a phrase) and so do **not** tile ``[0,
    n_words)`` contiguously. Blank lines are skipped.
    """
    raw: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw.setdefault(clip_base(row["audio_filename"]), []).append(row)

    clips: dict[str, list[Segment]] = {}
    for clip_id, rows in raw.items():
        rows.sort(key=lambda r: r["segment_index"])
        segments: list[Segment] = []
        for row in rows:
            n_words = len(row["uthmani"].split())
            segments.append(Segment(
                clip_id=clip_id,
                surah_ayah=row["surah_ayah"],
                segment_index=row["segment_index"],
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                n_words=n_words,
                word_start=row["word_start"],
                word_end=row["word_end"],
            ))
        clips[clip_id] = segments
    return clips


def load_pauses(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Read the VAD pause map ``{clip: [[start_s, end_s], ...]}`` into float tuples."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        clip: [(float(a), float(b)) for a, b in spans]
        for clip, spans in data.items()
    }


# A mid_word_closure pause's sidecar attribution is matched to its VAD pause by onset. Both
# come from the same VAD pause list (segment_score passes it straight to segment_clip), so
# the onsets are the same float printed to JSON twice; this tolerance only absorbs the
# round-trip, mirroring :data:`BOUNDARY_MATCH_TOL_S`.
ATTRIB_MATCH_TOL_S = 0.02


def load_pause_attributions(path: Path) -> dict[str, dict[str, int | None]]:
    """Read the pause-attribution sidecar into ``{clip: {onset_key: word_index}}``.

    :func:`tadabur.segment_score.write_pause_attributions` writes one row per clip with a
    ``pauses`` list of ``{start_s, end_s, kind, word_index}`` — **every** interior pause the
    segmenter saw, of every kind. All are indexed here (not just ``mid_word_closure``): the
    phoneme-aligned ``word_index`` is the word the decode had completed when the pause fell,
    computed identically regardless of kind, so a pause the segmenter split on (``waqf``)
    still carries the right word for the audit even if the scored manifest later dropped the
    neighbouring segment and the candidate layer re-reads that pause as an interior stop.

    Each pause is keyed by its onset rounded to the millisecond (:func:`_onset_key`) so
    :func:`clip_candidates` can look a pause's word up by ``start_s`` without a float-equality
    hazard, and its value is the phoneme-aligned word (``int``) or ``None`` when the segmenter
    could not place it. Keeping the ``None`` entries — rather than dropping them — lets the
    consumer tell an *explicitly unplaceable* pause (fall back to interpolation) apart from a
    pause **missing** from the sidecar (a stale / mismatched artifact, which it rejects).
    """
    attributions: dict[str, dict[str, int | None]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            placed: dict[str, int | None] = {}
            for pause in row["pauses"]:
                word_index = pause["word_index"]
                placed[_onset_key(float(pause["start_s"]))] = (
                    None if word_index is None else int(word_index)
                )
            attributions[row["clip_audio_filename"]] = placed
    return attributions


def _onset_key(start_s: float) -> str:
    """Millisecond-rounded onset key for matching a pause to its sidecar attribution."""
    return f"{round(start_s, 3):.3f}"


def _edge_times(segment: Segment, boundaries: list[int] | None) -> list[float]:
    """Clip times of this segment's word edges ``word_start .. word_end`` (inclusive).

    Returns ``word_end - word_start + 1`` times spanning ``[start_s, end_s]``. Interior
    edges are placed phoneme-proportionally from ``boundaries`` (the ayah's per-word
    phoneme offsets) when it is usable, else uniformly across the word count. A
    single-word segment yields just its two endpoints.
    """
    w0, w1 = segment.word_start, segment.word_end
    span = segment.end_s - segment.start_s
    n = w1 - w0
    if n <= 0:
        return [segment.start_s]

    fractions: list[float]
    usable = (
        boundaries is not None
        and len(boundaries) > w1
        and boundaries[w1] > boundaries[w0]
    )
    if usable:
        base = boundaries[w0]
        total = boundaries[w1] - base
        fractions = [(boundaries[w0 + i] - base) / total for i in range(n + 1)]
    else:
        fractions = [i / n for i in range(n + 1)]
    return [segment.start_s + f * span for f in fractions]


def _pause_is_boundary(pause_start: float, segments: list[Segment]) -> bool:
    """True if ``pause_start`` coincides with a segment end (a confirmed waqf split)."""
    return any(abs(pause_start - s.end_s) <= BOUNDARY_MATCH_TOL_S for s in segments[:-1])


def _word_at_time(segment: Segment, edge_times: list[float], t: float) -> int:
    """The Uthmani word index the pause falls *after*: the last word whose
    interpolated span ends at or before ``t``.

    Schema-wise ``word_index`` is the word a boundary falls *after* (see
    :class:`~tadabur.waqf_event_sampler.WaqfCandidate`), matching the waqf anchor.
    :func:`_edge_times` spreads a segment's words across its whole clip span
    *including* the pause silence, so a between-word pause lands inside the
    *following* word's inflated span; taking the last **completed** word keeps the
    marker on the word the reciter actually finished rather than the one about to
    start. Floors at ``segment.word_start`` for a stop inside the opening word.
    """
    completed = segment.word_start
    for i in range(len(edge_times) - 1):
        if edge_times[i + 1] <= t:
            completed = segment.word_start + i
        else:
            break
    return completed


def clip_candidates(
    segments: list[Segment],
    pauses: list[tuple[float, float]],
    word_boundaries: WordBoundaries,
    pause_attributions: dict[str, int | None] | None = None,
) -> list[WaqfCandidate]:
    """All waqf/wasl/mid-word-closure candidate boundaries for one clip.

    Emits a **waqf** for each interior segment boundary, a **wasl** for each word edge
    strictly inside a segment, and a **mid_word_closure** for each VAD pause that falls
    inside a segment (i.e. was not promoted to a boundary). Rows are ordered by clip
    time and assigned a contiguous ``boundary_index``.

    ``pause_attributions`` (``{onset_key: word_index}`` for *this clip* from
    :func:`load_pause_attributions`) is the phoneme-aligned word each pause fell after,
    computed the runtime-faithful way — Smith-Waterman aligning the decode to the
    reference, exactly as the app must at inference when no timestamps exist. Passing it
    switches the clip into phoneme-aligned mode: every interior pause **must** be in the
    map (the segmenter emits one attribution per VAD pause it saw, so a missing onset means
    a stale / mismatched sidecar and raises), a placed word (``int``) is used verbatim, and
    only an explicitly unplaceable pause (``None``) falls back to :func:`_word_at_time`'s
    time interpolation (which drifts a word because the pause silence inflates the segment
    span). ``None`` for the whole argument keeps the pure-interpolation legacy path, for a
    clip with no sidecar (e.g. a kept-whole skip) or a run given no sidecar at all.
    """
    if not segments:
        return []
    clip_id = segments[0].clip_id
    surah_ayah = segments[0].surah_ayah
    boundaries = word_boundaries(surah_ayah)
    edge_times = {s.segment_index: _edge_times(s, boundaries) for s in segments}

    raw: list[tuple[float, float, int, str]] = []  # (start_s, end_s, word_index, predicted)

    for seg in segments:
        times = edge_times[seg.segment_index]
        # wasl: word edges strictly interior to this segment (not its start/end).
        for i in range(1, seg.word_end - seg.word_start):
            edge = times[i]
            raw.append((edge, edge, seg.word_start + i - 1, WASL))

    # waqf: the gap between each pair of adjacent segments is a confirmed stop.
    # Anchor the marker to ``nxt.word_start - 1`` (the word right before the next
    # segment resumes), not ``prev.word_end - 1``. On a re-read the duplicate seam
    # word inflates ``prev.word_end`` (or a coverage gap leaves ``prev.word_end``
    # one short), so ``prev.word_end`` drifts from the audio; ``nxt.word_start`` is
    # firmly anchored by where the reciter picks the recitation back up. For a
    # contiguous seam the two are identical, so this is a no-op there. Floor at
    # ``prev.word_start`` so a full re-read from the ayah start (``nxt.word_start``
    # 0) cannot produce a negative index.
    for prev, nxt in zip(segments, segments[1:]):
        word_index = max(nxt.word_start - 1, prev.word_start)
        raw.append((prev.end_s, nxt.start_s, word_index, WAQF))

    # mid_word_closure: VAD pauses that landed inside a segment (never became a split).
    for pause_start, pause_end in pauses:
        if _pause_is_boundary(pause_start, segments):
            continue
        for seg in segments:
            if seg.start_s < pause_start < seg.end_s:
                word_index = _mid_word_index(
                    seg, edge_times[seg.segment_index], pause_start,
                    clip_id, pause_attributions,
                )
                raw.append((pause_start, pause_end, word_index, MID_WORD_CLOSURE))
                break

    raw.sort(key=lambda r: (r[0], r[1], r[2]))
    return [
        WaqfCandidate(
            clip_id=clip_id,
            audio_ref=clip_id,
            surah_ayah=surah_ayah,
            boundary_index=idx,
            word_index=word_index,
            start_s=start_s,
            end_s=end_s,
            predicted=predicted,
        )
        for idx, (start_s, end_s, word_index, predicted) in enumerate(raw)
    ]


def _mid_word_index(
    seg: Segment,
    edge_times: list[float],
    pause_start: float,
    clip_id: str,
    pause_attributions: dict[str, int | None] | None,
) -> int:
    """The word a mid-word-closure pause falls after — phoneme-aligned when available.

    With no sidecar (``pause_attributions is None``) this is the legacy time
    interpolation. With a sidecar the pause's phoneme-aligned word is used verbatim; an
    explicitly unplaceable pause (``None``) falls back to interpolation, and a pause the
    sidecar does not mention at all is a stale / mismatched artifact and raises rather than
    silently interpolating.
    """
    if pause_attributions is None:
        return _word_at_time(seg, edge_times, pause_start)
    key = _onset_key(pause_start)
    if key not in pause_attributions:
        raise ValueError(
            f"{clip_id}: interior pause at {pause_start:.3f}s has no phoneme-aligned "
            f"attribution in the sidecar — the pause-attribution artifact is stale or "
            f"from a different segmentation run."
        )
    aligned = pause_attributions[key]
    if aligned is not None:
        return aligned
    return _word_at_time(seg, edge_times, pause_start)


def build_candidates(
    segments_by_clip: dict[str, list[Segment]],
    pauses_by_clip: dict[str, list[tuple[float, float]]],
    word_boundaries: WordBoundaries,
    pause_attributions_by_clip: dict[str, dict[str, int | None]] | None = None,
) -> list[WaqfCandidate]:
    """Candidate boundaries for every clip, in clip-id then boundary order.

    When ``pause_attributions_by_clip`` is given (a sidecar was loaded), every clip is
    scored in phoneme-aligned mode: a clip present in the sidecar uses its per-pause
    attributions, and a clip absent from it is treated as an *empty* attribution map so any
    interior pause raises as a stale / mismatched artifact (``segment_clip`` emits one
    attribution per pause, so a clip with interior pauses is always present in a fresh
    sidecar). A pause-free clip has no interior pause to attribute and is unaffected.
    ``None`` disables the phoneme-aligned path entirely.
    """
    candidates: list[WaqfCandidate] = []
    for clip_id in sorted(segments_by_clip):
        attributions = (
            None if pause_attributions_by_clip is None
            else pause_attributions_by_clip.get(clip_id, {})
        )
        candidates.extend(clip_candidates(
            segments_by_clip[clip_id],
            pauses_by_clip.get(clip_id, []),
            word_boundaries,
            attributions,
        ))
    return candidates


def write_candidates(candidates: list[WaqfCandidate], path: Path) -> None:
    """Write the candidate manifest as JSONL, one :class:`WaqfCandidate` per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for candidate in candidates:
            f.write(json.dumps(asdict(candidate), ensure_ascii=False, sort_keys=True) + "\n")


def hafs_word_boundaries() -> WordBoundaries:
    """A cached ``surah_ayah -> per-word phoneme offsets`` map via the Hafs phonetizer.

    Wraps :func:`tadabur.waqf_segments.hafs_word_reference`, phonetizing each ayah once
    and returning its per-word offsets (``len == n_words + 1``). Returns ``None`` for an
    ayah the phonetizer cannot handle so callers fall back to uniform interpolation.
    """
    from .waqf_segments import _uthmani_words, hafs_word_reference

    reference = hafs_word_reference()
    cache: dict[str, list[int] | None] = {}

    def boundaries(surah_ayah: str) -> list[int] | None:
        if surah_ayah not in cache:
            try:
                _, offsets = reference(_uthmani_words(surah_ayah))
                cache[surah_ayah] = offsets
            except (KeyError, IndexError, ValueError):
                cache[surah_ayah] = None
        return cache[surah_ayah]

    return boundaries


def _resolve_pause_attributions(args) -> dict[str, dict[str, int | None]] | None:
    """Load the pause-attribution sidecar for :func:`main`, defaulting to the one
    ``segment_score`` writes next to the manifest.

    Returns ``None`` (legacy time-interpolation everywhere) when ``--no-pause-attrib`` is
    set or the default sidecar is absent — the latter keeps the tool runnable against an
    old manifest generated before the sidecar existed, with a clear warning. An explicit
    ``--pause-attrib`` that does not exist is a hard error rather than a silent downgrade.
    """
    if args.no_pause_attrib:
        return None
    if args.pause_attrib is not None:
        if not args.pause_attrib.exists():
            raise SystemExit(f"--pause-attrib {args.pause_attrib} does not exist.")
        return load_pause_attributions(args.pause_attrib)
    default = args.segment_manifest.with_suffix(
        args.segment_manifest.suffix + ".pause_attrib.jsonl"
    )
    if default.exists():
        print(f"Using phoneme-aligned pause attributions from {default}.")
        return load_pause_attributions(default)
    print(
        f"WARNING: no pause-attribution sidecar at {default}; mid_word_closure markers "
        f"fall back to time interpolation. Regenerate the manifest with "
        f"tadabur.segment_score to get phoneme-aligned markers."
    )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment-manifest", type=Path, required=True,
                        help="Segment manifest JSONL (tadabur.segment_score output).")
    parser.add_argument("--vad-pauses", type=Path, required=True,
                        help="VAD pause map JSON ({clip_audio_filename: [[start_s, end_s], ...]}).")
    parser.add_argument("--pause-attrib", type=Path, default=None,
                        help=(
                            "Pause-attribution sidecar JSONL (tadabur.segment_score "
                            "--out-pause-attrib) supplying each mid_word_closure's "
                            "phoneme-aligned word. Defaults to "
                            "<segment-manifest>.pause_attrib.jsonl; pass --no-pause-attrib "
                            "to force the legacy time-interpolation path."
                        ))
    parser.add_argument("--no-pause-attrib", action="store_true",
                        help="Ignore the pause-attribution sidecar and interpolate word "
                             "position from time (the pre-phoneme-alignment behaviour).")
    parser.add_argument("--out", type=Path, required=True, help="Output candidate manifest (JSONL).")
    args = parser.parse_args()

    segments_by_clip = read_segments(args.segment_manifest)
    pauses_by_clip = load_pauses(args.vad_pauses)
    pause_attributions_by_clip = _resolve_pause_attributions(args)
    candidates = build_candidates(
        segments_by_clip, pauses_by_clip, hafs_word_boundaries(),
        pause_attributions_by_clip,
    )
    write_candidates(candidates, args.out)

    by_class: dict[str, int] = {}
    for candidate in candidates:
        by_class[candidate.predicted] = by_class.get(candidate.predicted, 0) + 1
    counts = ", ".join(f"{by_class.get(c, 0)} {c}" for c in (WAQF, WASL, MID_WORD_CLOSURE))
    print(f"Wrote {len(candidates)} candidate boundaries from {len(segments_by_clip)} clips "
          f"to {args.out} ({counts}).")


if __name__ == "__main__":
    main()
