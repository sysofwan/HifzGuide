"""Scorer-side waqf post-processing reference: silence frames → snapped waqf events.

The waqf head (ADR-0004, ``docs/adr/0004-waqf-head-and-joint-whole-clip-fine-tune.md``)
emits a per-frame ``P(silence)`` on the Muaalem **40 ms** post-adapter lattice, alongside
the phoneme CTC output. Turning that frame signal into *waqf events* — the stops a reciter
made — stays **scorer-side**: the ADR keeps "waqf→word-boundary snapping and the 300/700 ms
post-processing" out of the model, "now reasoning over 40 ms frames ... cheap because the
phoneme alignment is in the same output". This module is the **Python reference** for that
post-processing. The production consumer is Muraja/Swift (out of scope here); this reference
is what the F2 event-level eval harness scores against the frozen F0 fixtures.

The pipeline is four torch-free, deterministic stages:

1. **Stitch** (:func:`stitch_silence`). The deployed model runs fixed 5 s windows with a
   center-trusted 1 s overlap (the A2 windowing contract frozen by #24 /
   :mod:`training.waqf_distill`). Per-window silence frames are recombined into one
   clip-length track by keeping, for each 40 ms frame, the posterior from the window whose
   **center is nearest** — so every interior stop is graded by the window that saw it in
   full context and the overlap's outer half of each window is discarded.

2. **Silence runs + 300/700 ms cleaning** (:func:`detect_pauses`). Frames are binarised
   (``P(silence) >= threshold`` — the argmax boundary of the VAD's two classes) and cleaned
   with the VAD's own training definition of a waqf: silences shorter than
   :data:`DEFAULT_MIN_SILENCE_MS` are not stops (merged into speech), speech islands shorter
   than :data:`DEFAULT_MIN_SPEECH_MS` are not speech (dropped), and the interior gaps between
   the surviving speech spans are the candidate pauses. This is the frame-level analogue of
   ``recitations_segmenter.clean_speech_intervals`` + :func:`tadabur.vad.pauses_from_intervals`,
   restated over the 40 ms lattice (300 ms ≈ 7–8 frames, 700 ms ≈ 17–18).

3. **Boundary snap, with mid-word-closure rejection** (:func:`snap_pauses`). A silence VAD
   detects *silence*, not a waqf, so a pause must be snapped to a **word edge** before it is
   an event — and a silence that fell *inside* a word (a qalqala closure on ق/ط, the hamza in
   شَيء) must be **rejected**, not fired as a stop. Timing alone cannot do this (a mid-word
   closure can sit tens of ms from a word edge), so the snap uses the **phoneme alignment**
   the same forward pass produced: each word owns a ``[start_frame, end_frame)`` speech span
   on the 40 ms lattice. A pause that overlaps a word's *interior* span is a mid-word closure;
   a pause that sits in the *gap* between one word's end and the next word's start is a waqf
   after that word. This mirrors :mod:`tadabur.waqf_detect` (map a pause to the last phoneme
   before it and ask whether that phoneme is at a word edge), moved from the phoneme-id
   sequence onto the frame lattice.

4. **madd is never a stop.** A madd (elongation) is *speech*, so its frames never form a
   ≥ 300 ms silence run and stage 2 emits no pause. Even a spurious silence dip during a madd
   is *inside* a word, so stage 3 rejects it as a closure rather than a waqf — the elongation
   is doubly protected from being mistaken for a stop.

Everything here is pure list / array logic over the frame lattice and a word-frame
alignment, so it is unit-testable — and unit-tested against the frozen F0 fixtures — with no
GPU, torch, or model forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The waqf head rides the Muaalem 40 ms post-adapter CTC lattice: the 20 ms encoder frames
# downsampled 2× by the single stride-2 adapter conv (ADR-0004; ``ml-model-transformation.md``;
# :data:`training.waqf_distill.SAMPLES_PER_STUDENT_FRAME`). The scorer reasons on that grid, so
# the frame duration is pinned here as the canonical scorer-side constant.
STUDENT_FRAME_MS = 40

# Binarisation threshold on ``P(silence)``. The VAD is a two-class (speech/silence) softmax, so
# ``P(silence) >= 0.5`` is exactly its argmax decision (:mod:`tadabur.vad`); exposed as a knob F2
# can calibrate on its partition (ADR-0004: "the eval slice only tunes the inference threshold").
DEFAULT_SILENCE_THRESHOLD = 0.5

# The waqf definition the Recitation VAD was trained on and the segmentation pass reuses
# (:mod:`tadabur.vad`): a stop is a silence ≥ 300 ms flanked by speech ≥ 700 ms. Reused here so
# the scorer's frame-level cleaning stays on-model instead of inventing a second threshold.
DEFAULT_MIN_SILENCE_MS = 300
DEFAULT_MIN_SPEECH_MS = 700

# Frames of overlap between a pause and a word's interior span tolerated before the pause is
# called mid-word. The teacher↔student feature extractors drift by ±1–2 frames (ADR-0004:
# "a 1–2 frame shift moves a boundary snap across a word edge"), so a word's aligned end may
# bleed a couple of frames into a genuine stop's silence; 2 frames (80 ms) absorbs that while
# staying far below the multi-hundred-ms overlap a real qalqala/hamza closure carves into a word.
DEFAULT_INTERIOR_TOL_FRAMES = 2


def seconds_to_frame(seconds: float, frame_ms: int = STUDENT_FRAME_MS) -> int:
    """The 40 ms lattice frame index a clip time (seconds) falls on (nearest frame)."""
    return round(seconds * 1000 / frame_ms)


def frame_to_seconds(frame: int, frame_ms: int = STUDENT_FRAME_MS) -> float:
    """The clip time (seconds) at the onset of 40 ms lattice frame ``frame``."""
    return frame * frame_ms / 1000


@dataclass(frozen=True)
class WindowFrames:
    """One fixed inference window's silence posteriors, placed on the clip 40 ms lattice.

    ``start_frame`` is the window's first 40 ms frame in **clip** coordinates (the
    ``Window.start_student_frame`` of :mod:`training.waqf_distill`); ``silence`` is that
    window's per-frame ``P(silence)``. The stitch (:func:`stitch_silence`) recombines a
    clip's windows into one track.
    """

    start_frame: int
    silence: np.ndarray


@dataclass(frozen=True)
class SilenceRun:
    """A half-open ``[start_frame, end_frame)`` run of silence on the 40 ms lattice."""

    start_frame: int
    end_frame: int

    @property
    def duration_ms(self) -> int:
        return (self.end_frame - self.start_frame) * STUDENT_FRAME_MS

    @property
    def start_s(self) -> float:
        return frame_to_seconds(self.start_frame)

    @property
    def end_s(self) -> float:
        return frame_to_seconds(self.end_frame)


@dataclass(frozen=True)
class WordSpan:
    """A word's speech extent on the 40 ms lattice, from the phoneme alignment.

    ``[start_frame, end_frame)`` are the word's first and one-past-last **speech** frames on
    the same 40 ms lattice the waqf head emits, so a pause can be tested against the word's
    interior vs the gap after it. Consecutive words are contiguous where the reciter did not
    stop and separated by a silence gap where they did.
    """

    word_index: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class WaqfEvent:
    """A detected stop: the ``pause`` and the ``word_index`` it falls after (its edge)."""

    word_index: int
    pause: SilenceRun


@dataclass(frozen=True)
class WaqfPostProcess:
    """The snap's output: confirmed ``waqf`` stops and ``rejected_closures`` (mid-word)."""

    waqf: list[WaqfEvent]
    rejected_closures: list[SilenceRun]


def stitch_silence(windows: list[WindowFrames]) -> np.ndarray:
    """Recombine per-window silence frames into one clip track (A2 center-trusted overlap).

    Each 40 ms clip frame is graded by the window whose **center is nearest** to it (ties to
    the earlier window). For the frozen 5 s window / 1 s overlap this splits the overlap at its
    midpoint — each window is authoritative over its central band and discards its outer 0.5 s
    — so no interior stop is trapped in a padding-affected window edge (ADR-0004 "Frozen
    windowing contract"; :func:`training.window_envelope.policy_options` "center-trusted
    overlap"). The first/last windows keep their outer frames because no neighbour covers them.
    A frame covered by **no** window is a gap in the tiling and fails loudly rather than leaving
    a hole in the track.
    """
    if not windows:
        return np.empty(0, dtype=np.float32)
    ordered = sorted(windows, key=lambda w: w.start_frame)
    total = max(w.start_frame + len(w.silence) for w in ordered)
    centers = [w.start_frame + len(w.silence) / 2 for w in ordered]
    track = np.full(total, np.nan, dtype=np.float32)
    for frame in range(total):
        owner = -1
        best = None
        for i, w in enumerate(ordered):
            if w.start_frame <= frame < w.start_frame + len(w.silence):
                distance = abs(frame - centers[i])
                if best is None or distance < best:
                    best = distance
                    owner = i
        if owner < 0:
            raise ValueError(
                f"frame {frame} of the {total}-frame clip track is covered by no window; "
                "the windows do not tile the clip"
            )
        w = ordered[owner]
        track[frame] = w.silence[frame - w.start_frame]
    return track


def _speech_runs(is_speech: np.ndarray) -> list[tuple[int, int]]:
    """Maximal ``[start, end)`` runs where ``is_speech`` is True, in frame order."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, speech in enumerate(is_speech):
        if speech and start is None:
            start = i
        elif not speech and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(is_speech)))
    return runs


def detect_pauses(
    silence: np.ndarray,
    *,
    frame_ms: int = STUDENT_FRAME_MS,
    threshold: float = DEFAULT_SILENCE_THRESHOLD,
    min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS,
) -> list[SilenceRun]:
    """The interior waqf pauses in a 40 ms silence track, cleaned by the 300/700 ms rules.

    Binarises each frame (``P(silence) >= threshold`` — the VAD's argmax boundary), then applies
    the VAD's on-model cleaning at the frame level: a silence gap shorter than ``min_silence_ms``
    is not a stop and its two speech spans are merged across it; a speech span shorter than
    ``min_speech_ms`` is not speech and is dropped; the interior gaps between the surviving
    speech spans are the pauses. Leading and trailing silence are clip edges, not interior
    stops, so they are excluded (there is no surviving speech before / after them). This mirrors
    ``recitations_segmenter.clean_speech_intervals`` (merge short silences → drop short speech)
    followed by :func:`tadabur.vad.pauses_from_intervals`, restated over the 40 ms lattice.
    """
    is_speech = np.asarray(silence, dtype=np.float32) < threshold
    speech_runs = _speech_runs(is_speech)
    if not speech_runs:
        return []

    merged: list[list[int]] = [list(speech_runs[0])]
    for start, end in speech_runs[1:]:
        gap_ms = (start - merged[-1][1]) * frame_ms
        if gap_ms < min_silence_ms:
            merged[-1][1] = end  # short silence: not a stop, merge the two speech spans
        else:
            merged.append([start, end])

    kept = [(s, e) for s, e in merged if (e - s) * frame_ms >= min_speech_ms]
    return [
        SilenceRun(prev_end, next_start)
        for (_, prev_end), (next_start, _) in zip(kept, kept[1:])
    ]


def _interior_overlap_frames(pause: SilenceRun, word: WordSpan) -> int:
    """Frames by which ``pause`` overlaps ``word``'s interior speech span (0 if disjoint)."""
    return max(
        0, min(pause.end_frame, word.end_frame) - max(pause.start_frame, word.start_frame)
    )


def snap_pauses(
    pauses: list[SilenceRun],
    words: list[WordSpan],
    *,
    interior_tol_frames: int = DEFAULT_INTERIOR_TOL_FRAMES,
) -> WaqfPostProcess:
    """Snap each pause to a word edge, rejecting the ones that fall inside a word.

    A pause that overlaps some word's interior speech span by more than ``interior_tol_frames``
    is a **mid-word closure** (a qalqala/hamza silence the word continues past) and is rejected.
    A pause that sits in the gap after a word — no interior overlap — is a **waqf** attributed to
    the last word that ended at or before it (its edge). A pause before the first word (leading
    silence) has no word to follow and is neither a stop nor a closure. Words are taken in
    ``start_frame`` order; the result lists are deterministic and in pause order.
    """
    ordered = sorted(words, key=lambda w: w.start_frame)
    waqf: list[WaqfEvent] = []
    rejected: list[SilenceRun] = []
    for pause in pauses:
        if any(_interior_overlap_frames(pause, w) > interior_tol_frames for w in ordered):
            rejected.append(pause)
            continue
        preceding = [
            w for w in ordered if w.end_frame <= pause.start_frame + interior_tol_frames
        ]
        if preceding:
            waqf.append(WaqfEvent(preceding[-1].word_index, pause))
    return WaqfPostProcess(waqf=waqf, rejected_closures=rejected)


def waqf_events(
    silence: np.ndarray,
    words: list[WordSpan],
    *,
    frame_ms: int = STUDENT_FRAME_MS,
    threshold: float = DEFAULT_SILENCE_THRESHOLD,
    min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS,
    interior_tol_frames: int = DEFAULT_INTERIOR_TOL_FRAMES,
) -> WaqfPostProcess:
    """Full reference: a clip's 40 ms silence track + word alignment → snapped waqf events."""
    pauses = detect_pauses(
        silence,
        frame_ms=frame_ms,
        threshold=threshold,
        min_silence_ms=min_silence_ms,
        min_speech_ms=min_speech_ms,
    )
    return snap_pauses(pauses, words, interior_tol_frames=interior_tol_frames)


def waqf_events_from_windows(
    windows: list[WindowFrames],
    words: list[WordSpan],
    **kwargs,
) -> WaqfPostProcess:
    """:func:`waqf_events` fed the stitched (:func:`stitch_silence`) per-window silence."""
    return waqf_events(stitch_silence(windows), words, **kwargs)
