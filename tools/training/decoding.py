"""One way to turn audio into a decode, for the teacher or any distilled student.

A **decode** is the phoneme string a model emits for some audio (``CONTEXT.md``). The
accuracy work under ADR-0011 asks for it two ways, of two kinds of model:

* **whole spans** -- one variable-length pass over each span, which is what the haraka
  tools (:mod:`training.tashkeel_eval`, :mod:`training.tashkeel_outcomes`) score; and
* **the streaming protocol** Muraja runs -- 5 s windows, 1 s hop, each window committing
  one second of its 125 CTC timesteps -- at the deployed block ``b=0`` or a later one;

of either the base teacher (a Hugging Face reference) or a student
(:mod:`training.distill_train` checkpoint). :class:`Decoder` is that one interface. A
**model reference** is resolved by one rule: a path to an existing *file* is a
distillation checkpoint; anything else (a hub id such as ``obadx/muaalem-model-v3_2``, or
a saved model directory such as :mod:`training.publish_hf` writes) is loaded with
``from_pretrained``.

**One numeric policy, chosen by the caller.** Both loaders hand back the same module for
the same weights, and :meth:`Decoder.load` casts it to ``weights_dtype`` and runs every
forward under the same CUDA bf16 autocast, whichever loader produced it -- so the loader
can never be the reason two decodes differ. ``weights_dtype`` is a required argument
because it moves decodes, and the repo has historically used both: the teacher has always
been decoded with bf16 weights (:func:`training.distill_train.load_teacher`, every cached
teacher decode in :mod:`training.decode_evalset`), and students with fp32 weights
(:func:`load_student_from_checkpoint`, every published agreement number). Passing the
historical dtype for a model reproduces its historical stream exactly; passing the same
dtype for two models is what makes them comparable. Spans and streams share the policy,
so a span-vs-stream difference is windowing, never numerics.

The streaming protocol
----------------------

Replicated from ``MuaalemInference.predictSplit`` and ``RealtimeTranscriber``:

* A 5 s window of audio is feature-extracted on its own (per-window normalization) and run
  to 125 CTC timesteps.
* :func:`scan_ctc` collapses the greedy argmax into contiguous runs of the same
  **non-blank** token -- each run is one segment with a midpoint.
* A window's 125 timesteps are five 25-step **blocks**, one per second. Block ``b`` of the
  window starting at second ``w`` covers absolute second ``[w+b, w+b+1)``, with ``b``
  seconds of left context and ``4-b`` of right. A segment belongs to the block its
  **midpoint** falls in, so a run straddling a boundary is owned by exactly one block.
* Each window commits block ``b``. Muraja commits ``b=0`` -- the oldest second of the
  buffer, with the full 4 s of right context (:data:`DEPLOYED_BLOCK`). Everything later
  stays provisional and is re-decoded by the next window.
* The window advances by 1 s and the next pass commits the next second.

Concatenating the committed segments across a clip gives the transcript the user sees.

**The final window is flushed.** Without it the last ``4-b`` seconds of every clip -- and for
a clip under 5 s, everything past block ``b`` -- is decoded and then thrown away. Muraja
flushes whatever is pending when speech stops, so the last window also commits every block
after ``b``. :data:`PROTOCOL_VERSION` records the flush because it moves every number built
on this protocol; ``flush_tail=False`` exists only to reproduce a pre-flush measurement.

**The startup rule (b > 0).** Absolute second ``s`` is committed by window ``s-b``, so
the first ``b`` seconds would need windows ``-b .. -1``, which do not exist. The first
window therefore also commits every block **before** ``b``: it is the only window that
ever covers those seconds, so this decodes them from real audio, with all the right
context the clip offers, and commits them at the same moment block ``b`` of that window
commits. The alternative -- left-padding ``b`` seconds of silence to manufacture the
missing windows -- feeds the model input the device never sees and changes the per-window
feature normalization. With the flush, every replayed second is committed exactly once at
any ``b``; at ``b=0`` the startup rule commits nothing extra, so the deployed protocol is
unchanged.

Gaps to the deployed protocol, none established to be harmless: the VAD gate that skips
inference during silence is ignored (it selects which regions are scored, and two models
need not disagree at the same rate inside and outside them); the preview inferences are
skipped (they never enter the transcript); and a tail of under one second past the last
full window is never decoded -- a 5.9 s clip is one window covering its first 5 s.

A segment straddling the commit boundary can be emitted twice: a run at steps 18-29 of one
window has midpoint 23.5 and commits; the same audio lands at steps 0-4 of the next window
and commits again. The stream concatenates without reconciliation, faithfully to
``predictSplit``. Whether the device dedupes is unverified here.

The commit rule (:func:`stream_emissions`) is pure numpy over per-window argmax rows, so
the protocol is tested on synthetic logits with no model at all, and importing this module
does not import torch.

Muraja's check sequence
-----------------------

The stream above is the transcript Muraja commits. What Muraja *grades* is every update its
transcriber hands the engine (``FollowAlongEngine.handleUpdate``): each full window's hop and
overlap and each preview of the pending audio, and then, when the reciter stops, the final
decode ``stop()`` returns. :meth:`Decoder.app_checks` records that sequence for one item, one
inference per check, as ``{hop, overlap, flush}`` checks (the request format of
``tools/muraja_harness``) ending in a ``stop`` check (:func:`make_check`). Paths are under
``ios/HifzGuide/`` of ``sysofwan/Muraja`` at ``99c326f`` (``training.muraja_policy.MURAJA_COMMIT``);
bare line numbers are in ``Models/RealtimeTranscriber+iOS.swift``.

* **Schedule** (:func:`check_schedule`). Nothing runs below 2 s pending (:568-571). With a full
  5 s window pending (:575-577; ``windowSamples`` 80,000, :253) the window is decoded, its hop
  and overlap sent (:612), the window advanced by 1 s (:616, :458-468; ``hopSamples`` 16,000,
  :254) and the preview throttle reset (:617-618), so a preview of what is left runs next.
  Otherwise a preview of all pending audio runs once 3,200 new samples (200 ms) have arrived
  since the last (:238-241, :624-662). The item ends with ``stop()``: one inference over
  everything pending (:317-336), whose decode the app hands to the engine's final alignment
  pass, ``handleFinalFlush`` (``FollowAlong/LifecycleCoordinator+iOS.swift:340-354``,
  ``FollowAlong/FollowAlongEngine.swift:897-914``), before the session's end settles the
  reader's word (``FollowAlong/FollowAlongManager+iOS.swift:881-882``). Audio is taken to
  arrive in 200 ms steps, counted in exact samples, and every inference to be instantaneous, so
  a window runs as soon as 5 s are pending and every window start falls on the step grid.
* **Featurisation** (:meth:`Decoder.device_rows`). Each update's audio is featurised on its own:
  per-mel-bin normalization over its own frames and an odd frame count padded with 1.0
  (``Models/MelFeatureExtractor+iOS.swift:91-153``; the teacher's ``SeamlessM4TFeatureExtractor``
  does the same with ``padding_value`` 1 and ``pad_to_multiple_of`` 2). One difference: a
  single frame (400-559 samples) has variance 0 on the device, so it normalizes to zeros
  (:107-115 there), where the extractor's ``ddof=1`` makes it NaN; :func:`device_pairs` takes
  the device's. The rest of the model's
  fixed 250 feature pairs repeat the last real pair (:156-162 there): a preview is not
  zero-padded audio, and even a full window (249 real pairs) carries one repeated pair. Only
  the first ``min(max((pairs + 1) // 2, 1), 125)`` output timesteps are decoded
  (``Models/MuaalemInference+iOS.swift:107-117, 132-135``).
* **Split** (``Models/MuaalemInference+iOS.swift:128-149``). A window's CTC segments whose
  midpoint is before timestep 25 form the hop; the rest are the overlap.
* **Text** (``Models/PhonemeVocabulary.swift:61-102``). Each part is decoded and stripped of
  padding artifacts (:func:`strip_padding_artifacts`) on its own (:583-584, :638, :683). An
  window or preview with nothing in it is not sent (:599, :649). The stop's CTC hallucination
  gate does not depend on the speech detector and is applied (:328-332;
  ``Models/CTCDecoder+iOS.swift:44-49, 121-139``).

Bypassed, because they need the device's speech detector (Apple's SoundAnalysis) or its clock:
the silence flush after 5 s without meaningful output (:545-556), so no silence flush ever fires
here; the VAD power gate (:561-566); and the hallucination gates on windows and previews (:603,
:651; both only when the detector reports silence). The reciter is therefore taken to be
speaking throughout the item, up to the stop. Not modelled either: inference latency (late
windows, skipped previews), the device's CoreML fp16 numerics and 6-bit palettization, and
page-change discards (``FollowAlong/UpdateCoalescer+iOS.swift``, which otherwise hands every
update to the engine in order, :13-16). None of these is known to move grades in one direction
only, so the recorded sequence is an estimate of the device's grades, not a bound on them.

The transcript the app commits from those checks (:func:`committed_text`) is every hop, the
stop's included (``FollowAlong/HeardStream.swift:4-6``, ``FollowAlong/FollowAlongEngine.swift
:608-625, 907-909``).

**Another commit block** (#148). Muraja has no such transcriber; ``block`` names a hypothetical
one that keeps the schedule above, and therefore every inference and its audio, and changes
only what each inference sends, as the stream's commit rule does (:func:`commit_bounds`). A
window starting at second ``w`` sends block ``b`` (``[w+b, w+b+1)``) as its hop and the rest of
its timesteps as its overlap. The first window has no earlier window, so it also sends the
blocks before ``b`` as hop (the startup rule). Once the buffer has advanced past the first
window, the buffer still starts ``b`` seconds before the committed frontier, so every
inference's first ``b`` seconds are already in the transcript and are not sent again: a window
drops its blocks before ``b``, and a preview or the stop drops its segments whose midpoint is
before timestep ``25 b``. The stop's hallucination gate still reads all its timesteps, as
``CTCDecoder.scan`` reports them. Each second of audio then enters the engine's query once,
as at ``b=0``; ``b=0`` is the deployed sequence unchanged (:func:`app_checks_mode`).

Identifying a stored decode
---------------------------

A stored decode is only comparable to another made the same way. :class:`DecodeFingerprint`
names everything besides the audio that decides one -- model reference, mode (whole spans,
or the stream's :func:`stream_protocol`, which folds in the commit block and the flush),
weights dtype, batch size and :data:`INFERENCE_POLICY` -- and tools that persist decodes or
outcomes record it, so a comparison across runs can refuse a mismatched or missing one.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from itertools import islice
from pathlib import Path

import numpy as np

from tadabur.phoneme_vocab import PHONEME_ID_TO_CHAR
from tadabur.phoneme_vocab import PHONEME_PAD_ID as BLANK_ID
from training.distill_data import SAMPLE_RATE, WINDOW_SAMPLES
from training.distill_student import (
    DEPLOYED_LOGIT_FRAMES,
    NUM_PHONEME_CLASSES,
    PHONEME_LEVEL,
    PRESETS,
    TEACHER_MODEL_ID,
    build_student,
)

# Timesteps of each 125-step window that become the user-visible transcript -- one second,
# one block (``MuaalemInference.predictSplit``; ``hopTimeSteps = outputTimeSteps / 5``).
CONFIRM_TIMESTEPS = 25

# The device advances its buffer by 1 s per pass: 25 timesteps, one block.
HOP_SAMPLES = SAMPLE_RATE

# Five 25-step blocks per 125-step window, one per second of the 5 s window.
NUM_BLOCKS = DEPLOYED_LOGIT_FRAMES // CONFIRM_TIMESTEPS

# The block Muraja commits today: the oldest second of the buffer.
DEPLOYED_BLOCK = 0

# Students are distilled on the teacher's own front-end, and a checkpoint carries no
# feature extractor of its own, so a checkpoint decodes through the teacher's.
STUDENT_FEATURE_EXTRACTOR = TEACHER_MODEL_ID

# The two weight precisions this repo has decoded with, by the torch dtype each names; see
# the module docstring.
WEIGHTS_DTYPES = {"bf16": "bfloat16", "fp32": "float32"}

# Bumped whenever the forward pass changes what a model decodes for the same weights and
# audio in a way the fingerprint's other fields do not name: feature extraction, the dtype
# the features enter at. v1 feeds fp32 features, under bf16 autocast on CUDA and plainly on
# CPU (the device and the autocast are fingerprinted separately). The haraka tools decoded
# whole spans differently before this module (``tadabur.inference``: bf16 features, no
# autocast); their outputs carry no fingerprint and are refused as legacy.
INFERENCE_POLICY = "fp32-features-v1"

# The decode mode of a whole-span decode; a stream's mode is its :func:`stream_protocol`.
SPANS = "whole-spans"


# --- The protocol, on argmax rows ------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """A contiguous run of one non-blank CTC token -- Swift's ``CTCSegment``."""

    token_id: int
    start_step: int
    end_step: int  # inclusive

    @property
    def midpoint(self) -> float:
        return (self.start_step + self.end_step) / 2.0


def scan_ctc(class_ids: np.ndarray) -> list[Segment]:
    """Collapse per-timestep argmax into non-blank runs, mirroring Swift's ``scanCTC``.

    A run ends when the token changes. Blank runs are tracked (they separate repeated
    tokens, which is the whole point of the CTC blank) but never emitted.
    """
    segments: list[Segment] = []
    current_token = -1
    current_start = 0

    for step, token in enumerate(int(t) for t in class_ids):
        if token == current_token:
            continue
        if current_token != BLANK_ID and current_token != -1:
            segments.append(Segment(current_token, current_start, step - 1))
        current_token = token
        current_start = step

    if current_token != BLANK_ID and current_token != -1:
        segments.append(Segment(current_token, current_start, len(class_ids) - 1))

    return segments


def segments_between(class_ids: np.ndarray, low: int, high: int) -> list[Segment]:
    """The segments whose midpoint lies in ``[low, high)`` -- the commit rule itself."""
    return [seg for seg in scan_ctc(class_ids) if low <= seg.midpoint < high]


def block_bounds(block: int) -> tuple[int, int]:
    """The ``[low, high)`` timesteps of block ``block`` -- one second of the window."""
    if not 0 <= block < NUM_BLOCKS:
        raise ValueError(f"block must be in [0, {NUM_BLOCKS}), got {block}")
    low = block * CONFIRM_TIMESTEPS
    return low, low + CONFIRM_TIMESTEPS


def stream_protocol(block: int = DEPLOYED_BLOCK, flush_tail: bool = True) -> str:
    """The name of the streaming protocol committing ``block``, with or without the flush.

    v1 was the unflushed stream; v2 flushes the last window. A stream committed at another
    block is a different protocol, so it gets a different name -- the deployed one is
    :data:`PROTOCOL_VERSION`.
    """
    block_bounds(block)
    name = "confirmed-stream-v2-flush" if flush_tail else "confirmed-stream-v1"
    return name if block == DEPLOYED_BLOCK else f"{name}-b{block}"


# The deployed protocol. Cached teacher decodes carry it (``training.decode_evalset``) so a
# manifest built under one protocol cannot be silently scored under another.
PROTOCOL_VERSION = stream_protocol()


def commit_bounds(
    window: int, last_window: int, block: int = DEPLOYED_BLOCK, flush_tail: bool = True
) -> tuple[int, int]:
    """The midpoint range ``[low, high)`` window ``window`` of ``0..last_window`` commits.

    Every window commits block ``block``. The first window also commits the blocks before it
    (the startup rule) and, with ``flush_tail``, the last window commits the blocks after it
    (the silence flush). See the module docstring for why each exists.
    """
    low, high = block_bounds(block)
    if window == 0:
        low = 0
    if flush_tail and window == last_window:
        high = DEPLOYED_LOGIT_FRAMES
    return low, high


def confirmed_tokens(
    class_ids: np.ndarray, confirm_timesteps: int = CONFIRM_TIMESTEPS
) -> list[int]:
    """The tokens one window commits under the deployed rule: midpoint < ``confirm_timesteps``.

    The single-window view for tools that compare models window by window rather than over
    a clip's stream.
    """
    return [seg.token_id for seg in segments_between(class_ids, 0, confirm_timesteps)]


@dataclass(frozen=True)
class Emission:
    """One committed token, tagged with where in the protocol it came from.

    The bare token stream is what a metric scores, but it cannot say *which part of the
    protocol* produced a given error -- and two fifths of scored timesteps come from the
    flushed final window. Provenance is carried here rather than recomputed by a second pass
    so that there stays exactly one implementation of the commit rule; a divergent second
    copy of this protocol is the defect that hid the missing silence flush.
    """

    token_id: int
    window: int
    start_step: int
    end_step: int  # inclusive
    is_final_window: bool
    # The block the stream commits -- not necessarily the one this token sits in, which
    # differs in the first window (startup) and the last (flush).
    block: int = DEPLOYED_BLOCK

    @property
    def midpoint(self) -> float:
        return (self.start_step + self.end_step) / 2.0

    @property
    def _seam(self) -> int:
        """The step where this window stops committing and the next window takes over."""
        return block_bounds(self.block)[1]

    @property
    def is_flush(self) -> bool:
        """Committed only because no later window exists to re-decode these timesteps."""
        return self.is_final_window and self.midpoint >= float(self._seam)

    @property
    def straddles_seam(self) -> bool:
        """The run crosses the commit boundary, so its commit is timing-sensitive.

        A segment ending one frame either side of the split is committed by a different
        window, which is how a student that is right about the token can still be charged an
        insertion or a deletion.
        """
        return self.start_step < self._seam <= self.end_step


def stream_emissions(
    rows: Sequence[np.ndarray], block: int = DEPLOYED_BLOCK, flush_tail: bool = True
) -> list[Emission]:
    """Replay the streaming protocol over one clip's per-window argmax rows.

    ``rows`` holds one row of 125 class ids per window, in window order. This is the only
    implementation of the commit rule; every stream in the repo is built by it.
    """
    last_window = len(rows) - 1
    return [
        Emission(
            token_id=seg.token_id,
            window=window,
            start_step=seg.start_step,
            end_step=seg.end_step,
            is_final_window=window == last_window,
            block=block,
        )
        for window, row in enumerate(rows)
        for seg in segments_between(row, *commit_bounds(window, last_window, block, flush_tail))
    ]


def clip_windows(num_samples: int, hop_samples: int = HOP_SAMPLES) -> list[int]:
    """Window starts for the streaming protocol: advance 1 s while audio remains.

    Only full windows run in steady state, so a clip shorter than one window yields a
    single (padded) pass and the tail past the last full window is not replayed -- the same
    place the real pipeline hands over to the silence flush.
    """
    if num_samples < WINDOW_SAMPLES:
        return [0]
    return list(range(0, num_samples - WINDOW_SAMPLES + 1, hop_samples))


def window_audio(samples: np.ndarray) -> list[np.ndarray]:
    """The 5 s windows the protocol runs over one clip, zero-padded to full length."""
    windows = []
    for start in clip_windows(len(samples)):
        chunk = samples[start : start + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        windows.append(chunk)
    return windows


def tokens_to_phonemes(token_ids: Iterable[int]) -> str:
    """Map already-collapsed CTC token ids to their phoneme characters.

    ``PHONEME_ID_TO_CHAR`` is a **tuple indexed by class id**, not a mapping. Testing
    ``id in PHONEME_ID_TO_CHAR`` therefore asks whether the *integer* is one of the
    characters, which is never true -- it silently yields an empty string, every gate
    scores 0.0, and the result reads as "the model produces nothing" rather than "the
    lookup is wrong". Guard the range explicitly instead.
    """
    return "".join(
        PHONEME_ID_TO_CHAR[t]
        for t in token_ids
        if t != BLANK_ID and 0 <= t < len(PHONEME_ID_TO_CHAR)
    )


# --- Muraja's check sequence (module docstring) ----------------------------------------

# The decode mode of a recorded check sequence (:meth:`Decoder.app_checks`). v1 ended the item
# with a silence flush; v2 ends it with the stop.
APP_CHECKS = "muraja-app-checks-v2"

# Audio arrives, and previews run, in steps of 200 ms (``previewMinNewSamples``).
PREVIEW_STEP_SAMPLES = 3_200
# No inference runs with less than 2 s pending.
MIN_PENDING_SAMPLES = 2 * SAMPLE_RATE
# The model's fixed input: 250 feature pairs, 5 s (``MelFeatureExtractor.fixedSeqLen``).
DEVICE_FEATURE_PAIRS = 250
# The analysis window: audio shorter than one frame has no features, and Swift's frame count
# misreads 241-399 samples as one frame, so the featuriser refuses it.
FRAME_SAMPLES = 400
# The frame hop: 10 ms.
FRAME_HOP_SAMPLES = 160

# The inferences of the schedule, and FLUSH, the silence flush the single-decode approximation
# ends an item with (``training.muraja_policy.single_decode_checks``).
WINDOW, PREVIEW, STOP, FLUSH = "window", "preview", "stop", "flush"


@dataclass(frozen=True)
class CheckSpan:
    """One inference the transcriber runs: its kind and the ``[start, end)`` samples it decodes."""

    kind: str
    start: int
    end: int


def check_schedule(num_samples: int) -> list[CheckSpan]:
    """Every inference the transcriber runs over an item of ``num_samples``, in order.

    At each 200 ms step of arrived audio: with a full window pending, the window (its hop the
    first second) and an advance of 1 s; then, with at least 2 s pending, a preview of all of
    it. The audio after the last whole step and everything still pending go to the stop.
    """
    spans: list[CheckSpan] = []
    start = 0
    for now in range(PREVIEW_STEP_SAMPLES, num_samples + 1, PREVIEW_STEP_SAMPLES):
        if now - start >= WINDOW_SAMPLES:
            spans.append(CheckSpan(WINDOW, start, start + WINDOW_SAMPLES))
            start += HOP_SAMPLES
        if now - start >= MIN_PENDING_SAMPLES:
            spans.append(CheckSpan(PREVIEW, start, now))
    spans.append(CheckSpan(STOP, start, num_samples))
    return spans


def make_check(kind: str, hop: str, overlap: str) -> dict | None:
    """What the engine receives for one inference, or ``None`` when it receives nothing.

    A window or preview is a ``handleUpdate`` and is dropped when it has nothing in it; a
    silence flush is a ``handleUpdate`` with ``flush`` set and is always sent. The stop is the
    phonemes ``stop()`` returns, always handed to ``handleFinalFlush`` (empty or not), and is
    marked ``stop``: it carries them as its hop, as the engine appends them as one
    (``FollowAlongEngine.swift:907-909``), and it is the session's last check.
    """
    if kind in (WINDOW, PREVIEW) and not (hop or overlap):
        return None
    check = {"hop": hop, "overlap": overlap, "flush": kind == FLUSH}
    return {**check, "stop": True} if kind == STOP else check


def committed_text(checks: Iterable[dict]) -> str:
    """The transcript the app commits from a check sequence: every hop, the stop's included."""
    return "".join(check["hop"] for check in checks)


def clusters(text: str) -> list[str]:
    """Swift Characters: a base scalar and the combining marks after it."""
    out: list[str] = []
    for char in text:
        if out and unicodedata.combining(char):
            out[-1] += char
        else:
            out.append(char)
    return out


_MADD_YA = "ۦ"
_NASALS = ("م", "ن")


def strip_padding_artifacts(text: str) -> str:
    """``PhonemeVocabulary.decodeAndStrip``'s strip, on whole Characters as Swift compares them.

    Leading: a madd ya, or a mim or nun that is alone or followed by a madd ya. Trailing: a
    madd ya, or a mim or nun preceded by one. A letter carrying a haraka is a different
    Character and is never stripped.
    """
    chars = clusters(text)
    while chars and (
        chars[0] == _MADD_YA or (chars[0] in _NASALS and chars[1:2] in ([], [_MADD_YA]))
    ):
        chars.pop(0)
    while chars and (
        chars[-1] == _MADD_YA or (chars[-1] in _NASALS and chars[-2:-1] == [_MADD_YA])
    ):
        chars.pop()
    return "".join(chars)


def device_output_steps(pairs: int) -> int:
    """The output timesteps the app decodes for ``pairs`` real feature pairs."""
    return min(max((pairs + 1) // 2, 1), DEPLOYED_LOGIT_FRAMES)


def likely_hallucination(class_ids: np.ndarray) -> bool:
    """``CTCDecoder.Stats.isLikelyHallucination`` over the decoded timesteps, its ratios in
    float32 as Swift computes them."""
    non_blank = class_ids[class_ids != BLANK_ID]
    if len(non_blank) == 0:
        return True
    if len(non_blank) < 3:
        return False
    blank_ratio = np.float32(len(class_ids) - len(non_blank)) / np.float32(len(class_ids))
    if blank_ratio >= np.float32(0.90):
        return True
    dominant_ratio = np.float32(np.bincount(non_blank).max()) / np.float32(len(non_blank))
    return bool(blank_ratio >= np.float32(0.75) and dominant_ratio >= np.float32(0.35))


def app_checks_mode(block: int = DEPLOYED_BLOCK) -> str:
    """The decode mode of a check sequence whose transcriber commits ``block``: :data:`APP_CHECKS`
    for the deployed block, a name of its own for any other (as :func:`stream_protocol`)."""
    block_bounds(block)
    return APP_CHECKS if block == DEPLOYED_BLOCK else f"{APP_CHECKS}-b{block}"


def app_check(span: CheckSpan, class_ids: np.ndarray, block: int = DEPLOYED_BLOCK) -> dict | None:
    """The update the transcriber sends for one inference, from its decoded timesteps, when it
    commits ``block`` (module docstring, "Another commit block").

    Once the first window has advanced the buffer (``span.start > 0``), the first ``block``
    seconds of every inference are already in the transcript and are not sent again. A window's
    hop is its block ``block`` (the first window's, everything before it too) and its overlap
    the rest.
    """
    segments = scan_ctc(class_ids)
    committed = block_bounds(block)[0] if span.start > 0 else 0

    def text(low: float, high: float = np.inf) -> str:
        selected = (seg.token_id for seg in segments if low <= seg.midpoint < high)
        return strip_padding_artifacts(tokens_to_phonemes(selected))

    if span.kind == WINDOW:
        split = min(block_bounds(block)[1], len(class_ids))
        return make_check(WINDOW, text(committed, split), text(split))
    if span.kind == PREVIEW:
        return make_check(PREVIEW, "", text(committed))
    return make_check(STOP, "" if likely_hallucination(class_ids) else text(committed), "")


def app_checks(
    spans: Sequence[CheckSpan], rows: Sequence[np.ndarray], block: int = DEPLOYED_BLOCK
) -> list[dict]:
    """The check sequence: each scheduled inference's update, the ones not sent left out."""
    if len(spans) != len(rows):
        raise ValueError(f"{len(spans)} scheduled inferences but {len(rows)} decoded rows")
    checks = (app_check(span, row, block) for span, row in zip(spans, rows))
    return [check for check in checks if check is not None]


# --- What produced a decode ------------------------------------------------------------


@dataclass(frozen=True)
class DecodeFingerprint:
    """Everything besides the audio that decides a stored decode: its identity.

    Two stored results may be compared only when every field but ``model`` agrees -- the
    model being the thing under comparison. Otherwise part of any difference between them is
    numerics or protocol rather than the models, and nothing downstream could tell how much.
    The b=0 stream's ``mode`` is :data:`PROTOCOL_VERSION`, so the streaming protocol's
    version and this fingerprint are one scheme, not two.

    Only :meth:`Decoder.fingerprint` builds one, from the decoder's *effective* settings --
    the dtype its weights actually hold, the device it actually runs on, whether that turns
    autocast on -- so a fingerprint cannot describe a decode other than the one it labels.
    """

    model: str
    mode: str  # SPANS, or a stream_protocol()
    weights_dtype: str
    batch_size: int
    device_type: str
    autocast: bool
    policy: str = INFERENCE_POLICY

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict | None, source: str) -> "DecodeFingerprint":
        """Read a stored fingerprint, refusing a legacy artifact that has none."""
        if record is None:
            raise ValueError(
                f"{source} records no decode fingerprint, so it predates "
                f"{INFERENCE_POLICY} and was decoded under numerics nothing can compare "
                "against. Regenerate it with the current tools."
            )
        expected = {f.name for f in fields(cls)}
        if set(record) != expected:
            raise ValueError(
                f"{source} has a malformed decode fingerprint: fields {sorted(record)}, "
                f"expected {sorted(expected)}."
            )
        return cls(**record)

    def check_comparable(self, other: "DecodeFingerprint", what: str) -> None:
        """Refuse to compare two decodes made under different settings."""
        mismatched = [
            f"{name}: {getattr(self, name)!r} vs {getattr(other, name)!r}"
            for name in (
                "policy", "mode", "weights_dtype", "batch_size", "device_type", "autocast"
            )
            if getattr(self, name) != getattr(other, name)
        ]
        if mismatched:
            raise ValueError(
                f"refusing to compare {what}: they were decoded under different settings "
                f"({'; '.join(mismatched)}), so part of any difference would be the decode, "
                "not the model. Regenerate the baseline with the same settings."
            )


# --- Loading a model reference ---------------------------------------------------------


MODEL_REF_HELP = "a hub id, a saved model directory, or a distillation checkpoint file"


def add_weights_dtype_argument(parser) -> None:
    """``--weights-dtype`` for a CLI that decodes through :meth:`Decoder.load`."""
    parser.add_argument(
        "--weights-dtype",
        choices=sorted(WEIGHTS_DTYPES),
        default="bf16",
        help="weight precision every model in the run is decoded at (default bf16, how the "
        "teacher has always been decoded; fp32 is how distill_eval decodes students). Two "
        "models are comparable only at the same value.",
    )


def _revision_kwargs(revision: str | None) -> dict:
    """``from_pretrained`` keyword arguments pinning ``revision``; none when it is not given,
    so an unpinned load is exactly what it always was."""
    return {} if revision is None else {"revision": revision}


def load_hf_model(model_ref: str, revision: str | None = None):
    """A Muaalem-architecture model saved for ``from_pretrained``, on CPU in eval mode.

    ``revision`` pins a hub id to one commit (``from_pretrained``'s own argument); without
    it the hub's default branch is fetched, which can move between two loads.

    Loaded through the vendored modeling code (not ``AutoModel``): the multi-level CTC class
    is not registered with transformers, and pinning the vendored copy keeps every tool on
    identical weights. Fails loudly if the phoneme head is not the 43-class vocabulary,
    which would silently corrupt every decode.
    """
    from tadabur.muaalem import (
        Wav2Vec2BertForMultilevelCTC,
        Wav2Vec2BertForMultilevelCTCConfig,
    )

    config = Wav2Vec2BertForMultilevelCTCConfig.from_pretrained(
        model_ref, **_revision_kwargs(revision)
    )
    classes = config.level_to_vocab_size[PHONEME_LEVEL]
    if classes != NUM_PHONEME_CLASSES:
        raise ValueError(
            f"{model_ref} phoneme head has {classes} classes, expected "
            f"{NUM_PHONEME_CLASSES} (tadabur.phoneme_vocab). Vocabulary drift -- the "
            "decode mapping would be corrupt."
        )
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
        model_ref, config=config, **_revision_kwargs(revision)
    )
    return model.eval()


def load_student_from_checkpoint(checkpoint_path: Path, device, use_ema: bool = False):
    """Rebuild the student described by a checkpoint and load its weights.

    Returns the run's persisted config alongside the model: the eval tools need it to
    refuse a split the checkpoint was not trained under.

    ``use_ema`` selects the averaged weights a run with ``--ema-decay`` stored beside the
    live ones. It raises rather than falling back when they are absent: silently scoring the
    live weights under an ``--ema`` flag would report the wrong model's number, and the two
    are meant to be compared.
    """
    import torch

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = state["config"]
    student = build_student(PRESETS[config["preset"]])
    if use_ema:
        # Reconstructed from the averager's shadow rather than read from a second stored
        # copy: the shadow holds only the floating-point tensors, so the live state supplies
        # the remaining buffers. Older checkpoints carry a materialised ``student_ema`` and
        # are still read directly.
        if "student_ema" in state:
            student.load_state_dict(state["student_ema"])
        elif "ema_state" in state:
            live = state["student"]
            shadow = state["ema_state"]["shadow"]
            student.load_state_dict(
                {
                    name: (shadow[name].to(value.dtype) if name in shadow else value)
                    for name, value in live.items()
                }
            )
        else:
            raise SystemExit(
                f"{checkpoint_path} carries no averaged weights -- it was trained without "
                f"--ema-decay. Drop --ema, or train a run that keeps an average."
            )
    else:
        student.load_state_dict(state["student"])
    student = student.to(device)
    student.eval()
    return student, config, state["step"]


# --- The interface ---------------------------------------------------------------------


class Decoder:
    """A loaded phoneme model, its feature extractor, and the one way they are run.

    Build with :meth:`load` from a model reference, or directly around a module a tool has
    already loaded (the distillation tools keep their teacher resident for other work);
    ``model_ref`` names the weights in the decoder's :meth:`fingerprint`. ``batch_size`` is
    fixed per decoder because, under bf16, it moves ~0.2% of characters: two decodes are
    comparable only at the same batch size.
    """

    def __init__(self, model_ref, model, extractor, device, batch_size: int = 16) -> None:
        import torch

        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.model_ref = str(model_ref)
        self.model = model
        self.extractor = extractor
        self.device = torch.device(device)
        self.batch_size = batch_size

    @property
    def autocast(self) -> bool:
        """Whether forwards run under bf16 autocast -- on CUDA only; CPU has no such path."""
        return self.device.type == "cuda"

    def fingerprint(self, mode: str) -> DecodeFingerprint:
        """The identity of a decode this decoder makes in ``mode`` (``SPANS`` or a stream).

        Read off the decoder as it actually runs rather than from what a caller asked for, so
        the same ``--weights-dtype`` on CPU and on CUDA fingerprints differently, as it decodes
        differently.
        """
        short = {torch_name: key for key, torch_name in WEIGHTS_DTYPES.items()}
        dtypes = {str(p.dtype).removeprefix("torch.") for p in self.model.parameters()}
        return DecodeFingerprint(
            model=self.model_ref,
            mode=mode,
            weights_dtype="+".join(sorted(short.get(d, d) for d in dtypes)),
            batch_size=self.batch_size,
            device_type=self.device.type,
            autocast=self.autocast,
        )

    @classmethod
    def load(
        cls,
        model_ref: str | Path,
        device,
        *,
        weights_dtype: str,
        batch_size: int = 16,
        use_ema: bool = False,
        revision: str | None = None,
    ) -> "Decoder":
        """Load a hub id, a saved model directory, or a distillation checkpoint file.

        ``weights_dtype`` is a key of :data:`WEIGHTS_DTYPES`. ``use_ema`` selects a
        checkpoint's averaged weights; asking for them from anything else is an error rather
        than a silent no-op. ``revision`` pins every hub fetch the load makes -- the model and
        its feature extractor for a hub id, the (teacher's) feature extractor for a checkpoint
        -- and is recorded in the fingerprint's ``model`` as ``<ref>@<revision>``. Without it
        the load is unchanged: the hub's default branch, and the bare reference.
        """
        import torch
        from transformers import SeamlessM4TFeatureExtractor

        device = torch.device(device)
        if weights_dtype not in WEIGHTS_DTYPES:
            raise ValueError(f"weights_dtype must be one of {sorted(WEIGHTS_DTYPES)}")
        if device.type != "cuda" and weights_dtype != "fp32":
            raise ValueError(
                f"{weights_dtype} weights on {device} would run without the CUDA autocast "
                "that reconciles them with fp32 features; decode on CUDA or with fp32."
            )
        path = Path(model_ref)
        if path.is_file():
            model, _, _ = load_student_from_checkpoint(path, torch.device("cpu"), use_ema)
            extractor_source = STUDENT_FEATURE_EXTRACTOR
        else:
            if use_ema:
                raise ValueError(
                    f"{model_ref} is not a distillation checkpoint, so it has no averaged "
                    "weights to select."
                )
            model = load_hf_model(str(model_ref), revision)
            extractor_source = str(model_ref)
        model = model.to(device=device, dtype=getattr(torch, WEIGHTS_DTYPES[weights_dtype]))
        extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            extractor_source, **_revision_kwargs(revision)
        )
        name = model_ref if revision is None else f"{model_ref}@{revision}"
        return cls(name, model, extractor, device, batch_size)

    def _phoneme_logits(self, features, attention_mask=None):
        """The phoneme head's float32 logits, ``(batch, timesteps, classes)``, on the device."""
        import torch

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            logits = self.model(
                features.to(self.device),
                attention_mask=attention_mask,
                return_dict=True,
            )["logits"][PHONEME_LEVEL]
        return logits.float()

    def _class_ids(self, features, attention_mask=None) -> np.ndarray:
        """Per-timestep argmax over the phoneme head, ``(batch, timesteps)``."""
        return self._phoneme_logits(features, attention_mask).argmax(dim=-1).cpu().numpy()

    def window_rows(self, windows: list[np.ndarray]) -> list[np.ndarray]:
        """Each full 5 s window's argmax row on the deployed 125-timestep lattice.

        Windows are extracted in batches but each is normalized on its own, as on device.
        All are the same length, so no attention mask is passed -- the fixed-shape forward
        the device runs.
        """
        rows: list[np.ndarray] = []
        for offset in range(0, len(windows), self.batch_size):
            features = self.extractor(
                windows[offset : offset + self.batch_size],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            ).input_features
            rows.extend(row[:DEPLOYED_LOGIT_FRAMES] for row in self._class_ids(features))
        return rows

    def emissions(
        self, samples: np.ndarray, block: int = DEPLOYED_BLOCK, flush_tail: bool = True
    ) -> list[Emission]:
        """The streaming protocol over one 16 kHz clip, committing ``block``, with provenance."""
        block_bounds(block)  # refuse a bad block before paying for the forward passes
        return stream_emissions(self.window_rows(window_audio(samples)), block, flush_tail)

    def decode_stream(
        self, samples: np.ndarray, block: int = DEPLOYED_BLOCK, flush_tail: bool = True
    ) -> str:
        """The decode the streaming protocol commits for one clip at ``block``."""
        return tokens_to_phonemes(e.token_id for e in self.emissions(samples, block, flush_tail))

    def device_rows(self, pieces: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Each piece of at most 5 s, featurised and decoded as the app decodes it (module
        docstring): its argmax over the output timesteps its real feature pairs reach.

        Each piece is extracted on its own, so it is normalized over its own frames, and the
        model's 250 feature pairs are filled by repeating its last real pair. Every forward is
        the fixed shape the device runs, so no attention mask is passed.
        """
        import torch

        rows: list[np.ndarray] = []
        for offset in range(0, len(pieces), self.batch_size):
            features, steps = [], []
            for piece in pieces[offset : offset + self.batch_size]:
                pairs, real = self.device_pairs(piece)
                features.append(pairs)
                steps.append(device_output_steps(real))
            class_ids = self._class_ids(torch.stack(features))
            if class_ids.shape[1] != DEPLOYED_LOGIT_FRAMES:
                raise ValueError(f"the model emits {class_ids.shape[1]} timesteps for 5 s, not 125")
            rows.extend(row[:n] for row, n in zip(class_ids, steps))
        return rows

    def device_pairs(self, piece: np.ndarray):
        """The model's 250 feature pairs for one piece as the app builds them, and how many of
        them are real (``Models/MelFeatureExtractor+iOS.swift:81-166``).

        The teacher's extractor computes the same features, except for a single frame
        (400-559 samples): its ``ddof=1`` variance is NaN there, while the device takes the
        variance as 0 (:107-115), so the frame normalizes to zeros. The padding frame after it
        is 1.0 on both.
        """
        import torch

        if not FRAME_SAMPLES <= len(piece) <= WINDOW_SAMPLES:
            raise ValueError(f"a piece of {len(piece)} samples is not one the app decodes")
        (pairs,) = self.extractor(
            [piece], sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        ).input_features
        if (len(piece) - FRAME_SAMPLES) // FRAME_HOP_SAMPLES == 0:  # one frame
            pairs = pairs.clone()
            pairs[0, : pairs.shape[1] // 2] = 0.0
        if torch.isnan(pairs).any():
            raise ValueError(f"the features of a {len(piece)}-sample piece are not finite")
        repeated = pairs[-1:].expand(DEVICE_FEATURE_PAIRS - len(pairs), -1)
        return torch.cat([pairs, repeated]), len(pairs)

    def app_checks(self, samples: np.ndarray) -> list[dict]:
        """The checks Muraja's transcriber hands its engine over one 16 kHz item, in order
        (:func:`check_schedule`, :func:`app_check`; fingerprinted as :data:`APP_CHECKS`)."""
        return self.app_checks_at(samples, (DEPLOYED_BLOCK,))[DEPLOYED_BLOCK]

    def app_checks_at(self, samples: np.ndarray, blocks: Sequence[int]) -> dict[int, list[dict]]:
        """The item's check sequence for a transcriber committing each of ``blocks``
        (:func:`app_checks_mode`), all cut from one inference per scheduled check: the blocks
        differ only in what each inference sends."""
        spans = check_schedule(len(samples))
        rows = self.device_rows([samples[s.start : s.end] for s in spans])
        return {block: app_checks(spans, rows, block) for block in blocks}

    def span_class_ids(self, spans: Iterable[np.ndarray]) -> list[np.ndarray]:
        """Each 16 kHz span's per-timestep argmax from one whole variable-length pass.

        Spans are batched with padding under a real attention mask, and each row is cut to
        its own valid length (the model's own length mapping, the one the CTC loss uses), so
        padding never reaches a row. ``spans`` is consumed lazily, one batch at a time, so a
        caller can stream them from disk. This is the frame-level view a tool needs to place
        pauses on words (:func:`tadabur.waqf_detect.segment_clip`); :meth:`decode_spans`
        collapses the same rows.
        """
        return self._span_rows(spans, lambda logits: logits.argmax(dim=-1).cpu().numpy())

    def span_log_posteriors(self, spans: Iterable[np.ndarray]) -> list[np.ndarray]:
        """Each span's per-timestep log-posteriors, ``(timesteps, classes)`` float32.

        The log-softmax of the same pass :meth:`span_class_ids` makes, cut the same way. A
        row's argmax is that method's row except where two logits are so close that the
        softmax's rounding ties them, so a caller that needs both should check them.
        """
        return self._span_rows(spans, lambda logits: logits.log_softmax(dim=-1).cpu().numpy())

    def _span_rows(self, spans: Iterable[np.ndarray], head) -> list[np.ndarray]:
        """``head(logits)`` of every span, batched and cut as :meth:`span_class_ids` says."""
        rows: list[np.ndarray] = []
        pending = iter(spans)
        while batch := list(islice(pending, self.batch_size)):
            extracted = self.extractor(
                [np.asarray(span, dtype=np.float32) for span in batch],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            mask = extracted.attention_mask.to(self.device)
            outputs = head(self._phoneme_logits(extracted.input_features, mask))
            valid = self.model._get_feat_extract_output_lengths(mask.sum(dim=1)).tolist()
            rows.extend(row[: int(length)] for row, length in zip(outputs, valid))
        return rows

    def decode_spans(self, spans: Iterable[np.ndarray]) -> list[str]:
        """Each 16 kHz span decoded whole: its :meth:`span_class_ids` row, collapsed."""
        return [
            tokens_to_phonemes(seg.token_id for seg in scan_ctc(row))
            for row in self.span_class_ids(spans)
        ]
