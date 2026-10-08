"""Fixed-window geometry: the 5 s training/inference window over the recitation and its lattices.

Every whole-clip training example is a **fixed window over the un-waqf-segmented recitation**
(the A2 frozen contract, #24), not an individual waqf segment. This module owns that grid and
the frame geometry it rests on, so the label builders (:mod:`training.windowed_labels`,
:mod:`training.segmented_labels`), the batch path, the duration envelope
(:mod:`training.window_envelope`) and the audit tooling all cut the same windows:

* **Two lattices.** The Muaalem feature extractor emits one **20 ms encoder frame** per 320
  samples at 16 kHz (named a *teacher frame* here, after the 20 ms lattice the encoder and
  the Recitation VAD share). Muaalem's single stride-2 adapter conv maps that to the **40 ms
  output lattice** the phoneme CTC head emits on (a *student frame*). One student frame owns
  two teacher frames; :func:`muaalem_lattice_length` is the exact conv relation and
  :func:`feature_frames_for_samples` the extractor's exact frame count for an audio span.

* **The window contract.** The window *length* (5 s / 250 feature frames) is the
  already-deployed inference window (``convert_to_coreml.py``, ``ml-model-transformation.md``).
  The window **spacing** is the legacy training-label grid ADR-0004 froze: a center-trusted
  1 s overlap (4 s hop / 200 feature frames). It is **not** the device's spacing — Muraja
  runs a 1 s hop and commits only each window's first second (ADR-0010; see ADR-0004's
  supersession notice). :class:`WindowContract` defaults to the training grid and takes
  the spacing as a parameter.

* **The recitation grid.** :func:`recitation_window_span` locates the recitation in the
  clip on the 40 ms lattice, :func:`enumerate_recitation_windows` tiles it with clip-relative
  window starts, and :func:`clip_recitation_windows` (the single entry point) snaps each
  window inward to the whole words it contains.

Everything here is torch-free and deterministic, so it is unit-tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

# Muaalem's single stride-2, kernel-3 adapter conv maps the 20 ms encoder lattice to
# the 40 ms CTC lattice (``ml-model-transformation.md``; config ``add_adapter``,
# ``num_adapter_layers=1``, ``adapter_kernel_size=3``, ``adapter_stride=2``). Pinned
# here so window lengths land on the exact frames the phoneme head — and the CTC target
# length — use, without loading the model.
ADAPTER_KERNEL = 3
ADAPTER_STRIDE = 2
ADAPTER_PADDING = ADAPTER_KERNEL // 2

# One 40 ms student frame consumes two 20 ms teacher (encoder) frames.
TEACHER_FRAMES_PER_STUDENT = 2

# The staged clips are 16 kHz mono (``tadabur.audio.TARGET_SAMPLE_RATE``) and the encoder
# frames them at 20 ms, so one teacher frame spans 320 samples. This is what lets the
# sample-domain window contract cut window waveforms on exact teacher-frame boundaries.
TARGET_SAMPLE_RATE = 16000
TEACHER_FRAME_MS = 20
SAMPLES_PER_TEACHER_FRAME = TARGET_SAMPLE_RATE * TEACHER_FRAME_MS // 1000  # 320

# A window start must land on an even teacher frame so its student frames line up with the
# 40 ms lattice (:class:`WindowContract` enforces this for the hop). The recitation origin
# the clip-relative grid is shifted to (:func:`recitation_window_span`) obeys the same
# rule, so it is floored to a whole student-frame pair: 2 teacher frames × 320 samples =
# 640 samples (40 ms). Flooring pulls in at most one 40 ms lead-in frame — well within the
# ±50 ms edge pad ``waqf_detect`` already leaves — and never drops recitation audio.
SAMPLES_PER_STUDENT_FRAME = SAMPLES_PER_TEACHER_FRAME * TEACHER_FRAMES_PER_STUDENT  # 640

# The constant offset in the feature extractor's frame count (see
# :func:`feature_frames_for_samples`): a 25 ms Kaldi analysis window over a 10 ms shift,
# stacked 2:1, loses half a 10 ms hop worth of samples relative to the naive
# ``num_samples // 320``.
FEATURE_FRAME_SAMPLE_OFFSET = 80


def feature_frames_for_samples(num_samples: int) -> int:
    """The 20 ms encoder frames the Muaalem feature extractor emits for ``num_samples``.

    ``SeamlessM4TFeatureExtractor`` computes 10 ms Kaldi fbank frames and stacks them
    ``stride=2``, dropping the odd remainder, which comes out **exactly**
    ``(num_samples - 80) // 320`` — *not* the naive ``num_samples // 320``, which
    over-counts by one for most window lengths (e.g. a 66 287-sample window: 207 vs the
    real 206). This one expression is what makes the phoneme CTC lattice
    (:mod:`training.windowed_labels`) match what the model actually emits. Verified
    against the real extractor in ``test_windowing``.
    """
    return max((num_samples - FEATURE_FRAME_SAMPLE_OFFSET) // SAMPLES_PER_TEACHER_FRAME, 0)


# The deployed fixed inference window: 250 feature frames ≈ 5 s at 20 ms
# (``convert_to_coreml.py`` ``FIXED_SEQ_LEN``). Its 40 ms length is 125.
DEPLOYED_WINDOW_FEATURE_FRAMES = 250

# The frozen training-label window spacing (#24, A2 HITL freeze): a 4 s hop = 1 s overlap
# over the 5 s window (center-trusted overlap). 200 feature frames is even, so every window
# still starts on an even teacher frame and its student frames line up with the clip's
# 40 ms lattice. This is the legacy training grid only; Muraja's inference uses a 1 s hop
# (ADR-0004 supersession notice, ADR-0010).
FROZEN_HOP_FEATURE_FRAMES = 200


def muaalem_lattice_length(feature_frames: int) -> int:
    """40 ms student-lattice length for a 20 ms encoder length ``feature_frames``.

    Mirrors ``Wav2Vec2BertModel._get_feat_extract_output_lengths`` for the single
    stride-2 adapter conv (kernel 3, pad 1): ``floor((T-1)/2) + 1 == ceil(T/2)``. For
    the fixed 5 s export window (T≈250) this is 125.
    """
    return (feature_frames + 2 * ADAPTER_PADDING - ADAPTER_KERNEL) // ADAPTER_STRIDE + 1


@dataclass(frozen=True)
class WindowContract:
    """How the un-waqf-segmented recitation is cut into fixed training windows.

    ``feature_frames`` is the window length on the 20 ms encoder grid — the deployed 5 s
    inference window (250). ``hop_feature_frames`` is the step between consecutive window
    starts on that grid; the default is :data:`FROZEN_HOP_FEATURE_FRAMES` (200 = a 4 s
    hop, 1 s overlap), the **center-trusted overlap** frozen by #24 (A2 HITL) for the
    training labels. It is not the device's spacing: Muraja runs a 1 s hop
    (ADR-0004 supersession notice, ADR-0010).

    Both are required to be **even** so every window starts on an even teacher frame and
    its student frames line up exactly with the clip's 40 ms lattice (``start // 2``);
    an odd start would split a teacher pair across two windows and reintroduce ±1-frame
    boundary drift. The window is cut in the **sample** domain (:attr:`window_samples` /
    :attr:`hop_samples`) because each window's audio is fed to the model on its own.
    """

    feature_frames: int = DEPLOYED_WINDOW_FEATURE_FRAMES
    hop_feature_frames: int = FROZEN_HOP_FEATURE_FRAMES

    def __post_init__(self) -> None:
        if self.feature_frames <= 0 or self.feature_frames % 2 != 0:
            raise ValueError(f"feature_frames must be a positive even int, got {self.feature_frames}")
        if self.hop_feature_frames <= 0 or self.hop_feature_frames % 2 != 0:
            raise ValueError(
                f"hop_feature_frames must be a positive even int, got {self.hop_feature_frames}"
            )

    @property
    def student_frames(self) -> int:
        """The full-window 40 ms lattice length (125 for the deployed 5 s window)."""
        return muaalem_lattice_length(self.feature_frames)

    @property
    def window_samples(self) -> int:
        """The window length in 16 kHz samples."""
        return self.feature_frames * SAMPLES_PER_TEACHER_FRAME

    @property
    def hop_samples(self) -> int:
        """The step between consecutive window starts, in 16 kHz samples."""
        return self.hop_feature_frames * SAMPLES_PER_TEACHER_FRAME


@dataclass(frozen=True)
class Window:
    """One fixed training window over a clip's 16 kHz waveform.

    ``start_sample`` is the (teacher-frame-aligned) sample offset and ``num_samples`` is
    the window's sample length, clamped at the clip end so a tail window covers fewer
    samples than a full window.
    """

    index: int
    start_sample: int
    num_samples: int

    @property
    def start_feature_frame(self) -> int:
        """The window start on the 20 ms encoder grid (an even frame, for provenance)."""
        return self.start_sample // SAMPLES_PER_TEACHER_FRAME

    @property
    def start_student_frame(self) -> int:
        """The window start on the clip's 40 ms lattice (``start_feature_frame // 2``)."""
        return self.start_feature_frame // TEACHER_FRAMES_PER_STUDENT


def recitation_window_span(start_s: float, end_s: float) -> tuple[int, int]:
    """Clip-relative ``(start_sample, num_samples)`` of the recitation to be windowed.

    The training unit is a fixed window over the **un-waqf-segmented recitation**, *not*
    the whole staged clip: a Tadabur clip keeps the previous ayah's tail as lead-in and
    sometimes a trailing word/takbir (``waqf_detect`` re-cuts the outer segment edges to
    the matched span, so ``start_s`` is generally > 0 and ``end_s`` < the clip duration).
    Windowing the clip instead would feed the CTC head neighbour-ayah audio with no target.
    ``start_sample`` is floored to a whole 40 ms student-frame pair
    (:data:`SAMPLES_PER_STUDENT_FRAME`) so every window still begins on the 40 ms lattice;
    ``num_samples`` spans to ``end_s``. The offset is **clip-relative**, so windows are
    keyed by the same clip-origin ``start_sample`` wherever they are cut.
    """
    start = (round(start_s * TARGET_SAMPLE_RATE) // SAMPLES_PER_STUDENT_FRAME) * SAMPLES_PER_STUDENT_FRAME
    end = round(end_s * TARGET_SAMPLE_RATE)
    return start, max(0, end - start)


def enumerate_windows(num_samples: int, contract: WindowContract) -> list[Window]:
    """Fixed training windows tiling ``num_samples`` of waveform under ``contract``.

    Windows start at samples ``0, hop_samples, 2*hop_samples, …`` while the start is
    inside the clip; each covers up to ``contract.window_samples`` samples, clamped at
    the clip end. An empty clip yields no windows. Every start is a multiple of
    ``hop_samples`` (an even number of 320-sample teacher frames), so each window begins
    on an even teacher frame and lands exactly on the clip's 40 ms lattice.
    """
    windows: list[Window] = []
    start = 0
    while start < num_samples:
        windows.append(
            Window(
                index=len(windows),
                start_sample=start,
                num_samples=min(contract.window_samples, num_samples - start),
            )
        )
        start += contract.hop_samples
    return windows


def enumerate_recitation_windows(
    recitation_start_sample: int, recitation_num_samples: int, contract: WindowContract
) -> list[Window]:
    """Fixed windows tiling the recitation span, with **clip-relative** start samples.

    Enumerates the same 0-based grid as :func:`enumerate_windows` over the recitation's
    ``recitation_num_samples`` and shifts every window start by ``recitation_start_sample``
    (a whole student-frame pair — see :func:`recitation_window_span`), so the returned
    ``Window.start_sample`` locates the window in the **whole clip** while its length and
    count come from the recitation. A **redundant trailing window** — one whose audio ends
    no later than the previous window's (pure overlap the previous window already
    covers) — is dropped, so the grid carries only windows
    with new center audio.
    """
    if recitation_start_sample % SAMPLES_PER_STUDENT_FRAME != 0:
        raise ValueError(
            f"recitation_start_sample {recitation_start_sample} must be a multiple of "
            f"{SAMPLES_PER_STUDENT_FRAME} (a 40 ms student-frame pair); use recitation_window_span"
        )
    windows: list[Window] = []
    prev_end = -1
    for w in enumerate_windows(recitation_num_samples, contract):
        end = w.start_sample + w.num_samples
        if end <= prev_end:
            break  # a fully-overlapped tail window: no new center audio, drop it
        windows.append(
            Window(
                index=w.index,
                start_sample=recitation_start_sample + w.start_sample,
                num_samples=w.num_samples,
            )
        )
        prev_end = end
    return windows


def snap_window_to_words(
    window: Window, word_times: tuple[float, ...]
) -> Window | None:
    """``window`` shrunk to the whole words its audio contains, or ``None`` if none fit.

    A fixed grid window's edge almost never falls on a word boundary, so its raw audio
    holds a fragment of the edge word: labelling that fragment as a whole word corrupts
    the CTC target, and omitting it leaves spoken audio with no target. Snapping the
    window **inward** to the first/last whole word (word ``j`` spans
    ``[word_times[j], word_times[j + 1])``, clip-relative seconds) removes both failure
    modes, and because the snapped span is a sub-span of the fixed window the frozen grid
    is preserved. Edges are rounded **in** to the window's own 640-sample (40 ms
    student-frame) lattice so the snapped window still starts and ends on a whole student
    frame; that trims at most one 40 ms frame into each edge word. Returns ``None`` when no
    whole word fits (the window sits inside one long word), and the caller drops that
    window from the grid deterministically.
    """
    start = window.start_sample
    end = start + window.num_samples
    first = last = None
    for index in range(len(word_times) - 1):
        onset = round(word_times[index] * TARGET_SAMPLE_RATE)
        offset = round(word_times[index + 1] * TARGET_SAMPLE_RATE)
        if onset >= start and offset <= end:
            if first is None:
                first = onset
            last = offset
    if first is None or last is None or last <= first:
        return None
    frame = SAMPLES_PER_STUDENT_FRAME
    snapped_start = start + -(-(first - start) // frame) * frame
    snapped_end = start + ((last - start) // frame) * frame
    if snapped_end <= snapped_start:
        return None
    return Window(
        index=window.index,
        start_sample=snapped_start,
        num_samples=snapped_end - snapped_start,
    )


def clip_recitation_windows(
    recitation_start_sample: int,
    recitation_num_samples: int,
    contract: WindowContract,
    word_times: tuple[float, ...] = (),
) -> list[Window]:
    """The clip's training windows: the fixed recitation grid, word-snapped when possible.

    With ``word_times`` this is :func:`enumerate_recitation_windows` followed by
    :func:`snap_window_to_words` per window (dropping the windows no whole word fits in);
    without it, the raw fixed grid. **This is the single entry point every window consumer
    uses** (:mod:`training.windowed_labels`, :mod:`training.segmented_labels`), so a clip's
    ``(index, start_sample, num_samples)`` triples are identical wherever they are cut.
    Window indices come from the nominal grid and may therefore skip.
    """
    windows = enumerate_recitation_windows(
        recitation_start_sample, recitation_num_samples, contract
    )
    if not word_times:
        return windows
    snapped = [snap_window_to_words(w, word_times) for w in windows]
    kept = [w for w in snapped if w is not None]
    # Snapping can collapse two neighbouring windows onto the same word run; keep the
    # first of each duplicate span so the grid carries no redundant training example.
    unique: list[Window] = []
    for window in kept:
        if unique and (
            window.start_sample == unique[-1].start_sample
            and window.num_samples == unique[-1].num_samples
        ):
            continue
        unique.append(window)
    return unique
