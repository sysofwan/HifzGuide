"""Waqf-head distillation soft labels: per-window VAD teacher pooled to Muaalem's 40 ms.

The waqf head (ADR-0004) is a per-frame silence classifier riding the Muaalem adapter
+ CTC output — the **40 ms** post-downsample lattice — distilled from the Recitation
VAD (``obadx/recitation-segmenter-v2``), whose frame classifier runs at **20 ms**.
Distillation is therefore **2:1**: the teacher's 20 ms silence posteriors are pooled to
the 40 ms grid before the KL. This module owns that pooling, the per-training-window
frame alignment, and the persisted soft-label artifact; the torch VAD forward pass
lives in :mod:`tadabur.vad`.

**The teacher runs over the training windows, not the whole clip.** The deployed /
student model sees fixed windows (5 s), and a transformer frame classifier's
window-local posteriors differ from a whole-clip pass (attention context and
window-edge padding change), so slicing whole-clip posteriors would not match the
student examples. The generator therefore cuts each clip's waveform into the same
fixed windows the student uses (:func:`slice_windows`), runs the VAD on each **window
waveform** (:class:`tadabur.vad.RecitationVad`), and pools each window's *own* 20 ms
posteriors to its exact 40 ms length.

The pooling and frame-alignment are **torch-free and deterministic** so they can be
unit-tested (golden fixtures) without a GPU. The pinned rule is:

* **Student frame ``i`` owns teacher frames ``2i`` and ``2i+1``** — a non-overlapping
  pair, left-anchored (frame 0 of both lattices starts at sample 0). Because a 1–2 frame
  drift between the two feature extractors moves a boundary snap across a word edge,
  anchoring the pairing at index 0 keeps every interior boundary on its true timestamp
  and confines the drift to the window tail, where :func:`pool_silence_2to1` reconciles it.
* **A student frame is silent iff both its teacher frames are** — so the pooled *silence*
  posterior is the **min** of the pair (equivalently, max-pool the speech posterior),
  matching ADR-0004's "a window is silent iff its two teacher frames are".

The window *length* (5 s / 250 feature frames) is the already-deployed inference window
(``convert_to_coreml.py``, ``ml-model-transformation.md``). The window **spacing**
(overlap / edge-ownership / stitch) is the inference contract frozen by #24 (A2 HITL): a
**center-trusted 1 s overlap** (4 s hop / 200 feature frames), used identically in train,
eval, and export (ADR-0004 "Frozen windowing contract"). :class:`WindowContract` defaults
to that spacing and takes it as a parameter. The exact generation contract (window/hop,
pooling rule, adapter + frame geometry, VAD id) is persisted as :class:`SoftLabelStore`
metadata, so a resumed run that would mix labels from a *different* contract **fails fast**
instead of silently corrupting the artifact; regenerating under a new contract is a fresh
store.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tadabur.manifest import ManifestRecord, read_records
from tadabur.vad import VAD_MODEL_ID

# Muaalem's single stride-2, kernel-3 adapter conv maps the 20 ms encoder lattice to
# the 40 ms CTC lattice (``ml-model-transformation.md``; config ``add_adapter``,
# ``num_adapter_layers=1``, ``adapter_kernel_size=3``, ``adapter_stride=2``). Pinned
# here so the soft labels land on the exact student frames the phoneme head — and the
# CTC target length — use, without loading the model to generate them.
ADAPTER_KERNEL = 3
ADAPTER_STRIDE = 2
ADAPTER_PADDING = ADAPTER_KERNEL // 2

# One 40 ms student frame consumes two 20 ms teacher frames.
TEACHER_FRAMES_PER_STUDENT = 2

# The staged clips are 16 kHz mono (``tadabur.audio.TARGET_SAMPLE_RATE``) and the VAD
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

# The deployed fixed inference window: 250 feature frames ≈ 5 s at 20 ms
# (``convert_to_coreml.py`` ``FIXED_SEQ_LEN``; ADR-0004). Its 40 ms length is 125.
DEPLOYED_WINDOW_FEATURE_FRAMES = 250

# The frozen window spacing (#24, A2 HITL freeze): a 4 s hop = 1 s overlap over the
# 5 s window (center-trusted overlap; see ADR-0004 "Frozen windowing contract"). 200
# feature frames is even, so every window still starts on an even teacher frame and its
# student frames line up with the clip's 40 ms lattice. Train, eval, and export use this
# identical spacing.
FROZEN_HOP_FEATURE_FRAMES = 200

# The pinned pooling rule, recorded in the store contract so a resume under a different
# rule is rejected rather than silently mixed into an existing artifact.
POOLING_RULE = "min-silence-2to1-left-anchored"


def muaalem_lattice_length(feature_frames: int) -> int:
    """40 ms student-lattice length for a 20 ms encoder length ``feature_frames``.

    Mirrors ``Wav2Vec2BertModel._get_feat_extract_output_lengths`` for the single
    stride-2 adapter conv (kernel 3, pad 1): ``floor((T-1)/2) + 1 == ceil(T/2)``. For
    the fixed 5 s export window (T≈250) this is 125, per ADR-0004.
    """
    return (feature_frames + 2 * ADAPTER_PADDING - ADAPTER_KERNEL) // ADAPTER_STRIDE + 1


def _reconcile_teacher_length(silence_20ms: np.ndarray, needed_frames: int) -> np.ndarray:
    """Left-anchor ``silence_20ms`` to exactly ``needed_frames`` teacher frames.

    Extra tail frames are dropped and a short tail is edge-held (repeat the last
    frame), so the ±few-frame feature-extractor drift is absorbed at the window end —
    never by shifting an interior frame boundary. An empty teacher (no frames) with a
    non-zero requirement is a data-integrity failure, not something to pad from nothing.
    """
    have = len(silence_20ms)
    if have == needed_frames:
        return silence_20ms
    if have > needed_frames:
        return silence_20ms[:needed_frames]
    if have == 0:
        raise ValueError(
            f"cannot reconcile 0 teacher frames up to {needed_frames}: no silence signal"
        )
    return np.concatenate(
        [silence_20ms, np.full(needed_frames - have, silence_20ms[-1], dtype=silence_20ms.dtype)]
    )


def pool_silence_2to1(silence_20ms: np.ndarray, num_student_frames: int) -> np.ndarray:
    """Pool 20 ms teacher silence posteriors to a 40 ms student lattice, 2:1.

    Returns one ``P(silence)`` per 40 ms student frame under the pinned rule (see the
    module docstring): student ``i`` = ``min(teacher[2i], teacher[2i+1])``. The teacher
    is first reconciled to exactly ``2*num_student_frames`` frames
    (:func:`_reconcile_teacher_length`), so a drifted teacher still yields exactly
    ``num_student_frames`` targets aligned to the Muaalem lattice.
    """
    if num_student_frames < 0:
        raise ValueError(f"num_student_frames must be non-negative, got {num_student_frames}")
    needed = TEACHER_FRAMES_PER_STUDENT * num_student_frames
    aligned = _reconcile_teacher_length(np.asarray(silence_20ms, dtype=np.float32), needed)
    pairs = aligned.reshape(num_student_frames, TEACHER_FRAMES_PER_STUDENT)
    return pairs.min(axis=1)


@dataclass(frozen=True)
class WindowContract:
    """How the un-waqf-segmented recitation is cut into fixed training windows.

    ``feature_frames`` is the window length on the 20 ms teacher/encoder grid — the
    deployed 5 s inference window (250). ``hop_feature_frames`` is the step between
    consecutive window starts on that grid; the default is
    :data:`FROZEN_HOP_FEATURE_FRAMES` (200 = a 4 s hop, 1 s overlap), the
    **center-trusted overlap** frozen by #24 (A2 HITL). Train, eval, and export use this
    identical spacing (ADR-0004 "Frozen windowing contract").

    Both are required to be **even** so every window starts on an even teacher frame and
    its student frames line up exactly with the clip's 40 ms lattice (``start // 2``);
    an odd start would split a teacher pair across two windows and reintroduce the
    ±1-frame boundary drift the alignment pins down. The window is cut in the **sample**
    domain (:attr:`window_samples` / :attr:`hop_samples`) because the VAD runs over the
    window *waveform*, not a slice of whole-clip posteriors.
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
        """The window length in 16 kHz samples the VAD waveform slice spans."""
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
    samples than a full window. The VAD is run on ``waveform[start_sample : start_sample
    + num_samples]`` to get this window's *own* 20 ms posteriors — the student length is
    then :func:`muaalem_lattice_length` of however many teacher frames the VAD emits, so
    the soft target is reconciled to each window's exact logit length.
    """

    index: int
    start_sample: int
    num_samples: int

    @property
    def start_feature_frame(self) -> int:
        """The window start on the 20 ms teacher grid (an even frame, for provenance)."""
        return self.start_sample // SAMPLES_PER_TEACHER_FRAME

    @property
    def start_student_frame(self) -> int:
        """The window start on the clip's 40 ms lattice (``start_feature_frame // 2``)."""
        return self.start_feature_frame // TEACHER_FRAMES_PER_STUDENT


def recitation_window_span(start_s: float, end_s: float) -> tuple[int, int]:
    """Clip-relative ``(start_sample, num_samples)`` of the recitation to be windowed.

    The training unit is a fixed window over the **un-waqf-segmented recitation**
    (ADR-0004), *not* the whole staged clip: a Tadabur clip keeps the previous ayah's
    tail as lead-in and sometimes a trailing word/takbir (``waqf_detect`` re-cuts the
    outer segment edges to the matched span, so ``start_s`` is generally > 0 and ``end_s``
    < the clip duration). Both the phoneme CTC labels (:mod:`training.windowed_labels`)
    and the waqf soft labels (:func:`generate_soft_labels`) window this same span so their
    per-window targets pair on one grid; windowing the clip instead would (a) feed the CTC
    head neighbour-ayah audio with no target and (b) put the two heads on different
    window origins/counts — silent joint-label corruption (ADR-0004 "same window
    contract"). ``start_sample`` is floored to a whole 40 ms student-frame pair
    (:data:`SAMPLES_PER_STUDENT_FRAME`) so every window still begins on the 40 ms lattice;
    ``num_samples`` spans to ``end_s``. The offset is **clip-relative**, so both artifacts
    key their windows by the same clip-origin ``start_sample``.
    """
    start = (round(start_s * TARGET_SAMPLE_RATE) // SAMPLES_PER_STUDENT_FRAME) * SAMPLES_PER_STUDENT_FRAME
    end = round(end_s * TARGET_SAMPLE_RATE)
    return start, max(0, end - start)


def enumerate_recitation_windows(
    recitation_start_sample: int, recitation_num_samples: int, contract: WindowContract
) -> list[Window]:
    """Fixed windows tiling the recitation span, with **clip-relative** start samples.

    Enumerates the same 0-based grid as :func:`enumerate_windows` over the recitation's
    ``recitation_num_samples`` and shifts every window start by ``recitation_start_sample``
    (a whole student-frame pair — see :func:`recitation_window_span`), so the returned
    ``Window.start_sample`` locates the window in the **whole clip** while its length and
    count come from the recitation. A **redundant trailing window** — one whose audio ends
    no later than the previous window's (pure overlap the previous window already covers,
    which the inference stitch discards) — is dropped, so the grid carries only windows
    with new center audio. This is the single grid both the phoneme labels and the waqf
    soft labels enumerate, guaranteeing identical ``(index, start_sample, num_samples)``
    per clip.
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


def slice_windows(
    waveform: np.ndarray, contract: WindowContract
) -> list[tuple[Window, np.ndarray]]:
    """Every training window of a clip paired with its **waveform slice**.

    Enumerates the windows under ``contract`` and cuts the matching sample span from
    ``waveform``, so each slice is fed to the VAD independently — window-local posteriors
    that match the student's fixed-window examples, not a whole-clip pass.
    """
    return [
        (w, np.asarray(waveform, dtype=np.float32)[w.start_sample : w.start_sample + w.num_samples])
        for w in enumerate_windows(len(waveform), contract)
    ]


def slice_recitation_windows(
    waveform: np.ndarray,
    recitation_start_sample: int,
    recitation_num_samples: int,
    contract: WindowContract,
) -> list[tuple[Window, np.ndarray]]:
    """Each recitation window paired with its waveform slice, clip-relative start samples.

    Like :func:`slice_windows` but over the recitation span
    (:func:`enumerate_recitation_windows`): the waveform slice is cut at the window's
    **clip-relative** ``start_sample`` (``waveform`` is the whole clip), so the VAD sees
    exactly the recitation audio the phoneme head is labelled on — no neighbour-ayah
    lead-in / trailing bleed — on the shared grid.
    """
    wave = np.asarray(waveform, dtype=np.float32)
    return [
        (w, wave[w.start_sample : w.start_sample + w.num_samples])
        for w in enumerate_recitation_windows(
            recitation_start_sample, recitation_num_samples, contract
        )
    ]


def pool_window_posteriors(window_silence_20ms: np.ndarray) -> np.ndarray:
    """Pool one window's own 20 ms VAD posteriors to its exact Muaalem 40 ms length.

    The student length is :func:`muaalem_lattice_length` of the teacher frames the VAD
    emitted for this window slice, so student frame ``j`` owns window teacher frames
    ``2j`` / ``2j+1`` (:func:`pool_silence_2to1`) — the pinned, drift-checked mapping,
    now anchored at the window start rather than the clip start.
    """
    return pool_silence_2to1(
        window_silence_20ms, muaalem_lattice_length(len(window_silence_20ms))
    )


def generation_contract(contract: WindowContract) -> dict:
    """The exact parameters the emitted soft labels depend on, for store metadata.

    Recorded once per :class:`SoftLabelStore` so a resume that would append labels
    generated under a *different* window/hop, pooling rule, adapter/frame geometry, or
    VAD fails fast instead of silently mixing incompatible targets (the labels are keyed
    only by ``audio_filename``, so a changed contract would otherwise skip every existing
    clip and leave stale arrays — silent training-label corruption).
    """
    return {
        "window_feature_frames": contract.feature_frames,
        "hop_feature_frames": contract.hop_feature_frames,
        "samples_per_teacher_frame": SAMPLES_PER_TEACHER_FRAME,
        "teacher_frames_per_student": TEACHER_FRAMES_PER_STUDENT,
        "adapter_kernel": ADAPTER_KERNEL,
        "adapter_stride": ADAPTER_STRIDE,
        "adapter_padding": ADAPTER_PADDING,
        "pooling_rule": POOLING_RULE,
        "vad_model_id": VAD_MODEL_ID,
    }


class SoftLabelStore:
    """Deterministic, idempotent on-disk store of per-window 40 ms silence soft labels.

    Each training window's targets are a ``.npy`` under ``root/silence_40ms/`` named
    ``{audio_filename}#w{index}.npy``; each clip contributes exactly one line to
    ``root/soft_labels.jsonl`` listing all of its windows (their sample span, start
    frames, and 40 ms length). Open with :meth:`open`, passing the :class:`WindowContract`
    the run generates under: the exact :func:`generation_contract` is written to
    ``root/contract.json`` on first open and **re-checked on every resume**, so a run
    that would append labels built under a different window/hop/pooling contract
    **fails fast** rather than silently skipping existing clips and leaving stale arrays.
    :meth:`has` reports clips already written so a matching resume skips them, and
    :meth:`write_clip` persists one clip's windows **atomically** — every window array
    first, then the single index line fsynced — so an interrupted multi-hour VAD run
    replays only the in-flight clip and never double-writes an index entry. Resume
    granularity is the clip because a clip's per-window VAD passes are batched and
    written together.
    """

    ARRAYS_SUBDIR = "silence_40ms"
    INDEX_NAME = "soft_labels.jsonl"
    CONTRACT_NAME = "contract.json"

    def __init__(self, root: Path, index_file, seen: set[str]) -> None:
        self.root = root
        self.arrays_dir = root / self.ARRAYS_SUBDIR
        self.index_path = root / self.INDEX_NAME
        self._index_file = index_file
        self._seen = seen

    @classmethod
    def open(cls, root: Path, contract: WindowContract) -> "SoftLabelStore":
        root = Path(root)
        (root / cls.ARRAYS_SUBDIR).mkdir(parents=True, exist_ok=True)
        cls._reconcile_contract(root / cls.CONTRACT_NAME, generation_contract(contract))
        index_path = root / cls.INDEX_NAME
        seen = _read_index_keys(index_path)
        index_file = open(index_path, "a", encoding="utf-8")
        return cls(root, index_file, seen)

    @staticmethod
    def _reconcile_contract(contract_path: Path, contract: dict) -> None:
        """Write the generation contract on first open, or fail fast if it changed.

        A resumed run whose contract differs from the stored one would corrupt the
        artifact (mixed pooling/window parameters under one ``audio_filename`` key), so
        it is rejected here instead — regenerating under a new contract is a fresh store.
        """
        if contract_path.exists():
            stored = json.loads(contract_path.read_text(encoding="utf-8"))
            if stored != contract:
                raise ValueError(
                    f"soft-label store at {contract_path.parent} was generated under a "
                    f"different contract {stored}; refusing to resume with {contract}. "
                    "Regenerate into a fresh --out-dir."
                )
            return
        tmp = contract_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(tmp, contract_path)

    def has(self, audio_filename: str) -> bool:
        return audio_filename in self._seen

    def write_clip(
        self,
        audio_filename: str,
        windows: list[tuple[Window, np.ndarray]],
        num_samples: int,
        recitation_start_sample: int = 0,
    ) -> None:
        """Persist all of one clip's per-window 40 ms targets and index the clip once.

        A no-op if ``audio_filename`` is already stored, so replaying an interrupted
        clip adds no duplicate array or index line. Window arrays are written before the
        index line and the line is fsynced, so a crash before the line leaves the clip
        un-indexed (re-done on resume), never half-indexed. ``recitation_start_sample`` is
        the clip-relative offset of the windowed recitation span (0 for a whole-clip run),
        recorded so the phoneme labels — keyed by the same clip-relative ``start_sample`` —
        join to these soft labels explicitly.
        """
        if audio_filename in self._seen:
            return
        entries = []
        for window, labels in windows:
            array_name = f"{audio_filename}#w{window.index}.npy"
            np.save(self.arrays_dir / array_name, np.asarray(labels, dtype=np.float32))
            entries.append(
                {
                    "window_index": window.index,
                    "start_sample": window.start_sample,
                    "num_samples": window.num_samples,
                    "start_student_frame": window.start_student_frame,
                    "num_student_frames": int(len(labels)),
                    "array_path": f"{self.ARRAYS_SUBDIR}/{array_name}",
                }
            )
        self._seen.add(audio_filename)
        self._index_file.write(
            json.dumps(
                {
                    "audio_filename": audio_filename,
                    "num_samples": int(num_samples),
                    "recitation_start_sample": int(recitation_start_sample),
                    "windows": entries,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        self._index_file.flush()
        os.fsync(self._index_file.fileno())

    def close(self) -> None:
        self._index_file.close()

    def __enter__(self) -> "SoftLabelStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _read_index_keys(index_path: Path) -> set[str]:
    """Recover the ``audio_filename`` keys already written to ``index_path``."""
    if not index_path.exists():
        return set()
    seen: set[str] = set()
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["audio_filename"])
    return seen


def _iter_present_clips(records: list[ManifestRecord], clips_dir: Path):
    """Yield ``(audio_filename, waveform)`` for staged clips, deterministic order.

    Reads each clip present under ``clips_dir`` (missing ones skipped, as
    :func:`tadabur.vad.compute_clip_pauses` does) in sorted ``audio_filename`` order, so
    the run is idempotent and resumable. ``soundfile`` is imported lazily to keep the
    module's pooling/windowing importable in a torch-free CPU env.
    """
    import soundfile as sf

    present = sorted(
        (r for r in records if (clips_dir / r.audio_filename).exists()),
        key=lambda r: r.audio_filename,
    )
    for record in present:
        waveform = sf.read(clips_dir / record.audio_filename, dtype="float32")[0]
        yield record.audio_filename, np.asarray(waveform, dtype=np.float32)


def generate_soft_labels(
    records: list[ManifestRecord],
    clips_dir: Path,
    out_dir: Path,
    *,
    recitation_spans: dict[str, tuple[int, int]] | None = None,
    contract: WindowContract | None = None,
    device: str = "cuda",
    dtype_str: str = "bfloat16",
    batch_size: int = 8,
) -> int:
    """Build the per-window waqf soft-label artifact for ``records`` under ``out_dir``.

    Loads the Recitation VAD once (:class:`tadabur.vad.RecitationVad`) and, for each
    staged clip in deterministic ``audio_filename`` order, cuts the waveform into the
    training windows the student sees, runs the VAD over each **window waveform**
    (batched), and pools each window's own 20 ms posteriors to its exact 40 ms length
    (:func:`pool_window_posteriors`) — window-local labels, not a sliced whole-clip pass.

    ``recitation_spans`` maps ``audio_filename`` → clip-relative ``(start_sample,
    num_samples)`` of the un-waqf-segmented recitation (from the ``clip_status`` sidecar,
    via :func:`recitation_window_span`). A clip with a span is windowed over **exactly
    that recitation span on the shared clip-relative grid**
    (:func:`slice_recitation_windows`), so its soft targets pair with the phoneme CTC
    labels window-for-window (ADR-0004 "same window contract"); a clip absent from the map
    (or ``recitation_spans is None``) falls back to whole-clip windowing. Each clip is
    written + fsynced to the :class:`SoftLabelStore` before the next runs, so the ``has``
    resume contract protects a multi-hour run: a crash keeps every clip already written,
    the full manifest is never materialized, and the store's contract metadata rejects a
    resume under different pooling/window params. Returns the number of clips newly written.
    """
    import torch

    from tadabur.vad import RecitationVad

    contract = contract or WindowContract()
    spans = recitation_spans or {}
    with SoftLabelStore.open(out_dir, contract) as store:
        pending = [r for r in records if not store.has(r.audio_filename)]
        written = 0
        with RecitationVad.load(
            device=torch.device(device), dtype=getattr(torch, dtype_str)
        ) as vad:
            for audio_filename, waveform in _iter_present_clips(pending, Path(clips_dir)):
                if audio_filename in spans:
                    start_sample, num_samples = spans[audio_filename]
                    windows = slice_recitation_windows(
                        waveform, start_sample, num_samples, contract
                    )
                else:
                    start_sample = 0
                    windows = slice_windows(waveform, contract)
                posteriors = vad.silence_posteriors(
                    [slice_wave for _, slice_wave in windows], batch_size=batch_size
                )
                labelled = [
                    (window, pool_window_posteriors(posterior))
                    for (window, _), posterior in zip(windows, posteriors)
                ]
                store.write_clip(
                    audio_filename,
                    labelled,
                    num_samples=len(waveform),
                    recitation_start_sample=start_sample,
                )
                written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="passing-subset JSONL manifest")
    parser.add_argument("--clips-dir", type=Path, required=True, help="staged 16 kHz clips directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="soft-label artifact directory")
    parser.add_argument(
        "--clip-status",
        type=Path,
        default=None,
        help="per-clip status sidecar (JSONL) from tadabur.segment_score; when given, each "
        "clip is windowed over its recitation span so the soft labels share the phoneme "
        "CTC labels' clip-relative window grid (ADR-0004). Omit for whole-clip windowing.",
    )
    parser.add_argument(
        "--window-feature-frames",
        type=int,
        default=DEPLOYED_WINDOW_FEATURE_FRAMES,
        help="window length on the 20 ms grid (deployed 5 s = 250)",
    )
    parser.add_argument(
        "--hop-feature-frames",
        type=int,
        default=None,
        help="window step on the 20 ms grid; default = frozen center-trusted 1 s overlap "
        "(4 s hop = 200 frames, #24 A2 freeze)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    contract = WindowContract(
        feature_frames=args.window_feature_frames,
        hop_feature_frames=args.hop_feature_frames or FROZEN_HOP_FEATURE_FRAMES,
    )
    records = read_records(args.manifest)
    recitation_spans = _recitation_spans_from_clip_status(args.clip_status)
    written = generate_soft_labels(
        records,
        args.clips_dir,
        args.out_dir,
        recitation_spans=recitation_spans,
        contract=contract,
        device=args.device,
        dtype_str=args.dtype,
        batch_size=args.batch_size,
    )
    print(f"Wrote {written} new soft-label clips to {args.out_dir} ({len(records)} in manifest).")


def _recitation_spans_from_clip_status(
    clip_status_path: Path | None,
) -> dict[str, tuple[int, int]] | None:
    """Clip-relative recitation ``(start_sample, num_samples)`` per clip, or ``None``.

    Reads the ``clip_status`` sidecar and maps every clip the segmenter could split
    safely (``skip_reason is None``) to its recitation window span
    (:func:`recitation_window_span`). Skipped clips are omitted: they are excluded from
    the phoneme labels too, so their soft labels are unpaired and windowing them over a
    (meaningless) recitation span would only waste work — they fall back to whole-clip.
    Returns ``None`` when no sidecar is given (whole-clip windowing for every clip).
    """
    if clip_status_path is None:
        return None
    from tadabur.clip_status import read_clip_status

    spans: dict[str, tuple[int, int]] = {}
    for status in read_clip_status(clip_status_path):
        if status.skip_reason is not None:
            continue
        spans[status.audio_filename] = recitation_window_span(
            status.recitation_start_s, status.recitation_end_s
        )
    return spans


if __name__ == "__main__":
    main()
