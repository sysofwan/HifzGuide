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
the protocol is tested on synthetic logits with no model at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from tadabur.phoneme_vocab import PHONEME_ID_TO_CHAR, PHONEME_PAD_ID
from training.distill_data import SAMPLE_RATE, WINDOW_SAMPLES
from training.distill_loss import BLANK_ID, CONFIRM_TIMESTEPS
from training.distill_student import (
    DEPLOYED_LOGIT_FRAMES,
    NUM_PHONEME_CLASSES,
    PHONEME_LEVEL,
    PRESETS,
    TEACHER_MODEL_ID,
    build_student,
)

# The device advances its buffer by 1 s per pass; at 125 timesteps per 5 s window that is
# 25 timesteps, which is also ``CONFIRM_TIMESTEPS`` -- one block.
HOP_SAMPLES = SAMPLE_RATE

# Five 25-step blocks per 125-step window, one per second of the 5 s window.
NUM_BLOCKS = DEPLOYED_LOGIT_FRAMES // CONFIRM_TIMESTEPS

# The block Muraja commits today: the oldest second of the buffer.
DEPLOYED_BLOCK = 0

# Bumped whenever the replayed **deployed** (b=0) protocol changes what a clip decodes to.
# Cached teacher decodes carry it (``training.decode_evalset``) so a manifest built under
# one protocol cannot be silently scored under another. v1 was the unflushed stream; v2
# flushes the last window. A stream committed at another block is a different protocol
# and is never cached under this version.
PROTOCOL_VERSION = "confirmed-stream-v2-flush"

# Students are distilled on the teacher's own front-end, and a checkpoint carries no
# feature extractor of its own, so a checkpoint decodes through the teacher's.
STUDENT_FEATURE_EXTRACTOR = TEACHER_MODEL_ID

# The two weight precisions this repo has decoded with; see the module docstring.
WEIGHTS_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


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
        if t != PHONEME_PAD_ID and 0 <= t < len(PHONEME_ID_TO_CHAR)
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


def load_hf_model(model_ref: str):
    """A Muaalem-architecture model saved for ``from_pretrained``, on CPU in eval mode.

    Loaded through the vendored modeling code (not ``AutoModel``): the multi-level CTC class
    is not registered with transformers, and pinning the vendored copy keeps every tool on
    identical weights. Fails loudly if the phoneme head is not the 43-class vocabulary,
    which would silently corrupt every decode.
    """
    from tadabur.muaalem import (
        Wav2Vec2BertForMultilevelCTC,
        Wav2Vec2BertForMultilevelCTCConfig,
    )

    config = Wav2Vec2BertForMultilevelCTCConfig.from_pretrained(model_ref)
    classes = config.level_to_vocab_size[PHONEME_LEVEL]
    if classes != NUM_PHONEME_CLASSES:
        raise ValueError(
            f"{model_ref} phoneme head has {classes} classes, expected "
            f"{NUM_PHONEME_CLASSES} (tadabur.phoneme_vocab). Vocabulary drift -- the "
            "decode mapping would be corrupt."
        )
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(model_ref, config=config)
    return model.eval()


def load_student_from_checkpoint(
    checkpoint_path: Path, device: torch.device, use_ema: bool = False
):
    """Rebuild the student described by a checkpoint and load its weights.

    Returns the run's persisted config alongside the model: the eval tools need it to
    refuse a split the checkpoint was not trained under.

    ``use_ema`` selects the averaged weights a run with ``--ema-decay`` stored beside the
    live ones. It raises rather than falling back when they are absent: silently scoring the
    live weights under an ``--ema`` flag would report the wrong model's number, and the two
    are meant to be compared.
    """
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
    already loaded (the distillation tools keep their teacher resident for other work).
    ``batch_size`` is fixed per decoder because, under bf16, it moves ~0.2% of characters:
    two decodes are comparable only at the same batch size.
    """

    def __init__(self, model, extractor, device: torch.device, batch_size: int = 16) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.model = model
        self.extractor = extractor
        self.device = torch.device(device)
        self.batch_size = batch_size

    @classmethod
    def load(
        cls,
        model_ref: str | Path,
        device: str | torch.device,
        *,
        weights_dtype: torch.dtype,
        batch_size: int = 16,
        use_ema: bool = False,
    ) -> "Decoder":
        """Load a hub id, a saved model directory, or a distillation checkpoint file.

        ``use_ema`` selects a checkpoint's averaged weights; asking for them from anything
        else is an error rather than a silent no-op.
        """
        from transformers import SeamlessM4TFeatureExtractor

        device = torch.device(device)
        if weights_dtype not in WEIGHTS_DTYPES.values():
            raise ValueError(f"weights_dtype must be one of {sorted(WEIGHTS_DTYPES)}")
        if device.type != "cuda" and weights_dtype != torch.float32:
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
            model = load_hf_model(str(model_ref))
            extractor_source = str(model_ref)
        model = model.to(device=device, dtype=weights_dtype)
        extractor = SeamlessM4TFeatureExtractor.from_pretrained(extractor_source)
        return cls(model, extractor, device, batch_size)

    @torch.no_grad()
    def _class_ids(self, features, attention_mask=None) -> np.ndarray:
        """Per-timestep argmax over the phoneme head, ``(batch, timesteps)``."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = self.model(
                features.to(self.device),
                attention_mask=attention_mask,
                return_dict=True,
            )["logits"][PHONEME_LEVEL]
        return logits.float().argmax(dim=-1).cpu().numpy()

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

    def decode_spans(self, spans: Iterable[np.ndarray]) -> list[str]:
        """Each 16 kHz span decoded whole, in one variable-length pass.

        Spans are batched with padding under a real attention mask, and each row is cut to
        its own valid length (the model's own length mapping, the one the CTC loss uses)
        before the collapse, so padding never reaches a decode. ``spans`` is consumed
        lazily, one batch at a time, so a caller can stream them from disk.
        """
        decodes: list[str] = []
        pending = iter(spans)
        while batch := list(islice(pending, self.batch_size)):
            extracted = self.extractor(
                [np.asarray(span, dtype=np.float32) for span in batch],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            mask = extracted.attention_mask.to(self.device)
            class_ids = self._class_ids(extracted.input_features, mask)
            valid = self.model._get_feat_extract_output_lengths(mask.sum(dim=1)).tolist()
            decodes.extend(
                tokens_to_phonemes(seg.token_id for seg in scan_ctc(row[: int(length)]))
                for row, length in zip(class_ids, valid)
            )
        return decodes
