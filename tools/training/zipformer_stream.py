"""The card's streaming protocol as arithmetic: ``tokens.txt``, CTC runs, and the ONNX loop.

This is the part of :mod:`training.zipformer_decode` that decides what the model emitted and
where -- the symbol table, the 40 ms timeline, the run collapse and the call loop -- with no
librosa, torch or ONNX import at module level. It lives apart so the parity gate's trace phase
(:func:`training.zipformer_parity.trace`) can drive the same ONNX with the reference front
end's features inside the reference runtime's throwaway venv, which holds only numpy, soundfile,
sherpa-onnx, kaldi-native-fbank and onnxruntime. Importing the card's decoder there would pull
in the corpus readers, and the trace must be the reference's frame outputs, never a copy of the
loop that produced ours.

* **No call is padded.** A call runs only while a whole :data:`INPUT_FRAMES` window is
  available -- sherpa-onnx's readiness rule, ``processed + input_frames < num_frames``. The
  zero-padded final call emitted tokens the reference runtime never produces; the parity gate
  caught that on the phone source and the one diagnosis round removed it (parity r1,
  2026-11-01).
* **One frame timeline per stream.** Each call yields its own log-prob frames (12 per 0.48 s
  chunk at the card's chunk 24), concatenated into one 40 ms timeline that
  :class:`Emission` indexes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: ``tokens.txt``'s blank. Ours is 0, so no zipformer id is ever one of ours.
BLANK_ID = 250

#: The card's analysis frame hop: 10 ms, so one encoder output step is four of them.
FRAME_SHIFT_SAMPLES = 160
#: One encoder output step, 40 ms at 16 kHz; the timeline :class:`Emission` indexes.
STEP_SAMPLES = 4 * FRAME_SHIFT_SAMPLES
#: The card's chunk 24 export: 48 frames advance, 12 log-prob frames per call, 61 input frames.
CHUNK_FRAMES = 48
INPUT_FRAMES = 61
#: The card's "1.05 s of silence at the end of an utterance", 16,800 samples at 16 kHz.
TAIL_SAMPLES = 16_800


@dataclass(frozen=True)
class Emission:
    """One non-blank CTC run: the unit's ``tokens.txt`` string and its frame span (inclusive)."""

    text: str
    start_step: int
    end_step: int

    @property
    def midpoint(self) -> float:
        """The mean of the run's first and last frame, as ``Segment.midpoint`` is ours."""
        return (self.start_step + self.end_step) / 2.0


def read_tokens(path: Path) -> dict[int, str]:
    """``tokens.txt`` as id -> unit string, refusing a table whose blank is not :data:`BLANK_ID`.

    Every line is ``<unit> <id>``; the unit may itself be a merged multi-character string.
    """
    text = Path(path).read_text(encoding="utf-8")
    tokens = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        unit, _, id_text = line.rpartition(" ")
        try:
            token_id = int(id_text)
        except ValueError as error:
            raise ValueError(f"{path}:{number} is not '<unit> <id>': {line!r}") from error
        tokens[token_id] = unit
    if tokens.get(BLANK_ID) != "<blank>":
        raise ValueError(
            f"{path} does not put <blank> at id {BLANK_ID} (found {tokens.get(BLANK_ID)!r}); "
            "the ids are not this model's CTC output classes"
        )
    return tokens


def scan_units(frames: Iterable[int], tokens: Mapping[int, str]) -> list[Emission]:
    """Collapse per-frame argmax into non-blank runs, each carrying its ``tokens.txt`` string.

    A run ends when the id changes; the blank breaks runs of one id and is never emitted. An id
    the table does not hold is refused rather than dropped: dropping it would silently shorten
    the transcript where the model said something the table cannot name.
    """
    units: list[Emission] = []
    previous: int | None = None
    for step, frame in enumerate(frames):
        token = int(frame)
        if token == BLANK_ID:
            previous = None
            continue
        if token not in tokens:
            raise ValueError(f"tokens.txt has no unit for id {token}")
        if token == previous:
            last = units[-1]
            units[-1] = Emission(last.text, last.start_step, step)
        else:
            units.append(Emission(tokens[token], step, step))
        previous = token
    return units


def detokenise(frames: Iterable[int], tokens: Mapping[int, str]) -> str:
    """The transcript of one frame sequence, by ``tokens.txt`` strings."""
    return "".join(emission.text for emission in scan_units(frames, tokens))


def featurize(samples: np.ndarray, fbank, *, scale: float) -> np.ndarray:
    """The card's front end: scale the samples, append the 1.05 s silence, one kaldi fbank pass.

    ``fbank`` receives a 1-D float32 waveform and the card's ``FBANK_OPTIONS`` as keywords and
    returns one 80-bin frame per 10 ms. ``scale`` is required: the card does not pin whether
    kaldi expects [-1, 1] or int16-scaled samples, and the gate decides it from PER.
    """
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1) * scale
    waveform = np.concatenate([waveform, np.zeros(TAIL_SAMPLES, dtype=np.float32)])
    features = fbank(waveform, **FBANK_OPTIONS)
    return np.asarray(features, dtype=np.float32)


#: The card's kaldi fbank: 80 bins, povey window, no dither, snip_edges false, high_freq -400.
FBANK_OPTIONS = {
    "num_mel_bins": 80,
    "sample_frequency": 16_000,
    "frame_shift": 10.0,
    "window_type": "povey",
    "dither": 0.0,
    "snip_edges": False,
    "high_freq": -400.0,
}

_NUMPY_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def _state_spec(spec) -> tuple[str, tuple[int, ...], type]:
    """One cache input's name, concrete shape and numpy dtype, from the ONNX session.

    The export leaves its batch axis symbolic (``"N"``); this decoder runs one stream at a
    time, so that axis is 1. Any other symbolic axis is refused: its size would not be known
    before the first call.
    """
    try:
        dtype = _NUMPY_DTYPES[spec.type]
    except KeyError as error:
        raise ValueError(f"the ONNX input {spec.name!r} has the type {spec.type!r}") from error
    shape = tuple(1 if dim == "N" else dim for dim in spec.shape)
    if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise ValueError(
            f"the ONNX input {spec.name!r} has the non-constant shape {spec.shape}; the "
            "streaming export's cache shapes must be known before the first call"
        )
    return spec.name, shape, dtype


def state_specs(session) -> list[tuple[str, tuple[int, ...], type]]:
    """The session's cache inputs as :func:`_state_spec` reads them, in the export's order."""
    return [_state_spec(spec) for spec in session.get_inputs() if spec.name != "x"]


def stream_frame_ids(session, features: np.ndarray,
                     specs: Sequence[tuple[str, tuple[int, ...], type]]) -> list[np.ndarray]:
    """Every call's argmax frame ids over ``features``, in call order, from a zero cache.

    A call is made only while a whole :data:`INPUT_FRAMES` window is available -- sherpa-onnx's
    readiness rule -- so no call is ever padded with zero feature rows. The card's 1.05 s tail
    still gives the last complete chunk the silence a delayed word needs to reach it.
    """
    state = {name: np.zeros(shape, dtype=dtype) for name, shape, dtype in specs}
    calls: list[np.ndarray] = []
    cursor = 0
    while cursor + INPUT_FRAMES < len(features):
        window = features[cursor:cursor + INPUT_FRAMES][None]
        outputs = session.run(None, {"x": window, **state})
        if len(outputs) != len(state) + 1:
            raise ValueError(
                f"the ONNX session returned {len(outputs)} outputs for {len(state)} cache "
                "inputs and log_probs"
            )
        calls.append(np.argmax(np.asarray(outputs[0])[0], axis=-1))
        state = {name: np.asarray(value) for (name, _, _), value in zip(specs, outputs[1:])}
        cursor += CHUNK_FRAMES
    return calls
