"""Decode audio with the Quran-Lab zipformer (v3.2, chunk 24), for the #245 screen.

The zipformer is **not** one of our models: its front end, its token ids and its streaming
protocol are the card's, so this module owns all three and shares none of them with
:mod:`training.decoding`.

* **Detokenisation is by ``tokens.txt`` strings.** The committed ``tokens.txt`` covers our 42
  non-blank classes as a *set* only: its order differs, its blank is id 250 (ours is 0), and ids
  42-249 are merged units such as ``اا``. :func:`training.decoding.tokens_to_phonemes` and
  :func:`~training.decoding.scan_ctc` therefore never see a zipformer id; :func:`scan_units`
  collapses runs against ``tokens.txt`` and yields each unit's string and frame span.
* **One continuous stream per item.** The kaldi fbank is computed once over the item plus
  1.05 s of silence (the card's flush), the encoder's cache state is carried from call to call,
  and every call's 12 log-prob frames form one 40 ms timeline. A window never restarts the
  model from zero: :func:`stream_checks` slices that one timeline with the transcriber's own
  schedule (:func:`training.decoding.check_schedule`).
* **Neither of our window guards applies.** The card's model is not ours to correct, so
  ``likely_hallucination`` and ``strip_padding_artifacts`` are not applied (#245 Q6).
* **A zipformer record carries ``spans`` and ``checks``, never ``stream_b0``** (#245 Q3): the
  students are scored from their b = 0 stream, the zipformer from its whole stream. Its
  fingerprints are CPU/fp32 under :data:`ZIPFORMER_POLICY`, a mode and policy no decode of ours
  holds, so :meth:`DecodeFingerprint.check_comparable` refuses to compare the two.

The fbank callable is injected: production passes ``torchaudio.compliance.kaldi.fbank`` (the
card's kaldi front end), while the tests pass a stand-in, so importing this module never
imports torch or our feature extractor. The sample scale is a required argument rather than a
default: the card does not say whether kaldi expects [-1, 1] or int16-scaled samples, and the
gate's PER pins it (``training.zipformer_gate.pin_scale``).

Usage (from ``tools/``)::

    import onnxruntime as ort
    from torchaudio.compliance import kaldi
    from training.zipformer_decode import ZipformerDecoder, read_tokens

    session = ort.InferenceSession("zipformer_p_arabic_v3.2.onnx")
    decoder = ZipformerDecoder(session, read_tokens(ZIPFORMER_DIR / "tokens.txt"),
                               lambda wave, **options: kaldi.fbank(
                                   torch.from_numpy(wave)[None], **options),
                               scale=32_768.0)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.distill_data import SAMPLE_RATE
from training.truth_baseline import Store

#: The card's committed symbol table, unit inventory and licence, beside the decodes they back.
ZIPFORMER_DIR = Path(__file__).resolve().parent.parent / "tadabur" / "zipformer_screen"

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

#: The card's kaldi fbank: 80 bins, povey window, no dither, snip_edges false, high_freq -400.
FBANK_OPTIONS = {
    "num_mel_bins": 80,
    "sample_frequency": SAMPLE_RATE,
    "frame_shift": 10.0,
    "window_type": "povey",
    "dither": 0.0,
    "snip_edges": False,
    "high_freq": -400.0,
}

#: The decode mode of each arm, and its fingerprint policy. Both differ from every mode and
#: policy our own decodes carry, so the two kinds can never be compared by accident.
ZIPFORMER_PROTOCOLS = {"spans": "zipformer-spans-v1", "checks": "zipformer-checks-v1"}
ZIPFORMER_POLICY = "kaldi-fbank-v1"

#: The model each committed decode file is scored from (#245 Q3).
SCORING_ARMS = {
    "base": "stream_b0",
    "c95": "stream_b0",
    "r2_candidate_ema": "stream_b0",
    "zipformer": "spans",
}


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

    ``fbank`` receives the 1-D float32 waveform and :data:`FBANK_OPTIONS` as keywords and
    returns one 80-bin frame per 10 ms. ``scale`` is required: the card does not pin whether
    kaldi expects [-1, 1] or int16-scaled samples, and the gate decides it from PER.
    """
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1) * scale
    waveform = np.concatenate([waveform, np.zeros(TAIL_SAMPLES, dtype=np.float32)])
    features = fbank(waveform, **FBANK_OPTIONS)
    return np.asarray(features, dtype=np.float32)


_NUMPY_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def _state_spec(spec) -> tuple[str, tuple[int, ...], type]:
    """One cache input's name, constant shape and numpy dtype, from the ONNX session."""
    try:
        dtype = _NUMPY_DTYPES[spec.type]
    except KeyError as error:
        raise ValueError(f"the ONNX input {spec.name!r} has the type {spec.type!r}") from error
    shape = tuple(spec.shape)
    if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise ValueError(
            f"the ONNX input {spec.name!r} has the non-constant shape {spec.shape}; the "
            "streaming export's cache shapes must be known before the first call"
        )
    return spec.name, shape, dtype


class ZipformerDecoder:
    """The card's streaming CTC decoder behind ours' arms: one continuous run per item.

    ``session`` is an ``onnxruntime.InferenceSession`` of the card's cache-aware streaming
    export (``x`` plus its cache inputs; ``log_probs`` first, then each cache's next value in
    input order). ``tokens`` is :func:`read_tokens`' table. ``scale`` is :func:`featurize`'s.
    """

    def __init__(self, session, tokens: Mapping[int, str], fbank, *, scale: float) -> None:
        self.session = session
        self.tokens = tokens
        self.fbank = fbank
        self.scale = scale
        inputs = session.get_inputs()
        if not any(spec.name == "x" for spec in inputs):
            raise ValueError("the ONNX session has no 'x' feature input")
        self._state_specs = [_state_spec(spec) for spec in inputs if spec.name != "x"]

    def stream(self, samples: np.ndarray) -> list[Emission]:
        """The item's units over one continuous stream: one fbank, one cache, one timeline."""
        features = featurize(samples, self.fbank, scale=self.scale)
        if not len(features):
            raise ValueError("the front end produced no frames")
        state = {name: np.zeros(shape, dtype=dtype)
                 for name, shape, dtype in self._state_specs}
        frames: list[np.ndarray] = []
        cursor = 0
        while cursor < len(features):
            chunk = np.zeros((INPUT_FRAMES, features.shape[1]), dtype=np.float32)
            piece = features[cursor:cursor + INPUT_FRAMES]
            chunk[:len(piece)] = piece
            outputs = self.session.run(None, {"x": chunk[None], **state})
            if len(outputs) != len(self._state_specs) + 1:
                raise ValueError(
                    f"the ONNX session returned {len(outputs)} outputs for "
                    f"{len(self._state_specs)} cache inputs and log_probs"
                )
            frames.append(np.argmax(np.asarray(outputs[0])[0], axis=-1))
            state = {name: np.asarray(value)
                     for (name, _, _), value in zip(self._state_specs, outputs[1:])}
            cursor += CHUNK_FRAMES
        return scan_units(np.concatenate(frames), self.tokens)

    def spans(self, samples: np.ndarray) -> str:
        """The item decoded whole, as one transcript (the ``spans`` arm)."""
        return "".join(emission.text for emission in self.stream(samples))

    def checks(self, samples: np.ndarray) -> list[dict]:
        """The item's check sequence, cut from its one stream (:func:`stream_checks`)."""
        return stream_checks(self.stream(samples), len(samples))


def _text(emissions: Iterable[Emission], low: float, high: float | None = None) -> str:
    """The emitted units whose midpoint lies in ``[low, high)``; ``None`` has no upper bound."""
    if high is None:
        return "".join(e.text for e in emissions if e.midpoint >= low)
    return "".join(e.text for e in emissions if low <= e.midpoint < high)


def stream_checks(emissions: list[Emission], num_samples: int) -> list[dict]:
    """The app's checks, sliced from one continuous zipformer stream.

    The schedule is the transcriber's own (:func:`training.decoding.check_schedule`), and each
    check takes the emissions whose midpoint lies in its window, as
    :func:`training.decoding.app_check` does in that window's own frame space. A window's hop
    is its first second and its overlap the rest; a preview sends everything since the window's
    start; the stop sends everything from its start, the 1.05 s flush included. Neither of our
    window guards applies (#245 Q6).

    Offline slicing changes only *when* an emission would have arrived, not what it says: an
    emission's text depends on audio up to the end of its chunk, so a check can lag the live
    stream by at most one 0.48 s call (#245 Q5).

    ``training.decoding`` imports torch transitively, so it is imported here rather than at
    module scope: this module must import without torch (#245 Q5).
    """
    from training import decoding

    checks: list[dict] = []
    for span in decoding.check_schedule(num_samples):
        low = span.start / STEP_SAMPLES
        if span.kind == decoding.WINDOW:
            hop_high = (span.start + decoding.HOP_SAMPLES) / STEP_SAMPLES
            check = decoding.make_check(
                decoding.WINDOW,
                _text(emissions, low, hop_high),
                _text(emissions, hop_high, span.end / STEP_SAMPLES),
            )
        elif span.kind == decoding.PREVIEW:
            check = decoding.make_check(
                decoding.PREVIEW, "", _text(emissions, low, span.end / STEP_SAMPLES))
        elif span.kind == decoding.STOP:
            check = decoding.make_check(decoding.STOP, _text(emissions, low), "")
        else:
            raise ValueError(f"unknown check kind {span.kind!r}")
        if check is not None:
            checks.append(check)
    return checks


def _zipformer_arms(decoder: ZipformerDecoder, span: np.ndarray) -> dict:
    """One item's two arms, as a zipformer record holds them (never ``stream_b0``)."""
    return {"spans": decoder.spans(span), "checks": decoder.checks(span)}


#: The zipformer's stores: whole-stream transcripts and check sequences, one record per item.
ZIPFORMER_STORES = {
    "zipformer": Store(ZIPFORMER_DIR / "decodes", ZIPFORMER_PROTOCOLS, _zipformer_arms)
}


def scoring_arm(model: str) -> str:
    """The arm a model's whole-item text is scored from (#245 Q3)."""
    try:
        return SCORING_ARMS[model]
    except KeyError:
        raise ValueError(
            f"{model!r} has no scoring arm; known models: {sorted(SCORING_ARMS)}"
        ) from None


def whole_item_text(line: Mapping, model: str) -> str:
    """The scored whole-item text of ``line``, refusing a zipformer line with a ``stream_b0``."""
    arm = scoring_arm(model)
    if arm == "spans" and "stream_b0" in line:
        raise ValueError(
            f"{model} is scored from its whole stream; a line holding a stream_b0 decode is "
            "one of ours, not this model's"
        )
    return line[arm]


def zipformer_fingerprint(model: str, arm: str):
    """``arm``'s decode fingerprint: CPU, fp32, batch 1, and a mode no decode of ours holds."""
    from training.decoding import DecodeFingerprint

    try:
        mode = ZIPFORMER_PROTOCOLS[arm]
    except KeyError:
        raise ValueError(
            f"the zipformer has no {arm!r} arm; its arms are {sorted(ZIPFORMER_PROTOCOLS)}"
        ) from None
    return DecodeFingerprint(
        model=model,
        mode=mode,
        weights_dtype="fp32",
        batch_size=1,
        device_type="cpu",
        autocast=False,
        policy=ZIPFORMER_POLICY,
    )
