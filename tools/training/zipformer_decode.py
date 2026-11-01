"""Decode audio with the Quran-Lab zipformer (v3.2, chunk 24), for the #245 screen.

The zipformer is **not** one of our models: its front end, its token ids and its streaming
protocol are the card's, so this module owns all three and shares none of them with
:mod:`training.decoding`.

* **Detokenisation is by ``tokens.txt`` strings.** The committed ``tokens.txt`` covers our 42
  non-blank classes as a *set* only: its order differs, its blank is id 250 (ours is 0), and ids
  42-249 are merged units such as ``اا``. :func:`training.decoding.tokens_to_phonemes` and
  :func:`~training.decoding.scan_ctc` therefore never see a zipformer id; :func:`scan_units`
  collapses runs against ``tokens.txt`` and yields each unit's string and frame span.
* **One continuous stream per item** (``spans`` and ``checks``). The kaldi fbank is computed
  once over the item plus 1.05 s of silence (the card's flush), the encoder's cache state is
  carried from call to call, and every call's 12 log-prob frames form one 40 ms timeline.
  :func:`stream_checks` slices that one timeline with the transcriber's own schedule
  (:func:`training.decoding.check_schedule`); a window never restarts the model from zero.
* **One fresh inference per check** (``checks-w1``, architect r2 + principal r1). Muraja runs
  an inference per check and gets several looks at a word; the single stream gives zipformer
  one. ``checks-w1`` therefore decodes every check of the schedule again, over
  ``[max(0, start - W), end)`` with ``W = 1 s`` (:data:`WARMUP_SAMPLES`, the transcriber's hop)
  from a zero cache, and drops only the warm-up emissions (midpoint before the check's start):
  everything else is kept, the 1.05 s flush included, so a delayed last word is not truncated.
  ``checks-w1-windows`` runs the fresh inferences for the windows only and takes previews and
  the stop from the one continuous stream (report-only). Both are zipformer-only arms: ours
  never decodes them.
* **The keep rules differ on purpose.** A single-stream check takes emissions with midpoint in
  its window (a straddling run reaches the later check, which is the same stream); a
  fresh-inference check takes everything from its start, because the delayed units after its end
  are what its own flush forces out.
* **Neither of our window guards applies.** The card's model is not ours to correct, so
  ``likely_hallucination`` and ``strip_padding_artifacts`` are not applied (#245 Q6).
* **The arms are per metric** (:func:`metric_arm`). Zipformer's app-outcome metrics read
  ``checks-w1``, its whole-item metrics read ``spans``; the students' read ``checks`` and
  ``stream_b0``. No zipformer arm is named ``stream_b0``, and the two report-only arms are
  never a zipformer verdict arm (:data:`REPORT_ONLY_ARMS`).
* **The fbank callable is injected.** Production passes
  ``torchaudio.compliance.kaldi.fbank`` (the card's kaldi front end), the tests a stand-in.
  Importing this module (and ``training.decoding``) does not import torch, so the geometry and
  the check slicing run on the Mac venv. The sample scale is required rather than defaulted: the
  card does not say whether kaldi expects [-1, 1] or int16-scaled samples, and the gate's PER
  pins it (``training.zipformer_gate.pin_scale``).

The two new arms live in :data:`ZIPFORMER_W1_STORES`, apart from :data:`ZIPFORMER_STORES`, since
they are decoded only on the sets the Muraja harness grades.

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

from training import decoding
from training.decoding import DecodeFingerprint
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

#: The warm-up a fresh per-check inference keeps before the check's start: one hop, 1 s.
WARMUP_SAMPLES = decoding.HOP_SAMPLES
#: The hop's width in output steps, the first second a window commits.
HOP_STEPS = WARMUP_SAMPLES // STEP_SAMPLES

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

#: The arm names: zipformer's single stream and its two per-check (warm-started) variants.
CHECKS_W1 = "checks-w1"
CHECKS_W1_WINDOWS = "checks-w1-windows"

#: The decode mode of each arm, and its fingerprint policy. All four differ from every mode and
#: policy our own decodes carry, so the two kinds can never be compared by accident.
ZIPFORMER_PROTOCOLS = {
    "spans": "zipformer-spans-v1",
    "checks": "zipformer-checks-v1",
    CHECKS_W1: "zipformer-checks-v1-w1000ms-flush1050ms",
    CHECKS_W1_WINDOWS: "zipformer-checks-v1-w1000ms-flush1050ms-windows",
}
ZIPFORMER_POLICY = "kaldi-fbank-v1"

#: The metrics the screen reports, split by which kind of arm reads them.
APP_METRICS = ("qdat_false_flags", "ship_a_catches")
WHOLE_ITEM_METRICS = (
    "qdat_per_class", "qdat_bench", "edits_heard", "ship_a_whole_item", "benchmark_per_class",
)
METRICS = APP_METRICS + WHOLE_ITEM_METRICS
#: Arms that are reported only; never zipformer's verdict arm (principal r1 §3).
REPORT_ONLY_ARMS = ("checks", CHECKS_W1_WINDOWS)

#: The arm each model's whole-item text is scored from (#245 Q3).
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
    """The card's streaming CTC decoder behind our arms: one stream, and one per check.

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
        # One item's emissions, held so ``spans`` and ``checks`` on the same item share one
        # stream. The array is held too, so its identity cannot be recycled under a stale hit.
        self._last_item: tuple[np.ndarray, list[Emission]] | None = None

    def _emissions(self, samples: np.ndarray) -> list[Emission]:
        """The item's emissions, from one stream shared by every arm that slices it."""
        if self._last_item is not None and self._last_item[0] is samples:
            return self._last_item[1]
        emissions = self._stream(samples, 0)
        self._last_item = (samples, emissions)
        return emissions

    def stream(self, samples: np.ndarray) -> list[Emission]:
        """The item's units over one continuous stream: one fbank, one cache, one timeline."""
        return self._emissions(samples)

    def _stream(self, samples: np.ndarray, offset_steps: int) -> list[Emission]:
        """One stream over ``samples``, its steps reported ``offset_steps`` later.

        A fresh zero cache, never carried in from another call or item.
        """
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
        emissions = scan_units(np.concatenate(frames), self.tokens)
        if not offset_steps:
            return emissions
        return [Emission(emission.text, emission.start_step + offset_steps,
                         emission.end_step + offset_steps)
                for emission in emissions]

    def spans(self, samples: np.ndarray) -> str:
        """The item decoded whole, as one transcript (the ``spans`` arm)."""
        return "".join(emission.text for emission in self._emissions(samples))

    def checks(self, samples: np.ndarray) -> list[dict]:
        """The checks cut from the item's one stream (the ``checks`` arm; :func:`stream_checks`)."""
        return stream_checks(self._emissions(samples), len(samples))

    def _fresh_check(self, samples: np.ndarray, span: decoding.CheckSpan) -> dict | None:
        """One check decoded again from a zero cache, with its one hop of warm-up."""
        begin = warmup_start(span)
        emissions = self._stream(samples[begin:span.end], begin // STEP_SAMPLES)
        return w1_check(span, emissions)

    def checks_w1(self, samples: np.ndarray) -> list[dict]:
        """The checks, each from its own fresh warm-started inference (the ``checks-w1`` arm)."""
        checks = (self._fresh_check(samples, span)
                  for span in decoding.check_schedule(len(samples)))
        return [check for check in checks if check is not None]

    def checks_w1_windows(self, samples: np.ndarray) -> list[dict]:
        """B': windows fresh, previews and the stop from one stream (``checks-w1-windows``)."""
        emissions = self._emissions(samples)
        checks: list[dict] = []
        for span in decoding.check_schedule(len(samples)):
            check = (self._fresh_check(samples, span) if span.kind == decoding.WINDOW
                     else _stream_check(span, emissions))
            if check is not None:
                checks.append(check)
        return checks


def _text(emissions: Iterable[Emission], low: float, high: float | None = None) -> str:
    """The emitted units whose midpoint lies in ``[low, high)``; ``None`` has no upper bound."""
    if high is None:
        return "".join(e.text for e in emissions if e.midpoint >= low)
    return "".join(e.text for e in emissions if low <= e.midpoint < high)


def warmup_start(span: decoding.CheckSpan) -> int:
    """The sample a fresh check starts at: one hop before its start, clipped at the item's 0."""
    return max(0, span.start - WARMUP_SAMPLES)


def _stream_check(span: decoding.CheckSpan, emissions: list[Emission]) -> dict | None:
    """One check of the single continuous stream, in that stream's absolute frame space."""
    low = span.start / STEP_SAMPLES
    if span.kind == decoding.WINDOW:
        hop_high = (span.start + decoding.HOP_SAMPLES) / STEP_SAMPLES
        return decoding.make_check(
            decoding.WINDOW,
            _text(emissions, low, hop_high),
            _text(emissions, hop_high, span.end / STEP_SAMPLES),
        )
    if span.kind == decoding.PREVIEW:
        return decoding.make_check(
            decoding.PREVIEW, "", _text(emissions, low, span.end / STEP_SAMPLES))
    if span.kind == decoding.STOP:
        return decoding.make_check(decoding.STOP, _text(emissions, low), "")
    raise ValueError(f"unknown check kind {span.kind!r}")


def stream_checks(emissions: list[Emission], num_samples: int) -> list[dict]:
    """The app's checks sliced from one continuous zipformer stream (the ``checks`` arm).

    The schedule is the transcriber's own (:func:`training.decoding.check_schedule`), and each
    check takes the emissions whose midpoint lies in its window, as
    :func:`training.decoding.app_check` does in that window's own frame space. A window's hop
    is its first second and its overlap the rest; a preview sends everything since the window's
    start; the stop sends everything from its start, the 1.05 s flush included. Neither of our
    window guards applies (#245 Q6).

    Offline slicing changes only *when* an emission would have arrived, not what it says: an
    emission's text depends on audio up to the end of its chunk, so a check can lag the live
    stream by at most one 0.48 s call (#245 Q5).
    """
    checks = (_stream_check(span, emissions)
              for span in decoding.check_schedule(num_samples))
    return [check for check in checks if check is not None]


def w1_check(span: decoding.CheckSpan, emissions: list[Emission]) -> dict | None:
    """One fresh warm-started inference's check (the ``checks-w1`` keep rule).

    Drop only the warm-up: an emission whose midpoint is before the check's ``start``. Keep
    everything else the inference emitted, including what arrives after ``end`` — that is the
    delayed last word the 1.05 s flush forces out, which ours never drops. A window's hop is
    its first second after ``start``; a preview or the stop sends everything it kept.
    """
    start = span.start / STEP_SAMPLES
    kept = [emission for emission in emissions if emission.midpoint >= start]
    if span.kind == decoding.WINDOW:
        split = start + HOP_STEPS
        return decoding.make_check(
            decoding.WINDOW, _text(kept, start, split), _text(kept, split))
    kept_text = "".join(emission.text for emission in kept)
    if span.kind == decoding.PREVIEW:
        return decoding.make_check(decoding.PREVIEW, "", kept_text)
    if span.kind == decoding.STOP:
        return decoding.make_check(decoding.STOP, kept_text, "")
    raise ValueError(f"unknown check kind {span.kind!r}")


def _zipformer_arms(decoder: ZipformerDecoder, span: np.ndarray) -> dict:
    """One item's single-stream arms, as a zipformer record holds them (never ``stream_b0``)."""
    return {"spans": decoder.spans(span), "checks": decoder.checks(span)}


def _zipformer_w1_arms(decoder: ZipformerDecoder, span: np.ndarray) -> dict:
    """One item's two per-check arms (never ``stream_b0``)."""
    return {CHECKS_W1: decoder.checks_w1(span),
            CHECKS_W1_WINDOWS: decoder.checks_w1_windows(span)}


#: The zipformer's single-stream stores, one record per item.
ZIPFORMER_STORES = {
    "zipformer": Store(ZIPFORMER_DIR / "decodes",
                       {arm: ZIPFORMER_PROTOCOLS[arm] for arm in ("spans", "checks")},
                       _zipformer_arms),
}

#: The zipformer's per-check stores, decoded only on the sets the Muraja harness grades.
ZIPFORMER_W1_STORES = {
    "zipformer": Store(ZIPFORMER_DIR / "w1_decodes",
                       {arm: ZIPFORMER_PROTOCOLS[arm] for arm in (CHECKS_W1, CHECKS_W1_WINDOWS)},
                       _zipformer_w1_arms),
}


def scoring_arm(model: str) -> str:
    """The arm a model's whole-item text is scored from (#245 Q3)."""
    try:
        return SCORING_ARMS[model]
    except KeyError:
        raise ValueError(
            f"{model!r} has no scoring arm; known models: {sorted(SCORING_ARMS)}"
        ) from None


def metric_arm(model: str, metric: str) -> str:
    """The arm ``model`` is read from for ``metric``: app outcomes and whole items differ.

    Zipformer's app-outcome metrics (Muraja's own engine over checks) read ``checks-w1`` — its
    several looks — and its whole-item metrics read ``spans``. Ours read ``checks`` and
    ``stream_b0`` as before. ``REPORT_ONLY_ARMS`` can never be returned for zipformer.
    """
    if model not in SCORING_ARMS:
        raise ValueError(
            f"{model!r} has no metric arms; known models: {sorted(SCORING_ARMS)}")
    if metric in APP_METRICS:
        return CHECKS_W1 if model == "zipformer" else "checks"
    if metric in WHOLE_ITEM_METRICS:
        return scoring_arm(model)
    raise ValueError(f"{metric!r} is not a metric; known metrics: {sorted(METRICS)}")


def whole_item_text(line: Mapping, model: str) -> str:
    """The scored whole-item text of ``line``, refusing a zipformer line with a ``stream_b0``."""
    arm = scoring_arm(model)
    if arm == "spans" and "stream_b0" in line:
        raise ValueError(
            f"{model} is scored from its whole stream; a line holding a stream_b0 decode is "
            "one of ours, not this model's"
        )
    if arm not in line:
        raise ValueError(f"{model}: the line has no {arm!r} decode")
    return line[arm]


def zipformer_fingerprint(model: str, arm: str) -> DecodeFingerprint:
    """``arm``'s decode fingerprint: CPU, fp32, batch 1, and a mode no decode of ours holds."""
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


#: What one check's inference decodes, in samples: ours its span, zipformer's warm-started
#: arm its warm-up and span plus the card's flushing tail.
DECODED_SAMPLES = {
    "ours": lambda span: span.end - span.start,
    CHECKS_W1: lambda span: span.end - warmup_start(span) + TAIL_SAMPLES,
}


def decoded_audio_ratio(arm: str, num_samples: int) -> float:
    """The compute table's audio decoded per audio second, over an item of ``num_samples``."""
    try:
        decoded = DECODED_SAMPLES[arm]
    except KeyError:
        raise ValueError(
            f"{arm!r} has no decoded-audio figure; known arms: {sorted(DECODED_SAMPLES)}"
        ) from None
    if num_samples <= 0:
        raise ValueError(f"an item has positive audio, got {num_samples} samples")
    return sum(decoded(span) for span in decoding.check_schedule(num_samples)) / num_samples
