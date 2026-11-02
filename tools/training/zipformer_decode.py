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
  **No call is padded**: a call runs only while a whole :data:`INPUT_FRAMES` window is
  available, which is sherpa-onnx's readiness rule. Our zero-padded final call emitted tokens
  the reference runtime never produces (the parity gate's phone differences), so the one
  diagnosis round removed it (parity r1, 2026-11-01). The tail's silence is still what lets
  the last *complete* chunk flush a delayed word out.
  :func:`stream_checks` slices that one timeline with the transcriber's own schedule
  (:func:`training.decoding.check_schedule`); a window never restarts the model from zero.
* **One fresh inference per check** (``checks-w1``, architect r2 + principal r1). Muraja runs
  an inference per check and gets several looks at a word; the single stream gives zipformer
  one. ``checks-w1`` therefore decodes every check of the schedule again, over
  ``[max(0, start - W), end)`` with ``W = 1 s`` (:data:`WARMUP_SAMPLES`, the transcriber's hop)
  from a zero cache, and drops only the warm-up emissions (midpoint before the check's start):
  everything else is kept, the 1.05 s flush included, so a delayed last word is not truncated.
  ``checks-w1-windows`` runs the fresh inferences for the windows only (reusing ``checks-w1``'s
  window results when both arms are decoded on one item) and takes previews and the stop from
  the one continuous stream (report-only). Both are zipformer-only arms: ours never decodes
  them.
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

    from training.zipformer_decode import ZipformerFiles, load_zipformer

    files = ZipformerFiles(onnx=Path("..."), int8=Path("..."),
                           text2phoneme=Path("..."), manifest=Path("..."),
                           revision="...", benchmark_revision="...")
    decoder = load_zipformer(files, scale=1.0)   # writes identity.json when given its path
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training import decoding
from training.decoding import DecodeFingerprint
from training.distill_data import SAMPLE_RATE
from training.external_decode import record_identity
from training.truth_baseline import Store, file_sha256
from training.zipformer_stream import (BLANK_ID, CHUNK_FRAMES, INPUT_FRAMES, STEP_SAMPLES,
                                       TAIL_SAMPLES, Emission, detokenise, featurize,
                                       read_tokens, scan_units, state_specs, stream_frame_ids)

#: The card's committed symbol table, unit inventory and licence, beside the decodes they back.
ZIPFORMER_DIR = Path(__file__).resolve().parent.parent / "tadabur" / "zipformer_screen"

#: The Hub repositories the card and the benchmark live in, and the files this screen reads.
MODEL_REPO = "Quran-Lab/zipformer_p-arabic-v3"
BENCHMARK_REPO = "Quran-Lab/quranic-asr-benchmark"
ONNX_FILE = "zipformer_p_arabic_v3.2.onnx"
INT8_FILE = "zipformer_p_arabic_v3.2.int8.onnx"
#: v3's own export: the model the card's PER table describes (its README table is dated
#: 2026-08-16, before v3.1). Decoded once as a decoder check, never on a comparison set.
CARD_ONNX_FILE = "zipformer_p_arabic_v3.onnx"
TEXT2PHONEME_FILE = "quran_text2phoneme.json"
MANIFEST_FILE = "benchmark.jsonl"
#: The card's left context: 256 frames at 50 Hz, about 5.1 s (its left-context probe README).
LEFT_CONTEXT_FRAMES = 256
#: The card's chunk: 24 output frames (0.48 s) per call, :data:`CHUNK_FRAMES` input frames.
CHUNK_SIZE = 24
#: The identity the decodes, and the gate's pinned scale, rest on.
IDENTITY_PATH = ZIPFORMER_DIR / "identity.json"

#: The protocol's constants and its run collapse are :mod:`training.zipformer_stream`'s,
#: imported below: the parity gate's trace drives the same ONNX with that module alone.
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


def _w1_mode(windows: bool) -> str:
    """The mode of a warm-started per-check arm, naming its two constants in milliseconds."""
    warm_up_ms = WARMUP_SAMPLES * 1000 // SAMPLE_RATE
    flush_ms = TAIL_SAMPLES * 1000 // SAMPLE_RATE
    return (f"zipformer-checks-v2-w{warm_up_ms}ms-flush{flush_ms}ms"
            + ("-windows" if windows else ""))


#: The decode mode of each arm, and its fingerprint policy. All four differ from every mode and
#: policy our own decodes carry, so the two kinds can never be compared by accident. ``v2`` is
#: the end-of-input repair (parity r1): a decode made under ``v1`` decoded a zero-padded final
#: chunk the reference runtime never decodes, so the two protocols must never be pooled.
ZIPFORMER_PROTOCOLS = {
    "spans": "zipformer-spans-v2",
    "checks": "zipformer-checks-v2",
    CHECKS_W1: _w1_mode(windows=False),
    CHECKS_W1_WINDOWS: _w1_mode(windows=True),
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


class ZipformerDecoder:
    """The card's streaming CTC decoder behind our arms: one stream, and one per check.

    ``session`` is an ``onnxruntime.InferenceSession`` of the card's cache-aware streaming
    export (``x`` plus its cache inputs; ``log_probs`` first, then each cache's next value in
    input order). ``tokens`` is :func:`read_tokens`' table. ``scale`` is :func:`featurize`'s.

    The decoder keeps **one item's** work: the continuous stream and each fresh check, so the
    arms called on one item share them. That work belongs to the exact array passed and to its
    contents (compared against a copy), so a different array, or a reused buffer rewritten in
    place, starts over rather than serving the previous item's text.
    """

    def __init__(self, session, tokens: Mapping[int, str], fbank, *, scale: float) -> None:
        self.session = session
        self.tokens = tokens
        self.fbank = fbank
        self.scale = scale
        inputs = session.get_inputs()
        if not any(spec.name == "x" for spec in inputs):
            raise ValueError("the ONNX session has no 'x' feature input")
        self._state_specs = state_specs(session)
        self._item_source: np.ndarray | None = None
        self._item_audio: np.ndarray | None = None
        self._item_emissions: list[Emission] | None = None
        self._item_checks: dict[tuple[int, int], dict | None] = {}

    def _start_item(self, samples: np.ndarray) -> None:
        """Point the one-item caches at ``samples``, clearing them for any other item.

        The caches belong to the exact array object passed. A different array is a different
        item (even when its samples happen to equal the last one's), and the same array with
        its contents changed is a different item too: its audio is compared against a copy, so
        a reused buffer is decoded again rather than served the previous item's text. The copy
        costs one item's audio per item, against that item's fbank and ONNX passes.
        """
        if (self._item_source is samples and self._item_audio is not None
                and np.array_equal(self._item_audio, samples)):
            return
        self._item_source = samples
        self._item_audio = np.array(samples, dtype=np.float32, copy=True)
        self._item_emissions = None
        self._item_checks = {}

    def _emissions(self, samples: np.ndarray) -> list[Emission]:
        """The item's emissions, from one stream shared by every arm that slices it."""
        self._start_item(samples)
        if self._item_emissions is None:
            self._item_emissions = self._stream(samples, 0)
        return self._item_emissions

    def stream(self, samples: np.ndarray) -> list[Emission]:
        """The item's units over one continuous stream: one fbank, one cache, one timeline."""
        return self._emissions(samples)

    def stream_calls(self, samples: np.ndarray) -> list[np.ndarray]:
        """Every call's argmax frame ids over one stream, in call order, from a zero cache.

        The loop is :func:`stream_frame_ids`, sherpa-onnx's readiness rule included, so the
        parity gate's own trace of the reference front end drives the model exactly as the
        arms do (see that function's rule and its one diagnosis round).
        """
        features = featurize(samples, self.fbank, scale=self.scale)
        if not len(features):
            raise ValueError("the front end produced no frames")
        return stream_frame_ids(self.session, features, self._state_specs)

    def _stream(self, samples: np.ndarray, offset_steps: int) -> list[Emission]:
        """One stream over ``samples``, its steps reported ``offset_steps`` later.

        A fresh zero cache, never carried in from another call or item.
        """
        emissions = scan_units(np.concatenate(self.stream_calls(samples)), self.tokens)
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

    def _decode_fresh_check(self, samples: np.ndarray,
                            span: decoding.CheckSpan) -> dict | None:
        """One check decoded again from a zero cache, with its one hop of warm-up.

        Records the result for the current item, so ``checks_w1_windows`` can reuse this span.
        """
        begin = warmup_start(span)
        emissions = self._stream(samples[begin:span.end], begin // STEP_SAMPLES)
        check = w1_check(span, emissions)
        self._item_checks[(span.start, span.end)] = check
        return check

    def _fresh_check(self, samples: np.ndarray, span: decoding.CheckSpan) -> dict | None:
        """``checks_w1``'s stored result for ``span``, or a fresh one if this item lacks it.

        The caller has already started the item; a different item clears the record.
        """
        key = (span.start, span.end)
        if key not in self._item_checks:
            self._decode_fresh_check(samples, span)
        return self._item_checks[key]

    def checks_w1(self, samples: np.ndarray) -> list[dict]:
        """The checks, each from its own fresh warm-started inference (the ``checks-w1`` arm)."""
        self._start_item(samples)
        checks = (self._decode_fresh_check(samples, span)
                  for span in decoding.check_schedule(len(samples)))
        return [check for check in checks if check is not None]

    def checks_w1_windows(self, samples: np.ndarray) -> list[dict]:
        """B': windows fresh, previews and the stop from one stream (``checks-w1-windows``).

        The windows reuse the fresh checks ``checks_w1`` already made for this item.
        """
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


@dataclass(frozen=True)
class ZipformerFiles:
    """The card's files a run fetched, and the Hub revisions ``tokens.txt``/the table came at.

    The ONNX, int8 and text-to-phoneme paths are the fetched copies outside the checkout;
    ``tokens.txt`` and ``phoneme_units.json`` are the committed copies the card's own hashes
    pin, so an identity record can name them by SHA-256 without a path.
    """

    onnx: Path
    int8: Path
    text2phoneme: Path
    manifest: Path
    revision: str
    benchmark_revision: str

    def identity(self, *, scale: float) -> dict:
        """The weights, front end and gate constants every decode under these files rests on."""
        return {
            "model_ref": MODEL_REPO,
            "hub_revision": self.revision,
            "benchmark": {
                "model_ref": BENCHMARK_REPO,
                "hub_revision": self.benchmark_revision,
                "manifest_sha256": file_sha256(self.manifest),
            },
            "files": {path.name: file_sha256(path)
                      for path in (self.onnx, self.int8, self.text2phoneme)},
            "tokens_sha256": file_sha256(ZIPFORMER_DIR / "tokens.txt"),
            "units_sha256": file_sha256(ZIPFORMER_DIR / "phoneme_units.json"),
            "chunk": CHUNK_SIZE,
            "chunk_frames": CHUNK_FRAMES,
            "input_frames": INPUT_FRAMES,
            "left_context_frames": LEFT_CONTEXT_FRAMES,
            "fbank": dict(FBANK_OPTIONS),
            "sample_scale": scale,
        }


def load_zipformer(files: ZipformerFiles, *, scale: float,
                   identity_path: Path | None = None) -> ZipformerDecoder:
    """The card's decoder on CPU, and the identity its decodes rest on when asked.

    The front end is the card's own (kaldi fbank, povey, no dither, ``snip_edges`` false) with
    :data:`TAIL_SAMPLES` of silence appended, and ``scale`` multiplying the waveform first:
    the card does not say whether kaldi wants [-1, 1] or int16-scaled samples, so the gate pins
    it (:func:`training.zipformer_gate.pin_scale`) before any screen set is decoded. ONNX runs
    with one thread per session: the screen parallelises whole items across processes, and a
    per-session thread pool would oversubscribe the box.

    ``identity_path`` writes the shared record (or refuses a run whose weights, table or
    settings differ). The record is the same shape as every other model's
    (:func:`training.external_decode.load_decoders`), so the arms' fingerprints key the same.
    """
    import onnxruntime as ort
    from torchaudio.compliance import kaldi

    def fbank(waveform, **options):
        """The card's kaldi fbank over one 1-D float32 waveform (torchaudio wants 2-D)."""
        import torch

        return kaldi.fbank(torch.from_numpy(np.asarray(waveform))[None], **options).numpy()

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(Path(files.onnx)), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    decoder = ZipformerDecoder(session, read_tokens(ZIPFORMER_DIR / "tokens.txt"), fbank,
                               scale=scale)
    if identity_path is not None:
        record = {
            "model": "zipformer",
            "identity": files.identity(scale=scale),
            "fingerprints": {arm: zipformer_fingerprint("zipformer", arm).as_dict()
                             for arm in ZIPFORMER_PROTOCOLS},
        }
        record_identity(Path(identity_path), record, label="zipformer")
    return decoder


#: What one check's inference decodes, in samples: ours its span, zipformer's warm-started
#: arm its warm-up and span plus the card's flushing tail; its windows-only arm keeps that
#: work for the windows and cuts previews and the stop from its one stream.
DECODED_SAMPLES = {
    "ours": lambda span: span.end - span.start,
    CHECKS_W1: lambda span: span.end - warmup_start(span) + TAIL_SAMPLES,
    CHECKS_W1_WINDOWS: lambda span: (span.end - warmup_start(span) + TAIL_SAMPLES
                                     if span.kind == decoding.WINDOW else 0),
}


def arms_decoded_samples(arms: Iterable[str], num_samples: int) -> int:
    """The audio one item's decode hands the model over ``arms`` decoded together.

    ``spans`` and ``checks`` are one stream. ``checks-w1`` decodes every check of the schedule
    from its own zero cache. ``checks-w1-windows`` adds one stream, and its windows' fresh
    inferences only when ``checks-w1`` did not already make them (the two arms share one
    decoder per item, so a run of both decodes each window once). This is the compute table's
    arithmetic and the timing probe's projection, from the same rule.
    """
    arms = set(arms)
    unknown = arms - set(ZIPFORMER_PROTOCOLS)
    if unknown:
        raise ValueError(
            f"not zipformer arms: {sorted(unknown)}; its arms are {sorted(ZIPFORMER_PROTOCOLS)}")
    if num_samples <= 0:
        raise ValueError(f"an item has positive audio, got {num_samples} samples")
    schedule = decoding.check_schedule(num_samples)
    total = num_samples if arms & {"spans", "checks"} else 0
    if CHECKS_W1 in arms:
        total += sum(DECODED_SAMPLES[CHECKS_W1](span) for span in schedule)
    if CHECKS_W1_WINDOWS in arms:
        total += num_samples  # its one stream
        if CHECKS_W1 not in arms:
            total += sum(DECODED_SAMPLES[CHECKS_W1_WINDOWS](span) for span in schedule)
    return total


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
