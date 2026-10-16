"""Acoustic perturbations of 16 kHz recitation: the grid #159's D2 decodes under.

``h448`` was distilled on Tadabur alone with every augmentation off
(:mod:`training.distill_student`), and it is 2-3x worse than the base teacher on app users'
and amateurs' audio (#156). D2 asks whether that is **acoustic fragility**: does ``h448``'s
decode move further from its own clean decode than the teacher's does when the same audio is
noisy, reverberant or phone-band? This module is the recipe, fixed and committed
(``tadabur/nonpro_gap/grid.json``, :func:`grid_record`) before any perturbed audio was decoded,
so a training arm can reuse it exactly.

Every condition is a pure function of the clean clip, its id and :data:`SEED`
(:func:`perturb`); every array is float32 at 16 kHz and the clip keeps its length. Nothing is
clipped: Tadabur already peaks above 1.0 and the feature extractor normalizes each window.

* **noise** (:data:`SNRS_DB`): a segment of one recording of
  ``obadx/freesound-commercial-50k-noise-only`` (:data:`NOISE`; speech and music filtered out,
  each recording under its own Freesound licence), the same segment for a clip at every SNR
  (:func:`noise_draw`), looped when shorter than the clip, scaled so the clip's mean power over
  the segment's is the SNR (:func:`add_noise`). The pool is the recordings of
  :data:`NOISE_SHARDS` that decode, last at least :data:`NOISE_MIN_SECONDS` and are not
  near-silent (:func:`pool_eligible`), ordered by Freesound id.
* **reverb** (:data:`RT60S_S`): the clip convolved with a synthetic room impulse response
  (:func:`synthetic_rir`): a unit direct path and a Gaussian tail from
  :data:`RIR_TAIL_ONSET_S` that decays 60 dB over the RT60, its energy :data:`DRR_DB` below the
  direct path's, drawn per clip; the result is scaled back to the clean clip's RMS.
* **band limit**: resampled to :data:`NARROWBAND_RATE` and back (a 4 kHz band, a telephone's
  sample rate; polyphase filters, :func:`band_limit`).
* **codec**: an Opus round trip at :data:`OPUS_BITRATE` bit/s, ``voip``, narrowband
  (:func:`opus_round_trip` under an explicit :class:`OpusConfig`, ``ffmpeg``'s ``libopus``),
  a low-bitrate phone call. The codec delays its output by about a sample and a half even
  with its pre-skip dropped, so every clip's decoded audio is moved back (:func:`advance`) by
  its configuration's lag, calibrated once per configuration on two fixed chirps and refused
  when they disagree (:func:`codec_lag`). The clip's own content never sets the shift.

**Every condition keeps the clip's timing**, to within :data:`MAX_RESIDUAL_LAG` samples, so a
perturbed clip's frames line up with the clean clip's: a training arm pairs clean teacher
frames with perturbed student frames through this module. The tolerance is set by the
model's frame hop, 20 ms or 320 samples at 16 kHz: 2 samples (0.125 ms) is under 1% of a
frame and cannot move a frame pairing. Zero is not reachable for the codec: its phase
response depends on frequency (a 300-900 Hz sweep lags about 0.8 samples, the whole band
about 1.6), so no single shift zeroes every band. The residual is verified on spectrally
distinct probes for every condition and every :data:`OPUS_CONFIGS` configuration
(:func:`verification_probes`, :func:`residual_lag`, :func:`codec_residuals`).

**Amended twice** (after review, before any training), the codec's alignment only; only the
codec condition was decoded again: v1 left the codec's lag in place; v2 removed a lag measured
on each clip, which some content measures wrongly; v3 (``acoustic-grid-v3``) removes the
codec's calibrated constant lag. (Review round 3 stated the timing contract as a tolerance,
made the codec's configuration explicit and calibrated per configuration; the grid's audio
did not change.)

Torch-free; the codec needs ``ffmpeg`` with ``libopus`` on the path.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from tadabur.audio import TARGET_SAMPLE_RATE
from tadabur.nonpro_sources import Dataset

GRID_VERSION = "acoustic-grid-v3"
#: Every draw (the sample, the noise segment, the impulse response) is salted with this.
SEED = 20261015

#: The noise corpus at its pinned revision. The collection states no licence of its own; each
#: recording carries its Freesound licence (``license``), recorded per recording drawn.
#: Downloaded to the GPU box's scratch only, never committed.
NOISE = Dataset(
    "freesound_noise", "obadx/freesound-commercial-50k-noise-only",
    "a5f48ab83b64ba1d35ee4ec363dae75d4e2c69d3",
    ("data/train-00000-of-00104.parquet", "data/train-00052-of-00104.parquet"),
    "per recording (Freesound: CC0-1.0, CC-BY-3.0/4.0, CC-BY-NC-3.0/4.0, CC-Sampling+)", False)
#: Two of the 104 shards, far apart in the collection's order (~1 GB, ~530 recordings).
NOISE_SHARDS = NOISE.shards
#: The ``license`` column's class names, in its order (the dataset card).
NOISE_LICENCES = ("CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "CC-BY-NC-3.0", "CC-BY-NC-4.0",
                  "CC-Sampling+")
NOISE_MIN_SECONDS = 1.0
#: -80 dBFS: a recording quieter than this is digital silence, which no SNR can scale.
NOISE_MIN_RMS = 1e-4
SNRS_DB = (20.0, 10.0, 5.0)

RT60S_S = (0.3, 0.6, 1.0)
#: Direct-to-reverberant ratio: a reciter near the phone.
DRR_DB = 3.0
#: Where the diffuse tail starts after the direct path (2.5 ms, ~0.9 m of extra path).
RIR_TAIL_ONSET_S = 0.0025

NARROWBAND_RATE = 8000

OPUS_BITRATE = 12_000
OPUS_APPLICATION = "voip"
#: ``libopus``'s narrowband cutoff: the codec keeps 4 kHz, as a phone call does.
OPUS_CUTOFF_HZ = 4000
#: ``libopus``'s frame size (its default, stated so a configuration names it).
OPUS_FRAME_MS = 20
#: Every condition keeps the clip's timing to within this many samples (module docstring).
MAX_RESIDUAL_LAG = 2.0
#: The codec's lag is measured within this many samples (1 ms); one outside is refused.
MAX_CODEC_LAG = 16
#: The calibration probes: slow chirps (a fast sweep codes poorly at this bitrate) up and
#: down the narrowband speech band.
CODEC_PROBE_SECONDS = 10.0
CODEC_PROBE_BAND_HZ = (300.0, 3400.0)
#: The two probes' lags may differ by this much (samples) before the calibration is refused.
CODEC_CALIBRATION_TOLERANCE = 0.25
#: A lag is measured only where the output's correlation with the reference, normalised by
#: both energies, peaks at least this high; anything weaker is refused, never guessed.
MIN_LAG_CORRELATION = 0.5

CLEAN, NOISE_KIND, REVERB, BAND_LIMIT, CODEC = "clean", "noise", "reverb", "band_limit", "codec"


@dataclass(frozen=True)
class Condition:
    """One cell of the grid: its name, its kind and its level (SNR in dB, RT60 in s, sample
    rate in Hz, bitrate in bit/s; none for clean)."""

    name: str
    kind: str
    level: float | None = None


GRID = (
    Condition("clean", CLEAN),
    *(Condition(f"noise_snr{snr:g}", NOISE_KIND, snr) for snr in SNRS_DB),
    *(Condition(f"reverb_rt{rt60:g}", REVERB, rt60) for rt60 in RT60S_S),
    Condition("band_8k", BAND_LIMIT, float(NARROWBAND_RATE)),
    Condition("opus_12k_nb", CODEC, float(OPUS_BITRATE)),
)
CONDITIONS = {c.name: c for c in GRID}


def grid_record() -> dict:
    """The grid as committed: every condition and every constant a condition reads."""
    return {
        "version": GRID_VERSION,
        "seed": SEED,
        "sample_rate": TARGET_SAMPLE_RATE,
        "conditions": [asdict(c) for c in GRID],
        "noise": {"repo": NOISE.repo, "revision": NOISE.revision, "shards": list(NOISE_SHARDS),
                  "licence": NOISE.licence, "min_seconds": NOISE_MIN_SECONDS,
                  "min_rms": NOISE_MIN_RMS, "snr": "clip mean power over the noise segment's",
                  "segment": "one recording and offset per clip (noise_draw), looped"},
        "reverb": {"rt60_s": list(RT60S_S), "drr_db": DRR_DB,
                   "tail_onset_s": RIR_TAIL_ONSET_S, "rir_length": "one RT60",
                   "tail": "Gaussian, drawn per clip, decaying 60 dB over the RT60",
                   "level": "scaled back to the clean clip's RMS"},
        "band_limit": {"sample_rate": NARROWBAND_RATE, "resampler": "scipy.signal.resample_poly"},
        "codec": {"codec": "opus (ffmpeg libopus)", "bitrate": OPUS_BITRATE,
                  "application": OPUS_APPLICATION, "cutoff_hz": OPUS_CUTOFF_HZ,
                  "alignment": "output moved back by the codec's constant lag, calibrated on "
                               f"an up and a down chirp ({CODEC_PROBE_BAND_HZ[0]:g}-"
                               f"{CODEC_PROBE_BAND_HZ[1]:g} Hz, {CODEC_PROBE_SECONDS:g} s) "
                               f"that must agree within {CODEC_CALIBRATION_TOLERANCE:g} samples",
                  "frame_ms": OPUS_FRAME_MS},
        "timing": f"every condition within {MAX_RESIDUAL_LAG:g} samples of the clean clip on "
                  "every verification probe (under 1% of the 20 ms frame hop)",
        "amended": "v2: the codec's output aligned to its input by a lag measured per clip "
                   "(v1 left a 1-2 sample lag); v3: by the codec's calibrated constant lag "
                   "instead (a per-clip measure can be wrong on some content); nothing else "
                   "changed",
        "clipping": "none; float32 throughout",
    }


def _unit(*parts: object) -> float:
    """A number in [0, 1) fixed by :data:`SEED` and ``parts``."""
    digest = hashlib.sha256(":".join(map(str, (SEED, *parts))).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _rng(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256(":".join(map(str, (SEED, *parts))).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _power(samples: np.ndarray) -> float:
    power = float(np.mean(np.square(samples, dtype=np.float64)))
    if not power > 0.0:
        raise ValueError("cannot set a level on silent audio")
    return power


# --- noise -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseRecording:
    """One recording of the noise pool, its provenance and its 16 kHz mono samples."""

    freesound_id: int
    licence: str
    username: str
    samples: np.ndarray = field(repr=False, compare=False)

    def as_dict(self) -> dict:
        return {"freesound_id": self.freesound_id, "licence": self.licence,
                "username": self.username,
                "seconds": round(len(self.samples) / TARGET_SAMPLE_RATE, 2),
                "samples_sha256": hashlib.sha256(self.samples.tobytes()).hexdigest()}


def pool_eligible(samples: np.ndarray) -> bool:
    """Long enough, finite, and not digital silence."""
    return (len(samples) >= NOISE_MIN_SECONDS * TARGET_SAMPLE_RATE
            and bool(np.isfinite(samples).all())
            and float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) > NOISE_MIN_RMS)


def noise_draw(pool_size: int, clip_id: str) -> tuple[int, float]:
    """Which recording of a pool of ``pool_size`` a clip gets, and where in it (a fraction of
    its length) its segment starts: the same at every SNR."""
    if pool_size < 1:
        raise ValueError("the noise pool is empty")
    return int(_unit("noise", clip_id) * pool_size), _unit("noise-offset", clip_id)


def noise_segment(noise: np.ndarray, offset: float, length: int) -> np.ndarray:
    """``length`` samples of ``noise`` from ``offset`` (a fraction), looping past its end."""
    start = int(offset * len(noise))
    return np.resize(np.roll(noise, -start), length)


def add_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """``clean`` plus ``noise`` (the same length) scaled to ``snr_db`` below it."""
    if len(noise) != len(clean):
        raise ValueError("the noise segment must be as long as the clip")
    gain = np.sqrt(_power(clean) / (_power(noise) * 10.0 ** (snr_db / 10.0)))
    return (clean + gain * noise.astype(np.float64)).astype(np.float32)


# --- reverb ------------------------------------------------------------------------------------


def synthetic_rir(rt60_s: float, rng: np.random.Generator) -> np.ndarray:
    """A unit direct path and a diffuse Gaussian tail from :data:`RIR_TAIL_ONSET_S` whose
    energy decays 60 dB over ``rt60_s`` and sums to :data:`DRR_DB` below the direct path's;
    one RT60 long."""
    length = int(round(rt60_s * TARGET_SAMPLE_RATE))
    onset = int(round(RIR_TAIL_ONSET_S * TARGET_SAMPLE_RATE))
    t = np.arange(length) / TARGET_SAMPLE_RATE
    tail = rng.standard_normal(length) * 10.0 ** (-3.0 * t / rt60_s)
    tail[:onset] = 0.0
    tail *= np.sqrt(10.0 ** (-DRR_DB / 10.0) / np.sum(np.square(tail)))
    tail[0] = 1.0
    return tail


def reverberate(clean: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """``clean`` convolved with ``rir``, cut to its length and scaled back to its RMS."""
    from scipy.signal import fftconvolve

    wet = fftconvolve(clean.astype(np.float64), rir)[: len(clean)]
    return (wet * np.sqrt(_power(clean) / _power(wet))).astype(np.float32)


# --- band limit and codec -----------------------------------------------------------------------


def band_limit(clean: np.ndarray, rate: int = NARROWBAND_RATE) -> np.ndarray:
    """Down to ``rate`` and back up, cut to the clip's length."""
    from math import gcd

    from scipy.signal import resample_poly

    step = gcd(TARGET_SAMPLE_RATE, rate)
    up, down = rate // step, TARGET_SAMPLE_RATE // step
    narrow = resample_poly(clean.astype(np.float64), up, down)
    return _fit(resample_poly(narrow, down, up), len(clean))


def _fit(samples: np.ndarray, length: int) -> np.ndarray:
    """Cut or zero-pad to ``length`` (a resampler or codec rounds the length)."""
    out = np.zeros(length, dtype=np.float32)
    out[: min(length, len(samples))] = samples[:length]
    return out


def _ffmpeg(args: list[str], data: bytes) -> bytes:
    result = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                             "-threads", "1", *args], input=data, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg {' '.join(args)}: {result.stderr.decode(errors='replace')}")
    return result.stdout


def lag_samples(output: np.ndarray, reference: np.ndarray, bound: int) -> float:
    """How many samples ``output`` lags ``reference`` (the same length): the cross-correlation
    peak, refined to a sub-sample by a parabola through it. Refused (``ValueError``) unless the
    reference has energy, the peak's correlation normalised by both energies reaches
    :data:`MIN_LAG_CORRELATION`, and the peak, searched for within four times ``bound``, lies
    inside ``bound``: a lag is measured on a known probe, never guessed."""
    from scipy.signal import correlate

    energy = float(np.sqrt(np.sum(np.square(output, dtype=np.float64))
                           * np.sum(np.square(reference, dtype=np.float64))))
    if not np.sqrt(np.mean(np.square(reference, dtype=np.float64))) > NOISE_MIN_RMS:
        raise ValueError("the reference is too quiet to measure a lag on")
    correlation = correlate(output.astype(np.float64), reference.astype(np.float64),
                            mode="full", method="fft")
    search = 4 * bound
    centre = len(reference) - 1
    window = correlation[centre - search:centre + search + 1]
    peak = int(np.argmax(window))
    if not window[peak] >= MIN_LAG_CORRELATION * energy:
        raise ValueError(f"no clear correlation peak ({window[peak] / energy:.3f} normalised)")
    if abs(peak - search) >= bound:
        raise ValueError(f"the lag {peak - search} is not within {bound} samples")
    before, at, after = window[peak - 1:peak + 2]
    curvature = before - 2 * at + after
    return peak - search + (0.5 * (before - after) / curvature if curvature else 0.0)


def advance(samples: np.ndarray, lag: float) -> np.ndarray:
    """``samples`` moved ``lag`` samples earlier (a fractional shift, by the spectrum's phase),
    zero-padded so nothing wraps round; the same length."""
    pad = int(np.ceil(abs(lag))) + 64
    padded = np.pad(samples.astype(np.float64), pad)
    spectrum = np.fft.rfft(padded) * np.exp(2j * np.pi * np.fft.rfftfreq(len(padded)) * lag)
    return np.fft.irfft(spectrum, len(padded))[pad:pad + len(samples)].astype(np.float32)


@dataclass(frozen=True)
class OpusConfig:
    """One Opus configuration: everything its lag depends on."""

    bitrate: int = OPUS_BITRATE
    application: str = OPUS_APPLICATION
    cutoff_hz: int = OPUS_CUTOFF_HZ
    frame_ms: int = OPUS_FRAME_MS

    @property
    def label(self) -> str:
        return f"{self.application}_{self.bitrate // 1000}k_cutoff{self.cutoff_hz}_{self.frame_ms}ms"


#: The grid's codec condition.
GRID_OPUS = OpusConfig()
#: The configurations a training arm's augmentation may use (#94): voip, narrowband, at
#: 8, 12 and 16 kbit/s; each is verified (:func:`codec_residuals`).
OPUS_CONFIGS = tuple(OpusConfig(bitrate) for bitrate in (8_000, 12_000, 16_000))


def _opus(clean: np.ndarray, config: OpusConfig) -> np.ndarray:
    """The Opus round trip as the codec leaves it: decoded back to 16 kHz (the decoder drops
    the pre-skip), not yet aligned."""
    raw = ["-f", "f32le", "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1"]
    encoded = _ffmpeg([*raw, "-i", "pipe:0", "-c:a", "libopus", "-b:a", str(config.bitrate),
                       "-application", config.application, "-cutoff", str(config.cutoff_hz),
                       "-frame_duration", str(config.frame_ms),
                       "-fflags", "+bitexact", "-flags:a", "+bitexact", "-f", "ogg", "pipe:1"],
                      clean.astype("<f4").tobytes())
    return _fit(np.frombuffer(_ffmpeg(["-i", "pipe:0", *raw, "pipe:1"], encoded),
                              dtype="<f4"), len(clean))


def _chirp(seconds: float, start_hz: float, end_hz: float) -> np.ndarray:
    from scipy.signal import chirp

    t = np.arange(int(seconds * TARGET_SAMPLE_RATE)) / TARGET_SAMPLE_RATE
    return (0.3 * chirp(t, start_hz, seconds, end_hz)).astype(np.float32)


def codec_probes() -> tuple[np.ndarray, np.ndarray]:
    """The calibration probes: a chirp up and a chirp down the narrowband speech band."""
    low, high = CODEC_PROBE_BAND_HZ
    return (_chirp(CODEC_PROBE_SECONDS, low, high), _chirp(CODEC_PROBE_SECONDS, high, low))


#: Each configuration's calibrated lag, measured once per process (:func:`codec_lag`).
_CODEC_LAGS: dict[OpusConfig, float] = {}


def codec_lag(config: OpusConfig) -> float:
    """``config``'s constant lag (samples): the mean of the two probes' measured lags, refused
    when they differ by more than :data:`CODEC_CALIBRATION_TOLERANCE`."""
    if config not in _CODEC_LAGS:
        lags = [lag_samples(_opus(probe, config), probe, MAX_CODEC_LAG)
                for probe in codec_probes()]
        if max(lags) - min(lags) > CODEC_CALIBRATION_TOLERANCE:
            raise ValueError(f"{config.label}: the codec's lag is unstable across the probes: "
                             f"{lags}")
        _CODEC_LAGS[config] = float(np.mean(lags))
    return _CODEC_LAGS[config]


def opus_round_trip(clean: np.ndarray, config: OpusConfig = GRID_OPUS) -> tuple[np.ndarray, float]:
    """Encoded to Opus under ``config``, decoded back to 16 kHz and moved back by the
    configuration's calibrated lag (:func:`codec_lag`, the same for every clip); the aligned
    audio and that lag."""
    lag = codec_lag(config)
    return advance(_opus(clean, config), lag), lag


# --- verifying the timing ----------------------------------------------------------------------

#: An impulse's sample in its probe.
IMPULSE_AT = 16_000


def verification_probes() -> dict[str, np.ndarray]:
    """Spectrally distinct probes none of which calibrated anything: 8 s chirps over the low,
    mid and high speech band and over the whole band up and down, and a unit impulse."""
    impulse = np.zeros(3 * IMPULSE_AT + 1, dtype=np.float32)
    impulse[IMPULSE_AT] = 1.0
    return {"low_300_900": _chirp(8, 300, 900), "mid_900_2000": _chirp(8, 900, 2000),
            "high_2000_3400": _chirp(8, 2000, 3400), "full_up": _chirp(8, 300, 3400),
            "full_down": _chirp(8, 3400, 300), "impulse": impulse}


def residual_lag(perturbed: np.ndarray, name: str, probe: np.ndarray) -> float:
    """How far ``perturbed`` still lags ``probe`` (samples): its largest sample's offset for
    the impulse, the measured lag (:func:`lag_samples`, refusing a weak peak) for a chirp."""
    if name == "impulse":
        return float(int(np.argmax(np.abs(perturbed))) - IMPULSE_AT)
    return lag_samples(perturbed, probe, MAX_CODEC_LAG)


def codec_residuals(configs: tuple[OpusConfig, ...] = OPUS_CONFIGS) -> dict:
    """Per configuration, its calibrated lag and each verification probe's raw and residual
    lag after the shift (samples, rounded to 3 places)."""
    out = {}
    for config in configs:
        probes = verification_probes()
        raw = {n: residual_lag(_opus(p, config), n, p) for n, p in probes.items()}
        shifted = {n: residual_lag(opus_round_trip(p, config)[0], n, p)
                   for n, p in probes.items()}
        out[config.label] = {"calibrated_lag": round(codec_lag(config), 3),
                             "raw": {n: round(v, 3) for n, v in raw.items()},
                             "residual": {n: round(v, 3) for n, v in shifted.items()}}
    return out


def ffmpeg_version() -> str:
    return _ffmpeg(["-version"], b"").decode().splitlines()[0]


# --- one condition -----------------------------------------------------------------------------


def perturb(clean: np.ndarray, condition: Condition, clip_id: str,
            pool: tuple[NoiseRecording, ...] = ()) -> tuple[np.ndarray, dict]:
    """``clean`` under ``condition``, and what was drawn or measured for it (the noise
    recording and offset, the codec's lag; nothing for the other kinds, whose draws are fixed
    by :data:`SEED`)."""
    if condition.kind == CLEAN:
        return clean, {}
    if condition.kind == NOISE_KIND:
        index, offset = noise_draw(len(pool), clip_id)
        recording = pool[index]
        segment = noise_segment(recording.samples, offset, len(clean))
        return (add_noise(clean, segment, condition.level),
                {"freesound_id": recording.freesound_id, "noise_offset": round(offset, 6)})
    if condition.kind == REVERB:
        return reverberate(clean, synthetic_rir(condition.level, _rng("rir", condition.name,
                                                                       clip_id))), {}
    if condition.kind == BAND_LIMIT:
        return band_limit(clean, int(condition.level)), {}
    if condition.kind == CODEC:
        coded, lag = opus_round_trip(clean, OpusConfig(bitrate=int(condition.level)))
        return coded, {"codec_lag": round(lag, 3)}
    raise ValueError(f"unknown condition kind {condition.kind!r}")


# --- the noise pool ----------------------------------------------------------------------------


def build_pool(cache_dir: Path) -> tuple[tuple[NoiseRecording, ...], dict]:
    """The eligible recordings of :data:`NOISE_SHARDS` (downloaded into ``cache_dir``, each shard
    deleted once read), ordered by Freesound id, and what was left out and why."""
    from tadabur.audio import decode_to_mono_16k
    from tadabur.nonpro_sources import iter_rows
    from tadabur.panel_seal import SealedPanelError

    pool, left_out = [], {"undecodable": 0, "ineligible": 0}
    for shard in NOISE_SHARDS:
        for row in iter_rows(NOISE, shard, cache_dir):
            try:
                samples = decode_to_mono_16k(row["audio"]["bytes"])
            except SealedPanelError:  # a seal violation is never a corrupt input
                raise
            except Exception:  # any format libsndfile cannot read is counted
                left_out["undecodable"] += 1
                continue
            if not pool_eligible(samples):
                left_out["ineligible"] += 1
                continue
            pool.append(NoiseRecording(int(row["freesound_id"]), NOISE_LICENCES[row["license"]],
                                       row["username"], samples))
    pool.sort(key=lambda r: r.freesound_id)
    return tuple(pool), left_out


def save_pool(pool: tuple[NoiseRecording, ...], path: Path) -> None:
    """The pool's samples as one ``.npz`` on scratch (never committed)."""
    np.savez(path, ids=np.array([r.freesound_id for r in pool]),
             lengths=np.array([len(r.samples) for r in pool]),
             samples=np.concatenate([r.samples for r in pool]))


def load_pool(path: Path, records: list[dict]) -> tuple[NoiseRecording, ...]:
    """The pool saved by :func:`save_pool`, checked against its committed ``records``."""
    with np.load(path) as data:  # each key read decompresses its whole array: read once
        ids, lengths, samples = data["ids"], data["lengths"], data["samples"]
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    pool = tuple(NoiseRecording(r["freesound_id"], r["licence"], r["username"],
                                samples[bounds[i]:bounds[i + 1]])
                 for i, r in enumerate(records))
    if [r.as_dict() for r in pool] != records or list(ids) != [r.freesound_id for r in pool]:
        raise ValueError(f"{path} is not the committed noise pool")
    return pool
