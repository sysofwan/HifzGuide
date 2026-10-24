"""An audio fingerprint for "is this the same recording?" across re-encodes and re-cuts (#144).

QUD's MP3s and Tadabur's WAVs of one recording differ in codec, sample rate, level and where
each dataset cut the ayah. A Haitsma–Kalker style fingerprint survives all of that and still
tells two recordings of the same text by the same voice apart, because it encodes the fine
time-frequency texture of the waveform rather than what was said:

* the 16 kHz signal is cut into frames of :data:`FRAME_S` every :data:`HOP_S`;
* each frame's energy is summed in :data:`BANDS` + 1 log-spaced bands between
  :data:`LOW_HZ` and :data:`HIGH_HZ`;
* bit ``m`` of frame ``n`` is the sign of the band-energy difference
  ``(E[n, m] − E[n, m+1]) − (E[n−1, m] − E[n−1, m+1])``: one 32-bit word per hop.

Two fingerprints are compared by their **bit error rate** (BER) at the best relative offset
(:func:`best_match`): the share of differing bits over the frames the two overlap. One
recording, re-encoded and re-cut, scores near 0; two different recordings score near 0.5.
Where the decision line lies between them is not assumed here: it is read off the committed
controls (:mod:`tadabur.qud_overlap`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import TARGET_SAMPLE_RATE

FRAME_S = 0.37
HOP_S = 0.0116
BANDS = 32
LOW_HZ = 300.0
HIGH_HZ = 2000.0
FRAME = round(FRAME_S * TARGET_SAMPLE_RATE)
HOP = round(HOP_S * TARGET_SAMPLE_RATE)
#: Frames a comparison must overlap to count (about 2 s).
MIN_OVERLAP = round(2.0 / HOP_S)


def _band_matrix() -> np.ndarray:
    """``(BANDS + 1, FRAME // 2 + 1)``: which FFT bins sum into each band."""
    freqs = np.fft.rfftfreq(FRAME, 1 / TARGET_SAMPLE_RATE)
    edges = np.geomspace(LOW_HZ, HIGH_HZ, BANDS + 2)
    return np.stack([(freqs >= lo) & (freqs < hi) for lo, hi in zip(edges[:-1], edges[1:])]
                    ).astype(np.float64)


_BANDS = _band_matrix()
_WINDOW = np.hanning(FRAME)
_WEIGHTS = (1 << np.arange(BANDS, dtype=np.uint64)).astype(np.uint64)


def fingerprint(samples: np.ndarray) -> np.ndarray:
    """One ``uint32`` per hop of a 16 kHz mono waveform (module docstring); empty when the
    waveform is shorter than two frames."""
    samples = np.asarray(samples, dtype=np.float64)
    if len(samples) < FRAME + HOP:
        return np.zeros(0, dtype=np.uint32)
    frames = np.lib.stride_tricks.sliding_window_view(samples, FRAME)[::HOP]
    power = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)) ** 2
    energy = power @ _BANDS.T  # (frames, BANDS + 1)
    across = energy[:, :-1] - energy[:, 1:]
    bits = (across[1:] - across[:-1]) > 0  # (frames - 1, BANDS)
    return (bits.astype(np.uint64) @ _WEIGHTS).astype(np.uint32)


def save_fingerprints(path: Path, prints: Mapping[str, tuple[str, np.ndarray]]) -> None:
    """Write ``clip -> (sha256, fingerprint)`` as one ``clip|sha256`` npz member each."""
    np.savez_compressed(path, **{f"{key}|{sha}": fp for key, (sha, fp) in prints.items()})


def load_fingerprints(path: Path) -> dict[str, np.ndarray]:
    """The fingerprints :func:`save_fingerprints` wrote, keyed by clip id again."""
    with np.load(path) as data:
        return {name.split("|")[0]: data[name] for name in data.files}


@dataclass(frozen=True)
class Match:
    """The best alignment of two fingerprints: its BER, the offset of ``b`` against ``a``
    (frames; ``b[i]`` lines up with ``a[i + offset]``) and how many frames overlap."""

    ber: float
    offset: int
    overlap: int


def best_match(a: np.ndarray, b: np.ndarray, min_overlap: int = MIN_OVERLAP) -> Match | None:
    """The lowest BER over every offset at which ``a`` and ``b`` overlap by at least
    ``min_overlap`` frames (or by all of the shorter one, when it is shorter than that);
    ``None`` when they cannot overlap that much."""
    need = min(min_overlap, len(a), len(b))
    if need == 0:
        return None
    best: Match | None = None
    for offset in range(-(len(b) - need), len(a) - need + 1):
        lo, hi = max(0, offset), min(len(a), offset + len(b))
        if hi - lo < need:
            continue
        differing = np.bitwise_count(a[lo:hi] ^ b[lo - offset:hi - offset]).sum()
        ber = float(differing) / ((hi - lo) * BANDS)
        if best is None or ber < best.ber:
            best = Match(round(ber, 4), offset, hi - lo)
    return best
