"""Lengthen one span of a recording without touching the rest: a hold, stretched in place.

ADR-0011 names stretching the hold as the synthetic edit for shaddah, and the shaddah probe
(#86, :mod:`training.shaddah_probe`) asks whether a longer hold flips a decode to a doubled
consonant. Both need the same edit: the samples ``[start, end)`` of a 16 kHz recording
played ``factor`` times as long, pitch unchanged, and every other sample left as it was.

The edit is **WSOLA** (waveform-similarity overlap-add) with a piecewise-linear time map:
output samples before ``start`` read the input one-to-one, the stretched span reads it at
``1 / factor`` speed, and everything after reads it one-to-one again, shifted by the added
length. Hann frames of :data:`FRAME` samples are overlap-added at half a frame apart, which
reconstructs the input exactly wherever every frame sits at its nominal position. Only the
frames that touch the stretched span may move, by up to :data:`TOLERANCE` samples, to the
position whose waveform best continues the previous frame -- which is what keeps a voiced
hold periodic instead of phasey. So the recording is bit-for-bit unchanged up to two frames
before the span and, shifted, from two frames after it.
"""

from __future__ import annotations

import numpy as np

# 20 ms frames and a +-5 ms search at 16 kHz: long enough to hold two pitch periods of a
# low voice, short enough that a frame stays inside one consonant.
FRAME = 320
HOP = FRAME // 2
TOLERANCE = 80


def added_samples(start: int, end: int, factor: float) -> int:
    """How many samples stretching ``[start, end)`` by ``factor`` adds."""
    return int(round((factor - 1.0) * (end - start)))


def stretch_span(samples: np.ndarray, start: int, end: int, factor: float) -> np.ndarray:
    """``samples`` with ``[start, end)`` played ``factor`` (>= 1) times as long, as float32."""
    if not 0 <= start < end <= len(samples):
        raise ValueError(f"span [{start}, {end}) is not inside {len(samples)} samples")
    if factor < 1.0:
        raise ValueError(f"factor must be at least 1, got {factor}")
    if factor == 1.0:
        # Exactly the input: the search below maximises an unnormalised correlation, which
        # can prefer a louder shifted frame even when nothing is to be stretched.
        return np.array(samples, dtype=np.float32, copy=True)
    x = np.asarray(samples, dtype=np.float64)
    added = added_samples(start, end, factor)
    out_len = len(x) + added
    stretched_end = end + added  # where the stretched span ends in the output

    def source(position: float) -> float:
        """The input position the output position ``position`` reads."""
        if position < start:
            return position
        if position < stretched_end:
            return start + (position - start) * (end - start) / (stretched_end - start)
        return position - added

    pad = FRAME + TOLERANCE
    padded = np.pad(x, (pad, pad + added + FRAME))
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / FRAME)  # periodic Hann
    out = np.zeros(out_len + 2 * FRAME)
    norm = np.zeros_like(out)
    previous = None
    for o in range(-HOP, out_len, HOP):
        nominal = int(round(source(o + HOP) - HOP))
        touches = o < stretched_end + FRAME and o + FRAME > start - FRAME
        position = nominal
        if touches and previous is not None:
            template = padded[pad + previous + HOP : pad + previous + HOP + FRAME]
            region = padded[pad + nominal - TOLERANCE : pad + nominal + TOLERANCE + FRAME]
            position = nominal - TOLERANCE + int(np.argmax(np.correlate(region, template)))
        out[o + HOP : o + HOP + FRAME] += window * padded[pad + position : pad + position + FRAME]
        norm[o + HOP : o + HOP + FRAME] += window
        previous = position
    out = out[HOP : HOP + out_len] / np.maximum(norm[HOP : HOP + out_len], 1e-8)
    return out.astype(np.float32)
