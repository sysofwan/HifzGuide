"""Sample-exact waveform edits for synthetic edits and their decoys (#88).

Two primitives, both pure numpy, both changing the waveform only around the span they are
given:

* :func:`crop` removes ``[start, start + length)``;
* :func:`replace_span` replaces ``[start, end)`` (empty for an insertion) with new material.

Every join is a Hann crossfade of :data:`FADE` samples centred on the cut, so no click is
introduced. Outside the crossfades the output is the input, sample for sample, before the
span and shifted by the length change after it; :func:`changed_region` names the output
samples that differ. :func:`fill` builds material of any length from a region of the
waveform, for stretching a hold.
"""

from __future__ import annotations

import numpy as np

#: Crossfade length at every join: 10 ms at 16 kHz. Even, so it centres on a cut.
FADE = 160

#: Pitch range searched when deciding whether a region is voiced (:func:`fill`).
MIN_PITCH_HZ = 70
MAX_PITCH_HZ = 400
#: Normalized autocorrelation above which a region is treated as periodic.
VOICED_CORRELATION = 0.5
#: Target length of one repeated unit: about two pitch periods of a typical voice.
UNIT_SECONDS = 0.03


def _ramp(fade: int) -> np.ndarray:
    """Fade-in weights of a Hann crossfade; the fade-out is ``1 - ramp``."""
    return 0.5 - 0.5 * np.cos(np.pi * (np.arange(fade) + 0.5) / fade)


def join(left: np.ndarray, right: np.ndarray, fade: int = FADE) -> np.ndarray:
    """``left`` then ``right``, overlapping by ``fade`` samples: ``len(left) + len(right) -
    fade`` samples, the overlap crossfaded."""
    if fade > min(len(left), len(right)):
        raise ValueError(f"cannot crossfade {fade} samples over {len(left)} and {len(right)}")
    ramp = _ramp(fade)
    overlap = left[len(left) - fade:] * (1.0 - ramp) + right[:fade] * ramp
    return np.concatenate([left[: len(left) - fade], overlap, right[fade:]]).astype(np.float32)


def crop(x: np.ndarray, start: int, length: int, fade: int = FADE) -> np.ndarray:
    """``x`` without ``[start, start + length)``: ``len(x) - length`` samples."""
    half = fade // 2
    if start - half < 0 or start + length + half > len(x) or length < 1:
        raise ValueError(f"crop [{start}, {start + length}) does not fit in {len(x)} samples")
    return join(x[: start + half], x[start + length - half:], fade)


def replace_span(
    x: np.ndarray, start: int, end: int, material: np.ndarray, fade: int = FADE
) -> np.ndarray:
    """``x`` with ``[start, end)`` replaced by ``material[fade // 2 : -fade // 2]``.

    ``material`` carries ``fade // 2`` samples of its own context on each side, which the
    two joins crossfade against ``x``'s, so the output holds ``len(x) - (end - start) +
    len(material) - fade`` samples. ``start == end`` inserts.
    """
    half = fade // 2
    if start - half < 0 or end + half > len(x) or end < start:
        raise ValueError(f"span [{start}, {end}) does not fit in {len(x)} samples")
    if len(material) < 2 * fade:
        raise ValueError(f"material of {len(material)} samples is too short to join")
    return join(join(x[: start + half], material, fade), x[end - half:], fade)


def changed_region(start: int, inserted: int, fade: int = FADE) -> tuple[int, int]:
    """The output samples :func:`replace_span` (``inserted`` new samples at ``start``) or
    :func:`crop` (``inserted = 0``) may change; everything else is the input's."""
    half = fade // 2
    return start - half, start + inserted + half


def _period(region: np.ndarray, sample_rate: int) -> int | None:
    """The pitch period of ``region`` in samples, or ``None`` when it is not periodic."""
    centred = region - region.mean()
    energy = float(np.dot(centred, centred))
    low, high = sample_rate // MAX_PITCH_HZ, sample_rate // MIN_PITCH_HZ
    if energy == 0.0 or len(centred) < 2 * high:
        return None
    correlations = [
        float(np.dot(centred[:-lag], centred[lag:])) / energy for lag in range(low, high + 1)
    ]
    best = int(np.argmax(correlations))
    return low + best if correlations[best] >= VOICED_CORRELATION else None


def fill(
    x: np.ndarray, region: tuple[int, int], length: int, at: int, seed: int,
    sample_rate: int = 16000, fade: int = FADE,
) -> np.ndarray:
    """``length`` samples of material sounding like ``x[region]``, to be inserted at ``at``
    by :func:`replace_span`.

    A periodic region (a vowel, a voiced hold) is extended pitch-synchronously: a unit of
    whole periods near its centre is repeated, starting in phase with the waveform where
    the material is joined in, so every repeat continues the one before. An aperiodic
    region (frication, a closure) is extended by units taken at seeded random offsets, so
    the noise never repeats audibly. Units are joined with :func:`join`.
    """
    lo, hi = region
    source = x[lo:hi].astype(np.float32)
    period = _period(source, sample_rate)
    unit = max(int(UNIT_SECONDS * sample_rate), 2 * fade)
    if period is not None:
        # Consecutive units overlap by ``fade``, so each advances ``unit - fade``: a whole
        # number of periods keeps every repeat in phase with the one before.
        whole = min(round(unit / period), (len(source) - fade) // period)
        unit = max(1, whole) * period + fade
    if unit > len(source):
        raise ValueError(f"region of {len(source)} samples is shorter than one {unit}-sample unit")
    rng = np.random.default_rng(seed)
    centre = (len(source) - unit) // 2
    if period is not None:
        # The first join crossfades the material against x from ``at - fade // 2``.
        centre -= (lo + centre - (at - fade // 2)) % period
        centre += period if centre < 0 else 0
    out = source[centre : centre + unit]
    while len(out) < length:
        offset = centre if period is not None else int(rng.integers(0, len(source) - unit + 1))
        out = join(out, source[offset : offset + unit], fade)
    return out[:length]
