"""Sample-exact waveform edits for synthetic edits and their decoys (#88).

Three operations, pure numpy, each changing the waveform only around the span it is given
and reporting how it did so (:class:`Rendered`):

* :func:`crop` removes ``length`` samples at ``start``;
* :func:`stretch` inserts ``length`` samples at ``at`` that sound like a region of the clip;
* :func:`splice` replaces ``[start, end)`` with material from elsewhere.

Every cut is joined with a Hann crossfade of :data:`FADE` samples centred on it, so no click
is introduced. Outside :attr:`Rendered.changed` (widened by half a crossfade) the output is
the input, sample for sample, before the span and shifted by the length change after it.

**Periodic regions keep their phase at every join.** Removing or inserting an arbitrary
number of samples in a vowel or a voiced hold joins two points of the cycle that do not
match, and the crossfade between them cancels part of the signal: an extension of 2,880
samples at 125 Hz leaves about a third of the energy at the join. So where the region is
periodic (:func:`period`), the length change is made of whole pitch periods, which join in
phase, and the remainder (at most half a period) is absorbed by resampling the
:data:`RETIME_PERIODS` periods that follow, a local pitch change of at most 4%. Aperiodic
regions (frication, a closure) have no phase to keep; they are cut or extended directly.
Which way a region is treated depends on the signal only, never on whether the change is
an edit or a decoy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Crossfade length at every join: 10 ms at 16 kHz. Even, so it centres on a cut.
FADE = 160

#: Pitch range searched when deciding whether a region is voiced (:func:`period`).
MIN_PITCH_HZ = 70
MAX_PITCH_HZ = 400
#: Normalized autocorrelation above which a region is treated as periodic.
VOICED_CORRELATION = 0.5
#: Target length of one repeated unit: about two pitch periods of a typical voice.
UNIT_SECONDS = 0.03
#: Periods resampled to absorb a periodic change's remainder (at most half a period), so
#: the local pitch moves by at most 1 / (2 * RETIME_PERIODS).
RETIME_PERIODS = 12

PERIODIC = "periodic"
APERIODIC = "aperiodic"
SPLICED = "splice"


@dataclass(frozen=True)
class Rendered:
    """An edited waveform, how it was made, and the source span ``[lo, hi)`` it replaced
    (``lo == hi`` for a pure insertion)."""

    samples: np.ndarray
    path: str
    period: int | None
    changed: tuple[int, int]


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


def _cut(x: np.ndarray, start: int, length: int, fade: int = FADE) -> np.ndarray:
    """``x`` without ``[start, start + length)``, the cut crossfaded."""
    half = fade // 2
    if start - half < 0 or start + length + half > len(x) or length < 1:
        raise ValueError(f"cut [{start}, {start + length}) does not fit in {len(x)} samples")
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


def _retime(x: np.ndarray, start: int, span: int, length: int) -> np.ndarray:
    """``x`` with ``[start, start + span)`` resampled to ``length`` samples. Both ends keep
    their samples, so the waveform stays continuous without a crossfade."""
    if start + span + 1 > len(x) or length < 1:
        raise ValueError(f"cannot retime [{start}, {start + span}) of {len(x)} samples")
    segment = x[start : start + span + 1].astype(np.float64)
    positions = np.linspace(0.0, span, length + 1)[:-1]
    resampled = np.interp(positions, np.arange(span + 1), segment)
    return np.concatenate([x[:start], resampled, x[start + span:]]).astype(np.float32)


def period(region: np.ndarray, sample_rate: int = 16000) -> int | None:
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


def _whole_periods(length: int, p: int) -> tuple[int, int]:
    """``length`` as whole periods plus a remainder of at most half a period."""
    count = round(length / p)
    return count, length - count * p


def crop(x: np.ndarray, start: int, length: int) -> Rendered:
    """``x`` with ``length`` samples removed at ``start``: ``len(x) - length`` samples.

    A periodic span loses whole periods at ``start`` and the remainder from the
    :data:`RETIME_PERIODS` periods after it (module docstring)."""
    p = period(x[start : start + length])
    if p is None:
        return Rendered(_cut(x, start, length), APERIODIC, None, (start, start + length))
    count, remainder = _whole_periods(length, p)
    span = RETIME_PERIODS * p
    y = _cut(x, start, count * p) if count else x
    return Rendered(_retime(y, start, span, span - remainder), PERIODIC, p,
                    (start, start + count * p + span))


def _periodic_material(
    source: np.ndarray, lo: int, p: int, length: int, at: int, fade: int
) -> np.ndarray:
    """``length`` samples of whole periods of ``source`` (which starts at ``lo`` in the
    clip), starting in phase with the clip at ``at - fade // 2``, where the first join
    crossfades it in."""
    unit = max(1, min(round(UNIT_SECONDS * 16000 / p), (len(source) - fade) // p)) * p + fade
    if unit > len(source):
        raise ValueError(f"region of {len(source)} samples is shorter than one {unit}-sample unit")
    centre = (len(source) - unit) // 2
    centre -= (lo + centre - (at - fade // 2)) % p
    centre += p if centre < 0 else 0
    out = source[centre : centre + unit]
    while len(out) < length:
        out = join(out, source[centre : centre + unit], fade)
    return out[:length]


def _noise_material(source: np.ndarray, length: int, seed: int, fade: int) -> np.ndarray:
    """``length`` samples of ``source``'s texture: units at seeded random offsets."""
    unit = max(int(UNIT_SECONDS * 16000), 2 * fade)
    if unit > len(source):
        raise ValueError(f"region of {len(source)} samples is shorter than one {unit}-sample unit")
    rng = np.random.default_rng(seed)
    out = source[(len(source) - unit) // 2 :][:unit]
    while len(out) < length:
        offset = int(rng.integers(0, len(source) - unit + 1))
        out = join(out, source[offset : offset + unit], fade)
    return out[:length]


def stretch(
    x: np.ndarray, at: int, length: int, region: tuple[int, int], seed: int, fade: int = FADE
) -> Rendered:
    """``x`` with ``length`` samples sounding like ``x[region]`` inserted at ``at``.

    A periodic region is extended by whole periods, in phase with the clip at both joins,
    and the remainder is absorbed by resampling the :data:`RETIME_PERIODS` periods after
    ``at``. An aperiodic one is extended by units at seeded random offsets."""
    lo, hi = region
    source = x[lo:hi].astype(np.float32)
    p = period(source)
    if p is None:
        material = _noise_material(source, length + fade, seed, fade)
        return Rendered(replace_span(x, at, at, material, fade), APERIODIC, None, (at, at))
    count, remainder = _whole_periods(length, p)
    span = RETIME_PERIODS * p
    y = x
    if count:
        material = _periodic_material(source, lo, p, count * p + fade, at, fade)
        y = replace_span(x, at, at, material, fade)
    return Rendered(_retime(y, at + count * p, span, span + remainder), PERIODIC, p,
                    (at, at + span))


def splice(x: np.ndarray, start: int, end: int, material: np.ndarray) -> Rendered:
    """``x`` with ``[start, end)`` replaced by ``material`` (with its join context)."""
    return Rendered(replace_span(x, start, end, material), SPLICED, None, (start, end))
