"""Excising the repeat — the differential's control clip, and the re-gate that vets it.

Muraja ADR-0016 decision 4's second oracle needs two clips per re-read: the recitation as
it was recited, and the same recitation with the repeated span **removed**. Both must
drive the engine to the same terminal state; where they do not, the divergence is
commit-and-trim's, not the reciter's. This module cuts the second one.

Where the cuts land is settled by :mod:`tadabur.seam`, which found the repeat in phoneme
space and measured what the VAD has to say at each edge of it. What that measurement
found is the reason this module looks the way it does: over the 50 seams of the shard-20
re-read pile, **no seam has a usable pause at both of its edges** (42% at the opening
edge, 8% at the closing one — ``docs/tadabur-excision-yield.md``). A pause-bounded
excision is therefore not on the table, and the ADR already said it need not be:

    They are deliberately *not* required to be pause-anchored ... Safety does not rest on
    the cut being pause-anchored, because every excised clip is re-run through the
    ``.balanced`` gate; if it does not come back clean the pair is discarded. A bad cut
    therefore becomes **yield loss, never a false finding**.

So a cut is placed at the CTC onset bounding the repeat, moved onto a VAD silence at
whichever edge happens to have one, and then **disbelieved** until the re-gate says
otherwise (:func:`validate_excision`). The bar is the pair's whole claim: the repeat is
gone (``max_insertion_run`` below the gate's own reject bar) and what is left is clean
recitation (``match_ratio`` at the strict threshold). A cut that ate a real phoneme fails
the first or the second and the pair is discarded — counted, never quietly dropped.

Every qualifying run is cut, not just the longest. A reciter can double back twice, and a
clip with a second repeat left in it would fail the re-gate on that repeat alone; cutting
one and keeping the pair would be asserting over a clip that still contains the thing the
control was supposed to remove.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scorer import MAX_INSERTION_RUN, STRICT, GateResult
from .seam import BoundaryPause, SeamCoverage

# Floor on what an excision may leave behind, matching
# :data:`tadabur.bleed_recut.MIN_RETAINED_S` and defensive for the same reason: a clip
# cut down this far is not a control, it is a fragment, and handing it to the gate would
# earn an answer about nothing. A real clip cannot reach it — the covered span sits
# between the cuts by construction — so tripping this means the geometry was wrong.
MIN_RETAINED_S = 0.5

# The re-gate bar (ADR-0016 decision 4, verbatim: ``match_ratio >= 0.75``,
# ``insertion_run < 5``). The ratio coincides with :data:`tadabur.scorer.STRICT`'s
# ``correct_threshold`` and is read from it rather than restated, because they mean the
# same thing here: the excised clip is being asked to look like an ordinary correct
# recitation, which is exactly what the strict gate's bar describes. Note this is *not*
# the floor #66 dropped from the mining predicate — that one admitted clips to the
# corpus, this one admits a **cut** whose whole purpose was to raise the ratio.
EXCISION_MIN_RATIO = STRICT.correct_threshold

REASON_ACCEPTED = "accepted"
REASON_NO_SEAM = "no_seam"
REASON_DEGENERATE_CUT = "degenerate_cut"
REASON_REPEAT_REMAINS = "repeat_remains"
REASON_RATIO_TOO_LOW = "ratio_too_low"


@dataclass(frozen=True)
class Cut:
    """One span an excision removes, in clip-relative seconds.

    ``pause_anchored_start`` / ``pause_anchored_end`` record whether each edge was moved
    onto a VAD silence or left on the CTC onset — the per-edge outcome
    :mod:`tadabur.seam` measured, carried through to the artifact so a yield figure can
    be split by it rather than argued about.
    """

    start_s: float
    end_s: float
    query_start: int
    query_end: int
    repeat_phonemes: int
    pause_anchored_start: bool = False
    pause_anchored_end: bool = False

    @property
    def seconds(self) -> float:
        """Audio the cut removes."""
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class ExcisionPlan:
    """Every cut one clip's excision makes, and what survives them.

    ``reason`` is non-empty only when there is nothing to excise or the geometry refused
    — a plan that reached the audio always carries at least one cut.
    """

    duration_s: float
    cuts: tuple[Cut, ...] = ()
    reason: str = REASON_NO_SEAM

    @property
    def removed_s(self) -> float:
        """Total audio the plan removes."""
        return sum(cut.seconds for cut in self.cuts)

    @property
    def retained_s(self) -> float:
        """Audio left after every cut."""
        return self.duration_s - self.removed_s

    @property
    def usable(self) -> bool:
        """Whether the plan has cuts worth making."""
        return bool(self.cuts)


def cut_time(boundary: BoundaryPause, onset_s: float) -> float:
    """Where one cut edge lands: the midpoint of its silence, or the CTC onset.

    Cutting through the middle of a silence rather than at its rim is
    :func:`tadabur.bleed_recut.recut_span`'s choice at an ayah edge, for the same reason
    — it puts the splice as far as possible from speech on either side. With no silence
    there is nothing to prefer, so the onset stands and the re-gate decides.
    """
    if boundary.pause_start_s is None or boundary.pause_end_s is None:
        return onset_s
    return (boundary.pause_start_s + boundary.pause_end_s) / 2.0


def plan_excision(seams: list[SeamCoverage], duration_s: float) -> ExcisionPlan:
    """The cuts that remove every repeat in one clip.

    Cuts are ordered and required to be disjoint and positive-length. An overlap would
    mean two runs the aligner reported as separate share audio, which is not a geometry
    to splice blind — the whole plan is refused rather than half of it applied, since a
    partial excision leaves a repeat in the control clip and the pair would fail the
    re-gate anyway, just later and less legibly.
    """
    if not seams:
        return ExcisionPlan(duration_s=duration_s, reason=REASON_NO_SEAM)

    cuts: list[Cut] = []
    for seam in sorted(seams, key=lambda seam: seam.span_start_s):
        cuts.append(
            Cut(
                start_s=cut_time(seam.start, seam.span_start_s),
                end_s=cut_time(seam.end, seam.span_end_s),
                query_start=seam.query_start,
                query_end=seam.query_end,
                repeat_phonemes=seam.repeat_phonemes,
                pause_anchored_start=seam.start.anchored,
                pause_anchored_end=seam.end.anchored,
            )
        )

    plan = ExcisionPlan(duration_s=duration_s, cuts=tuple(cuts), reason=REASON_ACCEPTED)
    previous_end = 0.0
    for cut in cuts:
        if cut.end_s <= cut.start_s or cut.start_s < previous_end:
            return ExcisionPlan(duration_s=duration_s, reason=REASON_DEGENERATE_CUT)
        previous_end = cut.end_s
    if plan.retained_s < MIN_RETAINED_S or previous_end > duration_s:
        return ExcisionPlan(duration_s=duration_s, reason=REASON_DEGENERATE_CUT)
    return plan


def excise(waveform: np.ndarray, sample_rate: int, plan: ExcisionPlan) -> np.ndarray:
    """The waveform with every planned cut removed, spliced in clip order."""
    kept: list[np.ndarray] = []
    cursor = 0
    for cut in plan.cuts:
        start = max(cursor, int(cut.start_s * sample_rate))
        kept.append(waveform[cursor:start])
        cursor = max(start, int(cut.end_s * sample_rate))
    kept.append(waveform[cursor:])
    return np.concatenate(kept) if kept else waveform


@dataclass(frozen=True)
class ExcisionValidation:
    """The re-gate verdict on an excised clip — the reason a pair is kept or thrown away.

    ``max_insertion_run_after`` is the load-bearing one: the control clip exists to be a
    recitation *without* the repeat, so a run still clearing the gate's reject bar means
    the cut missed. ``match_ratio_after`` catches the opposite failure, a cut that landed
    inside real recitation and took a phoneme with it.
    """

    accepted: bool
    reason: str
    match_ratio_before: float
    match_ratio_after: float
    max_insertion_run_before: int
    max_insertion_run_after: int
    leading_trim_after: int
    trailing_trim_after: int


def validate_excision(before: GateResult, after: GateResult) -> ExcisionValidation:
    """Judge an excision by re-gating the cut audio against the same reference."""
    if after.max_insertion_run >= MAX_INSERTION_RUN:
        reason = REASON_REPEAT_REMAINS
    elif after.match_ratio < EXCISION_MIN_RATIO:
        reason = REASON_RATIO_TOO_LOW
    else:
        reason = REASON_ACCEPTED
    return ExcisionValidation(
        accepted=reason == REASON_ACCEPTED,
        reason=reason,
        match_ratio_before=before.match_ratio,
        match_ratio_after=after.match_ratio,
        max_insertion_run_before=before.max_insertion_run,
        max_insertion_run_after=after.max_insertion_run,
        leading_trim_after=after.leading_trim,
        trailing_trim_after=after.trailing_trim,
    )
