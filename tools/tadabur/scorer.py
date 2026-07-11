"""The ``.balanced`` scorer gate — the Tadabur training-data filter (ADR-0001).

Ties together the three ported pieces — ``normalization`` (group collapse +
tajweed folds), ``smith_waterman`` (affine-gap local alignment), and
``phoneme_sifat`` (graduated articulatory substitution cost) — into the gate the
filter uses to decide whether a decoded clip matches its reference ayah well
enough to keep as a training example.

Per ADR-0001 the gate is Muraja's ``.balanced`` behaviour **verbatim**: normalize
both sides, run Smith-Waterman, and compare the alignment score against the query
phoneme count. We deliberately do **not** invent a bespoke matching score — no
``max(len(query), len(ref))`` denominator and no hard-coded ``0.70`` threshold
(those correspond to none of Muraja's tuned modes and break fixture parity).
``match_ratio = score / query_phoneme_count`` and the pass bar is the balanced
mode's ``correct_threshold`` (0.65), exactly as in Muraja's transcription check.

Note that Muraja's ``.balanced`` ``shaddahSuppression`` is a **word-scoring**
refinement (it excludes shadda-expansion gaps from the phoneme gate's gap count in
``QuranFollowAlong+WordScoring.swift``), *not* a normalization-time collapse:
``normalize_phonemes`` faithfully keeps shadda expansion doubled. This coarse
score-only filter gate does not reconstruct that per-word gap accounting, so the
flag is carried on ``ScoringParameters`` for fidelity but has no effect here.

The gate layers **one** Tadabur-only reject on top of the Muraja-faithful score: a
decode with a long interior *insertion* run (:data:`MAX_INSERTION_RUN`) fails as
repeated-phrase poison. Muraja's local aligner is built to *follow* a recitation on a
page, so it shrugs off inserted material; when the ayah is already known that leniency
lets a repeated phrase through, which this policy catches. It does not touch
``match_ratio`` or the parity-locked Smith-Waterman constants.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import contrast_attribution, phoneme_sifat
from .normalization import normalize_phonemes
from .smith_waterman import longest_insertion_run, smith_waterman

# Minimum non-space query phonemes before a match is trusted. Below this, short
# transcriptions produce spurious alignments. Verbatim from Muraja's
# checkTranscription guard.
MIN_QUERY_PHONEMES = 3

# Tadabur poison policy (NOT a Muraja parameter): a decode whose best local alignment
# contains an interior run of this many or more consecutive query-only phonemes
# (an insertion the reference does not contain) fails the gate outright, regardless of
# ``match_ratio``. Madd elongations are collapsed before alignment, so a run this long is
# a repeated word/phrase — a mislabelled ("poison") training example. This is a
# filter-side reject layered on top of the Muraja-faithful ``match_ratio``; it does not
# alter the score itself (the parity-locked Smith-Waterman constants are untouched).
MAX_INSERTION_RUN = 5


@dataclass(frozen=True)
class ScoringParameters:
    """The balanced-mode knobs the scorer consumes, verbatim from Muraja.

    ``correct_threshold`` is the pass bar for ``match_ratio``. ``soft_pairs_enabled``
    lets commonly-confused consonant pairs (``ذ↔ز`` …) count as soft rather than
    hard mismatches (see ``is_soft_mismatch``). ``shaddah_suppression`` mirrors
    Muraja's ``.balanced`` flag; there it excludes shadda-expansion gaps from the
    per-word phoneme-gate gap count (``QuranFollowAlong+WordScoring.swift``), a
    word-scoring refinement this coarse score-only gate does not reconstruct — it
    is retained here only for parameter fidelity with Muraja.
    """

    correct_threshold: float
    soft_pairs_enabled: bool
    shaddah_suppression: bool


# Muraja's ``ScoringParameters.balanced`` values: strict's correct_threshold 0.75
# relaxed to 0.65, soft pairs + shaddah suppression enabled (see FollowAlongTypes.swift).
BALANCED = ScoringParameters(
    correct_threshold=0.65,
    soft_pairs_enabled=True,
    shaddah_suppression=True,
)


@dataclass(frozen=True)
class GateResult:
    """Outcome of the scorer gate for one (predicted, reference) pair.

    ``match_ratio`` is the Muraja-faithful ``.balanced`` score (unaffected by the
    insertion-poison policy). ``max_insertion_run`` is the longest interior run of
    query-only phonemes in the alignment; ``passed`` is ``False`` when it reaches
    :data:`MAX_INSERTION_RUN` even if ``match_ratio`` clears the threshold.
    """

    passed: bool
    match_ratio: float
    max_insertion_run: int = 0


@dataclass(frozen=True)
class Scorer:
    """A phoneme scorer bound to a set of ``ScoringParameters``."""

    params: ScoringParameters

    def is_soft_mismatch(self, a: str, b: str) -> bool:
        """Whether two consonants form a soft mismatch under this scorer's mode."""
        return phoneme_sifat.is_soft_mismatch(a, b, self.params.soft_pairs_enabled)

    def attribute(self, predicted: str, reference: str) -> tuple[str, ...]:
        """The contrasts (soft pairs + shadda) present in this pair's alignment.

        Observational companion to :meth:`gate`: it re-runs the same
        normalization and Smith-Waterman purely to label a passer for the P3.5
        poison audit (#6), respecting this scorer's ``soft_pairs_enabled`` mode.
        It does not affect ``passed``/``match_ratio``. Returns a deterministic,
        codepoint-sorted tuple of contrast labels.
        """
        return contrast_attribution.attribute_contrasts(
            predicted, reference, self.params.soft_pairs_enabled
        )

    def gate(self, predicted: str, reference: str) -> GateResult:
        """Score a decoded ``predicted`` phoneme string against ``reference``.

        ``predicted`` is the model's raw decode and is normalized here;
        ``reference`` must already be normalized (the ``build_reference_phonemes``
        / cache form). Normalization is a Swift-faithful port that is **not
        idempotent** — it relies on combining marks to keep shadda gemination
        doubled while collapsing madd runs, and it strips those marks — so
        re-normalizing an already-normalized reference would wrongly collapse its
        shadda (``للاه`` → ``لاه``). Both strings are aligned with Smith-Waterman
        and scored as ``score / max(query_phoneme_count, 1)``. The pair passes
        when that ratio clears ``params.correct_threshold`` **and** the alignment
        has no interior insertion run of :data:`MAX_INSERTION_RUN` phonemes (a
        repeated-phrase poison reject that leaves ``match_ratio`` itself untouched).
        A query with fewer than ``MIN_QUERY_PHONEMES`` non-space phonemes, or no
        positive-scoring alignment, fails with ratio 0.0.
        """
        query = normalize_phonemes(predicted).normalized
        ref = reference
        query_phoneme_count = sum(1 for ch in query if ch != " ")
        if query_phoneme_count < MIN_QUERY_PHONEMES or not ref:
            return GateResult(passed=False, match_ratio=0.0)

        alignment = smith_waterman(query=query, reference=ref)
        if alignment.score <= 0:
            return GateResult(passed=False, match_ratio=0.0)

        match_ratio = alignment.score / max(query_phoneme_count, 1)
        insertion_run = longest_insertion_run(alignment.columns)
        return GateResult(
            passed=(
                match_ratio >= self.params.correct_threshold
                and insertion_run < MAX_INSERTION_RUN
            ),
            match_ratio=match_ratio,
            max_insertion_run=insertion_run,
        )


BALANCED_SCORER = Scorer(BALANCED)
