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
"""

from __future__ import annotations

from dataclasses import dataclass

from . import contrast_attribution, phoneme_sifat
from .normalization import normalize_phonemes
from .smith_waterman import smith_waterman

# Minimum non-space query phonemes before a match is trusted. Below this, short
# transcriptions produce spurious alignments. Verbatim from Muraja's
# checkTranscription guard.
MIN_QUERY_PHONEMES = 3


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
    """Outcome of the scorer gate for one (predicted, reference) pair."""

    passed: bool
    match_ratio: float


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
        when that ratio clears ``params.correct_threshold``. A query with fewer
        than ``MIN_QUERY_PHONEMES`` non-space phonemes, or no positive-scoring
        alignment, fails with ratio 0.0.
        """
        query = normalize_phonemes(predicted).normalized
        ref = reference
        query_phoneme_count = sum(1 for ch in query if ch != " ")
        if query_phoneme_count < MIN_QUERY_PHONEMES or not ref:
            return GateResult(passed=False, match_ratio=0.0)

        score = smith_waterman(query=query, reference=ref).score
        if score <= 0:
            return GateResult(passed=False, match_ratio=0.0)

        match_ratio = score / max(query_phoneme_count, 1)
        return GateResult(
            passed=match_ratio >= self.params.correct_threshold,
            match_ratio=match_ratio,
        )


BALANCED_SCORER = Scorer(BALANCED)
