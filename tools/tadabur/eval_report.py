"""Two-sided, targeted eval (ADR-0001) — the pure scoring core, torch-free.

ADR-0001 forbids aggregate PER/CER as the fine-tune metric: it can *improve* while
the very distinction we want to keep — the model's ability to tell a soft-pair
consonant from its confusable partner — collapses. Instead the eval is two-sided and
targeted, and this module owns its computation, decoupled from the GPU model pass
(:mod:`tadabur.eval_harness`) so it is unit-testable on hand-built decodes:

* a **per-phoneme confusion matrix** over the six soft pairs (aligned-column level)
  plus **shadda** (added/dropped gemination occurrences), so base-vs-fine-tuned can be
  compared directly (``compare_confusion`` / the harness diff);
* **should-accept recall** — the fraction of the curated acceptable-imperfect clips
  the model would ADMIT under Muraja's ``.strict`` mode (a false-negative is a
  ``.strict`` rejection of acceptable recitation, so recall *is* one-minus-the-false-
  negative-rate); and
* **should-reject discrimination** — the fraction of the curated genuinely-wrong clips
  ``.strict`` still REJECTS (the distinction is *retained*, not collapsed).

Accept/reject models Muraja's ``.strict`` mode (see :func:`strict_accept`): a decode
with any soft-pair substitution against its reference is a hard mismatch that fails —
that is precisely the tolerance ``.strict`` removes and ``.balanced`` keeps — and
otherwise the ported ``match_ratio`` must clear the ``.strict`` threshold. In this
score-only port ``match_ratio`` itself is mode-independent, so the recall/discrimination
shift between two checkpoints comes from the *decode* changing (the model learning to
emit the correct consonant, or collapsing onto it) — which is exactly what fine-tuning
moves. The Tadabur filter-side poison rejects (insertion-run, added-shadda) are **not**
applied here: they are training-data hygiene, not Muraja ``.strict`` behaviour, and the
success criterion is about Muraja.

The success criterion is recorded verbatim on every report (:data:`SUCCESS_CRITERION`)
so the base-model baseline this slice ships states, in Muraja's own vocabulary, what a
later fine-tuned model must beat.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from . import phoneme_sifat
from .contrast_attribution import shadda_events
from .eval_fixtures import ACCEPT, REJECT
from .normalization import normalize_phonemes
from .scorer import STRICT, STRICT_SCORER
from .smith_waterman import AlignedColumn, smith_waterman

# The eval scores against Muraja's ``.strict`` pass bar — the mode ADR-0001 wants the
# fine-tuned model to let Muraja default to.
STRICT_THRESHOLD = STRICT.correct_threshold

# The go/no-go criterion, in Muraja's own vocabulary (ADR-0001), recorded on every
# report so the baseline states what a fine-tuned model must beat.
SUCCESS_CRITERION = (
    "The fine-tuned model lets Muraja default to .strict (correct_threshold "
    f"{STRICT_THRESHOLD}, soft pairs off) WITHOUT raising false-negatives on "
    "acceptable recitation: should-accept recall rises vs the base model while "
    "should-reject discrimination is retained (not collapsed)."
)

# Predicted-symbol bucket for a soft-pair reference phoneme rendered as neither itself
# nor its confusable partner (a distant substitution or a dropped/gap column).
OTHER = "other"


@dataclass(frozen=True)
class ClipDecode:
    """One fixture clip decoded by the model under evaluation.

    ``verdict`` is :data:`tadabur.eval_fixtures.ACCEPT` or ``REJECT`` (the fixture set
    it came from). ``contrast`` is its audit bucket (a soft pair, ``shadda``, or
    ``marginal``). ``predicted`` is the model's raw decode (normalized here, as the
    gate does); ``reference`` is the clip's realized, already-normalized reference
    phoneme string (the segment manifest's cache form), used verbatim.
    """

    clip_id: str
    contrast: str
    verdict: str
    predicted: str
    reference: str


@dataclass(frozen=True)
class SideMetrics:
    """Accept/reject outcomes for one fixture set (overall or one contrast).

    ``accepted`` is how many of ``total`` clips clear the ``.strict`` gate. On the
    should-accept side the headline metric is :attr:`recall` (admitting acceptable
    recitation); on the should-reject side it is :attr:`discrimination` (still
    rejecting genuinely-wrong recitation). Both are ``None`` when ``total`` is 0.
    """

    total: int
    accepted: int

    @property
    def rejected(self) -> int:
        return self.total - self.accepted

    @property
    def recall(self) -> float | None:
        return self.accepted / self.total if self.total else None

    @property
    def discrimination(self) -> float | None:
        return self.rejected / self.total if self.total else None


@dataclass(frozen=True)
class SoftPairConfusion:
    """Aligned-column confusion for one soft pair, keyed by reference phoneme.

    ``matrix[ref_char][pred_symbol]`` counts alignment columns whose reference
    phoneme is ``ref_char`` (one of the pair's two consonants) and whose model
    rendering is ``pred_symbol`` — the same consonant (correct), its confusable
    partner (the soft substitution the fine-tune should reduce), or :data:`OTHER`.
    A model that has *collapsed* the pair shows both consonants rendered as one.
    """

    contrast: str
    matrix: dict[str, dict[str, int]]


@dataclass(frozen=True)
class ShaddaConfusion:
    """Gemination-mismatch occurrences across the whole eval set.

    ``added`` (decode doubled a single consonant) is the reject-worthy direction;
    ``dropped`` (decode omitted a geminated consonant) is the benign one (ADR-0003).
    Counted at occurrence level via :func:`tadabur.contrast_attribution.shadda_events`.
    """

    added: int
    dropped: int


@dataclass(frozen=True)
class ClipOutcome:
    """Whether one fixture clip cleared the ``.strict`` gate.

    The aggregate ``SideMetrics`` counts are enough to report one checkpoint, but not to
    *compare* two: the same clips are scored by every rung, so a rung-vs-rung difference
    is a paired observation and needs McNemar's test over the discordant clips rather than
    an unpaired test over the totals. Two rungs can post an identical ``accepted`` count
    while disagreeing on several clips in both directions, so the counts alone can hide
    real movement. Emitting the per-clip outcome keeps that test possible from the
    artifacts, without re-decoding.
    """

    clip_id: str
    contrast: str
    verdict: str
    accepted: bool


@dataclass(frozen=True)
class EvalReport:
    """The full two-sided eval of one model checkpoint over the curated fixtures."""

    model_id: str
    strict_threshold: float
    should_accept: SideMetrics
    should_reject: SideMetrics
    per_contrast: dict[str, dict[str, SideMetrics]]
    soft_pair_confusion: list[SoftPairConfusion]
    shadda_confusion: ShaddaConfusion
    clip_outcomes: tuple[ClipOutcome, ...] = ()
    success_criterion: str = SUCCESS_CRITERION

    def to_json_dict(self) -> dict:
        """A deterministic, human-readable JSON mapping of the whole report."""
        return {
            "model_id": self.model_id,
            "strict_threshold": self.strict_threshold,
            "success_criterion": self.success_criterion,
            "should_accept": _side_to_json(self.should_accept, "recall"),
            "should_reject": _side_to_json(self.should_reject, "discrimination"),
            "per_contrast": {
                contrast: {
                    "should_accept": _side_to_json(sides["should_accept"], "recall"),
                    "should_reject": _side_to_json(sides["should_reject"], "discrimination"),
                }
                for contrast, sides in sorted(self.per_contrast.items())
            },
            "clip_outcomes": [
                {
                    "clip_id": outcome.clip_id,
                    "contrast": outcome.contrast,
                    "verdict": outcome.verdict,
                    "accepted": outcome.accepted,
                }
                for outcome in sorted(self.clip_outcomes, key=lambda o: o.clip_id)
            ],
            "confusion_matrix": {
                "soft_pairs": {
                    conf.contrast: {
                        ref_char: dict(sorted(preds.items()))
                        for ref_char, preds in sorted(conf.matrix.items())
                    }
                    for conf in sorted(self.soft_pair_confusion, key=lambda c: c.contrast)
                },
                "shadda": {
                    "added": self.shadda_confusion.added,
                    "dropped": self.shadda_confusion.dropped,
                },
            },
        }


def _side_to_json(side: SideMetrics, headline: str) -> dict:
    value = side.recall if headline == "recall" else side.discrimination
    return {
        "total": side.total,
        "accepted": side.accepted,
        "rejected": side.rejected,
        headline: value,
    }


def _aligned_columns(predicted: str, reference: str) -> list[AlignedColumn]:
    """The Smith-Waterman columns for a decode vs its realized reference.

    Mirrors the gate exactly: ``predicted`` is normalized (the port's normalization
    is not idempotent), ``reference`` is the already-normalized cache form, used
    verbatim. Empty when either side normalizes to blank.
    """
    query = normalize_phonemes(predicted).normalized
    if not query.strip() or not reference.strip():
        return []
    return smith_waterman(query=query, reference=reference).columns


def strict_accept(clip: ClipDecode) -> bool:
    """Whether Muraja's ``.strict`` mode would admit this clip's decode.

    ``.strict``'s defining difference from ``.balanced`` is that soft pairs are OFF —
    a confusable-consonant substitution the ``.balanced`` scorer tolerates is, under
    ``.strict``, a **hard** mismatch that fails the clip. The score-only port's
    ``match_ratio`` is mode-independent (it always applies the graduated soft-pair
    penalty), so modelling ``.strict`` faithfully means layering that rule back on:
    a decode with any soft-pair substitution against its reference is rejected,
    otherwise it must clear the ``.strict`` ``match_ratio`` bar (:data:`STRICT_THRESHOLD`).
    This is what encodes the success criterion — an acceptable clip the fine-tuned
    model decodes *correctly* (no soft slip) passes ``.strict``; a genuinely-wrong clip
    whose distinct wrong phoneme the model still emits keeps the soft-pair substitution
    and is rejected, so a *collapsed* model (wrong sound decoded as the right phoneme)
    is what would silently start passing. The filter-side poison rejects are not applied
    — they are training-data hygiene, not ``.strict`` behaviour.

    **This models only the alignment half of ``.strict``, so it is a lower bound.** What this
    repo ported from Muraja is the alignment score; the app aligns in normalized space and then
    expands the alignment back to *original* space to compare the harakat that normalization
    stripped, emitting a ``tashkeelError`` word grade, and ``.strict`` additionally does not
    suppress shaddah-expansion gaps in its phoneme gate. Neither is reproduced here, so a decode
    with a wrong vowel or a missing shadda can pass this function while the app would flag it.
    See ADR-0005.
    """
    columns = _aligned_columns(clip.predicted, clip.reference)
    if _has_soft_pair_substitution(columns):
        return False
    return STRICT_SCORER.gate(clip.predicted, clip.reference).match_ratio >= STRICT.correct_threshold


def _has_soft_pair_substitution(columns: list[AlignedColumn]) -> bool:
    """Whether any aligned column substitutes one soft-pair consonant for its partner.

    These are exactly the mismatches ``.balanced`` forgives and ``.strict`` does not,
    so their presence is what turns a balanced pass into a strict rejection.
    """
    return any(
        col.query_char is not None
        and col.ref_char is not None
        and phoneme_sifat.is_soft_mismatch(col.query_char, col.ref_char, soft_pairs_enabled=True)
        for col in columns
    )


def _pred_symbol(query_char: str | None, ref_char: str, partner: str) -> str:
    """Bucket a soft-pair reference column's rendering: itself, partner, or OTHER."""
    if query_char == ref_char:
        return ref_char
    if query_char == partner:
        return partner
    return OTHER


def _soft_pair_confusion(clips: list[ClipDecode]) -> list[SoftPairConfusion]:
    """The per-phoneme confusion matrix over the six soft pairs, over all clips.

    Every clip's alignment columns are scanned once; each column whose reference
    phoneme sits on a soft pair contributes one count to that pair's matrix. Rows
    (reference phonemes) and columns (renderings) are always fully populated so two
    reports are directly comparable cell-for-cell.
    """
    counts: dict[str, Counter] = {c: Counter() for c in phoneme_sifat.soft_pair_contrasts()}
    for clip in clips:
        for col in _aligned_columns(clip.predicted, clip.reference):
            ref_char = col.ref_char
            if ref_char is None:
                continue
            partner = phoneme_sifat.soft_pair_partner(ref_char)
            if partner is None:
                continue
            contrast = phoneme_sifat.soft_pair_contrast(ref_char, partner)
            counts[contrast][(ref_char, _pred_symbol(col.query_char, ref_char, partner))] += 1

    result: list[SoftPairConfusion] = []
    for contrast in sorted(counts):
        members = contrast.split("\u2194")
        matrix = {
            ref_char: {sym: counts[contrast][(ref_char, sym)] for sym in (*members, OTHER)}
            for ref_char in members
        }
        result.append(SoftPairConfusion(contrast=contrast, matrix=matrix))
    return result


def _shadda_confusion(clips: list[ClipDecode]) -> ShaddaConfusion:
    added = dropped = 0
    for clip in clips:
        events = shadda_events(_aligned_columns(clip.predicted, clip.reference))
        added += events.added
        dropped += events.dropped
    return ShaddaConfusion(added=added, dropped=dropped)


def _side(clips: list[ClipDecode], verdict: str) -> SideMetrics:
    side = [c for c in clips if c.verdict == verdict]
    accepted = sum(1 for c in side if strict_accept(c))
    return SideMetrics(total=len(side), accepted=accepted)


def evaluate(clips: list[ClipDecode], model_id: str) -> EvalReport:
    """Score decoded fixture clips into a full two-sided :class:`EvalReport`.

    ``clips`` are the should-accept and should-reject fixtures already decoded by the
    model under evaluation (see :mod:`tadabur.eval_harness`). Overall and per-contrast
    should-accept recall / should-reject discrimination are computed against the
    ``.strict`` gate, alongside the soft-pair + shadda confusion matrix over the whole
    set.     Pure and deterministic: identical ``clips`` yield an identical report.
    """
    contrasts = sorted({c.contrast for c in clips})
    per_contrast = {
        contrast: {
            "should_accept": _side([c for c in clips if c.contrast == contrast], ACCEPT),
            "should_reject": _side([c for c in clips if c.contrast == contrast], REJECT),
        }
        for contrast in contrasts
    }
    return EvalReport(
        model_id=model_id,
        strict_threshold=STRICT.correct_threshold,
        should_accept=_side(clips, ACCEPT),
        should_reject=_side(clips, REJECT),
        per_contrast=per_contrast,
        soft_pair_confusion=_soft_pair_confusion(clips),
        shadda_confusion=_shadda_confusion(clips),
        clip_outcomes=tuple(
            ClipOutcome(
                clip_id=clip.clip_id,
                contrast=clip.contrast,
                verdict=clip.verdict,
                accepted=strict_accept(clip),
            )
            for clip in clips
        ),
    )
