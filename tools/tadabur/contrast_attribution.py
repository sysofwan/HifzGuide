"""Contrast attribution — which soft-pair or shadda contrasts admitted a passer.

Observational only. Given a ``(predicted, reference)`` phoneme pair the
``.balanced`` gate admitted, this reports the *set* of articulatory contrasts
present in its Smith-Waterman alignment — the six balanced soft pairs
(``ذ↔ز, ت↔ط, ض↔ظ, ك↔ق, س↔ص, ح↔ه``) and shadda present↔absent. It does **not**
recompute or affect the gate's ``passed``/``match_ratio`` (ADR-0001); it re-runs
the same ported normalization (``normalize_phonemes``, shadda stays doubled) and
alignment purely to *label* the passer for the P3.5 poison audit (#6).

The soft-pair vocabulary comes from ``phoneme_sifat`` (``soft_pair_contrast``),
never re-derived here. A shadda contrast is a doubled core aligned against a
single core: because normalization keeps shadda expansion doubled, that surfaces
as a gap/insertion column whose core repeats an immediately adjacent exact-match
column of the same core (the reference had ``cc`` and the query only ``c``, or
vice-versa).

Attribution reads the *local* alignment, so it sees the contrasts inside the
matched span — which is where they matter for a passer. A difference falling
entirely at the untrimmed leading/trailing edge of the alignment (e.g. a dropped
final shadda with no following context) is outside that span and is not reported;
this is a labelling heuristic for the human audit, not an exhaustive diff.
"""

from __future__ import annotations

from . import phoneme_sifat
from .normalization import normalize_phonemes
from .smith_waterman import AlignedColumn, smith_waterman

# The shadda present↔absent bucket, alongside the six soft-pair buckets.
SHADDA_CONTRAST = "shadda"

# The marginal ``match_ratio`` band just above threshold is audited too (#6); it
# is not a contrast but shares the worklist/fixture ``contrast`` vocabulary.
MARGINAL_CONTRAST = "marginal"


def all_contrasts() -> tuple[str, ...]:
    """The seven audit buckets: the six soft-pair contrasts (sorted) + shadda."""
    return tuple(sorted(phoneme_sifat.soft_pair_contrasts())) + (SHADDA_CONTRAST,)


def contrast_vocabulary() -> frozenset[str]:
    """Every label a worklist/fixture ``contrast`` field may carry (incl. marginal)."""
    return frozenset(all_contrasts()) | {MARGINAL_CONTRAST}


def _soft_pair_contrasts_in(
    columns: list[AlignedColumn], soft_pairs_enabled: bool
) -> set[str]:
    """Soft-pair substitution contrasts present among the alignment columns."""
    if not soft_pairs_enabled:
        return set()
    found: set[str] = set()
    for col in columns:
        if col.query_char is None or col.ref_char is None:
            continue
        label = phoneme_sifat.soft_pair_contrast(col.query_char, col.ref_char)
        if label is not None:
            found.add(label)
    return found


def _gap_core(col: AlignedColumn) -> str | None:
    """The single core of a one-sided (gap/insertion) column, else ``None``.

    A space is never a shadda core, so it is ignored.
    """
    if col.query_char is None and col.ref_char not in (None, " "):
        return col.ref_char
    if col.ref_char is None and col.query_char not in (None, " "):
        return col.query_char
    return None


def _is_exact_match(col: AlignedColumn, core: str) -> bool:
    return col.query_char == core and col.ref_char == core


def _has_shadda_contrast(columns: list[AlignedColumn]) -> bool:
    """Whether a doubled core is aligned against a single core anywhere.

    A shadda difference is a one-sided column (a dropped or inserted core) whose
    core equals an immediately adjacent exact-match column of the same core —
    i.e. one side had the core twice and the other once.
    """
    for idx, col in enumerate(columns):
        core = _gap_core(col)
        if core is None:
            continue
        neighbors = []
        if idx > 0:
            neighbors.append(columns[idx - 1])
        if idx + 1 < len(columns):
            neighbors.append(columns[idx + 1])
        if any(_is_exact_match(n, core) for n in neighbors):
            return True
    return False


def attribute_contrasts(
    predicted: str, reference: str, soft_pairs_enabled: bool = True
) -> tuple[str, ...]:
    """The sorted set of contrasts present in the ``predicted`` vs ``reference``
    alignment.

    ``predicted`` is the model's raw decode and is normalized here; ``reference``
    must already be normalized (the cache form). Normalization is not idempotent
    — re-normalizing an already-normalized reference would collapse its shadda
    doubling and mis-attribute shadda contrasts — so the reference is used
    verbatim. Aligns the two with the same Smith-Waterman used by the gate, and
    scans the aligned columns for soft-pair substitutions and shadda present↔absent
    differences. ``soft_pairs_enabled`` mirrors the scorer's mode (no soft pairs
    in strict). Returns a deterministic, codepoint-sorted tuple of contrast
    labels — empty when a passer matched cleanly on every contrast position.
    """
    query = normalize_phonemes(predicted).normalized
    ref = reference
    columns = smith_waterman(query=query, reference=ref).columns

    contrasts = _soft_pair_contrasts_in(columns, soft_pairs_enabled)
    if _has_shadda_contrast(columns):
        contrasts.add(SHADDA_CONTRAST)
    return tuple(sorted(contrasts))
