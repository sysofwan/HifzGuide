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

from collections.abc import Iterable
from dataclasses import dataclass

from . import phoneme_sifat
from .normalization import (
    PhonemeNormalization,
    _folded_core,
    _grapheme_clusters,
    cluster_offsets,
    normalize_phonemes,
)
from .smith_waterman import AlignedColumn, AlignmentResult, smith_waterman

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


@dataclass(frozen=True)
class ShaddaEvents:
    """Directional gemination-mismatch occurrences in one alignment.

    ``added`` counts *query-only* (insertion) cores the decode doubled that the
    reference has singly ("non-shadda made shadda"); ``dropped`` counts *reference-
    only* (gap) cores the decode omitted that the reference geminates ("omit when
    unsure"). Each is a one-sided column whose core equals an immediately adjacent
    exact-match column of the same core. The two directions are audited separately
    because the P3.5 audit (#6) found them asymmetric — added is 86% genuinely-wrong
    recitation, dropped 26% (ADR-0003) — and the eval's confusion matrix (#7) reports
    both to show whether that discrimination survives fine-tuning.
    """

    added: int
    dropped: int


def shadda_event_columns(columns: list[AlignedColumn]) -> list[tuple[int, int]]:
    """Every gemination-mismatch column, as ``(column index, matched neighbour index)``.

    A shadda difference is a one-sided column (a dropped or inserted core) whose
    core equals an immediately adjacent exact-match column of the same core — i.e.
    one side had the core twice and the other once. The neighbour is that exact-match
    column (the preceding one when both qualify). A query-only such column is an
    *added* gemination, a reference-only one a *dropped* gemination.
    """
    events: list[tuple[int, int]] = []
    for idx, col in enumerate(columns):
        core = _gap_core(col)
        if core is None:
            continue
        for neighbor in (idx - 1, idx + 1):
            if 0 <= neighbor < len(columns) and _is_exact_match(columns[neighbor], core):
                events.append((idx, neighbor))
                break
    return events


def shadda_events(columns: list[AlignedColumn]) -> ShaddaEvents:
    """Count added vs dropped gemination occurrences across ``columns``
    (:func:`shadda_event_columns`)."""
    events = shadda_event_columns(columns)
    added = sum(columns[idx].ref_char is None for idx, _ in events)
    return ShaddaEvents(added=added, dropped=len(events) - added)


def _has_shadda_contrast(columns: list[AlignedColumn]) -> bool:
    """Whether a doubled core is aligned against a single core anywhere (either
    direction) — the present↔absent shadda difference the P3.5 audit samples on."""
    events = shadda_events(columns)
    return events.added > 0 or events.dropped > 0


def has_added_shadda(columns: list[AlignedColumn]) -> bool:
    """Whether the *predicted* side carries a gemination the reference lacks.

    The directional half of the shadda present↔absent difference — a decode that
    doubled a consonant the reference has singly. This is the reject-worthy
    direction: in the P3.5 poison audit (#6) *added* shadda was 86% genuinely-wrong
    recitations (vs 26% for the *dropped* direction, the model's benign "omit when
    unsure" behaviour, ADR-0003), and since shadda is not a trainable phoneme-head
    class, admitting extra gemination has no training value. The mirror *dropped*
    direction is intentionally not rejected — it is kept, so the filter's shadda
    tolerance is asymmetric.
    """
    return shadda_events(columns).added > 0


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


#: A shadda site's change: the decode has a single consonant where the reference geminates,
#: or doubles one the reference has singly.
DROPPED = "dropped"
ADDED = "added"


@dataclass(frozen=True)
class ContrastSite:
    """One occurrence of a contrast, placed on the **raw** (tashkeel-bearing) reference.

    ``reference_index`` is the carrier consonant in the raw reference: for a soft pair the
    substituted letter, for a dropped shadda the first of the doubled pair, for an added
    one the single consonant the decode doubled. ``change`` is the decode's letter for a
    soft pair, and :data:`DROPPED` / :data:`ADDED` for shadda.
    """

    reference_index: int
    change: str


@dataclass(frozen=True)
class CarrierAlignment:
    """The normalized alignment of a decode against a raw reference, with every column's
    position on each side, so a column can be carried back to its raw-reference carrier.

    Built by :func:`align_on_carriers`. :func:`contrast_sites` and the truth-site scorer
    (:mod:`training.site_outcomes`) read one decode through the same object.
    """

    predicted: str
    raw_reference: str
    reference: PhonemeNormalization
    decode: PhonemeNormalization
    alignment: AlignmentResult
    #: The normalized reference / decode position each column consumes (``None``: a gap).
    positions: list[int | None]
    query_positions: list[int | None]
    cluster_starts: list[int]

    def carrier(self, ref_position: int) -> int:
        """The raw-reference character index of a normalized reference position: the
        offset of its group's first grapheme cluster, which starts with the core consonant."""
        return self.cluster_starts[self.reference.offset_map[ref_position][0]]

    def consonants(self) -> dict[int, str | None]:
        """The decode character aligned to each raw-reference carrier in the local alignment
        (:func:`aligned_consonants`)."""
        return {
            self.carrier(position): column.query_char
            for column, position in zip(self.alignment.columns, self.positions)
            if position is not None
        }

    def decoded_run(self, raw_indices: Iterable[int]) -> int:
        """How many consonants the decode holds where the reference has the run ``raw_indices``.

        ``raw_indices`` are the raw-reference indices of one run of a consonant (a geminate
        ``دد``, or a single ``د``). The decode's groups aligned to the run's normalized
        groups count, together with the decode's neighbouring groups of the same core that
        no reference character is aligned to: an inserted copy, or one the local alignment
        trimmed off its edge. Each group counts the raw consonants it stands for
        (:func:`_raw_cores`), so a bare ``دد`` that normalization merged counts two. 0 means
        the run is not aligned to that consonant at all.
        """
        wanted = set(raw_indices)
        run = {p for p in range(len(self.reference.normalized)) if self.carrier(p) in wanted}
        if not run:
            return 0
        core = self.reference.normalized[min(run)]
        consumed = {q for q, p in zip(self.query_positions, self.positions) if p is not None}
        held = {
            q
            for q, p in zip(self.query_positions, self.positions)
            if p in run and q is not None and self.decode.normalized[q] == core
        }
        if not held:
            return 0
        for step, edge in ((-1, min(held)), (1, max(held))):
            q = edge + step
            while 0 <= q < len(self.decode.normalized) and q not in consumed and (
                self.decode.normalized[q] == core
            ):
                held.add(q)
                q += step
        return sum(_raw_cores(self.predicted, self.decode, q) for q in held)


def align_on_carriers(predicted: str, raw_reference: str) -> CarrierAlignment:
    """Align as :func:`attribute_contrasts` does: both sides normalized, Smith-Waterman."""
    reference = normalize_phonemes(raw_reference)
    decode = normalize_phonemes(predicted)
    alignment = smith_waterman(query=decode.normalized, reference=reference.normalized)
    positions: list[int | None] = []
    query_positions: list[int | None] = []
    cursor, query_cursor = alignment.ref_start, alignment.query_start
    for column in alignment.columns:
        positions.append(None if column.ref_char is None else cursor)
        query_positions.append(None if column.query_char is None else query_cursor)
        cursor += column.ref_char is not None
        query_cursor += column.query_char is not None
    return CarrierAlignment(
        predicted, raw_reference, reference, decode, alignment, positions, query_positions,
        cluster_offsets(raw_reference),
    )


def aligned_consonants(predicted: str, raw_reference: str) -> dict[int, str | None]:
    """The decode character aligned to each raw-reference carrier inside the local alignment.

    This is the consonant-commitment rule the truth-site scorer reads at a pair site: the
    alignment :func:`contrast_sites` locates contrasts with, so the letter it reports at a
    carrier is the letter a contrast there would name. A value is the decode's normalized
    character (ghunna variants folded onto their base), or ``None`` where the decode has a
    gap. A carrier outside the local alignment span is absent.
    """
    return align_on_carriers(predicted, raw_reference).consonants()


def contrast_sites(predicted: str, raw_reference: str, contrast: str) -> list[ContrastSite]:
    """Where ``contrast`` (a pair label ``a↔b`` or :data:`SHADDA_CONTRAST`) occurs, by carrier.

    A pair label need not be one of the six soft pairs: ``ذ↔ظ`` is located the same way.

    The decode is aligned exactly as :func:`attribute_contrasts` aligns it (both sides
    normalized, Smith-Waterman), so a contrast this reports is one the attribution
    reports. Each contrast column is then carried back to the raw reference: a normalized
    character is a group whose first grapheme cluster starts with the core consonant, so
    the cluster's character offset is the carrier. A geminate whose two halves were both
    substituted is one site, on its first half. Sites are in reference order.

    A gemination mismatch is then **checked on the raw strings**, because normalization
    merges a run of *bare* same-core clusters into one character: a decode that keeps
    both consonants of a geminate but drops the haraka after it (``رَببسَ`` for
    ``رَببِسَ``) normalizes to a single ``ب`` and reads as a dropped shaddah. A dropped
    site is kept only if the decode's group aligned to the geminate holds one consonant
    of that core, an added site only if the reference's group holds one. The attribution
    and the scorer gate keep the normalized semantics; only these sites are filtered.
    """
    aligned = align_on_carriers(predicted, raw_reference)
    columns, positions = aligned.alignment.columns, aligned.positions

    if contrast == SHADDA_CONTRAST:
        sites = []
        for idx, neighbor in shadda_event_columns(columns):
            position = positions[idx]
            if position is None:  # the decode doubled the neighbour's single consonant
                if _raw_cores(raw_reference, aligned.reference, positions[neighbor]) == 1:
                    sites.append(ContrastSite(aligned.carrier(positions[neighbor]), ADDED))
            elif _raw_cores(predicted, aligned.decode, aligned.query_positions[neighbor]) == 1:
                # the reference's doubled pair is this column and its neighbour
                carrier = aligned.carrier(min(position, positions[neighbor]))
                sites.append(ContrastSite(carrier, DROPPED))
        return sorted(set(sites), key=lambda site: site.reference_index)

    pair = frozenset(contrast.split("\u2194"))
    found: dict[int, ContrastSite] = {}
    for column, position in zip(columns, positions):
        if (
            position is not None
            and column.query_char != column.ref_char
            and frozenset((column.query_char, column.ref_char)) == pair
        ):
            index = aligned.carrier(position)
            found[index] = ContrastSite(index, column.query_char)
    return [
        site
        for index, site in sorted(found.items())
        if not (index - 1 in found and raw_reference[index - 1] == raw_reference[index])
    ]


def _raw_cores(text: str, normalization: PhonemeNormalization, position: int) -> int:
    """How many consonants of its core the normalized character at ``position`` stands
    for in ``text``: 2 for a bare doubled consonant that normalization merged."""
    start, end = normalization.offset_map[position]
    core = normalization.normalized[position]
    return sum(
        _folded_core(cluster[0]) == core for cluster in _grapheme_clusters(text)[start:end]
    )
