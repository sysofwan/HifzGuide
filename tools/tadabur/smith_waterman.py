"""Smith-Waterman local alignment — Python port of ``SmithWatermanAlignment.swift``.

Aligns a decoded phoneme string (``query``) against an ayah's reference phoneme
string (``reference``) with affine gap penalties, so pauses and insertions cost
a single gap-open event rather than accumulating per character. Substitutions
are scored by articulatory closeness (``phoneme_sifat.graduated_mismatch_score``)
rather than a flat penalty, which is what lets the ``.balanced`` scorer tolerate
the confusable-consonant slips the ASR model makes on amateur recitation.

Ported verbatim from ``Muraja/ios/HifzGuide/FollowAlong/SmithWatermanAlignment.swift``
and validated against ``SmithWatermanTests.swift``. The scoring constants below
match Muraja exactly — do not retune them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .phoneme_sifat import graduated_mismatch_score

# Scoring parameters, identical to Muraja's smithWaterman(). These are the
# alignment costs (fixed across scoring modes), not per-word grading thresholds.
MATCH_SCORE = 1.0
MISMATCH_PENALTY = -0.5
BEST_MISMATCH = 0.2
GAP_OPEN = -0.5
GAP_EXTEND = -0.1


@dataclass(frozen=True)
class RefMatchInfo:
    """Per-reference-position alignment outcome.

    ``kind`` is ``"match"`` (exact), ``"mismatch"`` (different consonant),
    ``"gap"`` (ref char skipped), or ``"tashkeel"`` (consonant matches but
    harakat differ — carried by ``expected``/``heard``). Smith-Waterman itself
    only emits match/mismatch/gap; tashkeel is detected downstream.
    """

    kind: str
    expected: str | None = None
    heard: str | None = None


MATCH = RefMatchInfo("match")
MISMATCH = RefMatchInfo("mismatch")
GAP = RefMatchInfo("gap")


@dataclass(frozen=True)
class AlignedColumn:
    """One column of the recovered local alignment.

    ``query_char`` is ``None`` for a reference-only column (the query dropped a
    character — a gap), and ``ref_char`` is ``None`` for a query-only column (an
    insertion). Both are set for a match or a substitution. Unlike
    ``ref_matches``/``ref_to_query`` (indexed by reference position, so query
    insertions are invisible), the column list is the *complete* aligned
    sequence — which is what contrast attribution (issue #16) needs to see both
    directions of a shadda present↔absent difference.
    """

    query_char: str | None
    ref_char: str | None


def tashkeel(expected: str | None, heard: str | None) -> RefMatchInfo:
    """A ``RefMatchInfo`` for a consonant match whose harakat differ."""
    return RefMatchInfo("tashkeel", expected, heard)


@dataclass(frozen=True)
class AlignmentResult:
    """Best local alignment of ``query`` within ``reference``.

    ``ref_matches[i]`` / ``ref_to_query[i]`` describe reference position
    ``ref_start + i``: its outcome and the aligned query index (``-1`` for a
    gap). ``columns`` is the complete aligned column sequence (including query
    insertions, which the reference-indexed views omit). ``runner_up_scores``
    holds up to two next-best non-overlapping peak scores, used downstream to
    judge how unambiguous the match is.
    """

    score: float
    ref_start: int
    ref_end: int
    query_start: int
    query_end: int
    ref_matches: list[RefMatchInfo]
    ref_to_query: list[int]
    runner_up_scores: list[float]
    columns: list[AlignedColumn]


def _substitution_score(a: str, b: str) -> float:
    """Diagonal cell score for aligning query char ``a`` against ref char ``b``.

    Exact matches earn ``MATCH_SCORE`` (spaces contribute 0.0 so they don't
    inflate the score); everything else is the graduated articulatory penalty.
    """
    if a == b:
        return 0.0 if a == " " else MATCH_SCORE
    return graduated_mismatch_score(
        a, b,
        worst_penalty=MISMATCH_PENALTY,
        best_mismatch=BEST_MISMATCH,
        fallback=MISMATCH_PENALTY,
    )


def local_alignment_score(query: str, reference: str) -> float:
    """Best local-alignment score only — no traceback, ``O(n)`` memory.

    Equivalent to ``smith_waterman(query, reference).score`` but skips the
    traceback matrix and per-position bookkeeping. Used on the hot path where
    only the score matters.
    """
    q = list(query)
    r = list(reference)
    m = len(q)
    n = len(r)
    if m == 0 or n == 0:
        return 0.0

    # Pre-compute substitution scores so the inner DP loop avoids repeated
    # sifat-table lookups (matches the Swift score-table optimization).
    score_table = [_substitution_score(q[i], r[j]) for i in range(m) for j in range(n)]

    row_size = n + 1
    h_prev = [0.0] * row_size
    h_curr = [0.0] * row_size
    iq_prev = [-math.inf] * row_size
    iq_curr = [-math.inf] * row_size
    ir_prev = [-math.inf] * row_size
    ir_curr = [-math.inf] * row_size
    max_score = 0.0

    for i in range(1, m + 1):
        h_curr[0] = 0.0
        iq_curr[0] = -math.inf
        ir_curr[0] = -math.inf
        score_row = (i - 1) * n

        for j in range(1, n + 1):
            iq_curr[j] = max(h_curr[j - 1] + GAP_OPEN, iq_curr[j - 1] + GAP_EXTEND)
            ir_curr[j] = max(h_prev[j] + GAP_OPEN, ir_prev[j] + GAP_EXTEND)
            diag = h_prev[j - 1] + score_table[score_row + j - 1]
            val = max(0.0, diag, iq_curr[j], ir_curr[j])
            h_curr[j] = val
            if val > max_score:
                max_score = val

        h_prev, h_curr = h_curr, h_prev
        iq_prev, iq_curr = iq_curr, iq_prev
        ir_prev, ir_curr = ir_curr, ir_prev

    return max_score


def _find_runner_up_scores(column_max: list[float], max_j: int, min_sep: int) -> list[float]:
    """Up to two next-best alignment peaks separated from the best match.

    Scans ``column_max`` for strict local maxima at least ``min_sep`` columns
    from the best match (and from each other), then returns the top two scores —
    a heuristic for how unique the best alignment is.
    """
    n = len(column_max) - 1
    peaks: list[tuple[int, float]] = []
    for j in range(1, n + 1):
        s = column_max[j]
        if s <= 0:
            continue
        left_ok = j == 1 or s > column_max[j - 1]
        right_ok = j == n or s > column_max[j + 1]
        if left_ok and right_ok and abs(j - max_j) >= min_sep:
            peaks.append((j, s))

    peaks.sort(key=lambda peak: peak[1], reverse=True)
    scores: list[float] = []
    cols: list[int] = []
    for col, score in peaks:
        if len(scores) >= 2:
            break
        if all(abs(c - col) >= min_sep for c in cols):
            scores.append(score)
            cols.append(col)
    return scores


def smith_waterman(query: str, reference: str) -> AlignmentResult:
    """Best local alignment of ``query`` within ``reference`` with affine gaps.

    Returns the alignment score plus per-reference-position outcomes and the
    query index each reference position aligned to. Uses ``O(n)`` rolling score
    rows and a compact 1-byte-per-cell traceback matrix.
    """
    q = list(query)
    r = list(reference)
    m = len(q)
    n = len(r)
    if m == 0 or n == 0:
        return AlignmentResult(0.0, 0, 0, 0, 0, [], [], [], [])

    # Traceback pointer per cell (bit layout mirrors the Swift encoding):
    #   bits [1:0] = H source:  0=restart, 1=diag, 2=fromIq, 3=fromIr
    #   bit  [2]   = Iq source: 0=gap open, 1=gap extend
    #   bit  [3]   = Ir source: 0=gap open, 1=gap extend
    stride = n + 1
    trace = bytearray(stride * (m + 1))

    h_prev = [0.0] * stride
    h_curr = [0.0] * stride
    iq_prev = [-math.inf] * stride
    iq_curr = [-math.inf] * stride
    ir_prev = [-math.inf] * stride
    ir_curr = [-math.inf] * stride
    max_score = 0.0
    max_i = 0
    max_j = 0
    column_max = [0.0] * stride

    for i in range(1, m + 1):
        h_curr[0] = 0.0
        iq_curr[0] = -math.inf
        ir_curr[0] = -math.inf

        for j in range(1, n + 1):
            iq_from_h = h_curr[j - 1] + GAP_OPEN
            iq_from_iq = iq_curr[j - 1] + GAP_EXTEND
            if iq_from_h >= iq_from_iq:
                iq_val, iq_source = iq_from_h, 0
            else:
                iq_val, iq_source = iq_from_iq, 1
            iq_curr[j] = iq_val

            ir_from_h = h_prev[j] + GAP_OPEN
            ir_from_ir = ir_prev[j] + GAP_EXTEND
            if ir_from_h >= ir_from_ir:
                ir_val, ir_source = ir_from_h, 0
            else:
                ir_val, ir_source = ir_from_ir, 1
            ir_curr[j] = ir_val

            diag = h_prev[j - 1] + _substitution_score(q[i - 1], r[j - 1])

            val = 0.0
            h_source = 0
            if diag > val:
                val, h_source = diag, 1
            if iq_val > val:
                val, h_source = iq_val, 2
            if ir_val > val:
                val, h_source = ir_val, 3
            h_curr[j] = val

            trace[i * stride + j] = h_source | (iq_source << 2) | (ir_source << 3)

            if val > max_score:
                max_score = val
                max_i = i
                max_j = j
            if val > column_max[j]:
                column_max[j] = val

        h_prev, h_curr = h_curr, h_prev
        iq_prev, iq_curr = iq_curr, iq_prev
        ir_prev, ir_curr = ir_curr, ir_prev

    min_sep = max(m // 2, 3)
    runner_up_scores = _find_runner_up_scores(column_max, max_j, min_sep)

    # Traceback across the H/Iq/Ir matrices to recover per-position outcomes.
    i, j = max_i, max_j
    current = "h"
    trace_steps: list[tuple[int, int, RefMatchInfo]] = []
    columns_rev: list[AlignedColumn] = []
    while i > 0 and j > 0:
        tb = trace[i * stride + j]
        if current == "h":
            h_source = tb & 0x03
            if h_source == 0:
                break  # restart — alignment starts here
            if h_source == 1:
                info = MATCH if q[i - 1] == r[j - 1] else MISMATCH
                trace_steps.append((j - 1, i - 1, info))
                columns_rev.append(AlignedColumn(q[i - 1], r[j - 1]))
                i -= 1
                j -= 1
            elif h_source == 2:
                current = "iq"
            else:
                current = "ir"
        elif current == "iq":
            trace_steps.append((j - 1, -1, GAP))
            columns_rev.append(AlignedColumn(None, r[j - 1]))
            iq_source = (tb >> 2) & 0x01
            j -= 1
            current = "h" if iq_source == 0 else "iq"
        else:  # "ir"
            columns_rev.append(AlignedColumn(q[i - 1], None))
            ir_source = (tb >> 3) & 0x01
            i -= 1
            current = "h" if ir_source == 0 else "ir"

    trace_steps.reverse()
    columns = list(reversed(columns_rev))

    ref_start = j
    ref_end = max_j
    span = ref_end - ref_start
    ref_matches = [GAP] * span
    ref_to_query = [-1] * span
    for ref_idx, query_idx, info in trace_steps:
        local_idx = ref_idx - ref_start
        if 0 <= local_idx < span:
            ref_matches[local_idx] = info
            ref_to_query[local_idx] = query_idx

    return AlignmentResult(
        score=max_score,
        ref_start=ref_start,
        ref_end=ref_end,
        query_start=i,
        query_end=max_i,
        ref_matches=ref_matches,
        ref_to_query=ref_to_query,
        runner_up_scores=runner_up_scores,
        columns=columns,
    )
