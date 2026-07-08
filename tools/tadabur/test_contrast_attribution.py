"""Tests for contrast attribution (issue #16): which soft-pair / shadda contrasts
admitted a passing alignment, without altering the gate."""

from __future__ import annotations

import pytest

from tadabur.contrast_attribution import (
    MARGINAL_CONTRAST,
    SHADDA_CONTRAST,
    _has_shadda_contrast,
    all_contrasts,
    attribute_contrasts,
    contrast_vocabulary,
)
from tadabur.phoneme_sifat import soft_pair_contrast, soft_pair_contrasts
from tadabur.scorer import BALANCED_SCORER, Scorer, ScoringParameters
from tadabur.smith_waterman import AlignedColumn

# Soft-pair scalars.
DHAL, ZAI = "\u0630", "\u0632"
TAA, TAH = "\u062A", "\u0637"
DAAD, ZAH = "\u0636", "\u0638"
KAF, QAF = "\u0643", "\u0642"
SEEN, SAAD = "\u0633", "\u0635"
HAA, HA = "\u062D", "\u0647"

SOFT_PAIRS = [(DHAL, ZAI), (TAA, TAH), (DAAD, ZAH), (KAF, QAF), (SEEN, SAAD), (HAA, HA)]


# MARK: - contrast vocabulary


def test_all_contrasts_is_seven_buckets():
    contrasts = all_contrasts()
    assert len(contrasts) == 7
    assert contrasts[-1] == SHADDA_CONTRAST
    assert set(contrasts[:-1]) == set(soft_pair_contrasts())
    # Deterministic, sorted soft-pair labels.
    assert list(contrasts) == sorted(soft_pair_contrasts()) + [SHADDA_CONTRAST]


def test_contrast_vocabulary_adds_marginal():
    assert contrast_vocabulary() == frozenset(all_contrasts()) | {MARGINAL_CONTRAST}


# MARK: - soft-pair substitution detection (one test per pair)


@pytest.mark.parametrize("a,b", SOFT_PAIRS)
def test_soft_pair_substitution_is_attributed(a, b):
    # predicted has `a`, reference has `b`, surrounded by shared context so the
    # alignment is a clean substitution at that position.
    predicted = "\u0644" + a + "\u0645"  # ل a م
    reference = "\u0644" + b + "\u0645"  # ل b م
    contrasts = attribute_contrasts(predicted, reference)
    assert contrasts == (soft_pair_contrast(a, b),)


def test_hard_mismatch_is_not_attributed():
    # ف↔ه is not a soft pair — no contrast should be reported.
    contrasts = attribute_contrasts("\u0644\u0641\u0645", "\u0644\u0647\u0645")
    assert contrasts == ()


def test_clean_match_has_no_contrasts():
    assert attribute_contrasts("\u0628\u062A\u062B\u062C", "\u0628\u062A\u062B\u062C") == ()


def test_strict_mode_reports_no_soft_pairs():
    # soft_pairs_enabled=False: the substitution is a hard mismatch, not a contrast.
    assert attribute_contrasts("\u0644\u0635\u0645", "\u0644\u0633\u0645", soft_pairs_enabled=False) == ()


# MARK: - shadda present↔absent detection


def test_shadda_absent_in_query_is_attributed():
    # reference has an internal doubled core (لرَببُم → لرببم); the model dropped
    # one (لربم). The extra reference ب surfaces as an internal alignment gap.
    predicted = "\u0644\u0631\u0628\u0645"                      # لربم
    reference = "\u0644\u0631\u064e\u0628\u0628\u064f\u0645"    # لرَببُم → لرببم
    assert SHADDA_CONTRAST in attribute_contrasts(predicted, reference)


def test_shadda_present_in_query_is_attributed():
    # model produced an internal doubled core (لرَببُم → لرببم) the reference has
    # singly (لربم); the extra query ب surfaces as an internal insertion.
    predicted = "\u0644\u0631\u064e\u0628\u0628\u064f\u0645"    # لرَببُم → لرببم
    reference = "\u0644\u0631\u0628\u0645"                      # لربم
    assert SHADDA_CONTRAST in attribute_contrasts(predicted, reference)


def test_plain_deletion_is_not_shadda():
    # reference بتث, query drops the middle ت: a gap whose core matches neither
    # neighbour, so it is not a shadda contrast.
    columns = [
        AlignedColumn("\u0628", "\u0628"),  # ب match
        AlignedColumn(None, "\u062A"),      # ت gap (deletion)
        AlignedColumn("\u062B", "\u062B"),  # ث match
    ]
    assert not _has_shadda_contrast(columns)


def test_shadda_detected_from_hand_built_columns_both_directions():
    ba = "\u0628"
    deletion = [AlignedColumn(ba, ba), AlignedColumn(None, ba)]   # ref بب, query ب
    insertion = [AlignedColumn(ba, ba), AlignedColumn(ba, None)]  # query بب, ref ب
    assert _has_shadda_contrast(deletion)
    assert _has_shadda_contrast(insertion)


# MARK: - attribution is observational (does not touch the gate)


def test_attribution_does_not_change_gate():
    predicted, reference = "\u0635\u0644\u0645", "\u0633\u0644\u0645"  # صلم vs سلم (soft pair)
    before = BALANCED_SCORER.gate(predicted, reference)
    contrasts = BALANCED_SCORER.attribute(predicted, reference)
    after = BALANCED_SCORER.gate(predicted, reference)
    assert before == after
    assert before.passed  # the soft-pair clip still passes
    assert contrasts == (soft_pair_contrast(SEEN, SAAD),)


def test_scorer_attribute_respects_mode():
    predicted, reference = "\u0644\u0635\u0645", "\u0644\u0633\u0645"
    strict = Scorer(ScoringParameters(0.75, soft_pairs_enabled=False, shaddah_suppression=False))
    assert BALANCED_SCORER.attribute(predicted, reference) == (soft_pair_contrast(SEEN, SAAD),)
    assert strict.attribute(predicted, reference) == ()


def test_multiple_contrasts_are_sorted_and_deduped():
    # A pair exhibiting two distinct soft-pair substitutions reports both, sorted.
    predicted = "\u0635\u0644\u0637"  # ص ل ط
    reference = "\u0633\u0644\u062A"  # س ل ت
    contrasts = attribute_contrasts(predicted, reference)
    assert contrasts == tuple(sorted({soft_pair_contrast(SEEN, SAAD), soft_pair_contrast(TAA, TAH)}))
