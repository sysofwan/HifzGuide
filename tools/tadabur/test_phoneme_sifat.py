"""Parity tests for phoneme sifat/makhraj scoring, ported from Muraja's
PhonemeSifatTests.swift. Values must reproduce the Swift `.balanced` scorer."""

import pytest

from tadabur.phoneme_sifat import (
    Makhraj,
    PhonemeSifa,
    graduated_mismatch_score,
    is_soft_mismatch,
    makhraj_distance,
    phoneme_similarity,
)

# Consonant scalars used across the fixtures.
HAMZA = "\u0621"
ALIF = "\u0627"
BAA = "\u0628"
TAA = "\u062A"
THAA = "\u062B"
JEEM = "\u062C"
HAA = "\u062D"
KHAA = "\u062E"
DAAL = "\u062F"
DHAL = "\u0630"
ZAI = "\u0632"
SEEN = "\u0633"
SAAD = "\u0635"
DAAD = "\u0636"
TAH = "\u0637"
ZAH = "\u0638"
FAA = "\u0641"
QAF = "\u0642"
KAF = "\u0643"
HA = "\u0647"
WAW = "\u0648"
FATHA = "\u064E"
DAMMA = "\u064F"

ALL_CONSONANTS = [
    HAMZA, ALIF, BAA, TAA, THAA, JEEM, HAA, KHAA, DAAL, DHAL,
    "\u0631", ZAI, SEEN, "\u0634", SAAD, DAAD, TAH, ZAH, "\u0639",
    "\u063A", FAA, QAF, KAF, "\u0644", "\u0645", "\u0646", HA, WAW, "\u064A",
]


# MARK: - PhonemeSifa.matching_count


def _sifa(**overrides):
    base = dict(
        makhraj=Makhraj.JAWF, hams=False, shadeed="shadeed", mofakham=False,
        motbaq=False, safeer=False, qalqala=False, tikraar=False,
        tafashie=False, istitala=False, ghonna=False,
    )
    base.update(overrides)
    return PhonemeSifa(**base)


def test_matching_count_identical():
    sifa = _sifa(qalqala=True)
    assert sifa.matching_count(sifa) == PhonemeSifa.CATEGORY_COUNT


def test_matching_count_all_different():
    a = _sifa(
        hams=True, shadeed="shadeed", mofakham=True, motbaq=True,
        safeer=True, qalqala=True, tikraar=True, tafashie=True,
        istitala=True, ghonna=True,
    )
    b = _sifa(
        hams=False, shadeed="rikhw", mofakham=False, motbaq=False,
    )
    assert a.matching_count(b) == 0


def test_matching_count_partial():
    # Both jahr, both shadeed, differ on the rest.
    a = _sifa(hams=False, shadeed="shadeed", mofakham=True, motbaq=True, safeer=True)
    b = _sifa(
        hams=False, shadeed="shadeed", mofakham=False, motbaq=False,
        qalqala=True, tikraar=True, tafashie=True, istitala=True, ghonna=True,
    )
    assert a.matching_count(b) == 2


def test_category_count_is_10():
    assert PhonemeSifa.CATEGORY_COUNT == 10


# MARK: - phoneme_similarity


def test_similarity_identical_phonemes():
    assert phoneme_similarity(BAA, BAA) == pytest.approx(1.0)


def test_similarity_related_phonemes():
    # ب (shafawi) and ج (midTongue): identical sifat, distant makhraj (0.6).
    assert phoneme_similarity(BAA, JEEM) == pytest.approx(0.64, abs=1e-3)


def test_similarity_dissimilar_phonemes():
    assert phoneme_similarity(SAAD, BAA) < 0.5


def test_similarity_non_consonant():
    assert phoneme_similarity(FATHA, BAA) is None


def test_similarity_both_non_consonant():
    assert phoneme_similarity(FATHA, DAMMA) is None


# MARK: - graduated_mismatch_score


def test_graduated_score_identical_phonemes():
    score = graduated_mismatch_score(BAA, BAA, worst_penalty=-0.5, best_mismatch=0.2)
    assert score == pytest.approx(0.2, abs=1e-3)


def test_graduated_score_fallback_for_non_consonant():
    score = graduated_mismatch_score(FATHA, BAA, fallback=-0.5)
    assert score == pytest.approx(-0.5, abs=1e-3)


def test_graduated_score_interpolation():
    # ص and س: both sibilant (d=0.0), 8/10 sifat → sim = 0.92.
    assert phoneme_similarity(SAAD, SEEN) == pytest.approx(0.92, abs=1e-3)
    score = graduated_mismatch_score(SAAD, SEEN, worst_penalty=-0.5, best_mismatch=0.0)
    assert score == pytest.approx(-0.04, abs=1e-3)


def test_graduated_score_default_parameters():
    # ب↔ت: makhraj 0.5, 8/10 sifat → sim 0.62 → default best_mismatch 0.0.
    score = graduated_mismatch_score(BAA, TAA)
    assert score == pytest.approx(-0.19, abs=1e-3)


# MARK: - Lenient mode


def test_graduated_score_lenient_high_similarity():
    # ت and د: same makhraj, 8/10 sifat → sim 0.92 ≥ 0.9 → lenient boost 0.9.
    assert phoneme_similarity(TAA, DAAL) >= 0.9
    score = graduated_mismatch_score(
        TAA, DAAL, worst_penalty=-0.5, best_mismatch=0.2, lenient=True
    )
    assert score == pytest.approx(0.9, abs=1e-3)


def test_graduated_score_lenient_borderline():
    sim = phoneme_similarity(BAA, JEEM)
    assert sim == pytest.approx(0.64, abs=1e-3)
    score = graduated_mismatch_score(
        BAA, JEEM, worst_penalty=-0.5, best_mismatch=0.0, lenient=True
    )
    assert score == pytest.approx(-0.5 + sim * 0.5, abs=1e-3)


def test_graduated_score_lenient_dissimilar():
    sim = phoneme_similarity(SAAD, BAA)
    assert sim < 0.8
    score = graduated_mismatch_score(
        SAAD, BAA, worst_penalty=-0.5, best_mismatch=0.0, lenient=True
    )
    assert score == pytest.approx(-0.5 + sim * 0.5, abs=1e-3)


def test_graduated_score_strict_unchanged():
    strict = graduated_mismatch_score(BAA, JEEM, worst_penalty=-0.5, best_mismatch=0.2, lenient=False)
    default = graduated_mismatch_score(BAA, JEEM, worst_penalty=-0.5, best_mismatch=0.2)
    assert strict == pytest.approx(default, abs=1e-3)
    assert strict == pytest.approx(-0.052, abs=1e-3)


# MARK: - Soft mismatch pairs


def test_soft_mismatch_balanced_pairs():
    assert is_soft_mismatch(DHAL, ZAI, soft_pairs_enabled=True)
    assert is_soft_mismatch(ZAI, DHAL, soft_pairs_enabled=True)  # order-independent
    assert is_soft_mismatch(TAA, TAH, soft_pairs_enabled=True)
    assert is_soft_mismatch(DAAD, ZAH, soft_pairs_enabled=True)
    assert is_soft_mismatch(HAA, HA, soft_pairs_enabled=True)
    assert is_soft_mismatch(HA, HAA, soft_pairs_enabled=True)
    assert is_soft_mismatch(SEEN, SAAD, soft_pairs_enabled=True)
    assert is_soft_mismatch(KAF, QAF, soft_pairs_enabled=True)


def test_soft_mismatch_strict_always_false():
    assert not is_soft_mismatch(DHAL, ZAI, soft_pairs_enabled=False)
    assert not is_soft_mismatch(TAA, TAH, soft_pairs_enabled=False)
    assert not is_soft_mismatch(DAAD, ZAH, soft_pairs_enabled=False)
    assert not is_soft_mismatch(HAA, HA, soft_pairs_enabled=False)


def test_soft_mismatch_hard_pairs():
    assert not is_soft_mismatch(FAA, WAW, soft_pairs_enabled=True)
    assert not is_soft_mismatch(HAA, KHAA, soft_pairs_enabled=True)


# MARK: - Makhraj


def test_all_consonants_have_makhraj():
    for scalar in ALL_CONSONANTS:
        sim = phoneme_similarity(scalar, scalar)
        assert sim is not None, f"Consonant {scalar!r} missing from sifat table"
        assert sim == pytest.approx(1.0, abs=1e-3)


def test_makhraj_distance_symmetric():
    cases = [
        (Makhraj.HALQ_DEEP, Makhraj.HALQ_UPPER),
        (Makhraj.BACK_TONGUE, Makhraj.MID_TONGUE),
        (Makhraj.TIP_TONGUE, Makhraj.SIBILANT),
        (Makhraj.JAWF, Makhraj.SHAFAWI),
    ]
    for a, b in cases:
        assert makhraj_distance(a, b) == pytest.approx(makhraj_distance(b, a), abs=1e-3)


def test_makhraj_distance_same_group_zero():
    for m in Makhraj:
        assert makhraj_distance(m, m) == pytest.approx(0.0, abs=1e-3)


def test_makhraj_golden_ordering():
    sim_ta_tah = phoneme_similarity(TAA, TAH)
    sim_seen_saad = phoneme_similarity(SEEN, SAAD)
    sim_haa_ha = phoneme_similarity(HAA, HA)
    sim_dhal_zai = phoneme_similarity(DHAL, ZAI)
    sim_baa_taa = phoneme_similarity(BAA, TAA)
    sim_dhal_waw = phoneme_similarity(DHAL, WAW)
    sim_faa_ha = phoneme_similarity(FAA, HA)

    assert sim_ta_tah == pytest.approx(0.88, abs=1e-2)
    assert sim_seen_saad == pytest.approx(0.92, abs=1e-2)
    assert sim_haa_ha == pytest.approx(0.85, abs=1e-2)
    assert sim_dhal_zai == pytest.approx(0.96, abs=1e-2)
    assert sim_baa_taa == pytest.approx(0.62, abs=1e-2)
    assert sim_dhal_waw == pytest.approx(0.64, abs=1e-2)
    assert sim_faa_ha == pytest.approx(0.52, abs=1e-2)

    assert sim_haa_ha > sim_baa_taa
    assert sim_seen_saad > sim_baa_taa
    assert sim_dhal_zai > sim_dhal_waw
    assert sim_baa_taa > sim_faa_ha

    assert sim_dhal_waw < 0.65
    assert sim_faa_ha < 0.55


def test_makhraj_broken_pairs_fixed():
    assert phoneme_similarity(DHAL, WAW) == pytest.approx(0.64, abs=1e-2)
    assert phoneme_similarity(FAA, HA) == pytest.approx(0.52, abs=1e-2)
    assert phoneme_similarity(HAA, THAA) == pytest.approx(0.70, abs=1e-2)

    assert graduated_mismatch_score(DHAL, WAW, worst_penalty=-0.5, best_mismatch=0.2) < 0.0
    assert graduated_mismatch_score(FAA, HA, worst_penalty=-0.5, best_mismatch=0.2) < 0.0
