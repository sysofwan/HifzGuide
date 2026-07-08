"""Tests for the canonical 43-class phoneme vocabulary and CTC greedy decode.

The parity test is the load-bearing one: it asserts this repo's codified phoneme
vocabulary (which mirrors Muraja's ``PhonemeVocabulary`` / ``phonemeMap``) matches
the live ``obadx/muaalem-model-v3_2`` model's phoneme head **exactly**, and fails
loudly on any drift. It is skipped only when the model repo is unreachable (no
network / hub error) — never on a mismatch.
"""

from __future__ import annotations

import json

import pytest

from tadabur.inference import MODEL_ID, PHONEME_LEVEL
from tadabur.phoneme_vocab import (
    NUM_PHONEME_CLASSES,
    PHONEME_CHAR_TO_ID,
    PHONEME_ID_TO_CHAR,
    PHONEME_PAD_ID,
    greedy_ctc_decode,
)


def test_vocab_has_43_contiguous_bijective_classes():
    assert NUM_PHONEME_CLASSES == 43
    assert len(PHONEME_ID_TO_CHAR) == 43
    # Bijective: 43 distinct chars, ids exactly 0..42.
    assert len(PHONEME_CHAR_TO_ID) == 43
    assert set(PHONEME_CHAR_TO_ID.values()) == set(range(43))
    for idx, char in enumerate(PHONEME_ID_TO_CHAR):
        assert PHONEME_CHAR_TO_ID[char] == idx
    assert PHONEME_ID_TO_CHAR[PHONEME_PAD_ID] == "[PAD]"


def test_vocab_matches_live_muaalem_model():
    """The codified vocab must equal the model's phoneme head; drift fails loudly."""
    hf_hub_download = pytest.importorskip("huggingface_hub").hf_hub_download
    try:
        vocab_path = hf_hub_download(MODEL_ID, "vocab.json")
    except Exception as exc:  # network/hub unavailable — not a vocab mismatch
        pytest.skip(f"{MODEL_ID} unreachable: {exc}")

    with open(vocab_path, encoding="utf-8") as f:
        model_phoneme_vocab = json.load(f)[PHONEME_LEVEL]

    assert model_phoneme_vocab == PHONEME_CHAR_TO_ID, (
        "Muaalem phoneme vocabulary drifted from tadabur.phoneme_vocab — decoded "
        "phonemes / fine-tune labels would be corrupt. Reconcile the constant with "
        f"{MODEL_ID}/vocab.json and Muraja's PhonemeVocabulary."
    )


def test_greedy_decode_collapses_repeats_and_drops_blank():
    b = PHONEME_PAD_ID
    baa = PHONEME_CHAR_TO_ID["\u0628"]   # ب
    seen = PHONEME_CHAR_TO_ID["\u0633"]  # س
    # blanks dropped, consecutive repeats collapsed, blank-separated repeats kept.
    ids = [b, baa, baa, b, b, seen, seen, seen, b, baa, b]
    assert greedy_ctc_decode(ids) == "\u0628\u0633\u0628"  # بسب


def test_greedy_decode_empty_and_all_blank():
    assert greedy_ctc_decode([]) == ""
    assert greedy_ctc_decode([PHONEME_PAD_ID, PHONEME_PAD_ID]) == ""
