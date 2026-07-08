"""Tests for the canonical 43-class phoneme vocabulary and CTC greedy decode.

The parity here is triangular and load-bearing. A committed **Muraja snapshot**
(``fixtures/muraja_phoneme_vocabulary.json``, exported directly from Muraja's Swift
``PhonemeVocabulary``) is the authoritative reference. Both this repo's codified
``PHONEME_CHAR_TO_ID`` *and* the live ``obadx/muaalem-model-v3_2`` ``vocab.json`` are
asserted equal to that snapshot exactly. Because the snapshot comes from Muraja — a
third, independent source — the model constant and the live model can no longer agree
with each other while both silently drifting from what the iOS app actually decodes.

Only the live-model check is skipped when the model repo is unreachable (no network /
hub error) — never on a mismatch, and never the offline Muraja-vs-constant check.
"""

from __future__ import annotations

import json

import pytest

from tadabur.inference import MODEL_ID, PHONEME_LEVEL
from tadabur.muraja_phoneme_snapshot import load_muraja_phoneme_snapshot
from tadabur.phoneme_vocab import (
    NUM_PHONEME_CLASSES,
    PAD_TOKEN,
    PHONEME_CHAR_TO_ID,
    PHONEME_ID_TO_CHAR,
    PHONEME_PAD_ID,
    greedy_ctc_decode,
)


def _emittable_char_to_id(full_vocab: dict[str, int]) -> dict[str, int]:
    """Drop the CTC blank so a full char->id vocab can be compared to Muraja's.

    Muraja's ``PhonemeVocabulary`` maps only emittable phonemes (ids 1..42); the
    blank has no character. Both the model vocab and this repo's constant include a
    ``[PAD]`` blank entry, so strip it before comparing to the snapshot.
    """
    if full_vocab.get(PAD_TOKEN) != PHONEME_PAD_ID:
        raise AssertionError(
            f"Expected blank {PAD_TOKEN!r} at id {PHONEME_PAD_ID}; vocab has "
            f"{PAD_TOKEN!r} -> {full_vocab.get(PAD_TOKEN)!r}."
        )
    return {char: idx for char, idx in full_vocab.items() if char != PAD_TOKEN}


def test_vocab_has_43_contiguous_bijective_classes():
    assert NUM_PHONEME_CLASSES == 43
    assert len(PHONEME_ID_TO_CHAR) == 43
    # Bijective: 43 distinct chars, ids exactly 0..42.
    assert len(PHONEME_CHAR_TO_ID) == 43
    assert set(PHONEME_CHAR_TO_ID.values()) == set(range(43))
    for idx, char in enumerate(PHONEME_ID_TO_CHAR):
        assert PHONEME_CHAR_TO_ID[char] == idx
    assert PHONEME_ID_TO_CHAR[PHONEME_PAD_ID] == PAD_TOKEN


def test_constant_matches_muraja_snapshot():
    """This repo's codified vocab must equal Muraja's on-device PhonemeVocabulary."""
    muraja = load_muraja_phoneme_snapshot()
    assert NUM_PHONEME_CLASSES == muraja.vocab_size
    assert PHONEME_PAD_ID == muraja.blank_id
    assert _emittable_char_to_id(PHONEME_CHAR_TO_ID) == muraja.char_to_id, (
        "tadabur.phoneme_vocab drifted from Muraja's PhonemeVocabulary snapshot — "
        "decoded phonemes would no longer match what the iOS app decodes on-device. "
        "Reconcile PHONEME_ID_TO_CHAR with fixtures/muraja_phoneme_vocabulary.json."
    )


def test_live_model_vocab_matches_muraja_snapshot():
    """The live Muaalem phoneme head must equal Muraja's snapshot; drift fails loudly."""
    hf_hub_download = pytest.importorskip("huggingface_hub").hf_hub_download
    try:
        vocab_path = hf_hub_download(MODEL_ID, "vocab.json")
    except Exception as exc:  # network/hub unavailable — not a vocab mismatch
        pytest.skip(f"{MODEL_ID} unreachable: {exc}")

    with open(vocab_path, encoding="utf-8") as f:
        model_phoneme_vocab = json.load(f)[PHONEME_LEVEL]

    muraja = load_muraja_phoneme_snapshot()
    assert len(model_phoneme_vocab) == muraja.vocab_size
    assert _emittable_char_to_id(model_phoneme_vocab) == muraja.char_to_id, (
        f"{MODEL_ID} phoneme vocabulary drifted from Muraja's PhonemeVocabulary "
        "snapshot — the fine-tune labels / on-device decode would disagree. "
        "Reconcile the model, fixtures/muraja_phoneme_vocabulary.json, and Muraja."
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
