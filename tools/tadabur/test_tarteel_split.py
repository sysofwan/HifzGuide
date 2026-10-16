"""Tests for tadabur.tarteel_split: Tarteel v1's look / confirm split by clip hash (#160)."""

from __future__ import annotations

import hashlib

import pytest

from tadabur import tarteel_split as ts


def test_the_rule_is_the_pre_registered_one():
    """``sha256("tarteel-v1-split-2026-10:" + clip_id)``, first 8 bytes big-endian < 2**63."""
    for clip in ("0_1_0.wav", "1_1_1.wav", "100_10_4115049010.wav"):
        value = int.from_bytes(
            hashlib.sha256(f"tarteel-v1-split-2026-10:{clip}".encode()).digest()[:8], "big")
        assert ts.clip_hash(clip) == value
        assert ts.half(clip) == (ts.LOOK if value < 2**63 else ts.CONFIRM)
    assert (ts.half("0_1_0.wav"), ts.half("1_1_1.wav")) == (ts.LOOK, ts.CONFIRM)


def test_a_decode_key_is_no_clip_id():
    with pytest.raises(ValueError):
        ts.half("100_10_4115049010.wav#0")


def test_the_committed_split_is_the_rule_over_every_decoded_clip():
    record = ts.load_split()
    assert record == ts.split_record(ts.decoded_clips())
    assert record["clips"] == 8950 == sum(record["counts"].values())
    assert not set(record[ts.LOOK]) & set(record[ts.CONFIRM])
