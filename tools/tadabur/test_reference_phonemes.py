"""Tests for reference-phoneme generation and caching."""

import json

import generate_phonemes

from tadabur import reference_phonemes
from tadabur.normalization import normalize_phonemes


def test_build_normalizes_raw_phonemes():
    raw = {"1:1": "ببب", "2:1": "بَ"}
    built = reference_phonemes.build_reference_phonemes(raw)
    assert built == {"1:1": "ب", "2:1": "ب"}


def test_build_preserves_all_keys():
    raw = {"1:1": "بت", "114:6": "من"}
    assert set(reference_phonemes.build_reference_phonemes(raw)) == {"1:1", "114:6"}


def test_fallback_ayahs_are_present_and_normalized():
    # The 8 buggy ayahs must be carried through the fallback and normalized.
    raw = dict(generate_phonemes.FALLBACK_PHONEMES)
    built = reference_phonemes.build_reference_phonemes(raw)
    for key, phonemes in generate_phonemes.FALLBACK_PHONEMES.items():
        assert built[key] == normalize_phonemes(phonemes).normalized


def test_load_writes_cache_then_hits_it(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "reference_phonemes.json"
    monkeypatch.setattr(
        generate_phonemes,
        "generate_reference_phonemes",
        lambda: {"1:1": "ببب", "1:2": "تت"},
    )

    first = reference_phonemes.load_reference_phonemes(cache)
    assert cache.exists()
    assert first == {"1:1": "ب", "1:2": "ت"}

    # A warm cache must not rebuild — force build to fail if it is called.
    def _boom():
        raise AssertionError("cache miss: build was called on a warm cache")

    monkeypatch.setattr(generate_phonemes, "generate_reference_phonemes", _boom)
    assert reference_phonemes.load_reference_phonemes(cache) == first


def test_cache_is_deterministic_utf8(tmp_path, monkeypatch):
    cache = tmp_path / "reference_phonemes.json"
    monkeypatch.setattr(
        generate_phonemes,
        "generate_reference_phonemes",
        lambda: {"2:1": "بب", "10:1": "تت", "1:1": "ثث"},
    )

    reference_phonemes.load_reference_phonemes(cache, rebuild=True)
    first_bytes = cache.read_bytes()
    reference_phonemes.load_reference_phonemes(cache, rebuild=True)
    assert cache.read_bytes() == first_bytes  # idempotent, byte-identical

    text = cache.read_text(encoding="utf-8")
    assert "\\u" not in text  # Arabic written literally (ensure_ascii=False)
    assert list(json.loads(text)) == sorted(json.loads(text))  # stable key order
