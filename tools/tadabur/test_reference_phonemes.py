"""Tests for reference-phoneme generation and caching."""

import json

import pytest

import generate_phonemes

from tadabur import reference_phonemes

# Expected normalized (.balanced-scorer) references for the 8 fallback ayat.
# Shadda/repetition runs are collapsed to one core (رَببُ → رب), so these no
# longer carry the artifacts the review flagged in the corrupted cache.
_EXPECTED_FALLBACK_NORMALIZED = {
    "55:17": "رب لمشرقين ورب لمغربين",
    "90:8": "ءلم نجعلهۥ عينين",
    "90:9": "ولسانوشفتين",
    "90:10": "وهديناه نجدين",
    "106:1": "لءۦلاف قريش",
    "106:2": "ءۦلافهم رحلت شتاء وصيف",
    "106:3": "فليعبدۥ رب هاذ لبيت",
    "106:4": "ءلذۦ ءطعمهمنجۥعوءامنهمن خوف",
}


def test_build_normalizes_raw_phonemes():
    raw = {"1:1": "ببب", "2:1": "بَ"}
    built = reference_phonemes.build_reference_phonemes(raw)
    assert built == {"1:1": "ب", "2:1": "ب"}


def test_build_preserves_all_keys():
    raw = {"1:1": "بت", "114:6": "من"}
    assert set(reference_phonemes.build_reference_phonemes(raw)) == {"1:1", "114:6"}


def test_fallback_ayahs_are_present_and_normalized():
    # The 8 buggy ayahs must be carried through the fallback and collapsed to the
    # scorer's normalized form (no residual repetition artifacts).
    raw = dict(generate_phonemes.FALLBACK_PHONEMES)
    built = reference_phonemes.build_reference_phonemes(raw)
    assert built == _EXPECTED_FALLBACK_NORMALIZED


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


def _scorer_oracle(raw: str, sifa) -> str:
    """The .balanced scorer's normalized form of ``raw``: fold tajweed variants,
    then collapse each word with quran-transcript's ``chunck_phonemes``, keeping
    the first (core) char of each group. Word boundaries are preserved."""
    fold = {"\u06FE": "\u0645", "\u06BA": "\u0646"}
    folded = "".join(fold.get(ch, ch) for ch in raw)
    return " ".join(
        "".join(group[0] for group in sifa.chunck_phonemes(word))
        for word in folded.split(" ")
    )


def test_full_reference_set_matches_scorer_grouping():
    # Full-set parity: build the reference for all 6236 ayat through the real
    # generate → normalize path, then assert every value equals quran-transcript's
    # canonical chunck_phonemes grouping of the raw phonemes. This proves the cache
    # carries no residual repetition artifacts (55:17 is رب, not ربب), while still
    # allowing cores kept apart by a diacritic in the raw (104:1 ويللكل, where لُ
    # and للِ are distinct groups). The cache itself is a gitignored, regenerable
    # artifact, so this exercises the generator rather than a committed file.
    sifa = pytest.importorskip("quran_transcript.phonetics.sifa")
    raw = generate_phonemes.generate_reference_phonemes()
    built = reference_phonemes.build_reference_phonemes(raw)
    assert len(built) == generate_phonemes.TOTAL_AYAT
    assert built.keys() == raw.keys()
    assert set(generate_phonemes.FALLBACK_PHONEMES) <= built.keys()
    mismatches = {
        key: (built[key], _scorer_oracle(raw[key], sifa))
        for key in raw
        if built[key] != _scorer_oracle(raw[key], sifa)
    }
    assert not mismatches
