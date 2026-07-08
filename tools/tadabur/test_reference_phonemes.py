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


def _valid_raw_set() -> dict[str, str]:
    """A raw phoneme mapping that normalizes to a *valid* full cache: the 8
    fallback ayat plus synthetic filler to reach exactly TOTAL_AYAT keys."""
    raw = dict(generate_phonemes.FALLBACK_PHONEMES)
    i = 0
    while len(raw) < generate_phonemes.TOTAL_AYAT:
        raw[f"filler:{i}"] = "بت"
        i += 1
    return raw


def _use_raw(monkeypatch, raw: dict[str, str]) -> None:
    monkeypatch.setattr(
        generate_phonemes, "generate_reference_phonemes", lambda: dict(raw)
    )


def test_fallback_ayahs_are_present_and_normalized():
    # The 8 buggy ayahs must be carried through the fallback and collapsed to the
    # scorer's normalized form (no residual repetition artifacts).
    raw = dict(generate_phonemes.FALLBACK_PHONEMES)
    built = reference_phonemes.build_reference_phonemes(raw)
    assert built == _EXPECTED_FALLBACK_NORMALIZED


def test_load_writes_cache_then_hits_it(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "reference_phonemes.json"
    _use_raw(monkeypatch, _valid_raw_set())

    first = reference_phonemes.load_reference_phonemes(cache)
    assert cache.exists()
    assert len(first) == generate_phonemes.TOTAL_AYAT
    assert first["55:17"] == "رب لمشرقين ورب لمغربين"

    # A warm, valid cache must not rebuild — force build to fail if it is called.
    def _boom():
        raise AssertionError("cache miss: build was called on a warm cache")

    monkeypatch.setattr(generate_phonemes, "generate_reference_phonemes", _boom)
    assert reference_phonemes.load_reference_phonemes(cache) == first


def test_cache_is_deterministic_utf8(tmp_path, monkeypatch):
    cache = tmp_path / "reference_phonemes.json"
    _use_raw(monkeypatch, _valid_raw_set())

    reference_phonemes.load_reference_phonemes(cache, rebuild=True)
    first_bytes = cache.read_bytes()
    reference_phonemes.load_reference_phonemes(cache, rebuild=True)
    assert cache.read_bytes() == first_bytes  # idempotent, byte-identical

    text = cache.read_text(encoding="utf-8")
    assert "\\u" not in text  # Arabic written literally (ensure_ascii=False)
    payload = json.loads(text)
    assert payload["cache_version"] == reference_phonemes.CACHE_VERSION
    refs = payload["references"]
    assert list(refs) == sorted(refs)  # stable key order


def _write_raw_payload(cache, payload) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def test_stale_version_cache_is_rebuilt(tmp_path, monkeypatch):
    # A cache written by an older algorithm carries a different cache_version.
    cache = tmp_path / "reference_phonemes.json"
    _use_raw(monkeypatch, _valid_raw_set())
    valid_refs = reference_phonemes.build_reference_phonemes(_valid_raw_set())
    _write_raw_payload(
        cache, {"cache_version": "v0+norm0", "references": valid_refs}
    )

    rebuilt = {"flag": False}

    def _tracked():
        rebuilt["flag"] = True
        return _valid_raw_set()

    monkeypatch.setattr(generate_phonemes, "generate_reference_phonemes", _tracked)
    result = reference_phonemes.load_reference_phonemes(cache)
    assert rebuilt["flag"], "stale-version cache must trigger a rebuild"
    assert len(result) == generate_phonemes.TOTAL_AYAT
    # The rewritten cache now carries the current version.
    assert json.loads(cache.read_text(encoding="utf-8"))["cache_version"] == (
        reference_phonemes.CACHE_VERSION
    )


def test_partial_cache_is_rebuilt(tmp_path, monkeypatch):
    # A truncated cache (fewer than 6236 keys) must not be trusted.
    cache = tmp_path / "reference_phonemes.json"
    partial = reference_phonemes.build_reference_phonemes(
        dict(generate_phonemes.FALLBACK_PHONEMES)
    )
    _write_raw_payload(
        cache,
        {"cache_version": reference_phonemes.CACHE_VERSION, "references": partial},
    )
    _use_raw(monkeypatch, _valid_raw_set())

    result = reference_phonemes.load_reference_phonemes(cache)
    assert len(result) == generate_phonemes.TOTAL_AYAT


def test_bare_dict_cache_is_rebuilt(tmp_path, monkeypatch):
    # An old-format cache (bare surah:ayah dict, no envelope) must be rebuilt.
    cache = tmp_path / "reference_phonemes.json"
    _write_raw_payload(cache, {"55:17": "ربب", "1:1": "ب"})
    _use_raw(monkeypatch, _valid_raw_set())

    result = reference_phonemes.load_reference_phonemes(cache)
    assert len(result) == generate_phonemes.TOTAL_AYAT
    assert result["55:17"] == "رب لمشرقين ورب لمغربين"


def test_tampered_fallback_sentinel_cache_is_rebuilt(tmp_path, monkeypatch):
    # A full-size cache whose fallback value differs from the algorithm's output
    # (e.g. the previous broken normalization's ربب) is rejected.
    cache = tmp_path / "reference_phonemes.json"
    refs = reference_phonemes.build_reference_phonemes(_valid_raw_set())
    refs["55:17"] = "ربب لمشرقين ورب لمغربين"  # artifact from broken normalization
    _write_raw_payload(
        cache,
        {"cache_version": reference_phonemes.CACHE_VERSION, "references": refs},
    )
    _use_raw(monkeypatch, _valid_raw_set())

    result = reference_phonemes.load_reference_phonemes(cache)
    assert result["55:17"] == "رب لمشرقين ورب لمغربين"


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
