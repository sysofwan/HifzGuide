"""Tests for reference-phoneme generation and caching."""

import json

import pytest

import generate_phonemes

from tadabur import reference_phonemes

# Expected normalized (.balanced-scorer) references for the 8 fallback ayat.
# Swift's normalizePhonemes keeps shadda expansion DOUBLED (رَببُ → ربب, not رب):
# a same-core cluster carrying a combining mark starts a new group, so these carry
# doubled cores exactly as the app's .balanced scorer sees them. The downstream
# word scorer's shaddahSuppression, not normalization, neutralises the doubling.
_EXPECTED_FALLBACK_NORMALIZED = {
    "55:17": "ربب لمشرقين وربب لمغربين",
    "90:8": "ءلم نجعللهۥ عينين",
    "90:9": "ولسانووشفتين",
    "90:10": "وهديناه ننجدين",
    "106:1": "لءۦلاف قريش",
    "106:2": "ءۦلافهم رحلت ششتاء وصصيف",
    "106:3": "فليعبدۥ ربب هاذ لبيت",
    "106:4": "ءللذۦ ءطعمهممنجۥعووءامنهممن خوف",
}


def test_build_normalizes_raw_phonemes():
    raw = {"1:1": "ببب", "2:1": "بَ"}
    built = reference_phonemes.build_reference_phonemes(raw)
    assert built == {"1:1": "ب", "2:1": "ب"}


def test_build_preserves_all_keys():
    raw = {"1:1": "بت", "114:6": "من"}
    assert set(reference_phonemes.build_reference_phonemes(raw)) == {"1:1", "114:6"}


def _valid_raw_set() -> dict[str, str]:
    """A raw phoneme mapping that normalizes to a *valid* full cache: exactly the
    canonical 6236 ``surah:ayah`` keys, with the 8 fallback ayat carrying their
    real raw phonemes and every other ayah a placeholder."""
    raw = {key: "بت" for key in generate_phonemes.expected_ayah_keys()}
    raw.update(generate_phonemes.FALLBACK_PHONEMES)
    return raw


def _use_raw(monkeypatch, raw: dict[str, str]) -> None:
    monkeypatch.setattr(
        generate_phonemes, "generate_reference_phonemes", lambda: dict(raw)
    )


def test_fallback_ayahs_are_present_and_normalized():
    # The 8 buggy ayahs must be carried through the fallback and normalized by the
    # Swift-faithful group-collapse (shadda expansion stays doubled).
    raw = dict(generate_phonemes.FALLBACK_PHONEMES)
    built = reference_phonemes.build_reference_phonemes(raw)
    assert built == _EXPECTED_FALLBACK_NORMALIZED


def test_load_writes_cache_then_hits_it(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "reference_phonemes.json"
    _use_raw(monkeypatch, _valid_raw_set())

    first = reference_phonemes.load_reference_phonemes(cache)
    assert cache.exists()
    assert len(first) == generate_phonemes.TOTAL_AYAT
    assert first["55:17"] == "ربب لمشرقين وربب لمغربين"

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
    assert result["55:17"] == "ربب لمشرقين وربب لمغربين"


def test_wrong_key_set_cache_is_rebuilt(tmp_path, monkeypatch):
    # A full-size (6236-entry) cache that drops a real ayah (2:255) while carrying
    # an extra key must be rejected: exact canonical key-set match is required, so
    # count alone is not enough.
    cache = tmp_path / "reference_phonemes.json"
    refs = reference_phonemes.build_reference_phonemes(_valid_raw_set())
    assert "2:255" in refs
    refs["extra:0"] = refs.pop("2:255")  # still exactly TOTAL_AYAT entries
    assert len(refs) == generate_phonemes.TOTAL_AYAT
    _write_raw_payload(
        cache,
        {"cache_version": reference_phonemes.CACHE_VERSION, "references": refs},
    )
    _use_raw(monkeypatch, _valid_raw_set())

    result = reference_phonemes.load_reference_phonemes(cache)
    assert "2:255" in result
    assert "extra:0" not in result
    assert set(result) == generate_phonemes.expected_ayah_keys()


def test_tampered_fallback_sentinel_cache_is_rebuilt(tmp_path, monkeypatch):
    # A full-size cache whose fallback value differs from the algorithm's output
    # (e.g. the previous broken normalization's collapsed رب) is rejected and
    # rebuilt to the Swift-faithful doubled form.
    cache = tmp_path / "reference_phonemes.json"
    refs = reference_phonemes.build_reference_phonemes(_valid_raw_set())
    refs["55:17"] = "رب لمشرقين ورب لمغربين"  # collapsed artifact of old normalization
    _write_raw_payload(
        cache,
        {"cache_version": reference_phonemes.CACHE_VERSION, "references": refs},
    )
    _use_raw(monkeypatch, _valid_raw_set())

    result = reference_phonemes.load_reference_phonemes(cache)
    assert result["55:17"] == "ربب لمشرقين وربب لمغربين"


def _independent_swift_normalize(text: str) -> str:
    """A second, self-contained implementation of Swift's ``normalizePhonemes``
    group-collapse, written independently of the production ``normalize_phonemes``
    so the full-set test cross-checks behaviour rather than tautologically calling
    the same code. Mirrors the Swift loop: a bare core consumes following bare
    same-folded-core clusters plus one trailing standalone residual, but a
    same-core cluster carrying a combining mark starts a NEW group."""
    import unicodedata

    core = set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوياۥۦ۾ںـٲ")
    residual = set("\u064E\u064F\u0650\u0687\u0619\u06EA\u06DC")
    fold = {"\u06FE": "\u0645", "\u06BA": "\u0646"}

    clusters: list[list[str]] = []
    for ch in text:
        if clusters and unicodedata.combining(ch):
            clusters[-1].append(ch)
        else:
            clusters.append([ch])

    out: list[str] = []
    i, n = 0, len(clusters)
    while i < n:
        base = clusters[i][0]
        if base == " ":
            out.append(" ")
            i += 1
            continue
        if base in core:
            folded = fold.get(base, base)
            has_combining = len(clusters[i]) > 1
            i += 1
            if not has_combining:
                while i < n and fold.get(clusters[i][0], clusters[i][0]) == folded:
                    if len(clusters[i]) > 1:
                        break
                    i += 1
                if i < n and clusters[i][0] in residual:
                    i += 1
            out.append(folded)
            continue
        i += 1
    return "".join(out)


# Known-good Swift-normalized values for real quran-transcript ayat, hand-verified
# against Swift's group-collapse (shadda expansion doubled: للَ→لل, ررَ→رر, نّ→نن).
_SPOT_CHECK_REFERENCES = {
    "1:1": "بسم للاه ررحمان ررحۦم",
    "1:2": "ءلحمد لللاه ربب لعالمۦن",
    "112:1": "قل هو للاه ءحد",
    "114:1": "قل ءعۥذ بربب نناس",
}


def test_full_reference_set_is_swift_faithful():
    # Full-set parity: build the reference for all 6236 ayat through the real
    # generate → normalize path, then assert every value matches an independent
    # re-implementation of Swift's group-collapse (shadda expansion stays doubled,
    # e.g. 55:17 is ربب, not رب), carries no combining marks or tajweed variants,
    # and has clean word boundaries. The cache itself is a gitignored, regenerable
    # artifact, so this exercises the generator rather than a committed file.
    import unicodedata

    pytest.importorskip("quran_transcript")
    raw = generate_phonemes.generate_reference_phonemes()
    built = reference_phonemes.build_reference_phonemes(raw)
    assert len(built) == generate_phonemes.TOTAL_AYAT
    assert built.keys() == raw.keys()
    assert set(generate_phonemes.FALLBACK_PHONEMES) <= built.keys()

    # Hand-verified spot checks anchor the independent oracle to real Swift output.
    for key, expected in _SPOT_CHECK_REFERENCES.items():
        assert built[key] == expected, (key, built[key])

    mismatches = {
        key: (built[key], _independent_swift_normalize(raw[key]))
        for key in raw
        if built[key] != _independent_swift_normalize(raw[key])
    }
    assert not mismatches

    # Structural invariants: normalization emits only folded core scalars and
    # spaces — never combining marks, tajweed variants, or ragged spacing.
    for value in built.values():
        assert not any(unicodedata.combining(ch) for ch in value)
        assert "\u06FE" not in value and "\u06BA" not in value
        assert "  " not in value
        assert value == value.strip(" ") or value == ""
