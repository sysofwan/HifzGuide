"""Tests for the shared phonetizer entry point (``hafs_phonetizer``).

The real-phonetizer cases are skipped where quran-transcript is not installed.
"""

import ast
from pathlib import Path

import pytest

from hafs_phonetizer import pausal_taa_marbuta, phonetize

TOOLS_DIR = Path(__file__).resolve().parent


def test_pausal_taa_marbuta_rewrites_only_a_final_tanween_fatha_on_taa_marbuta():
    assert pausal_taa_marbuta("رَحْمَةًۭ") == "رَحْمَةَ"  # small low meem
    assert pausal_taa_marbuta("غُرْفَةًۢ") == "غُرْفَةَ"  # small high meem
    assert pausal_taa_marbuta("وَٰحِدَةً") == "وَٰحِدَةَ"
    for word in ("رَحْمَةٌۭ", "رَحْمَةٍۢ", "رَحْمَةَ", "عَلِيمًا", "بِنَآءًۭ", "قَالَ"):
        assert pausal_taa_marbuta(word) == word


def _ayah(surah: int, ayah: int) -> str:
    from quran_transcript import Aya

    return Aya(surah, ayah).get().uthmani


@pytest.mark.parametrize(
    "text, phonemes",
    [
        ("رَحْمَةًۭ", "رَحمَه"),  # 0.5.2 alone gives رَحمَتَاا
        ("جُمْلَةًۭ وَٰحِدَةًۭ", "جُملَتَوووَااحِدَه"),  # 25:32 at the waqf on وَٰحِدَةًۭ
        ("رَحْمَةٌۭ", "رَحمَه"),  # tanween damma was already right
        ("تَرْتِيلًۭا", "تَرتِۦۦلَاا"),  # madd al-iwad is untouched
        # 2:249's غُرْفَةًۢ at a waqf: 0.6.4 still gives غُرفَتَاا with the small high meem.
        ("غُرْفَةًۢ", "غُرفَه"),
        # Trailing whitespace is not a word: the ةً before it is still the last word.
        ("رَحْمَةًۭ ", "رَحمَه"),
        (" رَحْمَةًۭ\n", "رَحمَه"),
        ("جُمْلَةًۭ وَٰحِدَةًۭ \n", "جُملَتَوووَااحِدَه"),
        # Any whitespace separates words, as in quran_phonetizer: the second word survives
        # and the first ةً keeps its wasl form.
        ("وَٰحِدَةًۭ\nكَذَٰلِكَ", "وَااحِدَتَںںںكَذَاالِك"),
        ("وَٰحِدَةًۭ\tكَذَٰلِكَ", "وَااحِدَتَںںںكَذَاالِك"),
        ("وَٰحِدَةًۭ\u00a0كَذَٰلِكَ", "وَااحِدَتَںںںكَذَاالِك"),
        ("جُمْلَةًۭ\u00a0وَٰحِدَةًۭ", "جُملَتَوووَااحِدَه"),
    ],
)
def test_phonetize_gives_the_pausal_form_of_the_final_word(text, phonemes):
    pytest.importorskip("quran_transcript")
    assert phonetize(text).phonemes == phonemes


def test_taa_marbuta_with_tanween_fatha_keeps_its_wasl_form_inside_a_run():
    pytest.importorskip("quran_transcript")
    # 89:28 ٱرْجِعِىٓ إِلَىٰ رَبِّكِ رَاضِيَةًۭ مَّرْضِيَّةًۭ: the first ةً is in wasl, merging
    # its tanween into the meem; only the ayah-final one is pausal.
    assert phonetize(_ayah(89, 28)).phonemes.endswith("رَااضِيَتَممممَرضِييَه")


def test_phonetize_fixes_an_ayah_end():
    pytest.importorskip("quran_transcript")
    assert phonetize(_ayah(88, 4)).phonemes == "تَصلَاا نَاارَن حَاامِيَه"


def test_phonetize_mappings_index_into_the_text_as_given():
    pytest.importorskip("quran_transcript")
    text = "رَحْمَةًۭ"
    out = phonetize(text)
    assert len(out.mappings) == len(text)
    end = len(out.phonemes)
    taa = text.index("ة")
    assert out.phonemes[slice(*out.mappings[taa].pos)] == "ه"
    # The tanween and the small meem after it became nothing.
    assert [(m.pos, m.deleted) for m in out.mappings[taa + 1 :]] == [((end, end), True)] * 2


@pytest.mark.parametrize("text", ["رَحْمَةًۭ ", "رَحْمَةًۭ\nكَذَٰلِكَ", "جُمْلَةًۭ وَٰحِدَةًۭ \n"])
def test_phonetize_mappings_survive_whitespace_around_the_rewrite(text):
    pytest.importorskip("quran_transcript")
    out = phonetize(text)
    assert len(out.mappings) == len(text)
    for i, char in enumerate(text):  # each ة is ت in wasl and ه at the waqf
        if char == "ة":
            assert out.phonemes[slice(*out.mappings[i].pos)] in ("ت", "ه")


@pytest.mark.parametrize(
    "text",
    [
        "حَامِيَةࣰ",  # U+08F0 open fathatan
        "حَامِيَةٗ",  # QPC's spelling of 88:4:3 (U+0657 for the tanween)
        "عَلِيمࣰا",  # upstream raises IndexError on this one, at any position
        "رَحْمَةࣰ كَذَٰلِكَ",  # and passes it through into the phonemes in wasl
        "رَحْمَةًۭ ۝",  # ayah-end marker
        "رَحْمَةًۭ ١",  # ayah number
    ],
)
def test_phonetize_rejects_text_outside_the_uthmani_alphabet(text):
    pytest.importorskip("quran_transcript")
    with pytest.raises(ValueError, match="outside quran-transcript's Uthmani alphabet"):
        phonetize(text)


def test_the_qpc_spelling_is_rejected_and_quran_transcripts_own_text_is_not():
    pytest.importorskip("quran_transcript")
    import json

    qpc = json.loads(
        (TOOLS_DIR.parent / "data" / "qpc-hafs-word-by-word.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError):
        phonetize(qpc["88:4:3"]["text"])
    phonetize(_ayah(88, 4))  # quran-transcript's own text is what callers pass


def _phonetizer_imports(path: Path) -> list[str]:
    """Where ``path`` reaches quran-transcript's phonetizer or moshaf without this module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "quran_transcript"
        ):
            names = {alias.name for alias in node.names}
            if node.module.startswith("quran_transcript.phonetics") or (
                "quran_phonetizer" in names
            ):
                hits.append(f"{path.name}:{node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr == "quran_phonetizer":
            hits.append(f"{path.name}:{node.lineno}")
    return hits


def test_hafs_phonetizer_is_the_only_way_to_the_phonetizer():
    entry_point = TOOLS_DIR / "hafs_phonetizer.py"
    sources = sorted(p for p in TOOLS_DIR.rglob("*.py") if p != entry_point)
    assert len(sources) > 50  # the scan is really walking tools/
    assert _phonetizer_imports(entry_point)  # and really detects an import
    offenders = [hit for path in sources for hit in _phonetizer_imports(path)]
    assert offenders == []
