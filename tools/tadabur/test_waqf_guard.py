"""Ship A's waqf guard (#134): which items depend on wasl, through the real phonetizer; the
owner's notes; the decisions; and the committed sidecar against its inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tadabur import waqf_guard as g
from tadabur.listening_session import Verdict, write_verdicts
from tadabur.local_pause import calibrate, fires
from tadabur.ship_a_worklist import WORKLIST_PATH
from tadabur.truth_sites import HELD, SHADDAH


def item(surah_ayah: str, start: int, end: int, word: int, needle: str, mark: str = SHADDAH,
         prescribed: str = HELD) -> g.Item:
    """A segment of words ``[start, end)`` of an ayah, and a site on ``needle``'s first
    character in ayah word ``word`` (``mark`` ``word``: the whole word)."""
    pytest.importorskip("quran_transcript")
    reference, offsets = g.realize(surah_ayah, start, end)
    index = reference.index(needle, offsets[word - start])
    assert index < offsets[word - start + 1]
    return g.Item(f"{surah_ayah}@{word}", ("test",), "new_audit", "test", mark, prescribed,
                  "clip.wav", surah_ayah, start, offsets, reference, index)


def dependences(one: g.Item) -> list[tuple[int, str, str, str]]:
    return [(d.word, d.side, d.wasl, d.waqf) for d in g.classify(one).dependences]


@pytest.mark.parametrize("surah_ayah, start, end, word, needle, waqf", [
    ("6:151", 16, 20, 18, "نن", "نَحنُ"),          # إِمْلَاقٍ نَّحْنُ: tanween idgham
    ("2:103", 7, 11, 9, "لل", "لَو"),              # خَيْرٌ لَّوْ: tanween idgham, no ghunna
    ("7:168", 2, 6, 4, "مم", "مِنهُمُ"),            # أُمَمًا مِّنْهُمُ: tanween into meem
    ("3:122", 0, 4, 2, "طط", "طَااااءِفَتَاانِ"),   # هَمَّت طَّائِفَتَانِ: assimilation
])
def test_a_geminate_made_by_joining_two_words_depends_on_wasl(surah_ayah, start, end, word,
                                                             needle, waqf):
    (found,) = dependences(item(surah_ayah, start, end, word, needle))
    assert found[:2] == (word, g.AFTER) and found[3] == waqf


@pytest.mark.parametrize("surah_ayah, start, end, word, needle", [
    ("30:2", 0, 2, 1, "رر"),     # غُلِبَتِ الرُّومُ: a sun letter keeps its shaddah
    ("2:158", 0, 3, 1, "صص"),    # إِنَّ الصَّفَا
    ("37:8", 0, 3, 1, "سس"),     # يَسَّمَّعُونَ: inside one word
])
def test_a_sun_letter_or_in_word_geminate_does_not(surah_ayah, start, end, word, needle):
    assert g.classify(item(surah_ayah, start, end, word, needle)).dependence == g.NOT_DEPENDENT


def test_the_word_before_the_boundary_is_classified_too():
    """فَزَعٍ يَوْمَئِذٍ: the tanween's geminate sits on the word before; شَفَاعَةٌ وَ: its ta
    marbuta is ``ت`` only in wasl (``ه`` at a pause, #130)."""
    (geminate,) = dependences(item("27:89", 7, 10, 8, "يي"))
    assert geminate == (9, g.BEFORE, "فَزَعِيي", "فَزَع")
    (letter,) = dependences(item("2:254", 16, 19, 17, "ت", mark="ت↔ط", prescribed="ت"))
    assert letter == (18, g.BEFORE, "شَفَااعَتُوو", "شَفَااعَه")


def test_an_assimilated_letter_is_a_partner_letter_only_in_wasl():
    """The first ط of هَمَّت طَّائِفَتَانِ is the ت of هَمَّت: ``ت`` again after a pause."""
    (found,) = dependences(item("3:122", 0, 4, 2, "ط", mark="ت↔ط", prescribed="ط"))
    assert found[:2] == (2, g.AFTER)
    unchanged = item("3:122", 0, 4, 2, "ف", mark="ت↔ط", prescribed="ط")  # any later letter
    assert g.classify(unchanged).dependence == g.NOT_DEPENDENT


def test_a_sampled_word_depends_on_wasl_where_its_word_changes():
    def word(index: int) -> g.Item:
        return item("6:151", 16, 20, index, "", mark=g.WORD, prescribed=g.WORD)

    assert [d[:2] for d in dependences(word(17))] == [(18, g.BEFORE)]  # إِمْلَاقٍ ends at a pause
    assert [d[:2] for d in dependences(word(18))] == [(18, g.AFTER), (19, g.BEFORE)]
    assert g.classify(word(19)).dependence == g.NOT_DEPENDENT  # the segment's last word


def test_a_waqf_the_phonetizer_cannot_realize_is_counted_not_guessed():
    """خَيْرٌ at a pause ends in a leen madd on a sukun, which quran-transcript refuses."""
    found = g.classify(item("2:103", 7, 11, 8, "خ", mark=g.WORD, prescribed=g.WORD))
    assert (found.dependence, found.unsupported) == (g.UNSUPPORTED, (9,))


def test_a_reference_that_is_not_its_words_is_refused():
    one = item("2:103", 7, 11, 9, "لل")
    with pytest.raises(ValueError, match="realization"):
        g.classify(g.Item(**{**one.__dict__, "word_start": 6}))


def test_survives_keeps_arrangement():
    assert g.survives("ررُۥۥم", "ءَررُۥۥم", (0, 1))
    assert not g.survives("للَو", "لَو", (0, 1))
    assert not g.survives("ططَاا", "طَاا", (0,))  # the first half has no counterpart
    assert g.survives("ططَاا", "طَاا", (1,))


@pytest.mark.parametrize("note, says", [
    ("waqf", True), ("Waqf", True), ("This is between  a pause", True), ("he stopped", True),
    ("wasl", False), ("no waqf here", False), ("No pause", False),
    ("", None), ("Not hafs", None), ("Incorrect trim", None),
])
def test_owner_notes(note, says):
    assert g.owner_says(note) is says


def _row(dependence=g.DEPENDENT, paused=(True,)):
    return {"item_id": "x", "dependence": dependence,
            "boundaries": [{"paused": p} for p in paused],
            "detector_paused": any(p for p in paused)}


def test_the_owner_overrides_the_detector_both_ways():
    assert g.resolve(_row(), None) == (g.WAQF_REALIZED, g.LOCAL_CHECK)
    assert g.resolve(_row(), Verdict("x", "held", "wasl")) == (g.SITE, g.OWNER_NOTE)
    assert g.resolve(_row(paused=(False,)), Verdict("x", "held", "waqf")) == (
        g.WAQF_REALIZED, g.OWNER_NOTE)
    assert g.resolve(_row(paused=(False,)), Verdict("x", "held", "")) == (g.SITE, g.LOCAL_CHECK)
    assert g.resolve(_row(paused=(None, False)), None) == (g.SITE, g.UNPLACED)


def test_only_a_dependent_item_is_ever_realized():
    for dependence in (g.NOT_DEPENDENT, g.UNSUPPORTED):
        assert g.resolve(_row(dependence), Verdict("x", "held", "waqf")) == (g.SITE, None)


def test_realized_ids_read_the_verdicts_at_every_use(tmp_path: Path):
    guard, verdicts = tmp_path / "guard.jsonl", tmp_path / "verdicts.jsonl"
    rows = [{**_row(paused=(False,)), "item_id": "a"}, {**_row(), "item_id": "b"}]
    guard.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert g.realized_ids(guard, verdicts) == {"b"}
    write_verdicts({"a": Verdict("a", "not_held", "Waqf")}, verdicts)
    assert g.realized_ids(guard, verdicts) == {"a", "b"}


# --- the committed sidecar -------------------------------------------------------------------


def test_the_committed_guard_reads_the_committed_inputs():
    summary = json.loads(g.SUMMARY_PATH.read_text(encoding="utf-8"))
    for name, path in (("ship_a/sites.jsonl", g.SITES_PATH),
                       ("ship_a/worklist.jsonl", WORKLIST_PATH),
                       ("ship_a/word_sample.jsonl", g.WORD_SAMPLE_PATH),
                       ("ship_a/waqf_guard.boundaries.jsonl", g.BOUNDARIES_PATH)):
        assert summary["inputs"][name] == g.sha256_of(path), name
    assert {r["item_id"] for r in g.read_guard()} == {i.item_id for i in g.all_items()}


def test_the_committed_decisions_are_the_operating_points():
    """The calibration is re-derived from the measured fixture boundaries, and every
    dependent boundary's decision is the local check at that operating point."""
    summary = json.loads(g.SUMMARY_PATH.read_text(encoding="utf-8"))
    boundaries = g.read_boundaries()
    fixtures = [b for b in boundaries.values() if b["kind"] == g.FIXTURE and b["window"]]
    assert summary["calibration"] == json.loads(json.dumps(calibrate(fixtures)))
    point = summary["calibration"]["operating_point"]
    for row in g.read_guard():
        for b in row["boundaries"]:
            measured = boundaries[b["key"]]
            expected = None if measured["window"] is None else fires(
                measured["scores"], point["threshold"], point["duration"])
            assert (b["window"], b["paused"]) == (measured["window"], expected)
        assert row["detector_paused"] == any(b["paused"] for b in row["boundaries"])
