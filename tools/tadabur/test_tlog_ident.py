"""Acceptance tests for tlog S5 ayah identification (``tadabur.tlog_ident``, #190).

Committed before the module exists and **locked**: later turns may add tests but never weaken or
rewrite these. The rules are "Ayah identification" in the frozen
``docs/tlog-holdout-preregistration.md`` plus S5's dated 2026-10-25 (S5, pre-outcome) amendment in
``docs/tlog-ident-preregistration.md``, which cites the frozen text by path and SHA-256, plus the
owner-approved post-look b_u erratum in ``docs/tlog-ident-bu-amendment.md`` and the post-look
margin-definition pin in ``docs/tlog-ident-margin-amendment.md``.

Contract this file pins
-----------------------
``tadabur.tlog_ident`` exposes:

* Parameters ``K`` (20), ``GRAM`` (3), ``GRID`` (0.00 … 0.50 by 0.01), ``FLOOR_BINS``
  (``0-20, 21-40, 41-80, 81-160, 161-``), ``FLOOR_PERCENTILE`` (99), ``RECALL_CLIPS`` (1000),
  ``RECALL_MIN`` (0.999), ``SALT``, ``BASE_MODEL``/``BASE_REVISION``, and the ``label_source``
  values ``FILENAME``, ``BOTH``, ``IDENTIFIED``, ``NONE``.
* ``AyahReference(surah, ayah, reference, offsets)``: a normalised, space-free per-ayah reference
  and its ``len(words) + 1`` word offsets.
* ``ReferenceDatabase.build(ayat)``: ``ayat`` maps ``(surah, ayah)`` to its reference; per surah
  ``text`` and ``decoy``, per ayah ``window``/``decoy_window``; ``text_class``, ``class_ayat``,
  ``class_label``, ``shortlist``, ``all_windows``, ``best``, ``decoy_best``,
  ``decoy_exhaustive_best``.
* ``Match(surah, ayah_from, word_from, ayah_to, word_to, text_class, score)`` with ``straddles``;
* ``Identification`` with ``length``, ``match``, ``decoy``,
  ``exhaustive_decoy``, ``filename_class``, ``filename_score``, ``delta``, ``margin``,
  ``margin_class``, ``floor_ok``, ``overlaps_filename``, ``label_source``, ``assigned_label``.
  ``floor_ok`` is the S5b recitation verdict (#204): ``length >= L_MIN`` and
  ``real - own decoy >= DELTA``, computed inside ``identify``.
* ``score``, ``reverse_clusters``, ``floor_bin``, ``identify``, ``relabelled``, ``assigned``,
  ``derive_floors``, ``derive_bars``, ``noise_pass_by_convention``, ``floor_rule``,
  ``oracle_consequence``, ``ayah_reference``, ``load_references``,
  ``load_look_decodes``, ``recall_sample``, ``load_decode_store``.
* The committed artifacts under ``tlog/ident/``: ``references.json.gz``,
  ``real_mistake_clips.jsonl``, ``ident_calibration.json``, the whole-clip base decodes, the
  S5b confirmation draws, the oracle pass and the generated ``docs/tlog-ayah-identification.md``.

Ordering contract: ``shortlist`` and ``all_windows`` return scored windows best score first,
ties by ``(surah, ayah)``. A span's words are quran-transcript ``uthmani.split()`` words, 1-based.
A matched run is split at ayah boundaries, and a span's text class is the ayah with the most
words touched by a matched or substituted character (ties: the earlier ayah). ``derive_bars``
derives the amended ``b_u`` (precision of the clips it names) from its per-bar ``precision``
table and records the literal rule's ``b_u_literal`` for transparency.
Synthetic references here avoid adjacent equal cores, so ``normalize_phonemes`` (which ``identify``
applies to the decode) leaves them unchanged.

Test command, from ``tools/``::

    python -m pytest -q tadabur/test_tlog_ident.py
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TARTEEL_DIR = ROOT / "tools" / "tadabur" / "nonpro_eval" / "tarteel_v1"
SPLIT_PATH = TARTEEL_DIR / "split.json"
NOISE_POOL = ROOT / "tools" / "tadabur" / "nonpro_gap" / "noise_pool.json"
TRUTH_SITES = ROOT / "tools" / "tadabur" / "truth_sites" / "ship_a_verified.jsonl"
PREREGISTRATION = ROOT / "docs" / "tlog-ident-preregistration.md"
HOLDOUT_PREREGISTRATION = ROOT / "docs" / "tlog-holdout-preregistration.md"
BU_AMENDMENT = ROOT / "docs" / "tlog-ident-bu-amendment.md"
MARGIN_AMENDMENT = ROOT / "docs" / "tlog-ident-margin-amendment.md"
FLOOR_PREREGISTRATION = ROOT / "docs" / "tlog-ident-floor-preregistration.md"
#: The frozen hold-out text the S5 amendment cites; it is never edited (S3b's meta.json hashes it).
HOLDOUT_SHA256 = "204fe7f980488aff73402c56f1729f7f17f7c9471ea0664f6365bb8870b62086"
#: The S5 pre-registration is frozen too: the owner's b_u erratum cites this hash, never edits it.
S5_PREREGISTRATION_SHA256 = (
    "a383fcfdbd375c9e6c052d8032788a8eda69bd2d9fbd12fcd3b921b1fbb6a3ab")
BASE_REVISION = "01a1ef9fbe40d144ef845101e89ff924aed3fef5"


@pytest.fixture(scope="session")
def ident():
    return importlib.import_module("tadabur.tlog_ident")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- synthetic reference databases ------------------------------------------------------------


def _ayah(ident, surah: int, ayah: int, words: list[str]):
    """One reference over ``words``: their concatenation, with cumulative word offsets."""
    offsets = [0]
    for word in words:
        offsets.append(offsets[-1] + len(word))
    return ident.AyahReference(surah, ayah, "".join(words), tuple(offsets))


def _distinct_references(count: int) -> list[str]:
    """``count`` distinct four-letter references with no adjacent equal cores.

    Four consecutive letters of the alphabet (wrapping), with the fourth shifted by one for
    each further lap: distinct, adjacent characters always differ, and no two references
    repeat.
    """
    letters = "بتثجحخدذرزسشصضطظعغفقكلمنهوي"
    seen: list[str] = []
    for index in range(count):
        start, lap = index % len(letters), index // len(letters)
        positions = (start, start + 1, start + 2, start + 3 + lap)
        ref = "".join(letters[position % len(letters)] for position in positions)
        assert all(a != b for a, b in itertools.pairwise(ref)), ref
        assert ref not in seen
        seen.append(ref)
    return seen


@pytest.fixture
def two_ayat(ident):
    """Surah 1: 1:1 ``بجحد``, 1:2 ``خدذر`` (two words each)."""
    return ident.ReferenceDatabase.build([
        _ayah(ident, 1, 1, ["بج", "حد"]),
        _ayah(ident, 1, 2, ["خد", "ذر"]),
    ])


@pytest.fixture
def two_surahs(ident):
    """Surah 1: 1:1 ``بجح``; surah 2: 2:1 ``خدذ``."""
    return ident.ReferenceDatabase.build([
        _ayah(ident, 1, 1, ["بجح"]),
        _ayah(ident, 2, 1, ["خدذ"]),
    ])


def _overlaps(first, second) -> bool:
    """Whether two word spans share a word (the test's own reading of the rule)."""
    if first.surah != second.surah:
        return False
    first_from = (first.ayah_from, first.word_from)
    first_to = (first.ayah_to, first.word_to)
    second_from = (second.ayah_from, second.word_from)
    second_to = (second.ayah_to, second.word_to)
    return first_from <= second_to and second_from <= first_to


# --- the parameters ---------------------------------------------------------------------------


def test_the_parameters_are_the_pre_registered_ones(ident):
    assert ident.K == 20
    assert ident.GRAM == 3
    assert list(ident.GRID) == [round(0.01 * index, 2) for index in range(51)]
    assert ident.GRID[0] == 0.0 and ident.GRID[-1] == 0.5
    assert tuple(ident.FLOOR_BINS) == ("0-20", "21-40", "41-80", "81-160", "161-")
    assert ident.FLOOR_PERCENTILE == 99.0
    assert (ident.RECALL_CLIPS, ident.RECALL_MIN) == (1000, 0.999)
    assert (ident.FILENAME, ident.BOTH, ident.IDENTIFIED, ident.NONE) == (
        "filename", "both", "identified", "none")
    assert (ident.BASE_MODEL, ident.BASE_REVISION) == (
        "obadx/muaalem-model-v3_2", BASE_REVISION)
    # S5b (#204): the recitation rule is fixed in its own dated pre-registration.
    assert (ident.L_MIN, ident.DELTA) == (8, pytest.approx(0.30))
    assert ident.FLOOR_SALT == "tlog-s5b-floor-confirm-v1"
    assert ident.FLOOR_PREREGISTRATION_PATH == FLOOR_PREREGISTRATION


# --- the reference database: text classes, spans, decoys --------------------------------------


def test_a_repeated_reference_is_one_text_class_and_never_disagrees(ident):
    db = ident.ReferenceDatabase.build([
        _ayah(ident, 1, 1, ["بجح"]),
        _ayah(ident, 1, 2, ["بجح"]),
    ])
    first, second = db.text_class(1, 1), db.text_class(1, 2)
    assert first == second
    assert db.class_ayat(first) == ((1, 1), (1, 2))
    assert db.class_label(first) == "1:1"

    decoded = ident.identify("بجح", database=db, filename_ayah="1:2",
                             bars={"b": 0.0, "b_u": 0.0})
    assert decoded.filename_class == first
    assert decoded.match.text_class == first
    assert decoded.delta == 0.0
    assert decoded.label_source == ident.BOTH
    assert decoded.assigned_label == "1:2"
    # The best class is the filename's own class: no disagreement, so no relabel even at b = 0.
    assert decoded.overlaps_filename is False
    assert not ident.relabelled(decoded, 0.0)


def test_the_span_covers_two_ayat_and_straddles_them(ident, two_ayat):
    decoded = ident.identify("بجحدخدذر", database=two_ayat)
    match = decoded.match
    assert (match.surah, match.ayah_from, match.word_from) == (1, 1, 1)
    assert (match.ayah_to, match.word_to) == (2, 2)
    assert match.straddles is True
    assert match.score == pytest.approx(1.0)
    # Both ayat are fully aligned: the tie goes to the earlier ayah.
    assert match.text_class == two_ayat.text_class(1, 1)


def test_a_single_ayah_decode_does_not_straddle(ident, two_ayat):
    match = ident.identify("بجحد", database=two_ayat).match
    assert (match.ayah_from, match.word_from, match.ayah_to, match.word_to) == (1, 1, 1, 2)
    assert match.straddles is False


def test_a_run_crossing_a_boundary_credits_the_ayah_with_most_aligned_words(ident):
    """One aligned character in 1:1 and 46 in 1:2: the run is split, and 1:2 wins (2 words
    against 1). Attributing the whole run to the ayah it starts in would pick 1:1."""
    letters = "بتثجحخدذرزسشصضطظعغفقكلمنهوي"
    tail = "".join(letters[(index + 1) % len(letters)] for index in range(46))
    db = ident.ReferenceDatabase.build([
        _ayah(ident, 1, 1, ["ب"]),
        _ayah(ident, 1, 2, [tail[:23], tail[23:]]),
    ])
    match = ident.identify("ب" + tail, database=db).match
    assert match.score == pytest.approx(1.0)
    assert (match.ayah_from, match.word_from) == (1, 1)
    assert (match.ayah_to, match.word_to) == (2, 2)
    assert match.straddles is True
    assert match.text_class == db.text_class(1, 2)


def test_attribution_counts_words_not_characters(ident):
    """1:1 aligns four characters in one word; 1:2 three characters over three words. The
    ayah with the most aligned words wins, so the single-character words take it."""
    db = ident.ReferenceDatabase.build([
        _ayah(ident, 1, 1, ["بجحد"]),
        _ayah(ident, 1, 2, ["ز", "س", "ش"]),
    ])
    match = ident.identify("بجحدزسش", database=db).match
    assert match.score == pytest.approx(1.0)
    assert (match.ayah_from, match.word_from, match.ayah_to, match.word_to) == (1, 1, 2, 3)
    assert match.text_class == db.text_class(1, 2)


def test_reverse_clusters_reverses_a_base_with_its_marks(ident):
    # A fatha belongs to the base before it, so the cluster moves together: بَ ت ب -> ب ت بَ
    assert ident.reverse_clusters("بَتب") == "بتبَ"
    assert ident.reverse_clusters("بتب") == "بتب"[::-1]


def test_the_decoy_sequence_reverses_each_ayah_and_keeps_ayah_order(ident, two_ayat):
    assert two_ayat.text(1) == "بجحدخدذر"
    assert list(two_ayat.decoy(1)) == list(
        ident.reverse_clusters("بجحد") + ident.reverse_clusters("خدذر"))
    assert two_ayat.decoy_window(1, 1) == two_ayat.decoy(1)
    assert two_ayat.decoy_window(1, 2) == two_ayat.decoy(1)


# --- Δ and the margin -------------------------------------------------------------------------


def _rule_passes(monkeypatch, database):
    """The margin fixtures test the margin, not the decoy: pin the decoy best to 0 so the S5b
    rule passes whenever the real score clears DELTA (every fixture here scores at least 0.875).
    Patched on the instance so the search under test is unchanged."""
    monkeypatch.setattr(database, "decoy_best", lambda decode, *, k=None: 0.0)
    return database


def test_delta_is_measured_against_the_filename_ayahs_own_window(ident, monkeypatch):
    # The relabel rule needs the S5b verdict; the decoy is pinned so only the margin matters.
    db = _rule_passes(monkeypatch, ident.ReferenceDatabase.build([
        _ayah(ident, 1, 1, ["بجح"]), _ayah(ident, 2, 1, ["بتث", "جحخ", "دذر"])]))
    decoded = ident.identify("بتثجحخدذر", database=db, filename_ayah="1:1",
                             bars={"b": 0.5, "b_u": 0.5})
    assert decoded.match.score == pytest.approx(1.0)
    assert decoded.match.text_class == db.text_class(2, 1)
    assert decoded.filename_class == db.text_class(1, 1)
    assert decoded.filename_score == ident.score("بتثجحخدذر", db.window(1, 1))
    assert decoded.delta == pytest.approx(decoded.match.score - decoded.filename_score)
    assert decoded.delta >= 0.5  # the bar the fixture relabels at
    assert decoded.overlaps_filename is False
    assert decoded.floor_ok is True
    assert ident.relabelled(decoded, 0.5) is True
    assert decoded.label_source == ident.IDENTIFIED
    assert decoded.assigned_label == "2:1"


def test_the_margin_is_the_best_different_class_at_a_non_overlapping_location(ident):
    """Surah 1: 1:1 ``بجح``, 1:2 ``خدذ``; surah 2: 2:1 ``زرز``. The decode is 1:1's text. The
    same-class 1:2 window also scores 1.0 and is skipped; the runner-up is 2:1's class."""
    db = ident.ReferenceDatabase.build([
        _ayah(ident, 1, 1, ["بجح"]),
        _ayah(ident, 1, 2, ["خدذ"]),
        _ayah(ident, 2, 1, ["زرز"]),
    ])
    decoded = ident.identify("بجح", database=db)
    best = db.best("بجح").match
    others = [window.match for window in db.all_windows("بجح")
              if window.match.text_class != best.text_class
              and not _overlaps(window.match, best)]
    assert others  # the fixture has a different class at a non-overlapping location
    assert max(window.score for window in db.all_windows("بجح")) == pytest.approx(1.0)
    assert decoded.margin == pytest.approx(best.score - max(match.score for match in others))
    assert decoded.margin_class == max(others, key=lambda match: match.score).text_class
    assert decoded.margin_class == db.text_class(2, 1)


def test_adjacent_ayat_that_both_match_make_the_margin_zero(ident):
    """Both 1:1 and 1:2 match "بجح" perfectly at non-overlapping locations; the winning
    window's other ayah is a competitor, so an unlabelled row stays unnamed."""
    db = ident.ReferenceDatabase.build([
        ident.AyahReference(1, 1, "بجح", (0, 3)),
        ident.AyahReference(1, 2, "بجحد", (0, 4)),
    ])
    decoded = ident.identify("بجح", database=db,
                             bars={"b": 0.01, "b_u": 0.01})
    assert decoded.match.score == pytest.approx(1.0)
    assert decoded.margin == pytest.approx(0.0)
    assert decoded.margin_class == db.text_class(1, 2)
    assert decoded.label_source == ident.NONE
    assert decoded.assigned_label is None


def test_a_competitor_straddling_an_ayah_boundary_sets_the_margin(ident):
    """The runner-up starts mid-ayah and runs on into the next one, so it is one location.

    Splitting the window's free characters at the boundary loses it; rebasing the sliced ayah
    to 0 would restart it at word 1 and make it overlap the winner. Either error yields
    ``margin = 0.5`` and names the row instead of leaving it unnamed.
    """
    db = ident.ReferenceDatabase.build([
        ident.AyahReference(1, 1, "بجحدبج", (0, 4, 6)),
        ident.AyahReference(1, 2, "حد", (0, 1, 2)),
    ])
    shortlisted = db.shortlist("بجحد")
    best = shortlisted[0].match
    assert (best.ayah_from, best.word_from, best.ayah_to, best.word_to) == (1, 1, 1, 1)
    competitor = db.competitor("بجحد", shortlisted, best)
    assert (competitor.ayah_from, competitor.word_from) == (1, 2)
    assert (competitor.ayah_to, competitor.word_to) == (2, 2)
    assert competitor.text_class == db.text_class(1, 2)
    assert competitor.score == pytest.approx(1.0)

    decoded = ident.identify("بجحد", database=db,
                             bars={"b": 0.01, "b_u": 0.01})
    assert decoded.margin == pytest.approx(0.0)
    assert decoded.margin_class == db.text_class(1, 2)
    assert decoded.label_source == ident.NONE
    assert decoded.assigned_label is None


def test_a_joined_range_never_hides_the_different_class_beside_it(ident):
    """The joined range's single best alignment is the best class (1:2 repeats 1:1), so the
    joined search alone would drop 1:3 and report an uncontradicted margin of 1.0. The separate
    ayah slice must still be searched."""
    db = ident.ReferenceDatabase.build([
        ident.AyahReference(1, 1, "بجح", (0, 3)),
        ident.AyahReference(1, 2, "بجح", (0, 3)),
        ident.AyahReference(1, 3, "بجحخ", (0, 4)),
    ])
    shortlisted = db.shortlist("بجح")
    best = shortlisted[0].match
    competitor = db.competitor("بجح", shortlisted, best)
    assert competitor.text_class == db.text_class(1, 3)
    assert (competitor.ayah_from, competitor.word_from) == (3, 1)
    assert competitor.score == pytest.approx(1.0)

    decoded = ident.identify("بجح", database=db,
                             bars={"b": 0.01, "b_u": 0.01})
    assert decoded.margin == pytest.approx(0.0)
    assert decoded.margin_class == db.text_class(1, 3)
    assert decoded.label_source == ident.NONE


def test_a_tied_different_class_location_sets_the_margin(ident):
    """R3.1: the window centred at 1:3 holds two perfect locations. The first is 1:2, which
    repeats the winner's class; the second is 1:2 word 2 through 1:3 word 4, whose class is
    1:3. The margin is the difference against that second location, so it is zero."""
    db = ident.ReferenceDatabase.build([
        ident.AyahReference(1, 1, "بتثجبتثج", (0, 4, 8)),
        ident.AyahReference(1, 2, "بتثجبتثج", (0, 4, 8)),
        ident.AyahReference(1, 3, "بتثج", (0, 1, 2, 3, 4)),
    ])
    decoded = ident.identify("بتثجبتثج", database=db,
                             bars={"b": 0.01, "b_u": 0.01})
    assert decoded.match.score == pytest.approx(1.0)
    assert decoded.margin == pytest.approx(0.0)
    assert decoded.margin_class == db.text_class(1, 3)
    assert decoded.label_source == ident.NONE
    assert decoded.assigned_label is None


def test_a_lower_scoring_straddling_location_sets_the_margin(ident, monkeypatch):
    """R3.1: only a worse straddling location exists beside the same-class optimum. It scores
    0.875 (one substitution over eight characters), so the margin is 0.125, not the 0.625 the
    single-optimum search reports. The row is still named at the provisional b_u = 0.01."""
    db = _rule_passes(monkeypatch, ident.ReferenceDatabase.build([
        ident.AyahReference(1, 1, "بتثجبتثج", (0, 4, 8)),
        ident.AyahReference(1, 2, "بتثجبتثج", (0, 4, 8)),
        ident.AyahReference(1, 3, "بتثح", (0, 1, 2, 3, 4)),
    ]))
    shortlisted = db.shortlist("بتثجبتثج")
    best = shortlisted[0].match
    competitor = db.competitor("بتثجبتثج", shortlisted, best)
    assert competitor.text_class == db.text_class(1, 3)
    assert competitor.score == pytest.approx(0.875)

    decoded = ident.identify("بتثجبتثج", database=db,
                             bars={"b": 0.01, "b_u": 0.01})
    assert decoded.margin == pytest.approx(0.125)
    assert decoded.margin_class == db.text_class(1, 3)
    assert decoded.floor_ok is True
    assert decoded.label_source == ident.IDENTIFIED


def test_the_pinned_margin_misses_a_location_inside_a_shared_word(ident, monkeypatch):
    """R4.1 under the pin: the different-class location starts inside the single word a
    same-class optimum also uses, so no word span reaches it. ``docs/tlog-ident-margin-
    amendment.md`` records the case and its one-way direction — the margin is overstated (0.5
    here, where a character-level search gives 0), so the row is named, never left unnamed."""
    db = _rule_passes(monkeypatch, ident.ReferenceDatabase.build([
        ident.AyahReference(1, 1, "بتثجبتثج", (0, 8)),
        ident.AyahReference(1, 2, "بتثجبتثج", (0, 8)),
        ident.AyahReference(1, 3, "بتثج", (0, 1, 2, 3, 4)),
    ]))
    decoded = ident.identify("بتثجبتثج", database=db,
                             bars={"b": 0.01, "b_u": 0.01})
    assert decoded.match.score == pytest.approx(1.0)
    assert decoded.margin == pytest.approx(0.5)
    assert decoded.margin_class == db.text_class(1, 3)
    assert decoded.label_source == ident.IDENTIFIED
    assert decoded.assigned_label == "1:1"


def test_the_pinned_margin_misses_a_worse_location_inside_a_shared_word(ident, monkeypatch):
    """R4.1's second fixture: the only different-class location starts inside the shared word
    and scores 0.875, so a character-level search gives m ≤ 0.125. The pin reports 0.625 and
    names the row even at b_u = 0.50."""
    db = _rule_passes(monkeypatch, ident.ReferenceDatabase.build([
        ident.AyahReference(1, 1, "بتثجبتثج", (0, 8)),
        ident.AyahReference(1, 2, "بتثجبتثج", (0, 8)),
        ident.AyahReference(1, 3, "بتثح", (0, 1, 2, 3, 4)),
    ]))
    decoded = ident.identify("بتثجبتثج", database=db,
                             bars={"b": 0.01, "b_u": 0.50})
    assert decoded.margin == pytest.approx(0.625)
    assert decoded.margin_class == db.text_class(1, 3)
    assert decoded.label_source == ident.IDENTIFIED
    assert decoded.assigned_label == "1:1"


@pytest.fixture
def repeated_classes(ident):
    """1:1 and 1:2 share a class; 1:3 opens with their text but is its own class; 1:4 is
    unrelated. Enough shared text for a same-class optimum to sit beside a competitor."""
    ayat = {
        (1, 1): ident.AyahReference(1, 1, "بجحدبج", (0, 4, 6)),
        (1, 2): ident.AyahReference(1, 2, "بجحدبج", (0, 4, 6)),
        (1, 3): ident.AyahReference(1, 3, "بجحدخ", (0, 4, 5)),
        (1, 4): ident.AyahReference(1, 4, "خدذرز", (0, 2, 4, 6)),
    }
    return ayat, ident.ReferenceDatabase.build(ayat.values())


def _word_span_oracle(ident, ayat, db, decode):
    """The best different-class span over *every* word span of every shortlisted window.

    A word-span oracle, not a location-level one: it shares the pinned restriction of
    ``docs/tlog-ident-margin-amendment.md`` (a location inside a shared word is unreachable),
    and its point is that the engine's bounded search visits no fewer spans. The oracle owns the
    enumeration (no bound, no early exit) and the ranking; slicing, attribution and distance are
    the engine's, which the locked attribution tests pin.
    """
    shortlisted = db.shortlist(decode)
    best = shortlisted[0].match

    def rank(row, winner):
        return (row.score, winner, -row.surah, -row.ayah_from, -row.word_from,
                -row.ayah_to, -row.word_to)

    found = None
    spans_scored = 0
    for window in shortlisted:
        if window.match.text_class != best.text_class and not _overlaps(window.match, best):
            if found is None or rank(window.match, True) > found[0]:
                found = (rank(window.match, True), window.match)
        candidate = db._candidate(window.surah, window.ayah, decoy=False)
        spans: list[tuple[int, int, bool]] = []
        cursor = 0
        for other in (window.ayah - 1, window.ayah, window.ayah + 1):
            reference = ayat.get((window.surah, other))
            if reference is None:
                continue
            for word in range(1, reference.words + 1):
                spans.append((
                    cursor + reference.offsets[word - 1], cursor + reference.offsets[word],
                    best.surah == reference.surah
                    and best.ayah_from <= other <= best.ayah_to
                    and (other != best.ayah_from or word >= best.word_from)
                    and (other != best.ayah_to or word <= best.word_to)))
            cursor += len(reference.reference)
        runs: list[list[tuple[int, int]]] = []
        run: list[tuple[int, int]] = []
        for lo, hi, inside in spans:
            if inside:
                if run:
                    runs.append(run)
                    run = []
            else:
                run.append((lo, hi))
        if run:
            runs.append(run)
        for run in runs:
            for first in range(len(run)):
                for last in range(first, len(run)):
                    match = db._match(decode, db._sub_candidate(
                        candidate, run[first][0], run[last][1]))
                    spans_scored += 1
                    if match.text_class == best.text_class:
                        continue
                    if found is None or rank(match, False) > found[0]:
                        found = (rank(match, False), match)
    return best, (None if found is None else found[1]), spans_scored


def test_the_bounded_search_visits_no_fewer_spans_than_the_word_span_oracle(ident,
                                                                           repeated_classes):
    """The bounded search must reach the word-span oracle's margin and class on every decode."""
    ayat, db = repeated_classes
    for decode in ("بجحدبج", "بجحد", "بجح", "خدذ"):
        best, competitor, spans = _word_span_oracle(ident, ayat, db, decode)
        assert spans > 1  # the enumeration is more than the engine's few probes
        decoded = ident.identify(decode, database=db)
        assert decoded.match == best
        assert decoded.margin == pytest.approx(
            best.score - (competitor.score if competitor is not None else 0.0))
        assert decoded.margin_class == (competitor.text_class if competitor is not None
                                        else None)


def test_identification_record_is_the_committed_row_contract(ident, rule_database):
    """S6 and S7 must not invent two row shapes: one key set, the issue's field names, and the
    rule that produced the verdict on every row (architect r19: ``floor_rule``)."""
    decoded = ident.identify("طظعغفقكل", database=rule_database, filename_ayah="1:1",
                             bars={"b": 0.01, "b_u": 0.01})
    record = ident.identification_record(decoded, decode_mode="whole_clip")
    assert set(record) == {
        "surah", "ayah_from", "word_from", "ayah_to", "word_to", "straddles", "score",
        "length", "class_label", "decoy", "exhaustive_decoy", "delta", "margin",
        "margin_class", "id_floor_ok", "overlaps_filename", "filename_label",
        "label_source", "assigned_label", "floor_rule", "decode_mode"}
    assert record["floor_rule"] == ident.floor_rule()
    assert record["floor_rule"]["sha256"] == _sha256(FLOOR_PREREGISTRATION)
    assert (record["surah"], record["ayah_from"], record["word_from"]) == (2, 1, 1)
    assert record["straddles"] is False
    assert record["class_label"] == "2:1"
    assert record["id_floor_ok"] is True
    assert record["label_source"] == ident.IDENTIFIED
    assert record["assigned_label"] == "2:1"
    assert json.loads(json.dumps(record, ensure_ascii=False)) == record

    # A row with no match keeps every key: it is non_recitation, never "unverified".
    empty = ident.identification_record(ident.identify("", database=rule_database),
                                       decode_mode="whole_clip")
    assert set(empty) == set(record)
    assert (empty["surah"], empty["score"], empty["straddles"]) == (None, None, None)
    assert empty["id_floor_ok"] is False
    assert empty["floor_rule"] == record["floor_rule"]


def test_every_row_carries_margin_and_margin_class(ident, rule_database):
    """The margin is on every identification row, including one with no match, and the rule
    verdict never drops it."""
    decoded = ident.identify("بجحدذرزسط", database=rule_database)
    assert decoded.floor_ok is True
    assert decoded.margin is not None
    assert decoded.margin_class == rule_database.text_class(2, 1)

    empty = ident.identify("", database=rule_database)
    assert (empty.match, empty.margin, empty.margin_class) == (None, None, None)

    short = ident.identify("بجحدذر", database=rule_database)
    assert short.floor_ok is False
    assert short.margin is not None and short.margin_class is not None


# --- the shortlist ----------------------------------------------------------------------------


def test_the_shortlist_never_misses_the_exhaustive_best_on_a_fixture(ident):
    references = _distinct_references(60)
    db = ident.ReferenceDatabase.build([
        _ayah(ident, 1 + index // 30, 1 + index % 30, [reference])
        for index, reference in enumerate(references)
    ])
    for reference in references[:10]:
        shortlisted = db.shortlist(reference)
        assert 0 < len(shortlisted) <= ident.K
        exhaustive = max(window.score for window in db.all_windows(reference))
        assert shortlisted[0].score == pytest.approx(exhaustive)
        assert shortlisted[0].match.score == pytest.approx(1.0)


# --- the in-ayah mistake fixture ---------------------------------------------------------------


def test_an_in_ayah_mistake_keeps_its_label(ident, rule_database):
    """A dropped letter inside the right ayah keeps the filename label; the eight-character
    decode also clears the S5b rule, so the verdict is what the mistake bar reads."""
    decode = "بجحدذزسط"  # 1:1 without word 2's final ر: eight characters, one deletion
    decoded = ident.identify(decode, database=rule_database, filename_ayah="1:1",
                             bars={"b": 0.01, "b_u": 0.01})
    assert decoded.length == ident.L_MIN
    assert decoded.match.text_class == decoded.filename_class == rule_database.text_class(1, 1)
    assert decoded.match.ayah_from == decoded.match.ayah_to == 1
    assert decoded.floor_ok is True
    assert decoded.label_source == ident.BOTH
    assert decoded.assigned_label == "1:1"
    assert not ident.relabelled(decoded, 0.01)


# --- floors and the bars ----------------------------------------------------------------------


def test_the_floor_bins_are_the_pre_registered_length_bins(ident):
    assert [ident.floor_bin(length) for length in (0, 20, 21, 40, 41, 80, 81, 160, 161, 400)] == [
        "0-20", "0-20", "21-40", "21-40", "41-80", "41-80", "81-160", "81-160", "161-", "161-"]


@pytest.fixture
def rule_database(ident):
    """Two ayat, neither a decoy of the other: 1:1 ``بجحدذرزسط``, 2:1 ``طظعغفقكلم``."""
    return ident.ReferenceDatabase.build([
        _ayah(ident, 1, 1, ["بجح", "دذر", "زسط"]),
        _ayah(ident, 2, 1, ["طظع", "غفق", "كلم"]),
    ])


def test_a_decode_under_l_min_is_non_recitation_however_wide_its_margin(ident, rule_database):
    """S5b (#204): the rule is ``len >= L_MIN`` and ``real - own decoy >= DELTA``; the binned
    floor is retired for verdicts. Six characters fail on length even at Δ = 0.57."""
    decoded = ident.identify("بجحدذر", database=rule_database)
    assert decoded.length == 6 < ident.L_MIN
    assert decoded.match.score - decoded.decoy > ident.DELTA
    assert decoded.floor_ok is False
    assert decoded.label_source == ident.NONE

    empty = ident.identify("", database=rule_database)
    assert empty.length == 0 and empty.match is None
    assert empty.floor_ok is False


def test_a_decode_matching_its_own_decoy_is_non_recitation(ident):
    """The failure mode the rule exists for: eight characters that match the reversed ayah
    better than the real one (real 0.125, own decoy 1.0)."""
    database = ident.ReferenceDatabase.build([_ayah(ident, 1, 1, ["بتثجحخدذر"])])
    decoded = ident.identify("رذدخحجثت", database=database)
    assert decoded.length == 8 >= ident.L_MIN
    assert decoded.match.score == pytest.approx(0.125)
    assert decoded.decoy == pytest.approx(1.0)
    assert decoded.match.score - decoded.decoy < ident.DELTA
    assert decoded.floor_ok is False


def test_the_delta_boundary_is_strict(ident, rule_database, monkeypatch):
    """``< DELTA`` is non-recitation, so exactly DELTA passes."""
    score = ident.identify("بجحدذرزسط", database=rule_database).match.score
    for offset, expected in ((ident.DELTA, True), (ident.DELTA - 1e-9, False)):
        monkeypatch.setattr(rule_database, "decoy_best", lambda decode, *, k=ident.K,
                            _v=score - offset: _v)
        assert ident.identify("بجحدذرزسط", database=rule_database).floor_ok is expected


def test_real_and_decoy_scores_come_from_the_same_shortlist(ident, rule_database, monkeypatch):
    """The decoy best is the same decode's own search at the same k, never a second ranking."""
    calls: list[int] = []

    def spy(decode: str, *, k: int = ident.K) -> float:
        calls.append(k)
        return 0.1

    monkeypatch.setattr(rule_database, "decoy_best", spy)
    decoded = ident.identify("بجحدذرزسط", database=rule_database, k=7)
    assert calls == [7]
    assert decoded.decoy == 0.1
    assert decoded.floor_ok is (decoded.match.score - decoded.decoy >= ident.DELTA)


def test_a_non_recitation_row_is_never_relabelled_or_assigned(ident):
    """The verdict gates both rules: a below-rule row keeps its filename label and is never
    assigned."""
    failed = _identification(ident, floor_ok=False, delta=0.5, margin=0.5)
    assert not ident.relabelled(failed, 0.0)
    assert not ident.assigned(failed, 0.0)

    bars = {"b": 0.01, "b_u": 0.01}
    database = ident.ReferenceDatabase.build([
        _ayah(ident, 1, 1, ["بجح"]), _ayah(ident, 2, 1, ["خدذ"])])
    unlabelled = ident.identify("خدذ", database=database, bars=bars)  # three characters
    assert unlabelled.margin >= bars["b_u"]  # would be assigned without the rule
    assert unlabelled.label_source == ident.NONE
    assert unlabelled.assigned_label is None

    labelled = ident.identify("خدذ", database=database, filename_ayah="1:1", bars=bars)
    assert labelled.delta >= bars["b"]  # would be relabelled without the rule
    assert not ident.relabelled(labelled, bars["b"])
    assert labelled.label_source == ident.FILENAME
    assert labelled.assigned_label == "1:1"


def test_noise_pass_is_reported_under_the_three_empty_bin_conventions(ident):
    """A: an empty bin's rows fail; B: back off to the nearest lower calibrated bin (81-160);
    C: exclude them. Empty decodes never pass, and stay in the denominator."""
    def row(length: int, score: float | None):
        match = (None if score is None else
                 ident.Match(surah=1, ayah_from=1, word_from=1, ayah_to=1, word_to=1,
                             text_class=0, score=score))
        return ident.Identification(
            length=length, match=match, decoy=0.0, exhaustive_decoy=None,
            filename_class=None, filename_score=None, delta=None, margin=None,
            margin_class=None, floor_ok=None, overlaps_filename=None,
            label_source=ident.NONE, assigned_label=None)

    rows = [row(10, 0.9), row(10, 0.9), row(10, None),
            row(200, 0.9), row(200, 0.9), row(200, 0.2),
            row(100, 0.3), row(100, 0.5)]
    floors = {"0-20": 0.5, "81-160": 0.4}
    conventions = ident.noise_pass_by_convention(rows, floors)
    assert conventions["A"]["recordings"] == 8
    assert conventions["A"]["passed"] == 3  # two 0-20 hits and the 0.5 in 81-160
    assert conventions["A"]["pass"] == pytest.approx(3 / 8)
    assert conventions["B"]["recordings"] == 8
    assert conventions["B"]["passed"] == 5  # 161- rows are judged against 81-160's floor
    assert conventions["B"]["pass"] == pytest.approx(5 / 8)
    assert conventions["C"]["recordings"] == 5  # the three 161- rows are excluded
    assert conventions["C"]["passed"] == 3
    assert conventions["C"]["pass"] == pytest.approx(3 / 5)


def _identification(ident, *, match_class: int = 0, **changes):
    match = ident.Match(surah=1, ayah_from=1, word_from=1, ayah_to=1, word_to=1,
                        text_class=match_class, score=1.0)
    fields = dict(length=10, match=match, decoy=0.05, exhaustive_decoy=None,
                  filename_class=1, filename_score=1.0, delta=0.0, margin=0.25, margin_class=1,
                  floor_ok=True, overlaps_filename=False, label_source=ident.FILENAME,
                  assigned_label="1:1")
    fields.update(changes)
    return ident.Identification(**fields)


def test_derive_floors_is_the_type_7_99th_percentile_per_bin(ident):
    rows = [_identification(ident, length=10, decoy=value / 99) for value in range(100)]
    rows += [_identification(ident, length=200, decoy=0.5)]
    floors = ident.derive_floors(rows)
    assert floors["0-20"] == pytest.approx(98.01 / 99, abs=1e-12)
    assert floors["161-"] == pytest.approx(0.5)


def _look_population(ident):
    """985 own-class rows (955 from margin 0.02, 30 from 0.30), five relabel-eligible, ten
    never above the floor."""
    return (
        [_identification(ident, match_class=1, margin=0.02) for _ in range(955)]
        + [_identification(ident, match_class=2, margin=0.02, delta=0.10) for _ in range(5)]
        + [_identification(ident, match_class=1, margin=0.30) for _ in range(30)]
        + [_identification(ident, match_class=1, floor_ok=False, margin=None,
                           margin_class=None) for _ in range(10)]
    )


def test_derive_bars_b_u_is_the_amended_precision_rule(ident):
    """Owner-approved erratum (docs/tlog-ident-bu-amendment.md): b_u is the smallest bar at
    which >= 99% of the clips the identifier NAMES are their own class. The literal
    share-of-all result is still recorded alongside it for transparency."""
    # The one mistake clip's delta is 0.05, so b is the first grid value above it (0.06);
    # exactly the five relabel-eligible look clips relabel there.
    mistake = [_identification(ident, delta=0.05)]
    bars = ident.derive_bars(_look_population(ident), mistake)
    assert bars["b"] == pytest.approx(0.06)
    assert bars["look_relabel_at_b"] == pytest.approx(5 / 1000)
    assert bars["mistake_relabels_at_b"] == 0
    # 990 named at 0.00, 985 of them own-class: precision 985/990 >= 99%, so b_u = 0.00.
    assert bars["b_u"] == ident.GRID[0]
    assert bars["look_assigned_at_b_u"] == pytest.approx(985 / 990)
    # The literal rule's quantity is coverage over ALL clips (985/1000), which cannot rise.
    assert bars["b_u_literal"] is None
    assert bars["look_assigned_at_b_u_literal"] is None

    # Precision rises from 97% to 99.5% when the 25 low-margin rows drop out: b_u = 0.01,
    # the smallest bar that qualifies (not a later, even more precise one).
    rising = (
        [_identification(ident, match_class=1, margin=0.02) for _ in range(970)]
        + [_identification(ident, match_class=2, margin=0.005) for _ in range(25)]
        + [_identification(ident, match_class=2, margin=0.02) for _ in range(5)]
    )
    risen = ident.derive_bars(rising, mistake)
    assert risen["b_u"] == pytest.approx(0.01)
    assert risen["look_assigned_at_b_u"] == pytest.approx(970 / 975)
    assert risen["b_u_literal"] is None

    # A population already >= 99% own-class over all clips meets both rules at 0.00.
    mostly_own = (
        [_identification(ident, match_class=1, margin=0.02) for _ in range(995)]
        + [_identification(ident, match_class=2, margin=0.02) for _ in range(5)]
    )
    found = ident.derive_bars(mostly_own, mistake)
    assert found["b_u"] == ident.GRID[0]
    assert found["b_u_literal"] == ident.GRID[0]
    assert found["look_assigned_at_b_u"] == pytest.approx(0.995)
    assert found["look_assigned_at_b_u_literal"] == pytest.approx(0.995)

    never = [_identification(ident, delta=0.5) for _ in range(1000)]
    barred = ident.derive_bars(never, mistake)
    assert barred["b"] is None
    assert barred["no_bar"] is True


def test_derive_bars_reports_the_precision_table_that_sets_b_u(ident):
    """The per-bar table (assigned, precision, coverage): precision drives the amended b_u,
    while the literal rule's coverage is kept visible and never rises with the bar."""
    bars = ident.derive_bars(_look_population(ident), [_identification(ident, delta=0.05)])
    table = bars["precision"]
    assert table["used_for_b_u"] is True
    rows = table["by_bar"]
    assert [row["bar"] for row in rows] == list(ident.GRID)

    first = rows[0]
    assert first["assigned"] == 990  # 955 + 5 + 30 pass the floor; the ten floored rows do not
    assert first["precision"] == pytest.approx(985 / 990)
    assert first["coverage"] == pytest.approx(985 / 1000)
    at_030 = rows[ident.GRID.index(0.30)]
    assert at_030["assigned"] == 30
    assert at_030["precision"] == pytest.approx(1.0)
    assert at_030["coverage"] == pytest.approx(30 / 1000)
    last = rows[-1]
    assert last["assigned"] == 0
    assert last["precision"] is None
    assert last["coverage"] == 0.0
    coverage = [row["coverage"] for row in rows]
    assert all(later <= earlier for earlier, later in itertools.pairwise(coverage))
    assert bars["b_u"] == ident.GRID[0]
    assert bars["b_u_literal"] is None


# --- the reference builder, including the 8 fallback ayat -------------------------------------


@pytest.mark.parametrize("surah,ayah", [(55, 17), (90, 8), (90, 9), (90, 10),
                                        (106, 1), (106, 2), (106, 3), (106, 4)])
def test_the_eight_fallback_ayat_get_strictly_increasing_word_offsets(ident, surah, ayah):
    from generate_phonemes import FALLBACK_PHONEMES
    from quran_transcript import Aya

    words = Aya(surah, ayah).get().uthmani.split()
    reference = ident.ayah_reference(surah, ayah)
    offsets = reference.offsets
    assert len(offsets) == len(words) + 1
    assert offsets[0] == 0 and offsets[-1] == len(reference.reference)
    assert all(later > earlier for earlier, later in itertools.pairwise(offsets))
    from tadabur.normalization import normalize_phonemes

    assert reference.reference == normalize_phonemes(
        FALLBACK_PHONEMES[f"{surah}:{ayah}"].replace(" ", "")).normalized


def test_an_ordinary_ayah_reference_is_the_normalised_phonetization(ident):
    from quran_transcript import Aya

    reference = ident.ayah_reference(1, 1)
    words = Aya(1, 1).get().uthmani.split()
    assert len(reference.offsets) == len(words) + 1
    assert reference.offsets[0] == 0 and reference.offsets[-1] == len(reference.reference)
    assert all(later > earlier for earlier, later in itertools.pairwise(reference.offsets))


def test_the_committed_references_are_the_whole_mushaf(ident):
    db = ident.load_references()
    assert len(db.ayat) == 6236
    assert sorted({surah for surah, _ in db.ayat}) == list(range(1, 115))
    assert all(len(ayah.offsets) >= 2 for ayah in db.ayat.values())
    classes = {ayah.reference for ayah in db.ayat.values()}
    assert len(classes) == len({db.text_class(*key) for key in db.ayat})
    repeat = db.text_class(55, 13)
    assert len(db.class_ayat(repeat)) == 31
    assert db.class_label(repeat) == "55:13"


# --- the committed clip list and the calibration population -----------------------------------


def test_the_amendment_cites_the_frozen_holdout_text():
    assert _sha256(HOLDOUT_PREREGISTRATION) == HOLDOUT_SHA256
    assert HOLDOUT_SHA256 in PREREGISTRATION.read_text(encoding="utf-8")
    assert "Amended 2026-10-25 (S5, pre-outcome" in PREREGISTRATION.read_text(encoding="utf-8")


def test_the_bu_amendment_is_the_owner_approved_post_look_precision_rule():
    """The owner's dated erratum cites the frozen S5 pre-registration by SHA-256, is dated and
    post-look, and never edits that document (its own SHA pin is the real check)."""
    assert _sha256(PREREGISTRATION) == S5_PREREGISTRATION_SHA256
    text = BU_AMENDMENT.read_text(encoding="utf-8")
    assert "2026-10-25" in text
    assert "post-look" in text
    assert S5_PREREGISTRATION_SHA256 in text


def test_load_calibration_requires_the_s5b_floor_to_be_confirmed(ident, calibration, tmp_path,
                                                                 approval):
    """r19: S6 and S7 may label only under a confirmed floor. ``floor_status`` must be exactly
    "confirmed" — S5's binned "passed" is not confirmation — and provisional bars are refused.

    Success needs the frozen document's §7 owner approval too (r2), so the loadable cases pass
    the fixture record."""
    bars = {key: value for key, value in calibration["bars"].items()
            if key not in ("provisional", "provisional_reason")}
    path = tmp_path / "confirmed.json"
    path.write_text(json.dumps({**calibration, "floor_status": "confirmed", "bars": bars}),
                    encoding="utf-8")
    loaded, k = ident.load_calibration(path, approval_path=approval)
    assert loaded == bars
    assert k == calibration["parameters"]["k"] == ident.K

    for status in ("passed", "stopped: freesound 0.011 > 0.01"):
        path.write_text(json.dumps({**calibration, "floor_status": status, "bars": bars}),
                        encoding="utf-8")
        with pytest.raises(ValueError, match="not a confirmed calibration"):
            ident.load_calibration(path, approval_path=approval)

    path.write_text(json.dumps({**calibration, "floor_status": "confirmed",
                                "bars": {**bars, "provisional": True,
                                         "provisional_reason": "x"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="provisional"):
        ident.load_calibration(path, approval_path=approval)

    # The committed report loads exactly when its floor is confirmed *and* its margin definition
    # carries the owner's §7 approval; this PR adds the gate, not the approval record.
    if calibration["floor_status"] != "confirmed":
        with pytest.raises(ValueError, match="not a confirmed calibration"):
            ident.load_calibration()
    elif ident.MARGIN_APPROVAL_PATH.exists():
        loaded, k = ident.load_calibration()
        assert loaded == calibration["bars"]
        assert k == ident.K
    else:
        with pytest.raises(ValueError, match="owner approval"):
            ident.load_calibration()


def test_load_calibration_needs_the_owner_approval_of_the_margin(ident, calibration, tmp_path,
                                                                approval):
    """r1/r2: a confirmed floor and the oracle's mechanical ``final`` are not the owner's §7
    clearance. The bars stay unusable until a well-formed record cites the report's margin
    definition by SHA-256 *and* the report's ``b_u``; a matching record loads them."""
    bars = {key: value for key, value in calibration["bars"].items()
            if key not in ("provisional", "provisional_reason")}
    path = tmp_path / "margin.json"
    final = {**calibration["margin_definition"], "status": "final"}

    def written(definition):
        path.write_text(json.dumps({**calibration, "floor_status": "confirmed", "bars": bars,
                                    "margin_definition": definition}), encoding="utf-8")
        return path

    def written_approval(name, record):
        target = tmp_path / name
        target.write_text(json.dumps(record), encoding="utf-8")
        return target

    # Mechanical owner / missing status: refused before the approval is consulted.
    for definition in ({**final, "status": "owner"}, {}):
        with pytest.raises(ValueError, match="margin_definition.status"):
            ident.load_calibration(written(definition), approval_path=approval)

    # Confirmed floor + mechanical final + no approval record: refused, naming the approval.
    with pytest.raises(ValueError, match="owner approval"):
        ident.load_calibration(written(final), approval_path=tmp_path / "absent.json")

    base = _approval_record(calibration)
    # Each field the loader validates is load-bearing.
    cases = [
        ({**base, "schema": "not-it"}, "schema"),
        ({**base, "approved_by": "a bot"}, "approved_by"),
        ({**base, "approved_on": "27-10-2026"}, "approved_on"),
        ({k: v for k, v in base.items() if k != "doc"}, "does not cite"),
        ({**base, "doc": {"path": base["doc"]["path"], "sha256": "0" * 64}},
         "not the report's margin definition"),
        ({**base, "b_u": 0.42}, "approves b_u"),
        ({k: v for k, v in base.items() if k != "b_u"}, "does not record the approved b_u"),
    ]
    for record, message in cases:
        with pytest.raises(ValueError, match=message):
            ident.load_calibration(written(final),
                                   approval_path=written_approval("bad.json", record))

    # A matching approval loads the bars.
    loaded, k = ident.load_calibration(written(final), approval_path=approval)
    assert (loaded, k) == (bars, calibration["parameters"]["k"])


def test_approve_margin_writes_the_record_the_loader_accepts(ident, calibration, tmp_path,
                                                             monkeypatch):
    """r3: the approval needs a producer, not a hand-written JSON. The command records the live
    amendment's SHA-256 and the committed ``b_u``, refuses to overwrite, and the loader accepts
    exactly what it writes."""
    import tadabur.tlog_ident_confirm as confirm

    monkeypatch.setattr(confirm, "require_committed", lambda *args, **kwargs: None)
    path = tmp_path / "margin_approval.json"
    record = ident.approve_margin(204, path=path)

    assert record["schema"] == ident.MARGIN_APPROVAL_SCHEMA
    assert record["approved_by"] == "owner" and record["issue"] == 204
    assert record["b_u"] == calibration["bars"]["b_u"]
    assert record["doc"] == {"path": "docs/tlog-ident-margin-amendment.md",
                             "sha256": _sha256(MARGIN_AMENDMENT)}
    assert json.loads(path.read_text(encoding="utf-8")) == record

    loaded, k = ident.load_calibration(approval_path=path)
    assert (loaded, k) == (calibration["bars"], ident.K)
    with pytest.raises(ValueError, match="already records"):
        ident.approve_margin(204, path=path)


def test_load_identifier_binds_the_database_bars_and_engine(ident, calibration, approval,
                                                            tmp_path, monkeypatch):
    """r3 (I3.1): S6/S7 get the database, bars and k from one call, and every input the report
    records is compared with this checkout — a mismatch refuses instead of labelling on it."""
    database, bars, k = ident.load_identifier(approval_path=approval)
    assert (bars, k) == (calibration["bars"], calibration["parameters"]["k"])
    assert len(database.ayat) == calibration["reference"]["ayat"]
    record = ident.identification_record(
        ident.identify("بتثج", database=database, filename_ayah="1:1", k=k),
        decode_mode="whole_clip")
    assert record["floor_rule"] == ident.floor_rule()
    assert record["id_floor_ok"] is False  # 4 characters < L_MIN

    def mutated(mutate):
        report = json.loads(json.dumps(calibration))
        mutate(report)
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    reports = [
        (lambda r: r["reference"].__setitem__("sha256", "0" * 64), "not the calibrated"),
        (lambda r: r["reference"].__setitem__("ayat", 6235), "not the calibrated"),
        (lambda r: r["floor_rule"].__setitem__("sha256", "0" * 64), "carries the floor rule"),
    ]
    for mutate, message in reports:
        with pytest.raises(ValueError, match=message):
            ident.load_identifier(path=mutated(mutate), approval_path=approval)

    # The rule's values and the search library are bound to the code, not just to the report.
    monkeypatch.setattr(ident, "L_MIN", ident.L_MIN + 1)
    with pytest.raises(ValueError, match="the code's rule"):
        ident.load_identifier(approval_path=approval)
    monkeypatch.undo()

    monkeypatch.setattr(ident, "LIBRARY", {**ident.LIBRARY, "version": "1.4.0"})
    with pytest.raises(ValueError, match="this checkout's is"):
        ident.load_identifier(approval_path=approval)
    monkeypatch.undo()

    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.4.0")
    with pytest.raises(ValueError, match="installed edlib"):
        ident.load_identifier(approval_path=approval)


def test_the_reference_reader_does_not_import_the_phonetizer(ident, monkeypatch):
    """r3 (I3.6): S6/S7 read the committed database without quran-transcript installed; only
    rebuilding references needs the phonetizer."""
    import sys

    for module in ("hafs_phonetizer", "generate_phonemes"):
        monkeypatch.setitem(sys.modules, module, None)
    assert len(ident.load_references().ayat) == 6236


def test_the_reference_reader_refuses_an_incompatible_normalization_pin(ident, approval,
                                                                        monkeypatch):
    """r4 (R4.1): the reader keeps its dependency-free compatibility contract. A normalisation
    bump (which changes how a decode normalises) or a tokenisation change must not score new
    decodes against references built by the old algorithm, so both the reader and the consumer
    entry point refuse."""
    assert len(ident.load_references().ayat) == 6236

    monkeypatch.setattr(ident.normalization, "ALGORITHM_VERSION", "3")
    with pytest.raises(ValueError, match="normalization_algorithm_version"):
        ident.load_references()
    with pytest.raises(ValueError, match="normalization_algorithm_version"):
        ident.load_identifier(approval_path=approval)
    monkeypatch.undo()

    monkeypatch.setattr(ident, "WORD_TOKENISATION", "some-other tokenisation")
    with pytest.raises(ValueError, match="word_tokenisation"):
        ident.load_references()
    with pytest.raises(ValueError, match="word_tokenisation"):
        ident.load_identifier(approval_path=approval)


def test_pool_summary_reads_the_frozen_row_shape(ident):
    """A pool row carries the pass-1 ``speech_s``, never the clip's ``duration_s``: reading the
    wrong field failed the r5 calibration with a KeyError. Rows the codec floor rejected are
    skipped before any conversion, so their ``None`` never reaches ``float``."""
    rows = [
        {"split": "clean", "usable": True, "speech_s": 12.0, "filename_label": "1:1"},
        {"split": "clean", "usable": True, "speech_s": 25.0, "filename_label": "1:2"},
        {"split": "unclean", "usable": True, "speech_s": 75.0, "filename_label": None},
        {"split": "unclean", "usable": False, "speech_s": None, "filename_label": None},
    ]
    lengths, unlabelled = ident.pool_summary(rows)
    assert lengths["usable"] == 3
    assert lengths["field"] == "speech_s"
    assert (lengths["over_20_s"], lengths["over_30_s"], lengths["over_60_s"]) == (2, 1, 1)
    assert lengths["longest_s"] == 75.0
    assert unlabelled == 1


def test_the_page_renders_from_the_committed_report(ident, calibration):
    """The generated page must be ``render`` of the report, and the report carries the pool's
    speech histogram the page's pool section is read from."""
    assert ident.render(calibration) == ident.DOC_PATH.read_text(encoding="utf-8")
    pool = calibration["pool_lengths"]
    assert pool["source"] == "tadabur.tlog_overlap.iter_pool_rows"
    assert pool["field"] == "speech_s"  # the frozen row's length fact, not duration_s
    assert 0 < pool["over_60_s"] < pool["over_30_s"] < pool["over_20_s"] <= pool["usable"]
    assert pool["longest_s"] > 60


def test_pool_speech_seconds_make_no_floor_bin_claim(ident, calibration):
    """R6.1: VAD speech seconds are not the decode's length. A row with 19 s of speech can
    still decode past 160 characters into the floor-less bin, so neither the histogram nor the
    page may infer a row's floor bin from its speech duration."""
    lengths, _ = ident.pool_summary(
        [{"split": "unclean", "usable": True, "speech_s": 19.0, "filename_label": None}])
    letters = "بتثجحخدذرزسشصضطظعغفقكلمنهوي"
    long = (letters * 6)[:161]
    assert all(left != right for left, right in itertools.pairwise(long))
    db = ident.ReferenceDatabase.build([_ayah(ident, 1, 1, [long])])
    decoded = ident.identify(long, database=db)
    assert decoded.length == 161
    assert ident.floor_bin(decoded.length) == "161-"
    # The verdict is the S5b rule, not a bin: no bin can leave a row unverified.
    assert isinstance(decoded.floor_ok, bool)

    section = "\n".join(ident.pool_length_lines(calibration))
    for claim in ("calibrated bin", "at most", "needed to reach"):
        assert claim not in section
    assert "#192" in section


def test_the_margin_amendment_pins_the_committed_competitor_search(ident, calibration):
    """The dated amendment pins the word-span search, cites both frozen parents by SHA-256,
    names the engine commit it pins, and the calibration records its path and hash."""
    assert _sha256(HOLDOUT_PREREGISTRATION) == HOLDOUT_SHA256
    assert _sha256(PREREGISTRATION) == S5_PREREGISTRATION_SHA256
    text = MARGIN_AMENDMENT.read_text(encoding="utf-8")
    assert "2026-10-26" in text
    assert "post-look" in text
    assert "definitional" in text
    assert "one-directional" in text
    assert HOLDOUT_SHA256 in text
    assert S5_PREREGISTRATION_SHA256 in text
    assert "8138e8e" in text
    assert {key: calibration["margin_definition"][key] for key in ("path", "sha256")} == {
        "path": "docs/tlog-ident-margin-amendment.md",
        "sha256": _sha256(MARGIN_AMENDMENT)}
    # The oracle's mechanical consequence: "final" only when it changed neither b_u nor a look
    # assignment at b_u, otherwise the owner decides (architect r18/r19).
    assert calibration["margin_definition"]["status"] in ("final", "owner")


def test_the_real_mistake_clips_file_is_the_pinned_52_clips_over_46_ayat(ident):
    rows = [json.loads(line) for line in
            ident.REAL_MISTAKE_CLIPS_PATH.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 52
    assert [row["audio_filename"] for row in rows] == sorted(row["audio_filename"] for row in rows)
    sites = [site for row in rows for site in row["sites"]]
    assert len(sites) == 56 == len(set(sites))
    # The 52 clips cover 46 distinct ayat: one recited ayah can appear in several clips.
    assert len({row["surah_ayah"] for row in rows}) == 46
    assert all(len(row["audio_sha256"]) == 64 for row in rows)
    assert _sha256(ident.REAL_MISTAKE_CLIPS_PATH) in PREREGISTRATION.read_text(encoding="utf-8")

    sites_by_id = {}
    for line in TRUTH_SITES.read_text(encoding="utf-8").splitlines():
        site = json.loads(line)
        sites_by_id[site["site_id"]] = site
    for row in rows:
        for site_id in row["sites"]:
            site = sites_by_id[site_id]
            assert site["audio_filename"] == row["audio_filename"]
            assert site["surah_ayah"] == row["surah_ayah"]
            assert site["heard"] != site["prescribed"]


def test_the_shortlist_recall_sample_is_seeded_and_look_only(ident):
    look = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["look"]
    sample = ident.recall_sample(look)
    assert len(sample) == ident.RECALL_CLIPS
    assert len(set(sample)) == ident.RECALL_CLIPS
    assert set(sample) <= set(look)
    ranked = sorted(look, key=lambda clip: hashlib.sha256(
        f"{ident.SALT}:{clip}".encode("utf-8")).digest())
    assert list(sample) == ranked[:ident.RECALL_CLIPS]


def test_only_tarteel_look_clips_enter_the_calibration(ident, calibration):
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    decodes = ident.load_look_decodes()
    assert len(decodes) == len({decode.audio_filename for decode in decodes}) == 4537
    assert {decode.audio_filename for decode in decodes} <= set(split["look"])
    assert not {decode.audio_filename for decode in decodes} & set(split["confirm"])
    assert calibration["split"] == {
        "path": "tools/tadabur/nonpro_eval/tarteel_v1/split.json",
        "sha256": _sha256(SPLIT_PATH), "look": 4537}
    assert calibration["decodes"]["look"]["sha256"] == _sha256(ident.DECODES_PATH)
    assert calibration["decodes"]["look"]["clips"] == 4537


@pytest.fixture(scope="module")
def calibration(ident):
    return json.loads(ident.CALIBRATION_PATH.read_text(encoding="utf-8"))


def _approval_record(calibration, **overrides):
    """A valid owner approval of the committed margin definition (frozen floor document §7)."""
    record = {
        "schema": "tadabur.tlog_ident/margin-approval-v1",
        "issue": 204,
        "doc": {"path": calibration["margin_definition"]["path"],
                "sha256": calibration["margin_definition"]["sha256"]},
        "b_u": calibration["bars"]["b_u"],
        "approved_on": "2026-10-27",
        "approved_by": "owner",
    }
    record.update(overrides)
    return record


@pytest.fixture
def approval(tmp_path, calibration):
    """A recorded owner approval of the committed margin definition (frozen floor doc §7).

    The real record is absent until the owner approves at the S5b check-in, so every test that
    needs a loadable calibration writes this fixture instead.
    """
    path = tmp_path / "margin_approval.json"
    path.write_text(json.dumps(_approval_record(calibration)), encoding="utf-8")
    return path


def test_the_calibration_rederives_from_its_committed_inputs(ident, calibration):
    assert calibration["schema"] == "tadabur.tlog_ident/calibration-v1"
    assert calibration["issue"] == 204
    assert calibration["preregistration"] == {
        "path": "docs/tlog-ident-preregistration.md",
        "sha256": _sha256(PREREGISTRATION),
        "parent": {"path": "docs/tlog-holdout-preregistration.md", "sha256": HOLDOUT_SHA256}}
    assert calibration["bu_amendment"] == {
        "path": "docs/tlog-ident-bu-amendment.md", "sha256": _sha256(BU_AMENDMENT)}
    # S5b's own pre-registration, which cites every frozen parent by SHA-256.
    assert calibration["floor_preregistration"] == {
        "path": "docs/tlog-ident-floor-preregistration.md",
        "sha256": ident.FLOOR_PREREGISTRATION_SHA256,
        "parents": {"holdout": HOLDOUT_SHA256,
                    "ident": S5_PREREGISTRATION_SHA256,
                    "bu": _sha256(BU_AMENDMENT)}}
    assert calibration["floor_preregistration"]["sha256"] == _sha256(FLOOR_PREREGISTRATION)
    assert calibration["floor_rule"] == ident.floor_rule()
    assert calibration["real_mistake"]["sha256"] == _sha256(ident.REAL_MISTAKE_CLIPS_PATH)
    assert calibration["parameters"]["k"] == ident.K
    assert calibration["parameters"]["gram"] == ident.GRAM
    assert calibration["parameters"]["grid"] == list(ident.GRID)

    database = ident.load_references()
    look = ident.load_look_decodes()
    scored = [ident.identify(decode.spans, database=database,
                             filename_ayah=decode.surah_ayah) for decode in look]
    # The binned floors are the retired record and still derive from the committed decodes.
    floors = ident.derive_floors(scored)
    assert set(floors) == set(ident.FLOOR_BINS) - {"161-"}
    assert {bin_: round(value, 6) for bin_, value in floors.items()} == {
        bin_: round(value, 6) for bin_, value in calibration["floors"].items()}
    # The verdict is the S5b rule inside ``identify``; nothing is applied afterwards.
    assert all(isinstance(row.floor_ok, bool) for row in scored)

    clips = [json.loads(line) for line in
             ident.REAL_MISTAKE_CLIPS_PATH.read_text(encoding="utf-8").splitlines()]
    decodes = ident.load_decode_store(ident.REAL_MISTAKE_DECODES_PATH)
    mistakes = [ident.identify(decodes["items"][clip["audio_filename"]]["spans"],
                               database=database, filename_ayah=clip["surah_ayah"])
                for clip in clips]
    assert len(mistakes) == calibration["mistake"]["clips"] == 52
    assert calibration["mistake"]["passed"] == sum(row.floor_ok for row in mistakes)
    assert calibration["mistake"]["passed"] == 52  # the bar: every mistake clip still passes

    bars = ident.derive_bars(scored, mistakes)
    assert bars["b"] == calibration["bars"]["b"]
    assert calibration["bars"]["b_u"] in ident.GRID
    assert bars["b_u"] == calibration["bars"]["b_u"]
    assert bars["b_u_literal"] == calibration["bars"]["b_u_literal"]
    assert bars["look_assigned_at_b_u"] == pytest.approx(
        calibration["bars"]["look_assigned_at_b_u"])
    assert bars["look_relabel_at_b"] == pytest.approx(calibration["bars"]["look_relabel_at_b"])
    assert bars["mistake_relabels_at_b"] == calibration["bars"]["mistake_relabels_at_b"]
    assert bars["no_bar"] is (bars["b"] is None)
    # b and b_u are final only under a confirmed floor; L_min and DELTA never move.
    assert (ident.L_MIN, ident.DELTA) == (8, pytest.approx(0.30))
    assert calibration["bars"].get("provisional") is (calibration["floor_status"] != "confirmed")
    assert calibration["bars"]["precision"]["by_bar"] == bars["precision"]["by_bar"]

    retention = sum(row.floor_ok for row in scored) / len(scored)
    assert calibration["look"]["retention"] == pytest.approx(retention)
    assert calibration["look"]["relabel_rate_at_b"] == pytest.approx(
        bars["look_relabel_at_b"])
    tail = [row for decode, row in zip(look, scored) if decode.cer_wav >= 0.2]
    assert len(tail) == calibration["look"]["tail"]["clips"]
    assert calibration["look"]["tail"]["retention"] == pytest.approx(
        sum(row.floor_ok for row in tail) / len(tail))

    # The 445-recording pool: its binned A/B/C record (the S5 stop) and the rule's own rate.
    pool = json.loads(NOISE_POOL.read_text(encoding="utf-8"))["recordings"]
    noise = ident.load_decode_store(ident.NOISE_DECODES_PATH)
    assert set(noise["items"]) == {str(record["freesound_id"]) for record in pool}
    noise_rows = [ident.identify(item["spans"], database=database)
                  for item in noise["items"].values()]
    conventions = ident.noise_pass_by_convention(noise_rows, floors)
    assert calibration["noise"]["recordings"] == len(pool) == 445
    reported = calibration["noise"]["conventions"]
    for key in ("A", "B", "C"):
        assert reported[key]["recordings"] == conventions[key]["recordings"]
        assert reported[key]["passed"] == conventions[key]["passed"]
        assert reported[key]["pass"] == pytest.approx(conventions[key]["pass"])
    # B (back off to 81-160) is the S5 headline, kept as history: the retired binned rule.
    assert calibration["noise"]["pass"] == pytest.approx(conventions["B"]["pass"])
    assert calibration["noise"]["floor_passed"] == conventions["B"]["passed"]
    rule_passed = sum(row.floor_ok for row in noise_rows)
    assert calibration["noise"]["rule_pass"]["clips"] == 445
    assert calibration["noise"]["rule_pass"]["passed"] == rule_passed
    assert calibration["noise"]["rule_pass"]["pass"] == pytest.approx(rule_passed / 445)

    # The five bars are the S5b floor's status; the S5 record above is history.
    assert calibration["floor_status"] == calibration["confirm"]["status"]
    assert calibration["confirm"]["salt"] == ident.FLOOR_SALT

    guard = calibration["shortlist_guard"]
    assert guard["clips"] == ident.RECALL_CLIPS == 1000
    assert guard["salt"] == ident.SALT
    assert guard["k"] >= ident.K
    assert guard["passed"] is (guard["recall"] >= ident.RECALL_MIN)
    assert calibration["parameters"]["recall"]["minimum"] == ident.RECALL_MIN
    assert calibration["throughput"]["ms_per_clip"] > 0

    # b_u's reach: the unclean rows without a filename label that pass codec and silence and
    # survive S4, through the one S6/S7 reader.
    from tadabur.tlog_overlap import iter_pool_rows

    usable = sum(1 for row in iter_pool_rows()
                 if row["split"] == "unclean" and row["filename_label"] is None
                 and row["usable"])
    assert calibration["unlabelled_unclean"] == {
        "source": "tadabur.tlog_overlap.iter_pool_rows", "usable": usable}

    assert ident.DOC_PATH.is_file()
    assert all(path.name.endswith(".base.json") for path in ident.DECODES_DIR.iterdir())
    for path in (ident.REAL_MISTAKE_DECODES_PATH, ident.NOISE_DECODES_PATH):
        store = ident.load_decode_store(path)
        assert store["identity"] == {"model_ref": ident.BASE_MODEL,
                                     "hub_revision": ident.BASE_REVISION}


def test_the_calibration_scores_every_population_with_the_guard_k(ident, monkeypatch):
    """When the guard needs k = 40, the look scores, floors, bars, exhaustive pass and noise
    must all be measured with k = 40, and the report must record it.

    This test is about ``k``'s propagation, not the five bars: the confirmation is stubbed, so
    the empty synthetic mistake population never reaches ``five_bars``, which requires all 52
    clips (``test_a_short_real_mistake_population_cannot_move_the_bar``).
    """
    query = "بتثجحخدذرزسش"
    grams = [query[index:index + 3] for index in range(len(query) - 2)]
    bait = "قفق".join(reversed(grams))
    true = "بتصجحصدذصزسش"
    database = ident.ReferenceDatabase.build(
        [ident.AyahReference(surah, 1, bait, (0, len(bait))) for surah in range(1, 23)]
        + [ident.AyahReference(23, 1, true, (0, len(true)))])
    decodes = [ident.Decode(f"k{index}", f"clip{index}", "23:1", 1, 1, 0.0, query)
               for index in range(1000)]
    pool = json.loads(NOISE_POOL.read_text(encoding="utf-8"))["recordings"]
    noise_items = {str(record["freesound_id"]):
                   {"spans": "", "samples_sha256": record["samples_sha256"]}
                   for record in pool}

    monkeypatch.setattr(ident, "load_look_decodes", lambda: decodes)
    monkeypatch.setattr(ident, "real_mistake_clips", lambda: [])
    monkeypatch.setattr(ident, "load_decode_store",
                        lambda path: {"items": noise_items}
                        if Path(path) == ident.NOISE_DECODES_PATH else {"items": {}})
    import tadabur.tlog_ident_confirm as confirm
    monkeypatch.setattr(confirm, "confirm_block",
                        lambda *args, **kwargs: {"status": "confirmed", "stores": {}})
    import tadabur.tlog_overlap as overlap
    monkeypatch.setattr(overlap, "iter_pool_rows", lambda: iter(()))

    calls = []
    original = ident.identify

    def spy(*args, **kwargs):
        calls.append(kwargs.get("k"))
        return original(*args, **kwargs)

    monkeypatch.setattr(ident, "identify", spy)
    report = ident._calibration(database)
    assert report["shortlist_guard"]["k"] == 40
    assert report["parameters"]["k"] == 40
    assert set(calls) == {40}
    assert len(calls) == 1000 + 1000 + len(noise_items)


@pytest.mark.slow
def test_the_shortlist_guard_rederives_on_the_seeded_sample(ident, calibration):
    """k = 20's only evidence is the guard's recall; re-derive it clip by clip.

    A parity pass over a frozen result (1,000 clips, each scored against every window of the
    mushaf), measured at 497 s on the Mac: the merge gate runs it, the inner loop does not.
    """
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    decodes = {decode.audio_filename: decode for decode in ident.load_look_decodes()}
    sample = ident.recall_sample(split["look"])
    guard = calibration["shortlist_guard"]
    assert guard["clips"] == len(sample) == 1000
    assert guard["salt"] == ident.SALT
    database = ident.load_references()
    recalled = sum(database.best_score(decodes[clip].spans, k=guard["k"])
                   == database.exhaustive_best_score(decodes[clip].spans)
                   for clip in sample)
    assert recalled == guard["recalled"] == 1000
    assert guard["recall"] == recalled / len(sample)
    assert guard["passed"] is (guard["recall"] >= ident.RECALL_MIN)

    # Cheap shape checks the heavy re-derivation does not cover.
    assert set(calibration["exhaustive_decoy"]["floors"]) == set(calibration["floors"])
    assert 0.0 <= calibration["exhaustive_decoy"]["look_retention"] <= 1.0
    assert sum(calibration["look"]["bins"].values()) == calibration["look"]["clips"]
    assert calibration["throughput"]["clips"] == calibration["look"]["clips"]
    assert calibration["throughput"]["seconds"] > 0


def test_the_noise_store_is_the_committed_pool(ident, calibration):
    """The noise stop's input must be the pool's bytes, not a re-fetch."""
    pool = json.loads(NOISE_POOL.read_text(encoding="utf-8"))["recordings"]
    expected = {str(record["freesound_id"]): record["samples_sha256"] for record in pool}
    store = ident.load_decode_store(ident.NOISE_DECODES_PATH)
    assert {key: item["samples_sha256"] for key, item in store["items"].items()} == expected
    assert calibration["noise"]["samples_sha256"] == {
        "recordings": len(expected), "mismatched": 0}
