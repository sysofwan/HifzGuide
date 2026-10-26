"""Acceptance tests for tlog S5 ayah identification (``tadabur.tlog_ident``, #190).

Committed before the module exists and **locked**: later turns may add tests but never weaken or
rewrite these. The rules are "Ayah identification" in the frozen
``docs/tlog-holdout-preregistration.md`` plus S5's dated 2026-10-25 (S5, pre-outcome) amendment in
``docs/tlog-ident-preregistration.md``, which cites the frozen text by path and SHA-256.

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
  ``Window(surah, ayah, match)``; ``Identification`` with ``length``, ``match``, ``decoy``,
  ``exhaustive_decoy``, ``filename_class``, ``filename_score``, ``delta``, ``margin``,
  ``margin_class``, ``floor_ok``, ``overlaps_filename``, ``label_source``, ``assigned_label``.
* ``score``, ``reverse_clusters``, ``floor_bin``, ``identify``, ``apply_floors``, ``relabelled``,
  ``assigned``, ``derive_floors``, ``derive_bars``, ``ayah_reference``, ``load_references``,
  ``load_look_decodes``, ``recall_sample``, ``load_decode_store``.
* The committed artifacts under ``tlog/ident/``: ``references.json.gz``,
  ``real_mistake_clips.jsonl``, ``ident_calibration.json``, the whole-clip base decodes and the
  generated ``docs/tlog-ayah-identification.md``.

Ordering contract: ``shortlist`` and ``all_windows`` return scored windows best score first,
ties by ``(surah, ayah)``. A span's words are quran-transcript ``uthmani.split()`` words, 1-based.
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
#: The frozen hold-out text the S5 amendment cites; it is never edited (S3b's meta.json hashes it).
HOLDOUT_SHA256 = "204fe7f980488aff73402c56f1729f7f17f7c9471ea0664f6365bb8870b62086"
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
    """``count`` distinct four-letter references with no adjacent equal cores."""
    letters = "بتثجحخدذرزسشصضطظعغفقكلمنهوي"
    seen: list[str] = []
    for index in range(count):
        ref = "".join(letters[(index * step + shift) % len(letters)]
                      for step, shift in ((7, 0), (11, 3), (13, 5), (17, 7)))
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


def test_reverse_clusters_reverses_a_base_with_its_marks(ident):
    assert ident.reverse_clusters("بَتب") == "بتَب"
    assert ident.reverse_clusters("بتب") == "بتب"[::-1]


def test_the_decoy_sequence_reverses_each_ayah_and_keeps_ayah_order(ident, two_ayat):
    assert two_ayat.text(1) == "بجحدخدذر"
    assert list(two_ayat.decoy(1)) == list(
        ident.reverse_clusters("بجحد") + ident.reverse_clusters("خدذر"))
    assert two_ayat.decoy_window(1, 1) == two_ayat.decoy(1)
    assert two_ayat.decoy_window(1, 2) == two_ayat.decoy(1)


# --- Δ and the margin -------------------------------------------------------------------------


def test_delta_is_measured_against_the_filename_ayahs_own_window(ident, two_surahs):
    decoded = ident.identify("خدذ", database=two_surahs, filename_ayah="1:1",
                             bars={"b": 0.5, "b_u": 0.5})
    assert decoded.match.score == pytest.approx(1.0)
    assert decoded.match.text_class == two_surahs.text_class(2, 1)
    assert decoded.filename_class == two_surahs.text_class(1, 1)
    assert decoded.filename_score == ident.score("خدذ", two_surahs.window(1, 1))
    assert decoded.delta == pytest.approx(decoded.match.score - decoded.filename_score)
    assert decoded.delta == pytest.approx(1.0)
    assert decoded.overlaps_filename is False
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


def test_an_in_ayah_mistake_keeps_its_label(ident):
    """Two substitutions and a skipped word inside the right ayah keep the filename label."""
    db = ident.ReferenceDatabase.build([_ayah(ident, 1, 1, ["بجح", "دذر", "زسط"])])
    assert db.text(1) == "بجحدذرزسط"
    decode = "بجحدذز"  # word 2's final ر went to ز; the whole third word is skipped
    decoded = ident.identify(decode, database=db, filename_ayah="1:1",
                             bars={"b": 0.01, "b_u": 0.01})
    assert decoded.match.text_class == decoded.filename_class == db.text_class(1, 1)
    assert decoded.match.ayah_from == decoded.match.ayah_to == 1
    assert decoded.label_source == ident.BOTH
    assert decoded.assigned_label == "1:1"
    assert not ident.relabelled(decoded, 0.01)


# --- floors and the bars ----------------------------------------------------------------------


def test_the_floor_bins_are_the_pre_registered_length_bins(ident):
    assert [ident.floor_bin(length) for length in (0, 20, 21, 40, 41, 80, 81, 160, 161, 400)] == [
        "0-20", "0-20", "21-40", "21-40", "41-80", "41-80", "81-160", "81-160", "161-", "161-"]


def test_floor_pass_is_strictly_above_the_bin_floor(ident, two_surahs):
    decoded = ident.identify("بجح", database=two_surahs, floors={ident.floor_bin(3): 0.0})
    assert decoded.floor_ok is True
    at_floor = ident.identify("بجح", database=two_surahs, floors={ident.floor_bin(3): 1.0})
    assert at_floor.floor_ok is False
    assert at_floor.label_source == ident.NONE
    assert at_floor.assigned_label is None
    # Without floors the engine reports no floor verdict rather than assuming one.
    assert ident.identify("بجح", database=two_surahs).floor_ok is None


def _identification(ident, **changes):
    match = ident.Match(surah=1, ayah_from=1, word_from=1, ayah_to=1, word_to=1,
                        text_class=0, score=1.0)
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


def test_derive_bars_is_the_smallest_grid_value_meeting_both_rules(ident):
    # 960 assigned from 0.02 up, 30 only from 0.30, 10 never above the floor; five look clips
    # carry delta 0.10 and the one mistake clip delta 0.05.
    look = (
        [_identification(ident, margin=0.02) for _ in range(955)]
        + [_identification(ident, margin=0.02, delta=0.10) for _ in range(5)]
        + [_identification(ident, margin=0.30) for _ in range(30)]
        + [_identification(ident, floor_ok=False, margin=None, margin_class=None)
           for _ in range(10)]
    )
    mistake = [_identification(ident, delta=0.05)]
    bars = ident.derive_bars(look, mistake)
    assert bars["b"] == pytest.approx(0.06)
    assert bars["look_relabel_at_b"] == pytest.approx(5 / 1000)
    assert bars["mistake_relabels_at_b"] == 0
    assert bars["b_u"] == pytest.approx(0.03)
    assert bars["no_bar"] is (bars["b"] is None)

    never = [_identification(ident, delta=0.5) for _ in range(1000)]
    barred = ident.derive_bars(never, mistake)
    assert barred["b"] is None
    assert barred["no_bar"] is True


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


def test_the_real_mistake_clips_file_is_the_pinned_52_over_56(ident):
    rows = [json.loads(line) for line in
            ident.REAL_MISTAKE_CLIPS_PATH.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 52
    assert [row["audio_filename"] for row in rows] == sorted(row["audio_filename"] for row in rows)
    sites = [site for row in rows for site in row["sites"]]
    assert len(sites) == 56 == len(set(sites))
    assert len({row["surah_ayah"] for row in rows}) == 52
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
    assert calibration["decodes"]["look"]["sha256"] == _sha256(ident.LOOK_DECODES_PATH)
    assert calibration["decodes"]["look"]["clips"] == 4537


@pytest.fixture(scope="module")
def calibration(ident):
    return json.loads(ident.CALIBRATION_PATH.read_text(encoding="utf-8"))


def test_the_calibration_rederives_from_its_committed_inputs(ident, calibration):
    assert calibration["schema"] == "tadabur.tlog_ident/calibration-v1"
    assert calibration["issue"] == 190
    assert calibration["preregistration"] == {
        "path": "docs/tlog-ident-preregistration.md",
        "sha256": _sha256(PREREGISTRATION),
        "parent": {"path": "docs/tlog-holdout-preregistration.md", "sha256": HOLDOUT_SHA256}}
    assert calibration["real_mistake"]["sha256"] == _sha256(ident.REAL_MISTAKE_CLIPS_PATH)
    assert calibration["parameters"]["k"] == ident.K
    assert calibration["parameters"]["gram"] == ident.GRAM
    assert calibration["parameters"]["grid"] == list(ident.GRID)

    database = ident.load_references()
    look = ident.load_look_decodes()
    scored = [ident.identify(decode.spans, database=database,
                             filename_ayah=decode.surah_ayah) for decode in look]
    floors = ident.derive_floors(scored)
    assert set(floors) == set(ident.FLOOR_BINS)
    assert {bin_: round(value, 6) for bin_, value in floors.items()} == {
        bin_: round(value, 6) for bin_, value in calibration["floors"].items()}
    applied = [ident.apply_floors(row, floors) for row in scored]

    clips = [json.loads(line) for line in
             ident.REAL_MISTAKE_CLIPS_PATH.read_text(encoding="utf-8").splitlines()]
    decodes = ident.load_decode_store(ident.REAL_MISTAKE_DECODES_PATH)
    mistakes = [ident.apply_floors(ident.identify(
        decodes["items"][clip["audio_filename"]]["spans"], database=database,
        filename_ayah=clip["surah_ayah"]), floors) for clip in clips]
    assert len(mistakes) == calibration["mistake"]["clips"] == 52

    bars = ident.derive_bars(applied, mistakes)
    assert bars["b"] == calibration["bars"]["b"]
    assert bars["b_u"] == calibration["bars"]["b_u"]
    assert bars["look_relabel_at_b"] == pytest.approx(calibration["bars"]["look_relabel_at_b"])
    assert bars["mistake_relabels_at_b"] == calibration["bars"]["mistake_relabels_at_b"]
    assert bars["no_bar"] is (bars["b"] is None)

    retention = sum(row.floor_ok for row in applied) / len(applied)
    assert calibration["look"]["retention"] == pytest.approx(retention)
    assert calibration["look"]["relabel_rate_at_b"] == pytest.approx(
        bars["look_relabel_at_b"])
    tail = [row for decode, row in zip(look, applied) if decode.cer_wav >= 0.2]
    assert len(tail) == calibration["look"]["tail"]["clips"]
    assert calibration["look"]["tail"]["retention"] == pytest.approx(
        sum(row.floor_ok for row in tail) / len(tail))

    pool = json.loads(NOISE_POOL.read_text(encoding="utf-8"))["recordings"]
    noise = ident.load_decode_store(ident.NOISE_DECODES_PATH)
    assert set(noise["items"]) == {str(record["freesound_id"]) for record in pool}
    noise_scored = [ident.apply_floors(ident.identify(
        item["spans"], database=database), floors) for item in noise["items"].values()]
    noise_pass = sum(row.floor_ok for row in noise_scored) / len(noise_scored)
    assert calibration["noise"]["recordings"] == len(pool) == 445
    assert calibration["noise"]["pass"] == pytest.approx(noise_pass)

    stopped = calibration["stop_conditions"]
    assert stopped["look_retention"]["limit"] == 0.90
    assert stopped["noise_pass"]["limit"] == 0.01
    assert stopped["bar"]["limit"] is None
    assert stopped["look_retention"]["value"] == pytest.approx(retention)
    assert stopped["look_retention"]["fired"] is (retention < 0.90)
    assert stopped["noise_pass"]["value"] == pytest.approx(noise_pass)
    assert stopped["noise_pass"]["fired"] is (noise_pass > 0.01)
    assert stopped["bar"]["value"] == bars["b"]
    assert stopped["bar"]["fired"] is (bars["b"] is None)

    guard = calibration["shortlist_guard"]
    assert guard["clips"] == ident.RECALL_CLIPS
    assert guard["salt"] == ident.SALT
    assert guard["k"] >= ident.K
    assert guard["passed"] is (guard["recall"] >= ident.RECALL_MIN)
    assert calibration["parameters"]["recall"]["minimum"] == ident.RECALL_MIN
    assert calibration["throughput"]["ms_per_clip"] > 0

    assert ident.DOC_PATH.is_file()
    assert all(path.name.endswith(".base.json") for path in ident.DECODES_DIR.iterdir())
    for path in (ident.REAL_MISTAKE_DECODES_PATH, ident.NOISE_DECODES_PATH):
        store = ident.load_decode_store(path)
        assert store["identity"] == {"model_ref": ident.BASE_MODEL,
                                     "hub_revision": ident.BASE_REVISION}
