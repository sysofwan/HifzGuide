"""Ship A's onset guard (#138): exposure, the owner's notes, the candidates' scores on known
cases, the operating-point rule, the owner check, and the committed sidecar. Torch-free."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tadabur import onset_guard as o
from tadabur import owner_notes, waqf_guard
from tadabur.exclusions import Exclusions, current
from tadabur.listening_session import Verdict, read_verdicts, write_verdicts
from tadabur.phoneme_vocab import PHONEME_CHAR_TO_ID, PHONEME_ID_TO_CHAR, PHONEME_PAD_ID
from tadabur.waqf_guard import Item, all_items

#: The owner's four trim notes found while listening (#138), on shaddah sites at index 2.
FLAGGED = {
    "new_audit:tadabur_spk0017_S3_A37_a74cb622_000020.wav#0@2:shaddah": "Trimming issues",
    "new_audit:tadabur_spk0034_S3_A37_ab548534_000017.wav#0@2:shaddah": "Incorrect trim",
    "new_audit:tadabur_spk0307_S1_A194_81a47bca_000131.wav#0@2:shaddah": "May have trimming issues",
    "new_audit:tadabur_spk0360_S11_A49_4672c14f_000050.wav#0@2:shaddah":
        "Incurrect split (tha missing)",
}


@pytest.mark.parametrize("note, says", [
    ("Trimming issues", True), ("Incorrect trim", True), ("May have trimming issues", True),
    ("Incurrect split (tha missing)", True), ("CUT  off", True), ("trimmed", True),
    ("no trim", False), ("not cut", False), ("No  Trim", False), ("didn't cut", False),
    ("didn’t cut", False), ("the clip is not trimmed", False),
    ("", None), ("Not hafs", None), ("waqf", None), ("trimmed? no trim", None),
])
def test_owner_notes(note, says):
    assert o.owner_says(note) is says


@pytest.mark.parametrize("note, waqf, trim", [
    ("no waqf, trimmed", False, True), ("trimmed, no waqf", False, True),
    ("no trim, waqf", True, False), ("no waqf trimmed", False, True),
    ("waqf but no trim", True, False), ("paused and trimmed", True, True),
    ("not trimmed; wasl", False, False), ("no waqf; no trim", False, False),
])
def test_a_negation_governs_only_its_own_claim(note, waqf, trim):
    """#138: the waqf and the trim claim of one note are read apart."""
    assert (waqf_guard.owner_says(note), o.owner_says(note)) == (waqf, trim)


def _legacy_waqf(note: str) -> bool | None:
    """#134's waqf reader as it was (any of the three words before a pause word negates it),
    the oracle the scoped reader must agree with on the owner's committed notes."""
    import re
    pause = {"waqf", "pause", "paused", "pauses", "pausing", "stop", "stopped", "stops",
             "stopping"}
    words = re.findall(r"[a-z']+", " ".join(note.lower().replace("\u2019", "'").split()))
    claims = set()
    for k, word in enumerate(words):
        if word in pause or word == "wasl":
            negated = any(w in owner_notes.NEGATORS or w.endswith("n't")
                          for w in words[max(0, k - 3):k])
            claims.add((word in pause) != negated)
    return claims.pop() if len(claims) == 1 else None


def test_every_committed_note_reads_as_the_waqf_guard_read_it():
    notes = {v.note for v in read_verdicts().values() if v.note}
    assert notes and all(waqf_guard.owner_says(n) is _legacy_waqf(n) for n in notes)


def test_the_four_flagged_sites_are_trimmed_at_the_committed_verdicts():
    """The owner's committed notes, not manufactured ones, take every issue site out."""
    verdicts = read_verdicts()
    assert all(o.owner_says(verdicts[i].note) for i in FLAGGED)
    assert set(FLAGGED) <= o.trimmed_ids()
    assert set(FLAGGED) <= current().by_reason[o.ONSET_TRIMMED]


def item(reference_index: int, start: int = 0, excerpt: int = 0) -> Item:
    return Item("x", ("test",), "new_audit", "s", "shaddah", "held", "clip.wav", "1:1", 0,
                (0, 6, 12), "ءَللَذِ ممَااا", reference_index, start, 16000, excerpt)


def test_an_item_is_exposed_when_its_mark_is_read_in_the_segments_first_word():
    assert o.exposure(item(2)) == o.EXPOSED
    assert o.exposure(item(7)) == o.NOT_EXPOSED
    assert o.cut_of(item(2)) == o.FILE_START
    assert o.cut_of(item(2, start=480, excerpt=0)) == o.FILE_START  # the owner hears sample 0
    assert o.cut_of(item(2, start=480, excerpt=320)) == o.SEGMENTER_CUT


# --- the decisions ---------------------------------------------------------------------------


def _row(item_id="x", exposure=o.EXPOSED, trimmed=None):
    return {"item_id": item_id, "exposure": exposure, "detector_trimmed": trimmed}


def test_the_owner_overrides_the_detector_both_ways():
    assert o.resolve(_row(trimmed=True), None) == (o.ONSET_TRIMMED, o.DETECTOR)
    assert o.resolve(_row(trimmed=True), Verdict("x", "held", "no trim")) == (o.SITE, o.OWNER_NOTE)
    assert o.resolve(_row(trimmed=False), Verdict("x", "held", "Incorrect trim")) == (
        o.ONSET_TRIMMED, o.OWNER_NOTE)
    assert o.resolve(_row(trimmed=False), None) == (o.SITE, o.DETECTOR)
    assert o.resolve(_row(), None) == (o.SITE, None)  # the detector does not decide
    assert o.resolve(_row(exposure=o.NOT_EXPOSED), Verdict("x", "held", "trim")) == (
        o.ONSET_TRIMMED, o.OWNER_NOTE)


def test_trimmed_ids_read_the_verdicts_at_every_use(tmp_path: Path):
    guard, verdicts = tmp_path / "guard.jsonl", tmp_path / "verdicts.jsonl"
    o.write_jsonl(guard, [_row("a"), _row("b", trimmed=True)])
    assert o.trimmed_ids(guard, verdicts) == {"b"}
    write_verdicts({"a": Verdict("a", "held", "Trimming issues"),
                    "b": Verdict("b", "held", "no trim")}, verdicts)
    assert o.trimmed_ids(guard, verdicts) == {"a"}


def test_the_four_flagged_sites_are_trimmed_by_the_owners_notes(tmp_path: Path):
    """The committed sidecar holds every flagged site; their notes take them out, a negated
    note brings them back, and the shared exclusion carries the decision."""
    verdicts = tmp_path / "verdicts.jsonl"
    write_verdicts({i: Verdict(i, "held", n) for i, n in FLAGGED.items()}, verdicts)
    assert set(FLAGGED) <= o.trimmed_ids(verdicts_path=verdicts)
    assert set(FLAGGED) <= current(verdicts).by_reason[o.ONSET_TRIMMED]
    write_verdicts({i: Verdict(i, "held", "no trim") for i in FLAGGED}, verdicts)
    assert not set(FLAGGED) & o.trimmed_ids(verdicts_path=verdicts)


def test_an_exclusion_lists_each_item_once_under_its_first_reason():
    excluded = Exclusions.of(waqf_realized={"a"}, onset_trimmed={"a", "b"})
    assert excluded.listed(["c", "b", "a"]) == {"waqf_realized": ["a"], "onset_trimmed": ["b"]}
    assert "b" in excluded and "c" not in excluded
    with pytest.raises(ValueError):
        Exclusions.of(other={"a"})


# --- synthetic trims and the scores ----------------------------------------------------------


def test_a_clear_onset_follows_a_silent_pause():
    levels = np.full(200, -60.0)
    levels[100:] = -20.0  # speech from frame 100 (1.00 s)
    levels[99] = -50.0  # a rise above the silence before it
    onset = o.clear_onset(levels, word_s=1.0, pause_start=0)
    assert onset is not None and onset.sample == 99 * o.LEVEL_FRAME
    assert o.clear_onset(levels, word_s=1.0, pause_start=90) is None  # lead not in the pause
    levels[80:99] = -20.0  # no silence before the word
    assert o.clear_onset(levels, word_s=1.0, pause_start=0) is None


def test_the_depths_cut_from_the_onset_and_only_a_cut_into_the_letter_is_a_trim():
    clip = {"audio_filename": "c.wav", "segment_index": 1, "onset_sample": 16000,
            "segment_end_sample": 48000, "reference": "ءَب", "split": o.CALIBRATION,
            "reciter_id": 3}
    rows = o.spans([], [clip])
    assert [(r["depth_ms"], r["start_sample"], r["trimmed"]) for r in rows] == sorted(
        [(d, 16000 + 16 * d, d > 0) for d in o.DEPTHS_MS], key=lambda x: str(x[0]))
    assert all(r["key"].startswith("synthetic:c.wav#1@") for r in rows)


def test_the_first_letter_takes_its_marks_with_it():
    labels, first = o.labels_of("ءَللَ ذِ")
    assert first == 2 and len(labels) == 7
    with pytest.raises(ValueError):
        o.labels_of("x")


def test_leading_missing_counts_the_reference_before_the_decode_matches():
    assert o.leading_missing("للَذِۦۦنَ", "ءَللَذِۦۦنَ") == 2
    assert o.leading_missing("ءَللَذِۦۦنَ", "ءَللَذِۦۦنَ") == 0
    assert o.leading_missing("ب", "ءَل") == 3


def _posteriors(text: str, frames_per_char: int = 3) -> np.ndarray:
    """Log-posteriors that emit ``text`` (each character for a few frames, blanks between)."""
    ids = []
    for char in text:
        ids += [PHONEME_CHAR_TO_ID[char]] * frames_per_char + [PHONEME_PAD_ID]
    probabilities = np.full((len(ids), len(PHONEME_ID_TO_CHAR)), 0.01)
    probabilities[np.arange(len(ids)), ids] = 1.0
    return np.log(probabilities / probabilities.sum(axis=1, keepdims=True))


def test_the_ctc_scores_tell_a_present_onset_from_a_missing_one():
    reference = "ءَللَذِ"
    present = o.ctc_scores(_posteriors(reference), reference)
    missing = o.ctc_scores(_posteriors(reference[2:]), reference)
    assert present["ctc_onset_llr"] > 0 > missing["ctc_onset_llr"]
    assert present["ctc_first_posterior"] > 0.5 > missing["ctc_first_posterior"]
    assert (present["decode_leading_missing"], missing["decode_leading_missing"]) == (0, 2)
    assert missing["decode_head"] == reference[2:]


# --- calibration and the owner check ---------------------------------------------------------


def _scored(values, trimmed, name="ctc_onset_llr", split=o.CALIBRATION):
    return [{"scores": {name: v}, "trimmed": t, "split": split, "reciter_id": k,
             "depth_ms": 40 if t else 0} for k, (v, t) in enumerate(zip(values, trimmed))]


def test_the_operating_point_is_the_highest_recall_at_the_precision_floor():
    candidate = o.CANDIDATES[0]  # trimmed when low
    rows = _scored([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    rows += _scored([-6] * 9 + [5], [1] * 9 + [0])
    chosen = o.operating_point(candidate, o.curve(candidate, rows))
    assert chosen["threshold"] == 3 and chosen["recall"] == 1.0
    assert o.operating_point(candidate, o.curve(candidate, _scored([0, 0], [1, 0]))) is None


def _check_rows(trimmed_ids, called_ids):
    return [{"item_id": i, "exposure": o.EXPOSED,
             "scores": {c.name: 1.0 if i in called_ids else 0.0 for c in o.CANDIDATES}}
            for i in sorted(set(trimmed_ids) | set(called_ids) | {"clean"})]


def _calibration():
    """Every candidate at 1.0, ``onset_level_db`` chosen: it fires on the rows scored 1.0."""
    return {"chosen": "onset_level_db",
            "candidates": {c.name: {"operating_point": 1.0} for c in o.CANDIDATES}}


def test_a_detector_that_fires_everywhere_is_contradicted_by_the_owner():
    verdicts = {i: Verdict(i, "held", n) for i, n in
                [("a", "Trimming issues"), ("b", ""), ("c", ""), ("clean", "")]}
    agrees = o.owner_check(_check_rows({"a"}, {"a"}), verdicts, _calibration())
    assert not agrees["contradicted"] and agrees["candidates"]["onset_level_db"]["recall"] == 1.0
    loud = o.owner_check(_check_rows({"a"}, {"a", "b", "c", "clean"}), verdicts, _calibration())
    assert loud["contradicted"]
    assert [r["item_id"] for r in loud["relisten"]] == ["b", "c", "clean"]
    missed = o.owner_check(_check_rows({"a"}, set()), verdicts, _calibration())
    assert missed["contradicted"]
    assert o.decider(_calibration(), missed) is None


# --- the committed sidecar -------------------------------------------------------------------


def test_the_committed_guard_reads_the_committed_inputs():
    summary = json.loads(o.SUMMARY_PATH.read_text(encoding="utf-8"))
    for name, path in (("ship_a/onset_guard.synthetic.jsonl", o.SYNTHETIC_PATH),
                       ("ship_a/onset_guard.scores.jsonl", o.SCORES_PATH)):
        assert summary["inputs"][name] == o.sha256_of(path), name
    rows = o.read_jsonl(o.GUARD_PATH)
    assert [r["item_id"] for r in rows] == [i.item_id for i in all_items()]
    scored = o.read_jsonl(o.SCORES_PATH)
    assert [{k: v for k, v in r.items() if k != "scores"} for r in scored] == o.spans(
        all_items(), o.read_jsonl(o.SYNTHETIC_PATH))


def test_the_committed_calibration_is_re_derived_from_the_scores():
    """The synthetic halves are reciter-disjoint, the calibration is the rule's on the
    committed scores, and the detector decides only where the summary says it does."""
    summary = json.loads(o.SUMMARY_PATH.read_text(encoding="utf-8"))
    synthetic = [r for r in o.read_jsonl(o.SCORES_PATH) if r["kind"] == o.SYNTHETIC]
    halves = [{r["reciter_id"] for r in synthetic if r["split"] == s} for s in o.SPLITS]
    assert not halves[0] & halves[1]
    assert all(r["split"] == o.reciter_split(r["reciter_id"]) for r in synthetic)
    calibration, _ = o.calibrate(synthetic)
    assert summary["calibration"] == json.loads(json.dumps(calibration))
    deciding = summary["detector_decides"]
    for row in o.read_jsonl(o.GUARD_PATH):
        if row["exposure"] == o.EXPOSED and deciding is None:
            assert row["detector_trimmed"] is None
