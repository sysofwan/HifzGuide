"""Tests for synthetic edits, their decoys and the blind-check worklist (#88).

The waveform tests run on synthetic recitations in which every letter is a pure tone of its
own pitch, so "which letter is at the carrier" and "how long is it held" can be read off
the samples: an edit must change only its target span, a decoy must change the timing by
the same amount somewhere else and leave the carrier alone, and each item's label must say
what its audio now holds.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from tadabur.staged_audio import IndexRow, StagedClip, load_staged_clips
from tadabur.synthetic_edit_plan import (
    AT_CARRIER,
    CONSONANT_SWAP,
    CROP,
    DECOY,
    EDIT,
    IN_MADD,
    MIN_DECOY_DISTANCE,
    SHADDAH_ADDED,
    SHADDAH_REMOVED,
    SPLICE,
    STRETCH,
    TimedClip,
    Token,
    anchored_positions,
    consonant_swaps,
    labelled_reference,
    plan_pairs,
    select_pairs,
    shaddah_added,
    shaddah_removed,
)
from tadabur.synthetic_edits import (
    EDITS_PATH,
    SALT,
    WORKLIST_PATH,
    blind_check,
    check_manifest,
    edit_frame,
    edit_reciters,
    evaluation_reciters,
    exposure_rows,
    h448_unseen_shards,
    item_record,
    output_filename,
    read_frame,
    read_items,
    render,
    teacher_agrees,
)
from tadabur.truth_sites import HELD, NOT_HELD, PENDING, SHADDAH, load_truth_sites, parse_site
from tadabur.waveform_edits import FADE, changed_region, crop, fill, join, replace_span

UNIT = 3200  # samples per letter in the synthetic recitations (0.2 s)
SHA = "a" * 64
RECITER = 7


# --- synthetic recitations ---------------------------------------------------------------


_CHARS = "بلسصكشدمنرتفاَُِ"


def _pitch(char: str) -> float:
    """A distinct pitch per character, 15 Hz apart, all inside the voiced range."""
    return 100.0 + 15.0 * _CHARS.index(char)


def _recitation(name: str, reference: str) -> tuple[TimedClip, np.ndarray]:
    """A clip where each reference character is ``UNIT`` samples of its own tone, and the
    teacher emitted each one over the middle half of its stretch (a perfect decode)."""
    t = np.arange(UNIT) / 16000.0
    waveform = np.concatenate(
        [0.3 * np.sin(2 * np.pi * _pitch(c) * (t + i * UNIT / 16000.0)) for i, c in
         enumerate(reference)]
    ).astype(np.float32)
    tokens = tuple(Token(c, i * UNIT + UNIT // 4, i * UNIT + 3 * UNIT // 4)
                   for i, c in enumerate(reference))
    clip = TimedClip(name, RECITER, len(waveform), reference, tokens,
                     anchored_positions(reference, reference))
    return clip, waveform


def _letter_at(samples: np.ndarray, start: int, end: int, letters: str) -> str:
    """Which of ``letters`` has the strongest tone in ``samples[start:end]``."""
    t = np.arange(end - start) / 16000.0
    power = {c: abs(np.dot(samples[start:end], np.exp(-2j * np.pi * _pitch(c) * t)))
             for c in letters}
    return max(power, key=power.get)


def _run_length(samples: np.ndarray, centre: int, letter: str, others: str) -> int:
    """How many samples around ``centre`` the tone of ``letter`` holds, in 400-sample
    hops: the duration of a held letter."""
    hop, total = 400, 0
    for step in (-hop, hop):
        position = centre if step > 0 else centre - hop
        while 0 <= position and position + hop <= len(samples) and \
                _letter_at(samples, position, position + hop, letter + others) == letter:
            total += hop
            position += step
    return total


# A geminate س (index 4), a single fricative ش between harakat (index 9), and a long madd
# well away from both (indices 17-20).
GEMINATE_REFERENCE = "بَلَسسَكُشَدَمُبِ" + "اااا" + "نَدُرَ"
SWAP_TARGET = "بَلَكُسَمُفَدُنِ" + "اااا" + "رَ"
SWAP_DONORS = "دَكَصَبُتَلُسَرِ"


# --- the waveform primitives -------------------------------------------------------------


def test_crop_changes_only_the_crossfade_around_the_cut():
    x = np.random.default_rng(0).standard_normal(20000).astype(np.float32)
    y = crop(x, 8000, 1500)
    lo, hi = changed_region(8000, 0)
    assert len(y) == len(x) - 1500
    assert np.array_equal(y[:lo], x[:lo])
    assert np.array_equal(y[hi:], x[hi + 1500:])


def test_replace_span_changes_only_the_span_and_its_crossfades():
    x = np.random.default_rng(1).standard_normal(20000).astype(np.float32)
    material = np.random.default_rng(2).standard_normal(3000 + FADE).astype(np.float32)
    y = replace_span(x, 6000, 8000, material)
    lo, hi = changed_region(6000, 3000)
    assert len(y) == len(x) + 1000
    assert np.array_equal(y[:lo], x[:lo])
    assert np.array_equal(y[hi:], x[8000 + FADE // 2:])
    assert np.array_equal(y[lo + FADE:hi - FADE], material[FADE:-FADE])


def test_join_crossfades_and_refuses_an_overlap_longer_than_either_side():
    a, b = np.ones(400, np.float32), np.zeros(400, np.float32)
    y = join(a, b)
    assert len(y) == 800 - FADE and y[0] == 1 and y[-1] == 0
    assert np.all(np.diff(y) <= 0)
    with pytest.raises(ValueError):
        join(a[:10], b)


def test_fill_extends_a_periodic_region_in_phase():
    t = np.arange(16000) / 16000.0
    x = np.sin(2 * np.pi * 125.0 * t).astype(np.float32)  # period 128 samples
    material = fill(x, (4000, 8000), 47 * 128 + FADE, at=6000, seed=0)
    y = replace_span(x, 6000, 6000, material)
    # Stretched by whole periods, a sine stays a sine: no jump beyond the tone's own slope.
    assert np.max(np.abs(np.diff(y))) <= np.max(np.abs(np.diff(x))) * 1.05


def test_fill_of_noise_is_seeded_and_does_not_simply_repeat():
    x = np.random.default_rng(3).standard_normal(8000).astype(np.float32)
    a = fill(x, (1000, 3000), 4000, at=3000, seed=5)
    assert np.array_equal(a, fill(x, (1000, 3000), 4000, at=3000, seed=5))
    assert not np.array_equal(a, fill(x, (1000, 3000), 4000, at=3000, seed=6))


# --- planning on synthetic timings -------------------------------------------------------


def test_anchoring_needs_equal_context_on_both_sides():
    anchors = anchored_positions("abcdefg", "abcXefg")
    assert set(anchors) == set()  # every equal block is too short for two each side
    anchors = anchored_positions("abcdefghij", "abcdefghij")
    assert set(anchors) == set(range(2, 8))
    # Word spaces are not in the decode: they neither break a block nor get anchored.
    anchors = anchored_positions("abc defghij", "abcdefghij")
    assert set(anchors) == {2, 4, 5, 6, 7, 8} and anchors[4] == 3


def test_shaddah_removed_crops_the_held_span_and_its_decoy_an_equal_madd_span():
    clip, _ = _recitation("g.wav", GEMINATE_REFERENCE)
    (pair,) = shaddah_removed(clip, SALT)
    assert (pair.reference_index, pair.mark, pair.prescribed, pair.edited) == (4, SHADDAH, HELD, NOT_HELD)
    assert (pair.edit.place, pair.edit.kind) == (AT_CARRIER, CROP)
    assert (pair.edit.start_sample, pair.edit.end_sample) == (4 * UNIT + UNIT // 2, 5 * UNIT + UNIT // 2)
    assert (pair.decoy.place, pair.decoy.kind) == (IN_MADD, CROP)
    assert pair.decoy.length_change == pair.edit.length_change < 0
    assert abs(pair.decoy.start_sample - pair.edit.start_sample) >= MIN_DECOY_DISTANCE
    assert labelled_reference(pair, EDIT) == GEMINATE_REFERENCE[:5] + GEMINATE_REFERENCE[6:]
    assert labelled_reference(pair, DECOY) == GEMINATE_REFERENCE


def test_shaddah_added_stretches_a_single_fricative_and_its_decoy_a_madd_equally():
    clip, _ = _recitation("g.wav", GEMINATE_REFERENCE)
    (pair,) = shaddah_added(clip, 1600, SALT)
    assert (pair.reference_index, pair.prescribed, pair.edited) == (9, NOT_HELD, HELD)
    assert (pair.edit.kind, pair.decoy.kind) == (STRETCH, STRETCH)
    assert pair.edit.length_change == pair.decoy.length_change == 1600
    assert pair.decoy.place == IN_MADD
    assert labelled_reference(pair, EDIT) == GEMINATE_REFERENCE[:10] + "ش" + GEMINATE_REFERENCE[10:]


def test_a_clip_without_a_neutral_madd_yields_no_pair():
    clip, _ = _recitation("g.wav", "بَلَسسَكُشَدَمُبِنَدُرَ")
    assert shaddah_removed(clip, SALT) == [] and shaddah_added(clip, 1600, SALT) == []


def test_a_swap_takes_both_donors_from_the_same_reciter_with_the_same_haraka():
    target, _ = _recitation("t.wav", SWAP_TARGET)
    donors, _ = _recitation("d.wav", SWAP_DONORS)
    pairs = {(p.audio_filename, p.reference_index): p for p in consonant_swaps([target, donors], SALT)}
    pair = pairs[("t.wav", 6)]  # سَ in the target
    assert (pair.operation, pair.mark, pair.prescribed, pair.edited) == (CONSONANT_SWAP, "س↔ص", "س", "ص")
    assert (pair.edit.kind, pair.decoy.kind) == (SPLICE, SPLICE)
    assert pair.edit.donor.letter == "ص" and pair.decoy.donor.letter == "س"
    for donor in (pair.edit.donor, pair.decoy.donor):
        assert donor.audio_filename == "d.wav"
        assert SWAP_DONORS[donor.reference_index + 1] == SWAP_TARGET[7]  # same haraka
    assert pair.edit.length_change == pair.decoy.length_change == 0
    assert labelled_reference(pair, EDIT) == SWAP_TARGET[:6] + "ص" + SWAP_TARGET[7:]


def test_a_swap_needs_a_same_letter_donor_for_its_decoy():
    target, _ = _recitation("t.wav", SWAP_TARGET)
    only_other, _ = _recitation("d.wav", "دَكَصَبُتَلُرَ")
    assert [p for p in consonant_swaps([target, only_other], SALT)
            if p.audio_filename == "t.wav"] == []


def test_selection_is_hash_ordered_and_caps_clips_and_reciters():
    clip, _ = _recitation("g.wav", GEMINATE_REFERENCE)
    pairs, length = plan_pairs([clip], SALT)
    assert length == UNIT
    assert {p.operation for p in pairs} == {SHADDAH_REMOVED, SHADDAH_ADDED}
    assert select_pairs(pairs, quota=5, per_reciter=1, salt=SALT) == sorted(
        pairs, key=lambda p: p.pair_id)
    duplicated = pairs + [dataclasses.replace(pairs[0], reference_index=99)]
    assert len(select_pairs(duplicated, quota=5, per_reciter=5, salt=SALT)) == len(pairs)


# --- rendering: what each item's audio holds ---------------------------------------------


def _render_pair(pair, waveforms):
    return {role: render(pair, role, waveforms.__getitem__, seed=1) for role in (EDIT, DECOY)}


def _shift(change, position):
    """Where a source sample at ``position`` lands after ``change`` (before or after it)."""
    return position if position < change.start_sample else position + change.length_change


def test_shaddah_removed_shortens_only_the_hold_and_the_decoy_only_the_madd():
    clip, x = _recitation("g.wav", GEMINATE_REFERENCE)
    (pair,) = shaddah_removed(clip, SALT)
    rendered = _render_pair(pair, {"g.wav": x})
    for role, (y, gain) in rendered.items():
        change = pair.edit if role == EDIT else pair.decoy
        lo, hi = changed_region(change.start_sample, change.inserted_samples)
        assert gain is None and len(y) == len(x) + change.length_change
        assert np.array_equal(y[:lo], x[:lo])
        assert np.array_equal(y[hi:], x[hi - change.length_change:])
    held, neighbours = 5 * UNIT, "َك"  # the centre of the doubled س
    hold = _run_length(x, held, "س", neighbours)
    edit, decoy = rendered[EDIT][0], rendered[DECOY][0]
    assert _run_length(edit, pair.edit.start_sample, "س", neighbours) <= hold - 1200
    assert _run_length(decoy, held, "س", neighbours) == hold
    madd, neighbours = 19 * UNIT, "ِن"
    length = _run_length(x, madd, "ا", neighbours)
    assert _run_length(decoy, _shift(pair.decoy, madd - UNIT), "ا", neighbours) <= length - 1200
    assert _run_length(edit, _shift(pair.edit, madd), "ا", neighbours) == length


def test_shaddah_added_lengthens_only_the_fricative_and_the_decoy_only_the_madd():
    clip, x = _recitation("g.wav", GEMINATE_REFERENCE)
    (pair,) = shaddah_added(clip, 1600, SALT)
    edit, _ = render(pair, EDIT, {"g.wav": x}.__getitem__, seed=1)
    decoy, _ = render(pair, DECOY, {"g.wav": x}.__getitem__, seed=1)
    carrier, neighbours = 9 * UNIT + UNIT // 2, "َُ"
    before = _run_length(x, carrier, "ش", neighbours)
    assert _run_length(edit, carrier, "ش", neighbours) >= before + 1200
    assert _run_length(decoy, carrier, "ش", neighbours) == before
    assert len(edit) == len(decoy) == len(x) + 1600


def test_a_swap_edit_says_the_other_letter_and_its_decoy_the_same_letter():
    target, x = _recitation("t.wav", SWAP_TARGET)
    donors, d = _recitation("d.wav", SWAP_DONORS)
    pair = next(p for p in consonant_swaps([target, donors], SALT)
                if p.audio_filename == "t.wav" and p.reference_index == 6)
    carrier = (6 * UNIT + UNIT // 8, 7 * UNIT - UNIT // 8)
    for role in (EDIT, DECOY):
        y, gain = render(pair, role, {"t.wav": x, "d.wav": d}.__getitem__, seed=1)
        change = pair.edit if role == EDIT else pair.decoy
        lo, hi = changed_region(change.start_sample, change.inserted_samples)
        assert len(y) == len(x) and gain == pytest.approx(1.0, abs=0.05)
        assert np.array_equal(y[:lo], x[:lo]) and np.array_equal(y[hi:], x[hi:])
        label = pair.edited if role == EDIT else pair.prescribed
        assert _letter_at(y, *carrier, "سص") == label


# --- the manifest ------------------------------------------------------------------------


def _staged(name: str, num_samples: int, reciter: int = RECITER) -> StagedClip:
    return StagedClip(name, 100, 1, reciter, "2:2", num_samples, SHA, ("synthetic_edit",))


def _items(pair, waveforms):
    clips = {name: _staged(name, len(w)) for name, w in waveforms.items()}
    items = []
    for role in (EDIT, DECOY):
        y, gain = render(pair, role, waveforms.__getitem__, seed=1)
        item_id = f"{pair.pair_id}:{role}"
        output = {"audio_filename": output_filename(item_id), "num_samples": len(y),
                  "audio_sha256": SHA}
        items.append(item_record(pair, role, clips, gain, output, seed=1))
    return items


def test_manifest_rows_carry_provenance_and_labels_that_follow_the_edit():
    target, x = _recitation("t.wav", SWAP_TARGET)
    donors, d = _recitation("d.wav", SWAP_DONORS)
    pair = next(p for p in consonant_swaps([target, donors], SALT) if p.audio_filename == "t.wav")
    edit, decoy = _items(pair, {"t.wav": x, "d.wav": d})
    check_manifest([edit, decoy])
    assert edit["label"] == pair.edited and decoy["label"] == pair.prescribed
    assert edit["change"]["donor"]["letter"] == pair.edited
    assert edit["change"]["donor"]["reciter_id"] == edit["source"]["reciter_id"]
    assert edit["output"]["audio_filename"].startswith("se_")
    assert "edit" not in edit["output"]["audio_filename"]


@pytest.mark.parametrize("mutate", [
    lambda e, d: d.update(label=e["label"]),
    lambda e, d: d.update(length_change=d["length_change"] + 1),
    lambda e, d: e.update(labelled_reference=e["reference"]),
    lambda e, d: d.update(reference_index=d["reference_index"] + 1),
    lambda e, d: e["change"]["donor"].update(reciter_id=RECITER + 1),
    lambda e, d: e["output"].update(audio_filename="edit.wav"),
    lambda e, d: e["output"].update(num_samples=e["output"]["num_samples"] + 1),
])
def test_check_manifest_refuses_a_broken_pair(mutate):
    target, x = _recitation("t.wav", SWAP_TARGET)
    donors, d = _recitation("d.wav", SWAP_DONORS)
    pair = next(p for p in consonant_swaps([target, donors], SALT) if p.audio_filename == "t.wav")
    edit, decoy = _items(pair, {"t.wav": x, "d.wav": d})
    mutate(edit, decoy)
    with pytest.raises(ValueError):
        check_manifest([edit, decoy])
    with pytest.raises(ValueError):
        check_manifest([edit])


def test_the_blind_check_mixes_edits_and_decoys_without_repeating_a_recitation():
    items = []
    for n in range(12):
        clip, x = _recitation(f"g{n:02d}.wav", GEMINATE_REFERENCE)
        for pair in shaddah_removed(clip, SALT) + shaddah_added(clip, 1600, SALT):
            items += _items(pair, {clip.audio_filename: x})
    sites = blind_check(items, per_operation=4)
    by_output = {i["output"]["audio_filename"]: i for i in items}
    chosen = [by_output[s.audio_filename] for s in sites]
    assert len(sites) == 8
    assert {i["role"] for i in chosen} == {EDIT, DECOY}
    assert len({i["source"]["audio_filename"] for i in chosen}) == 8
    for site in sites:
        parse_site(dataclasses.asdict(site), site.site_id)
        assert site.heard == PENDING and site.source == "synthetic_edit"
        assert site.site_id.startswith("synthetic_edit:") and "edit:" not in site.site_id[15:]
    assert blind_check(items, per_operation=4) == sites


def test_the_teacher_check_reads_the_label_at_the_carrier():
    held = GEMINATE_REFERENCE  # س doubled at 4
    single = held[:5] + held[6:]
    edit = {"labelled_reference": single, "reference_index": 4, "label": NOT_HELD}
    decoy = {"labelled_reference": held, "reference_index": 4, "label": HELD}
    assert teacher_agrees(edit, single) and not teacher_agrees(edit, held)
    assert teacher_agrees(decoy, held) and not teacher_agrees(decoy, single)


# --- the frame and disjointness ----------------------------------------------------------


def _index_row(name: str, reciter: int, shard: int, seconds: float = 8.0) -> IndexRow:
    return IndexRow(name, shard, 0, reciter, "2:2", seconds)


def test_the_frame_takes_only_reciters_absent_from_every_unseen_shard():
    unseen = [_index_row(f"u{s}.wav", 1, s) for s in h448_unseen_shards()]
    train = [_index_row("a.wav", 1, 22), _index_row("b.wav", 2, 23),
             _index_row("c.wav", 2, 24, seconds=60.0)]
    frame, excluded, evaluation = edit_frame(train, unseen)
    assert [r.audio_filename for r in frame] == ["b.wav"]
    assert excluded == {"reciter_in_unseen_shards": 1, "duration": 1}
    assert evaluation == [1]
    with pytest.raises(ValueError):
        edit_frame(train, unseen[1:])  # an unseen shard missing from the index
    with pytest.raises(ValueError):
        edit_frame([_index_row("d.wav", 3, 39)], unseen)


def test_committed_edit_sources_and_donors_are_disjoint_from_every_evaluation_item():
    items = read_items(EDITS_PATH)
    frame = read_frame()
    used = edit_reciters(items)
    assert used and used <= {int(r) for r in frame["clips_per_reciter"]}
    assert not used & evaluation_reciters()
    # The frame's record of evaluation reciters covers every registered evaluation clip.
    registry = load_staged_clips()
    assert {c.reciter_id for c in registry.values() if "synthetic_edit" not in c.uses} \
        <= set(frame["evaluation_reciters"])


def test_committed_items_use_registered_audio_and_the_worklist_names_them():
    items = read_items(EDITS_PATH)
    registry = load_staged_clips()
    for item in items:
        source = registry[item["source"]["audio_filename"]]
        assert (source.audio_sha256, source.reciter_id) == (
            item["source"]["audio_sha256"], item["source"]["reciter_id"])
        assert "synthetic_edit" in source.uses
        donor = item["change"]["donor"]
        if donor is not None:
            assert registry[donor["audio_filename"]].audio_sha256 == donor["audio_sha256"]
    sites = load_truth_sites(WORKLIST_PATH)
    outputs = {i["output"]["audio_filename"]: i for i in items}
    assert 24 <= len(sites) <= 36
    for site in sites:
        item = outputs[site.audio_filename]
        assert site.audio_sha256 == item["output"]["audio_sha256"]
        assert (site.reference, site.reference_index, site.mark, site.prescribed) == (
            item["reference"], item["reference_index"], item["mark"], item["prescribed"])
    assert blind_check(items) == sites
    assert json.loads(json.dumps(sites[0].site_id)) == sites[0].site_id


def test_exposure_rows_have_the_registry_shape_and_cover_every_edit_reciter():
    items = read_items(EDITS_PATH)
    rows = exposure_rows(items)
    assert set(rows) == {"synthetic_edit.source", "synthetic_edit.donor"}
    fields = {"audio_filename", "shard", "row_index", "reciter_id", "start_sample",
              "end_sample", "audio_sha256"}
    assert all(set(r) == fields for use in rows.values() for r in use)
    assert {r["reciter_id"] for use in rows.values() for r in use} == edit_reciters(items)
    assert all(r["start_sample"] is None for r in rows["synthetic_edit.source"])
    assert all(r["start_sample"] < r["end_sample"] for r in rows["synthetic_edit.donor"])
