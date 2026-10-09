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
    Unusable,
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
    render_pair,
    stage_in_parallel,
    teacher_agrees,
    write_item,
)
from tadabur.truth_sites import HELD, NOT_HELD, PENDING, SHADDAH, load_truth_sites, parse_site
from tadabur.waveform_edits import (
    APERIODIC,
    FADE,
    DoesNotFit,
    PERIODIC,
    crop,
    join,
    replace_span,
    stretch,
)

UNIT = 3200  # samples per letter in the synthetic recitations (0.2 s)
SHA = "a" * 64
RECITER = 7


# --- synthetic recitations ---------------------------------------------------------------


_CHARS = "بلسصكشدمنرتفاَُِ"


def _pitch(char: str) -> float:
    """A distinct pitch per character, 15 Hz apart, all inside the voiced range."""
    return 100.0 + 15.0 * _CHARS.index(char)


def _recitation(
    name: str, reference: str, unit: int = UNIT, level: float = 0.3, noise: str = ""
) -> tuple[TimedClip, np.ndarray]:
    """A clip where each reference character is ``unit`` samples of its own tone (or of
    seeded white noise, for the characters in ``noise``), and the teacher emitted each one
    over the middle half of its stretch (a perfect decode)."""
    t = np.arange(unit) / 16000.0
    rng = np.random.default_rng(0)
    waveform = np.concatenate([
        level * (rng.uniform(-1, 1, unit) if c in noise else
                 np.sin(2 * np.pi * _pitch(c) * (t + i * unit / 16000.0)))
        for i, c in enumerate(reference)
    ]).astype(np.float32)
    tokens = tuple(Token(c, i * unit + unit // 4, i * unit + 3 * unit // 4)
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


def _sine(hz: float, n: int = 24000, level: float = 0.5) -> np.ndarray:
    return (level * np.sin(2 * np.pi * hz * np.arange(n) / 16000.0)).astype(np.float32)


def _min_period_rms(y: np.ndarray, p: int) -> float:
    """The quietest one-period window of ``y``: a join that cancels the signal shows here."""
    energy = np.convolve(np.square(y.astype(np.float64)), np.ones(p), mode="valid") / p
    return float(np.sqrt(energy.min()))


def test_an_aperiodic_crop_changes_only_the_crossfade_around_the_cut():
    x = np.random.default_rng(0).standard_normal(20000).astype(np.float32)
    r = crop(x, 8000, 1500)
    half = FADE // 2
    assert (r.path, r.changed) == (APERIODIC, (8000, 9500))
    assert len(r.samples) == len(x) - 1500
    assert np.array_equal(r.samples[:8000 - half], x[:8000 - half])
    assert np.array_equal(r.samples[8000 + half:], x[9500 + half:])


def test_replace_span_changes_only_the_span_and_its_crossfades():
    x = np.random.default_rng(1).standard_normal(20000).astype(np.float32)
    material = np.random.default_rng(2).standard_normal(3000 + FADE).astype(np.float32)
    y = replace_span(x, 6000, 8000, material)
    half = FADE // 2
    assert len(y) == len(x) + 1000
    assert np.array_equal(y[:6000 - half], x[:6000 - half])
    assert np.array_equal(y[9000 + half:], x[8000 + half:])
    assert np.array_equal(y[6000 + half:9000 - half], material[FADE:-FADE])


def test_join_crossfades_and_refuses_an_overlap_longer_than_either_side():
    a, b = np.ones(400, np.float32), np.zeros(400, np.float32)
    y = join(a, b)
    assert len(y) == 800 - FADE and y[0] == 1 and y[-1] == 0
    assert np.all(np.diff(y) <= 0)
    with pytest.raises(ValueError):
        join(a[:10], b)


@pytest.mark.parametrize("hz", [125.0, 133.0, 210.0])
@pytest.mark.parametrize("length", [2880, 2000, 1111])
def test_a_periodic_stretch_keeps_its_phase_at_both_joins(hz, length):
    """Not a whole number of periods (2,880 samples at 125 Hz is 22.5): before the fix the
    second join met the clip in opposite phase and cancelled two thirds of the energy."""
    x = _sine(hz)
    r = stretch(x, 9000, length, (6000, 12000), seed=0)
    lo, hi = r.changed
    assert r.path == PERIODIC and len(r.samples) == len(x) + length
    assert np.array_equal(r.samples[:lo - FADE // 2], x[:lo - FADE // 2])
    assert np.array_equal(r.samples[hi + length:], x[hi:])
    full = 0.5 / np.sqrt(2)
    assert _min_period_rms(r.samples, r.period) >= 0.9 * full
    assert np.max(np.abs(np.diff(r.samples))) <= np.max(np.abs(np.diff(x))) * 1.1


@pytest.mark.parametrize("hz", [125.0, 133.0, 210.0])
@pytest.mark.parametrize("length", [2880, 2000, 1111])
def test_a_periodic_crop_keeps_its_phase_at_the_join(hz, length):
    x = _sine(hz)
    r = crop(x, 9000, length)
    lo, hi = r.changed
    assert r.path == PERIODIC and len(r.samples) == len(x) - length
    assert np.array_equal(r.samples[:lo - FADE // 2], x[:lo - FADE // 2])
    assert np.array_equal(r.samples[hi - length:], x[hi:])
    assert _min_period_rms(r.samples, r.period) >= 0.9 * 0.5 / np.sqrt(2)


def test_a_short_region_shrinks_its_unit_rather_than_taking_a_truncated_one():
    """A 640-sample region at 100 Hz: aligning a four-period unit to the clip's phase
    pushes it past the region's end. NumPy would hand back 560 samples and advance by 2.5
    periods (min-period energy 22%); the unit must shrink to whole periods instead."""
    x = _sine(100.0)
    r = stretch(x, 8320, 2880, (8000, 8640), seed=0)
    assert r.path == PERIODIC and r.period == 160
    assert len(r.samples) == len(x) + 2880
    assert _min_period_rms(r.samples, r.period) >= 0.9 * 0.5 / np.sqrt(2)


def test_a_span_outside_the_waveform_is_refused_not_truncated():
    x = _sine(125.0, n=4000)
    with pytest.raises(DoesNotFit):
        stretch(x, 2000, 1000, (3500, 4600), seed=0)  # region runs past the end
    with pytest.raises(DoesNotFit):
        crop(x, 3000, 1500)
    with pytest.raises(DoesNotFit):
        stretch(x, 2000, 1000, (1000, 1200), seed=0)  # shorter than one unit


def test_an_aperiodic_stretch_is_seeded_and_does_not_simply_repeat():
    x = np.random.default_rng(3).standard_normal(8000).astype(np.float32)
    a = stretch(x, 3000, 4000, (1000, 3000), seed=5)
    assert a.path == APERIODIC and a.changed == (3000, 3000)
    assert np.array_equal(a.samples, stretch(x, 3000, 4000, (1000, 3000), seed=5).samples)
    assert not np.array_equal(a.samples, stretch(x, 3000, 4000, (1000, 3000), seed=6).samples)


def test_an_item_that_would_clip_is_never_written(tmp_path):
    with pytest.raises(ValueError):
        write_item(np.array([0.5, 1.2, -0.3], np.float32), tmp_path / "x.wav")
    assert not (tmp_path / "x.wav").exists()
    output = write_item(np.array([0.5, -0.9, 0.1] * 100, np.float32), tmp_path / "y.wav")
    assert output["num_samples"] == 300 and output["peak"] == pytest.approx(0.9)


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


def _shift(change, position):
    """Where a source sample at ``position`` lands after ``change`` (before or after it)."""
    return position if position < change.start_sample else position + change.length_change


def _assert_untouched_outside(item, x, change):
    """The item is its source outside the span it replaced (and half a crossfade)."""
    lo, hi = item.rendered.changed
    half = FADE // 2
    y = item.rendered.samples
    assert len(y) == len(x) + change.length_change
    assert np.array_equal(y[:lo - half], x[:lo - half])
    assert np.array_equal(y[hi + change.length_change + half:], x[hi + half:])


def test_shaddah_removed_shortens_only_the_hold_and_the_decoy_only_the_madd():
    clip, x = _recitation("g.wav", GEMINATE_REFERENCE)
    (pair,) = shaddah_removed(clip, SALT)
    rendered = render_pair(pair, {"g.wav": x}.__getitem__)
    for role, item in rendered.items():
        _assert_untouched_outside(item, x, pair.edit if role == EDIT else pair.decoy)
        assert item.gain is None
    assert rendered[EDIT].rendered.path == rendered[DECOY].rendered.path
    held, neighbours = 5 * UNIT, "َك"  # the centre of the doubled س
    hold = _run_length(x, held, "س", neighbours)
    edit, decoy = rendered[EDIT].rendered.samples, rendered[DECOY].rendered.samples
    assert _run_length(edit, pair.edit.start_sample, "س", neighbours) <= hold - 1200
    assert _run_length(decoy, held, "س", neighbours) == hold
    madd, neighbours = 19 * UNIT, "ِن"
    length = _run_length(x, madd, "ا", neighbours)
    assert _run_length(decoy, _shift(pair.decoy, madd - UNIT), "ا", neighbours) <= length - 1200
    assert _run_length(edit, _shift(pair.edit, madd), "ا", neighbours) == length


def test_shaddah_added_lengthens_only_the_fricative_and_the_decoy_only_the_madd():
    clip, x = _recitation("g.wav", GEMINATE_REFERENCE)
    (pair,) = shaddah_added(clip, 1600, SALT)
    rendered = render_pair(pair, {"g.wav": x}.__getitem__)
    for role, item in rendered.items():
        _assert_untouched_outside(item, x, pair.edit if role == EDIT else pair.decoy)
    edit, decoy = rendered[EDIT].rendered.samples, rendered[DECOY].rendered.samples
    carrier, neighbours = 9 * UNIT + UNIT // 2, "َُ"
    before = _run_length(x, carrier, "ش", neighbours)
    assert _run_length(edit, carrier, "ش", neighbours) >= before + 1200
    assert _run_length(decoy, carrier, "ش", neighbours) == before
    assert len(edit) == len(decoy) == len(x) + 1600


def test_a_pair_whose_items_would_be_made_differently_is_refused():
    """A noisy fricative stretched against a voiced madd: the two would take different
    methods, a difference a model could learn instead of the edit's identity."""
    clip, x = _recitation("g.wav", GEMINATE_REFERENCE, noise="ش")
    (pair,) = shaddah_added(clip, 1600, SALT)
    with pytest.raises(Unusable) as refusal:
        render_pair(pair, {"g.wav": x}.__getitem__)
    assert refusal.value.reason == "render_path"


def _swap(donor_unit: int = UNIT, donor_level: float = 0.3, target_level: float = 0.3):
    target, x = _recitation("t.wav", SWAP_TARGET, level=target_level)
    donors, d = _recitation("d.wav", SWAP_DONORS, unit=donor_unit, level=donor_level)
    pairs = [p for p in consonant_swaps([target, donors], SALT)
             if p.audio_filename == "t.wav" and p.reference_index == 6]
    return pairs, {"t.wav": x, "d.wav": d}


@pytest.mark.parametrize("donor_unit", [2800, UNIT, 3600])
def test_a_swap_brings_in_the_carrier_and_haraka_and_no_other_phoneme(donor_unit):
    """Donors recited faster or slower than the target still fit: the edit says the other
    letter, the decoy the same one, and both keep the target's preceding letter, its
    haraka and the letter after it."""
    (pair,), audio = _swap(donor_unit)
    x = audio["t.wav"]
    rendered = render_pair(pair, audio.__getitem__)
    for role, item in rendered.items():
        change = pair.edit if role == EDIT else pair.decoy
        _assert_untouched_outside(item, x, change)
        y = item.rendered.samples
        label = pair.edited if role == EDIT else pair.prescribed
        letters = "سصَُِمف"
        middle = UNIT // 4, 3 * UNIT // 4
        assert _letter_at(y, 5 * UNIT + middle[0], 5 * UNIT + middle[1], letters) == "ُ"
        assert _letter_at(y, 6 * UNIT + middle[0], 6 * UNIT + middle[1], letters) == label
        assert _letter_at(y, 7 * UNIT + middle[0], 7 * UNIT + middle[1], letters) == "َ"
        assert _letter_at(y, 8 * UNIT + middle[0], 8 * UNIT + middle[1], letters) == "م"
        assert item.gain == pytest.approx(1.0, abs=0.1)
        donor = change.donor
        s = donor_unit
        # The window, crossfade context included, stays between the donor's previous
        # emission and the emission after its haraka, and reaches into the haraka.
        j = donor.reference_index
        assert donor.start_sample - FADE // 2 >= (j - 1) * s + 3 * s // 4
        assert donor.end_sample >= (j + 1) * s + s // 2
        assert donor.end_sample + FADE // 2 <= (j + 2) * s + s // 4


def test_a_donor_whose_timing_cannot_fit_is_incompatible():
    pairs, _ = _swap(donor_unit=2000)
    assert pairs == []


def test_a_donor_far_louder_or_quieter_than_the_span_is_refused():
    (pair,), audio = _swap(donor_level=0.9)
    with pytest.raises(Unusable) as refusal:
        render_pair(pair, audio.__getitem__)
    assert refusal.value.reason == "donor_level"


def test_a_pair_that_would_clip_is_refused():
    (pair,), audio = _swap(donor_level=0.5, target_level=0.95)
    spike = audio["d.wav"].copy()
    spike[pair.edit.donor.start_sample + 200] = 0.6  # 0.6 x the ~1.9 level match > 1
    audio["d.wav"] = spike
    with pytest.raises(Unusable) as refusal:
        render_pair(pair, audio.__getitem__)
    assert refusal.value.reason == "peak"


# --- the manifest ------------------------------------------------------------------------


def _staged(name: str, num_samples: int, reciter: int = RECITER) -> StagedClip:
    return StagedClip(name, 100, 1, reciter, "2:2", num_samples, SHA, ("synthetic_edit",))


def _items(pair, waveforms):
    clips = {name: _staged(name, len(w)) for name, w in waveforms.items()}
    items = []
    for role, item in render_pair(pair, waveforms.__getitem__).items():
        y = item.rendered.samples
        item_id = f"{pair.pair_id}:{role}"
        output = {"audio_filename": output_filename(item_id), "num_samples": len(y),
                  "audio_sha256": SHA, "peak": float(np.max(np.abs(y)))}
        items.append(item_record(pair, role, clips, item, output))
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
    lambda e, d: e["output"].update(peak=1.01),
    lambda e, d: d["render"].update(path="periodic"),
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


def _clip_entry(name: str, sha: str) -> StagedClip:
    return StagedClip(name, 100, 1, RECITER, "2:2", 16000, sha * 64, ("synthetic_edit",))


def test_an_interrupted_parallel_staging_keeps_every_recorded_checksum():
    """Worker 0 stages a new clip, checkpoints and dies; worker 1 holds a clip that was
    already registered and never checkpoints. No write may drop that clip's checksum,
    and a resume restores the registry exactly."""
    a, b, c = _clip_entry("a.wav", "a"), _clip_entry("b.wav", "b"), _clip_entry("c.wav", "c")
    stale = _clip_entry("gone.wav", "d")  # registered once, no longer requested
    known = {"a.wav": a, "b.wav": b, "gone.wav": stale}
    groups = [{"a.wav": frozenset(), "c.wav": frozenset()}, {"b.wav": frozenset()}]
    writes: list[dict] = []

    def interrupted(requests, checkpoint):
        if "c.wav" in requests:
            checkpoint({"a.wav": a, "c.wav": c})
            raise RuntimeError("worker died")
        return {"b.wav": b}, []

    with pytest.raises(RuntimeError):
        stage_in_parallel(groups, known, interrupted, writes.append, workers=2)
    assert writes and all({"a.wav", "b.wav", "gone.wav"} <= set(w) for w in writes)
    assert writes[-1]["b.wav"] == b and writes[-1]["c.wav"] == c

    def resumed(requests, checkpoint):
        found = {name: writes[-1][name] for name in requests}
        checkpoint(found)
        return found, []

    resumed_writes: list[dict] = []
    staged, missing = stage_in_parallel(groups, writes[-1], resumed, resumed_writes.append, 2)
    assert staged == {"a.wav": a, "b.wav": b, "c.wav": c} and missing == []
    assert all("gone.wav" in w for w in resumed_writes[:-1])
    assert resumed_writes[-1] == staged  # pruned only once every worker finished


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
