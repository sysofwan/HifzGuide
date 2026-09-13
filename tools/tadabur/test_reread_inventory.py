"""Tests for the pre-stream inventory of re-read material on disk."""

from __future__ import annotations

from tadabur.clip_status import ClipStatus
from tadabur.manifest import ManifestRecord
from tadabur.reread_inventory import take_inventory

# Same 20-phoneme reference as tadabur.test_rejects; a decode repeating phonemes 5..10
# is a clean re-read, the unrepeated reference is a clean pass.
REFERENCE = "بتثجحخدذرزسشصضطظعغفق"
RE_READ_DECODE = REFERENCE[:10] + REFERENCE[5:10] + REFERENCE[10:]
REFERENCES = {"3:82": REFERENCE, "3:83": REFERENCE}


def _record(name: str, decode: str = REFERENCE, reciter: int = 1, ayah: str = "3:82"):
    return ManifestRecord(
        audio_filename=name,
        surah_ayah=ayah,
        match_ratio=0.9,
        ayah_duration_s=10.0,
        reciter_id=reciter,
        predicted_phonemes=decode,
    )


def _status(name: str, re_reads: int = 0, skip: str | None = None, duration: float = 10.0):
    return ClipStatus(
        audio_filename=name,
        surah_ayah="3:82",
        reciter_id=1,
        n_words=5,
        duration_s=duration,
        re_reads=re_reads,
        skip_reason=skip,
    )


def test_counts_clips_reciters_and_the_re_read_subset():
    records = [
        _record("a.wav", reciter=1),
        _record("b.wav", reciter=2, ayah="3:83"),
        _record("c.wav", reciter=2),
    ]
    statuses = [_status("a.wav", re_reads=1, duration=12.0),
                _status("b.wav", re_reads=2, duration=8.0),
                _status("c.wav")]

    inventory = take_inventory(records, statuses, None, REFERENCES)

    assert inventory.clips == 3
    assert inventory.reciters == 2
    assert inventory.re_read_clips == 2
    assert inventory.re_read_reciters == 2
    assert inventory.re_read_ayat == 2
    assert inventory.re_read_audio_seconds == 20.0
    assert inventory.re_reads_histogram == {"0": 1, "1": 1, "2": 1}


def test_audio_presence_is_checked_against_the_staged_directory(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "a.wav").write_bytes(b"")
    records = [_record("a.wav"), _record("gone.wav")]
    statuses = [_status("a.wav", re_reads=1), _status("gone.wav", re_reads=1)]

    inventory = take_inventory(records, statuses, clips_dir, REFERENCES)

    # A manifest row with no WAV is not a usable clip and is counted as such.
    assert inventory.clips == 2
    assert inventory.clips_with_audio == 1
    assert inventory.re_read_clips == 2
    assert inventory.re_read_clips_with_audio == 1


def test_re_gating_recovers_the_insertion_run_the_run_discarded():
    # The whole point of keeping predicted_phonemes: the repeat length is recomputable
    # offline, without a GPU and without the original GateResult.
    records = [_record("clean.wav"), _record("reread.wav", decode=RE_READ_DECODE)]
    statuses = [_status("clean.wav"), _status("reread.wav", re_reads=1)]

    inventory = take_inventory(records, statuses, None, REFERENCES)

    assert inventory.insertion_run_histogram == {"0": 1, "5": 1}
    assert inventory.re_read_insertion_run_histogram == {"5": 1}


def test_a_segmenter_re_read_flag_need_not_be_a_phoneme_space_repeat():
    # The disagreement the report turns on: the segmenter flags a re-read seam on a
    # clip whose decode contains no repeated span at all.
    records = [_record("flagged.wav")]
    statuses = [_status("flagged.wav", re_reads=1)]

    inventory = take_inventory(records, statuses, None, REFERENCES)

    assert inventory.re_read_clips == 1
    assert inventory.re_read_insertion_run_histogram == {"0": 1}
    assert inventory.clean_re_reads == 0


def test_skip_reasons_are_tallied_and_repeated_recitation_singled_out():
    records = [_record("a.wav"), _record("b.wav"), _record("c.wav")]
    statuses = [
        _status("a.wav"),
        _status("b.wav", skip="repeated_recitation"),
        _status("c.wav", skip="phonetizer_unsupported"),
    ]

    inventory = take_inventory(records, statuses, None, REFERENCES)

    assert inventory.skip_reasons == {
        "none": 1, "phonetizer_unsupported": 1, "repeated_recitation": 1
    }
    assert inventory.repeated_recitation_clips == 1
    assert inventory.repeated_recitation_with_audio == 1


def test_a_record_without_a_status_row_still_counts_in_the_totals():
    # It was filtered, it just was never segmented; dropping it would understate the
    # subset and the insertion-run histogram it contributes to.
    records = [_record("scored.wav", decode=RE_READ_DECODE), _record("segmented.wav")]
    statuses = [_status("segmented.wav")]

    inventory = take_inventory(records, statuses, None, REFERENCES)

    assert inventory.clips == 2
    assert inventory.insertion_run_histogram == {"0": 1, "5": 1}
    assert inventory.re_reads_histogram == {"0": 1}


def test_a_passing_subset_yields_no_clean_re_reads():
    # The structural claim behind the recommendation: a clip clearing the gate's
    # insertion-run bar is what "passing" means, so no passer can meet the mining
    # predicate however many re-read seams the segmenter flagged on it.
    records = [_record(f"p{i}.wav") for i in range(5)]
    statuses = [_status(f"p{i}.wav", re_reads=1) for i in range(5)]

    inventory = take_inventory(records, statuses, None, REFERENCES)

    assert inventory.re_read_clips == 5
    assert inventory.clean_re_reads == 0
