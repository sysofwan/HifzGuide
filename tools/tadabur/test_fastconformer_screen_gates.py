"""Tests for #267's gates (steps 1-2): each is a refusal with a message, not a judgement.

* identity: the revision and the full 64-hex ``.nemo`` SHA-256 (``d29d19d7…3e31``, the LFS oid at revision ``7f32349d``) are pinned;
  matching only the prefix and suffix the issue quotes is not a match;
* mapping gate: at least 95% of sites and edits located;
* parity gate (only if ONNX is used): at least 50 Ship A clips, at least 49 with equal text;
* card sanity: CTC WER on a 200-clip EveryAyah sample, stop only above 2x the card's 7.61.
"""

from __future__ import annotations

import re

import pytest

from tadabur.fastconformer_screen.gates import (
    CARD_CLIPS,
    CARD_WER,
    NEMO_SHA256,
    REVISION,
    ScreenStop,
    card_gate,
    mapping_gate,
    parity_gate,
    require_identity,
)


def test_screen_stop_is_a_runtime_error():
    assert issubclass(ScreenStop, RuntimeError)


# --- identity ---------------------------------------------------------------------------------


def test_the_pinned_identity_is_complete():
    assert re.fullmatch(r"[0-9a-f]{64}", NEMO_SHA256)
    assert NEMO_SHA256.startswith("d29d19d7") and NEMO_SHA256.endswith("3e31")
    assert re.fullmatch(r"[0-9a-f]{40}", REVISION)       # a Hub commit, never a branch name
    assert REVISION.startswith("7f32349d")


def test_the_pinned_identity_passes():
    require_identity({"revision": REVISION, "nemo_sha256": NEMO_SHA256})


def test_a_hash_with_only_the_quoted_prefix_and_suffix_is_refused():
    forged = NEMO_SHA256[:8] + "0" * 48 + NEMO_SHA256[-8:]
    with pytest.raises(ScreenStop, match="sha"):
        require_identity({"revision": REVISION, "nemo_sha256": forged})


def test_another_revision_is_refused():
    with pytest.raises(ScreenStop, match="revision"):
        require_identity({"revision": "0" * 40, "nemo_sha256": NEMO_SHA256})


@pytest.mark.parametrize("missing", ["revision", "nemo_sha256"])
def test_an_identity_missing_a_pin_is_refused(missing):
    record = {"revision": REVISION, "nemo_sha256": NEMO_SHA256}
    del record[missing]
    with pytest.raises(ScreenStop, match=missing):
        require_identity(record)


# --- mapping gate -----------------------------------------------------------------------------


def test_exactly_95_percent_located_passes():
    mapping_gate(located=950, total=1000)


def test_just_under_95_percent_located_stops_with_the_counts():
    with pytest.raises(ScreenStop, match="949"):
        mapping_gate(located=949, total=1000)


def test_nothing_to_locate_stops():
    with pytest.raises(ScreenStop):
        mapping_gate(located=0, total=0)


# --- parity gate (ONNX only) ----------------------------------------------------------------------


def test_parity_with_fifty_clips_and_forty_nine_equal_passes():
    parity_gate(equal=49, clips=50)


def test_parity_with_one_more_difference_stops():
    with pytest.raises(ScreenStop, match="48"):
        parity_gate(equal=48, clips=50)


def test_parity_on_fewer_than_fifty_clips_stops_even_if_all_equal():
    with pytest.raises(ScreenStop, match="49"):
        parity_gate(equal=49, clips=49)


# --- the card sanity line ---------------------------------------------------------------------------


def test_the_cards_ctc_wer_is_7_61_on_a_200_clip_sample():
    assert (CARD_WER, CARD_CLIPS) == (7.61, 200)


def test_a_wer_of_exactly_twice_the_cards_passes():
    card_gate(wer=2 * CARD_WER, clips=CARD_CLIPS)


def test_a_wer_above_twice_the_cards_stops():
    with pytest.raises(ScreenStop, match="WER"):
        card_gate(wer=2 * CARD_WER + 0.01, clips=CARD_CLIPS)


def test_the_card_line_on_another_sample_size_is_refused():
    with pytest.raises(ScreenStop, match="200"):
        card_gate(wer=7.0, clips=199)
