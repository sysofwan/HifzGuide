"""Tests for cutting the ``h448`` b=0 stream of a whole clip into its pool segments (#87)."""

from __future__ import annotations

from training.decoding import Emission

from tadabur.pool_stream import SEGMENT_MARGIN_S, commit_time, segment_decodes
from tadabur.phoneme_vocab import PHONEME_ID_TO_CHAR

BA, TA = PHONEME_ID_TO_CHAR.index("ب"), PHONEME_ID_TO_CHAR.index("ت")


def _emission(token: int, window: int, step: int) -> Emission:
    return Emission(token_id=token, window=window, start_step=step, end_step=step,
                    is_final_window=False)


def test_a_token_is_timed_by_its_window_and_the_centre_of_its_run():
    assert commit_time(_emission(BA, 3, 0)) == 3.02  # window 3 starts at 3 s; step 0 is 40 ms
    assert commit_time(Emission(BA, 0, 10, 14, False)) == 0.5


def test_each_segment_keeps_the_tokens_committed_inside_it_and_its_margin():
    emissions = [_emission(BA, 0, 0), _emission(TA, 3, 0), _emission(BA, 5, 25)]
    spans = {"a#0": (16000, 32000), "a#1": (80000, 96000)}  # 1-2 s and 5-6 s
    margin_steps = round(SEGMENT_MARGIN_S / 0.04)
    assert segment_decodes(emissions, spans) == {"a#0": "", "a#1": "ب"}
    near = [_emission(TA, 0, 25 - margin_steps)]  # just inside the 0.5 s before segment a#0
    assert segment_decodes(near, spans)["a#0"] == "ت"
