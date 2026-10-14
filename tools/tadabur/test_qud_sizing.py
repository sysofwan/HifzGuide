"""Tests for tadabur.qud_sizing: carrier counting, outcome classes and the arithmetic."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from tadabur import qud_sizing as qz

# A geminate ط inside one word, then a ن geminate split over two words ("اَن" + "نَا").
REFERENCE = "سَططَ" + "اَن" + "نَا"
OFFSETS = [0, 5, 8, 11]


def test_carrier_counts_per_direction_letter_and_in_word_geminate():
    counts = qz.carrier_counts(REFERENCE, OFFSETS)
    assert counts["س→ش"] == counts["س→ص"] == 1
    assert counts["ط→ت"] == counts["ط→ق"] == 1  # the geminate is one carrier
    assert counts["letter:س"] == counts["letter:ط"] == 1  # a letter in two pairs, once
    assert counts[qz.GEMINATE] == 1  # the cross-word نن is left out


def test_quran_carriers_lists_what_the_phonetizer_cannot_handle():
    def reference_of(words):
        if words == ["bad"]:
            raise KeyError("leen")
        return REFERENCE, OFFSETS

    counts, unsupported = qz.quran_carriers(reference_of, [(1, 1, ["ok"]), (1, 2, ["bad"])])
    assert list(counts) == ["1:1"] and unsupported == ["1:2"]


@pytest.mark.parametrize(("h448", "base", "expected"), [
    ("ط", "ط", qz.AGREED),
    ("ق", "ط", qz.DISAGREED),
    (None, "ت", qz.DISAGREED),
    (None, "ط", qz.ABSTAINED),
    (None, None, qz.ABSTAINED),
])
def test_outcome_classes(h448, base, expected):
    assert qz.outcome("ط", h448, base) == expected


def test_covered_ayat_expands_ranges():
    surahs = {"1": {"ayat": "1-3,5"}, "114": {"ayat": "6"}}
    assert qz.covered_ayat(surahs) == ["1:1", "1:2", "1:3", "1:5", "114:6"]


def test_estimate_mushaf_counts_each_carrier_once_for_minutes():
    rates = {"letter:س": {"agreed": 0.9, "disagreed": 0.1, "abstained": 0.0},
             "letter:ط": {"agreed": 0.5, "disagreed": 0.5, "abstained": 0.0},
             qz.GEMINATE: {"agreed": 0.8, "disagreed": 0.0, "abstained": 0.2}}
    counts = Counter({"س→ش": 10, "س→ص": 10, "ط→ت": 4, "ط→ق": 4, "letter:س": 10,
                      "letter:ط": 4, qz.GEMINATE: 5})
    out = qz.estimate_mushaf(counts, rates, {"pair": 60.0, qz.GEMINATE: 30.0})
    assert out["directions"]["ط→ق"] == {"carriers": 4, "agreed": 2.0, "disagreed": 2.0,
                                        "abstained": 0.0}
    # 1 س + 2 ط disagreements at 60 s; 1 abstained geminate at 30 s.
    assert out["listening_minutes"] == {"disagreed": 3.0, "abstained": 0.5}


def test_seconds_per_site_from_the_committed_worklist_summary():
    summary = json.loads(qz.WORKLIST_SUMMARY_PATH.read_text(encoding="utf-8"))
    seconds = qz.seconds_per_site(summary)
    assert seconds == {"pair": round(91.9 * 60 / 422, 2), qz.GEMINATE: round(17.9 * 60 / 93, 2)}
