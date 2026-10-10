"""Record Muraja's own grades for the parity cases into ``parity_fixtures.json``.

The cases are GPT-6 Astra's counterexamples to the dropped Python port of Muraja's grading
(#123 review, round 1), moved onto a real ayah (2:80) so the real engine can replay them; the
soft-pair flag ADR-0012 switches; and one case per remaining cell of ADR-0011 §2's amended table.
The expected values are whatever the compiled Swift returns; ``training/test_muraja_harness.py``
replays every case and requires identical output.

Usage (from ``tools/``, with the harness built by ``muraja_harness/build.sh``)::

    python -m muraja_harness.make_parity_fixtures
"""

from __future__ import annotations

import json
from pathlib import Path

from training.muraja_policy import SAMPLE_RATE, clusters, require_harness, single_decode_checks

FIXTURES_PATH = Path(__file__).parent / "parity_fixtures.json"

#: 2:80's phoneme words as Muraja's quran.db holds them (1-based), joined without spaces as the
#: model emits them.
AYAH = (
    "وَقَاالُۥۥ لَںںں تَمَسسَنَ ننننَاارُ ءِللَاااا ءَييَاامَ ممممَعدُۥۥدَتَںںں قُل ءَتتَخَذتُم "
    "عِںںںدَ للَااهِ عَهدَںںں فَلَيي يُخلِفَ للَااهُ عَهدَهُۥۥۥۥ ءَم تَقُۥۥلُۥۥنَ عَلَ للَااهِ مَاا لَاا "
    "تَعلَمُۥۥن"
).split(" ")


def words(first: int, last: int, **replaced: str) -> str:
    """Words ``first..last`` (1-based) run together, some replaced: ``w3="..."``."""
    return "".join(replaced.get(f"w{n}", AYAH[n - 1]) for n in range(first, last + 1))


BALANCED = {"name": "balanced", "mode": "balanced"}
STRICT = {"name": "strict", "mode": "strict"}
ALLOWANCES_OFF = {"name": "allowances_off", "mode": "balanced", "soft_pairs_enabled": False, "shaddah_suppression": False}
SOFT_PAIRS_OFF = {"name": "soft_pairs_off", "mode": "balanced", "soft_pairs_enabled": False}
SHADDAH_OFF = {"name": "shaddah_off", "mode": "balanced", "shaddah_suppression": False}


#: Swift Characters the cadence spreads over one second when a case replays a whole decode;
#: about a reciter's pace in the staged clips.
CHARACTERS_PER_SECOND = 8


def recited(decode: str, pace: float = CHARACTERS_PER_SECOND) -> list[dict]:
    """The checks the app's cadence runs over ``decode`` recited at ``pace`` Characters a second
    (``single_decode_checks``)."""
    seconds = len(clusters(decode)) / pace
    return single_decode_checks(decode, round(seconds * SAMPLE_RATE))


def preview(text: str) -> dict:
    return {"hop": "", "overlap": text, "flush": False}


def flush(text: str) -> dict:
    return {"hop": text, "overlap": "", "flush": True}


BARE_DOUBLE = "تَمَسسنَ"  # the geminate decoded as a bare double, its fatha lost

CASES = [
    {
        # Astra P1-1: a geminate decoded as a bare double merges into one group, so a gap
        # discards the group's tashkeel; graded from Muraja's own alignment.
        "case": "bare_double_geminate",
        "item": "2:80 w3 bare double",
        "start_word": 1,
        "report_words": [3],
        "checks": recited(words(1, 7, w3=BARE_DOUBLE)),
        "scorings": [BALANCED, STRICT, SHADDAH_OFF],
    },
    {
        # Astra P1-2: a check that has heard only part of a word grades it as Muraja's overlap
        # and coverage say, and the end word holds a non-correct grade back.
        "case": "partial_word",
        "item": "2:80 half of w4",
        "start_word": 1,
        "report_words": [3, 4],
        # recited slowly, so the reader is placed before the session ends
        "checks": recited(words(1, 3) + "ننن", pace=3),
        "scorings": [BALANCED, STRICT],
    },
    {
        # Astra P1-3: placement keeps harakat; للَااهُ (w15) is not للَااهِ (w11, w20).
        "case": "placement_by_harakat",
        "item": "2:80 from w14",
        "start_word": 14,
        "report_words": [11, 15, 20],
        "checks": recited(words(14, 18)),
        "scorings": [BALANCED],
    },
    {
        # Astra P1-4: switching ADR-0012's allowances off keeps the protected exemptions: a
        # dropped haraka on an interior و stays unflagged.
        "case": "protected_waw_exemption",
        "item": "2:80 w1 bare waw",
        "start_word": 1,
        "report_words": [1],
        "checks": recited(words(1, 7, w1="وقَاالُۥۥ")),
        "scorings": [BALANCED, ALLOWANCES_OFF],
    },
    {
        # Astra P2-6: low grades wait in the hold buffer, and a held grade is replaced only by a
        # higher rank; the kept evidence is the engine's.
        "case": "hold_buffer",
        "item": "2:80 w9 heard wrong twice",
        "start_word": 1,
        "report_words": [9],
        # once the reader is placed, w9 is heard badly (wrong, score 0.235), then a little less
        # badly (wrong, 0.322) as the ayah goes on: wrong grades wait in the hold buffer and
        # the engine's skip marking replaces them
        "checks": recited(words(1, 8))[:-1] + [
            preview(words(5, 8) + "ءَسسَعَصطُن" + words(10, 10)),
            *(preview(words(5, 8) + "ءَتسَعَصطُن" + words(10, n)) for n in range(11, 15)),
            flush(words(5, 8) + "ءَتسَعَصطُن" + words(10, 15)),
        ],
        "scorings": [BALANCED],
    },
    {
        # ADR-0012's soft-pair flag: ذ heard as ز in ءَتتَخَذتُم, forgiven in balanced only
        # while softPairsEnabled holds.
        "case": "soft_pair_flag",
        "item": "2:80 w9 zay",
        "start_word": 1,
        "report_words": [9],
        "checks": recited(words(1, 12, w9="ءَتتَخَزتُم")),
        "scorings": [BALANCED, SOFT_PAIRS_OFF, STRICT],
    },
]


LENIENT = {"name": "lenient", "mode": "lenient"}
MODES = [STRICT, BALANCED, LENIENT]


def table_cell(case: str, word: int, heard: str) -> dict:
    """One cell of ADR-0011 §2's amended table: ``word`` heard as ``heard``, in every mode."""
    return {
        "case": case,
        "item": f"2:80 w{word} {case}",
        "start_word": 1,
        "report_words": [word],
        "checks": recited(words(1, min(word + 3, len(AYAH)), **{f"w{word}": heard})),
        "scorings": MODES,
    }


#: The other cells of the amended table (ADR-0011 §2, 2026-10-09).
CASES += [
    table_cell("missing_haraka", 3, "تمَسسَنَ"),
    table_cell("wrong_haraka", 3, "تِمَسسَنَ"),
    table_cell("haraka_added_at_sukun", 9, "ءَتتَخَذَتُم"),
    table_cell("word_initial_geminate_single", 11, "لَااهِ"),
    table_cell("single_decoded_double", 14, "يُخللِفَ"),
    table_cell("dhal_heard_as_zah", 9, "ءَتتَخَظتُم"),
]


def request(case: dict) -> dict:
    return {key: value for key, value in case.items() if key != "case"} | {"surah": 2, "ayah": 80}


def main() -> None:
    harness = require_harness()
    build, results = harness.run(request(case) for case in CASES)
    by_item: dict[str, list[dict]] = {}
    for result in results:
        by_item.setdefault(result["item"], []).append(result)
    fixtures = {
        "build": build,
        "cases": [{**case, "surah": 2, "ayah": 80, "expected": by_item[case["item"]]} for case in CASES],
    }
    FIXTURES_PATH.write_text(json.dumps(fixtures, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {FIXTURES_PATH}")


if __name__ == "__main__":
    main()
