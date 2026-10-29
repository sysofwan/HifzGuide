"""The round-2 catch-loss audit's scorer (#225): ``tadabur.catch_loss_audit.score``.

``score(directory)`` reads ``worklist.jsonl`` (only each row's ``id``), ``key.jsonl`` and the
owner's ``verdicts.jsonl`` (the listening UI's word question: ``correct``, ``mistake`` or
``unclear``) from ``directory``, writes ``result.json`` there and returns it. The rules are the
pre-registration's:

* **gate**: among the removed stratum's decisive verdicts (``correct`` or ``mistake``), the share
  that is ``mistake`` is at most 20% (exactly 20% passes);
* **cannot certify** when fewer than 80 removed items are decisive — an absolute floor, whatever
  the stratum's size — and it wins over both pass and fail;
* **reported**: the added stratum's share, both strata by cause, unclear counts, and the items
  whose flag is a harder pair (both letters of a substitution in {س, ص, ث}; the flagging model's
  substitutions: C95's for removed, the candidate's for added).

``result`` holds ``verdict`` (``pass``, ``fail`` or ``cannot_certify``), ``bars``, ``removed`` and
``added`` (each ``{decisive, real, correct, unclear, real_share, wilson}``), ``by_cause[stratum]
[cause]`` (the same counts), ``noisy[stratum]`` (``{items, decisive, real}``) and ``caveats``.
"""

from __future__ import annotations

import json

import pytest

from tadabur import catch_loss_audit as audit
from tadabur.listening_session import Verdict, write_verdicts

REFUSED = (SystemExit, ValueError)
CONSONANT = "other_consonant"
MISTAKE, CORRECT, UNCLEAR = "mistake", "correct", "unclear"


def _item(stratum, heard, causes=(CONSONANT,), consonants=()):
    return stratum, heard, list(causes), [list(pair) for pair in consonants]


def _stratum(stratum, real, correct, unclear, **fields):
    return [_item(stratum, heard, **fields)
            for heard, count in ((MISTAKE, real), (CORRECT, correct), (UNCLEAR, unclear))
            for _ in range(count)]


def _removed(real=0, correct=0, unclear=0, **fields):
    return _stratum("removed", real, correct, unclear, **fields)


def _added(real=0, correct=0, unclear=0, **fields):
    return _stratum("added", real, correct, unclear, **fields)


def _plant(directory, items) -> None:
    """Write an audit's three files: one worklist row and key row per item, and a verdict for
    each item whose answer is not ``None``."""
    worklist, key, verdicts = [], [], {}
    for n, (stratum, heard, causes, consonants) in enumerate(items):
        ident = f"id{n:03d}"
        flagged = {"outcome": "flagged", "causes": causes, "consonants": consonants}
        clear = {"outcome": "not_flagged", "causes": [], "consonants": []}
        removed = stratum == "removed"
        worklist.append({"id": ident})
        key.append({"id": ident, "stratum": stratum, "probability": 0.5,
                    "c95": flagged if removed else clear, "candidate": clear if removed else flagged})
        if heard is not None:
            verdicts[ident] = Verdict(ident, heard)
    for name, rows in (("worklist.jsonl", worklist), ("key.jsonl", key)):
        (directory / name).write_text("".join(json.dumps(row) + "\n" for row in rows),
                                      encoding="utf-8")
    write_verdicts(verdicts, directory / "verdicts.jsonl")


def _score(directory, items) -> dict:
    directory.mkdir(exist_ok=True)
    _plant(directory, items)
    return audit.score(directory)


# --- the gate ------------------------------------------------------------------------------------


@pytest.mark.parametrize("real, correct, unclear, verdict", [
    (20, 80, 0, "pass"),       # exactly 20% of 100 passes
    (21, 79, 0, "fail"),
    (0, 100, 0, "pass"),
    (16, 64, 20, "pass"),      # 80 decisive certifies, and 16 of 80 is exactly 20%
    (17, 63, 20, "fail"),
    (0, 80, 20, "pass"),
    (80, 0, 20, "fail"),
    (0, 79, 21, "cannot_certify"),   # 79 decisive: a pass-looking result is not certified
    (79, 0, 21, "cannot_certify"),   # ... and neither is a fail-looking one
    (0, 0, 100, "cannot_certify"),
])
def test_the_gate_at_the_20_percent_and_80_decisive_boundaries(
        tmp_path, real, correct, unclear, verdict):
    result = _score(tmp_path, _removed(real, correct, unclear))

    assert result["verdict"] == verdict
    assert result["removed"]["real"] == real
    assert result["removed"]["correct"] == correct
    assert result["removed"]["unclear"] == unclear
    assert result["removed"]["decisive"] == real + correct
    assert result["bars"] == {"real_at_most": 0.20, "decisive_at_least": 80}


def test_a_stratum_smaller_than_100_still_needs_80_decisive_and_no_more(tmp_path):
    assert _score(tmp_path / "enough", _removed(8, 77))["verdict"] == "pass"
    assert _score(tmp_path / "short", _removed(0, 79))["verdict"] == "cannot_certify"


def test_the_real_mistake_share_and_its_wilson_interval_are_reported(tmp_path):
    result = _score(tmp_path, _removed(20, 80))

    assert result["removed"]["real_share"] == 0.2
    low, high = result["removed"]["wilson"]
    assert (low, high) == pytest.approx((0.1334, 0.2888), abs=1e-4)


def test_with_no_decisive_verdict_there_is_no_share_to_report(tmp_path):
    result = _score(tmp_path, _removed(unclear=100))

    assert result["removed"]["real_share"] is None
    assert result["removed"]["wilson"] is None


def test_the_added_stratum_is_reported_and_never_changes_the_verdict(tmp_path):
    result = _score(tmp_path, [*_removed(10, 90), *_added(30, 10, 10)])

    assert result["verdict"] == "pass"
    assert result["added"]["real_share"] == 0.75
    assert (result["added"]["decisive"], result["added"]["unclear"]) == (40, 10)
    sparse = _score(tmp_path / "sparse", _removed(10, 90) + _added(5))
    assert sparse["verdict"] == "pass"  # few added verdicts never make a pass uncertifiable


# --- by cause and the harder pairs ---------------------------------------------------------------


def test_each_stratum_is_split_by_the_flagging_models_causes(tmp_path):
    items = [
        *_removed(real=3, correct=1, causes=("tashkeel",)),
        *_removed(real=1, correct=5, unclear=2, causes=(CONSONANT,)),
        *_removed(real=2, causes=("tashkeel", CONSONANT)),  # a word counts in each of its causes
        *_added(real=1, correct=1, causes=("shaddah",)),
    ]

    result = _score(tmp_path, items)

    by_cause = result["by_cause"]
    assert {cause: (c["real"], c["correct"], c["unclear"])
            for cause, c in by_cause["removed"].items()} == {
        "tashkeel": (5, 1, 0), CONSONANT: (3, 5, 2)}
    assert {cause: (c["real"], c["correct"]) for cause, c in by_cause["added"].items()} == {
        "shaddah": (1, 1)}


def test_a_harder_pair_flag_is_marked_noisy_by_the_flagging_models_substitutions(tmp_path):
    noisy = (("soft_pair", "س→ص"),)
    items = [
        *_removed(real=2, correct=1, consonants=noisy),
        *_removed(real=1, consonants=(("soft_pair", "ث→س"),)),
        *_removed(correct=1, consonants=(("a", "ق→ط"), ("b", "ص→ث"))),  # any substitution of the word
        *_removed(real=5, correct=5, consonants=(("soft_pair", "س→ش"),)),  # easy by ear
        *_removed(real=3, consonants=(("other_consonant", "ث→ط"),)),       # only one letter in the set
        *_removed(real=4, causes=("tashkeel",)),                            # no substitution
        *_added(real=1, correct=2, consonants=(("soft_pair", "ص→س"),)),
        *_added(real=7, consonants=(("soft_pair", "ك→ق"),)),
    ]

    result = _score(tmp_path, items)

    assert result["noisy"]["removed"] == {"items": 5, "decisive": 5, "real": 3}
    assert result["noisy"]["added"] == {"items": 3, "decisive": 3, "real": 1}
    assert result["removed"]["real"] == 2 + 1 + 5 + 3 + 4  # noisy verdicts stay in the gate


# --- what is written and what is refused ---------------------------------------------------------


def test_the_result_is_written_beside_the_files_and_states_the_position_caveat(tmp_path):
    result = _score(tmp_path, _removed(10, 90))

    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == result
    assert result["caveats"] and all(isinstance(text, str) for text in result["caveats"])
    assert any("position" in text for text in result["caveats"])


def test_the_scorer_refuses_while_a_worklist_item_has_no_verdict(tmp_path):
    items = _removed(10, 89) + [_item("removed", None)]

    _plant(tmp_path, items)

    with pytest.raises(REFUSED, match="verdict"):
        audit.score(tmp_path)
    assert not (tmp_path / "result.json").exists()
