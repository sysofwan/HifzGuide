"""The phonetizer's Uthmani words are public (#311 PR3, scout Q3).

Committed before the PR3 code exists and **locked**: later turns may add tests but never weaken or
rewrite these. The draw's worklist words come from quran-transcript, not from ``quran.db`` (the two
differ in spelling for nearly every drawn ayah), so the listening UI must verify the worklist
against the same source: ``tadabur.waqf_segments.uthmani_words(surah_ayah) -> list[str]`` is the
public name of ``_uthmani_words`` (same behaviour), and no other module reaches for the private one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tadabur.waqf_segments import uthmani_words

TADABUR = Path(__file__).resolve().parents[1]
DRAW = Path(__file__).resolve().parent / "s5_draw"


def test_the_public_accessor_returns_an_ayahs_uthmani_words_in_order():
    words = uthmani_words("1:2")
    assert len(words) == 4 and all(isinstance(word, str) and word for word in words)
    assert uthmani_words("112:1") == uthmani_words("112:1")


def test_the_public_accessor_is_the_private_ones_behaviour():
    from tadabur.waqf_segments import _uthmani_words

    for label in ("1:1", "2:255", "36:1", "112:4"):
        assert uthmani_words(label) == _uthmani_words(label)


def test_it_reproduces_every_committed_worklist_word():
    rows = [json.loads(line) for line in
            (DRAW / "worklist.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows[:60]:
        assert uthmani_words(row["surah_ayah"])[row["word_index"]] == row["word"]


def test_an_unknown_ayah_is_refused_not_guessed():
    with pytest.raises(Exception):
        uthmani_words("999:1")


def test_no_module_but_the_defining_one_reaches_for_the_private_name():
    offenders = []
    for path in sorted(TADABUR.rglob("*.py")):
        if path.name == "waqf_segments.py" or path.name.startswith("test_"):
            continue
        if re.search(r"\b_uthmani_words\b", path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(TADABUR)))
    assert offenders == []
