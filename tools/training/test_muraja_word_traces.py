"""``word_traces`` on a batch's results (#306): a trace reads only its own item's records.

The harness returns one result per request per engine run, so a batch holds several items under
the same scorings. A trace that indexed them by scoring alone would label one item's grades with
another item's check text; these tests pin the item-and-scoring key instead.
"""

from __future__ import annotations

import pytest

from training import muraja_policy as mp
from training.muraja_policy import SHIPPED_DEFAULT, STRICT
from training.test_muraja_policy import W3_LETTERS

#: Two checks whose hop/overlap text differs per request, so a wrong item's text is visible.
CHECKS = [{"hop": "", "overlap": "أَ", "flush": False},
          {"hop": "ل", "overlap": "بِ", "flush": True}]


def request(item: str, words: list[int]) -> dict:
    return {"item": item, "surah": 2, "ayah": 80, "start_word": 2, "report_words": words,
            "checks": CHECKS, "scorings": mp.engine_runs([STRICT, SHIPPED_DEFAULT])}


def result(item: str, scoring: mp.Scoring, word: int, quality: str) -> dict:
    kept = {"word": word, "quality": quality, "score": 0.7, "withheld": None, "errors": []}
    return {"item": item, "scoring": scoring.engine_name, "page": 1, "parameters": {},
            "checks": [{"position": "2:80:2", "statuses": [kept]},
                       {"position": "2:80:2", "statuses": []}],
            "final": [kept], "letters": [dict(W3_LETTERS, word=word)]}


def results_of(item: str, word: int, quality: str) -> list[dict]:
    return [result(item, STRICT, word, quality), result(item, SHIPPED_DEFAULT, word, quality)]


def test_a_batch_trace_equals_each_item_s_trace_of_its_own_results():
    first, second = request("arm#1", [3]), request("arm#2", [9])
    batch = results_of("arm#1", 3, "tashkeelError") + results_of("arm#2", 9, "correct")
    for item, word, own in ((first, 3, batch[:2]), (second, 9, batch[2:])):
        assert mp.word_traces(item, batch, [STRICT, SHIPPED_DEFAULT]) == \
            mp.word_traces(item, own, [STRICT, SHIPPED_DEFAULT])
        rows = mp.word_traces(item, batch, [STRICT, SHIPPED_DEFAULT])
        assert [(row["item"], row["word"]) for row in rows] == [(item["item"], word)] * 2


def test_a_trace_raises_when_the_run_holds_no_result_for_the_request():
    other = results_of("arm#2", 9, "correct")
    with pytest.raises(ValueError, match="arm#1"):
        mp.word_traces(request("arm#1", [3]), other, [STRICT])
