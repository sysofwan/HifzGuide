"""Round 2's catch-loss mark is the worklist's own word coordinate (#309, A5b, S1.5).

The worklist names a 0-based canonical (printed) word; ``stratify``'s ``position`` is the
0-based position among placed phoneme words. Those diverge where printed words share a phoneme
word, so the mark must carry ``word_index`` rather than reuse the position. The worklist is
look-tranche, so the committed count is also the reported ``marked`` and stays visible.
"""

from __future__ import annotations

from tadabur.followalong_audit.strata import stratify
from tadabur.followalong_audit.strata_run import CATCH_LOSS_WORDS, catch_loss_marks

DETECTORS = ("base", "c95", "r2_candidate_ema", "r3_candidate_ema")


def word(position: int, word_index: int) -> dict:
    return {"key": "a#0", "position": position, "word_index": word_index, "app": "passed",
            "oracle": None, "causes": [],
            "detectors": {model: False for model in DETECTORS}}


def test_the_committed_worklist_marks_its_150_words():
    marks = catch_loss_marks()
    assert len(marks) == CATCH_LOSS_WORDS == 150
    assert all(isinstance(key, str) and isinstance(index, int) for key, index in marks)


def test_the_mark_is_the_worklist_word_index_not_the_phoneme_position():
    assert stratify([word(position=0, word_index=5)], catch_loss={("a#0", 5)})[0]["catch_loss"]
    assert not stratify([word(position=0, word_index=5)],
                        catch_loss={("a#0", 0)})[0]["catch_loss"]
