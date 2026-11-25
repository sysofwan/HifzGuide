"""The candidate strata S4 draws from (#309): total, single-valued, and the pools they size.

A word is graded twice over the same population: the App arm's strict outcome and the four
detector decodes' flags, both re-graded at the audit's Muraja commit. The strata split the
comparison into what each side says:

* **S5** — the App left the word ungraded: no phone verdict exists, so nothing else can decide it;
* **S1** — the App flags it and the oracle passes it: the candidate false flag, pending S2's
  oracle outcome (``S1_PENDING`` until then) and explained when the oracle flags it too
  (``ORACLE_EXPLAINED``);
* **S3** — the App passes it and the base detector, or at least two detectors, flag it: a
  candidate miss;
* **S4** — the App passes it and nothing flags it: the pool the next tranche is drawn from;
* **RESIDUAL** — the App passes it and exactly one non-base detector flags it.

:func:`assign_stratum` reads the precedence in that order, so every combination of outcomes lands
in exactly one stratum; :func:`stratify` adds the round-2 catch-loss mark (A5b) and
:func:`pool_sizes` counts each pool, overall and among the catch-loss words.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

S5 = "S5"
S1 = "S1"
S1_PENDING = "S1-pending"
ORACLE_EXPLAINED = "oracle-explained"
S3 = "S3"
S4 = "S4"
RESIDUAL = "residual"

#: Every stratum, in report order.
STRATA = (S5, S1, S1_PENDING, ORACLE_EXPLAINED, S3, S4, RESIDUAL)

#: The detector names, as the committed decodes name them.
DETECTORS = ("base", "c95", "r2_candidate_ema", "r3_candidate_ema")

_APP_OUTCOMES = ("flagged", "passed", "ungraded")
_ORACLE_OUTCOMES = ("flagged", "passed")


def assign_stratum(word: Mapping) -> str:
    """The one stratum the word's App outcome, oracle outcome and detector flags put it in."""
    app = word["app"]
    if app not in _APP_OUTCOMES:
        raise ValueError(f"{app!r} is not an App outcome: {_APP_OUTCOMES}")
    if app == "ungraded":
        return S5
    if app == "flagged":
        oracle = word["oracle"]
        if oracle == "flagged":
            return ORACLE_EXPLAINED
        if oracle == "passed":
            return S1
        if oracle is None:
            return S1_PENDING
        raise ValueError(f"{oracle!r} is not an oracle outcome: {(*_ORACLE_OUTCOMES, None)}")
    flags = word["detectors"]
    if flags["base"]:
        return S3
    flagging = sum(1 for name in DETECTORS if flags[name])
    if flagging >= 2:
        return S3
    return S4 if not flagging else RESIDUAL


def covered(word: Mapping) -> bool:
    """Whether the audio holds the word (A1).

    An App-graded word is covered: its grade is evidence of audio at the position. An App-ungraded
    word is covered when the Teacher transcript aligns to it, and the transcript must have said so
    — a missing ``transcript_covered`` is a bug in the join, not an uncovered word.
    """
    return True if word["app"] != "ungraded" else word["transcript_covered"]


def stratify(words: Iterable[Mapping],
             catch_loss: Iterable[tuple[str, int]] = ()) -> list[dict]:
    """Each word with its stratum and whether round 2's catch-loss audit marked it (A5b).

    ``catch_loss`` is keyed ``(item key, word_index)`` — the 0-based canonical word the worklist
    names. A word with no ``word_index`` (a fixture without one) is matched on its ``position``,
    the same coordinate only where one printed word is one phoneme word.
    """
    marked = set(catch_loss)
    return [{**word,
             "stratum": assign_stratum(word),
             "catch_loss": (word["key"], word["word_index"] if "word_index" in word
                            else word["position"]) in marked}
            for word in words]


def pool_sizes(rows: Sequence[Mapping]) -> dict[str, dict[str, int]]:
    """How many words each stratum holds, and how many of those are catch-loss words."""
    sizes = {"strata": {name: 0 for name in STRATA},
             "catch_loss": {name: 0 for name in STRATA}}
    for row in rows:
        stratum = row["stratum"]
        sizes["strata"][stratum] += 1
        if row["catch_loss"]:
            sizes["catch_loss"][stratum] += 1
    return sizes


def s1_by_cause_and_base(rows: Sequence[Mapping]) -> dict[tuple[str, bool], int]:
    """The App flags the oracle does not (yet) share, once per cause, keyed by base-detector flag.

    ``S1_PENDING`` words are counted here too: until S2 has run they are the pool's best estimate,
    and a word carries each of its causes once however many causes the flag names.
    """
    counts: dict[tuple[str, bool], int] = {}
    for row in rows:
        if row["stratum"] not in (S1, S1_PENDING):
            continue
        base = bool(row["detectors"]["base"])
        for cause in set(row["causes"]):
            counts[(cause, base)] = counts.get((cause, base), 0) + 1
    return dict(sorted(counts.items()))


def s1_records(rows: Sequence[Mapping]) -> list[dict]:
    """The S1 pool as the summary records it: one ``{cause, base_flags, words}`` line each."""
    return [{"cause": cause, "base_flags": base, "words": words}
            for (cause, base), words in s1_by_cause_and_base(rows).items()]
