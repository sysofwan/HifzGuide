"""Pattern grouping for the S2 findings (#308): candidate Muraja issues at ten words.

A **finding** is one word the engine graded wrongly — a false flag under a perfect decode or a
planted mistake it missed. A **pattern** pools findings that agree on the requested dimensions
(reference form, error kind, position, boundary) and counts the distinct ayah words it covers,
``(surah, ayah, word)``: one word flagged in strict and in balanced, or in several clips of the
same ayah, is one word of the pattern. :data:`CANDIDATE_WORDS` is the PRD's threshold for a
candidate Muraja issue, and :func:`issue_body` writes the draft the owner files in the private
repo (the note keeps the drafts; this repo quotes no Muraja source).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

#: How many distinct words a pattern must cover to be a candidate Muraja issue (PRD decision 1).
CANDIDATE_WORDS = 10

#: How many example findings a pattern keeps, in ayah order, with their replay evidence.
EXAMPLES = 3

#: The finding fields an example keeps: the word it is, how it reads and the evidence.
_EXAMPLE_FIELDS = ("item", "surah", "ayah", "word", "mode", "quality", "error_kind", "position",
                   "boundary", "form", "statuses")


def group_findings(findings: Sequence[Mapping],
                   dimensions: Sequence[str]) -> list[dict]:
    """Pool ``findings`` by ``dimensions``, ordered by words then key.

    Every finding must carry every requested dimension (a missing one raises ``KeyError``);
    the key names each dimension with its value, and the count of distinct ayah words is what
    makes a candidate — not the number of findings.
    """
    pools: dict[tuple, list[Mapping]] = {}
    for finding in findings:
        key = tuple(finding[dimension] for dimension in dimensions)
        pools.setdefault(key, []).append(finding)

    patterns = []
    for key, pool in pools.items():
        ordered = sorted(pool, key=_word_order)
        words = {(finding["surah"], finding["ayah"], finding["word"]) for finding in ordered}
        patterns.append({
            "key": dict(zip(dimensions, key)),
            "words": len(words),
            "findings": len(ordered),
            "candidate": len(words) >= CANDIDATE_WORDS,
            "examples": [{field: finding[field] for field in _EXAMPLE_FIELDS}
                         for finding in ordered[:EXAMPLES]],
        })
    patterns.sort(key=lambda pattern: (-pattern["words"],
                                       json.dumps(pattern["key"], sort_keys=True,
                                                  ensure_ascii=False)))
    return patterns


def _word_order(finding: Mapping) -> tuple:
    """A finding's place in ayah order, then by mode and item, so examples are stable."""
    return (finding["surah"], finding["ayah"], finding["word"], finding["mode"],
            finding["item"])


def issue_body(pattern: Mapping) -> str:
    """The ready-to-file Muraja issue body for a candidate pattern; raises below the threshold."""
    if not pattern["candidate"] or pattern["words"] < CANDIDATE_WORDS:
        raise ValueError(f"a pattern of {pattern['words']} words is not a candidate issue")
    key = ", ".join(f"{dimension}: {pattern['key'][dimension]}"
                    for dimension in sorted(pattern["key"]))
    lines = [
        f"## Follow-along pattern: {pattern['words']} words ({key})",
        "",
        "The follow-along audit's perfect-decode sweep (HifzGuide #308) graded these words"
        f" wrongly over {pattern['findings']} finding(s): each word the engine does not grade"
        " correct is a follow-along or reference fault, because the decode was the reference.",
        "",
        "| ayah | word | mode | form | quality | error | position | boundary |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for example in pattern["examples"]:
        lines.append(
            f"| {example['surah']}:{example['ayah']} | {example['word']} | {example['mode']} | "
            f"{example['form']} | {example['quality']} | {example['error_kind']} | "
            f"{example['position']} | {example['boundary']} |")
    lines += ["", "Replay evidence (per-check statuses of the first example):", "", "```json",
              json.dumps(pattern["examples"][0]["statuses"], ensure_ascii=False,
                         sort_keys=True),
              "```"]
    return "\n".join(lines) + "\n"
