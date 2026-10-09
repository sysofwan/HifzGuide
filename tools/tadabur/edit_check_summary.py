"""What the synthetic-edit blind check heard, per operation: the hand-off to #90 (#107).

ADR-0011 §3 admits a synthetic edit type as training signal only after a blind listen
confirms the edits sound real; #90 decides which types pass. This summarises the owner's
edit-check verdicts (``listening_session/verdicts.jsonl``, written by the blind UI) by
joining the blind check (``synthetic_edits/blind_check.jsonl``) to the edit manifest
(``synthetic_edits/edits.jsonl``) through each item's opaque output file and checksum.

Per operation, and per operation and mark, for the edits and the decoys apart:

* **naturalness**: ``natural`` / ``unnatural`` / ``unclear``, and the natural rate over the
  items judged natural or unnatural (``unclear`` leaves the denominator, as everywhere);
* **heard as labelled**: the item's label was heard at the carrier (for an edit, the
  changed state; for a decoy, the unchanged one, so its rate is "decoys unchanged"), the
  other state was heard, or ``unclear``; the rate is over the items heard as either state.

A rate is ``null`` while its denominator is empty. Items with no verdict yet are counted
as ``pending``. No threshold is applied here: which types pass is #90's call, under the
acceptance rules.

**Never shown in the UI.** This module reads the edit manifest, which names every item's
operation and role. The listening UI (:mod:`tadabur.tashkeel_audit_ui`) never imports it
and has no route to it; run it only after the session.

Usage (from ``tools/``)::

  python -m tadabur.edit_check_summary [--out summary.json]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from training.tashkeel_eval import write_text_atomically

from .listening_session import (
    NATURAL,
    NATURALNESS,
    UNNATURAL,
    VERDICTS_PATH,
    Verdict,
    hearable,
    read_verdicts,
)
from .synthetic_edits import EDITS_PATH, WORKLIST_PATH, read_items
from .synthetic_edit_plan import DECOY, EDIT, OPERATIONS
from .truth_sites import UNCLEAR, TruthSite, load_truth_sites

ROLES = (EDIT, DECOY)


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def tally(answers: list[tuple[dict, Verdict | None]]) -> dict:
    """Counts and rates for one cell: each blind-check item's manifest row and its verdict
    (``None`` while unanswered)."""
    given = [(item, v) for item, v in answers if v is not None]
    natural = sum(v.natural == NATURAL for _, v in given)
    unnatural = sum(v.natural == UNNATURAL for _, v in given)
    labelled = sum(v.heard == item["label"] for item, v in given)
    unclear = sum(v.heard == UNCLEAR for _, v in given)
    otherwise = len(given) - labelled - unclear
    return {
        "items": len(answers),
        "pending": len(answers) - len(given),
        "natural": natural,
        "unnatural": unnatural,
        "natural_unclear": len(given) - natural - unnatural,
        "natural_rate": _rate(natural, natural + unnatural),
        "heard_as_labelled": labelled,
        "heard_otherwise": otherwise,
        "heard_unclear": unclear,
        "heard_as_labelled_rate": _rate(labelled, labelled + otherwise),
    }


def matched_items(sites: list[TruthSite], items: list[dict]) -> list[tuple[TruthSite, dict]]:
    """Each blind-check site with its manifest row, matched by output file and checksum,
    and checked to describe the same reference, carrier and mark."""
    by_output = {item["output"]["audio_filename"]: item for item in items}
    pairs = []
    for site in sites:
        item = by_output.get(site.audio_filename)
        if item is None or item["output"]["audio_sha256"] != site.audio_sha256:
            raise ValueError(f"{site.site_id}: no manifest item renders {site.audio_filename}")
        if (site.reference, site.reference_index, site.mark, site.prescribed) != (
                item["reference"], item["reference_index"], item["mark"], item["prescribed"]):
            raise ValueError(f"{site.site_id}: the blind check and the manifest disagree")
        pairs.append((site, item))
    return pairs


def summarize(
    sites: list[TruthSite], items: list[dict], verdicts: Mapping[str, Verdict]
) -> dict:
    """Per operation, and per operation and mark, :func:`tally` for edits and decoys."""
    answers: list[tuple[dict, Verdict | None]] = []
    for site, item in matched_items(sites, items):
        verdict = verdicts.get(site.site_id)
        if verdict is not None and (verdict.heard not in hearable(site.mark)
                                    or verdict.natural not in NATURALNESS):
            raise ValueError(f"{site.site_id}: {verdict} does not answer both questions")
        answers.append((item, verdict))

    def cells(key) -> dict:
        groups: dict[str, list] = {}
        for item, verdict in answers:
            groups.setdefault(key(item), []).append((item, verdict))
        return {
            name: {role: tally([(i, v) for i, v in group if i["role"] == role])
                   for role in ROLES}
            for name, group in sorted(groups.items(), key=lambda g: g[0])
        }

    by_operation = cells(lambda item: item["operation"])
    return {
        "items": len(answers),
        "answered": sum(v is not None for _, v in answers),
        "operations": {op: by_operation[op] for op in OPERATIONS if op in by_operation},
        "operation_marks": cells(lambda item: f"{item['operation']} {item['mark']}"),
    }


def _table(summary: dict) -> str:
    def rate(value: float | None) -> str:
        return "-" if value is None else f"{value:.0%}"

    lines = [f"{summary['answered']} of {summary['items']} items answered",
             f"{'operation':28s} {'role':6s} {'n':>3s} {'natural':>9s} {'as labelled':>12s}"]
    marks = summary["operation_marks"]
    split = [m for m in marks if sum(n.split()[0] == m.split()[0] for n in marks) > 1]
    for name, roles in [*summary["operations"].items(), *((m, marks[m]) for m in split)]:
        for role, cell in roles.items():
            if cell["items"]:
                lines.append(f"{name:28s} {role:6s} {cell['items']:3d} "
                             f"{rate(cell['natural_rate']):>9s} "
                             f"{rate(cell['heard_as_labelled_rate']):>12s}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--blind-check", type=Path, default=WORKLIST_PATH)
    parser.add_argument("--edits", type=Path, default=EDITS_PATH)
    parser.add_argument("--verdicts", type=Path, default=VERDICTS_PATH)
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON here")
    args = parser.parse_args()
    summary = summarize(load_truth_sites(args.blind_check), read_items(args.edits),
                        read_verdicts(args.verdicts))
    if args.out is not None:
        write_text_atomically(args.out, json.dumps(summary, indent=2, ensure_ascii=False,
                                                   sort_keys=True) + "\n")
    print(_table(summary))


if __name__ == "__main__":
    main()
