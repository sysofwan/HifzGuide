"""What the tashkeel audit says about over-strictness, base vs candidate (#60).

The counterfactual gate (:mod:`training.counterfactual_eval`) measures one direction only:
how often a fine-tune stops flagging a vowel error it should have caught. That is the
*cost* side. ADR-0003's actual goal is the other side — the base Muaalem checkpoint was
trained on professional reciters and declines to mark vowels that amateurs do in fact
produce, and the fine-tune exists to stop it rejecting correct recitation.

This module turns the mined-and-adjudicated worklist into that second number. It joins
:mod:`training.tashkeel_worklist` sites to :mod:`tadabur.tashkeel_fixtures` verdicts and
keeps only sites where the listener heard the **reference** colour: at those, and only
those, a checkpoint that failed to reproduce the vowel was demonstrably wrong about correct
recitation. Sites where the reciter said something else, said nothing, or could not be made
out carry no information about over-strictness and are reported, not scored.

**The worklist is a sample, so the report is two layers.** The audited sites give a
*proportion* per direction — of the positions the candidate recovered, what share were
genuinely correct recitation — with a Wilson interval. Those proportions are then scaled
onto the population counts the mining run recorded, giving a confirmed false-rejection rate
per reference vowel for each checkpoint. The interval on the difference is formed from the
worst-case combination of the two Wilson intervals: cruder than a joint interval, and
deliberately so, because it cannot understate the uncertainty the sampling introduced.

Concordant positions never reach the audit (see :mod:`training.tashkeel_worklist`), which is
sound for the *difference* — it is a function of the discordant cells alone — but means this
module can say nothing about the absolute quality of either checkpoint. It answers "which
one rejects correct recitation more often, and by how much", not "how good is it".
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from training.counterfactual_eval import wilson_interval
from training.tashkeel_worklist import RECOVERED, REGRESSED, TashkeelSite, read_worklist

from .tashkeel_fixtures import UNCLEAR, Adjudication, read_adjudications


@dataclass(frozen=True)
class DirectionResult:
    """One direction's audit outcome: how many mined sites were genuinely correct recitation.

    ``confirmed`` / ``audited`` is the share of this direction's discordant positions at
    which the listener heard the reference colour, i.e. the share that is real
    over-strictness rather than the reciter having said something else. ``population`` is
    how many such positions the mining run found in total, so ``estimate`` scales the
    audited proportion back onto the corpus.
    """

    direction: str
    audited: int
    confirmed: int
    said_otherwise: int
    unclear: int
    population: int

    @property
    def interval(self) -> tuple[float, float]:
        """95% Wilson interval on the confirmed share of this direction."""
        return wilson_interval(self.confirmed, self.audited) if self.audited else (0.0, 1.0)

    @property
    def estimate(self) -> float:
        """Population positions estimated to be genuine over-strictness by this checkpoint."""
        share = self.confirmed / self.audited if self.audited else 0.0
        return self.population * share

    def bounds(self) -> tuple[float, float]:
        """Population positions, low and high, from the Wilson interval on the share."""
        low, high = self.interval
        return self.population * low, self.population * high

    def to_dict(self) -> dict:
        low, high = self.interval
        estimate_low, estimate_high = self.bounds()
        return {
            "direction": self.direction,
            "audited": self.audited,
            "confirmed": self.confirmed,
            "said_otherwise": self.said_otherwise,
            "unclear": self.unclear,
            "population": self.population,
            "confirmed_share": round(self.confirmed / self.audited, 4) if self.audited else None,
            "confirmed_share_ci95": [round(low, 4), round(high, 4)],
            "estimated_population_sites": round(self.estimate, 1),
            "estimated_population_sites_ci95": [
                round(estimate_low, 1), round(estimate_high, 1)
            ],
        }


def _classify(site: TashkeelSite, verdict: Adjudication) -> str:
    """Whether this adjudication confirms over-strictness, refutes it, or says nothing."""
    if verdict.verdict == UNCLEAR:
        return UNCLEAR
    if verdict.heard_vowel == site.reference_vowel:
        return "confirmed"
    return "said_otherwise"


def summarize_direction(
    sites: list[TashkeelSite], adjudications: dict[str, Adjudication], direction: str, population: int
) -> DirectionResult:
    """Fold one direction's adjudicated sites into a :class:`DirectionResult`."""
    tallies = {"confirmed": 0, "said_otherwise": 0, UNCLEAR: 0}
    for site in sites:
        if site.direction != direction:
            continue
        verdict = adjudications.get(site.site_id)
        if verdict is None:
            continue
        tallies[_classify(site, verdict)] += 1
    return DirectionResult(
        direction=direction,
        audited=sum(tallies.values()),
        confirmed=tallies["confirmed"],
        said_otherwise=tallies["said_otherwise"],
        unclear=tallies[UNCLEAR],
        population=population,
    )


def compare(
    sites: list[TashkeelSite],
    adjudications: dict[str, Adjudication],
    population: dict,
) -> dict:
    """The two-sided over-strictness comparison, base against candidate.

    ``recovered`` counts positions the **base** got wrong and the candidate right — the
    base's confirmed false rejections. ``regressed`` counts the reverse. Both are rates per
    reference vowel, so they can be read against the aggregate recall the eval reports.

    The reported interval on the difference combines each direction's Wilson bounds at their
    worst: the candidate looks best when its own confirmed count is at its lower bound and
    the base's at its upper, and worst the other way round. That is wider than a joint
    interval would be. Given the counterfactual gate's history of a too-narrow interval
    reading as a result, an interval that can only be too cautious is the right trade.
    """
    total_vowels = population["reference_vowels"]
    recovered = summarize_direction(sites, adjudications, RECOVERED, population[RECOVERED])
    regressed = summarize_direction(sites, adjudications, REGRESSED, population[REGRESSED])

    base_low, base_high = recovered.bounds()
    candidate_low, candidate_high = regressed.bounds()
    rate = (lambda count: count / total_vowels if total_vowels else 0.0)
    difference = rate(recovered.estimate) - rate(regressed.estimate)
    return {
        "reference_vowels": total_vowels,
        "recovered": recovered.to_dict(),
        "regressed": regressed.to_dict(),
        "base_false_rejection_rate": round(rate(recovered.estimate), 5),
        "candidate_false_rejection_rate": round(rate(regressed.estimate), 5),
        # Positive means the base falsely rejects correct recitation more often than the
        # candidate does — the over-strictness ADR-0003 set out to reduce.
        "acceptance_gain": round(difference, 5),
        "acceptance_gain_ci95": [
            round(rate(base_low) - rate(candidate_high), 5),
            round(rate(base_high) - rate(candidate_low), 5),
        ],
        "audited": recovered.audited + regressed.audited,
        "pending": sum(1 for s in sites if s.site_id not in adjudications),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, required=True,
                        help="mined worklist JSONL (training.tashkeel_worklist).")
    parser.add_argument("--adjudications", type=Path, required=True,
                        help="listener verdicts JSONL (tadabur.tashkeel_audit_ui).")
    parser.add_argument("--summary", type=Path, default=None,
                        help="the worklist's '.summary.json' sidecar, holding the population "
                             "counts (default: alongside --worklist).")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the report here as well as to stdout.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary_path = args.summary or args.worklist.with_suffix(
        args.worklist.suffix + ".summary.json"
    )
    if not summary_path.is_file():
        raise SystemExit(
            f"{summary_path} not found. The population counts it holds are what turn the "
            "audited sample into a rate; without them the report would silently read as a "
            "census of the corpus. Pass --summary."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = {
        "base": summary.get("base"),
        "candidate": summary.get("candidate"),
        "coverage": summary.get("coverage"),
        **compare(
            read_worklist(args.worklist),
            read_adjudications(args.adjudications),
            summary["population"],
        ),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
