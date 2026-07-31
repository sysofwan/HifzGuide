"""What the tashkeel audit says about over-strictness, base vs candidate (#60).

The counterfactual gate (:mod:`training.counterfactual_eval`) measures one direction only:
how often a fine-tune stops flagging a vowel error it should have caught. That is the
*cost* side. ADR-0003's actual goal is the other side — the base Muaalem checkpoint was
trained on professional reciters and declines to mark vowels that amateurs do in fact
produce, and the fine-tune exists to stop it rejecting correct recitation.

This module turns the mined-and-adjudicated worklist into that second number. It joins
:mod:`training.tashkeel_worklist` sites to :mod:`tadabur.tashkeel_fixtures` verdicts and
counts a site as over-strictness only where the listener heard the **reference** colour: at
those, and only those, a checkpoint that failed to reproduce the vowel was demonstrably
wrong about correct recitation.

**Estimation is stratified, because the sample is.** The worklist draws up to
``per_bucket`` sites per ``(direction, colour)``, so fatha and kasra are sampled at wildly
different rates relative to how often they occur. Pooling a direction's verdicts and scaling
the pooled share onto the direction's population would weight each colour by its *sample*
size instead of its *population* size — if fatha recoveries are usually genuine and kasra
recoveries usually are not, that pooled estimator is biased, badly. Each stratum is
therefore estimated against its own population count and the strata are summed.

**Unclear verdicts leave the denominator**, rather than counting against confirmation. A
recording nobody can make out is not evidence that the reciter said the wrong vowel. This
assumes the unclear sites would have resolved like the audible ones in their stratum; the
report carries ``unclear_share`` so a reader can see how much weight that assumption bears.

**The interval is simultaneous, not a difference of two 95% intervals.** Subtracting two
independent 95% bounds gives at most 90.25% joint coverage, so the naive construction is not
merely crude — it is *anti*-conservative and would be mislabelled. Every stratum bound
entering the difference is widened by Bonferroni so the reported interval holds at 95%
overall.

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
from statistics import NormalDist

from training.counterfactual_eval import wilson_interval
from training.tashkeel_worklist import RECOVERED, REGRESSED, TashkeelSite, read_worklist

from .tashkeel_fixtures import UNCLEAR, Adjudication, read_adjudications

#: Overall error rate the reported interval is built to hold at.
ALPHA = 0.05

CONFIRMED = "confirmed"
SAID_OTHERWISE = "said_otherwise"


def component_z(components: int, alpha: float = ALPHA) -> float:
    """Per-bound z so a two-sided difference over ``components`` strata holds at ``1-alpha``.

    Bonferroni. The reported interval's two ends consume ``2 * components`` one-sided
    stratum bounds — each end takes an upper bound from one direction and a lower bound from
    the other — and the interval misses the truth if *any* of them does. Leaving each at
    1.96 would give roughly ``(1 - alpha) ** components`` coverage, about 74% across six
    strata rather than 95%, so each is widened to ``alpha / (2 * components)``.

    At one stratum per direction this reduces to the familiar 1.96, which is the only case
    where differencing two plain 95% intervals would have been defensible.
    """
    if components <= 0:
        raise ValueError("a simultaneous interval needs at least one component bound")
    return NormalDist().inv_cdf(1 - alpha / (2 * components))


@dataclass(frozen=True)
class StratumResult:
    """One ``(direction, colour)`` cell: what the audit found, scaled onto its population.

    ``scoreable`` excludes ``unclear``, so ``confirmed / scoreable`` is the share of the
    *adjudicable* sites at which the reciter really did say the reference colour.
    """

    direction: str
    vowel_name: str
    confirmed: int
    said_otherwise: int
    unclear: int
    population: int

    @property
    def audited(self) -> int:
        return self.confirmed + self.said_otherwise + self.unclear

    @property
    def scoreable(self) -> int:
        return self.confirmed + self.said_otherwise

    def share(self, fallback: float) -> float:
        """Confirmed share of this stratum, or ``fallback`` when nothing is adjudicable.

        A stratum nobody has reached yet has no share of its own. Borrowing the direction's
        pooled share keeps the point estimate from silently reading as *zero* over-strictness
        in that colour; the bounds stay at the full ``(0, population)``, so the borrowing
        never makes the interval look better informed than the audit is.
        """
        return self.confirmed / self.scoreable if self.scoreable else fallback

    def estimate(self, fallback: float) -> float:
        return self.population * self.share(fallback)

    def bounds(self, z: float) -> tuple[float, float]:
        """Population sites, low and high, from this stratum's Wilson interval."""
        if not self.scoreable:
            return 0.0, float(self.population)
        low, high = wilson_interval(self.confirmed, self.scoreable, z)
        return self.population * low, self.population * high

    def to_dict(self, fallback: float, z: float) -> dict:
        low, high = self.bounds(z)
        return {
            "vowel": self.vowel_name,
            "audited": self.audited,
            "confirmed": self.confirmed,
            "said_otherwise": self.said_otherwise,
            "unclear": self.unclear,
            "population": self.population,
            "confirmed_share": (
                round(self.confirmed / self.scoreable, 4) if self.scoreable else None
            ),
            "estimated_population_sites": round(self.estimate(fallback), 1),
            "estimated_population_sites_bounds": [round(low, 1), round(high, 1)],
        }


@dataclass(frozen=True)
class DirectionResult:
    """One direction's strata, summed — the checkpoint's confirmed false rejections."""

    direction: str
    strata: list[StratumResult]

    @property
    def confirmed(self) -> int:
        return sum(s.confirmed for s in self.strata)

    @property
    def scoreable(self) -> int:
        return sum(s.scoreable for s in self.strata)

    @property
    def audited(self) -> int:
        return sum(s.audited for s in self.strata)

    @property
    def unclear(self) -> int:
        return sum(s.unclear for s in self.strata)

    @property
    def population(self) -> int:
        return sum(s.population for s in self.strata)

    @property
    def pooled_share(self) -> float:
        """Descriptive only — never scaled onto the population; see the module docstring."""
        return self.confirmed / self.scoreable if self.scoreable else 0.0

    def estimate(self) -> float:
        return sum(s.estimate(self.pooled_share) for s in self.strata)

    def bounds(self, z: float) -> tuple[float, float]:
        pairs = [s.bounds(z) for s in self.strata]
        return sum(low for low, _ in pairs), sum(high for _, high in pairs)

    def to_dict(self, z: float) -> dict:
        low, high = self.bounds(z)
        return {
            "direction": self.direction,
            "audited": self.audited,
            "confirmed": self.confirmed,
            "said_otherwise": self.scoreable - self.confirmed,
            "unclear": self.unclear,
            "unclear_share": round(self.unclear / self.audited, 4) if self.audited else None,
            "population": self.population,
            "pooled_confirmed_share": round(self.pooled_share, 4) if self.scoreable else None,
            "estimated_population_sites": round(self.estimate(), 1),
            "estimated_population_sites_bounds": [round(low, 1), round(high, 1)],
            "strata": [s.to_dict(self.pooled_share, z) for s in self.strata],
        }


def _classify(site: TashkeelSite, verdict: Adjudication) -> str:
    """Whether this adjudication confirms over-strictness, refutes it, or says nothing."""
    if verdict.verdict == UNCLEAR:
        return UNCLEAR
    if verdict.heard_vowel == site.reference_vowel:
        return CONFIRMED
    return SAID_OTHERWISE


def summarize_direction(
    sites: list[TashkeelSite],
    adjudications: dict[str, Adjudication],
    direction: str,
    strata_population: dict[str, int],
) -> DirectionResult:
    """Fold one direction's adjudicated sites into per-colour strata.

    ``strata_population`` is the mining run's count of this direction's sites per colour.
    Every colour it names becomes a stratum even when nothing in it has been audited yet, so
    an unfinished audit widens the interval instead of quietly shrinking the population.
    """
    tallies: dict[str, dict[str, int]] = {
        vowel: {CONFIRMED: 0, SAID_OTHERWISE: 0, UNCLEAR: 0} for vowel in strata_population
    }
    for site in sites:
        if site.direction != direction:
            continue
        verdict = adjudications.get(site.site_id)
        if verdict is None:
            continue
        if site.vowel_name not in tallies:
            raise ValueError(
                f"site {site.site_id} is a {direction}/{site.vowel_name} site but the "
                "summary sidecar records no population for that stratum — the worklist and "
                "the summary came from different mining runs."
            )
        tallies[site.vowel_name][_classify(site, verdict)] += 1
    return DirectionResult(
        direction=direction,
        strata=[
            StratumResult(
                direction=direction,
                vowel_name=vowel,
                confirmed=tally[CONFIRMED],
                said_otherwise=tally[SAID_OTHERWISE],
                unclear=tally[UNCLEAR],
                population=strata_population[vowel],
            )
            for vowel, tally in sorted(tallies.items())
        ],
    )


def compare(
    sites: list[TashkeelSite],
    adjudications: dict[str, Adjudication],
    population: dict,
) -> dict:
    """The two-sided over-strictness comparison, base against candidate.

    ``recovered`` counts positions the **base** got wrong and the candidate right — the
    base's confirmed false rejections. ``regressed`` counts the reverse. Both are expressed
    as rates per reference vowel, so they can be read against the aggregate recall
    :mod:`training.tashkeel_eval` reports over the same windows.

    A positive ``acceptance_gain`` means the base falsely rejects correct recitation more
    often than the candidate does — the over-strictness ADR-0003 set out to reduce.
    """
    total_vowels = population["reference_vowels"]
    strata = population["strata"]
    recovered = summarize_direction(sites, adjudications, RECOVERED, strata[RECOVERED])
    regressed = summarize_direction(sites, adjudications, REGRESSED, strata[REGRESSED])

    components = len(recovered.strata) + len(regressed.strata)
    z = component_z(components)
    base_low, base_high = recovered.bounds(z)
    candidate_low, candidate_high = regressed.bounds(z)

    def rate(count: float) -> float:
        return count / total_vowels if total_vowels else 0.0

    return {
        "reference_vowels": total_vowels,
        "recovered": recovered.to_dict(z),
        "regressed": regressed.to_dict(z),
        "base_false_rejection_rate": round(rate(recovered.estimate()), 5),
        "candidate_false_rejection_rate": round(rate(regressed.estimate()), 5),
        "acceptance_gain": round(rate(recovered.estimate()) - rate(regressed.estimate()), 5),
        "acceptance_gain_ci95": [
            round(rate(base_low) - rate(candidate_high), 5),
            round(rate(base_high) - rate(candidate_low), 5),
        ],
        "interval_method": {
            "family": "Wilson per stratum, Bonferroni-corrected, summed",
            "alpha": ALPHA,
            "components": components,
            "component_z": round(z, 4),
        },
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
                        help="the worklist's '.summary.json' sidecar, holding the per-stratum "
                             "population counts (default: alongside --worklist).")
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
            f"{summary_path} not found. The per-stratum population counts it holds are what "
            "turn the audited sample into a rate; without them the report would silently "
            "read as a census of the corpus. Pass --summary."
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
