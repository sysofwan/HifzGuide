"""Assemble the committed summary the App-arm doc renders from (#309).

The summary is the run's one machine-readable result: the population and its failures, both
scorings' flag rates, P2's bridge, P3's cadence and the candidate pools. S2 (#308) fills the
``oracle`` section in; until then it stays ``null`` and the doc renders its parts as pending.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tadabur.followalong_audit.rates import flag_rates
from training.muraja_policy import AUDIT_SCORINGS


def build_summary(grades: Sequence[Mapping], *, clips: int, failed: Sequence[Mapping],
                  p2: Mapping, p3: Mapping, pools: Mapping,
                  oracle: Mapping | None = None) -> dict:
    """The doc's whole input, from a finished run's tables."""
    return {
        "population": {"clips": clips, "failed": list(failed)},
        "rates": {scoring.name: flag_rates(grades, scoring.name) for scoring in AUDIT_SCORINGS},
        "oracle": dict(oracle) if oracle is not None else None,
        "p2": dict(p2),
        "p3": dict(p3),
        "pools": pools["pools"],
        "s1_by_cause_and_base": pools["s1_by_cause_and_base"],
    }
