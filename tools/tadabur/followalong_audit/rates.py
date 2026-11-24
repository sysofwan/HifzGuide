"""Flags per 1,000 graded words, by cause, with a clip-bootstrap interval (#309).

Reads ``word_grade_rows`` lines. The denominator is the **graded** words: an ungraded word is the
S5 stratum, not a pass. A word counts once per cause it carries, so the cause rows can sum past
the flagged total. The interval resamples whole clips (``item``): a clip's words are not
independent, and a rate that treated them as such would be narrower than the data supports.

The clusters are the clips' aggregates, not their words: a word list of ~90,000 rows resampled
4,000 times per statistic is minutes of pure Python, while the same sums over ~5,700 one-line
aggregates are the same arithmetic in a fraction of the time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tadabur.tlog_mining.candidate_inventory import TRUST_BOOTSTRAP_ITERATIONS, TRUST_BOOTSTRAP_SEED
from training.decode_evalset import cluster_bootstrap
from training.word_flags import CAUSES


def flag_rates(rows: Sequence[Mapping], scoring: str, *,
               seed: int = TRUST_BOOTSTRAP_SEED,
               iterations: int = TRUST_BOOTSTRAP_ITERATIONS) -> dict:
    """The flagged words per 1,000 graded words under ``scoring``, overall and per cause."""
    clusters = _clip_aggregates(rows, scoring)
    graded = sum(cluster["graded"] for cluster in clusters)
    flagged = sum(cluster["flagged"] for cluster in clusters)
    total = {
        "graded": graded,
        "flagged": flagged,
        "per_1000": _per_1000(flagged, graded),
        "interval": list(_interval(clusters, "flagged", seed, iterations)),
    }
    causes = {}
    for cause in CAUSES:
        hits = sum(cluster[cause] for cluster in clusters)
        causes[cause] = {
            "flagged": hits,
            "per_1000": _per_1000(hits, graded),
            "interval": list(_interval(clusters, cause, seed, iterations)),
        }
    return {**total, "causes": causes}


def _clip_aggregates(rows: Sequence[Mapping], scoring: str) -> list[dict]:
    """Each clip once: its graded words, flagged words and per-cause flagged words."""
    by_item: dict[str, dict] = {}
    for row in rows:
        cell = row[scoring]
        cluster = by_item.setdefault(row["item"], {"item": row["item"], "graded": 0,
                                                   "flagged": 0, **{c: 0 for c in CAUSES}})
        if not cell["graded"]:
            continue
        cluster["graded"] += 1
        if cell["flagged"]:
            cluster["flagged"] += 1
            for cause in set(cell["causes"]):
                if cause in cluster:
                    cluster[cause] += 1
    return [by_item[item] for item in sorted(by_item)]


def _per_1000(hits: int, graded: int) -> float:
    return 1000.0 * hits / graded if graded else 0.0


def _interval(clusters: Sequence[Mapping], field: str, seed: int,
              iterations: int) -> tuple[float, float]:
    def statistic(sample: list) -> float:
        return _per_1000(sum(cluster[field] for cluster in sample),
                         sum(cluster["graded"] for cluster in sample))

    return cluster_bootstrap(list(clusters), lambda cluster: cluster["item"], statistic,
                             seed=seed, iterations=iterations)
