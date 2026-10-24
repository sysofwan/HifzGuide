"""Acceptance tests for ``tadabur.speaker_cluster`` (#186, tlog S2).

The pre-registered procedure (``docs/tlog-holdout-preregistration.md``, "Groups and roles",
items 2-3): L2-normalised embeddings, exact cosine k-nearest neighbours (k = 20), an edge where
cosine >= tau, connected components, and the 1% cap that re-clusters an oversized component at
tau + 0.02, repeatedly. The τ pick, the chaining and text-independence tests and the
same-reciter recall are read from the components BEFORE the cap; the post-cap numbers are
reported alongside for information (architect decision 2026-10-24).

L2-normalised rows are the input contract; cosine is float64. Cluster ids are named by the
smallest member clip id (lexicographic), never by run or edge order, so the same input gives the
same ids whatever order the rows arrive in; ties in the k-th neighbour are broken by clip id.

Test command, from the repo root::

    python -m pytest -q tools/tadabur/test_speaker_cluster.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tadabur.speaker_cluster import (
    GRID,
    K,
    Clustering,
    choose_tau,
    cluster,
    decide,
    knn_edges,
    metrics,
    no_chaining,
    normalise,
    text_independent,
)


# --- the pre-registered constants ------------------------------------------------------------


def test_the_grid_is_the_preregistered_tau_grid():
    assert K == 20
    assert GRID == tuple(round(0.30 + 0.02 * step, 2) for step in range(31))
    assert (GRID[0], GRID[-1], len(GRID)) == (0.30, 0.90, 31)


# --- embeddings and the kNN graph ------------------------------------------------------------


def test_normalise_makes_unit_rows_in_float64():
    out = normalise(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
    assert out.dtype == np.float64
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert np.allclose(normalise(out), out)  # idempotent


def test_normalise_refuses_a_zero_row():
    with pytest.raises(ValueError):
        normalise(np.array([[0.0, 0.0]], dtype=np.float64))


def _unit(degrees: float) -> list[float]:
    return [math.cos(math.radians(degrees)), math.sin(math.radians(degrees))]


#: Four rows on the unit circle: 0, 9, 20 and 180 degrees. cosines: 1-2 .988, 2-3 .982,
#: 1-3 .940, and 4 is nearly opposite all of them.
CIRCLE_IDS = ("a", "b", "c", "d")
CIRCLE = np.array([_unit(0.0), _unit(9.0), _unit(20.0), _unit(180.0)], dtype=np.float64)


def test_the_knn_graph_is_the_union_of_each_row_s_neighbours():
    # k = 1: a->b, b->a, c->b, d->c, so the undirected union is a-b, b-c, c-d. The highly
    # similar a-c pair (.940) is in neither list and must not appear, even at tau = 0.
    edges = knn_edges(CIRCLE_IDS, CIRCLE, tau=0.0, k=1)
    assert edges == {frozenset({"a", "b"}), frozenset({"b", "c"}), frozenset({"c", "d"})}
    assert frozenset({"a", "c"}) not in edges


def test_an_edge_needs_a_cosine_at_least_tau_among_the_k_nearest():
    # k = 2: a-b (.988), a-c (.940), b-c (.982) and d's two nearest are c and b, both negative.
    edges = knn_edges(CIRCLE_IDS, CIRCLE, tau=0.95)
    assert edges == {frozenset({"a", "b"}), frozenset({"b", "c"})}
    # Tau 1.0 leaves nothing: no pair reaches a cosine of one.
    assert knn_edges(CIRCLE_IDS, CIRCLE, tau=1.0) == set()


def _circle_clusters(tau: float) -> Clustering:
    return cluster(CIRCLE_IDS, CIRCLE, tau)


def test_connected_components_name_each_cluster_by_its_smallest_member():
    # tau 0.9 keeps a-b, b-c (and a-c) but not c-d: one 3-row cluster plus d alone.
    found = _circle_clusters(0.9)
    assert found.pre_cap == {"a": "a", "b": "a", "c": "a", "d": "d"}
    # renamed because the ids sort differently: the smallest member is what names it
    ids = ("c", "b", "a")
    renamed = cluster(ids, CIRCLE[:3], 0.9)
    assert renamed.pre_cap == {"a": "a", "b": "a", "c": "a"}


def test_the_same_input_gives_the_same_clusters_whatever_the_row_order():
    rng = np.random.default_rng(20261024)
    ids = [f"clip{i:03d}" for i in range(40)]
    # Duplicate rows make exact cosine ties, so the tie-break must follow the clip id.
    base = rng.normal(size=(6, 8))
    embeddings = np.repeat(base, 7, axis=0)[:40]
    first = cluster(ids, embeddings, 0.5)
    second = cluster(ids, embeddings, 0.5)
    assert first == second
    order = list(range(len(ids)))[::-1]
    permuted = cluster([ids[i] for i in order], embeddings[order], 0.5)
    assert permuted == first
    assert knn_edges([ids[i] for i in order], embeddings[order], 0.5) \
        == knn_edges(ids, embeddings, 0.5)


# --- the 1% re-split cap ---------------------------------------------------------------------


def _one_hot(index: int, width: int = 128) -> list[float]:
    row = [0.0] * width
    row[index] = 1.0
    return row


def _hundred_with_a_pair(cosine: float) -> tuple[list[str], np.ndarray]:
    """98 mutually orthogonal rows plus one pair at exactly ``cosine``."""
    ids = [f"c{index:03d}" for index in range(98)] + ["p", "q"]
    rows = [_one_hot(index) for index in range(98)]
    rows.append(_one_hot(98))
    q = _one_hot(98)
    q[98] = cosine
    q[99] = math.sqrt(1.0 - cosine * cosine)
    rows.append(q)
    return ids, normalise(np.array(rows, dtype=np.float64))


def test_the_cap_re_splits_a_component_over_one_percent_at_tau_plus_0_02():
    ids, embeddings = _hundred_with_a_pair(0.51)  # 1% of 100 rows is one row
    found = cluster(ids, embeddings, 0.50)
    assert found.pre_cap["p"] == found.pre_cap["q"] == "p"  # one 2-row component at tau
    assert found.post_cap["p"] == "p" and found.post_cap["q"] == "q"  # split at 0.52
    assert found.resplits >= 1
    sizes = np.unique(list(found.post_cap.values()), return_counts=True)[1]
    assert sizes.max() <= 1


def test_the_cap_terminates_when_a_component_cannot_be_split_further():
    # Identical rows stay one component at every threshold above tau: the procedure must stop
    # rather than loop, and report the component it could not break.
    ids, embeddings = _hundred_with_a_pair(1.0)
    found = cluster(ids, embeddings, 0.50)
    assert found.pre_cap["p"] == found.pre_cap["q"] == "p"
    assert found.post_cap["p"] == found.post_cap["q"] == "p"
    assert isinstance(found.resplits, int) and found.resplits >= 1


# --- the validation metrics ------------------------------------------------------------------


#: Two clusters over four clips: {a, b} and {c, d}; reciters 1, 1, 2, 2; ayat 7:1, 8:1, 7:1,
#: 9:1. Pairs: same reciter a-b and c-d (both clustered); same ayah a-c (not); different ayah
#: a-d, b-c, b-d (none clustered).
FOUR = Clustering(
    pre_cap={"a": "a", "b": "a", "c": "c", "d": "c"},
    post_cap={"a": "a", "b": "a", "c": "c", "d": "c"},
    resplits=0,
)
RECITERS = {"a": 1, "b": 1, "c": 2, "d": 2}
AYAHS = {"a": "7:1", "b": "8:1", "c": "7:1", "d": "9:1"}


def test_the_metrics_count_disjoint_pair_strata():
    found = metrics(FOUR, RECITERS, AYAHS)
    assert found["rows"] == 4
    assert found["largest_reciter"] == 2
    for block in ("pre_cap", "post_cap"):
        assert found[block]["largest_cluster"] == 2
        assert found[block]["largest_cluster_share"] == 0.5
        assert found[block]["recall"] == 1.0
        assert found[block]["same_reciter"] == {"pairs": 2, "clustered": 2, "rate": 1.0}
        assert found[block]["same_ayah"] == {"pairs": 1, "clustered": 0, "rate": 0.0}
        assert found[block]["different_ayah"] == {"pairs": 3, "clustered": 0, "rate": 0.0}


def test_a_same_ayah_pair_is_not_counted_as_a_different_ayah_pair():
    found = Clustering(pre_cap={"a": "a", "b": "a"}, post_cap={"a": "a", "b": "a"}, resplits=0)
    out = metrics(found, {"a": 1, "b": 2}, {"a": "7:1", "b": "7:1"})
    assert out["pre_cap"]["same_ayah"] == {"pairs": 1, "clustered": 1, "rate": 1.0}
    assert out["pre_cap"]["different_ayah"] == {"pairs": 0, "clustered": 0, "rate": 0.0}
    assert out["pre_cap"]["same_reciter"] == {"pairs": 0, "clustered": 0, "rate": 0.0}


# --- the pre-registered rule -----------------------------------------------------------------


def _entry(*, largest_cluster: int, largest_reciter: int = 20, recall: float = 1.0,
           same_ayah: float = 0.0, different_ayah: float = 0.0,
           post_largest: int | None = None) -> dict:
    """One τ's metrics, shaped as :func:`tadabur.speaker_cluster.metrics` returns them."""
    def block(largest: int, same: float, different: float) -> dict:
        return {
            "largest_cluster": largest,
            "largest_cluster_share": largest / 2000,
            "recall": recall,
            "same_reciter": {"pairs": 100, "clustered": 0, "rate": recall},
            "same_ayah": {"pairs": 600, "clustered": 0, "rate": same},
            "different_ayah": {"pairs": 1_000_000, "clustered": 0, "rate": different},
        }
    return {"rows": 2000, "largest_reciter": largest_reciter,
            "pre_cap": block(largest_cluster, same_ayah, different_ayah),
            "post_cap": block(post_largest if post_largest is not None else largest_cluster,
                              same_ayah, different_ayah)}


def test_no_chaining_is_three_times_the_largest_reciter():
    assert no_chaining(_entry(largest_cluster=60, largest_reciter=20))
    assert not no_chaining(_entry(largest_cluster=61, largest_reciter=20))


def test_text_independence_needs_both_the_ratio_and_the_absolute_ceiling():
    assert text_independent(_entry(largest_cluster=50, same_ayah=0.01, different_ayah=0.005))
    assert not text_independent(_entry(largest_cluster=50, same_ayah=0.011, different_ayah=0.02))
    assert not text_independent(_entry(largest_cluster=50, same_ayah=0.009,
                                       different_ayah=0.004))
    assert text_independent(_entry(largest_cluster=50, same_ayah=0.0, different_ayah=0.0))


def test_the_tau_is_the_lowest_that_meets_both_conditions():
    per_tau = {
        0.30: _entry(largest_cluster=61),                      # chaining
        0.32: _entry(largest_cluster=50, same_ayah=0.02),      # not text independent
        0.34: _entry(largest_cluster=50),
        0.36: _entry(largest_cluster=50),
    }
    assert choose_tau(per_tau) == 0.34


def test_the_rule_reads_the_pre_cap_components_only():
    # The cap would hide chaining: pre-cap 61 rows, post-cap 5. The τ is not chosen.
    hidden = _entry(largest_cluster=61, largest_reciter=20, post_largest=5)
    assert not no_chaining(hidden)
    assert choose_tau({0.30: hidden}) is None


def test_the_run_passes_only_if_recall_at_the_chosen_tau_is_0_9_or_more():
    low = {0.30: _entry(largest_cluster=40, recall=0.899)}
    at_bar = {0.30: _entry(largest_cluster=40, recall=0.9)}
    assert decide({"a": low}) == {"fallback": True}
    assert decide({"a": at_bar}) == {"model": "a", "tau": 0.30}


def test_the_first_passing_model_wins_and_the_fallback_is_stated():
    passed = {0.30: _entry(largest_cluster=40)}
    failed = {0.30: _entry(largest_cluster=40, recall=0.5)}
    assert decide({"a": passed, "b": passed}) == {"model": "a", "tau": 0.30}
    assert decide({"a": failed, "b": passed}) == {"model": "b", "tau": 0.30}
    assert decide({"a": failed, "b": failed}) == {"fallback": True}
    # A model with no τ meeting (i) and (ii) does not pass, whatever its recall elsewhere.
    no_tau = {0.30: _entry(largest_cluster=61),
              0.32: _entry(largest_cluster=40, recall=1.0, same_ayah=0.02)}
    assert decide({"a": no_tau, "b": passed}) == {"model": "b", "tau": 0.30}


def test_the_cap_fraction_is_the_preregistered_one_percent():
    from tadabur import speaker_cluster

    assert float(speaker_cluster.CAP_FRACTION) == 0.01
    assert speaker_cluster.CAP_STEP == 0.02
