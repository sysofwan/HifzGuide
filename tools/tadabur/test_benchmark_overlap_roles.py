"""Tests for the sealed-role derivation (#245, self-check r1): the plan must fail closed."""

from __future__ import annotations

from tadabur import benchmark_overlap
from tadabur.tlog_index import ROLES, TRAIN


def test_every_non_train_role_is_sealed():
    assert set(benchmark_overlap.sealed_roles()) == set(ROLES) - {TRAIN}
    assert TRAIN not in benchmark_overlap.sealed_roles()


def test_a_role_added_to_the_index_is_sealed_not_decoded(monkeypatch):
    monkeypatch.setattr(benchmark_overlap, "ROLES", (*ROLES, "look2"))
    plan = benchmark_overlap.plan_disjointness(["c"], {"c": [("look2", "row-1")]}, set())
    assert plan.decode == ()
    assert plan.sealed == {"c": ("look2",)}
    assert plan.by_role["look2"] == 1
