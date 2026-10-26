"""The suite's own contract: the ``slow`` tier is declared, so ``-m "not slow"`` is a real tier.

``scripts/test.sh`` runs ``-m "not slow"`` in the inner loop and everything (slow tier included)
at the merge gate. The marker must be declared for the tier to be discoverable
(``pytest --markers``, ``--strict-markers``) and for a misspelt marker to fail loudly instead of
silently running in the inner loop.
"""

from __future__ import annotations


def test_the_slow_tier_marker_is_declared(pytestconfig):
    declared = {line.split(":", 1)[0].strip() for line in pytestconfig.getini("markers")}
    assert "slow" in declared, (
        "declare `slow` in tools/pytest.ini; scripts/test.sh selects the inner loop with "
        "-m 'not slow' and the merge gate with --full")
