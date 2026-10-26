"""Shared fixtures for the `tools/` suite.

The round's coverage table is a frozen read of committed data that costs minutes to build (it
phonetizes every Tarteel v1 clip to find the ones without a reference). Three tests read the same
record, so it is built once per session; the `xdist_group("coverage")` marker keeps them on one
worker, where they share that one build.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def coverage_record() -> dict:
    """The round's frozen coverage table (``training.consonant_readout.coverage``)."""
    from training.consonant_readout import coverage

    return coverage()
