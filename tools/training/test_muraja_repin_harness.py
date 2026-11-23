"""Muraja's own engine under strict, and the per-word trace, through the built harness (#306), at
the frozen pin (99c326f) and at 028bc51. Each test skips where its harness is not built
(``bash muraja_harness/build.sh [<commit>]``, macOS, ``gh`` auth)."""

from __future__ import annotations

import json

import pytest

from muraja_harness.make_parity_fixtures import FIXTURES_PATH, request
from training import device_logs as dl
from training import muraja_policy as mp
from training.muraja_policy import (
    HARNESS_PROTOCOL,
    LOG_REPLAY_BINARY,
    MURAJA_COMMIT,
    SHIPPED_DEFAULT,
    Harness,
)

CURRENT = "028bc513eeb233caa7240a5c37c423da59f3d9d7"
COMMITS = [MURAJA_COMMIT, CURRENT]
FIXTURES = {c["case"]: c for c in json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))["cases"]}
#: What Muraja's ``ScoringParameters.strict`` switches (``FollowAlong/FollowAlongTypes.swift:111-139``):
#: no soft pairs, no shaddah suppression, the phoneme gate on, haraka drops counted.
STRICT_FLAGS = {"softPairsEnabled": False, "shaddahSuppression": False,
                "phonemeGateEnabled": True, "suppressHarakaDrop": False}
BALANCED_FLAGS = {"softPairsEnabled": True, "shaddahSuppression": True,
                  "phonemeGateEnabled": True, "suppressHarakaDrop": False}


def harness_at(commit: str, binary: str = mp.ITEM_BINARY) -> Harness:
    harness = Harness.locate(commit=commit, binary=binary)
    if harness is None:
        pytest.skip(f"the Muraja harness at {commit[:7]} is not built here")
    return harness


def graded(commit: str, case: str, scorings: list[mp.Scoring]) -> tuple[dict, list[dict]]:
    harness = harness_at(commit)
    asked = {**request(FIXTURES[case]), "scorings": mp.engine_runs(scorings)}
    _, results = harness.run([asked])
    return asked, results


@pytest.mark.parametrize("commit", COMMITS)
def test_the_harness_built_for_a_commit_names_that_commit_and_speaks_the_protocol(commit):
    for binary in (mp.ITEM_BINARY, LOG_REPLAY_BINARY):
        harness = harness_at(commit, binary)
        assert harness.muraja_commit == commit and harness.build["muraja_commit"] == commit
        assert harness.build["protocol_version"] == HARNESS_PROTOCOL
        assert "mode" in harness.build["scoring_keys"]


@pytest.mark.parametrize("commit", COMMITS)
def test_strict_runs_muraja_s_strict_parameters_and_balanced_its_own(commit):
    _, results = graded(commit, "missing_haraka", [mp.STRICT, SHIPPED_DEFAULT])
    by_name = {r["scoring"]: r for r in results}
    assert by_name[mp.STRICT.engine_name]["parameters"] == STRICT_FLAGS
    assert by_name[SHIPPED_DEFAULT.engine_name]["parameters"] == BALANCED_FLAGS


@pytest.mark.parametrize("commit", COMMITS)
def test_strict_does_not_forgive_the_bare_double_that_balanced_forgives(commit):
    _, results = graded(commit, "bare_double_geminate", [mp.STRICT, SHIPPED_DEFAULT])
    quality = {r["scoring"]: {s["word"]: s["quality"] for s in r["final"]}[3] for r in results}
    assert quality == {mp.STRICT.engine_name: "minor", SHIPPED_DEFAULT.engine_name: "correct"}


@pytest.mark.parametrize("commit", COMMITS)
def test_a_trace_from_the_engine_agrees_with_what_the_engine_printed(commit):
    scorings = [mp.STRICT, SHIPPED_DEFAULT]
    asked, results = graded(commit, "missing_haraka", scorings)
    rows = mp.word_traces(asked, results, scorings)
    assert rows and {r["scoring"] for r in rows} == {"strict", "shipped_default"}
    printed = {r["scoring"]: r for r in results}
    for row in rows:
        result = printed[next(s for s in scorings if s.name == row["scoring"]).engine_name]
        kept = next(s for s in result["final"] if s["word"] == row["word"])
        assert (row["quality"], row["score"], row["errors"]) == (kept["quality"], kept["score"], kept["errors"])
        assert len(row["checks"]) == len(asked["checks"]) == len(result["checks"])
        for index, check in enumerate(row["checks"]):
            assert (check["hop"], check["overlap"]) == (asked["checks"][index]["hop"], asked["checks"][index]["overlap"])
            own = [s for s in result["checks"][index]["statuses"] if s["word"] == row["word"]]
            assert check["status"] == (own[0] if own else None)
        assert row["setting_checks"] == list(mp.setting_checks(result["checks"], row["word"], kept))
        assert all(row["checks"][i]["status"] is not None for i in row["setting_checks"])


@pytest.mark.skipif(Harness.locate(binary=LOG_REPLAY_BINARY) is None, reason="the pin's log replay is not built here")
def test_the_log_replay_runs_a_log_under_strict_at_the_pin():
    from training.device_logs import baseline_differences, replay_logs

    harness = harness_at(MURAJA_COMMIT, LOG_REPLAY_BINARY)
    log = "surah2_page4_strict_shaddah"
    path = dl.log_dir(MURAJA_COMMIT) / f"{log}.log"
    _, replays = replay_logs(harness, {log: path}, [SHIPPED_DEFAULT, mp.STRICT])
    assert replays[log][mp.STRICT.engine_name].words
    baseline = json.loads((dl.log_dir(MURAJA_COMMIT) / f"{log}.baseline.json").read_text(encoding="utf-8"))
    assert baseline_differences(replays[log][SHIPPED_DEFAULT.engine_name], baseline) == []


@pytest.mark.slow
def test_all_74_muraja_device_logs_reproduce_their_baselines_at_028bc51():
    """Gate P1's first half: Muraja's own logs against Muraja's own baselines at the audit's
    commit, balanced, as #151 did; exact. A failure blocks the re-pin: do not loosen it."""
    harness = harness_at(CURRENT, LOG_REPLAY_BINARY)
    directory = dl.log_dir(CURRENT)
    logs = sorted(p.stem for p in directory.glob("*.log"))
    assert len(logs) == 74 and len(list(directory.glob("*.baseline.json"))) == 74
    _, replays = dl.replay_logs(harness, {log: directory / f"{log}.log" for log in logs}, [SHIPPED_DEFAULT])
    differing = {}
    for log in logs:
        baseline = json.loads((directory / f"{log}.baseline.json").read_text(encoding="utf-8"))
        if diff := dl.baseline_differences(replays[log][SHIPPED_DEFAULT.engine_name], baseline):
            differing[log] = diff
    assert differing == {}
