"""The #178 data derivation: #94's consumed shard prefix, and what is left to train on.

C95's arm (#94) streamed only the head of the seed-9494 shard permutation -- 10,000 steps at
~150 steps per shard is about 67 of the 345 shards -- so #178 continues C95 on the rest. The
prefix has to be derived exactly rather than estimated: a shard left in by mistake is audio
the candidate has already augmented over, and a shard dropped by mistake shrinks the corpus
for no reason.

The cross-check is `OBSERVED_COMPLETIONS` below: the 66 `[data] exposure exclusion through
shard N` lines of `/root/scratch/issue-94/logs/aug94_candidate_v2.log` (cuda-dev, run
2026-10-16T02:17Z-05:18Z), in the order they were logged. The line for a shard is printed
when the *next* shard's first row arrives, so 66 completions mean the run entered a 67th
shard and stopped inside it at step 10,000 (the log's last shard line sits between steps
9900 and 9950). The evidence for the run's own stop is in the same log's `[done] 10000 steps`
line.

The remaining shards are registered as the round's exposure use with membership `exact`
(:func:`test_the_rounds_use_is_exact_on_exactly_the_remaining_shards`): the training stream's
exact input, not an unknown subset of it.
"""

from __future__ import annotations

import pathlib

import pytest

#: #94's seed (`h448` used 1234; both #94 arms used 9494).
SOURCE_SEED = 9494

#: #178's own seed: the continuation's shard order and augmentation draws.
RUN_SEED = 178

#: Shard-completion lines of #94's C95 log, in order. See the module docstring.
OBSERVED_COMPLETIONS = (
    169, 354, 207, 312, 227, 63, 382, 251, 217, 333, 213, 212, 57, 49, 322, 244, 363, 133,
    345, 211, 201, 258, 266, 198, 59, 314, 349, 298, 157, 325, 372, 246, 216, 138, 342, 161,
    370, 103, 307, 51, 331, 365, 145, 237, 256, 46, 283, 374, 101, 152, 257, 34, 109, 43,
    219, 195, 112, 205, 352, 328, 136, 319, 165, 26, 338, 304,
)


def _h448_shards() -> list[int]:
    """`h448`'s 345 `--stream-shards`, the corpus both `h448` and C95 trained on."""
    from tadabur.shard_reader import parse_shard_spec
    from training.decode_evalset import training_shard_spec

    return parse_shard_spec(training_shard_spec())


def _consumed() -> tuple[int, ...]:
    from training.consumed_prefix import consumed_prefix

    return consumed_prefix(_h448_shards(), OBSERVED_COMPLETIONS, SOURCE_SEED)


# --- the permutation and the log's cross-check ---------------------------------------


def test_the_seed_9494_permutation_reproduces_the_order_94_logged():
    from training.consumed_prefix import shard_permutation

    shards = _h448_shards()
    assert len(shards) == 345
    assert shard_permutation(shards, SOURCE_SEED)[: len(OBSERVED_COMPLETIONS)] == list(
        OBSERVED_COMPLETIONS
    )


def test_the_derivation_reads_the_streams_own_order():
    """One definition of the order: the derivation must not drift from what is streamed."""
    from training.consumed_prefix import shard_permutation
    from training.distill_stream import StreamingWindowDataset

    shards = _h448_shards()
    dataset = StreamingWindowDataset(shards, seed=SOURCE_SEED)
    assert dataset._my_shards() == shard_permutation(shards, SOURCE_SEED)


def test_the_log_parser_reads_the_completion_order():
    from training.consumed_prefix import completion_order

    text = (
        "[  9850/10000] loss 0.2204 kl 0.2204 feat    off\n"
        "[data] exposure exclusion through shard 169 {'use': 'aug.training', 'rows_seen': 1000}\n"
        "[  9950/10000] loss 0.2238 kl 0.2238 feat    off\n"
        "[data] exposure exclusion through shard 354 {'use': 'aug.training', 'rows_seen': 2000}\n"
        "[done] 10000 steps in 3.02h\n"
    )
    assert completion_order(text) == [169, 354]


def test_the_consumed_prefix_is_the_completed_shards_plus_the_one_in_progress():
    from training.consumed_prefix import shard_permutation

    consumed = _consumed()
    expected = shard_permutation(_h448_shards(), SOURCE_SEED)[: len(OBSERVED_COMPLETIONS) + 1]
    assert len(consumed) == 67
    assert list(consumed) == expected
    assert consumed[-1] == 114  # the 67th of the permutation: entered, not finished


def test_a_log_that_does_not_match_the_permutation_is_refused():
    """A wrong order would silently leave audio C95 already trained on in the corpus."""
    from training.consumed_prefix import consumed_prefix

    reordered = list(OBSERVED_COMPLETIONS)
    reordered[3], reordered[4] = reordered[4], reordered[3]
    with pytest.raises(ValueError, match="permutation"):
        consumed_prefix(_h448_shards(), reordered, SOURCE_SEED)


# --- what is left to train on --------------------------------------------------------


def test_the_remaining_shards_are_the_345_minus_the_consumed_prefix():
    from training.consumed_prefix import remaining_shards

    shards = _h448_shards()
    remaining = remaining_shards(shards, _consumed())
    assert len(remaining) == 278
    assert remaining == sorted(remaining)
    assert not set(remaining) & set(_consumed())
    assert set(remaining) | set(_consumed()) == set(shards)


def test_the_cli_chains_repeated_logs_each_with_its_own_seed(tmp_path):
    """The launcher's one derivation call (#223): `--log PATH:SEED`, repeated in run order,
    so neither the seed nor the log pairing can go dead in a shell constant."""
    import subprocess
    import sys

    from tadabur.shard_reader import shard_spec
    from training.consumed_prefix import remaining_after_logs, shard_permutation

    after_94 = remaining_shards_for_test()
    streamed_178 = shard_permutation(after_94, RUN_SEED)[:66]
    log_94 = tmp_path / "aug94.log"
    log_94.write_text(
        "".join(
            f"[data] exposure exclusion through shard {shard} {{'use': 'x'}}\n"
            for shard in OBSERVED_COMPLETIONS
        ),
        encoding="utf-8",
    )
    log_178 = tmp_path / "train178.log"
    log_178.write_text(
        "".join(
            f"[data] exposure exclusion through shard {shard} {{'use': 'x'}}\n"
            for shard in streamed_178
        ),
        encoding="utf-8",
    )
    done = subprocess.run(
        [
            sys.executable, "-m", "training.consumed_prefix",
            "--log", f"{log_94}:{SOURCE_SEED}",
            "--log", f"{log_178}:{RUN_SEED}",
            "--monitor-seed", "2026",
        ],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    expected = remaining_after_logs([(log_94, SOURCE_SEED), (log_178, RUN_SEED)])
    assert f"stream={shard_spec(expected)}" in done.stdout
    assert "monitor=" in done.stdout and "position=" in done.stdout


def remaining_shards_for_test() -> list[int]:
    """The 278 shards #94 left, the frame #178's log is read against."""
    from training.consumed_prefix import remaining_shards

    return remaining_shards(_h448_shards(), _consumed())


def test_the_remaining_spec_round_trips_through_the_streams_parser():
    from tadabur.shard_reader import parse_shard_spec
    from training.consumed_prefix import remaining_shards, shard_spec

    assert shard_spec([1, 2, 3, 7]) == "1-3,7"
    remaining = remaining_shards(_h448_shards(), _consumed())
    assert parse_shard_spec(shard_spec(remaining)) == remaining


def test_the_rounds_use_is_exact_on_exactly_the_remaining_shards():
    from tadabur.exposure import (
        CONSONANT_TRAINING,
        TRAINING_AND_TUNING_USES,
        load_registry,
    )
    from training.consumed_prefix import remaining_shards

    remaining = remaining_shards(_h448_shards(), _consumed())
    registry = load_registry()
    use = registry.shard_uses[CONSONANT_TRAINING]
    assert list(use.shards) == remaining
    assert use.membership == "exact"  # the stream's exact input, not an unknown subset
    assert not use.shared_baseline
    # A training use, so the stream drops the sealed panel's reciters and every row tied to
    # an evaluation recording before decoding (acceptance rules §6).
    assert CONSONANT_TRAINING in TRAINING_AND_TUNING_USES


def test_a_log_with_no_completions_is_refused():
    """An empty, rotated or wrongly-pathed log would otherwise 'prove' an empty prefix and
    move every shard #94 consumed back into the corpus."""
    from training.consumed_prefix import consumed_prefix

    with pytest.raises(ValueError, match="no shard-completion"):
        consumed_prefix(_h448_shards(), [], SOURCE_SEED)


def test_a_second_continuation_leaves_only_what_both_runs_left(tmp_path):
    """#223's control trains after #94 *and* #178: a shard either run streamed is audio the
    arm would be measured against having already seen."""
    from training.consumed_prefix import (
        consumed_prefix,
        remaining_after_logs,
        remaining_shards,
        shard_permutation,
    )

    after_94 = remaining_shards(_h448_shards(), _consumed())
    streamed_178 = shard_permutation(after_94, RUN_SEED)[:66]
    log_94 = tmp_path / "aug94.log"
    log_94.write_text(
        "".join(
            f"[data] exposure exclusion through shard {shard} {{'use': 'x'}}\n"
            for shard in OBSERVED_COMPLETIONS
        ),
        encoding="utf-8",
    )
    log_178 = tmp_path / "train178.log"
    log_178.write_text(
        "".join(
            f"[data] exposure exclusion through shard {shard} {{'use': 'x'}}\n"
            for shard in streamed_178
        ),
        encoding="utf-8",
    )
    remaining = remaining_after_logs([(log_94, SOURCE_SEED), (log_178, RUN_SEED)])
    assert remaining == remaining_shards(
        after_94, consumed_prefix(after_94, streamed_178, RUN_SEED)
    )
    assert len(remaining) == 278 - 67  # #94 left 278; #178 entered 67 of them
    assert not set(remaining) & (set(_consumed()) | set(streamed_178))


# --- the shard validation reads, and the guard that keeps it out of training ----------


def test_the_monitor_shard_is_the_last_of_the_runs_seed_permutation():
    """Validation clips come from this shard, so the run must never stream it."""
    from training.consumed_prefix import monitoring_shard, remaining_shards, shard_permutation

    remaining = remaining_shards(_h448_shards(), _consumed())
    monitor, position = monitoring_shard(remaining, RUN_SEED)
    assert monitor == shard_permutation(remaining, RUN_SEED)[-1] == 90
    assert position == len(remaining) - 1 == 277


def test_the_monitor_guard_fires_only_once_the_run_enters_the_shard():
    """A shard's line is printed when it is *left*: entering needs one fewer completion."""
    from training.consumed_prefix import monitor_reached, remaining_shards

    remaining = remaining_shards(_h448_shards(), _consumed())
    assert not monitor_reached(remaining, [0] * 276, RUN_SEED)
    assert monitor_reached(remaining, [0] * 277, RUN_SEED)
