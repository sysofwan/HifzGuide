"""The benchmark's disjointness plan (#245, step 1): which clips may be decoded and counted.

The 600 clips of ``Quran-Lab/quranic-asr-benchmark`` are fingerprinted against every tlog row
(:mod:`tadabur.tlog_overlap`'s index, line and vote threshold; its ``SETS`` are untouched) and
each hit names the tlog row's pre-registered role. This module turns those hits into the plan
every model's decode list follows, **before any clip is decoded**:

* a clip matching a sealed row — every pre-registered role but ``train``
  (:func:`sealed_roles`), so a tranche added later is sealed too — is dropped from every
  model's decode list: decoding it would read a sealed tranche;
* a clip matching a row ``tlog.training`` consumed is decoded but excluded from the verdict and
  reported apart: the candidate may have trained on it;
* every other clip is decoded and counts, a match of an unconsumed ``train`` row included (that
  row is not in the schedule the candidate was trained from, so it is not exposure).

Tadabur exposure needs no audio pass here: the benchmark and the Tadabur shards are disjoint by
reciter name, screened separately. The plan is pure: it reads no audio and no fingerprint file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from tadabur.tlog_index import ROLES, TRAIN

#: Table categories that are not roles: whether a ``train`` hit was consumed, and no hit.
TRAIN_CONSUMED = "train_consumed"
TRAIN_UNCONSUMED = "train_unconsumed"
NO_HIT = "no_hit"


@dataclass(frozen=True)
class DisjointnessPlan:
    """The clips each stage sees, and how many clips hit each category.

    ``decode`` is every clip that may be decoded, in input order; ``trained`` and ``verdict``
    split it by exposure to ``tlog.training``; ``sealed`` names each dropped clip with the
    sealed roles it hit; ``by_role`` counts clips once per category hit.
    """

    decode: tuple[str, ...]
    verdict: tuple[str, ...]
    trained: tuple[str, ...]
    sealed: dict[str, tuple[str, ...]]
    by_role: dict[str, int]


def sealed_roles() -> tuple[str, ...]:
    """Every pre-registered role that is not ``train``: exactly the sealed tranches today.

    Derived from :data:`tadabur.tlog_index.ROLES` at call time, so a role pre-registered later
    is sealed (dropped before any decode) rather than silently decoded and scored.
    """
    return tuple(role for role in ROLES if role != TRAIN)


def plan_disjointness(clips: Sequence[str], hits: Mapping[str, Iterable[tuple[str, str]]],
                      consumed: Iterable[str]) -> DisjointnessPlan:
    """Plan ``clips`` from their fingerprint ``hits`` (``(role, row)`` each) and consumed rows.

    A clip absent from ``hits`` is refused rather than assumed clean: an unqueried clip is an
    unverified one. A hit whose role is not pre-registered is refused too.
    """
    consumed = set(consumed)
    sealed_tranches = sealed_roles()
    by_role = {role: 0 for role in sealed_tranches}
    by_role.update({TRAIN_CONSUMED: 0, TRAIN_UNCONSUMED: 0, NO_HIT: 0})
    decode: list[str] = []
    trained: list[str] = []
    sealed_clips: dict[str, tuple[str, ...]] = {}
    for clip in clips:
        if clip not in hits:
            raise ValueError(f"{clip} has no fingerprint result; refusing to assume it clean")
        clip_hits = list(hits[clip])
        roles = set()
        for role, _ in clip_hits:
            if role not in ROLES:
                raise ValueError(f"{clip} hits a row with the unknown role {role!r}")
            roles.add(role)
        for role in sealed_tranches:
            if role in roles:
                by_role[role] += 1
        train_rows = [row for role, row in clip_hits if role == TRAIN]
        consumed_hit = any(row in consumed for row in train_rows)
        if consumed_hit:
            by_role[TRAIN_CONSUMED] += 1
        if any(row not in consumed for row in train_rows):
            by_role[TRAIN_UNCONSUMED] += 1
        if not clip_hits:
            by_role[NO_HIT] += 1
        hit_sealed = tuple(dict.fromkeys(role for role, _ in clip_hits
                                         if role in sealed_tranches))
        if hit_sealed:
            sealed_clips[clip] = hit_sealed
            continue
        decode.append(clip)
        if consumed_hit:
            trained.append(clip)
    trained_set = set(trained)
    verdict = tuple(clip for clip in decode if clip not in trained_set)
    return DisjointnessPlan(
        decode=tuple(decode),
        verdict=verdict,
        trained=tuple(trained),
        sealed=sealed_clips,
        by_role=by_role,
    )
