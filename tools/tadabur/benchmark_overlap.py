"""The benchmark's disjointness plan (#245, step 1): which clips may be decoded and counted.

The 600 clips of ``Quran-Lab/quranic-asr-benchmark`` are fingerprinted against every tlog row
(:mod:`tadabur.tlog_overlap`'s index, line and vote threshold; its ``SETS`` are untouched) and
each hit names the tlog row's pre-registered role. This module turns those hits into the plan
every model's decode list follows, **before any clip is decoded**:

* a clip matching a sealed row — every pre-registered role but ``train``
  (:func:`sealed_roles`), so a tranche added later is sealed too — is dropped from every
  model's decode list: decoding it would read a sealed tranche;
* a clip matching a row the candidate's run consumed (its ``tlog_consumed.jsonl.gz``) is
  decoded but excluded from the verdict and reported apart: the candidate trained on it;
* a clip matching only pool rows the candidate never read stays in the verdict, but is named
  apart when its row is in a shard the candidate partly read: the pool is 17% consumed, so a
  hit there is a weaker, speaker-level exposure the verdict reports as a sensitivity split;
* every other clip is decoded and counts, an unconsumed row outside those shards included (the
  candidate read no row near it).

Tadabur exposure needs no audio pass here: the benchmark and the Tadabur shards are disjoint by
reciter name, screened separately. The plan is pure: it reads no audio and no fingerprint file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from tadabur.tlog_index import ROLES, TRAIN

#: Table categories that are not roles: whether a ``train`` hit was consumed, its
#: shard-level sibling, and no hit.
TRAIN_CONSUMED = "train_consumed"
TRAIN_UNCONSUMED = "train_unconsumed"
POOL_UNCONSUMED_SAME_GROUP = "pool_unconsumed_same_group"
NO_HIT = "no_hit"


@dataclass(frozen=True)
class DisjointnessPlan:
    """The clips each stage sees, and how many clips hit each category.

    ``decode`` is every clip that may be decoded, in input order; ``trained`` and ``verdict``
    split it by exposure to the rows the candidate read; ``same_group`` is the part of
    ``verdict`` that hit an unread pool row in a shard the candidate partly read (the
    sensitivity split); ``sealed`` names each dropped clip with the sealed roles it hit;
    ``by_role`` counts clips once per category hit.
    """

    decode: tuple[str, ...]
    verdict: tuple[str, ...]
    trained: tuple[str, ...]
    same_group: tuple[str, ...]
    sealed: dict[str, tuple[str, ...]]
    by_role: dict[str, int]


def row_group(row: str) -> str:
    """A tlog row key's shard, ``split:shard_file`` (the key is ``split:shard_file:row``).

    The shard is the unit the schedule downloads and a run reads whole clips from, so the
    shards a run touched are the speakers and sessions it heard.
    """
    return row.rsplit(":", 1)[0]


def sealed_roles() -> tuple[str, ...]:
    """Every pre-registered role that is not ``train``: exactly the sealed tranches today.

    Derived from :data:`tadabur.tlog_index.ROLES` at call time, so a role pre-registered later
    is sealed (dropped before any decode) rather than silently decoded and scored.
    """
    return tuple(role for role in ROLES if role != TRAIN)


def plan_disjointness(clips: Sequence[str], hits: Mapping[str, Iterable[tuple[str, str]]],
                      consumed: Iterable[str],
                      read_groups: Iterable[str] = ()) -> DisjointnessPlan:
    """Plan ``clips`` from their fingerprint ``hits`` (``(role, row)`` each) and read rows.

    ``consumed`` is the row keys the candidate's run logged; ``read_groups`` the shards those
    keys belong to (:func:`row_group`), the sensitivity split's grouping. A clip absent from
    ``hits`` is refused rather than assumed clean: an unqueried clip is an unverified one. A
    hit whose role is not pre-registered is refused too.
    """
    consumed = set(consumed)
    read_groups = set(read_groups)
    sealed_tranches = sealed_roles()
    by_role = {role: 0 for role in sealed_tranches}
    by_role.update({TRAIN_CONSUMED: 0, TRAIN_UNCONSUMED: 0, POOL_UNCONSUMED_SAME_GROUP: 0,
                    NO_HIT: 0})
    decode: list[str] = []
    trained: list[str] = []
    same_group: list[str] = []
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
        unconsumed_rows = [row for row in train_rows if row not in consumed]
        if consumed_hit:
            by_role[TRAIN_CONSUMED] += 1
        if unconsumed_rows:
            by_role[TRAIN_UNCONSUMED] += 1
        same_shard = any(row_group(row) in read_groups for row in unconsumed_rows)
        if same_shard and not consumed_hit:
            by_role[POOL_UNCONSUMED_SAME_GROUP] += 1
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
        elif same_shard:
            same_group.append(clip)
    trained_set = set(trained)
    verdict = tuple(clip for clip in decode if clip not in trained_set)
    return DisjointnessPlan(
        decode=tuple(decode),
        verdict=verdict,
        trained=tuple(trained),
        same_group=tuple(same_group),
        sealed=sealed_clips,
        by_role=by_role,
    )
