"""The pre-registered speaker-cluster procedure and its validation rule (#186, tlog S2).

`docs/tlog-holdout-preregistration.md` ("Groups and roles", items 2-3) fixes the procedure and
is frozen; this module is its only implementation, shared with S3b:

* L2-normalised embeddings, one per clip, from the first 20 s at 16 kHz mono;
* exact cosine k-nearest neighbours (**k = 20**), computed in blocks, so the kNN graph is the
  union of each row's k neighbours whichever way the blocks fall;
* an undirected edge wherever the cosine is **>= tau**; clusters are the connected components;
* the **1% cap**: any component over 1% of rows is re-clustered by the same procedure inside
  itself, at **tau + 0.02** and then +0.02 a round, until no component exceeds the cap or a
  round cannot split one further (an unsplittable component is reported, not looped on);
* deterministic: the rows are sorted by clip id before any arithmetic, and a cluster is named
  by its **smallest member clip id**, never by run or edge order. Exact cosine ties at the
  k-th neighbour are all included, so a tie never depends on iteration order.

Reading of "k nearest": a neighbour is any clip whose cosine reaches the row's k-th largest
cosine. That makes the k-th-place tie group complete rather than arbitrarily cut, and it is
what the per-row threshold below computes.

The τ pick, the chaining and text-independence tests and the same-reciter recall are read from
the components **before** the cap; the post-cap components are reported alongside for
information (architect decision 2026-10-24). Similarities are float64; the arithmetic runs on
CUDA in blocks when a GPU is present, and on the CPU otherwise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

#: The pre-registered neighbour count.
K = 20
#: The pre-registered tau grid: 0.30, 0.32, ..., 0.90 (31 values).
GRID = tuple(round(0.30 + 0.02 * step, 2) for step in range(31))
#: A component over this share of rows is re-clustered (1%).
CAP_FRACTION = 1 / 100
#: Each re-split round raises the threshold by this much above the previous one.
CAP_STEP = 0.02

#: (i) no chaining: the largest cluster is at most this multiple of the largest reciter's clips.
CHAINING_FACTOR = 3
#: (ii) text-independence: same-ayah/different-reciter co-cluster share at most this ...
SAME_AYAH_CEILING = 0.01
#: ... and at most this multiple of the different-ayah/different-reciter share.
TEXT_RATIO = 2
#: The run passes when same-reciter pairwise recall at the chosen tau reaches this.
RECALL_FLOOR = 0.9

#: Rows scored per kNN block, and columns per score chunk (sized for ~2 GiB of float64).
BLOCK_ROWS = 2048
BLOCK_COLUMNS = 16384


@dataclass(frozen=True)
class Clustering:
    """One threshold's clusters, before and after the 1% cap, plus the rounds it took."""

    pre_cap: Mapping[str, str]
    post_cap: Mapping[str, str]
    resplits: int


# --- embeddings and the kNN graph ------------------------------------------------------------


def normalise(embeddings) -> np.ndarray:
    """The input as float64 with unit rows; a zero row has no cosine and is refused."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must be a 2-D array, not {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("an all-zero embedding row has no cosine")
    return matrix / norms


def _canonical(clip_ids: Sequence[str], embeddings) -> tuple[list[str], np.ndarray]:
    """The ids sorted, with their embeddings; the one canonical order every computation uses."""
    ids = list(clip_ids)
    matrix = normalise(embeddings)
    if len(ids) != len(matrix):
        raise ValueError(f"{len(ids)} clip ids for {len(matrix)} embeddings")
    if len(set(ids)) != len(ids):
        raise ValueError("clip ids must be unique")
    order = sorted(range(len(ids)), key=ids.__getitem__)
    return [ids[index] for index in order], matrix[order]


def _block_topk(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Each row's ``k`` largest cosine values and their column indices, self excluded.

    Exact and blocked: rows in :data:`BLOCK_ROWS`, columns in :data:`BLOCK_COLUMNS`, merging
    each chunk's top ``k`` into the running one. float64, on CUDA when available.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.as_tensor(np.ascontiguousarray(matrix), dtype=torch.float64, device=device)
    rows_total = tensor.shape[0]
    values = torch.empty((rows_total, k), dtype=torch.float64, device=device)
    indices = torch.empty((rows_total, k), dtype=torch.int64, device=device)
    for start in range(0, rows_total, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, rows_total)
        block = tensor[start:stop]
        row_index = torch.arange(start, stop, device=device)
        running_values = running_indices = None
        for column_start in range(0, rows_total, BLOCK_COLUMNS):
            column_stop = min(column_start + BLOCK_COLUMNS, rows_total)
            scores = block @ tensor[column_start:column_stop].T
            column_index = torch.arange(column_start, column_stop, device=device)
            scores[row_index[:, None] == column_index[None, :]] = -torch.inf
            want = min(k, column_stop - column_start)
            chunk_values, chunk_indices = torch.topk(scores, want, dim=1)
            chunk_indices = chunk_indices + column_start
            if running_values is None:
                running_values, running_indices = chunk_values, chunk_indices
                continue
            merged_values = torch.cat([running_values, chunk_values], dim=1)
            merged_indices = torch.cat([running_indices, chunk_indices], dim=1)
            picked = torch.topk(merged_values, min(k, merged_values.shape[1]), dim=1)
            running_values = picked.values
            running_indices = merged_indices.gather(1, picked.indices)
        values[start:stop] = running_values
        indices[start:stop] = running_indices
    return values.cpu().numpy(), indices.cpu().numpy()


def knn_edges(clip_ids: Sequence[str], embeddings, tau: float, k: int = K) -> set[frozenset[str]]:
    """The undirected kNN pairs whose cosine is at least ``tau``, as two-id frozensets."""
    ids, matrix = _canonical(clip_ids, embeddings)
    rows = len(ids)
    if rows < 2:
        return set()
    values, indices = _block_topk(matrix, min(k, rows - 1))
    edges = {frozenset((ids[row], ids[column]))
             for row in range(rows)
             for value, column in zip(values[row], indices[row])
             if value >= tau and column != row}
    # The k-th cosine may be tied with columns the top-k cut; add the whole tie group, so the
    # graph never depends on which of equal values happened to be picked.
    kth = values[:, -1]
    for row in np.nonzero((values[:, :-1] == kth[:, None]).any(axis=1))[0]:
        if kth[row] < tau:
            continue
        ties = np.nonzero(matrix[row] @ matrix.T >= kth[row])[0]
        for column in ties:
            if column != row:
                edges.add(frozenset((ids[row], ids[column])))
    return edges


def _components(ids: Iterable[str], edges: Iterable[frozenset[str]]) -> dict[str, str]:
    """Connected components as ``id -> smallest member id``."""
    parent = {clip: clip for clip in ids}

    def root(clip: str) -> str:
        while parent[clip] != clip:
            parent[clip] = parent[parent[clip]]
            clip = parent[clip]
        return clip

    for edge in sorted(edges, key=sorted):
        first, second = sorted(edge)
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            # Union towards the smaller clip id, so every root is its component's smallest.
            parent[max(first_root, second_root)] = min(first_root, second_root)
    return {clip: root(clip) for clip in parent}


def cluster(clip_ids: Sequence[str], embeddings, tau: float, k: int = K) -> Clustering:
    """The pre-registered clustering at ``tau``, with the 1% cap applied to its components."""
    ids, matrix = _canonical(clip_ids, embeddings)
    position = {clip: index for index, clip in enumerate(ids)}
    pre_cap = _components(ids, knn_edges(ids, matrix, tau, k))
    post_cap = dict(pre_cap)
    cap = CAP_FRACTION * len(ids)
    threshold = tau
    resplits = 0
    while True:
        sizes = Counter(post_cap.values())
        oversized = sorted(name for name, size in sizes.items() if size > cap)
        if not oversized:
            break
        threshold += CAP_STEP
        resplits += 1
        changed = False
        for name in oversized:
            members = sorted(clip for clip, cluster_id in post_cap.items()
                             if cluster_id == name)
            if len(members) < 2:
                continue
            within = matrix[[position[clip] for clip in members]]
            pieces = _components(members, knn_edges(members, within, threshold, k))
            if len(set(pieces.values())) < 2:
                continue  # unsplittable at this threshold: report it, do not loop on it
            post_cap.update(pieces)
            changed = True
        if not changed:
            break
    return Clustering(pre_cap=pre_cap, post_cap=post_cap, resplits=resplits)


# --- the validation metrics ------------------------------------------------------------------


def _pairs(counts: Iterable[int]) -> int:
    return sum(count * (count - 1) // 2 for count in counts)


def _strata(groups: Mapping[str, str], reciters: Mapping[str, object],
            ayahs: Mapping[str, object]) -> dict[str, int]:
    """Clustered pairs in each of the three disjoint strata, by inclusion-exclusion."""
    by_reciter = Counter((groups[clip], reciters[clip]) for clip in groups)
    by_ayah = Counter((groups[clip], ayahs[clip]) for clip in groups)
    by_both = Counter((groups[clip], reciters[clip], ayahs[clip]) for clip in groups)
    same_reciter = _pairs(by_reciter.values())
    same_ayah = _pairs(by_ayah.values()) - _pairs(by_both.values())
    total = _pairs(Counter(groups.values()).values())
    return {"same_reciter": same_reciter, "same_ayah": same_ayah,
            "different_ayah": total - same_reciter - same_ayah}


def _rate(clustered: int, pairs: int) -> float:
    return clustered / pairs if pairs else 0.0


def _block(clusters: Mapping[str, str], reciters: Mapping[str, object],
           ayahs: Mapping[str, object], totals: Mapping[str, int], rows: int) -> dict:
    clustered = _strata(clusters, reciters, ayahs)
    sizes = Counter(clusters.values())
    largest = max(sizes.values()) if sizes else 0
    block = {"largest_cluster": largest, "largest_cluster_share": largest / rows if rows else 0.0,
             "recall": _rate(clustered["same_reciter"], totals["same_reciter"])}
    for stratum in ("same_reciter", "same_ayah", "different_ayah"):
        block[stratum] = {"pairs": totals[stratum], "clustered": clustered[stratum],
                          "rate": _rate(clustered[stratum], totals[stratum])}
    return block


def metrics(clustering: Clustering, reciters: Mapping[str, object],
            ayahs: Mapping[str, object]) -> dict:
    """The pre-cap and post-cap numbers the rule and the report read, for one clustering."""
    ids = list(clustering.pre_cap)
    if set(ids) != set(clustering.post_cap):
        raise ValueError("the pre- and post-cap clusterings cover different clips")
    rows = len(ids)
    one_group = {clip: "" for clip in ids}
    totals = _strata(one_group, reciters, ayahs)
    per_reciter = Counter(reciters[clip] for clip in ids)
    return {
        "rows": rows,
        "largest_reciter": max(per_reciter.values()) if per_reciter else 0,
        "resplits": clustering.resplits,
        "pre_cap": _block(clustering.pre_cap, reciters, ayahs, totals, rows),
        "post_cap": _block(clustering.post_cap, reciters, ayahs, totals, rows),
    }


# --- the pre-registered rule -----------------------------------------------------------------


def no_chaining(entry: Mapping) -> bool:
    """(i) the pre-cap largest cluster is at most 3x the largest single reciter's clips."""
    return entry["pre_cap"]["largest_cluster"] <= CHAINING_FACTOR * entry["largest_reciter"]


def text_independent(entry: Mapping) -> bool:
    """(ii) the same-ayah/different-reciter co-cluster share is <= 2x the different-ayah share
    and <= 1% (both pre-cap)."""
    same_ayah = entry["pre_cap"]["same_ayah"]["rate"]
    different_ayah = entry["pre_cap"]["different_ayah"]["rate"]
    return same_ayah <= SAME_AYAH_CEILING and same_ayah <= TEXT_RATIO * different_ayah


def choose_tau(per_tau: Mapping[float, Mapping]) -> float | None:
    """The lowest tau on the grid at which both conditions hold, or ``None``."""
    for tau in sorted(per_tau):
        if no_chaining(per_tau[tau]) and text_independent(per_tau[tau]):
            return tau
    return None


def decide(per_model: Mapping[str, Mapping[float, Mapping]]) -> dict:
    """The pre-registered choice: the first candidate model that passes, or the fallback."""
    for model in per_model:  # A, then B: the caller's order is the pre-registered one
        tau = choose_tau(per_model[model])
        if tau is None:
            continue
        if per_model[model][tau]["pre_cap"]["recall"] >= RECALL_FLOOR:
            return {"model": model, "tau": tau}
    return {"fallback": True}
