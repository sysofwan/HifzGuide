"""Phase B's decode (#263): the mined rows through the teacher, C95 and round 2's EMA candidate.

The row selection is :func:`tadabur.tlog_mining.selection.phase_b_rows`; this module is what turns
those rows into the three committed ``tlog.mined.<model>.jsonl.gz`` files. One clip is read whole
and decoded into both stores the round-2 confirm decodes carry — the app's check sequence
(``checks``) and the whole-clip span decode (``spans``) — by
:func:`tadabur.tlog_eval_decode.run_shard`, which appends one line per clip per model to
``tlog.<tranche>.<shard>.<model>.jsonl`` so a killed run resumes by clip key.

:func:`decode_batch` is one worker's batch: the budget is measured from the worker's spawn and
checked **before each row**, and a row is atomic across models and stores (all three lines are
written together), so a batch never leaves a half-decoded clip behind. ``done`` is true only when
every row holds every model, which is what lets the give-way driver tell "no row left" from "the
budget ran out".

The stores are declared here, not in ``tlog_eval_decode``: the shared decoder's own runs stay
check-only, and phase B is the first mining pass to read ``spans``. The record shape is the shared
one — ``label_source`` is added at merge time from the manifest row, so the round-2 files and the
decoder's ``_record`` stay untouched.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from tadabur import tlog_eval_decode as ted
from training.external_decode import resume_keys
from training.nonpro_decode import item_key
from training.truth_baseline import CHECK_STORE, PROTOCOLS, Store

from .batch import BatchResult

#: Phase B's models: the teacher, C95 and round 2's chosen candidate at its EMA weights.
MINED_MODELS = ("base", "c95", "r2_candidate_ema")
#: The committed merged files, and the identity records' prefix in ``meta.json``.
MINED_PREFIX = "tlog.mined"
#: Where this store's decodes are committed.
MINED_DECODES_DIR = Path(__file__).parent / "decodes"


def _spans(decoder, span) -> dict:
    return {"spans": decoder.decode_spans([span])[0]}


#: Phase B's two stores: the check sequence the app plays, and the whole-clip span decode.
MINED_STORES = {"checks": CHECK_STORE,
                "spans": Store(MINED_DECODES_DIR, {"spans": PROTOCOLS["spans"]}, _spans)}


def grouped_by_shard(rows: Sequence[Mapping]) -> dict[str, list[dict]]:
    """The rows by ``shard_file``, each shard's rows in manifest row order."""
    by_shard: dict[str, list[dict]] = {}
    for row in rows:
        by_shard.setdefault(row["shard_file"], []).append(dict(row))
    for shard_rows in by_shard.values():
        shard_rows.sort(key=lambda row: row["row"])
    return by_shard


def decode_batch(rows: Sequence[Mapping], parts_dir: Path, *, role: str, decoders: dict,
                 stream, word_counts: Callable[[str], int], budget_seconds: float,
                 clock: Callable[[], float], share_features: bool,
                 load_audio: Callable[[bytes], object] | None = None,
                 clip_cache: Path | None = None,
                 stores: Mapping = MINED_STORES) -> BatchResult:
    """Decode ``rows``' pending clips, shard by shard, until every row holds every model.

    Each row goes through :func:`tadabur.tlog_eval_decode.run_shard`, which skips the models that
    already hold it, so a resumed batch decodes only what the parts do not. The budget is checked
    before each row; a started row is never cut off. A failing decode propagates and leaves the
    rows written so far in place.
    """
    started = clock()
    written = 0
    for shard_file, shard_rows in sorted(grouped_by_shard(rows).items()):
        held = _held(parts_dir, shard_file, role, decoders)
        for row in shard_rows:
            if item_key(row["audio_path"]) in held:
                continue
            if clock() - started >= budget_seconds:
                return BatchResult(done=False, written=written)
            ted.run_shard(shard_file, [row], parts_dir, decoders, stream=stream,
                          word_counts=word_counts, load_audio=load_audio, stores=stores,
                          clip_cache=clip_cache, share_features=share_features, role=role)
            written += 1
    return BatchResult(done=not pending(rows, parts_dir, role=role,
                                        models=tuple(decoders)), written=written)


def pending(rows: Sequence[Mapping], parts_dir: Path, *, role: str,
            models: Sequence[str] = MINED_MODELS) -> bool:
    """Whether any of ``rows`` lacks a record in one of ``models``' parts."""
    for shard_file, shard_rows in grouped_by_shard(rows).items():
        held = _held(parts_dir, shard_file, role, models)
        if any(item_key(row["audio_path"]) not in held for row in shard_rows):
            return True
    return False


def _held(parts_dir: Path, shard_file: str, role: str, models: Sequence[str]) -> set[str]:
    """The shard's keys every model holds: a row is done only when all of them hold it."""
    held = [resume_keys(ted.decodes_path(parts_dir, shard_file, model, role=role))
            for model in models]
    return set.intersection(*held) if held else set()
