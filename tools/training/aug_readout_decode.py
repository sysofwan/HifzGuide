"""Decode #95's readout sets with #94's arms on the GPU box, in the order the readout allows.

One run decodes one shard of one set (:data:`training.aug_readout.SETS`) with the models named,
through :mod:`training.external_decode`'s loop (the truth baseline's protocol and identity
checks, resumable, timed), reading each recording once for every model. Each line is the item's
provenance exactly as the corpus's own decoder writes it (so the readout checks it against the
committed base decode of the same item) and its decodes under the set's protocols:

* Tarteel v1's halves and RetaSy: each clip, its recorded ``checks``
  (:mod:`training.nonpro_decode`'s items, filtered to the half by :func:`tadabur.tarteel_split.half`);
* qdat: ``checks`` and ``stream_b0`` (qdat_bench reads the stream);
* QUD: the frozen salted half of its segments (:func:`training.aug_readout.load_qud_selection`),
  ``checks`` (:mod:`training.qud_decode`'s items);
* Tadabur dev, the clips ``h448`` holds no copy of: ``stream_b0`` and the blank count of the argmax
  frames the stream read (:func:`stream_with_blanks`), as #159's clean pass decoded them.

**The order** (:mod:`training.aug_readout_gate`): the look half only for the candidate's live and
EMA weights; every other set only from a checkout where ``choice.json``, ``look.json`` and the
look's claim are committed (a git clone, never an rsync'd copy), for both arms at the chosen
weights (and, on Tadabur, ``h448`` and the base teacher, for the blank share), the candidate's
checkpoint the one the choice read. An arm's checkpoint must hold #94's final step.

Decodes go to ``<out>/<set>.<shard>.<model>.jsonl``, each model's identity to
``<out>/<set>.<model>.identity.json``, the run's timing to ``<out>/<set>.<shard>.timing.jsonl``.

Usage (on the GPU box, from a copy of ``tools/``; ``--shard`` indexes the set's shards, see
``--list-shards``)::

  flock /root/scratch/gpu.lock env OMP_NUM_THREADS=1 python -m training.aug_readout_decode \\
      --set tarteel_v1.look --shard 0 --models candidate_live,candidate_ema \\
      --out-dir /root/scratch/issue-95/decodes --cache-dir /root/scratch/issue-95/hf_cache
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np

from tadabur.tarteel_split import half
from training.aug_readout import load_qud_selection
from training.aug_readout_spec import AUG95, SPECS, RoundSpec
from training.aug_readout_spec import spec as round_spec
from training.aug_readout_gate import (
    BLANKS,
    QDAT_SET,
    QUD_OTHER_SET,
    QUD_SET,
    RETASY_SET,
    SETS,
    STREAM,
    TADABUR_SET,
    TARTEEL_CONFIRM,
    TARTEEL_LOOK,
    Layout,
    allowed_models,
    committed_choice,
    model_spec,
)
from training.external_decode import Recording, append_timing, decode_recordings, load_decoders
from training.truth_baseline import CHECK_STORE, CHECKS, DECODE_STORE, PROTOCOLS, Store


def stream_with_blanks(decoder, span: np.ndarray) -> dict:
    """The deployed b = 0 stream over ``span`` (tail flushed), exactly
    :meth:`training.decoding.Decoder.decode_stream`, and ``[blank, frames]`` over every argmax
    frame of the windows it read."""
    from training.decoding import BLANK_ID, stream_emissions, tokens_to_phonemes, window_audio

    rows = decoder.window_rows(window_audio(span))
    text = tokens_to_phonemes(e.token_id for e in stream_emissions(rows, 0, True))
    return {STREAM: text, BLANKS: [int(sum(int((r == BLANK_ID).sum()) for r in rows)),
                                   int(sum(int(r.size) for r in rows))]}


#: ``stream_b0`` and its blank count, under the stream's own fingerprint.
STREAM_STORE = Store(DECODE_STORE.directory, {STREAM: PROTOCOLS[STREAM]}, stream_with_blanks)
STORES = {CHECKS: CHECK_STORE, STREAM: STREAM_STORE}


def shards(set_name: str) -> list[str]:
    """The shards a set's runs index: a dataset's parquet files, QUD's ``<mushaf>:<shard>``,
    Tadabur's one directory of clips."""
    from tadabur.nonpro_sources import QDAT, RETASY, TARTEEL_V1
    from tadabur.qud_overlap import PILOT
    from training.qud_decode import mushaf_shards

    if set_name in (TARTEEL_LOOK, TARTEEL_CONFIRM):
        return list(TARTEEL_V1.shards)
    if set_name == QDAT_SET:
        return list(QDAT.shards)
    if set_name == RETASY_SET:
        return list(RETASY.shards)
    if set_name in (QUD_SET, QUD_OTHER_SET):
        return [f"{slug}:{shard}" for slug in PILOT for shard in mushaf_shards(slug)]
    return ["clips"]


def kept(recordings: Iterator[Recording], keep: Callable[[str], bool]) -> Iterator[Recording]:
    """Only the items ``keep`` names, of recordings that hold any (the rest are never read)."""
    for recording in recordings:
        keys = tuple(k for k in recording.keys if keep(k))
        if not keys:
            continue

        def load(recording=recording, keys=frozenset(keys)):
            samples, records = recording.load()
            return samples, [r for r in records if r["key"] in keys]

        yield Recording(keys, load)


def _clip_of(key: str) -> str:
    return key.rsplit("#", 1)[0]


def recordings(set_name: str, shard: str, cache_dir: Path, clips_dir: Path | None
               ) -> Iterator[Recording]:
    """One shard's recordings of a set, each holding only the set's items."""
    from tadabur.nonpro_sources import QDAT, RETASY, TARTEEL_V1
    from training import nonpro_decode, qud_decode

    if set_name in (TARTEEL_LOOK, TARTEEL_CONFIRM):
        which = set_name.split(".")[1]
        return kept(nonpro_decode.recordings(TARTEEL_V1, shard, cache_dir),
                    lambda key: half(_clip_of(key)) == which)
    if set_name in (QDAT_SET, RETASY_SET):
        return nonpro_decode.recordings(QDAT if set_name == QDAT_SET else RETASY, shard,
                                        cache_dir)
    if set_name in (QUD_SET, QUD_OTHER_SET):
        from training.consonant_readout import load_qud_complement

        slug, parquet = shard.split(":", 1)
        selection = (load_qud_selection() if set_name == QUD_SET
                     else load_qud_complement())
        return kept(qud_decode.recordings(slug, parquet, cache_dir), selection.__contains__)
    if clips_dir is None:
        raise SystemExit(f"{TADABUR_SET} reads the decode_evalset's clips: pass --clips-dir")
    return tadabur_recordings(clips_dir)


def tadabur_recordings(clips_dir: Path) -> Iterator[Recording]:
    """Each dev clip ``h448`` holds no copy of, clean, recorded as #159's clean pass records it."""
    from training.decode_evalset import read_clip_audio
    from training.nonpro_gap import committed_grid, dev_clips

    seen = set(committed_grid()["sample"]["h448_seen"])
    for clip in dev_clips():
        if clip.filename in seen:
            continue

        def load(clip=clip):
            clean = read_clip_audio(clips_dir, clip.filename)
            return clean, [{"key": clip.filename, "reciter_id": clip.reciter_id,
                            "clean_sha256": hashlib.sha256(clean.tobytes()).hexdigest(),
                            "condition": "clean", "start_sample": 0, "end_sample": len(clean)}]

        yield Recording((clip.filename,), load)


def check_models(set_name: str, models: Sequence[str], layout: Layout | None = None,
                 checkpoint_sha256: Callable[[str], str] | None = None,
                 spec: RoundSpec = AUG95) -> None:
    """Refuse a model the readout's order does not allow on this set now, before any recording
    is read: a gated set needs the round's committed choice (:func:`committed_choice`: tracked
    by git, unchanged from ``HEAD``, bound to the committed look), its weights only, and the
    candidate's checkpoint the one the choice read and the round pinned. ``layout`` defaults to
    the round's own root, never another round's."""
    layout = layout if layout is not None else Layout(spec.root)
    if not SETS[set_name].gated:
        allowed = allowed_models(set_name, None, spec)
        choice = None
    else:
        choice = committed_choice(layout, spec=spec).choice
        allowed = allowed_models(set_name, choice, spec)
    if refused := [m for m in models if m not in allowed]:
        raise SystemExit(f"{set_name} may be decoded now only for {list(allowed)}, not {refused}")
    if choice is None:
        return
    if checkpoint_sha256 is None:
        from training.truth_baseline import file_sha256 as checkpoint_sha256
    for model in models:
        if model in choice["identities"]:
            found = checkpoint_sha256(model_spec(model, spec=spec).ref)
            if found != choice["identities"][model]["checkpoint_sha256"]:
                raise SystemExit(f"{model}: the checkpoint is not the one the choice read")


def run(set_name: str, shard_index: int, models: Sequence[str], out_dir: Path, cache_dir: Path,
        clips_dir: Path | None, spec: RoundSpec = AUG95) -> None:
    check_models(set_name, models, spec=spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = shards(set_name)[shard_index]
    decoders = load_decoders(lambda model: out_dir / f"{set_name}.{model}.identity.json",
                             {m: model_spec(m, spec=spec) for m in models})
    paths = {m: out_dir / f"{set_name}.{shard_index}.{m}.jsonl" for m in models}
    stores = {p: STORES[p] for p in SETS[set_name].protocols}
    timing = decode_recordings(recordings(set_name, shard, cache_dir, clips_dir), decoders, paths,
                               f"{set_name} {shard}", stores)
    append_timing(out_dir / f"{set_name}.{shard_index}.timing.jsonl",
                  {"set": set_name, "shard": shard, **timing})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", required=True, choices=sorted(SETS))
    parser.add_argument("--list-shards", action="store_true", help="print the set's shards")
    parser.add_argument("--shard", type=int, help="index into the set's shards")
    parser.add_argument("--models", help="comma-separated, e.g. candidate_live,candidate_ema")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--clips-dir", type=Path, help="the decode_evalset's clips (Tadabur)")
    parser.add_argument("--spec", default=AUG95.name, choices=sorted(SPECS),
                        help="the round the models and the choice belong to")
    args = parser.parse_args()
    if args.list_shards:
        print("\n".join(f"{i} {s}" for i, s in enumerate(shards(args.set))))
        return
    if None in (args.shard, args.models, args.out_dir, args.cache_dir):
        parser.error("--shard, --models, --out-dir and --cache-dir are required")
    run(args.set, args.shard, args.models.split(","), args.out_dir, args.cache_dir,
        args.clips_dir, spec=round_spec(args.spec))


if __name__ == "__main__":
    main()
