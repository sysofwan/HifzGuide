"""Decode #245's comparison sets with our three models, from the sets' own sources.

The zipformer screen compares one external model against the base teacher, the round-2 candidate
(``r2_candidate_ema``) and C95 on sets those three had never been decoded on. This module is that
decode, in the shared loop (:func:`training.external_decode.decode_recordings`), with the round-2
spec's pinned identities, so every line carries the provenance and arms the readout reads:

* **benchmark** — the clips :func:`training.zipformer_run.decode_plan` allows, never a sealed one,
  each line recording the clip's ``outcome`` (``verdict``, ``trained`` or ``same_group``) beside
  its source and reference text. Rows come from the pinned fetch record, so the manifest and the
  plan are the ones the disjointness pass screened.
* **edits** — the 300 synthetic-edit items, each line recording the operation, mark and role the
  item's audited predicate reads, with the audio's SHA-256 checked against the manifest before it
  is decoded.

Zipformer itself is decoded by :mod:`training.zipformer_decode`; this module only ever loads our
models. qdat and Ship A are not re-decoded here: #156 and Ship A's committed decodes already
carry every arm the readout reads. Usage (on the GPU box, from a copy of ``tools/``)::

  flock /root/scratch/gpu.lock env OMP_NUM_THREADS=1 python -m training.zip_screen_decode \\
      --set benchmark --models base,c95,r2_candidate_ema \\
      --work-dir /root/scratch/zipformer/data --out-dir /root/scratch/zipformer/decodes

  flock /root/scratch/gpu.lock env OMP_NUM_THREADS=1 python -m training.zip_screen_decode \\
      --set edits --audio-dir /root/scratch/issue-117/stage/edits/audio \\
      --out-dir /root/scratch/zipformer/decodes
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterator, Sequence
from pathlib import Path

from tadabur.audio import decode_to_mono_16k
from training.aug_readout_gate import model_spec
from training.aug_readout_spec import CONSONANT2, RoundSpec
from training.external_decode import (
    Recording,
    append_timing,
    decode_recordings,
    load_decoders,
)
from training.truth_baseline import DECODE_STORE

BENCHMARK_SET = "benchmark"
EDITS_SET = "edits"
#: The three models this screen decodes: the teacher and the two candidates named in #245.
MODELS = ("base", "c95", "r2_candidate_ema")
#: The benchmark plan's outcomes, in the order a line records them.
OUTCOMES = ("verdict", "trained", "same_group")

#: What each set decodes, by store label (:func:`training.external_decode.decode_recordings`
#: takes a label per store it times): the benchmark and the edits are whole items (``spans``
#: beside ``stream_b0`` in one pass).
STORES = {BENCHMARK_SET: {"decode": DECODE_STORE}, EDITS_SET: {"decode": DECODE_STORE}}


def _clip_outcome(plan: dict) -> dict[str, str]:
    """Each decodable clip's outcome, from the plan's three exclusive lists."""
    outcome = {clip: "verdict" for clip in plan["verdict"]}
    outcome.update({clip: "trained" for clip in plan["trained"]})
    outcome.update({clip: "same_group" for clip in plan["same_group"]})
    missing = set(plan["decode"]) - set(outcome)
    if missing:
        raise ValueError(f"the plan decodes {len(missing)} clips it gives no outcome, e.g. "
                         f"{sorted(missing)[0]}")
    return outcome


def benchmark_recordings(work_dir: Path) -> Iterator[Recording]:
    """The plan's decodable benchmark clips, read from the pinned fetch's own copy.

    A sealed clip never reaches this: the plan drops it before any decode, and the plan is bound
    to the manifest's SHA-256 (:func:`training.zipformer_run.decode_plan`).
    """
    from tadabur.nonpro_sources import audio_sha256
    from training.zipformer_run import decode_plan, load_fetch, manifest_rows

    manifest = Path(work_dir) / load_fetch(work_dir)["manifest"]["path"]
    plan = decode_plan(manifest)
    outcome = _clip_outcome(plan)
    rows = {row["id"]: row for row in manifest_rows(manifest)}
    for clip in plan["decode"]:
        row = rows[clip]
        path = manifest.parent / row["audio"]

        def load(clip=clip, row=row, path=path):
            raw = path.read_bytes()
            samples = decode_to_mono_16k(raw)
            return samples, [{
                "key": clip, "audio_filename": clip, "audio_sha256": audio_sha256(raw),
                "source": row["source"], "reference_text": row["reference_text"],
                "outcome": outcome[clip], "start_sample": 0, "end_sample": len(samples),
            }]

        yield Recording((clip,), load)


def edit_recordings(audio_dir: Path) -> Iterator[Recording]:
    """The 300 rendered edit items, each checked against the manifest before it is decoded."""
    from tadabur.synthetic_edits import read_items
    from tadabur.truth_sites import audio_sha256

    for item in read_items():
        path = Path(audio_dir) / item["output"]["audio_filename"]
        expected = item["output"]["audio_sha256"]
        if not path.is_file():
            raise ValueError(f"{item['item_id']}: the edit audio {path} is not there")
        if (found := audio_sha256(path)) != expected:
            raise ValueError(f"{path} is {found}, not the manifest's {expected}")

        def load(item=item, path=path, expected=expected):
            samples = decode_to_mono_16k(path.read_bytes())
            return samples, [{
                "key": item["item_id"], "audio_filename": item["output"]["audio_filename"],
                "audio_sha256": expected, "operation": item["operation"],
                "mark": item["mark"], "role": item["role"], "label": item["label"],
                "start_sample": 0, "end_sample": len(samples),
            }]

        yield Recording((item["item_id"],), load)


def recordings(set_name: str, work_dir: Path | None,
               audio_dir: Path | None) -> Iterator[Recording]:
    """The set's recordings; each source's own directory is required, never defaulted."""
    if set_name == BENCHMARK_SET:
        if work_dir is None:
            raise SystemExit(f"{BENCHMARK_SET} reads the pinned fetch: pass --work-dir")
        return benchmark_recordings(work_dir)
    if audio_dir is None:
        raise SystemExit(f"{EDITS_SET} reads the rendered audio: pass --audio-dir")
    return edit_recordings(audio_dir)


def decodes_path(out_dir: Path, set_name: str, model: str) -> Path:
    return out_dir / f"{set_name}.{model}.jsonl"


def identity_path(out_dir: Path, set_name: str, model: str) -> Path:
    return out_dir / f"{set_name}.{model}.identity.json"


def run(set_name: str, models: Sequence[str], out_dir: Path, work_dir: Path | None,
        audio_dir: Path | None, limit: int | None = None,
        spec: RoundSpec = CONSONANT2) -> None:
    """Decode ``set_name`` for ``models``, appending to the files already there (resumable).

    ``limit`` decodes only the set's first items: the smoke run that proves the sources and the
    identities on the box before a multi-hour job is left alone.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decoders = load_decoders(lambda model: identity_path(out_dir, set_name, model),
                             {model: model_spec(model, spec=spec) for model in models})
    paths = {model: decodes_path(out_dir, set_name, model) for model in decoders}
    source = recordings(set_name, work_dir, audio_dir)
    timing = decode_recordings(
        source if limit is None else itertools.islice(source, limit), decoders, paths,
        f"{set_name} {','.join(models)}", STORES[set_name])
    append_timing(out_dir / f"{set_name}.timing.jsonl",
                  {"set": set_name, "models": list(models), **timing})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", required=True, choices=sorted(STORES))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, help="the benchmark fetch's work directory")
    parser.add_argument("--audio-dir", type=Path, help="the rendered edit audio")
    parser.add_argument("--limit", type=int, help="decode only the set's first items (smoke)")
    args = parser.parse_args()
    run(args.set, args.models.split(","), args.out_dir, args.work_dir, args.audio_dir,
        limit=args.limit)


if __name__ == "__main__":
    main()
