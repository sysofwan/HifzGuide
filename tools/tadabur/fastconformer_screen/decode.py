"""#267's decode driver: CPU only, with the owner's PAUSE and RAM floors at batch boundaries.

``--device cuda`` is refused: the GPU belongs to round 3 (#263), a GPU batch would have to run
under its non-blocking yield helper, and this screen never needed one — it decodes a few hours
of audio and its run note records 0 GPU-hours. The stops that matter on a shared box (PAUSE,
disk, RAM) are checked before every batch, and a stop leaves the plain decodes resumable.

:func:`preflight` is the refusal the run cannot skip: the set must be decodable, and the dated
erratum announcing the teacher's letter-level bars must be committed first. ``main`` takes its
plan from there, so there is one refusal path and no second one inside the loop.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from tadabur.fastconformer_screen.gates import ScreenStop
from tadabur.fastconformer_screen.sets import (ERRATUM_PATH, decode_plan, require_decodable,
                                               require_erratum)
from tadabur.fastconformer_screen.verdict import Rates
from tadabur.tlog_mining.give_way import Gates

#: CPU is the default and the only device this screen decodes on.
DEFAULT_DEVICE = "cpu"
BATCH_SIZE = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The run's command line: a set, a device (CPU; cuda is refused with a message) and
    an output directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=("ship-a", "edits", "qdat"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=DEFAULT_DEVICE)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--nemo", type=Path, help="the verified .nemo file")
    parser.add_argument("--dir", type=Path, help="the set's audio (Ship A clips, edit audio, "
                                                "or qdat's streaming cache)")
    parser.add_argument("--pause-file", type=Path,
                        default=Path("/root/scratch/tlog-mining/PAUSE"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def require_cpu(device: str) -> None:
    """Refuse a GPU decode: round 3 owns the GPU and this screen records 0 GPU-hours."""
    if device not in ("cpu", "cuda"):
        raise ScreenStop(f"unknown device {device!r}: only cpu and cuda are accepted, "
                         f"and only cpu is run")
    if device != DEFAULT_DEVICE:
        raise ScreenStop(
            f"--device {device}: this screen decodes on CPU only; the GPU belongs to round 3 "
            f"and any GPU batch would have to run under #263's yield helper (#267)")


def preflight(set_name: str, rates: Rates, *, erratum_path: Path = ERRATUM_PATH) -> list[str]:
    """Refuse a round-3 set or a missing/stale erratum, then return the set's decode plan."""
    require_decodable(set_name)
    require_erratum(rates, erratum_path)
    return decode_plan(set_name)


# --- the NeMo runtime: the model is loaded here, in the isolated environment -------------------

#: The committed decodes and their identities.
DECODES = Path(__file__).parent / "decodes"
MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
NEMO_FILE = "stt_ar_fastconformer_hybrid_large_pcd_v1.0.nemo"
#: The pinned file size, beside the SHA-256 in :mod:`tadabur.fastconformer_screen.gates`.
NEMO_SIZE = 459_233_280
SAMPLE_RATE = 16_000

NOTICE = """\
NVIDIA Arabic FastConformer, used as a detector only (#267).

Model: nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0
File:  stt_ar_fastconformer_hybrid_large_pcd_v1.0.nemo (459,233,280 bytes)
Revision: 7f32349d952f42a28dce979ba73270aa2bbdfa89
SHA-256: d29d19d7c054a5fc010ac6815e9cbb0dd1b21a30e0a7f7f2982e1fecaf0c3e31
Licence: CC-BY-4.0 (attribution only). Copyright NVIDIA Corporation.

The decodes in this directory are the model's CTC-greedy whole-item text. They are
evidence for one screen; the model is never a label, reference, teacher target or
training signal, and its weights are never fine-tuned or ensembled (ADR-0011 §3).
The model's training data (EveryAyah, MASC and others) is NVIDIA's; its provenance is
not redistributed here. No audio and no weights are committed.
"""


def sha256_file(path: Path) -> str:
    """A file's SHA-256, read in blocks (never the whole file in memory)."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_nemo(destination: Path) -> Path:
    """Download the pinned ``.nemo`` and refuse a changed size or hash."""
    from huggingface_hub import hf_hub_download

    from tadabur.fastconformer_screen.gates import NEMO_SHA256, REVISION

    path = Path(hf_hub_download(MODEL_ID, NEMO_FILE, revision=REVISION,
                                local_dir=str(destination)))
    if path.stat().st_size != NEMO_SIZE:
        raise ScreenStop(f"{path} is {path.stat().st_size} bytes, not the pinned {NEMO_SIZE}")
    found = sha256_file(path)
    if found != NEMO_SHA256:
        raise ScreenStop(f"{path} is {found}, not the pinned {NEMO_SHA256}")
    return path


def load_ctc_model(nemo_path: Path, device: str = DEFAULT_DEVICE):
    """The pinned model with the CTC head in greedy decoding, loaded in this environment."""
    from nemo.collections.asr.models import ASRModel

    from tadabur.fastconformer_screen.gates import REVISION, require_identity

    require_identity({"revision": REVISION, "nemo_sha256": sha256_file(nemo_path)})
    model = ASRModel.restore_from(str(nemo_path), map_location=device)
    model.change_decoding_strategy(decoder_type="ctc")
    model.eval()
    return model


def transcribe(model, samples) -> str:
    """The model's greedy CTC text for one whole item."""
    import numpy as np

    out = model.transcribe([np.asarray(samples, dtype=np.float32)], batch_size=1, verbose=False)
    first = out[0]
    while isinstance(first, (list, tuple)) and len(first) == 1:
        first = first[0]
    return first.text if hasattr(first, "text") else str(first)


def fingerprint(device: str) -> dict:
    """What decoded the text: the device, the decoder and the environment's versions."""
    from importlib.metadata import PackageNotFoundError, version

    def pinned(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {"device_type": device, "decoder": "ctc-greedy", "sample_rate": SAMPLE_RATE,
            "model": MODEL_ID, "nemo": pinned("nemo_toolkit"), "torch": pinned("torch"),
            "numpy": pinned("numpy")}


def _write_identity(out_dir: Path, set_name: str, nemo_path: Path, device: str) -> None:
    from tadabur.fastconformer_screen.gates import NEMO_SHA256, REVISION

    record = {"identity": {"revision": REVISION, "nemo_sha256": NEMO_SHA256,
                           "nemo_size": NEMO_SIZE, "file": nemo_path.name},
              "fingerprint": fingerprint(device)}
    (out_dir / f"{set_name}.fastconformer.identity.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def decode_set(set_name: str, *, plan: Sequence[str], directory: Path, out_dir: Path,
               nemo_path: Path, model, gates: Gates, device: str = DEFAULT_DEVICE,
               limit: int | None = None, batch_size: int = BATCH_SIZE) -> dict:
    """Decode every pending item of one set to whole-item CTC text, resuming by key.

    ``plan`` is :func:`tadabur.fastconformer_screen.sets.decode_plan`'s answer, and only its
    items are decoded: a G6 item or a round-3 input cannot reach the loop even when the source
    directory holds it. CPU only — the sets are a few hours of audio and the GPU belongs to
    round 3. The gates are checked before the next item is fetched, so a PAUSE or a low floor
    never triggers the shard read (or the model call) the next batch would need; the decodes
    written so far stay resumable.
    """
    import json
    import time

    from training.external_decode import resume_keys
    from training.zipformer_screen_decode import pending_items, recordings

    require_cpu(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_identity(out_dir, set_name, Path(nemo_path), device)
    # plain while the run is resumable; the launcher gzips it when the set is complete
    path = out_dir / f"{set_name}.fastconformer.jsonl"
    keep = set(plan)
    done = resume_keys(path) & keep
    items = audio_seconds = 0
    started = time.time()
    source = pending_items(recordings(set_name, Path(directory)), done, keep)
    with open(path, "a", encoding="utf-8") as out:
        try:
            while limit is None or items < limit:
                if items % batch_size == 0:
                    reason = gates.stop_reason()
                    if reason is not None:
                        # a stopped invocation still records what it decoded: the timing file
                        # is the set's provenance, so a stop must not be invisible in it
                        return _finish(out_dir, set_name, reason, items, audio_seconds,
                                       started, device)
                try:
                    record, span = next(source)
                except StopIteration:
                    break
                line = {**record, "ctc": transcribe(model, span)}
                out.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
                out.flush()
                items += 1
                audio_seconds += len(span) / SAMPLE_RATE
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()  # a stopped stream deletes the shard it was reading
    return _finish(out_dir, set_name, "done", items, audio_seconds, started, device)


def _finish(out_dir: Path, set_name: str, stopped: str, items: int, audio_seconds: float,
            started: float, device: str) -> dict:
    """Record what this invocation decoded, whether it finished or a gate stopped it.

    One line per invocation: their ``items`` sum to the committed set's lines, so a PAUSE or
    RAM stop is part of the set's provenance instead of a gap in it.
    """
    from training.external_decode import append_timing

    summary = {"stopped": stopped, "items": items,
               "audio_seconds": round(audio_seconds, 1),
               "wall_seconds": round(time.time() - started, 1), "device": device}
    append_timing(out_dir / f"{set_name}.fastconformer.timing.jsonl", summary)
    return summary


# --- the command line the launcher runs --------------------------------------------------------

#: The bars the erratum announced, as committed before any decode (see ``write_bars``).
BARS_PATH = Path(__file__).parent / "teacher_bars.json"


def read_bars(path: Path = BARS_PATH) -> Rates:
    """The pre-decode teacher bars, committed beside the erratum."""
    record = json.loads(path.read_text(encoding="utf-8"))
    return Rates(**record)


def main(argv: list[str] | None = None) -> None:
    """Decode one set with the pinned model, after the erratum's bars are checked."""
    from tadabur.fastconformer_screen.gates import available_ram_gb, free_disk_gb

    args = parse_args(argv)
    if args.set is None or args.out_dir is None or args.nemo is None or args.dir is None:
        raise SystemExit("--set, --out-dir, --nemo and --dir are required")
    plan = preflight(args.set, read_bars())  # the erratum is the run's first artifact
    require_cpu(args.device)
    gates = Gates(args.pause_file, free_disk_gb=lambda: free_disk_gb(args.out_dir),
                  available_ram_gb=available_ram_gb)
    started = time.time()
    if (reason := gates.stop_reason()) is not None:
        # the model is not restored and nothing is read: the stop is still part of the record
        args.out_dir.mkdir(parents=True, exist_ok=True)
        summary = _finish(args.out_dir, args.set, reason, 0, 0.0, started, args.device)
        print(json.dumps(summary), flush=True)
        raise SystemExit(3)
    model = load_ctc_model(args.nemo, args.device)
    summary = decode_set(args.set, plan=plan, directory=args.dir, out_dir=args.out_dir,
                         nemo_path=args.nemo, model=model, gates=gates,
                         device=args.device, limit=args.limit, batch_size=args.batch_size)
    print(json.dumps(summary), flush=True)
    if summary["stopped"] != "done":
        raise SystemExit(3)  # a gate stopped the run: the plain decodes stay resumable


if __name__ == "__main__":
    main()
