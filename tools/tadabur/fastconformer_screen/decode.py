"""#267's decode driver: CPU by default, and any GPU batch through #263's yield helper.

``decode_batches`` is torch- and NeMo-free: the model is a callable the caller supplies, so the
loop's gate behaviour is testable without the runtime. On a CPU (the default) the two stops
that still matter on a shared box, the owner's PAUSE file and the RAM floor, are honoured at
batch boundaries and the GPU lock is never touched. On a GPU a batch runs only under
``gpu.lock`` taken non-blockingly by :func:`tadabur.tlog_mining.give_way.run_give_way`, never
waited for; the driver is CPU by default because the sets are a few hours of audio.

:func:`preflight` is the refusal the run cannot skip: the set must be decodable, and the dated
erratum announcing the teacher's letter-level bars must be committed first.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from tadabur.fastconformer_screen.gates import ScreenStop
from tadabur.fastconformer_screen.sets import (ERRATUM_PATH, decode_plan, require_decodable,
                                               require_erratum)
from tadabur.fastconformer_screen.verdict import Rates
from tadabur.tlog_mining.give_way import Gates, GpuLock, run_give_way

#: CPU is the default: the sets are small and the GPU belongs to round 3 (#263).
DEFAULT_DEVICE = "cpu"
BATCH_SIZE = 8


@dataclass
class DecodeRun:
    """Where the loop stopped (``done``, ``pause``, ``disk`` or ``ram``) and what it decoded."""

    stopped: str
    lines: list[dict] = field(default_factory=list)
    batches: int = 0
    held_seconds: float = 0.0
    yielded_seconds: float = 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The run's command line: a set, a device (CPU unless asked) and an output directory."""
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


def decode_batches(items: Iterable, decode: Callable[[object], dict], *,
                   device: str, lock: GpuLock, gates: Gates, batch_size: int,
                   sleep: Callable[[float], None] = time.sleep) -> DecodeRun:
    """Decode the items in order, one batch at a time, stopping at a batch boundary.

    ``decode`` turns one item into its record. On the CPU the lock is never touched; on CUDA
    every batch runs under ``run_give_way``'s non-blocking lock and its gates.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError(f"unknown device {device!r}: only cpu and cuda are supported")
    items = list(items)
    run = DecodeRun("done")

    if device == "cpu":
        for start in range(0, len(items), batch_size):
            reason = gates.stop_reason()
            if reason is not None:
                run.stopped = reason
                return run
            run.lines.extend(decode(item) for item in items[start:start + batch_size])
            run.batches += 1
        return run

    position = 0

    def batch(_seconds: float) -> bool:
        nonlocal position
        run.lines.extend(decode(item) for item in items[position:position + batch_size])
        position += batch_size
        return position >= len(items)

    result = run_give_way(batch, lock=lock, gates=gates, sleep=sleep)
    run.stopped, run.batches = result.stopped, result.batches
    run.held_seconds, run.yielded_seconds = result.held_seconds, result.yielded_seconds
    return run


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
    import json

    from tadabur.fastconformer_screen.gates import NEMO_SHA256, REVISION

    record = {"identity": {"revision": REVISION, "nemo_sha256": NEMO_SHA256,
                           "nemo_size": NEMO_SIZE, "file": nemo_path.name},
              "fingerprint": fingerprint(device)}
    (out_dir / f"{set_name}.fastconformer.identity.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def decode_set(set_name: str, *, directory: Path, out_dir: Path, nemo_path: Path, model,
               gates: Gates, device: str = DEFAULT_DEVICE, limit: int | None = None,
               batch_size: int = BATCH_SIZE) -> dict:
    """Decode every pending item of one set to whole-item CTC text, resuming by key.

    CPU only: the sets are a few hours of audio and the GPU belongs to round 3. The gates stop
    at an item boundary; the decodes written so far stay resumable.
    """
    import json
    import time

    from training.external_decode import append_timing, resume_keys
    from training.zipformer_screen_decode import pending_items, recordings

    require_decodable(set_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_identity(out_dir, set_name, Path(nemo_path), device)
    # plain while the run is resumable; the launcher gzips it when the set is complete
    path = out_dir / f"{set_name}.fastconformer.jsonl"
    done = resume_keys(path)
    items = audio_seconds = 0
    started = time.time()
    source = pending_items(recordings(set_name, Path(directory)), done)
    with open(path, "a", encoding="utf-8") as out:
        try:
            for record, span in source:
                if items % batch_size == 0:
                    reason = gates.stop_reason()
                    if reason is not None:
                        return {"stopped": reason, "items": items,
                                "audio_seconds": round(audio_seconds, 1),
                                "wall_seconds": round(time.time() - started, 1)}
                line = {**record, "ctc": transcribe(model, span)}
                out.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
                out.flush()
                items += 1
                audio_seconds += len(span) / SAMPLE_RATE
                if limit is not None and items >= limit:
                    break
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()  # a stopped stream deletes the shard it was reading
    summary = {"stopped": "done", "items": items, "audio_seconds": round(audio_seconds, 1),
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


def _free_disk_gb(path: Path) -> float:
    import shutil

    return shutil.disk_usage(path).free / 1e9


def _available_ram_gb() -> float:
    """The box's available RAM, or a large constant when it cannot be read."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 1e6


def main(argv: list[str] | None = None) -> None:
    """Decode one set with the pinned model, after the erratum's bars are checked."""
    import json as _json

    args = parse_args(argv)
    if args.set is None or args.out_dir is None or args.nemo is None or args.dir is None:
        raise SystemExit("--set, --out-dir, --nemo and --dir are required")
    bars = read_bars()
    require_erratum(bars, ERRATUM_PATH)  # the erratum is the run's first artifact
    gates = Gates(args.pause_file, free_disk_gb=lambda: _free_disk_gb(args.out_dir),
                  available_ram_gb=_available_ram_gb)
    model = load_ctc_model(args.nemo, args.device)
    summary = decode_set(args.set, directory=args.dir, out_dir=args.out_dir,
                         nemo_path=args.nemo, model=model, gates=gates,
                         device=args.device, limit=args.limit, batch_size=args.batch_size)
    print(_json.dumps(summary), flush=True)
    if summary["stopped"] != "done":
        raise SystemExit(3)  # a gate stopped the run: the plain decodes stay resumable


if __name__ == "__main__":
    main()
