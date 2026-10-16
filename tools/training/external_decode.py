"""Decode an external corpus with the base teacher and ``h448``, every arm (#144, #156).

An **external corpus** is audio from outside Tadabur that no model has heard (QUD's
professional mushafs, #144; the non-professional app-user corpora, #156). Its items are spans
of its recordings; each is decoded by both models exactly as the truth baseline decodes its
items (:mod:`training.truth_baseline`: bf16 weights, batch size 1, CUDA), so every corpus is
comparable with it and with the others:

* ``spans``: the item decoded whole;
* ``stream_b0``: the deployed b = 0 stream over the item, tail flushed;
* ``checks``: the item's recorded check sequence (:meth:`training.decoding.Decoder.app_checks`),
  what Muraja's transcriber would hand its engine, for the Muraja harness.

Each model's identity and fingerprints are checked against the truth baseline's frozen ones
(:func:`training.truth_baseline.expected_fingerprint`, :func:`~training.truth_baseline.model_identity`)
and written once per corpus part beside its decodes; a later run must find the same. Decodes are
appended one item per line as they are made, so a run resumes where it stopped (an incomplete
final record is cut off first); each line is the item's provenance record (its audio's checksum
and its sample span among it) followed by its three decodes. The run's wall time per model and
arm is appended to a timing file. The work is CPU-bound (one feature extraction and one batch-1
forward per check), so several runs share the GPU, each with few CPU threads.

A corpus hands this loop its :class:`Recording` s; the loop owns resuming, identity, decoding,
writing and timing (:func:`decode_recordings`).
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tadabur.audio import TARGET_SAMPLE_RATE
from training.truth_baseline import (
    BATCH_SIZE,
    CHECK_PROTOCOLS,
    CHECK_STORE,
    CHECKS,
    DECODE_STORE,
    DEFAULT_MODELS,
    DEVICE_TYPE,
    PROTOCOLS,
    WEIGHTS_DTYPE,
    Store,
    expected_fingerprint,
    model_identity,
)

MODES = {**PROTOCOLS, **CHECK_PROTOCOLS}
#: The truth baseline's two stores' decode functions: ``spans`` and ``stream_b0`` together,
#: then the check sequence; each is timed on its own.
STORES = {"spans+stream_b0": DECODE_STORE, CHECKS: CHECK_STORE}
#: Items between progress lines.
PROGRESS_EVERY = 200


@dataclass(frozen=True)
class Recording:
    """One recording of a corpus: the keys of the items it holds, and how to read it: its 16 kHz
    mono samples and one provenance record per key (``key``, ``start_sample``, ``end_sample``
    and whatever the corpus records), read only when an item is still to be decoded."""

    keys: tuple[str, ...]
    load: Callable[[], tuple[np.ndarray, list[dict]]]


def resume_keys(path: Path) -> set[str]:
    """The items a decodes file already holds, after cutting off an incomplete final record
    (a run killed mid-write), so the next record appended starts on a line of its own. A
    malformed record anywhere else is corruption, not an interruption, and is refused."""
    if not path.exists():
        return set()
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
        with open(path, "r+b") as f:
            f.truncate(len(data))
    keys = set()
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        try:
            keys.add(json.loads(line)["key"])
        except (json.JSONDecodeError, KeyError) as error:
            raise ValueError(f"{path}:{number} is not a decode record") from error
    return keys


def load_decoders(identity_path: Callable[[str], Path]) -> dict:
    """Both models, checked against the truth baseline's frozen protocol and against the
    identity a previous run recorded at ``identity_path(model)``."""
    from training.decoding import Decoder

    decoders = {}
    for name, ref in DEFAULT_MODELS.items():
        decoder = Decoder.load(ref, DEVICE_TYPE, weights_dtype=WEIGHTS_DTYPE,
                               batch_size=BATCH_SIZE)
        record = {
            "model": name,
            "identity": model_identity(ref, decoder),
            "fingerprints": {arm: decoder.fingerprint(mode).as_dict()
                             for arm, mode in MODES.items()},
        }
        for arm, mode in MODES.items():
            if record["fingerprints"][arm] != expected_fingerprint(ref, mode):
                raise ValueError(f"{name} decodes {arm} as {record['fingerprints'][arm]}")
        path = identity_path(name)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != record:
                raise ValueError(f"{path} records other weights or settings than {name} has now")
        else:  # runs go side by side: each writes through its own temporary file
            temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, path)
        decoders[name] = decoder
    return decoders


def decode_recordings(recordings: Iterable[Recording], decoders: dict,
                      paths: dict[str, Path], label: str,
                      stores: Mapping[str, Store] = STORES) -> dict:
    """Decode every item of ``recordings`` not yet in its model's file (``paths`` by model)
    into each of ``stores`` (every arm by default), appending each item as one line; return
    the run's timing (items, audio and wall seconds, and decode seconds per model and store).
    ``label`` names the run in progress lines."""
    done = {name: resume_keys(path) for name, path in paths.items()}
    seconds: dict[str, dict[str, float]] = {name: defaultdict(float) for name in decoders}
    audio_seconds = 0.0
    items = 0
    started = time.time()
    files = {name: open(path, "a", encoding="utf-8") for name, path in paths.items()}
    try:
        for recording in recordings:
            if all(key in done[name] for key in recording.keys for name in decoders):
                continue
            samples, records = recording.load()
            for record in records:
                span = samples[record["start_sample"]:record["end_sample"]]
                audio_seconds += len(span) / TARGET_SAMPLE_RATE
                items += 1
                for name, decoder in decoders.items():
                    if record["key"] in done[name]:
                        continue
                    line = dict(record)
                    for store_label, store in stores.items():
                        tick = time.time()
                        line.update(store.decode(decoder, span))
                        seconds[name][store_label] += time.time() - tick
                    files[name].write(json.dumps(line, ensure_ascii=False,
                                                 separators=(",", ":")) + "\n")
                    files[name].flush()
                if items % PROGRESS_EVERY == 0:
                    print(f"{label}: {items} items, {audio_seconds:.0f} s of audio in "
                          f"{time.time() - started:.0f} s", flush=True)
    finally:
        for handle in files.values():
            handle.close()
    return {"items": items, "audio_seconds": round(audio_seconds, 1),
            "wall_seconds": round(time.time() - started, 1),
            "decode_seconds": {n: {a: round(s, 1) for a, s in arms.items()}
                               for n, arms in seconds.items()}}


def append_timing(path: Path, timing: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(timing) + "\n")
    print(json.dumps(timing), flush=True)
