"""Decode an external corpus with the base teacher and ``h448``, every arm (#144, #156), or with
other models named by :class:`ModelSpec` (the augmentation readout's arms, #95).

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


@dataclass(frozen=True)
class ModelSpec:
    """One model to decode with: its reference (a hub id or a distillation checkpoint), and for
    a checkpoint, which weights (``use_ema``: the averaged ones), the training step it must hold
    (``step``) and the augmentation recipe it was trained under (``recipe``: the version its
    config's ``augment`` names, or :data:`NO_AUGMENTATION` for none); ``None`` checks nothing.
    A checkpoint's live and averaged weights share one reference and so one fingerprint; the
    identity record tells them apart."""

    ref: str
    use_ema: bool = False
    step: int | None = None
    recipe: str | None = None

    def identity_extra(self) -> dict:
        """What the identity record adds to :func:`training.truth_baseline.model_identity`:
        nothing for the truth baseline's two models, so their committed records still match."""
        if not self.use_ema and self.step is None and self.recipe is None:
            return {}
        return {"weights": "ema" if self.use_ema else "live", "step": self.step,
                **({} if self.recipe is None else {"recipe": self.recipe})}


#: The truth baseline's two models, as every external corpus decodes them.
DEFAULT_SPECS = {name: ModelSpec(ref) for name, ref in DEFAULT_MODELS.items()}
#: The recipe of a run trained without augmentation (its config's ``augment`` is empty).
NO_AUGMENTATION = "none"


def checkpoint_facts(path: Path) -> tuple[int, str]:
    """The training step a distillation checkpoint was saved at, and the augmentation recipe its
    config records, read without its weights. Training augments whenever ``augment`` is
    non-empty, so only an absent or empty ``augment`` is :data:`NO_AUGMENTATION`; a non-empty one
    must name its recipe by a ``version`` string other than that reserved name, or the
    checkpoint is refused."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    augment = state["config"].get("augment")
    if not augment:
        return int(state["step"]), NO_AUGMENTATION
    version = augment.get("version") if isinstance(augment, dict) else None
    if not isinstance(version, str) or not version or version == NO_AUGMENTATION:
        raise ValueError(f"{path}: an augment config that names no recipe ({version!r}); "
                         "training augmented under it")
    return int(state["step"]), version


def load_decoders(identity_path: Callable[[str], Path],
                  models: Mapping[str, ModelSpec] = DEFAULT_SPECS) -> dict:
    """Each model of ``models`` (both truth-baseline models by default), checked against the
    truth baseline's frozen protocol, against the step and recipe its spec names (before its
    weights are loaded) and against the identity a previous run recorded at
    ``identity_path(model)``."""
    from training.decoding import Decoder

    decoders = {}
    for name, spec in models.items():
        if spec.step is not None or spec.recipe is not None:
            step, recipe = checkpoint_facts(Path(spec.ref))
            if spec.step is not None and step != spec.step:
                raise ValueError(f"{name}: {spec.ref} holds step {step}, not {spec.step}")
            if spec.recipe is not None and recipe != spec.recipe:
                raise ValueError(f"{name}: {spec.ref} was trained under {recipe!r}, "
                                 f"not {spec.recipe!r}")
        decoder = Decoder.load(spec.ref, DEVICE_TYPE, weights_dtype=WEIGHTS_DTYPE,
                               batch_size=BATCH_SIZE, use_ema=spec.use_ema)
        record = {
            "model": name,
            "identity": {**model_identity(spec.ref, decoder), **spec.identity_extra()},
            "fingerprints": {arm: decoder.fingerprint(mode).as_dict()
                             for arm, mode in MODES.items()},
        }
        for arm, mode in MODES.items():
            if record["fingerprints"][arm] != expected_fingerprint(spec.ref, mode):
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
