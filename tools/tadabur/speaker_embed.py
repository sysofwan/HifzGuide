"""The two pre-registered speaker-embedding candidates, at pinned Hub revisions (#186).

``docs/tlog-holdout-preregistration.md`` names the candidates and their order: (A)
``speechbrain/spkrec-ecapa-voxceleb``, then (B) ``microsoft/wavlm-base-plus-sv``. The revision
of each is pinned here, resolved on 2026-10-24 **before any embedding was computed**, and a
snapshot is fetched at that revision rather than whatever ``main`` holds later.

Every clip is embedded from its first 20 s at 16 kHz mono, zero-padded to
:data:`SPAN_SAMPLES` with the true length passed as the mask, in float32 weights. The encoder
returns un-normalised float64 rows; :func:`tadabur.speaker_cluster.normalise` L2-normalises
them where the procedure needs it.

The heavy imports (torch, transformers, speechbrain) happen inside :func:`encoder`, so the
procedural modules and their tests do not need speechbrain installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The embedding excerpt: the first 20 s at 16 kHz mono (the pre-registration's window).
SPAN_SECONDS = 20
SAMPLE_RATE = 16000
SPAN_SAMPLES = SPAN_SECONDS * SAMPLE_RATE
#: Clips per encoder call; a 20 s batch is ~20 MB of float32 and fits any card here.
BATCH = 16


@dataclass(frozen=True)
class Model:
    """One candidate: its key in the report, its Hub repo, its pinned revision and its width."""

    key: str
    repo_id: str
    revision: str
    dim: int


#: In the pre-registered order; S2 tries A first and only falls back to B if A fails.
MODELS = (
    Model("a", "speechbrain/spkrec-ecapa-voxceleb",
          "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286", 192),
    Model("b", "microsoft/wavlm-base-plus-sv",
          "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e", 512),
)


def _padded(waveforms) -> tuple:
    """The batch as float32 ``(n, SPAN_SAMPLES)`` plus each clip's true relative length."""
    import torch

    matrix = torch.zeros((len(waveforms), SPAN_SAMPLES), dtype=torch.float32)
    lengths = torch.ones(len(waveforms), dtype=torch.float32)
    for index, waveform in enumerate(waveforms):
        samples = np.asarray(waveform, dtype=np.float32)[:SPAN_SAMPLES]
        if samples.size == 0:
            raise ValueError(f"clip {index} is empty")
        matrix[index, :samples.size] = torch.from_numpy(np.ascontiguousarray(samples))
        lengths[index] = samples.size / SPAN_SAMPLES
    return matrix, lengths


def _chunks(waveforms, size: int):
    for start in range(0, len(waveforms), size):
        yield waveforms[start:start + size]


def _ecapa(model: Model, device: str, batch: int):
    """(A) SpeechBrain's ECAPA-TDNN speaker encoder, loaded from a revision-pinned snapshot."""
    import torch
    from huggingface_hub import snapshot_download
    from speechbrain.inference.speaker import EncoderClassifier

    directory = snapshot_download(model.repo_id, revision=model.revision)
    classifier = EncoderClassifier.from_hparams(source=directory,
                                                run_opts={"device": device})

    def encode(waveforms):
        outputs = []
        for chunk in _chunks(list(waveforms), batch):
            padded, lengths = _padded(chunk)
            with torch.no_grad():
                embeddings = classifier.encode_batch(padded.to(device), lengths.to(device))
            outputs.append(embeddings.reshape(len(chunk), -1).double().cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, model.dim))

    return encode


def _wavlm(model: Model, device: str, batch: int):
    """(B) Microsoft's WavLM x-vector model, revision-pinned through ``transformers``."""
    import torch
    from transformers import WavLMForXVector

    network = WavLMForXVector.from_pretrained(model.repo_id,
                                              revision=model.revision).to(device).eval()

    def encode(waveforms):
        outputs = []
        for chunk in _chunks(list(waveforms), batch):
            padded, lengths = _padded(chunk)
            mask = (torch.arange(SPAN_SAMPLES).unsqueeze(0)
                    < (lengths * SPAN_SAMPLES).round().long().unsqueeze(1)).long()
            with torch.no_grad():
                embeddings = network(input_values=padded.to(device),
                                     attention_mask=mask.to(device)).embeddings
            outputs.append(embeddings.reshape(len(chunk), -1).double().cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, model.dim))

    return encode


def encoder(model: Model, *, device: str = "cuda", batch: int = BATCH):
    """A callable from a list of 16 kHz mono float waveforms to ``(n, dim)`` float64 rows."""
    if model.key == "a":
        return _ecapa(model, device, batch)
    if model.key == "b":
        return _wavlm(model, device, batch)
    raise ValueError(f"unknown model {model.key!r}")
