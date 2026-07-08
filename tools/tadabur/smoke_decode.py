"""Walking-skeleton smoke test: decode one streamed Tadabur clip end-to-end.

Streams a single clip from ``FaisaI/tadabur`` (never materializing the 1400 h),
resamples it to 16 kHz mono, loads Muaalem in bf16 on the CUDA GPU, runs one
variable-length forward pass, and greedy-CTC-decodes the phoneme head to a sanity
phoneme string. Records the model's VRAM footprint. This proves the PyTorch
inference->decode path (``tadabur.inference``) works before the filter is built on
top of it (issue #2 / PRD #1 Phase 0).

Usage:
  python -m tadabur.smoke_decode [--index N] [--model-id ID] [--device cuda]
"""

from __future__ import annotations

import argparse
import itertools

import torch
from datasets import Audio, load_dataset

from .audio import TARGET_SAMPLE_RATE, decode_to_mono_16k
from .inference import MODEL_ID, MuaalemPhonemeModel

DATASET_ID = "FaisaI/tadabur"
AUDIO_COLUMN = "audio"


def stream_clip(
    dataset_id: str, config_name: str | None, split: str, index: int
) -> dict:
    """Stream ``dataset_id`` and return the ``index``-th clip with undecoded audio.

    Reads the audio feature with ``decode=False`` — this streams only the raw WAV
    bytes (decoded later by ``tadabur.audio``) and avoids ``datasets``' ``torchcodec``
    audio-decoder dependency. Nothing is materialized to disk. Note the default
    config's shards are a single ~2.4 GB / 1000-row Parquet row group, so yielding
    even one clip pulls that whole row group; the small-row-group ``preview`` config
    (``--config-name preview``) is far faster for a smoke test.
    """
    dataset = load_dataset(
        dataset_id, name=config_name, split=split, streaming=True
    )
    dataset = dataset.cast_column(AUDIO_COLUMN, Audio(decode=False))
    clip = next(itertools.islice(iter(dataset), index, index + 1), None)
    if clip is None:
        raise IndexError(f"{dataset_id}[{split}] has no clip at index {index}.")
    return clip


def _mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID, help="HF dataset id.")
    parser.add_argument(
        "--config-name",
        default=None,
        help="Dataset config name (e.g. 'preview' for fast small-row-group streaming).",
    )
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument(
        "--index", type=int, default=0, help="Which streamed clip to decode."
    )
    parser.add_argument("--model-id", default=MODEL_ID, help="HF model id.")
    parser.add_argument(
        "--device", default="cuda", help="Torch device (default: cuda)."
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    print(f"Streaming {args.dataset}[{args.split}] clip #{args.index} ...")
    clip = stream_clip(args.dataset, args.config_name, args.split, args.index)
    waveform = decode_to_mono_16k(clip[AUDIO_COLUMN]["bytes"])
    duration_s = len(waveform) / TARGET_SAMPLE_RATE
    print(
        f"  reciter={clip.get('reciter_id')} "
        f"surah:ayah={clip.get('surah_id')}:{clip.get('ayah_id')} "
        f"duration={duration_s:.2f}s samples={len(waveform)} "
        f"sr={TARGET_SAMPLE_RATE}"
    )

    print(f"Loading {args.model_id} (bf16) on {device} ...")
    model = MuaalemPhonemeModel.load(args.model_id, device=device)
    if is_cuda:
        print(f"  backbone VRAM allocated: {_mib(torch.cuda.memory_allocated(device)):.1f} MiB")

    result = model.decode(waveform, TARGET_SAMPLE_RATE)
    if is_cuda:
        print(f"  peak VRAM after forward: {_mib(torch.cuda.max_memory_allocated(device)):.1f} MiB")

    print(
        f"Decoded {result.num_logit_frames} phoneme-head frames "
        f"(from {result.num_feature_frames} feature frames, variable-length):"
    )
    print(f"  reference (uthmani): {clip.get('text_ar_uthmani')}")
    print(f"  decoded phonemes   : {result.phonemes}")


if __name__ == "__main__":
    main()
