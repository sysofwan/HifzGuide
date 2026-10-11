"""The shipped student's streaming decode of the mining pool: ``h448`` at b=0 (#87).

The acceptance rules' comparator for today's system is ``h448`` decoded with the deployed
streaming protocol at commit block ``b=0`` with no blank bias (§3). The listening session
(:mod:`tadabur.listening_session`) stratifies its tashkeel sites by that decode as well as
by the base teacher's, so the sites the shipped model leaves empty are drawn on purpose
rather than by luck.

Each pool clip is decoded **whole**, as the device hears it: the staged 16 kHz clip goes
through :meth:`training.decoding.Decoder.decode_stream`'s protocol (5 s windows, 1 s hop,
block 0 committed, the last window flushed), never the UI's excerpts. The committed cache
then keeps, per kept pool segment, the committed tokens whose **commit time** falls inside
the segment's span widened by :data:`SEGMENT_MARGIN_S` either side (:func:`segment_decodes`):
the segment's references are matched against that text by local alignment, which trims the
margin. A token's commit time is its window's start plus the centre of its run of
timesteps.

The cache records the decode fingerprint (model, ``stream_protocol(0)``, weights dtype,
batch size, device, autocast, policy) and the checkpoint file's SHA-256.

Usage (on the GPU box, from ``tools/``; under the shared GPU lock)::

  flock /root/scratch/gpu.lock python -m tadabur.pool_stream \\
      --checkpoint runs/h448_stream/checkpoint.pt --audio-dir stage/clips
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from training.decoding import DEPLOYED_BLOCK, HOP_SAMPLES, Emission, tokens_to_phonemes
from training.distill_data import SAMPLE_RATE
from training.distill_student import DEPLOYED_LOGIT_FRAMES

from .mining_pool import MINING_POOL_DIR, PoolClip, load_manifest, segment_key

H448_DECODES_PATH = MINING_POOL_DIR / "h448_stream_decodes.json"

#: Seconds either side of a segment whose committed tokens are kept with it.
SEGMENT_MARGIN_S = 0.5
#: One CTC timestep of a 5 s window.
STEP_S = 5.0 / DEPLOYED_LOGIT_FRAMES


def commit_time(emission: Emission) -> float:
    """Seconds into the clip at which ``emission``'s run of timesteps is centred."""
    return emission.window * HOP_SAMPLES / SAMPLE_RATE + (emission.midpoint + 0.5) * STEP_S


def segment_decodes(
    emissions: Iterable[Emission], spans: Mapping[str, tuple[int, int]]
) -> dict[str, str]:
    """For each ``key -> (start_sample, end_sample)``, the committed text whose commit time
    lies in the span widened by :data:`SEGMENT_MARGIN_S` on both sides."""
    timed = [(commit_time(e), e.token_id) for e in emissions]
    decodes = {}
    for key, (start, end) in spans.items():
        low = start / SAMPLE_RATE - SEGMENT_MARGIN_S
        high = end / SAMPLE_RATE + SEGMENT_MARGIN_S
        decodes[key] = tokens_to_phonemes(token for t, token in timed if low <= t < high)
    return decodes


def kept_spans(clip: PoolClip) -> dict[str, tuple[int, int]]:
    return {
        segment_key(clip.audio_filename, seg.segment_index): (seg.start_sample, seg.end_sample)
        for seg in clip.segments
        if seg.kept
    }


def load_stream_decodes(path: Path = H448_DECODES_PATH) -> tuple[dict, dict[str, str]]:
    """The cache's provenance (fingerprint, checkpoint checksum, cut rule) and its decodes
    keyed by :func:`tadabur.mining_pool.segment_key`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k != "decodes"}, data["decodes"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_record(
    clips: list[PoolClip], registry: Mapping, checkpoint: Path, audio_dir: Path,
    device: str = "cuda", batch_size: int = 16,
) -> dict:
    """``h448``'s committed stream for every kept segment of ``clips`` (each clip decoded
    whole from its verified staged file), with the decode fingerprint and checkpoint
    checksum: the cache :func:`load_stream_decodes` reads."""
    from training.decoding import Decoder, stream_protocol

    from .audio import read_audio
    from .staged_audio import verify_staged

    clips = [c for c in clips if kept_spans(c)]
    # The student's historical numerics: fp32 weights under the CUDA bf16 autocast.
    decoder = Decoder.load(checkpoint, device, weights_dtype="fp32", batch_size=batch_size)
    decodes: dict[str, str] = {}
    for number, clip in enumerate(clips, 1):
        verify_staged(registry[clip.audio_filename], audio_dir)
        samples, rate = read_audio(audio_dir / clip.audio_filename, dtype="float32")
        if rate != SAMPLE_RATE:
            raise ValueError(f"{clip.audio_filename} is {rate} Hz, not {SAMPLE_RATE}")
        decodes.update(segment_decodes(decoder.emissions(samples, DEPLOYED_BLOCK),
                                       kept_spans(clip)))
        if number % 250 == 0:
            print(f"  {number}/{len(clips)} clips", flush=True)
    return {
        "decode_fingerprint": decoder.fingerprint(stream_protocol(DEPLOYED_BLOCK)).as_dict(),
        "checkpoint_sha256": _sha256(checkpoint),
        "segment_margin_s": SEGMENT_MARGIN_S,
        "decodes": dict(sorted(decodes.items())),
    }


def write_stream_record(record: dict, path: Path) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False, indent=0, sort_keys=True) + "\n",
                    encoding="utf-8")


def main() -> None:
    from .staged_audio import load_staged_clips

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="the h448 distillation checkpoint (live weights)")
    parser.add_argument("--audio-dir", type=Path, required=True, help="staged pool clips")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=H448_DECODES_PATH)
    args = parser.parse_args()

    registry = load_staged_clips()
    record = stream_record(load_manifest(registry=registry), registry, args.checkpoint,
                           args.audio_dir, args.device, args.batch_size)
    write_stream_record(record, args.out)
    print(f"{len(record['decodes'])} segments -> {args.out}")


if __name__ == "__main__":
    main()
