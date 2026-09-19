"""Stream the full Tadabur corpus for distillation without ever storing it.

``training.distill_data`` reads windows from clips already on disk, which caps the corpus
at whatever has been staged -- 61.8 hours, and ADR-0010 records that as the binding
constraint on the first full run: char accuracy plateaued at 84.58%, gaining 0.76 points
between steps 20k and 40k. The fix is more audio, and the whole corpus is 385 shards of
~2.4 GB, i.e. ~950 GB. No machine here has that.

It does not need to. Distillation reads each window **once**: the teacher generates its
target on the fly, so nothing has to be revisited and nothing has to be kept.
``tadabur.shard_reader.iter_shard_rows`` already fetches a shard, iterates it in small
Arrow batches, and with ``delete_after`` removes the blob when done. This module wraps that
as a torch ``IterableDataset``, so peak disk is **one shard per worker** (~2.5 GB) however
long the run is.

The arithmetic that makes it work, measured on this box:

* a shard downloads in **37 s** (2.47 GB at 66 MB/s);
* a shard holds ~1000 clips ≈ 3.4 h of audio ≈ 4,920 windows ≈ **154 steps** at batch 32;
* 154 steps at ~0.95 steps/s is **162 s** of training.

Download is ~4.4x faster than consumption **per worker**, so one stream stays ahead of the
GPU and the disk never grows. Note that ``distill_train`` defaults to 8 workers and each
fetches its own shard, so the real run has 8 concurrent downloads sharing one link and up
to ~20 GB of shards in flight -- the headroom above is per-stream, not aggregate. Over all 385 shards that is ~1,310 hours
and ~1.89M windows -- **one epoch is ~59,000 steps**, more than the 40,000 the staged-corpus
run used. Past one epoch the loader restarts with the same ``seed`` and replays the same
shard permutation, so ``--steps`` beyond ~59,000 does repeat windows.

**Train and validation are split by shard, not by clip.** The staged ``clips_v2`` corpus is
the filtered output of shards 0-19, so training on those shards would leak into the
existing validation clips and make the new number incomparable to the 84.58% baseline.
:data:`DEFAULT_TRAIN_SHARDS` therefore starts at 20, leaving the whole of 0-19 -- and hence
every staged validation clip -- unseen.

Usage::

    # What one shard yields, and how fast (downloads and deletes a single shard)
    python -m training.distill_stream --shards 200 --probe

    # Used by distill_train via --stream-shards
    python -m training.distill_train --preset h384 --stream-shards 20-384 \\
        --audio-root ../tadabur/audit_run/clips_v2 --out-dir runs/h384_full

Linux + CUDA (the class subclasses ``IterableDataset``, so torch is a hard import).
"""

from __future__ import annotations

import argparse
import random
import time

import torch

from training.distill_data import (
    FEATURE_DIM,
    FEATURE_FRAMES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    DEFAULT_HOP_SECONDS,
    DEFAULT_MIN_WINDOW_SECONDS,
    window_starts,
)

# Shards 0-19 produced the staged ``clips_v2`` corpus, so they are reserved: training on
# them would leak into the validation clips every reported number so far is measured on.
HELD_OUT_SHARDS = range(0, 20)
DEFAULT_TRAIN_SHARDS = "20-384"

# Windows buffered before yielding, to break up the strong correlation of a stream that
# arrives clip-by-clip and shard-by-shard: a pure stream would hand the optimiser ~5000
# consecutive windows from one shard's reciters.
#
# Sized against RAM, **per worker**, and the sizing was got wrong once already. A buffered
# window is a 5 s float32 waveform: 80,000 x 4 = **320 KB**, so 4096 of them is 1.3 GB per
# worker, and three workers died on this 24 GB box once each also held a 2.5 GB shard and
# pyarrow's batch buffers. 1024 is ~330 MB per worker.
#
# Note that buffering waveforms is the *more* expensive choice: extracted features are
# 250 x 160 x 4 = 160 KB, half as much. It is kept because the shuffle then reorders raw
# audio and extraction happens on the way out, which keeps the buffer independent of the
# feature extractor -- but if RAM ever binds again, buffering features is the cheaper fix.
DEFAULT_SHUFFLE_BUFFER = 1024


def decode_row_waveform(row: dict):
    """The 16 kHz mono waveform for one streamed Tadabur row."""
    from tadabur.audio import decode_to_mono_16k

    return decode_to_mono_16k(row["audio"]["bytes"])


def iter_row_windows(
    waveform,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    min_window_seconds: float = DEFAULT_MIN_WINDOW_SECONDS,
):
    """Yield every fixed-length window of one clip, zero-padded like the device pads."""
    import numpy as np

    hop_samples = int(SAMPLE_RATE * hop_seconds)
    min_samples = int(SAMPLE_RATE * min_window_seconds)

    for start in window_starts(len(waveform), WINDOW_SAMPLES, hop_samples, min_samples):
        chunk = waveform[start : start + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        yield chunk


class StreamingWindowDataset(torch.utils.data.IterableDataset):
    """Fixed ``(250, 160)`` feature windows streamed from Tadabur parquet shards.

    Must genuinely subclass ``IterableDataset``: ``DataLoader`` dispatches on
    ``isinstance``, not on the presence of ``__iter__``, and a duck-typed class is silently
    treated as map-style and asked for its ``len()``.

    Each DataLoader worker takes a disjoint slice of the shard list (``shards[worker::n]``)
    and fetches, consumes and **deletes** its shards one at a time, so peak disk is one
    shard per worker regardless of how many shards the run covers.

    Yields only input features, exactly like
    :class:`training.distill_data.DistillWindowDataset` -- the teacher runs online in the
    training step and produces targets from this same tensor.
    """

    def __init__(
        self,
        shard_indices: list[int],
        model_id: str = "obadx/muaalem-model-v3_2",
        hop_seconds: float = DEFAULT_HOP_SECONDS,
        shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
        seed: int = 1234,
        delete_after: bool = True,
    ) -> None:
        overlap = sorted(set(shard_indices) & set(HELD_OUT_SHARDS))
        if overlap:
            raise ValueError(
                f"shards {overlap} produced the staged validation clips; training on them "
                f"would leak. Use {DEFAULT_TRAIN_SHARDS} or another disjoint range."
            )
        self.shard_indices = list(shard_indices)
        self.model_id = model_id
        self.hop_seconds = hop_seconds
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.delete_after = delete_after
        self._extractor = None

    @property
    def extractor(self):
        if self._extractor is None:
            from transformers import SeamlessM4TFeatureExtractor

            self._extractor = SeamlessM4TFeatureExtractor.from_pretrained(self.model_id)
        return self._extractor

    def _my_shards(self) -> list[int]:
        """This worker's disjoint slice of the shard list, permuted per run."""
        shards = list(self.shard_indices)
        random.Random(self.seed).shuffle(shards)

        info = torch.utils.data.get_worker_info()
        if info is None:
            return shards
        return shards[info.id :: info.num_workers]

    def extract(self, waveform):
        """One window's features, pinned to the static ``(250, 160)`` ANE shape."""
        extracted = self.extractor(
            waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False
        )
        features = extracted.input_features[0]
        if features.shape[0] < FEATURE_FRAMES:
            features = torch.nn.functional.pad(
                features, (0, 0, 0, FEATURE_FRAMES - features.shape[0])
            )
        return features[:FEATURE_FRAMES]

    def _raw_windows(self):
        """Every window of every clip in this worker's shards, in stream order."""
        from tadabur.shard_reader import iter_shard_rows

        for row in iter_shard_rows(
            self._my_shards(), delete_after=self.delete_after
        ):
            try:
                waveform = decode_row_waveform(row)
            except Exception:
                # A corrupt row must not kill a 17-hour stream.
                continue
            yield from iter_row_windows(waveform, self.hop_seconds)

    def __iter__(self):
        """Shuffle-buffered feature windows.

        Waveforms are buffered rather than features. That is the **more** expensive choice
        -- a 5 s float32 waveform is 320 KB against a feature window's 160 KB, see
        :data:`DEFAULT_SHUFFLE_BUFFER` -- and is kept only so the shuffle stays independent
        of the feature extractor. If RAM binds again, buffering features is the cheaper fix.
        """
        rng = random.Random(self.seed + 1)
        buffer: list = []

        for waveform in self._raw_windows():
            buffer.append(waveform)
            if len(buffer) >= self.shuffle_buffer:
                index = rng.randrange(len(buffer))
                buffer[index], buffer[-1] = buffer[-1], buffer[index]
                yield self.extract(buffer.pop())

        rng.shuffle(buffer)
        for waveform in buffer:
            yield self.extract(waveform)


def probe(shard_index: int, hop_seconds: float = DEFAULT_HOP_SECONDS) -> dict:
    """Download one shard, measure what it yields, delete it. Bounded disk by construction."""
    from tadabur.shard_reader import iter_shard_rows

    started = time.time()
    clips = 0
    windows = 0
    seconds = 0.0

    for row in iter_shard_rows([shard_index], delete_after=True):
        try:
            waveform = decode_row_waveform(row)
        except Exception:
            continue
        clips += 1
        seconds += len(waveform) / SAMPLE_RATE
        windows += sum(1 for _ in iter_row_windows(waveform, hop_seconds))

    elapsed = time.time() - started
    return {
        "shard": shard_index,
        "clips": clips,
        "audio_hours": round(seconds / 3600, 2),
        "windows": windows,
        "steps_at_batch_32": windows // 32,
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Tadabur shards for distillation")
    parser.add_argument("--shards", default="200", help='shard spec, e.g. "20-384"')
    parser.add_argument("--hop-seconds", type=float, default=DEFAULT_HOP_SECONDS)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="download one shard, report what it yields, and delete it",
    )
    args = parser.parse_args()

    from tadabur.shard_reader import NUM_SHARDS, parse_shard_spec

    indices = parse_shard_spec(args.shards)

    if args.probe:
        report = probe(indices[0], args.hop_seconds)
        for key, value in report.items():
            print(f"  {key:<20} {value}")
        full = len(parse_shard_spec(DEFAULT_TRAIN_SHARDS))
        print(f"\nExtrapolated over {full} training shards ({DEFAULT_TRAIN_SHARDS}):")
        print(f"  audio_hours          {report['audio_hours'] * full:,.0f}")
        print(f"  windows              {report['windows'] * full:,}")
        print(f"  steps_at_batch_32    {report['steps_at_batch_32'] * full:,}")
        return

    print(f"{len(indices)} shards of {NUM_SHARDS}: {indices[:5]}...{indices[-3:]}")
    print(f"held out (never trained on): {HELD_OUT_SHARDS.start}-{HELD_OUT_SHARDS.stop - 1}")
    print(f"peak disk: one shard per worker (~2.5 GB), independent of shard count")


if __name__ == "__main__":
    main()
