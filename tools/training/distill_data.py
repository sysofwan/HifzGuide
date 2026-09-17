"""Fixed-window recitation audio for size distillation: discovery, splitting, features.

Size distillation needs no labels -- the teacher supplies every target -- so this module
takes plain 16 kHz mono recitation audio and turns it into the exact tensor the deployed
model eats: ``(250, 160)`` per window, one window per training example.

Three things are pinned here because getting any of them wrong silently trains the student
on a distribution the deployed pipeline never produces.

**Windows are cut before features are extracted, never after.** ``SeamlessM4TFeatureExtractor``
normalizes each utterance to zero mean and unit variance *per mel bin* over whatever it is
given. Extracting a whole clip and slicing it would normalize over the clip; the device
normalizes over the 5 s window (``MuaalemInference.prepareFeatures``, called with
``windowAudio``). Those produce different numbers, so this module slices the **waveform**
into windows and feature-extracts each window on its own -- the same reasoning
``training.waqf_distill`` applies to its VAD teacher.

**Short windows are kept and zero-padded, not dropped.** The device pads too: Muraja's
~5 previews per audio-second run on a *partially filled* buffer, padded to the static 250
frames, and they are 5 of the ~6 inferences per second. A student that only ever saw full
windows would be trained off-distribution for most of its actual calls. :func:`window_starts`
therefore emits a final short window whenever it carries at least ``min_window_seconds`` of
real audio, and :class:`DistillWindowDataset` zero-pads it exactly as the device does.

**The split is by clip, not by window.** Windows from one recitation overlap and share a
reciter; splitting by window would put near-duplicates on both sides and make validation
agreement meaningless. :func:`split_clips` hashes the clip *filename* so the assignment is
stable across runs, machines, and any re-ordering of the directory walk.

Usage::

    # Inventory what a corpus root would yield, without extracting anything
    python -m training.distill_data --audio-root ../tadabur/audit_run/clips_v2

    # Same, as JSON, with a different training stride
    python -m training.distill_data --audio-root <dir> --hop-seconds 1.25 --json

Runs anywhere for the inventory path (needs only ``soundfile``); the dataset class needs
torch and ``transformers``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# The frozen deployed contract. A window is 5 s of 16 kHz audio -> 250 stride-2 feature
# frames -> 125 CTC timesteps. Mirrors ``training.distill_student``; duplicated as
# literals so this module does not drag in the student architecture just to window audio.
SAMPLE_RATE = 16000
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
FEATURE_FRAMES = 250
FEATURE_DIM = 160

# Training stride between window starts. Unrelated to the device's 1 s hop: that hop exists
# to bound confirmation latency, whereas this one only controls how much the training
# windows overlap, i.e. how many examples a fixed corpus yields. 2.5 s (50% overlap) roughly
# doubles the example count while keeping successive windows meaningfully different.
DEFAULT_HOP_SECONDS = 2.5

# A trailing window shorter than this carries too little signal to be worth a teacher pass.
# Well below the window length on purpose -- short padded windows are in-distribution.
DEFAULT_MIN_WINDOW_SECONDS = 1.0

DEFAULT_VAL_FRACTION = 0.02
SPLIT_SEED = "muaalem-distill-v1"


def window_starts(
    num_samples: int,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int | None = None,
    min_samples: int | None = None,
) -> list[int]:
    """Start offsets of every training window in a clip of ``num_samples``.

    Walks from 0 in ``hop_samples`` steps. A start is emitted while at least
    ``min_samples`` of real audio remain, so the final window may be short (and is padded
    downstream). A clip shorter than ``min_samples`` yields nothing.
    """
    hop = hop_samples if hop_samples is not None else int(SAMPLE_RATE * DEFAULT_HOP_SECONDS)
    minimum = (
        min_samples
        if min_samples is not None
        else int(SAMPLE_RATE * DEFAULT_MIN_WINDOW_SECONDS)
    )
    if hop <= 0:
        raise ValueError(f"hop_samples must be positive, got {hop}")
    if num_samples < minimum:
        return []

    starts: list[int] = []
    start = 0
    while start + minimum <= num_samples:
        starts.append(start)
        # A window that already reaches the end of the clip is the last one; stepping
        # further would only emit windows made mostly of padding.
        if start + window_samples >= num_samples:
            break
        start += hop
    return starts


def clip_split(filename: str, val_fraction: float = DEFAULT_VAL_FRACTION) -> str:
    """``"val"`` or ``"train"`` for one clip, deterministically from its filename.

    Hashing the name (not the path, not an index) keeps the split identical across
    machines and immune to directory-walk order, so a validation agreement number stays
    comparable between runs.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    digest = hashlib.sha256(f"{SPLIT_SEED}:{filename}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "val" if bucket < val_fraction else "train"


def split_clips(
    paths: list[Path], val_fraction: float = DEFAULT_VAL_FRACTION
) -> tuple[list[Path], list[Path]]:
    """Partition clips into (train, val) by :func:`clip_split`."""
    train, val = [], []
    for path in paths:
        (val if clip_split(path.name, val_fraction) == "val" else train).append(path)
    return train, val


def discover_clips(audio_root: Path, pattern: str = "*.wav") -> list[Path]:
    """Every clip under ``audio_root``, sorted so the listing is reproducible."""
    return sorted(audio_root.rglob(pattern))


@dataclass(frozen=True)
class WindowRef:
    """One training example: a clip and the sample offset the window starts at."""

    path: Path
    start_sample: int

    @property
    def key(self) -> str:
        return f"{self.path.name}#{self.start_sample}"


def build_window_index(
    paths: list[Path],
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    min_window_seconds: float = DEFAULT_MIN_WINDOW_SECONDS,
) -> list[WindowRef]:
    """Enumerate every window in every clip, reading durations only (no audio decode).

    ``soundfile.info`` reads the header, so a 60-hour corpus indexes in seconds rather
    than the tens of minutes a full decode would take. Unreadable files are skipped rather
    than failing the run -- a corpus assembled by a long filtering pipeline reliably
    contains a few truncated writes.
    """
    import soundfile as sf

    hop_samples = int(SAMPLE_RATE * hop_seconds)
    min_samples = int(SAMPLE_RATE * min_window_seconds)

    refs: list[WindowRef] = []
    for path in paths:
        try:
            info = sf.info(str(path))
        except Exception:
            continue
        if info.samplerate != SAMPLE_RATE:
            # Resampling would silently change the mel front end; the corpus is already
            # 16 kHz, so a mismatch means something upstream is wrong. Skip loudly-ish.
            continue
        for start in window_starts(info.frames, WINDOW_SAMPLES, hop_samples, min_samples):
            refs.append(WindowRef(path, start))
    return refs


@dataclass(frozen=True)
class CorpusInventory:
    """What a corpus root yields, before any feature extraction happens."""

    num_clips: int
    num_readable_clips: int
    total_hours: float
    num_train_windows: int
    num_val_windows: int

    @property
    def num_windows(self) -> int:
        return self.num_train_windows + self.num_val_windows

    def as_dict(self) -> dict:
        return {
            "num_clips": self.num_clips,
            "num_readable_clips": self.num_readable_clips,
            "total_hours": round(self.total_hours, 2),
            "num_train_windows": self.num_train_windows,
            "num_val_windows": self.num_val_windows,
            "num_windows": self.num_windows,
        }


def inventory(
    audio_root: Path,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    min_window_seconds: float = DEFAULT_MIN_WINDOW_SECONDS,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> CorpusInventory:
    """Count clips, hours and windows for a corpus root."""
    import soundfile as sf

    paths = discover_clips(audio_root)
    train_paths, val_paths = split_clips(paths, val_fraction)

    total_frames = 0
    readable = 0
    for path in paths:
        try:
            info = sf.info(str(path))
        except Exception:
            continue
        readable += 1
        total_frames += info.frames

    return CorpusInventory(
        num_clips=len(paths),
        num_readable_clips=readable,
        total_hours=total_frames / SAMPLE_RATE / 3600,
        num_train_windows=len(build_window_index(train_paths, hop_seconds, min_window_seconds)),
        num_val_windows=len(build_window_index(val_paths, hop_seconds, min_window_seconds)),
    )


class DistillWindowDataset:
    """Fixed ``(250, 160)`` feature windows, extracted one window at a time.

    A torch ``Dataset`` in duck-typed form (``__len__`` / ``__getitem__``) so the module
    imports without torch. The feature extractor is built lazily per worker process:
    ``SeamlessM4TFeatureExtractor`` is not fork-safe to share, and each DataLoader worker
    needs its own.

    Yields only the input features. There are no targets here by design -- the teacher runs
    online in the training step and produces them from this same tensor, which keeps
    teacher and student pointwise aligned on identical input.
    """

    def __init__(
        self,
        refs: list[WindowRef],
        model_id: str = "obadx/muaalem-model-v3_2",
        window_samples: int = WINDOW_SAMPLES,
    ) -> None:
        self.refs = refs
        self.model_id = model_id
        self.window_samples = window_samples
        self._extractor = None

    def __len__(self) -> int:
        return len(self.refs)

    @property
    def extractor(self):
        if self._extractor is None:
            from transformers import SeamlessM4TFeatureExtractor

            self._extractor = SeamlessM4TFeatureExtractor.from_pretrained(self.model_id)
            if self._extractor.sampling_rate != SAMPLE_RATE:
                raise ValueError(
                    f"{self.model_id} extractor is {self._extractor.sampling_rate} Hz, "
                    f"not {SAMPLE_RATE} Hz"
                )
        return self._extractor

    def load_window(self, ref: WindowRef):
        """The window's waveform, zero-padded to the full window length like the device."""
        import numpy as np
        import soundfile as sf

        samples, rate = sf.read(
            str(ref.path),
            start=ref.start_sample,
            frames=self.window_samples,
            dtype="float32",
            always_2d=False,
        )
        if rate != SAMPLE_RATE:
            raise ValueError(f"{ref.path} is {rate} Hz, not {SAMPLE_RATE} Hz")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if len(samples) < self.window_samples:
            samples = np.pad(samples, (0, self.window_samples - len(samples)))
        return samples

    def __getitem__(self, index: int):
        import torch

        ref = self.refs[index]
        samples = self.load_window(ref)
        extracted = self.extractor(
            samples, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False
        )
        features = extracted.input_features[0]

        # The extractor's frame count follows from the padded window length, but pin it:
        # the ANE input is static, so anything other than exactly 250 frames is a bug that
        # would otherwise surface as a shape error deep in the training step.
        if features.shape[0] != FEATURE_FRAMES:
            if features.shape[0] > FEATURE_FRAMES:
                features = features[:FEATURE_FRAMES]
            else:
                features = torch.nn.functional.pad(
                    features, (0, 0, 0, FEATURE_FRAMES - features.shape[0])
                )
        return features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory a recitation corpus for size distillation"
    )
    parser.add_argument(
        "--audio-root", type=Path, required=True, help="directory searched recursively for .wav"
    )
    parser.add_argument("--hop-seconds", type=float, default=DEFAULT_HOP_SECONDS)
    parser.add_argument("--min-window-seconds", type=float, default=DEFAULT_MIN_WINDOW_SECONDS)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.audio_root.is_dir():
        raise SystemExit(f"not a directory: {args.audio_root}")

    inv = inventory(
        args.audio_root, args.hop_seconds, args.min_window_seconds, args.val_fraction
    )

    if args.json:
        print(json.dumps(inv.as_dict(), indent=2, ensure_ascii=False))
        return

    print(f"Corpus: {args.audio_root}")
    print(f"  clips            {inv.num_clips:,} ({inv.num_readable_clips:,} readable)")
    print(f"  audio            {inv.total_hours:.1f} hours")
    print(f"  windows          {inv.num_windows:,} at a {args.hop_seconds:g}s stride")
    print(f"    train          {inv.num_train_windows:,}")
    print(f"    val            {inv.num_val_windows:,}")


if __name__ == "__main__":
    main()
