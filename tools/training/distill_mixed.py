"""The round-2 mixed window stream: 3 parts tlog to 1 part Tadabur, in every batch.

:class:`training.distill_stream.StreamingWindowDataset` streams Tadabur alone. This module
adds the second ingredient without touching that stream: :class:`MixedWindowDataset` reads
the control's (#223) Tadabur dataset exactly as it is -- its shard order, shuffle buffer,
seed, exposure exclusion and augmentation -- and interleaves windows from
:func:`tadabur.tlog_pool.iter_training_rows` (the frozen #224 shard schedule) with them, so
every batch of ``mix.size`` is exactly ``mix.tlog`` tlog and ``mix.tadabur`` Tadabur windows.

**Composition is by construction, not by sampling odds.** Windows are emitted in a repeat of
``mix.tlog`` tlog followed by ``mix.tadabur`` Tadabur; the batch size is the mix's size, so
a batch boundary is always a pattern boundary, and :func:`check_batch_composition` asserts it
on every batch the trainer reads. The stream stops the moment either side runs out -- never a
short batch and never a skewed one -- so ``drop_last`` has nothing to drop.

**Augmentation is keyed per clip, and by the same recipe.** The tlog side is cut into 5 s
windows at the Tadabur state's hop, after the whole clip has been perturbed (as Tadabur's
stream does), under ``(f"{split}:{shard_file}", row)``. That string tag cannot collide with a
Tadabur ``(shard:int, row)`` key, so a tlog clip never shares a draw with a Tadabur row and
the shared Tadabur rows keep the control's draws unchanged. The teacher reads the clean
window; the student the perturbed one.

**What a run reports.** Every consumed tlog row key is appended by the reader to the run's
``tlog_consumed.jsonl.gz``, and :func:`per_source_stats` turns the logits of one batch into
per-source KL, teacher/student blank ratios and the blank-collapse margin, so the run can say
where a divergence or a collapse comes from.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from tadabur.tlog_pool import iter_training_rows
from training.distill_loss import agreement_stats, frame_weights, weighted_kl
from training.distill_stream import (
    Augment,
    StreamingWindowDataset,
    cut_window,
    row_window_starts,
    window_shuffle_seed,
)

#: The two sources' tags in a mixed item's third element. Distinct ints so a ``DataLoader``
#: collates them into one tensor the trainer can count.
SOURCE_TLOG = 0
SOURCE_TADABUR = 1

#: The metrics record's suffix for each source tag (``kl_tlog``, ``kl_tadabur``, ...).
SOURCE_NAMES = {SOURCE_TLOG: "tlog", SOURCE_TADABUR: "tadabur"}

#: Under the run's out dir: where the tlog shards' downloads are cached, and the reader's
#: consumed-keys log. The log is evidence of the prefix the run read, so it lives with the
#: run rather than in a scratch directory.
TLOG_CACHE_DIRNAME = "tlog_cache"
TLOG_RUN_LOG_NAME = "tlog_consumed.jsonl.gz"

_PART_NAMES = frozenset(SOURCE_NAMES.values())


@dataclass(frozen=True)
class Mix:
    """How many windows of each source make one batch."""

    tlog: int
    tadabur: int

    @property
    def size(self) -> int:
        return self.tlog + self.tadabur


def parse_mix(spec: str, batch_size: int) -> Mix:
    """``"tlog=24,tadabur=8"`` (either order) as a :class:`Mix` of ``batch_size`` windows.

    Refuses a missing, extra, duplicated, non-integer or non-positive part, and a size that
    is not the batch's: a mix that does not fill a batch would make the per-batch composition
    depend on ``drop_last`` instead of on the stream.
    """
    counts: dict[str, int] = {}
    for part in spec.split(","):
        name, separator, value = part.partition("=")
        if not separator or name not in _PART_NAMES or name in counts:
            raise ValueError(
                f"{spec!r}: every part must be tlog=<n> or tadabur=<n> once, got {part!r}"
            )
        try:
            count = int(value)
        except ValueError:
            raise ValueError(f"{spec!r}: {name} is not an integer: {value!r}") from None
        if count <= 0:
            raise ValueError(f"{spec!r}: {name} must be positive, got {count}")
        counts[name] = count
    if set(counts) != _PART_NAMES:
        missing = sorted(_PART_NAMES - set(counts))
        raise ValueError(f"{spec!r}: no {' or '.join(missing)} part")
    mix = Mix(tlog=counts["tlog"], tadabur=counts["tadabur"])
    if mix.size != batch_size:
        raise ValueError(
            f"{spec!r} is {mix.size} windows, not the batch size {batch_size}: every batch "
            f"must be exactly the mix"
        )
    return mix


def batch_counts(sources: torch.Tensor) -> dict[str, int]:
    """The per-source window counts of one batch, as the metrics record states them."""
    return {f"batch_{name}": int((sources == source).sum())
            for source, name in SOURCE_NAMES.items()}


def check_batch_composition(sources: torch.Tensor, mix: Mix) -> None:
    """Raise unless ``sources`` is exactly ``mix``'s windows.

    Called on every batch the trainer reads: the composition is the round's defining input,
    and a stream that silently drifts to 23 + 9 would still train.
    """
    counts = batch_counts(sources)
    expected = {"batch_tlog": mix.tlog, "batch_tadabur": mix.tadabur}
    if int(sources.numel()) != mix.size or counts != expected:
        raise ValueError(
            f"a mixed batch must be {mix.tlog} tlog + {mix.tadabur} tadabur windows, got "
            f"{int(sources.numel())} ({counts})"
        )


class MixedWindowDataset(torch.utils.data.IterableDataset):
    """``(teacher, student, source)`` items over the Tadabur and tlog streams.

    ``tadabur`` is the control's :class:`~training.distill_stream.StreamingWindowDataset`,
    read through its own ``__iter__``: its shard permutation, shuffle buffer, exclusion,
    augmentation and extraction are the Tadabur side's, so with an all-Tadabur mix the items
    are the control's stream. ``schedule`` is the committed #224 schedule and the remaining
    arguments are :func:`tadabur.tlog_pool.iter_training_rows`'s, except ``shuffle_buffer``,
    ``seed`` and ``augment``, which come from ``tadabur`` so both sides share them.

    The tlog side is windowed, augmented per clip and shuffle-buffered by the same buffer
    length and window-shuffle seed as the Tadabur side, then extracted through the Tadabur
    dataset's ``extract_pair``. Both streams are closed when iteration ends, so a run that
    stops because one side ran out still flushes the tlog consumed-keys log and releases its
    open shards.
    """

    def __init__(
        self,
        tadabur: StreamingWindowDataset,
        *,
        mix: Mix,
        schedule,
        cache_dir,
        run_log,
        rows=None,
        stream=None,
        load_audio=None,
    ) -> None:
        self.tadabur = tadabur
        self.mix = mix
        self.schedule = schedule
        self.cache_dir = cache_dir
        self.run_log = run_log
        self.rows = rows
        self.stream = stream
        self.load_audio = load_audio

    def _tlog_pairs(self):
        """``(clean, augmented)`` waveform pairs for every tlog window, in schedule order.

        Each clip is perturbed whole -- once, before windowing -- under
        ``(f"{split}:{shard_file}", row)``, and the two clips are cut at the same starts, so a
        teacher window and the student window it is compared against cover the same audio.
        The reader is consumed lazily, one clip at a time, so a run that stops early leaves
        most of its shards unread and unlogged.
        """
        augment: Augment | None = self.tadabur.augment
        reader = iter_training_rows(
            self.rows,
            cache_dir=self.cache_dir,
            schedule=self.schedule,
            stream=self.stream,
            load_audio=self.load_audio,
            run_log=self.run_log,
        )
        for (split, shard_file, row), samples in reader:
            augmented = (
                None if augment is None
                else augment(samples, f"{split}:{shard_file}", row)
            )
            for start in row_window_starts(len(samples), self.tadabur.hop_seconds):
                clean = cut_window(samples, start)
                yield clean, (clean if augmented is None else cut_window(augmented, start))

    def _tlog_windows(self):
        """``_tlog_pairs`` through the Tadabur stream's shuffle buffer and extraction."""
        rng = random.Random(window_shuffle_seed(self.tadabur.seed))
        buffer: list = []

        for pair in self._tlog_pairs():
            buffer.append(pair)
            if len(buffer) >= self.tadabur.shuffle_buffer:
                index = rng.randrange(len(buffer))
                buffer[index], buffer[-1] = buffer[-1], buffer[index]
                yield self.tadabur.extract_pair(buffer.pop())

        rng.shuffle(buffer)
        for pair in buffer:
            yield self.tadabur.extract_pair(pair)

    def __iter__(self):
        pattern = ((SOURCE_TLOG,) * self.mix.tlog
                   + (SOURCE_TADABUR,) * self.mix.tadabur)
        tadabur = iter(self.tadabur)
        tlog = self._tlog_windows()
        try:
            while True:
                batch = []
                for source in pattern:
                    if source == SOURCE_TLOG:
                        pair = next(tlog, None)
                        if pair is None:
                            return
                        batch.append((*pair, SOURCE_TLOG))
                    else:
                        item = next(tadabur, None)
                        if item is None:
                            return
                        batch.append((*item, SOURCE_TADABUR))
                yield from batch
        finally:
            # Both `__iter__`s and the reader are generators: closing them flushes the
            # reader's consumed-keys log and drops the shards each one has open.
            tlog.close()
            tadabur.close()


def per_source_stats(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    sources: torch.Tensor,
    loss_config,
) -> dict[str, float]:
    """KL, blank ratios and blank-collapse margin per source, for one batch.

    ``sources`` must be on the logits' device. The KL is the run's own weighted KL over each
    source's rows (:func:`training.distill_loss.weighted_kl` with the teacher-derived frame
    weights), so the per-source numbers add up to the loss the step optimised rather than
    measuring something else. A batch with no window from one source is a composition bug
    the caller asserts against; it raises here rather than reporting a missing source as
    zero.
    """
    stats: dict[str, float] = {}
    for source, name in SOURCE_NAMES.items():
        mask = sources == source
        if not bool(mask.any()):
            raise ValueError(f"the batch holds no {name} window: {batch_counts(sources)}")
        student, teacher = student_logits[mask], teacher_logits[mask]
        stats[f"kl_{name}"] = float(weighted_kl(
            student,
            teacher,
            frame_weights(
                teacher,
                nonblank_weight=loss_config.nonblank_weight,
                confirm_timesteps=loss_config.confirm_timesteps,
                confirm_weight=loss_config.confirm_weight,
            ),
            temperature=loss_config.temperature,
        ))
        blank = agreement_stats(student, teacher)
        for field in ("teacher_blank_ratio", "student_blank_ratio", "blank_collapse_margin"):
            stats[f"{field}_{name}"] = float(getattr(blank, field))
    return stats
