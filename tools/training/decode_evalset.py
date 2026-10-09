"""A held-out clip set with the teacher's decode frozen into it.

Size distillation is behavioural cloning of the teacher's **phoneme decode**. The thing to
measure is therefore the thing being cloned: the teacher's confirmed phoneme stream against
the student's, string to string.

It is worth being explicit about what is *not* measured here, because this work spent two
passes measuring it by mistake. ``tadabur.scorer`` aligns a decode against an ayah reference
with Smith-Waterman and thresholds the result. That machinery answers two other questions:
which clips are clean enough for the ADR-0001 **fine-tune corpus** (the filter, whose two
poison rejects its own comments call "NOT a Muraja parameter"), and whether Muraja **advances**
the reciter (ADR-0005, a different threshold again). Neither is a distillation question -- no
reference string, no aligner and no threshold appears in "does the student emit what the
teacher emits" -- and ADR-0008 already records that the gate "should not be the headline
metric at all". So this module caches the teacher's decoded string and nothing downstream of
it.

**Batch size is part of the protocol.** The teacher is bit-identical at a fixed batch size
(120 clips re-decoded at the manifest's batch 32: zero edits) and drifts **0.17% of
characters** at batch 4, from bf16 accumulation over different padding and matmul groupings.
That is the same order as the difference between two checkpoints, so the manifest records the
batch size it was built with and evaluations inherit it rather than choosing their own.

**The teacher is decoded once and frozen.** Every evaluation used to decode both models,
paying the teacher's 42 ms/window again for every student measured. The teacher is
deterministic and the protocol is fixed, so its decode is a property of the clip: it is
computed here and written into the manifest, and :mod:`training.distill_eval` then scores a
student against it. Besides halving the work it makes two checkpoints comparable by
construction rather than by hoping the teacher ran identically twice. The manifest records
the teacher id, the decode protocol and the bounds it was built under, and
:func:`check_provenance` refuses one whose rules no longer match the code loading it.

**A uniform sample, because the metric has no threshold.** Pooled character accuracy over
~168k phonemes is informative on any sample of clips; there is nothing to saturate and no
decision boundary to enrich around, so the set is a plain reservoir draw over eligible clips.
(An earlier revision drew a second sample stratified by the scorer's ``match_ratio``, for a
pass/fail metric that is no longer computed. Legacy manifests carry it; it is **not** a
uniform draw, so :func:`load_manifest` drops it rather than pooling it into a fidelity
estimate.)

**Clips are stored as 32-bit float.** The manifest caches the teacher's decode of the
waveform in memory and the student reads the file back, so anything lossy between them is a
difference charged entirely to the student. A PCM_16 round trip is lossy enough to matter:
measured on 60 clips it changed the teacher's own decoded string on 12 and Tadabur audio peaks
above 1.0, so it clips as well.

**Provenance: strided shards, never contiguous.** Candidates come from Tadabur shards no
training run may touch. Shards 0-19 produced the staged ``clips_v2`` corpus, so they are
already reserved; :func:`gate_eval_shards` reserves a strided block on top of that, held out
from streaming runs and never staged, so one set is valid for the staged-corpus baseline and
every streaming successor. Strided rather than a block at one end because nothing guarantees
shard order does not group reciters, and a tail block would then be a distribution shift
rather than a sample.

**A dev/test boundary inside the set**, split by **reciter** -- one voice's recordings share
channel, pace and style, so a clip-level split puts near-duplicates on both sides and a
"held-out" test number quietly measures the dev set again.

The candidates are **unfiltered**: Tadabur's staging filter never saw them. For cloning that
is right, since the student has to reproduce the teacher on whatever it is given.

Usage::

    python -m training.decode_evalset --out-dir ../tadabur/gate_eval
    python -m training.decode_evalset --out-dir ../tadabur/gate_eval --describe
    python -m training.decode_evalset --print-training-shards

Linux + CUDA for the build; loading, the statistics and the shard arithmetic are torch-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tadabur.panel_seal import SealedPanelError

# Bumped whenever the manifest layout, the strata, or anything the cached teacher decisions
# depend on changes, so a stale set is refused rather than silently reused.
SCHEMA_VERSION = "decode-evalset-v2"

MANIFEST_FILENAME = "manifest.json"
CLIPS_DIRNAME = "clips"

# Every 19th shard from 20: 20 shards spread across the corpus rather than a block at one
# end (see the module docstring on why contiguous is the wrong choice). 20 shards yield
# ~20,000 candidates, comfortably more than the sample needs; the count was originally set by
# a ratio-stratified second sample that no longer exists, and is kept because a larger
# candidate pool costs only scan time and reserves only 5% of the training corpus.
GATE_EVAL_SHARD_STRIDE = 19
GATE_EVAL_SHARD_START = 20
NUM_TADABUR_SHARDS = 385

# The evaluation is a pooled character accuracy over ~168k phonemes, not a per-clip
# proportion, so n is generous at 2,000: the reciter-clustered interval on it is +/-0.6
# points, which resolves the differences between checkpoints this work produces.
DEFAULT_TARGET = 2000

# Matches ``tadabur.filter.MAX_AYAH_DURATION_S``: the staging filter drops clips longer than
# this, so including them would measure the student on audio the product pipeline never
# reaches -- and each one costs ~46 teacher window passes.
MAX_CLIP_SECONDS = 50.0
# Below this there is not enough audio for one confirmed window to mean anything.
MIN_CLIP_SECONDS = 1.5

# Above this many discordant pairs the exact sum is replaced by the normal approximation:
# the exact form costs O(n^2) big-integer work and the approximation is indistinguishable at
# these counts, while ``2.0 ** trials`` would simply overflow past 1023.
EXACT_BINOMIAL_LIMIT = 1000

DEFAULT_SEED = 20260918
# Stable across rebuilds so "the test half" means the same reciters in every set.
SPLIT_SALT = "gate-evalset-reciter-split-v1"


def read_clip_audio(clips_dir: Path, filename: str):
    """One frozen clip's waveform: mono, float32, at the expected rate.

    Every tool that scores the frozen set reads its audio through here, because the two
    guards are load-bearing rather than incidental. A clip at another rate would be decoded
    as the wrong duration by a pipeline that never resamples, and a stereo clip averaged in
    one tool but not another would hand two tools different audio for the same filename --
    the kind of silent divergence that makes two measurements quietly incomparable.

    float32 is also not incidental: writing the set as PCM_16 moved the *teacher's own*
    decode on 12 of 60 clips, and Tadabur audio peaks at 1.037 so it clips.
    """
    import soundfile as sf

    from training.distill_data import SAMPLE_RATE

    from tadabur.audio import read_audio

    samples, rate = read_audio(clips_dir / filename, dtype="float32")
    if rate != SAMPLE_RATE:
        raise SystemExit(f"{filename} is {rate} Hz, not {SAMPLE_RATE}")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples


def scoring_batch_size(evalset: "EvalSet", requested: int = 0) -> int:
    """The batch to decode at: the manifest's, unless explicitly overridden.

    The teacher is bit-identical at a fixed batch size and moves **0.17% of characters**
    between batch 4 and 32 -- bf16 accumulation over different padding and matmul groupings.
    That is the same order as a real gain, so letting it float would let a rerun look like
    progress. Inheriting the manifest's value is what keeps two checkpoints comparable; an
    override is honoured but warned about, because its numbers are not comparable to
    anything else scored against this set.
    """
    manifest_batch = evalset.provenance.get("batch_size", 16)
    if not requested:
        return manifest_batch
    if requested != manifest_batch:
        print(
            f"[warn] scoring at batch {requested}, manifest built at {manifest_batch}: "
            "expect ~0.2% of characters to move for that reason alone. Comparisons across "
            "batch sizes are refused.",
            flush=True,
        )
    return requested


def gate_eval_shards() -> list[int]:
    """The shards reserved for evaluation -- strided, never trained on."""
    return list(
        range(GATE_EVAL_SHARD_START, NUM_TADABUR_SHARDS, GATE_EVAL_SHARD_STRIDE)
    )


def training_shard_spec(
    held_out_below: int = GATE_EVAL_SHARD_START,
    num_shards: int = NUM_TADABUR_SHARDS,
    reserved_shards: list[int] | None = None,
) -> str:
    """The ``--stream-shards`` spec that excludes both held-out blocks.

    Shards below ``held_out_below`` produced the staged corpus; the evaluation shards are
    removed on top of that. Returned as the compact range spec ``training.distill_stream``
    already parses, so a run cannot be configured with a hand-typed range that clips the
    reservation.

    ``reserved_shards`` must be the shards a set was **actually built on**, which is not
    always :func:`gate_eval_shards`: ``--shards`` can override it. Defaulting to the
    canonical block while a set was built on some other shard is how the advertised leakage
    guard quietly stops guarding, so the CLI reads this from the built manifest.
    """
    reserved = set(gate_eval_shards() if reserved_shards is None else reserved_shards)
    available = [i for i in range(held_out_below, num_shards) if i not in reserved]
    if not available:
        raise ValueError("no training shards left after the reservation")

    terms: list[str] = []
    start = previous = available[0]
    for index in available[1:]:
        if index == previous + 1:
            previous = index
            continue
        terms.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    terms.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(terms)


def reciter_split(reciter_id: int) -> str:
    """``"dev"`` or ``"test"`` for one reciter, stably and without a stored table.

    Split by reciter rather than by clip: one reciter's recordings share channel, pace and
    style, so a clip-level split puts near-duplicates on both sides and a "held-out" test
    number quietly measures the dev set again.
    """
    digest = hashlib.sha256(f"{SPLIT_SALT}:{reciter_id}".encode("utf-8")).digest()
    return "test" if digest[0] % 2 else "dev"


@dataclass(frozen=True)
class EvalClip:
    """One frozen clip and the teacher decode cached for it.

    ``teacher_text`` is the whole target: the confirmed phoneme stream the student has to
    reproduce, and it is the only thing here a distillation metric reads.

    Scorer-derived fields (``match_ratio`` and the filter's poison flags) are deliberately
    absent. An earlier revision carried them "for provenance", but ``build`` never populated
    them once the scorer left this module, so every new manifest wrote zeros for all 2,000
    clips and the promise was false. Legacy ``gate-evalset-v1`` files on disk still contain
    real values; read the JSON directly if a filter-side question ever needs them.
    """

    filename: str
    surah_ayah: str
    reciter_id: int
    shard: int
    duration_s: float
    split: str
    teacher_text: str

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "surah_ayah": self.surah_ayah,
            "reciter_id": self.reciter_id,
            "shard": self.shard,
            "duration_s": round(self.duration_s, 2),
            "split": self.split,
            "teacher_text": self.teacher_text,
        }


@dataclass(frozen=True)
class EvalSet:
    """A built set: its clips, its provenance, and what the scan had to discard."""

    schema_version: str
    clips: tuple[EvalClip, ...]
    num_scanned: int
    num_skipped: int
    provenance: dict

    def fingerprint(self) -> str:
        """A stable id for *this* evaluation: its clips, their cached truth, its rules.

        Two sets built from the same shards under different protocols share filenames, so
        comparing a saved decisions file by name alone can silently score one checkpoint
        against another's truth. Hashing the clip list with the teacher's decodes and the
        provenance makes that mismatch detectable.
        """
        payload = json.dumps(
            {
                "schema": self.schema_version,
                "provenance": self.provenance,
                "clips": [[c.filename, c.teacher_text] for c in self.clips],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def subset(self, split: str = "both") -> tuple[EvalClip, ...]:
        """All clips, or one side of the reciter split."""
        if split == "both":
            return self.clips
        if split not in ("dev", "test"):
            raise ValueError(f"unknown split {split!r}")
        return tuple(clip for clip in self.clips if clip.split == split)

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "provenance": self.provenance,
            "num_scanned": self.num_scanned,
            "num_skipped": self.num_skipped,
            "num_clips": len(self.clips),
            "clips": [clip.as_dict() for clip in self.clips],
        }


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance between two token streams (token ids or phoneme characters).

    Torch-free, so the truth-site scorer's teacher-agreement rule
    (:mod:`training.acceptance_stats`) and the distillation tools share one recurrence.
    """
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (token_a != token_b),
                )
            )
        previous = current
    return previous[-1]


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Thin wrapper over the project's existing implementation.

    ``training.counterfactual_eval`` already owns this and ``tadabur.tashkeel_acceptance``
    already imports it from there; re-deriving it here would be a second copy of a formula
    that is easy to get subtly wrong near the extremes. Re-exported rather than imported
    at the call sites only so this module stays the single place the evaluation's statistics
    are looked up.
    """
    from training.counterfactual_eval import wilson_interval as canonical

    return canonical(successes, trials, z)


def cluster_bootstrap(
    rows: list,
    cluster_of,
    statistic,
    *,
    seed: int,
    iterations: int = 4000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap that resamples whole **clusters**, not observations.

    Every interval this evaluation reports is over clips that are not independent: 2,000 of
    them come from 286 reciters, and whether the student matches the teacher correlates
    within a voice. An independent-sample interval is therefore too narrow on exactly the
    question a ship decision asks -- measured here, about 23% too narrow.

    ``statistic`` pools a list of rows into one number, so the same resampling serves a
    proportion, a pooled ratio of sums, and a paired difference of pooled ratios. Those were
    three separate loops with three different percentile conventions and three different
    hardcoded seeds until this existed; one of them indexed ``draws[100], draws[3899]``
    against a literal ``range(4000)``, which is a wrong interval the moment either number
    moves. Written once, the arithmetic is wrong at most once.

    Resampling clusters makes no assumption about the size of the correlation, which is the
    right trade when there is one clustering variable and enough of them to resample. The
    total row count varies between draws, as it should: a corpus with a different set of
    reciters really would have a different number of clips.
    """
    if not rows:
        return (0.0, 0.0)
    grouped: dict = {}
    for row in rows:
        grouped.setdefault(cluster_of(row), []).append(row)
    keys = list(grouped)

    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sample: list = []
        for _ in keys:
            sample.extend(grouped[keys[rng.randrange(len(keys))]])
        draws.append(statistic(sample))
    draws.sort()
    low = draws[int(alpha / 2 * iterations)]
    high = draws[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (low, high)


# The on-disk format id. Legacy sets carry the old name; the layout they describe is the
# one this code reads, so they are accepted rather than forcing a 90-minute rebuild.
COMPATIBLE_SCHEMAS = (SCHEMA_VERSION, "gate-evalset-v1")


def load_manifest(out_dir: Path) -> EvalSet:
    """Read a built set, refusing one this code cannot honestly interpret.

    Legacy sets carry a second sample stratified by the scorer's ``match_ratio``, drawn for a
    pass/fail metric this code no longer computes. It is **not** a uniform draw, so pooling it
    into a character-accuracy estimate would bias that estimate toward clips near a threshold
    that no longer appears anywhere. Those clips are dropped on load.
    """
    path = Path(out_dir) / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version not in COMPATIBLE_SCHEMAS:
        raise SystemExit(
            f"{path} was written by schema {version!r}; this code reads "
            f"{COMPATIBLE_SCHEMAS}. Rebuild it -- the cached teacher decodes are only valid "
            f"for the protocol the manifest was built under."
        )

    records = payload["clips"]
    uniform = [r for r in records if r.get("in_population", True)]
    clips = tuple(
        EvalClip(
            filename=r["filename"],
            surah_ayah=r["surah_ayah"],
            reciter_id=r["reciter_id"],
            shard=r["shard"],
            duration_s=r["duration_s"],
            split=r["split"],
            teacher_text=r["teacher_text"],
        )
        for r in uniform
    )
    return EvalSet(
        schema_version=version,
        clips=clips,
        num_scanned=payload["num_scanned"],
        num_skipped=payload["num_skipped"],
        provenance={**payload["provenance"], "dropped_stratified_clips":
                    len(records) - len(uniform)},
    )


def check_provenance(evalset: EvalSet, teacher_model_id: str) -> None:
    """Refuse a manifest whose cached teacher decisions were made under other rules.

    A different teacher, a different decode protocol or a different confirmation split all
    change the cached decode without changing the file. The failure
    mode is a plausible-looking agreement number computed against stale truth, which is
    exactly the class of silent wrongness these tools exist not to produce. Every such field
    is checked **here**, in one place -- a second provenance check somewhere else is how one
    of them ends up unchecked.
    """
    from training.decoding import PROTOCOL_VERSION
    from training.distill_loss import CONFIRM_TIMESTEPS

    # No threshold here: nothing this set is used for has one.
    expected = {
        "teacher_model_id": teacher_model_id,
        "confirm_timesteps": CONFIRM_TIMESTEPS,
        "protocol_version": PROTOCOL_VERSION,
    }
    mismatches = [
        f"  {key}: manifest={evalset.provenance.get(key)!r} current={value!r}"
        for key, value in expected.items()
        if evalset.provenance.get(key) != value
    ]
    if mismatches:
        raise SystemExit(
            "refusing to use this evaluation set: its cached teacher decisions were "
            "produced under different rules.\n" + "\n".join(mismatches)
        )


class _Reservoir:
    """Uniform sample of at most ``quota`` items, in one streaming pass.

    Two passes over the candidates would be simpler, but a pass costs a re-download and a
    full teacher decode of every shard, so the sample must be chosen while streaming.
    Reservoir sampling keeps the sample uniform over everything seen; taking the first N
    instead would sample by shard order, i.e. by reciter.
    """

    def __init__(self, quota: int, seed: int) -> None:
        self.quota = quota
        self.rng = random.Random(seed)
        self.seen = 0
        self.kept: list = []

    def offer(self, item) -> bool:
        """Consider one candidate; returns whether it is currently in the sample."""
        self.seen += 1
        if len(self.kept) < self.quota:
            self.kept.append(item)
            return True
        index = self.rng.randrange(self.seen)
        if index < self.quota:
            self.kept[index] = item
            return True
        return False

    @property
    def is_full(self) -> bool:
        return len(self.kept) >= self.quota


def _clip_filename(shard: int, row_index: int, surah_id: int, ayah_id: int) -> str:
    """A name that carries its own provenance and parses with the staged-clip pattern.

    ``surah_id`` is Tadabur's **0-indexed** array position, matching the staged
    ``clips_v2`` filenames, so ``training.distill_gate.parse_surah_ayah`` reads these
    exactly as it reads the staged ones. The manifest carries the canonical key as well;
    the filename is for humans and for tools that only have a path.
    """
    return f"tadabur_sh{shard:03d}_i{row_index:05d}_S{surah_id}_A{ayah_id}_.wav"


def build(
    out_dir: Path,
    shards: list[int],
    target: int,
    row_sample: float,
    max_scan: int,
    seed: int,
    batch_size: int,
) -> EvalSet:
    """Scan the reserved shards once, decode each clip with the teacher, sample, freeze.

    One pass does everything: the teacher decode is the expensive part and is never repeated,
    the sample is a reservoir so it stays uniform over the whole scan, and shard blobs are
    deleted as they are consumed so peak disk is one shard.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import SeamlessM4TFeatureExtractor

    from tadabur.audio import decode_to_mono_16k
    from tadabur.dataset_source import canonical_surah_ayah
    from tadabur.reference_phonemes import load_reference_phonemes
    from tadabur.shard_reader import iter_shard_rows
    from training.distill_data import SAMPLE_RATE
    from training.decoding import PROTOCOL_VERSION, Decoder
    from training.distill_loss import CONFIRM_TIMESTEPS
    from training.distill_student import TEACHER_MODEL_ID
    from training.distill_train import load_teacher

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to build the evaluation set")
    device = torch.device("cuda")

    clips_dir = Path(out_dir) / CLIPS_DIRNAME
    clips_dir.mkdir(parents=True, exist_ok=True)

    teacher = Decoder(
        TEACHER_MODEL_ID,
        load_teacher(device),
        SeamlessM4TFeatureExtractor.from_pretrained(TEACHER_MODEL_ID),
        device,
        batch_size,
    )
    # Only to resolve which ayah a clip is, so a row with no canonical reference can be
    # skipped -- the reference string itself plays no part in the decode or the metric.
    references = load_reference_phonemes()

    reservoir = _Reservoir(target, seed)
    # Rows are skipped *before* the teacher pass, so a sub-1.0 rate buys coverage across all
    # reserved shards at a fraction of the GPU cost -- which is the right trade: a scan that
    # exhausts its budget inside the first few shards samples reciters, not the corpus.
    sampler = random.Random(seed ^ 0x5EED)

    written: set[str] = set()
    num_scanned = num_skipped = 0
    started = time.time()

    for position, shard in enumerate(shards):
        for row_index, row in enumerate(iter_shard_rows([shard], delete_after=True)):
            if max_scan and num_scanned >= max_scan:
                break
            if row_sample < 1.0 and sampler.random() >= row_sample:
                continue

            key = canonical_surah_ayah(int(row["surah_id"]), int(row["ayah_id"]))
            if references.get(key) is None:
                num_skipped += 1
                continue
            try:
                samples = decode_to_mono_16k(row["audio"]["bytes"])
            except SealedPanelError:  # a seal violation is never a corrupt input
                raise
            except Exception:
                num_skipped += 1
                continue
            duration = len(samples) / SAMPLE_RATE
            if not MIN_CLIP_SECONDS <= duration <= MAX_CLIP_SECONDS:
                num_skipped += 1
                continue

            text = teacher.decode_stream(samples)
            num_scanned += 1
            reciter_id = int(row["reciter_id"])
            candidate = EvalClip(
                filename=_clip_filename(
                    shard, row_index, int(row["surah_id"]), int(row["ayah_id"])
                ),
                surah_ayah=key,
                reciter_id=reciter_id,
                shard=shard,
                duration_s=duration,
                split=reciter_split(reciter_id),
                teacher_text=text,
            )

            if reservoir.offer(candidate) and candidate.filename not in written:
                # 32-bit float, not PCM_16, and this is not a preference. The manifest caches
                # the teacher's decode of the waveform *in memory*; the student will read the
                # file back, and any difference between the two is charged to the student.
                # Measured on 60 clips, a PCM_16 round trip changed the teacher's own decoded
                # string on 12 of them, and Tadabur peaks above 1.0 (measured 1.037) so
                # PCM_16 clips real signal. FLOAT round-trips bit-exactly.
                sf.write(
                    str(clips_dir / candidate.filename),
                    np.asarray(samples, dtype="float32"),
                    SAMPLE_RATE,
                    subtype="FLOAT",
                )
                written.add(candidate.filename)

            if num_scanned % 100 == 0:
                rate = num_scanned / max(1e-6, time.time() - started)
                print(
                    f"  shard {shard} ({position + 1}/{len(shards)}) scanned "
                    f"{num_scanned} at {rate:.2f} clips/s -- kept "
                    f"{len(reservoir.kept)}/{reservoir.quota}",
                    flush=True,
                )
        if max_scan and num_scanned >= max_scan:
            break

    selected = {clip.filename: clip for clip in reservoir.kept}
    final = tuple(sorted(selected.values(), key=lambda c: c.filename))

    # Reservoir sampling writes every clip it ever accepts and evicts some later, so the
    # directory is a superset of the final sample. Sweep the difference at the end rather
    # than refcounting during the scan.
    for path in clips_dir.glob("*.wav"):
        if path.name not in selected:
            path.unlink()

    evalset = EvalSet(
        schema_version=SCHEMA_VERSION,
        clips=final,
        num_scanned=num_scanned,
        num_skipped=num_skipped,
        provenance={
            "teacher_model_id": TEACHER_MODEL_ID,
            "protocol_version": PROTOCOL_VERSION,
            "confirm_timesteps": CONFIRM_TIMESTEPS,
            "shards": shards,
            "row_sample": row_sample,
            "max_scan": max_scan,
            "seed": seed,
            "target": target,
            # Recorded because it changes the decode. The teacher is bit-identical at a fixed
            # batch size -- re-decoding 120 clips at batch 32 reproduced the cached strings
            # exactly, 0 edits -- but at batch 4 it drifts 0.17% of characters, from bf16
            # accumulation over different padding and matmul groupings. That is the same
            # order as a real improvement between two checkpoints, so it cannot float.
            "batch_size": batch_size,
            "min_clip_seconds": MIN_CLIP_SECONDS,
            "max_clip_seconds": MAX_CLIP_SECONDS,
            "source": "tadabur-shards-unfiltered",
            "elapsed_s": round(time.time() - started, 1),
        },
    )
    (Path(out_dir) / MANIFEST_FILENAME).write_text(
        json.dumps(evalset.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return evalset


def describe(evalset: EvalSet) -> str:
    """A human summary of what a built set contains."""
    import statistics

    clips = evalset.clips
    reciters = {clip.reciter_id for clip in clips}
    phonemes = sum(len(clip.teacher_text) for clip in clips)
    lines = [
        f"decode evaluation set -- {evalset.schema_version}",
        f"  scanned                {evalset.num_scanned} clips "
        f"({evalset.num_skipped} skipped) from shards {evalset.provenance['shards']}",
        f"  kept                   {len(clips)} clips over {len(reciters)} reciters",
        f"  teacher phonemes       {phonemes:,} "
        f"(median {statistics.median(len(c.teacher_text) for c in clips):.0f} per clip)",
        f"  clip duration          median "
        f"{statistics.median(c.duration_s for c in clips):.1f}s",
    ]
    for split in ("dev", "test"):
        lines.append(f"    {split:<20} {len(evalset.subset(split))} clips")
    dropped = evalset.provenance.get("dropped_stratified_clips", 0)
    if dropped:
        lines.append(
            f"  dropped on load        {dropped} clips from a legacy ratio-stratified sample "
            f"-- not a uniform draw, so not pooled into a fidelity estimate"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a held-out clip set with the teacher's decode frozen into it"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("../tadabur/gate_eval"))
    parser.add_argument(
        "--shards",
        default="",
        help="override the reserved shard list (default: every "
        f"{GATE_EVAL_SHARD_STRIDE}th shard from {GATE_EVAL_SHARD_START})",
    )
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument(
        "--row-sample",
        type=float,
        default=1.0,
        help="fraction of rows to decode, applied BEFORE the teacher pass. Below 1.0 it "
        "buys coverage of every reserved shard at a fraction of the GPU cost.",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=0,
        help="stop after this many decoded clips (0 = the whole reservation). A budget that "
        "runs out inside the first shards samples reciters, not the corpus -- prefer "
        "--row-sample.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--describe", action="store_true", help="summarise a built set")
    parser.add_argument(
        "--print-training-shards",
        action="store_true",
        help="print the --stream-shards spec that excludes every reserved shard",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.print_training_shards:
        # From the built manifest when there is one, so a set built with --shards does not
        # get a spec that still permits the shards it was drawn from.
        manifest = Path(args.out_dir) / MANIFEST_FILENAME
        reserved = (
            load_manifest(args.out_dir).provenance.get("shards")
            if manifest.exists()
            else None
        )
        print(training_shard_spec(reserved_shards=reserved))
        return

    if args.describe:
        evalset = load_manifest(args.out_dir)
        print(json.dumps(evalset.as_dict(), indent=2, ensure_ascii=False)
              if args.json else describe(evalset))
        return

    from tadabur.shard_reader import parse_shard_spec

    shards = parse_shard_spec(args.shards) if args.shards else gate_eval_shards()
    evalset = build(
        out_dir=args.out_dir,
        shards=shards,
        target=args.target,
        row_sample=args.row_sample,
        max_scan=args.max_scan,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print()
    print(describe(evalset))
    print(f"\nwrote {args.out_dir / MANIFEST_FILENAME}")
    print(
        f"training runs must use --stream-shards "
        f"{training_shard_spec(reserved_shards=shards)}"
    )


if __name__ == "__main__":
    main()
