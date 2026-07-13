"""Fixed-window duration/memory envelope for the waqf-head fine-tune (ADR-0004, A1).

The deployed CoreML pipeline runs the Muaalem backbone over **fixed 5 s windows** at a
**40 ms** post-adapter CTC lattice (``ml-model-transformation.md``); the ANE requires a
static ``(1, 250, 160)`` input, so the training window is pinned to match inference
(ADR-0004). This module turns that constraint into the *measured envelope* A2 (#24, the
HITL windowing-contract freeze) needs, and lays out provisional windowing options — it
makes **no product decision**.

It answers three questions, all torch-free so they run without a GPU:

* **How long are the recitations and their waqf segments?** :func:`duration_histogram`
  over the whole (un-waqf-segmented) clips (:func:`load_whole_clip_durations`) and over
  the waqf segments (:func:`load_segment_durations`). A recitation longer than one window
  must be tiled, so the fraction exceeding a candidate window length is what decides how
  central stitching is to the design.

* **What is the per-window CTC logit length, and does a batch fit 16 GB?** Each candidate
  window's 40 ms lattice length is :func:`training.waqf_distill.muaalem_lattice_length`
  of its 20 ms frame count (250 → 125 for the deployed 5 s window). :func:`memory_estimate`
  models activation memory with the standard activation-recomputation formula so the
  16 GB RTX 5060 Ti budget (ADR-0004's OOM consequence) becomes a per-window max batch.

* **How should windows overlap, own edges, and stitch?** :func:`policy_options` enumerates
  the candidate window/overlap/edge-ownership/stitch policies (non-overlapping tiling vs a
  center-trusted overlap) with a :func:`recommendation` for A2 to confirm.

The whole-clip durations come from the filter's passing manifest (``ManifestRecord``); the
waqf-segment durations from a segment manifest (``tadabur.segment_score``). Both are
gitignored generated data, so :func:`main` renders a self-contained markdown report
(:func:`render_report`) that is committed as ``docs/window-envelope.md`` — the artifact A2
reads.

Usage:
  python -m training.window_envelope \\
      --whole-manifest tadabur/audit_run/passing_subset_full.jsonl \\
      --segment-manifest tadabur/audit_run/segment_manifest_v4.jsonl \\
      --out docs/window-envelope.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from tadabur.manifest import read_records
from training.waqf_distill import (
    TEACHER_FRAME_MS,
    muaalem_lattice_length,
)

# ---------------------------------------------------------------------------
# Frame ↔ time geometry (shared with training.waqf_distill, restated for reports).
# ---------------------------------------------------------------------------

# One 20 ms encoder/teacher frame per this many seconds; 50 frames per second, so the
# deployed 5 s window is 250 feature frames (its 40 ms lattice length is 125).
FEATURE_FRAMES_PER_SECOND = 1000 // TEACHER_FRAME_MS  # 50

# The deployed, ANE-fixed inference window: 5 s = 250 feature frames (see ADR-0004,
# convert_to_coreml.py FIXED_SEQ_LEN). Training must match it, so it is the window *cap*.
DEPLOYED_WINDOW_SECONDS = 5.0

# ---------------------------------------------------------------------------
# Memory model. Muaalem is a 24-layer Wav2Vec2-BERT Conformer (hidden 1024, 16 heads,
# intermediate 4096); its FP32 checkpoint is 2424 MB (ml-model-transformation.md), so
# ~606M parameters. The fine-tune is LoRA on the phoneme head with the backbone frozen
# (ADR-0001), so trainable params / gradients / optimizer state are a rounding error next
# to the frozen backbone weights and the activations, which dominate the budget.
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 1024
NUM_LAYERS = 24
NUM_HEADS = 16
PARAM_COUNT = 606_000_000  # 2424 MB FP32 / 4 bytes (ml-model-transformation.md)
BF16_BYTES = 2

# Activation memory per standard transformer layer, stored in 2-byte precision, from
# Korthikanti et al. 2022 ("Reducing Activation Recomputation in Large Transformer
# Models"): sbh·(34 + 5·a·s/h) bytes, s=seq, b=batch, h=hidden, a=heads. The 5·a·s/h term
# is the attention-score buffer — quadratic in s, which is why a whole-clip window is
# infeasible while a 5 s window is cheap.
ACTIVATION_BASE_COEFF = 34
ACTIVATION_ATTENTION_COEFF = 5
# A Wav2Vec2-BERT Conformer layer is heavier than a plain transformer layer (macaron
# double-FFN + a convolution module), so the plain-transformer formula is scaled up. An
# estimate, not a measurement — ADR-0004 still requires verifying one real batch fits.
CONFORMER_ACTIVATION_FACTOR = 1.5

# Fixed VRAM overhead: CUDA context, cuDNN/attention workspaces, allocator fragmentation,
# the fp32 CTC log-prob buffer. A conservative flat estimate.
FIXED_OVERHEAD_GB = 1.5

# RTX 5060 Ti has 16 GB; the driver/display reserve leaves ~15 GB usable for the process.
TOTAL_VRAM_GB = 16.0
USABLE_VRAM_GB = 15.0

_BYTES_PER_GB = 1024 ** 3


def seconds_to_feature_frames(seconds: float) -> int:
    """20 ms feature-frame count spanning ``seconds`` (rounded to a whole frame)."""
    return round(seconds * FEATURE_FRAMES_PER_SECOND)


def feature_frames_to_seconds(feature_frames: int) -> float:
    """Seconds spanned by ``feature_frames`` 20 ms frames."""
    return feature_frames / FEATURE_FRAMES_PER_SECOND


# ---------------------------------------------------------------------------
# Duration histograms.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DurationHistogram:
    """The sorted durations of one population, binned for display and queryable by window.

    ``sorted_durations`` is the full population, ascending — kept so the exact fraction
    fitting *any* candidate window is answerable (:meth:`fraction_at_most`), not just the
    fixed display bins. ``edges`` are the ``len(counts) + 1`` bin boundaries in seconds; the
    final bin is the open-ended ``>= edges[-2]`` tail (a duration past the last edge still
    lands there). ``percentiles`` maps 50/90/95/99 to their durations.
    """

    label: str
    edges: tuple[float, ...]
    counts: tuple[int, ...]
    sorted_durations: tuple[float, ...]
    percentiles: dict[int, float]

    @property
    def count(self) -> int:
        return len(self.sorted_durations)

    @property
    def minimum(self) -> float:
        return self.sorted_durations[0]

    @property
    def maximum(self) -> float:
        return self.sorted_durations[-1]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.sorted_durations)

    def fraction_at_most(self, seconds: float) -> float:
        """Fraction of the population no longer than ``seconds`` — one window's coverage."""
        fits = sum(value <= seconds for value in self.sorted_durations)
        return fits / self.count

    def fraction_exceeding(self, seconds: float) -> float:
        """Fraction longer than ``seconds`` — the share that must be tiled across windows."""
        return 1.0 - self.fraction_at_most(seconds)


_HISTOGRAM_PERCENTILES = (50, 90, 95, 99)


def duration_histogram(
    label: str,
    durations: list[float],
    edges: tuple[float, ...],
) -> DurationHistogram:
    """Sort and bin ``durations``, keeping the population for exact window queries.

    Each duration lands in the last bin whose left edge it meets, so the final bin is the
    open-ended ``>= edges[-2]`` tail. Percentiles use the nearest-rank convention. Raises on
    an empty population — an empty duration manifest is a data-integrity failure, not a
    valid histogram.
    """
    if len(edges) < 2:
        raise ValueError(f"need at least two bin edges, got {edges!r}")
    if not durations:
        raise ValueError(f"no durations for population {label!r}")

    ordered = sorted(durations)
    counts = [0] * (len(edges) - 1)
    for value in ordered:
        counts[_bin_index(value, edges)] += 1

    return DurationHistogram(
        label=label,
        edges=tuple(edges),
        counts=tuple(counts),
        sorted_durations=tuple(ordered),
        percentiles={p: _nearest_rank(ordered, p) for p in _HISTOGRAM_PERCENTILES},
    )


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    """Index of the bin holding ``value``; the last bin is the open-ended tail."""
    for index in range(len(edges) - 2):
        if value < edges[index + 1]:
            return index
    return len(edges) - 2


def _nearest_rank(ordered: list[float], percentile: int) -> float:
    """Nearest-rank percentile of a pre-sorted list (percentile in ``[0, 100]``)."""
    rank = max(0, min(len(ordered) - 1, round(percentile / 100 * len(ordered)) - 1))
    return ordered[rank]


def load_whole_clip_durations(manifest_path: Path) -> list[float]:
    """Whole-recitation durations from a filter passing manifest (``ManifestRecord``).

    Each passing Tadabur clip is one whole, un-waqf-segmented recitation of an ayah; its
    ``ayah_duration_s`` is the duration of the 16 kHz waveform the filter scored.
    """
    return [record.ayah_duration_s for record in read_records(manifest_path)]


def load_segment_durations(manifest_path: Path) -> list[float]:
    """Waqf-segment durations from a ``tadabur.segment_score`` manifest.

    A segment row carries the ``[start_s, end_s)`` span of its parent clip that the segment
    covers; the segment's own duration is ``end_s - start_s``. Reads the JSONL directly
    (the segment manifest is a superset of ``ManifestRecord`` with the span fields).
    """
    durations: list[float] = []
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            durations.append(float(row["end_s"]) - float(row["start_s"]))
    return durations


# ---------------------------------------------------------------------------
# Per-window logit length + 16 GB memory estimate.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryEstimate:
    """Estimated peak training VRAM for one micro-batch of a given window length.

    All figures are GB. ``activation_gb`` is the dominant, seq-length-dependent term;
    ``weights_gb`` is the frozen bf16 backbone; ``overhead_gb`` is the fixed context/
    workspace estimate. ``checkpointed`` records whether per-layer activation checkpointing
    is assumed. ``fits_usable`` is ``total_gb <= USABLE_VRAM_GB`` — an estimate ADR-0004
    still requires confirming with one real batch.
    """

    feature_frames: int
    batch: int
    checkpointed: bool
    activation_gb: float
    weights_gb: float
    overhead_gb: float
    total_gb: float
    fits_usable: bool


def _activation_bytes_per_layer(feature_frames: int, batch: int) -> float:
    """Stored-activation bytes for one Conformer layer at this seq length and batch."""
    per_token = ACTIVATION_BASE_COEFF + (
        ACTIVATION_ATTENTION_COEFF * NUM_HEADS * feature_frames / HIDDEN_SIZE
    )
    plain = feature_frames * batch * HIDDEN_SIZE * per_token
    return plain * CONFORMER_ACTIVATION_FACTOR


def _activation_bytes(feature_frames: int, batch: int, checkpointed: bool) -> float:
    """Peak stored-activation bytes across the backbone.

    Without checkpointing every layer keeps its full activations. With per-layer
    checkpointing only each layer's input (``2·s·b·h`` bytes) is kept, and a single layer's
    full activations reappear during its backward recompute — the peak.
    """
    per_layer = _activation_bytes_per_layer(feature_frames, batch)
    if not checkpointed:
        return NUM_LAYERS * per_layer
    layer_input = BF16_BYTES * feature_frames * batch * HIDDEN_SIZE
    return NUM_LAYERS * layer_input + per_layer


def memory_estimate(
    feature_frames: int, batch: int, checkpointed: bool = True
) -> MemoryEstimate:
    """Estimate peak training VRAM for ``batch`` windows of ``feature_frames`` each."""
    if feature_frames <= 0 or batch <= 0:
        raise ValueError(f"feature_frames and batch must be positive, got {feature_frames}, {batch}")
    weights_gb = PARAM_COUNT * BF16_BYTES / _BYTES_PER_GB
    activation_gb = _activation_bytes(feature_frames, batch, checkpointed) / _BYTES_PER_GB
    total_gb = weights_gb + FIXED_OVERHEAD_GB + activation_gb
    return MemoryEstimate(
        feature_frames=feature_frames,
        batch=batch,
        checkpointed=checkpointed,
        activation_gb=activation_gb,
        weights_gb=weights_gb,
        overhead_gb=FIXED_OVERHEAD_GB,
        total_gb=total_gb,
        fits_usable=total_gb <= USABLE_VRAM_GB,
    )


def max_micro_batch(feature_frames: int, checkpointed: bool = True) -> int:
    """Largest micro-batch of ``feature_frames`` windows whose estimate fits usable VRAM.

    Returns 0 only if a single window cannot fit; with activation checkpointing even a
    whole-clip window fits at batch 1, so the deployed 5 s window is far from that limit and
    grad-accumulation recovers any effective batch size.
    """
    batch = 0
    while memory_estimate(feature_frames, batch + 1, checkpointed).fits_usable:
        batch += 1
    return batch


@dataclass(frozen=True)
class WindowCandidate:
    """One candidate window length with its measured logit length, coverage, and fit.

    ``student_frames`` is the post-adapter 40 ms CTC logit length (ADR-0004's
    ``target_len < frames`` is checked against this, not the 20 ms feature count).
    ``clip_coverage`` / ``segment_coverage`` are the fraction of whole clips / waqf segments
    that fit inside one window (``1 - exceedance``); the rest must be tiled.
    ``max_micro_batch`` is the fitting micro-batch under bf16 + activation checkpointing.
    """

    seconds: float
    feature_frames: int
    student_frames: int
    clip_coverage: float
    segment_coverage: float
    max_micro_batch: int
    deployed: bool


# Provisional per-clip cap: ~99th percentile of whole-clip durations, so the per-clip
# window count and the longest concatenated CTC target A2/#25 must preflight are bounded.
# Clips beyond it are flagged for A2, not silently truncated. Provisional — A2 owns it.
_PROVISIONAL_CAP_SECONDS = 40.0

_CANDIDATE_SECONDS = (3.0, DEPLOYED_WINDOW_SECONDS, 8.0, 10.0, 20.0, 40.0)


def window_candidates(
    clips: DurationHistogram,
    segments: DurationHistogram,
    candidate_seconds: tuple[float, ...] = _CANDIDATE_SECONDS,
) -> list[WindowCandidate]:
    """Build the candidate-window table across ``candidate_seconds``.

    The 5 s deployed window is flagged ``deployed`` — the ANE-fixed cap A2 is confirming;
    the longer candidates quantify the coverage-vs-memory trade that motivates keeping it.
    """
    return [
        WindowCandidate(
            seconds=seconds,
            feature_frames=(frames := seconds_to_feature_frames(seconds)),
            student_frames=muaalem_lattice_length(frames),
            clip_coverage=clips.fraction_at_most(seconds),
            segment_coverage=segments.fraction_at_most(seconds),
            max_micro_batch=max_micro_batch(frames),
            deployed=seconds == DEPLOYED_WINDOW_SECONDS,
        )
        for seconds in candidate_seconds
    ]


# ---------------------------------------------------------------------------
# Windowing policy options (edge ownership + stitch), and the A2 recommendation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyOption:
    """A candidate windowing policy: how windows overlap, own edges, and stitch.

    ``overlap_seconds`` is the window-to-window overlap (0 for a non-overlapping tiling);
    ``hop_seconds`` is ``window - overlap``. ``edge_ownership`` describes how a waqf or word
    straddling a window boundary is assigned to exactly one window; ``stitch`` describes how
    the per-window 40 ms silence frames are recombined into one clip-length track.
    """

    name: str
    window_seconds: float
    overlap_seconds: float
    edge_ownership: str
    stitch: str
    trade_off: str

    @property
    def hop_seconds(self) -> float:
        return self.window_seconds - self.overlap_seconds

    @property
    def windows_per_multiple(self) -> float:
        """Windows emitted per window-worth of audio — the relative compute vs a tiling."""
        return self.window_seconds / self.hop_seconds


# The waqf scorer-side post-processing reasons over pauses up to ~700 ms and snaps to word
# boundaries (ADR-0004); a boundary is only safely owned if it sits at least this far inside
# some window, clear of the padding-affected edge frames. 1 s (25 frames) comfortably
# exceeds the 700 ms window plus a short word, so a 1 s overlap is the smallest that keeps
# every interior stop away from a seam.
_OVERLAP_MARGIN_SECONDS = 1.0


def policy_options() -> list[PolicyOption]:
    """The candidate windowing policies for A2 to choose between (no decision made here)."""
    return [
        PolicyOption(
            name="non-overlapping tiling (current provisional default)",
            window_seconds=DEPLOYED_WINDOW_SECONDS,
            overlap_seconds=0.0,
            edge_ownership=(
                "A window owns exactly the audio in its [start, start+5s) span. A waqf pause "
                "or word straddling a tile boundary is split: its onset falls in the earlier "
                "window and its tail in the later one, and neither sees the whole pause. The "
                "boundary event is attributed to the window containing the frame where "
                "silence crosses the threshold (the earlier window for a pause that starts "
                "before the seam). Matches training.waqf_distill.WindowContract's default "
                "(hop == window)."
            ),
            stitch=(
                "Concatenate each window's 40 ms silence frames end-to-end in window order; "
                "student frame counts sum exactly to the clip lattice length because tiles "
                "do not overlap. Zero reconciliation, but a pause or qalqala closure sitting "
                "on a seam is seen half in each window and can be missed or double-counted."
            ),
            trade_off=(
                "Cheapest (one window pass per 5 s of audio) and already implemented, but the "
                "seam is a blind spot exactly where waqf detection matters."
            ),
        ),
        PolicyOption(
            name="center-trusted overlap",
            window_seconds=DEPLOYED_WINDOW_SECONDS,
            overlap_seconds=_OVERLAP_MARGIN_SECONDS,
            edge_ownership=(
                "Windows step by a 4 s hop with 1 s overlap, so every clip position (except "
                "the outermost 0.5 s) is interior to at least one window. The overlap region "
                "is owned by the window whose center is nearer the frame — each window is "
                "authoritative only over its central [0.5s, 4.5s) band, discarding its outer "
                "0.5 s. A 1 s overlap exceeds the 700 ms waqf post-processing window plus a "
                "short word, so no interior stop is ever trapped in a discarded edge."
            ),
            stitch=(
                "For each 40 ms frame keep the silence posterior from the window that owns it "
                "(nearer center); no averaging, so a boundary is graded by the window that "
                "saw it in full context. Requires the frozen hop/overlap and the 2:1 pooling "
                "to be applied identically in train, eval, and export (ADR-0004)."
            ),
            trade_off=(
                "~1.25x the window passes of a tiling (5 s / 4 s hop) — negligible at 5 s "
                "windows, where a micro-batch of dozens still fits 16 GB — in exchange for no "
                "seam blind spot. The overlap size is the one free parameter A2 must set."
            ),
        ),
    ]


def recommendation(candidates: list[WindowCandidate]) -> str:
    """The provisional recommendation this analysis hands to A2 (not a product decision)."""
    deployed = next(c for c in candidates if c.deployed)
    clip_exceed = 1.0 - deployed.clip_coverage
    seg_exceed = 1.0 - deployed.segment_coverage
    cap_frames = seconds_to_feature_frames(_PROVISIONAL_CAP_SECONDS)
    cap_windows = -(-cap_frames // deployed.feature_frames)  # ceil
    return (
        f"Keep the deployed **{deployed.seconds:g} s** window "
        f"({deployed.feature_frames} feature frames -> {deployed.student_frames} frames on "
        f"the 40 ms lattice). It is **fixed by the ANE**, which requires a static input shape "
        f"(`ml-model-transformation.md`), so training must match inference regardless of what "
        f"the duration data alone might suggest. Training memory is **not** the binding "
        f"constraint at this length: an estimated micro-batch of {deployed.max_micro_batch} "
        f"windows fits {USABLE_VRAM_GB:g} GB usable under bf16 + activation checkpointing, and "
        f"grad-accumulation recovers any effective batch. "
        f"Because ~{clip_exceed:.0%} of whole recitations and ~{seg_exceed:.0%} of waqf "
        f"segments exceed one window, multi-window stitching is the **norm**, so the real "
        f"decision A2 owns is the **overlap / edge-ownership / stitch policy, not the window "
        f"length**. Provisional recommendation: move from the current non-overlapping tiling "
        f"to the **center-trusted 1 s overlap**, so no interior waqf lands on a seam blind "
        f"spot; the exact overlap is A2's to freeze. The current code default "
        f"(`training.waqf_distill.WindowContract`, hop == window) stays valid until then. "
        f"A **provisional cap** of ~{_PROVISIONAL_CAP_SECONDS:g} s "
        f"(~99th percentile of whole clips, {cap_frames} feature frames, ~{cap_windows} "
        f"windows) bounds the per-clip window count and the longest CTC target A2/#25 must "
        f"preflight; clips beyond it are flagged for A2 rather than silently truncated."
    )


# ---------------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------------

_CLIP_EDGES = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 90.0)
_SEGMENT_EDGES = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0)
_DISPLAY_WINDOWS = (5.0, 10.0, 15.0, 20.0)
_BAR_WIDTH = 40


def _histogram_block(hist: DurationHistogram) -> str:
    peak = max(hist.counts) or 1
    lines = [f"#### {hist.label} (n={hist.count})", "", "```"]
    for index, count in enumerate(hist.counts):
        low = hist.edges[index]
        high = hist.edges[index + 1]
        span = f"[{low:>4.0f},{high:>4.0f})" if index < len(hist.counts) - 1 else f">= {low:>3.0f}s   "
        bar = "#" * round(_BAR_WIDTH * count / peak)
        lines.append(f"{span}  {count:>6d}  {bar}")
    lines.append("```")
    lines.append("")
    percentiles = "  ".join(f"p{p}={value:.1f}s" for p, value in hist.percentiles.items())
    lines.append(f"min={hist.minimum:.1f}s  {percentiles}  max={hist.maximum:.1f}s  mean={hist.mean:.1f}s")
    lines.append("")
    exceed = "  ".join(
        f">{window:g}s: {hist.fraction_exceeding(window):.0%}" for window in _DISPLAY_WINDOWS
    )
    lines.append(f"share exceeding one window — {exceed}")
    return "\n".join(lines)


def _candidate_table(candidates: list[WindowCandidate]) -> str:
    header = (
        "| window | feat frames (20 ms) | logit frames (40 ms) | whole-clip coverage "
        "| waqf-seg coverage | max micro-batch @15 GB |\n"
        "|---|---|---|---|---|---|"
    )
    rows = [
        f"| {c.seconds:g} s{' **(deployed)**' if c.deployed else ''} | {c.feature_frames} "
        f"| {c.student_frames} | {c.clip_coverage:.0%} | {c.segment_coverage:.0%} "
        f"| {c.max_micro_batch if c.max_micro_batch else 'does not fit'} |"
        for c in candidates
    ]
    return "\n".join([header, *rows])


def _memory_detail(feature_frames: int) -> str:
    single = memory_estimate(feature_frames, 1)
    full = memory_estimate(feature_frames, 1, checkpointed=False)
    return (
        f"Per-window estimate at the {feature_frames_to_seconds(feature_frames):g} s window "
        f"(bf16): frozen backbone {single.weights_gb:.1f} GB + fixed overhead "
        f"{single.overhead_gb:.1f} GB + {single.activation_gb:.2f} GB activations per window "
        f"with per-layer activation checkpointing ({full.activation_gb:.2f} GB without). "
        f"Activation memory grows ~quadratically with window length (the attention-score "
        f"buffer), so a whole-clip window OOMs without checkpointing; but at the fixed 5 s "
        f"window memory is comfortable either way. The window length is therefore pinned by "
        f"the ANE fixed-shape inference contract, not by the training memory budget."
    )


def _policy_block(options: list[PolicyOption]) -> str:
    blocks = []
    for option in options:
        blocks.append(
            f"### {option.name}\n\n"
            f"- **Window / overlap**: {option.window_seconds:g} s window, "
            f"{option.overlap_seconds:g} s overlap "
            f"({option.hop_seconds:g} s hop, {option.windows_per_multiple:.2f}x window passes).\n"
            f"- **Edge ownership**: {option.edge_ownership}\n"
            f"- **Stitch**: {option.stitch}\n"
            f"- **Trade-off**: {option.trade_off}"
        )
    return "\n\n".join(blocks)


def render_report(
    clips: DurationHistogram,
    segments: DurationHistogram,
    candidates: list[WindowCandidate],
    options: list[PolicyOption],
) -> str:
    """Render the full measured envelope as the markdown A2 (#24) consumes."""
    deployed = next(c for c in candidates if c.deployed)
    return "\n".join(
        [
            "# Fixed-window duration/memory envelope (P7.A1, input to A2 #24)",
            "",
            "> Generated by `python -m training.window_envelope` over the filter passing",
            "> manifest and a segment manifest (both gitignored). Regenerate to refresh the",
            "> measured numbers. This slice **makes no product decision** — it measures the",
            "> envelope and lays out options for A2 (#24, the HITL windowing-contract freeze).",
            "",
            "## 1. Duration histograms",
            "",
            "The deployed pipeline runs fixed **5 s** windows at a **40 ms** lattice (ADR-0004,",
            "`ml-model-transformation.md`). A recitation or waqf segment longer than one window",
            "must be tiled across windows, so the tail of these distributions is what makes the",
            "windowing/overlap/stitch contract load-bearing rather than incidental.",
            "",
            _histogram_block(clips),
            "",
            _histogram_block(segments),
            "",
            "## 2. Per-window logit length + 16 GB fit",
            "",
            f"The post-adapter 40 ms logit length is `muaalem_lattice_length(feature_frames)` "
            f"(the single stride-2 adapter conv); the deployed 250-frame window yields "
            f"{deployed.student_frames} logit frames. ADR-0004's `target_len < frames` preflight "
            f"is checked against this 40 ms length, not the 20 ms feature count.",
            "",
            _candidate_table(candidates),
            "",
            _memory_detail(deployed.feature_frames),
            "",
            f"Budget: {TOTAL_VRAM_GB:g} GB physical (RTX 5060 Ti), ~{USABLE_VRAM_GB:g} GB usable. "
            "These are **estimates** from the activation-recomputation formula (Korthikanti et "
            "al. 2022) with a Conformer factor; ADR-0004 still requires verifying one real batch "
            "fits before committing.",
            "",
            "## 3. Candidate windowing policies (edge ownership + stitch)",
            "",
            _policy_block(options),
            "",
            "## 4. Recommendation for A2",
            "",
            recommendation(candidates),
        ]
    )


def build_report(whole_manifest: Path, segment_manifest: Path) -> str:
    """Load both manifests and render the full envelope report."""
    clips = duration_histogram(
        "Whole recitations (un-waqf-segmented clips)",
        load_whole_clip_durations(whole_manifest),
        _CLIP_EDGES,
    )
    segments = duration_histogram(
        "Waqf segments",
        load_segment_durations(segment_manifest),
        _SEGMENT_EDGES,
    )
    candidates = window_candidates(clips, segments)
    return render_report(clips, segments, candidates, policy_options())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--whole-manifest",
        type=Path,
        required=True,
        help="Filter passing manifest (JSONL of ManifestRecord) — whole-clip durations.",
    )
    parser.add_argument(
        "--segment-manifest",
        type=Path,
        required=True,
        help="tadabur.segment_score manifest (JSONL with start_s/end_s) — waqf-segment durations.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the markdown report here (default: stdout).",
    )
    args = parser.parse_args()

    report = build_report(args.whole_manifest, args.segment_manifest)
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"Wrote {args.out} ({len(report)} chars).")


if __name__ == "__main__":
    main()
