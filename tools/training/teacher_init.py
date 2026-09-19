"""Start the student from the teacher's weights instead of from noise.

Every distillation run so far began at random init and spent its first ~2,000 steps escaping
an all-blank basin. The teacher already contains a solution; the only reason the student
cannot simply load it is width -- 1024 hidden against 384, 4096 FFN against 1536, 16 attention
heads against 6. This module closes that gap by **structured selection**: it keeps a subset of
the teacher's channels, heads and FFN units and copies their weights verbatim, producing a
student that is a genuine sub-network of the teacher rather than an approximation of one.

**Selection, not projection, and the reason is specific to this backbone.** A PCA rotation
``P`` of the residual stream preserves more variance per retained dimension, and it is what
the literature reaches for first. It is the wrong tool here, three times over. LayerNorm does
not commute with a rotation (``LN(P^T x) != P^T LN(x)``), so every one of the ~130 norm sites
becomes an approximation. The residual stream is added to in every block, so a rotation has to
be globally consistent or the adds stop meaning anything. And -- decisively -- transformers'
``Wav2Vec2BertSelfAttention`` applies the rotary embedding to the **hidden states, before**
``linear_q``/``linear_k``, in ``num_heads`` contiguous blocks of the *input* space; after an
arbitrary rotation those blocks are groups of unrelated directions. Selection keeps every
student channel equal to one teacher channel, so LayerNorm gains index-select exactly, the
residual stream stays interpretable, and the transplant is verifiable by reading it.

**What the LayerNorms still need.** Selection is exact per coordinate but the *normalisation*
is not: the student normalises over 384 dimensions and the teacher over 1024, and because the
selected channels are the high-activation ones, the student's standard deviation is
systematically larger. Left alone that scales every norm output down and the transplant is
worthless. :func:`layernorm_correction` measures the two moments on the calibration batch and
folds them into the copied affine, so ``LN_384(x[sel]) * y' + b'`` reproduces
``(LN_1024(x) * y + b)[sel]`` to first order. Two scalars per site, one forward pass.

**What is selected, and by what.** All three criteria are the *measured contribution to the
residual stream*, not a weight norm -- a large weight on a dead channel buys nothing:

* residual channels, by mean absolute activation pooled over every layer's output, each layer
  standardised first so late-layer scale does not own the ranking;
* attention heads, by the norm of the vector each head actually adds
  (``||W_out[:, head] @ head_output||``), measured per layer;
* FFN units, by ``E[|activation|] * ||W_down[:, unit]||`` -- the Minitron criterion, and again
  the size of the contribution rather than of the parameter.

**Query and key are the interesting question, so they are a flag.** The teacher's ``linear_q``
and ``linear_k`` were fitted with a ``relative_key`` positional bias -- a separate learned
``q . E[clamp(j - i, -64, 8)]`` term that rotary has no slot for -- and on *unrotated* inputs.
Copying them transfers a content matcher into a model whose positional mechanism is different
and whose q/k inputs are pre-rotated. That could help (the content geometry is still
information) or hurt (a confidently wrong attention pattern is worse to unlearn than a diffuse
one). ``--qk`` runs it as an experiment: ``random`` leaves them at init, ``copy`` transplants
them, ``damp`` transplants them scaled toward uniform attention. Value and output projections,
and the whole adapter attention -- which has **no** positional embedding at all
(``is_adapter_attention=True``) -- always copy.

The output is a checkpoint in ``training.distill_train``'s format, so it is consumed by
``--init-from`` with no change to the training loop.

Usage::

    # Build the initialisation and report how far it already is from the teacher
    python -m training.teacher_init --preset h384 \\
        --audio-root ../tadabur/audit_run/clips_v2 \\
        --out runs/h384_tinit/checkpoint.pt --qk random

    # Then train from it
    python -m training.distill_train --preset h384 --stream-shards ... \\
        --init-from runs/h384_tinit/checkpoint.pt --learning-rate 5e-5

Linux + CUDA.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from training.distill_student import (
    PRESETS,
    TEACHER_HIDDEN_SIZE,
    TEACHER_MODEL_ID,
    TEACHER_NUM_HEADS,
    StudentSpec,
    build_student,
)

# Head dimension is 64 for the teacher (1024/16) and for every student preset, which is what
# makes head *selection* possible at all: a selected head is copied whole, with its internal
# geometry -- the q.k scaling, the 64-dim subspace -- untouched.
TEACHER_HEAD_DIM = TEACHER_HIDDEN_SIZE // TEACHER_NUM_HEADS

# Tokens sampled per forward when scoring heads, which is the one criterion that needs a
# matmul inside the hook. 4096 tokens is ~16 windows' worth and the ranking is stable well
# below that; the full 250-frame batch would cost 24x more for no change in the ordering.
HEAD_SCORE_TOKENS = 4096

LAYER_INDEX = re.compile(r"\.layers\.(\d+)\.")


def _layer_key(name: str) -> str:
    """The module path a per-layer statistic is filed under."""
    return name


def _teacher_forward(teacher, features, **kwargs):
    """One teacher forward under the same autocast the training step uses.

    ``load_teacher`` returns a **bf16** teacher, so calling it on fp32 features outside
    autocast raises a dtype mismatch inside the first Linear. Every forward here goes through
    this, which also keeps the activations these statistics are measured on identical to the
    ones training actually produces.
    """
    if features.is_cuda:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return teacher(features, **kwargs)
    return teacher(features, **kwargs)


def load_teacher_weights(model_id: str = TEACHER_MODEL_ID):
    """The teacher in **fp32 on the CPU**, for the weight copy only.

    ``training.distill_train.load_teacher`` casts to bf16, which is right for running the
    teacher and wrong for copying it: bf16 keeps 8 mantissa bits, so a transplant taken from
    it starts the student on rounded weights for no reason. The statistics passes use the
    bf16 GPU teacher; only the copy uses this one, and it never touches the GPU.
    """
    from tadabur.muaalem import (
        Wav2Vec2BertForMultilevelCTC,
        Wav2Vec2BertForMultilevelCTCConfig,
    )

    config = Wav2Vec2BertForMultilevelCTCConfig.from_pretrained(model_id)
    weights = Wav2Vec2BertForMultilevelCTC.from_pretrained(model_id, config=config)
    weights.eval()
    return weights


@dataclass
class CalibrationStats:
    """Everything measured from the teacher on real audio, before anything is copied.

    Collected in two passes because the LayerNorm moments depend on the channel selection,
    which the first pass produces. Both passes are forward-only and cost seconds.
    """

    residual_importance: torch.Tensor
    head_importance: dict[str, torch.Tensor] = field(default_factory=dict)
    ffn_importance: dict[str, torch.Tensor] = field(default_factory=dict)
    conv_importance: dict[str, torch.Tensor] = field(default_factory=dict)
    layernorm_moments: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )

    def captured_variance_share(self, selected: torch.Tensor) -> float:
        """Share of the pooled residual importance the selected channels carry.

        Reported because it is the one number that says whether selection was a reasonable
        choice against a rotation: at 384 of 1024 channels an even spread would give 0.375,
        and anything well above that is concentration the selection is exploiting.
        """
        total = float(self.residual_importance.sum())
        return float(self.residual_importance[selected].sum()) / max(1e-12, total)


def _is_hidden_layernorm(module, hidden_size: int) -> bool:
    return (
        isinstance(module, torch.nn.LayerNorm)
        and tuple(module.normalized_shape) == (hidden_size,)
    )


@torch.no_grad()
def collect_importance(teacher, batches: list[torch.Tensor]) -> CalibrationStats:
    """Pass one: what each residual channel, attention head and FFN unit actually contributes.

    Hooks accumulate into fixed-size vectors rather than storing activations: the FFN sites
    alone would be 48 tensors of ``(B, 250, 4096)``, which is tens of gigabytes across a
    calibration batch, and none of it is needed beyond its column sums.
    """
    device = next(teacher.parameters()).device
    hidden = teacher.config.hidden_size
    num_heads = teacher.config.num_attention_heads
    head_dim = hidden // num_heads
    intermediate = teacher.config.intermediate_size
    residual = torch.zeros(hidden, device=device, dtype=torch.float32)
    head_scores: dict[str, torch.Tensor] = {}
    ffn_scores: dict[str, torch.Tensor] = {}
    conv_scores: dict[str, torch.Tensor] = {}
    handles = []

    def attention_hook(name):
        def hook(module, args):
            # ``args[0]`` is the concatenation of every head's output, the exact tensor
            # ``linear_out`` mixes back into the residual stream.
            flat = args[0].reshape(-1, hidden).float()
            step = max(1, flat.shape[0] // HEAD_SCORE_TOKENS)
            flat = flat[::step]
            weight = module.weight.float()
            scores = head_scores.setdefault(
                name, torch.zeros(num_heads, device=device, dtype=torch.float32)
            )
            for head in range(num_heads):
                span = slice(head * head_dim, (head + 1) * head_dim)
                contribution = flat[:, span] @ weight[:, span].T
                scores[head] += contribution.norm(dim=-1).mean()
        return hook

    def ffn_hook(name):
        def hook(module, args):
            # ``args[0]`` is post-activation, so |a_u| * ||W_down[:, u]|| is the magnitude of
            # unit u's contribution -- not a weight norm dressed up as importance.
            activations = args[0].reshape(-1, intermediate).float()
            column_norms = module.weight.float().norm(dim=0)
            scores = ffn_scores.setdefault(
                name,
                torch.zeros(intermediate, device=device, dtype=torch.float32),
            )
            scores += activations.abs().mean(dim=0) * column_norms
        return hook

    def conv_hook(name):
        def hook(module, args):
            # The conv module's internal channels are NOT the residual stream. They are a
            # separate learned space that happens to have the same width, produced by the
            # GLU and consumed by this 1x1 convolution. Ranking them by residual importance
            # -- which is what a single global selection would do -- keeps an arbitrary
            # subset of them. Input here is channels-first: (batch, channel, time).
            activations = args[0].float()
            column_norms = module.weight.float().squeeze(-1).norm(dim=0)
            scores = conv_scores.setdefault(
                name,
                torch.zeros(activations.shape[1], device=device, dtype=torch.float32),
            )
            scores += activations.abs().mean(dim=(0, 2)) * column_norms
        return hook

    for name, module in teacher.named_modules():
        if name.endswith("self_attn.linear_out"):
            handles.append(module.register_forward_pre_hook(attention_hook(name)))
        elif name.endswith(".output_dense"):
            handles.append(module.register_forward_pre_hook(ffn_hook(name)))
        elif name.endswith("conv_module.pointwise_conv2"):
            handles.append(module.register_forward_pre_hook(conv_hook(name)))

    try:
        for features in batches:
            outputs = _teacher_forward(
                teacher, features, output_hidden_states=True, return_dict=True
            )
            for state in outputs["hidden_states"]:
                state = state.float()
                # Standardise per layer before pooling: without it the deepest layers' scale
                # decides the ranking for all 24.
                scale = state.pow(2).mean().sqrt().clamp_min(1e-6)
                residual += state.abs().mean(dim=(0, 1)) / scale
    finally:
        for handle in handles:
            handle.remove()

    return CalibrationStats(
        residual_importance=residual.cpu(),
        head_importance={k: v.cpu() for k, v in head_scores.items()},
        ffn_importance={k: v.cpu() for k, v in ffn_scores.items()},
        conv_importance={k: v.cpu() for k, v in conv_scores.items()},
    )


@torch.no_grad()
def collect_layernorm_moments(
    teacher, batches: list[torch.Tensor], selection: "Selection | torch.Tensor"
) -> dict[str, tuple[float, float, float, float]]:
    """Pass two: each LayerNorm's mean and sd over all 1024 dims and over the kept 384.

    These four numbers per site are what :func:`layernorm_correction` folds into the copied
    affine. Without them the student's norms are systematically mis-scaled -- the selected
    channels are the loud ones, so their standard deviation is the larger of the two -- and
    everything downstream of every norm is wrong by a constant factor.
    """
    moments: dict[str, list[float]] = {}
    handles = []
    device = next(teacher.parameters()).device
    hidden = teacher.config.hidden_size

    def index_for(site: str) -> torch.Tensor:
        chosen = (
            selection.index_for_layernorm(site)
            if isinstance(selection, Selection)
            else selection
        )
        return chosen.to(device)

    def hook(name):
        index = index_for(name)

        def capture(module, args):
            x = args[0].float()
            full_mean = x.mean(dim=-1)
            full_sd = x.var(dim=-1, unbiased=False).add(module.eps).sqrt()
            kept = x.index_select(-1, index)
            kept_mean = kept.mean(dim=-1)
            kept_sd = kept.var(dim=-1, unbiased=False).add(module.eps).sqrt()
            totals = moments.setdefault(name, [0.0, 0.0, 0.0, 0.0, 0.0])
            totals[0] += float(full_mean.mean())
            totals[1] += float(full_sd.mean())
            totals[2] += float(kept_mean.mean())
            totals[3] += float(kept_sd.mean())
            totals[4] += 1.0
        return capture

    for name, module in teacher.named_modules():
        if _is_hidden_layernorm(module, hidden):
            handles.append(module.register_forward_pre_hook(hook(name)))

    try:
        for features in batches:
            _teacher_forward(teacher, features, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    return {
        name: (
            totals[0] / totals[4],
            totals[1] / totals[4],
            totals[2] / totals[4],
            totals[3] / totals[4],
        )
        for name, totals in moments.items()
    }


@torch.no_grad()
def collect_branch_gains(
    teacher, batches: list[torch.Tensor], selection: "Selection"
) -> dict[str, float]:
    """Pass three: how much each kept branch must be scaled to replace the whole one.

    A Conformer block adds four branch outputs into the residual stream, and each is a *sum*
    over units -- 16 attention heads, 4096 FFN units, 1024 conv channels. Keeping 6, 1536 and
    384 of them keeps a fraction of that sum, and the received wisdom is to scale what
    survives by ``16/6`` to make up the difference.

    This measures the right number instead: the least-squares optimum
    ``a = <kept, full> / <kept, kept>``, restricted to the residual channels the student
    actually has, fitted per branch per layer on the calibration batch and folded into the
    copied output weight.

    **Read the value, not just apply it.** The closed form is ``1 + <kept, dropped>/<kept,
    kept>``, so it is 1 exactly when the dropped units contribute *orthogonally* to the kept
    ones -- and then no rescaling recovers anything, because what is missing is a direction
    the student cannot express, not a magnitude. A gain near ``16/6`` would mean the units
    are largely redundant and the branch really was just quieter. Which regime this teacher
    is in is an empirical question the printed median answers, and applying ``16/6`` blind
    would amplify noise in the first regime while looking like a principled fix.
    """
    device = next(teacher.parameters()).device
    residual = selection.residual.to(device)
    numerator: dict[str, float] = {}
    denominator: dict[str, float] = {}
    handles = []

    def accumulate(name: str, kept: torch.Tensor, full: torch.Tensor) -> None:
        kept = kept.index_select(-1, residual)
        full = full.index_select(-1, residual)
        numerator[name] = numerator.get(name, 0.0) + float((kept * full).sum())
        denominator[name] = denominator.get(name, 0.0) + float((kept * kept).sum())

    def linear_hook(name, columns):
        def hook(module, args):
            activations = args[0].reshape(-1, args[0].shape[-1]).float()
            weight = module.weight.float()
            index = columns.to(device)
            accumulate(
                name,
                activations.index_select(-1, index) @ weight.index_select(1, index).T,
                activations @ weight.T,
            )
        return hook

    def conv_hook(name, channels):
        def hook(module, args):
            # (batch, channel, time) -- transposed to match the linear case.
            activations = args[0].transpose(1, 2).reshape(-1, args[0].shape[1]).float()
            weight = module.weight.float().squeeze(-1)
            index = channels.to(device)
            accumulate(
                name,
                activations.index_select(-1, index) @ weight.index_select(1, index).T,
                activations @ weight.T,
            )
        return hook

    for name, module in teacher.named_modules():
        if name in selection.heads:
            handles.append(
                module.register_forward_pre_hook(linear_hook(name, selection.heads[name]))
            )
        elif name in selection.ffn:
            handles.append(
                module.register_forward_pre_hook(linear_hook(name, selection.ffn[name]))
            )
        elif name in selection.conv:
            handles.append(
                module.register_forward_pre_hook(conv_hook(name, selection.conv[name]))
            )

    try:
        for features in batches:
            _teacher_forward(teacher, features, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    return {
        name: numerator[name] / denominator[name]
        for name in numerator
        if denominator.get(name, 0.0) > 1e-9
    }


def layernorm_correction(
    gamma: torch.Tensor,
    beta: torch.Tensor,
    selected: torch.Tensor,
    moments: tuple[float, float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The copied affine that makes a 384-dim norm reproduce the 1024-dim one on kept coords.

    Teacher coordinate c produces ``g_c (x_c - m_f) / s_f + b_c``; the student produces
    ``g'_c (x_c - m_k) / s_k + b'_c`` over the kept subset. Matching the linear term gives
    ``g' = g * s_k / s_f`` and then matching the constant gives
    ``b' = b + g * (m_k - m_f) / s_f``. Exact to the extent the per-token moments are
    replaced by their means, which is why they are measured rather than assumed.
    """
    full_mean, full_sd, kept_mean, kept_sd = moments
    scale = kept_sd / max(1e-6, full_sd)
    kept_gamma = gamma.index_select(0, selected)
    kept_beta = beta.index_select(0, selected)
    return (
        kept_gamma * scale,
        kept_beta + kept_gamma * (kept_mean - full_mean) / max(1e-6, full_sd),
    )


def select_top(scores: torch.Tensor, count: int) -> torch.Tensor:
    """The ``count`` highest-scoring indices, returned in ascending index order.

    Sorted by index, not by score: a selection that reorders channels would scramble the
    correspondence between the residual stream's position and the rotary embedding's
    ``head_size`` blocks, and would make the transplant unreadable against the teacher.
    """
    if count > scores.numel():
        raise ValueError(f"cannot select {count} of {scores.numel()}")
    return torch.sort(torch.topk(scores, count).indices).values


def select_heads(
    scores: torch.Tensor, num_heads: int, head_dim: int = TEACHER_HEAD_DIM
) -> torch.Tensor:
    """Channel indices of the ``num_heads`` most important heads, as one flat index vector.

    A head is kept whole. That is the point of selecting heads rather than projecting the
    query/key/value spaces: inside a head the 64-dimensional geometry the teacher learned --
    which directions the dot product rewards, at what scale -- survives untouched, whereas any
    mixing across heads destroys it before the softmax ever sees it.
    """
    chosen = torch.sort(torch.topk(scores, num_heads).indices).values
    return torch.cat(
        [
            torch.arange(head * head_dim, (head + 1) * head_dim)
            for head in chosen.tolist()
        ]
    )


@dataclass(frozen=True)
class Selection:
    """Which of the teacher's channels, heads and FFN units the student keeps.

    ``heads`` and ``ffn`` are keyed by the **module path of the consumer** -- the
    ``linear_out`` and ``output_dense`` whose inputs were measured -- so a parameter can look
    up its own layer's selection without an index arithmetic convention that could drift
    between the two passes.
    """

    residual: torch.Tensor
    heads: dict[str, torch.Tensor]
    ffn: dict[str, torch.Tensor]
    conv: dict[str, torch.Tensor] = field(default_factory=dict)

    def index_for_layernorm(self, site: str) -> torch.Tensor:
        """Which channels a LayerNorm site normalises over.

        Almost every hidden-width norm sits on the residual stream. ``depthwise_layer_norm``
        does not -- it normalises the conv module's internal channels, so measuring its
        moments over the residual selection compares two different spaces and produces a
        correction that is worse than none.
        """
        if site.endswith("conv_module.depthwise_layer_norm"):
            consumer = site.replace("depthwise_layer_norm", "pointwise_conv2")
            if consumer in self.conv:
                return self.conv[consumer]
        return self.residual


def choose(
    spec: StudentSpec, stats: CalibrationStats, head_dim: int = TEACHER_HEAD_DIM
) -> Selection:
    """Rank and cut every axis to the student's shape.

    ``head_dim`` must match on both sides -- it is 64 for the teacher and for every preset --
    or a "selected head" is not a head. Passed explicitly so the transplant can be exercised
    on small models in a test rather than only at the real sizes.
    """
    if spec.hidden_size % spec.num_attention_heads != 0 or (
        spec.hidden_size // spec.num_attention_heads != head_dim
    ):
        raise ValueError(
            f"{spec.name} has head_dim "
            f"{spec.hidden_size // spec.num_attention_heads}, teacher has {head_dim}; "
            f"head selection requires them to be equal"
        )
    return Selection(
        residual=select_top(stats.residual_importance, spec.hidden_size),
        heads={
            name: select_heads(scores, spec.num_attention_heads, head_dim)
            for name, scores in stats.head_importance.items()
        },
        ffn={
            name: select_top(scores, spec.intermediate_size)
            for name, scores in stats.ffn_importance.items()
        },
        conv={
            name: select_top(scores, spec.hidden_size)
            for name, scores in stats.conv_importance.items()
        },
    )


def _consumer_path(parameter_name: str, leaf: str, consumer: str) -> str:
    """``a.b.linear_q.weight`` -> ``a.b.linear_out`` etc: the module whose input was scored."""
    return parameter_name[: parameter_name.rindex(f".{leaf}.")] + f".{consumer}"


def _glu_rows(residual: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """Row indices of a GLU pointwise conv: the value half and its paired gate half.

    ``nn.GLU(dim=1)`` splits the output channels down the middle, so output channel ``c`` is
    gated by channel ``c + hidden``. Selecting the two halves independently would pair each
    kept value channel with somebody else's gate, which is a silent way to produce a
    plausible-looking model that computes nothing the teacher computes.
    """
    return torch.cat([residual, residual + hidden_size])


@torch.no_grad()
def transplant(
    student,
    teacher,
    selection: Selection,
    moments: dict[str, tuple[float, float, float, float]],
    qk_mode: str = "random",
    qk_scale: float = 0.5,
    branch_gains: dict[str, float] | None = None,
) -> dict:
    """Copy the selected sub-network of ``teacher`` into ``student`` in place.

    Every student parameter is matched to the identically-named teacher parameter and sliced
    by the rule its role demands. A parameter with no teacher counterpart (the student has
    none) or one this function declines to copy (``linear_q``/``linear_k`` under
    ``--qk random``) keeps its random initialisation and is reported, so "what did not get
    copied" is an output rather than something to be inferred by reading the code.
    """
    if qk_mode not in ("random", "copy", "damp"):
        raise ValueError(f"unknown --qk mode {qk_mode!r}")
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    residual = selection.residual
    hidden = teacher.config.hidden_size
    gains = branch_gains or {}
    copied: list[str] = []
    skipped: list[str] = []

    def source(name: str) -> torch.Tensor:
        return teacher_state[name].detach().float().cpu()

    for name, target in student_state.items():
        if name not in teacher_state:
            skipped.append(f"{name} (no teacher counterpart)")
            continue

        value: torch.Tensor | None = None

        # --- the head and the 160-dim front end copy without any selection on one axis ---
        if name.startswith("level_to_lm_head.phonemes"):
            weight = source(name)
            value = weight.index_select(1, residual) if weight.dim() == 2 else weight
        elif name.startswith("wav2vec2_bert.feature_projection.layer_norm"):
            value = source(name)
        elif name == "wav2vec2_bert.feature_projection.projection.weight":
            value = source(name).index_select(0, residual)
        elif name == "wav2vec2_bert.feature_projection.projection.bias":
            value = source(name).index_select(0, residual)

        # --- LayerNorms over the hidden dimension, with the measured scale correction ---
        # Matched on the module name, not on a "_layer_norm" suffix: the conv module's is
        # plain ``conv_module.layer_norm``, and a suffix test silently misses it and leaves
        # 24 norms at random init inside an otherwise transplanted network.
        elif name.rsplit(".", 1)[0].endswith("layer_norm"):
            site = name.rsplit(".", 1)[0]
            gamma, beta = source(f"{site}.weight"), source(f"{site}.bias")
            kept = selection.index_for_layernorm(site)
            if site in moments:
                gamma, beta = layernorm_correction(gamma, beta, kept, moments[site])
            else:
                gamma = gamma.index_select(0, kept)
                beta = beta.index_select(0, kept)
            value = gamma if name.endswith(".weight") else beta

        # --- attention ---
        elif ".self_attn.linear_" in name:
            leaf = name.split(".self_attn.linear_")[1].split(".")[0]
            consumer = _consumer_path(name, f"linear_{leaf}", "linear_out")
            heads = selection.heads[consumer]
            is_query_or_key = leaf in ("q", "k")
            adapter_attention = ".adapter." in name
            if is_query_or_key and not adapter_attention and qk_mode == "random":
                # The teacher fitted these with a relative_key positional bias and unrotated
                # inputs; see the module docstring. Left at init on purpose.
                skipped.append(f"{name} (--qk random)")
                continue
            weight = source(name)
            if leaf == "out":
                if weight.dim() == 2:
                    # Six of sixteen heads survive; the fitted gain is the least-squares best
                    # rescaling of what is left (see collect_branch_gains -- it is ~1.0 when
                    # the dropped heads were contributing orthogonally).
                    value = weight.index_select(0, residual).index_select(1, heads)
                    value = value * gains.get(consumer, 1.0)
                else:
                    value = weight.index_select(0, residual)
            else:
                value = (
                    weight.index_select(0, heads).index_select(1, residual)
                    if weight.dim() == 2
                    else weight.index_select(0, heads)
                )
            if is_query_or_key and not adapter_attention and qk_mode == "damp":
                # Scaling q and k toward zero flattens the initial attention distribution, so
                # a wrong copied pattern is a weak prior the optimiser can move rather than a
                # confident one it must first unlearn.
                value = value * qk_scale

        # --- feed-forward ---
        elif ".intermediate_dense." in name:
            units = selection.ffn[_consumer_path(name, "intermediate_dense", "output_dense")]
            weight = source(name)
            value = (
                weight.index_select(0, units).index_select(1, residual)
                if weight.dim() == 2
                else weight.index_select(0, units)
            )
        elif ".output_dense." in name:
            consumer = name.rsplit(".", 2)[0] + ".output_dense"
            units = selection.ffn[consumer]
            weight = source(name)
            if weight.dim() == 2:
                value = weight.index_select(0, residual).index_select(1, units)
                value = value * gains.get(consumer, 1.0)
            else:
                value = weight.index_select(0, residual)

        # --- convolutions ---
        elif name.endswith("pointwise_conv1.weight"):
            # Output channels live in the conv module's own space, input channels in the
            # residual stream. The GLU pairs output channel c with channel c + hidden, so the
            # two halves must be selected together or every kept value channel is gated by a
            # stranger.
            internal = selection.conv[
                name.replace("pointwise_conv1.weight", "pointwise_conv2")
            ]
            value = (
                source(name).index_select(0, _glu_rows(internal, hidden))
                .index_select(1, residual)
            )
        elif ".residual_conv." in name or ".self_attn_conv." in name:
            # The adapter's convolutions are different: their GLU output *is* the residual
            # stream (one branch is added straight back), so both halves index by it.
            rows = _glu_rows(residual, hidden)
            weight = source(name)
            value = (
                weight.index_select(0, rows).index_select(1, residual)
                if weight.dim() == 3
                else weight.index_select(0, rows)
            )
        elif name.endswith("depthwise_conv.weight"):
            internal = selection.conv[
                name.replace("depthwise_conv.weight", "pointwise_conv2")
            ]
            value = source(name).index_select(0, internal)
        elif name.endswith("pointwise_conv2.weight"):
            consumer = name.rsplit(".", 1)[0]
            internal = selection.conv[consumer]
            value = (
                source(name).index_select(0, residual).index_select(1, internal)
                * gains.get(consumer, 1.0)
            )

        if value is None:
            skipped.append(f"{name} (no rule)")
            continue
        if value.shape != target.shape:
            raise ValueError(
                f"transplant shape mismatch for {name}: produced {tuple(value.shape)}, "
                f"student expects {tuple(target.shape)}"
            )
        target.copy_(value.to(target.dtype))
        copied.append(name)

    student.load_state_dict(student_state)
    return {
        "copied": len(copied),
        "left_random": len(skipped),
        "left_random_names": skipped,
        "copied_parameters": int(
            sum(student_state[name].numel() for name in copied)
        ),
    }


@torch.no_grad()
def initial_agreement(student, teacher, batches: list[torch.Tensor]) -> dict:
    """How close the initialisation already is, before a single gradient step.

    Three numbers, because they fail in different places. ``kl`` is the objective itself.
    ``frame_agreement`` says whether the argmax is right anywhere. ``decoded_agreement``
    scores the confirmed stream, which is the only one that tracks the release gate -- and a
    transplant that improves KL while leaving decoded agreement at zero is a transplant that
    has learned the blank distribution and nothing else.
    """
    from training.distill_eval import confirmed_tokens, levenshtein
    from training.distill_loss import CONFIRM_TIMESTEPS, agreement_stats, frame_weights, weighted_kl

    totals = {"kl": 0.0, "confirmed_agreement": 0.0, "nonblank_agreement": 0.0}
    edits = tokens = 0

    for features in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_logits = teacher(features, return_dict=True)["logits"]["phonemes"]
            student_logits = student(features, return_dict=True)["logits"]["phonemes"]
        totals["kl"] += float(
            weighted_kl(student_logits, teacher_logits, frame_weights(teacher_logits))
        )
        stats = agreement_stats(student_logits, teacher_logits)
        # confirmed_agreement, not overall: overall is dominated by blank, which is 67% of
        # frames, so an initialisation that predicts blank everywhere would score 0.67 there
        # and read as two-thirds of the way to the teacher.
        totals["confirmed_agreement"] += stats.confirmed_agreement
        totals["nonblank_agreement"] += stats.nonblank_agreement

        student_ids = student_logits.argmax(dim=-1).cpu().numpy()
        teacher_ids = teacher_logits.argmax(dim=-1).cpu().numpy()
        for student_row, teacher_row in zip(student_ids, teacher_ids):
            reference = confirmed_tokens(teacher_row, CONFIRM_TIMESTEPS)
            edits += levenshtein(reference, confirmed_tokens(student_row, CONFIRM_TIMESTEPS))
            tokens += len(reference)

    count = max(1, len(batches))
    return {
        "kl": round(totals["kl"] / count, 4),
        "confirmed_agreement": round(totals["confirmed_agreement"] / count, 4),
        "nonblank_agreement": round(totals["nonblank_agreement"] / count, 4),
        "decoded_agreement": round(1.0 - edits / max(1, tokens), 4),
    }


def write_checkpoint(path: Path, student, spec: StudentSpec, report: dict) -> None:
    """Save in ``training.distill_train``'s checkpoint format so ``--init-from`` reads it.

    ``load_student_weights`` wants ``student`` and optionally ``projector``; the eval tools
    want ``config['preset']`` and ``step``. ``step`` is 0 because no optimisation has
    happened -- writing anything else would make a metrics log read as if it had.
    """
    from training.distill_loss import DEFAULT_TAP_LAYERS, FeatureProjector
    from training.distill_train import TrainConfig

    projector = FeatureProjector(
        spec.hidden_size, TEACHER_HIDDEN_SIZE, len(DEFAULT_TAP_LAYERS)
    )
    config = TrainConfig(preset=spec.name, audio_root="", out_dir=str(path.parent))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": 0,
            "student": student.state_dict(),
            "projector": projector.state_dict(),
            "config": asdict(config),
            "teacher_init": report,
        },
        path,
    )


def load_calibration_batches(
    audio_root: Path, num_windows: int, batch_size: int, device
) -> list[torch.Tensor]:
    """Real windows, not noise: every importance criterion here is an expectation over data."""
    from torch.utils.data import DataLoader

    from training.distill_data import DistillWindowDataset, build_window_index, discover_clips
    from training.distill_train import _init_worker

    clips = discover_clips(audio_root)
    if not clips:
        raise SystemExit(f"no .wav files under {audio_root}")
    refs = build_window_index(clips)[:num_windows]
    loader = DataLoader(
        DistillWindowDataset(refs),
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        worker_init_fn=_init_worker,
    )
    return [batch.to(device) for batch in loader]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialise a thin student from a selected sub-network of the teacher"
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="h384")
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--num-windows",
        type=int,
        default=256,
        help="calibration windows. The channel and head rankings are stable well below this;"
        " it is the LayerNorm moments that want a few hundred.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--qk",
        choices=("random", "copy", "damp"),
        default="random",
        help="what to do with encoder query/key projections. The teacher fitted them under a "
        "relative_key positional bias on unrotated inputs, so copying them is an open "
        "question rather than an obvious win -- run all three.",
    )
    parser.add_argument("--qk-scale", type=float, default=0.5)
    parser.add_argument(
        "--no-branch-gain",
        action="store_true",
        help="skip the least-squares rescaling of each kept branch. Only to measure what "
        "the rescaling is worth -- without it every branch under-drives the residual stream "
        "by roughly the fraction of units that were dropped.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="also score a randomly-initialised student on the same windows, which is the "
        "only thing that makes the transplant's numbers mean anything",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")

    from training.distill_train import load_teacher, seed_everything

    seed_everything(1234)
    spec = PRESETS[args.preset]
    teacher = load_teacher(device)
    batches = load_calibration_batches(
        args.audio_root, args.num_windows, args.batch_size, device
    )
    print(f"[calib] {len(batches)} batches of {args.batch_size} windows", flush=True)

    stats = collect_importance(teacher, batches)
    selection = choose(spec, stats, teacher.config.hidden_size // teacher.config.num_attention_heads)
    stats.layernorm_moments = collect_layernorm_moments(teacher, batches, selection)
    gains = {} if args.no_branch_gain else collect_branch_gains(teacher, batches, selection)
    print(
        f"[select] {spec.hidden_size}/{TEACHER_HIDDEN_SIZE} channels carry "
        f"{stats.captured_variance_share(selection.residual):.1%} of the pooled residual "
        f"importance (an even spread would be "
        f"{spec.hidden_size / TEACHER_HIDDEN_SIZE:.1%})",
        flush=True,
    )
    print(f"[select] {len(stats.layernorm_moments)} LayerNorm sites corrected", flush=True)

    if gains:
        values = sorted(gains.values())
        print(
            f"[select] {len(gains)} branch gains fitted, median "
            f"{values[len(values) // 2]:.2f}, range "
            f"[{values[0]:.2f}, {values[-1]:.2f}] -- 1.0 means the dropped units contributed "
            f"orthogonally and no rescaling recovers them",
            flush=True,
        )

    student = build_student(spec).to(device)
    weights = load_teacher_weights()
    report = transplant(
        student, weights, selection, stats.layernorm_moments, args.qk, args.qk_scale, gains
    )
    del weights
    student.eval()
    report["qk_mode"] = args.qk
    report["branch_gains"] = {name: round(value, 4) for name, value in gains.items()}
    report["captured_importance_share"] = round(
        stats.captured_variance_share(selection.residual), 4
    )
    report["initialised"] = initial_agreement(student, teacher, batches)

    if args.baseline:
        random_student = build_student(spec).to(device)
        random_student.eval()
        report["random_init"] = initial_agreement(random_student, teacher, batches)
        del random_student

    write_checkpoint(args.out, student, spec, report)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"\nTeacher initialisation -- {spec.name}, --qk {args.qk}")
    print(
        f"  copied                {report['copied']} tensors "
        f"({report['copied_parameters'] / 1e6:.1f}M parameters)"
    )
    print(f"  left at random init   {report['left_random']} tensors")
    for name in report["left_random_names"][:8]:
        print(f"    {name}")
    if report["left_random"] > 8:
        print(f"    ... and {report['left_random'] - 8} more")
    print("  before any training:")
    for label, key in (("transplanted", "initialised"), ("random init", "random_init")):
        if key in report:
            scores = report[key]
            print(
                f"    {label:<14} kl {scores['kl']:.4f}  "
                f"confirmed {scores['confirmed_agreement']:.3f}"
                f"  non-blank {scores['nonblank_agreement']:.3f}"
                f"  DECODED {scores['decoded_agreement']:.3f}"
            )
    print(f"\nwrote {args.out} -- train with --init-from {args.out}")


if __name__ == "__main__":
    main()
