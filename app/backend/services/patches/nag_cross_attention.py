"""Monkey-patch: Normalized Attention Guidance (NAG) on video text cross-attention.

The distilled LTX models run unguided (CFG = 1), so a negative prompt has no
negative branch. NAG (Chen et al., 2025) guides inside the text cross-attention
instead: one extra K/V attention per layer, no second forward pass.

For each guided layer, with ``s`` = scale, ``a`` = alpha, ``t`` = tau (KJNodes
``LTX2_NAG`` convention):

    Z~  = s * Z_pos - (s - 1) * Z_neg
    r   = ||Z~||_1 / ||Z_pos||_1            (L1 over the feature dim)
    Z^  = Z~ * min(r, t) / r
    out = a * Z^ + (1 - a) * Z_pos

``Z_pos`` and ``Z_neg`` share one query. They are the attention outputs after
``V`` and before ``to_out`` and the per-head gate, which both stay shared.

How the negative context reaches the layer: the pipeline concatenates the
negative embedding to ``Modality.context`` along the sequence axis (see
``DistilledPipelineWithSkipStage2._prepare_run``). The transformer then applies
``caption_projection`` to both halves. In the AdaLN branch the context is
modulated before attention (``context * (1 + scale_kv) + shift_kv``), and the
modulation broadcasts over the sequence axis, so it reaches both halves too.
This patch only splits the halves inside ``attn2``.

Only ``BasicAVTransformerBlock.attn2`` is guided. ``audio_attn2``, self-attention
and the audio/video cross-attention modules keep the upstream call.

Strict no-op: with no active ``nag_scope`` the upstream method runs unchanged.

Usage:
    import services.patches.nag_cross_attention  # noqa: F401
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock

from services.fast_video_pipeline.nag_config import NagConfig

assert callable(getattr(BasicAVTransformerBlock, "_apply_text_cross_attention", None)), (
    "BasicAVTransformerBlock._apply_text_cross_attention missing - "
    "re-verify this patch on rev bump"
)
_original_apply_text_cross_attention = BasicAVTransformerBlock._apply_text_cross_attention


@dataclass(frozen=True)
class NagState:
    config: NagConfig
    # Sequence length of the positive text context. Tokens after it are the negative context.
    positive_length: int


# One generation runs at a time (GenerationHandler serializes them), so a module global is
# enough. It is not a ContextVar: torch.compile traces a plain global read.
_active: NagState | None = None


def active_state() -> NagState | None:
    return _active


@contextmanager
def nag_scope(state: NagState) -> Iterator[None]:
    """Turn NAG on for the denoise calls inside the block. Always turns it off on exit."""
    global _active
    if _active is not None:
        raise RuntimeError("NAG scope is already active")
    _active = state
    try:
        yield
    finally:
        _active = None


def normalized_attention_guidance(
    z_pos: torch.Tensor, z_neg: torch.Tensor, *, scale: float, alpha: float, tau: float
) -> torch.Tensor:
    guidance = z_pos * scale - z_neg * (scale - 1.0)
    # Norms accumulate in float32 without a full float32 copy of the token tensor.
    norm_pos = torch.linalg.vector_norm(z_pos, ord=1, dim=-1, keepdim=True, dtype=torch.float32)
    norm_guidance = torch.linalg.vector_norm(guidance, ord=1, dim=-1, keepdim=True, dtype=torch.float32)
    ratio = torch.nan_to_num(norm_guidance / norm_pos, nan=10.0)
    clip = torch.where(ratio > tau, norm_pos * tau / (norm_guidance + 1e-7), torch.ones_like(ratio))
    guidance = guidance * clip.to(guidance.dtype)
    return guidance * alpha + z_pos * (1.0 - alpha)


def _attend(attn: Attention, q: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    k = attn.k_norm(attn.to_k(context))
    v = attn.to_v(context)
    if mask is None:
        return attn.attention_function(q, k, v, attn.heads)
    return attn.masked_attention_function(q, k, v, attn.heads, mask)


class _NagTextAttention:
    """Stands in for ``attn2`` inside the upstream cross-attention helpers.

    Same call shape as ``Attention.forward`` for a text cross-attention call
    (no RoPE, no perturbation mask).
    """

    def __init__(self, attn: Attention, state: NagState) -> None:
        self._attn = attn
        self._state = state

    def __call__(
        self, x: torch.Tensor, context: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attn = self._attn
        config = self._state.config
        if context is None:
            raise RuntimeError("NAG text attention needs a context")
        length = self._state.positive_length
        q = attn.q_norm(attn.to_q(x))
        # The text mask is None in ltx-pipelines. A mask, if present, fits the positive half only.
        z_pos = _attend(attn, q, context[:, :length], mask)
        z_neg = _attend(attn, q, context[:, length:], None)
        out = normalized_attention_guidance(
            z_pos, z_neg, scale=config.scale, alpha=config.alpha, tau=config.tau
        )
        if attn.to_gate_logits is not None:
            batch, tokens, _ = out.shape
            gates = 2.0 * torch.sigmoid(attn.to_gate_logits(x))
            out = (out.view(batch, tokens, attn.heads, attn.dim_head) * gates.unsqueeze(-1)).view(
                batch, tokens, attn.heads * attn.dim_head
            )
        return attn.to_out(out)


def _apply_text_cross_attention(
    self: BasicAVTransformerBlock,
    x_normed: torch.Tensor,
    context: torch.Tensor,
    attn: Any,
    scale_shift_table: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor | None,
    timestep_embedding: Any,
    prompt_timestep: torch.Tensor | None,
    context_mask: torch.Tensor | None,
    cross_attention_adaln: bool = False,
) -> torch.Tensor:
    state = _active
    if state is None or attn is not getattr(self, "attn2", None):
        return _original_apply_text_cross_attention(
            self,
            x_normed,
            context,
            attn,
            scale_shift_table,
            prompt_scale_shift_table,
            timestep_embedding,
            prompt_timestep,
            context_mask,
            cross_attention_adaln,
        )
    if context.shape[1] <= state.positive_length:
        raise RuntimeError("NAG is active but the text context has no negative tokens")
    if cross_attention_adaln and prompt_timestep is not None and prompt_timestep.shape[1] != 1:
        # A per-token prompt modulation would not broadcast over the concatenated halves.
        raise RuntimeError("NAG needs a prompt timestep with one entry per sample")
    return _original_apply_text_cross_attention(
        self,
        x_normed,
        context,
        _NagTextAttention(attn, state),
        scale_shift_table,
        prompt_scale_shift_table,
        timestep_embedding,
        prompt_timestep,
        context_mask,
        cross_attention_adaln,
    )


BasicAVTransformerBlock._apply_text_cross_attention = _apply_text_cross_attention  # type: ignore[method-assign]
