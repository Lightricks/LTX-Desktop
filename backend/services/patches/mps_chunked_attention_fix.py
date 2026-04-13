"""Monkey-patch: chunked scaled_dot_product_attention for MPS (Apple Silicon).

On MPS, PyTorch does not support Flash Attention or memory-efficient attention
(xformers/FlashAttention-3 are CUDA-only).  LTX uses PytorchAttention, which
calls torch.nn.functional.scaled_dot_product_attention with the full Q/K/V
tensors.  For long video sequences this allocates an attention matrix of size
O(N²) where N can exceed 10 000 tokens, producing a ~26 GB MTLBuffer that
immediately OOMs:

    MPSCore: failed assertion `Failed to allocate private MTLBuffer for size
    28341043200'

This patch replaces PytorchAttention.__call__ on MPS with a chunked
implementation that processes the query sequence in fixed-size chunks, keeping
peak memory at O(N × chunk_size) instead of O(N²).  CUDA and CPU paths are
left entirely unchanged.

The chunk size is controlled by the environment variable
LTX_MPS_ATTN_CHUNK_SIZE (default: 512).  Larger chunks are faster but use
more memory; smaller chunks are safer on constrained hardware.

Remove this patch once ltx-core ships a memory-efficient attention path for
MPS (e.g. via torch.nn.attention.SDPBackend.CHUNKED_PREFILL or a first-party
chunked kernel).

Usage:
    import services.patches.mps_chunked_attention_fix  # noqa: F401
"""

from __future__ import annotations

import os
from typing import Any

import torch

import ltx_core.model.transformer.attention as _attn_module
from ltx_core.model.transformer.attention import PytorchAttention

_DEFAULT_CHUNK_SIZE = 512
_CHUNK_SIZE: int = int(os.environ.get("LTX_MPS_ATTN_CHUNK_SIZE", _DEFAULT_CHUNK_SIZE))

_original_pytorch_attention_call = PytorchAttention.__call__


def _chunked_mps_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Chunked SDPA for MPS.

    Splits Q into chunks of _CHUNK_SIZE tokens and accumulates the output,
    attending each Q-chunk against the full K/V.  This keeps the intermediate
    attention scores at O(chunk_size × N) instead of O(N²).

    Args:
        q, k, v: [B, S, H*D] tensors (ltx_core's packed format before reshaping)
        heads:   number of attention heads
        mask:    optional attention mask — may arrive in any of the shapes that
                 PytorchAttention / SDPA accept:
                   2-D [seq_q, seq_k]
                   3-D [B_or_1, seq_q, seq_k]
                   4-D [B_or_1, H_or_1, seq_q, seq_k]
                 A mask whose Q-dimension does not match seq_q (e.g. size 0 for
                 the text-encoder's audio-only path) is passed through as-is and
                 never sliced, letting SDPA broadcast it correctly.

    Returns:
        Output tensor in the same [B, S, H*D] packed format.
    """
    b, seq_q, _ = q.shape
    dim_head = q.shape[-1] // heads

    # Reshape to [B, H, S, D] (standard SDPA layout)
    q = q.view(b, seq_q, heads, dim_head).transpose(1, 2)
    k = k.view(b, k.shape[1], heads, dim_head).transpose(1, 2)
    v = v.view(b, v.shape[1], heads, dim_head).transpose(1, 2)

    # Normalise mask to 4-D [B, H, seq_q_mask, seq_k] so we can inspect dims.
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    # Determine whether we can safely slice the mask along the Q dimension.
    # The mask Q-dim (dim 2) may be:
    #   == seq_q  → slice per chunk
    #   == 1      → broadcast singleton, pass as-is to every chunk
    #   == 0      → empty / no-op mask (e.g. text-encoder audio-only path);
    #               treat as None so SDPA doesn't try to broadcast 0 → chunk_size
    if mask is not None and mask.shape[2] == 0:
        mask = None
    mask_seq_q = mask.shape[2] if mask is not None else 0
    can_slice_mask = mask is not None and mask_seq_q == seq_q

    chunks: list[torch.Tensor] = []
    for start in range(0, seq_q, _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, seq_q)
        q_chunk = q[:, :, start:end, :]

        if can_slice_mask:
            assert mask is not None  # narrowing for pyright
            mask_chunk: torch.Tensor | None = mask[:, :, start:end, :]
        else:
            mask_chunk = mask  # broadcast as-is (covers size-0 and size-1 cases)

        chunk_out = torch.nn.functional.scaled_dot_product_attention(
            q_chunk, k, v, attn_mask=mask_chunk, dropout_p=0.0, is_causal=False
        )
        chunks.append(chunk_out)

    # Reassemble and return in packed [B, S, H*D] format
    out = torch.cat(chunks, dim=2)           # [B, H, S, D]
    out = out.transpose(1, 2).reshape(b, seq_q, heads * dim_head)
    return out


def _patched_pytorch_attention_call(
    self: PytorchAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch to chunked attention on MPS, original on other devices."""
    if q.device.type == "mps":
        return _chunked_mps_attention(q, k, v, heads, mask)
    return _original_pytorch_attention_call(self, q, k, v, heads, mask)


# Apply patch.
assert hasattr(PytorchAttention, "__call__") and callable(PytorchAttention.__call__), (
    "PytorchAttention.__call__ not found — was it renamed? "
    "The mps_chunked_attention_fix patch needs updating."
)
PytorchAttention.__call__ = _patched_pytorch_attention_call  # type: ignore[method-assign]
