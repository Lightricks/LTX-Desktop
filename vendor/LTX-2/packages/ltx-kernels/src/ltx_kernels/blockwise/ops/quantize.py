# ruff: noqa: N803, ANN001, ANN202
"""Blockwise FP8 activation quantize / dequantize, and torch weight quantizers.
The activation ops are registered via :func:`torch.library.triton_op` so they
appear as single FX nodes (``torch.ops.ltx_kernels.blockwise_quantize`` /
``...blockwise_dequantize``) -- the compile passes need stable identities to
bind against. The torch weight quantizers run at load time (not in the compiled
graph) and produce the FP8/FP6 ``.weight`` + ``.weight_scale`` buffers that
:class:`ltx_kernels.blockwise.linear.BlockwiseFP8Linear` / ``BlockwiseFP6Linear``
expect.
"""

import logging
from typing import Tuple

import torch
from torch.library import triton_op, wrap_triton
from triton import jit
from triton import language as tl

from ltx_kernels.blockwise._common import (
    _BLOCK_SIZE,
    _FP6_SCALE_MAX,
    _FP8_SCALE_MAX,
    _empty_tma_aligned_scales,
    _quantize,
)

logger = logging.getLogger(__name__)


@jit
def _block_quantize_kernel(
    X,
    Out_scales,
    X_out,
    tma_aligned_mn: int,
    H: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCALE_MAX: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    x_ptr = X + token_idx * H + BLOCK_SIZE * tl.arange(0, NUM_BLOCKS)[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]
    x = tl.load(x_ptr).to(tl.float32)

    o_quant, o_scales = _quantize(x, SCALE_MAX, NUM_BLOCKS, BLOCK_SIZE)
    x_out_ptr = (
        X_out + token_idx * H + BLOCK_SIZE * tl.arange(0, NUM_BLOCKS)[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    tl.store(x_out_ptr, o_quant)
    out_scales_ptr = Out_scales + token_idx + tma_aligned_mn * tl.arange(0, NUM_BLOCKS)[:, None]
    tl.store(out_scales_ptr, o_scales)


@triton_op("ltx_kernels::blockwise_quantize", mutates_args=())
def blockwise_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-128-element blockwise FP8 quantization of a bf16 activation.
    Args:
        x: ``(B, T, H)`` bf16 input. ``H`` must be divisible by 128.
    Returns:
        ``(fp8_e4m3fn (B, T, H), fp32 scales (B*T, H//128))`` -- the scales
        tensor has TMA-padded backing storage for downstream FP8 GEMM consumers.
    """
    b, s, h = x.shape
    num_rows = b * s
    num_blocks = h // _BLOCK_SIZE
    out = torch.empty((num_rows, h), device=x.device, dtype=torch.float8_e4m3fn)
    scales = _empty_tma_aligned_scales(num_rows, num_blocks, x.device)
    wrap_triton(_block_quantize_kernel)[(num_rows,)](
        x,
        scales,
        out,
        H=h,
        BLOCK_SIZE=_BLOCK_SIZE,
        NUM_BLOCKS=num_blocks,
        tma_aligned_mn=scales.shape[1],
        SCALE_MAX=_FP8_SCALE_MAX,
    )
    return out.view(b, s, h), scales.t()[:num_rows]


@jit
def _blockwise_dequantize_kernel(
    X,
    Scales,
    Out,
    scales_row_stride: int,
    scales_col_stride: int,
    H: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    scales = tl.load(Scales + token_idx * scales_row_stride + scales_col_stride * tl.arange(0, NUM_BLOCKS)).to(
        tl.float32
    )
    scales = tl.broadcast_to(scales[:, None], (NUM_BLOCKS, BLOCK_SIZE))
    x_offsets = BLOCK_SIZE * tl.arange(0, NUM_BLOCKS)[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]
    x = tl.load(X + token_idx * H + x_offsets).to(tl.float32)
    tl.store(Out + token_idx * H + x_offsets, (x * scales).to(tl.bfloat16))


@triton_op("ltx_kernels::blockwise_dequantize", mutates_args=())
def blockwise_dequantize(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`blockwise_quantize`.
    Args:
        x: ``(B, T, H)`` fp8_e4m3fn input.
        scales: ``(B*T, H//128)`` fp32 per-block scales.
    Returns:
        ``(B, T, H)`` bf16 output.
    """
    b, s, h = x.shape
    block_size = _BLOCK_SIZE
    num_blocks = h // block_size
    num_rows = b * s
    out = torch.empty((num_rows, h), device=x.device, dtype=torch.bfloat16)
    wrap_triton(_blockwise_dequantize_kernel)[(num_rows,)](
        x,
        scales,
        out,
        scales_row_stride=scales.stride(0),
        scales_col_stride=scales.stride(1),
        H=h,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=block_size,
    )
    return out.view(b, s, h)


# ---------------------------------------------------------------------------
# Torch weight quantizers (load-time; not part of the compiled graph)
# ---------------------------------------------------------------------------


def blockwise_quantize_weights(
    w: torch.Tensor, block_size: int = _BLOCK_SIZE, scale_max: float = _FP8_SCALE_MAX
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise-quantize a 2D weight ``[out, in]`` to fp8 + per-128-block fp32 scale."""
    w = w.view(w.shape[0] // block_size, block_size, w.shape[1] // block_size, block_size).transpose(1, 2).contiguous()
    w_absmax = w.float().abs().view(w.shape[0], w.shape[1], block_size * block_size).max(dim=-1, keepdim=False).values
    w_scales = scale_max / w_absmax
    w_quant = (w.float() * w_scales[:, :, None, None].float()).to(torch.float8_e4m3fn)
    w_quant = w_quant.transpose(1, 2).contiguous()
    w_quant = w_quant.view(w.shape[0] * block_size, -1)
    return w_quant, (1 / w_scales).contiguous()


def fp8_blockwise_quantize_weights_torch(
    x: torch.Tensor, block_size: int = _BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    return blockwise_quantize_weights(x, block_size, _FP8_SCALE_MAX)


def fp6_blockwise_quantize_weights_torch(
    x: torch.Tensor, block_size: int = _BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    return blockwise_quantize_weights(x, block_size, _FP6_SCALE_MAX)
