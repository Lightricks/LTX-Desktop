# ruff: noqa: N803, ANN001, ANN202
# Triton @jit kernels follow conventions that conflict with Python's lint rules:
# pointer args are CapWord (``X``), constexprs are UPPER_CASE (``BLOCK_SIZE``),
# pointer args carry no Python-level type, and the return type is opaque.
"""Shared blockwise-quant constants and Triton helpers.
Single home for the per-128-element FP8 quantize device function and the TMA
alignment math, reused by the blockwise quantize op (:mod:`.ops.quantize`) and
by the fused activation ops that quantize their output
(:mod:`ltx_kernels.ops.ada_zero`, :mod:`ltx_kernels.ops.gated_attention`).
"""

import torch
from triton import jit
from triton import language as tl

_BLOCK_SIZE = 128
_FP8_SCALE_MAX = 448.0
_FP6_SCALE_MAX = 0.1172


def _get_tma_aligned_size(n: int, element_size: int) -> int:
    """Round ``n`` up to the TMA element-count multiple for ``element_size`` bytes."""
    num_elems_tma = 16 // element_size
    return ((n + num_elems_tma - 1) // num_elems_tma) * num_elems_tma


def _empty_tma_aligned_scales(num_rows: int, num_blocks: int, device: torch.device) -> torch.Tensor:
    """Allocate ``(num_blocks, tma_aligned_rows)`` FP32 scale storage.
    Kernels write this tensor itself; the op returns ``.t()[:num_rows]``, the MN-major
    ``(num_rows, num_blocks)`` view. Writing through that view instead makes inductor guard
    on whether ``num_rows`` is TMA-aligned, so a dynamic row count recompiles per alignment.
    """
    return torch.empty((num_blocks, _get_tma_aligned_size(num_rows, 4)), dtype=torch.float, device=device)


@jit
def _quantize(x, scale_max: tl.constexpr, NUM_BLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Per-block FP8 quantize: returns (fp8 codes, 1/scale per block)."""
    x = tl.reshape(x, (NUM_BLOCKS, BLOCK_SIZE))
    x_abs = tl.abs(x)
    x_abs = tl.broadcast_to(x_abs, (NUM_BLOCKS, BLOCK_SIZE))
    x_absmax = tl.max(x_abs, axis=1)[:, None]
    x_scales = scale_max / x_absmax
    x_quant = (x_scales * x).to(tl.float8e4nv)
    x_out_scales = 1.0 / x_scales
    return x_quant, x_out_scales
