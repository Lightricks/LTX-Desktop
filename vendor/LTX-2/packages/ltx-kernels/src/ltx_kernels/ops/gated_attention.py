# ruff: noqa: N803, ANN001, ANN202
# Triton @jit kernels follow conventions that conflict with Python's lint
# rules: pointer args are CapWord (``X``, ``Out``), constexprs are UPPER_CASE
# (``NUM_HEADS``, ``BLOCK_SIZE``), pointer args have no Python-level type, and
# the return type is opaque at the Python level.
"""Per-head gated attention.
Applies a per-head sigmoid gate to a multi-head attention output::
    gates = 2 * sigmoid(gate_logits)        # (B, T, H)
    out = attn_out.view(B, T, H, D) * gates.unsqueeze(-1)
    return out.view(B, T, H * D)
The ``2 *`` factor keeps zero-init at identity: ``2 * sigmoid(0) = 1``.
Two custom ops are registered:
- ``ltx_kernels::gated_attention`` -- bf16 in, bf16 out.
- ``ltx_kernels::gated_attention_quantized`` -- bf16 in, returns
  ``(float8_e4m3fn, fp32 scales)`` with per-128-element blockwise scales for
  downstream blockwise-FP8 consumers.
The bf16 op is the structural-fusion target; the compile backend's
FuseTrailingQuantize pass rewrites ``blockwise_quantize(gated_attention(...))``
into ``gated_attention_quantized(...)`` where the gated output feeds only FP8
consumers (e.g. ``to_out``). Both share one ``@jit`` kernel via its ``QUANTIZE``
flag.
"""

from typing import Tuple

import torch
from torch.library import triton_op, wrap_triton
from triton import jit
from triton import language as tl

from ltx_kernels.blockwise._common import _BLOCK_SIZE, _FP8_SCALE_MAX, _empty_tma_aligned_scales, _quantize


@jit
def _gated_attention_kernel(  # noqa: PLR0913
    X,
    Gate_Logits,
    Out,
    Out_scales,
    gate_batch_stride: int,
    gate_token_stride: int,
    seqlen: int,
    tma_aligned_mn: int,
    H: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    DIM_HEAD: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    QUANTIZE: tl.constexpr,
    SCALE_MAX: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)

    # Gates are addressed through forwarded strides, so a batch- or seq-broadcast
    # gate arrives expanded with a 0 stride and needs no special case.
    gate_ptr = (
        Gate_Logits
        + (token_idx // seqlen) * gate_batch_stride
        + (token_idx % seqlen) * gate_token_stride
        + tl.arange(0, NUM_HEADS)
    )
    gate_logits = tl.load(gate_ptr).to(tl.float32)
    gates = 2.0 * tl.sigmoid(gate_logits)
    gates = tl.broadcast_to(gates[:, None], (NUM_HEADS, DIM_HEAD))

    offsets = tl.arange(0, NUM_HEADS)[:, None] * DIM_HEAD + tl.arange(0, DIM_HEAD)[None, :]
    x = tl.load(X + token_idx * H + offsets).to(tl.float32)

    gated = x * gates

    if QUANTIZE:
        o_quant, o_scales = _quantize(gated, SCALE_MAX, NUM_BLOCKS, BLOCK_SIZE)
        out_ptr = (
            Out + token_idx * H + BLOCK_SIZE * tl.arange(0, NUM_BLOCKS)[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]
        )
        tl.store(out_ptr, o_quant)
        scales_ptr = Out_scales + token_idx + tma_aligned_mn * tl.arange(0, NUM_BLOCKS)[:, None]
        tl.store(scales_ptr, o_scales)
    else:
        out_ptr = Out + token_idx * H + offsets
        tl.store(out_ptr, gated)


@triton_op("ltx_kernels::gated_attention", mutates_args=())
def gated_attention(x: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
    """BF16 per-head gated attention.
    Args:
        x: ``(B, T, H * D)`` bf16 attention output.
        gate_logits: ``(B, T, H)`` per-head gate logits.
    Returns:
        ``(B, T, H * D)`` bf16 gated output.
    """
    b, t, h = x.shape
    num_heads = gate_logits.shape[-1]
    dim_head = h // num_heads
    num_blocks = h // _BLOCK_SIZE
    num_rows = b * t

    # The kernel walks x row-major from one base pointer and reads a gate row as
    # ``ptr + arange(0, NUM_HEADS)`` off forwarded strides: x has to be contiguous, the
    # gate needs an innermost stride of 1, and a broadcast gate has to arrive expanded
    # so its stride is 0 rather than its dim short.
    x = x.contiguous()
    if gate_logits.stride(-1) != 1:
        gate_logits = gate_logits.contiguous()
    gate_logits = gate_logits.expand(b, t, num_heads)

    out = torch.empty((num_rows, h), device=x.device, dtype=torch.bfloat16)

    wrap_triton(_gated_attention_kernel)[(num_rows,)](
        x,
        gate_logits,
        out,
        None,
        gate_batch_stride=gate_logits.stride(0),
        gate_token_stride=gate_logits.stride(1),
        seqlen=t,
        H=h,
        NUM_HEADS=num_heads,
        DIM_HEAD=dim_head,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=_BLOCK_SIZE,
        tma_aligned_mn=0,
        QUANTIZE=False,
        SCALE_MAX=_FP8_SCALE_MAX,
    )

    return out.view(b, t, h)


@triton_op("ltx_kernels::gated_attention_quantized", mutates_args=())
def gated_attention_quantized(x: torch.Tensor, gate_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP8-output per-head gated attention. Pair with a blockwise-FP8 consumer.
    Args:
        x: ``(B, T, H * D)`` bf16 attention output.
        gate_logits: ``(B, T, H)`` per-head gate logits.
    Returns:
        ``(fp8_e4m3fn output, fp32 per-128-element blockwise scales)``.
    """
    b, t, h = x.shape
    num_heads = gate_logits.shape[-1]
    dim_head = h // num_heads
    num_blocks = h // _BLOCK_SIZE
    num_rows = b * t

    x = x.contiguous()
    if gate_logits.stride(-1) != 1:
        gate_logits = gate_logits.contiguous()
    gate_logits = gate_logits.expand(b, t, num_heads)

    out = torch.empty((num_rows, h), device=x.device, dtype=torch.float8_e4m3fn)
    scales = _empty_tma_aligned_scales(num_rows, num_blocks, x.device)

    wrap_triton(_gated_attention_kernel)[(num_rows,)](
        x,
        gate_logits,
        out,
        scales,
        gate_batch_stride=gate_logits.stride(0),
        gate_token_stride=gate_logits.stride(1),
        seqlen=t,
        H=h,
        NUM_HEADS=num_heads,
        DIM_HEAD=dim_head,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=_BLOCK_SIZE,
        tma_aligned_mn=scales.shape[1],
        QUANTIZE=True,
        SCALE_MAX=_FP8_SCALE_MAX,
    )

    return out.view(b, t, h), scales.t()[:num_rows]
