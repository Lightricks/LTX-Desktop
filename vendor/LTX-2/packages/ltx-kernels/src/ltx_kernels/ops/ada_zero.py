# ruff: noqa: N803, ANN001, ANN202
"""AdaZero: fused RMSNorm + per-token affine (AdaLN-Zero style).
Computes::
    out = rms_norm(x, eps) * (1 + scale) + shift
where ``scale`` and ``shift`` are broadcast either across the seq dim (shape
``(B, 1, H)``) or per-token (shape ``(B, T, H)``).
Two custom ops are registered:
- ``ltx_kernels::ada_zero`` -- bf16 in, bf16 out.
- ``ltx_kernels::ada_zero_quantized`` -- bf16 in, returns
  ``(float8_e4m3fn, fp32 scales)`` with per-128-element blockwise scales for
  downstream blockwise-FP8 consumers.
The bf16 ``ada_zero`` is the structural-fusion target; the compile backend's
FuseTrailingQuantize pass rewrites ``blockwise_quantize(ada_zero(...))`` into
``ada_zero_quantized(...)`` wherever the normed output feeds only FP8 consumers
(both share one ``@jit`` kernel via its ``QUANTIZE`` flag).
"""

from typing import Tuple

import torch
from torch.library import triton_op, wrap_triton
from triton import jit
from triton import language as tl

from ltx_kernels.blockwise._common import _BLOCK_SIZE, _FP8_SCALE_MAX, _empty_tma_aligned_scales, _quantize


@jit
def _ada_zero_kernel(  # noqa: PLR0913
    X,
    Norm_Scale,
    Norm_Shift,
    Out_scales,
    X_out,
    norm_scale_batch_stride: int,
    norm_shift_batch_stride: int,
    norm_scale_token_stride: int,
    norm_shift_token_stride: int,
    seqlen: int,
    tma_aligned_mn: int,
    eps: float,
    H: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    QUANTIZE: tl.constexpr,
    SCALE_MAX: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    batch_id = token_idx // seqlen

    # Both affine tensors are addressed purely through forwarded strides, so a batch-
    # or seq-broadcast one arrives expanded with a 0 stride and needs no special case.
    x_ptr = X + token_idx * H + tl.arange(0, H)
    norm_scale_ptr = (
        Norm_Scale
        + batch_id * norm_scale_batch_stride
        + (token_idx % seqlen) * norm_scale_token_stride
        + tl.arange(0, H)
    )
    norm_shift_ptr = (
        Norm_Shift
        + batch_id * norm_shift_batch_stride
        + (token_idx % seqlen) * norm_shift_token_stride
        + tl.arange(0, H)
    )

    norm_scales = tl.load(norm_scale_ptr)
    norm_shift = tl.load(norm_shift_ptr)
    x = tl.load(x_ptr)

    x_sqr = x * x
    x_norm = tl.sum(x_sqr) / H
    x_norm = tl.rsqrt(x_norm + eps)
    x = (x * x_norm).to(tl.bfloat16)
    x = x * (1.0 + norm_scales) + norm_shift

    if QUANTIZE:
        o_quant, o_scales = _quantize(x, SCALE_MAX, NUM_BLOCKS, BLOCK_SIZE)
        x_out_ptr = (
            X_out + token_idx * H + BLOCK_SIZE * tl.arange(0, NUM_BLOCKS)[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]
        )
        tl.store(x_out_ptr, o_quant)
        out_scales_ptr = Out_scales + token_idx + tma_aligned_mn * tl.arange(0, NUM_BLOCKS)[:, None]
        tl.store(out_scales_ptr, o_scales)
    else:
        x_out_ptr = X_out + token_idx * H + tl.arange(0, H)
        tl.store(x_out_ptr, x)


@triton_op("ltx_kernels::ada_zero", mutates_args=())
def ada_zero(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """BF16 RMSNorm + per-token affine.
    Args:
        x: ``(B, T, H)`` bf16 input.
        scale: ``(B, 1, H)`` or ``(B, T, H)`` bf16 per-token scale.
        shift: ``(B, 1, H)`` or ``(B, T, H)`` bf16 per-token shift.
        eps: RMSNorm epsilon.
    Returns:
        ``(B, T, H)`` bf16 output.
    """
    b, s, h = x.shape
    num_rows = b * s
    # The kernel walks x row-major from one base pointer, and reads an affine row as
    # ``ptr + arange(0, H)`` off forwarded batch and token strides: x has to be
    # contiguous, the affine tensors need an innermost stride of 1, and a broadcast one
    # has to arrive expanded so its stride is 0 rather than its dim short.
    x = x.contiguous()
    if scale.stride(-1) != 1:
        scale = scale.contiguous()
    if shift.stride(-1) != 1:
        shift = shift.contiguous()
    scale = scale.expand(b, s, h)
    shift = shift.expand(b, s, h)
    out = torch.empty_like(x)
    wrap_triton(_ada_zero_kernel)[(num_rows,)](
        x,
        scale,
        shift,
        None,
        out,
        norm_scale_batch_stride=scale.stride(0),
        norm_shift_batch_stride=shift.stride(0),
        norm_scale_token_stride=scale.stride(1),
        norm_shift_token_stride=shift.stride(1),
        seqlen=s,
        eps=eps,
        H=h,
        NUM_BLOCKS=0,
        BLOCK_SIZE=_BLOCK_SIZE,
        tma_aligned_mn=0,
        QUANTIZE=False,
        SCALE_MAX=_FP8_SCALE_MAX,
    )
    return out


@triton_op("ltx_kernels::ada_zero_quantized", mutates_args=())
def ada_zero_quantized(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP8-output RMSNorm + per-token affine.
    Returns ``(fp8_e4m3fn output, fp32 per-128-element blockwise scales)``.
    """
    b, s, h = x.shape
    num_rows = b * s
    num_blocks = h // _BLOCK_SIZE
    x = x.contiguous()
    if scale.stride(-1) != 1:
        scale = scale.contiguous()
    if shift.stride(-1) != 1:
        shift = shift.contiguous()
    scale = scale.expand(b, s, h)
    shift = shift.expand(b, s, h)
    out = torch.empty((num_rows, h), device=x.device, dtype=torch.float8_e4m3fn)
    scales = _empty_tma_aligned_scales(num_rows, num_blocks, x.device)
    wrap_triton(_ada_zero_kernel)[(num_rows,)](
        x,
        scale,
        shift,
        scales,
        out,
        norm_scale_batch_stride=scale.stride(0),
        norm_shift_batch_stride=shift.stride(0),
        norm_scale_token_stride=scale.stride(1),
        norm_shift_token_stride=shift.stride(1),
        seqlen=s,
        eps=eps,
        H=h,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=_BLOCK_SIZE,
        tma_aligned_mn=scales.shape[1],
        QUANTIZE=True,
        SCALE_MAX=_FP8_SCALE_MAX,
    )
    return out.view(b, s, h), scales.t()[:num_rows]
