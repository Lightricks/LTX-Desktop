# ruff: noqa: N803, ANN001, ANN202, PLR0913
"""Fused residual update and RMS normalization."""

from typing import Tuple

import torch
from torch.library import triton_op, wrap_triton
from triton import jit
from triton import language as tl


@jit
def _rms_sum_mult_kernel(
    X,
    Y,
    Z,
    Norm_Scale,
    Norm_Shift,
    Residual_out,
    Normed_out,
    seqlen: int,
    z_batch_stride: int,
    z_token_stride: int,
    norm_scale_batch_stride: int,
    norm_shift_batch_stride: int,
    norm_scale_token_stride: int,
    norm_shift_token_stride: int,
    eps: float,
    H: tl.constexpr,
    IS_Z_BROADCAST: tl.constexpr,
    IS_Z_H_BROADCAST: tl.constexpr,
    BROADCAST_ADALN_SEQLEN: tl.constexpr,
    APPLY_ADALN: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    batch_id = token_idx // seqlen

    x_ptr = X + token_idx * H + tl.arange(0, H)
    y_ptr = Y + token_idx * H + tl.arange(0, H)
    if IS_Z_BROADCAST:
        z_ptr = Z + batch_id * z_batch_stride
    else:
        z_ptr = Z + batch_id * z_batch_stride + (token_idx % seqlen) * z_token_stride
    z_ptr += 0 if IS_Z_H_BROADCAST else tl.arange(0, H)
    x = tl.load(x_ptr)
    y = tl.load(y_ptr)
    z = tl.load(z_ptr)

    out = x + y * z
    tl.store(Residual_out + token_idx * H + tl.arange(0, H), out)

    out_sqr = out * out
    out_norm = tl.sum(out_sqr, axis=0) / H
    out_inv = tl.rsqrt(out_norm + eps)
    out *= out_inv

    if APPLY_ADALN:
        if BROADCAST_ADALN_SEQLEN:
            norm_scale_ptr = Norm_Scale + batch_id * norm_scale_batch_stride + tl.arange(0, H)
            norm_shift_ptr = Norm_Shift + batch_id * norm_shift_batch_stride + tl.arange(0, H)
        else:
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
        norm_scale = tl.load(norm_scale_ptr)
        norm_shift = tl.load(norm_shift_ptr)
        out = out.to(tl.bfloat16)
        out = out * (1.0 + norm_scale) + norm_shift

    tl.store(Normed_out + token_idx * H + tl.arange(0, H), out)


@triton_op("ltx_kernels::rms_fma", mutates_args=())
def rms_fma(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the gated residual update and normalize the updated value."""
    b, s, h = x.shape
    residual = torch.empty_like(x)
    normed = torch.empty_like(x)
    wrap_triton(_rms_sum_mult_kernel)[(b * s, 1, 1)](
        x,
        y,
        gate,
        None,
        None,
        residual,
        normed,
        seqlen=s,
        z_batch_stride=gate.stride(0),
        z_token_stride=0 if gate.ndim == 2 else gate.stride(1),
        norm_scale_batch_stride=0,
        norm_shift_batch_stride=0,
        norm_scale_token_stride=0,
        norm_shift_token_stride=0,
        eps=eps,
        H=h,
        IS_Z_BROADCAST=gate.ndim == 2 or gate.shape[1] == 1,
        IS_Z_H_BROADCAST=gate.shape[-1] == 1,
        BROADCAST_ADALN_SEQLEN=False,
        APPLY_ADALN=False,
    )
    return residual, normed


@triton_op("ltx_kernels::rms_fma_adaln", mutates_args=())
def rms_fma_adaln(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the gated residual update, RMSNorm, and query modulation."""
    b, s, h = x.shape
    residual = torch.empty_like(x)
    attn_input = torch.empty_like(x)
    wrap_triton(_rms_sum_mult_kernel)[(b * s, 1, 1)](
        x,
        y,
        gate,
        scale,
        shift,
        residual,
        attn_input,
        seqlen=s,
        z_batch_stride=gate.stride(0),
        z_token_stride=0 if gate.ndim == 2 else gate.stride(1),
        norm_scale_batch_stride=scale.stride(0),
        norm_shift_batch_stride=shift.stride(0),
        norm_scale_token_stride=0 if scale.ndim == 2 else scale.stride(1),
        norm_shift_token_stride=0 if shift.ndim == 2 else shift.stride(1),
        eps=eps,
        H=h,
        IS_Z_BROADCAST=gate.ndim == 2 or gate.shape[1] == 1,
        IS_Z_H_BROADCAST=gate.shape[-1] == 1,
        BROADCAST_ADALN_SEQLEN=scale.ndim == 2 or scale.shape[1] == 1,
        APPLY_ADALN=True,
    )
    return residual, attn_input
