"""NVFP4 quantize + block-scaled GEMM, wrapped as Dynamo-safe custom ops.
The CUDA kernels live in the ``nvfp4_cpp`` extension. Layout contract, scale math and
padding rules are documented in ``packages/ltx-kernels/docs/NVFP4.md``; the short version:
* data: FP4 E2M1, two values per ``uint8``, element ``2j`` in the **high** nibble when
  ``hi_first=True`` (default);
* block scales: FP8 E4M3, one per 16 elements, in the cuBLAS 128x4 tiled layout, shape
  ``(roundup(rows, 128), roundup(K // 16, 4))``;
* per-tensor scale: fp32 *decode* scale ``amax / (448 * 6)``.
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Optional

import torch

E4M3_MAX = 448.0
E2M1_MAX = 6.0
GLOBAL_RANGE = E4M3_MAX * E2M1_MAX  # 2688.0
BLOCK_SIZE = 16

# Shared install hint used by extension/SM errors and by ``unavailable_reason``. The arch
# must match the target device -- 10.0 datacenter Blackwell, 11.0 Jetson Thor, 12.0 consumer
# Blackwell -- because each builds an arch-specific cubin that will not launch on the others.
REBUILD_HINT = (
    "TORCH_CUDA_ARCH_LIST=<10.0|11.0|12.0> uv pip install -e packages/ltx-kernels "
    "--no-build-isolation (or `uv sync --group kernels`)"
)


def _round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def scale_shape(rows: int, k: int) -> tuple[int, int]:
    """Shape of the cuBLAS-tiled block-scale array for an ``(rows, k)`` operand."""
    return _round_up(rows, 128), _round_up(k // BLOCK_SIZE, 4)


@functools.lru_cache(maxsize=1)
def _extension() -> ModuleType:
    try:
        import nvfp4_cpp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - build-dependent
        raise RuntimeError(
            f"ltx_kernels.nvfp4 requires the 'nvfp4_cpp' extension. Rebuild on a Blackwell host: {REBUILD_HINT}."
        ) from exc
    return nvfp4_cpp


@functools.lru_cache(maxsize=8)
def _check_sm(device_index: int) -> None:
    major, minor = torch.cuda.get_device_capability(device_index)
    if major < 10:
        raise RuntimeError(
            f"ltx_kernels.nvfp4 needs a Blackwell GPU (SM >= 10.0) for FP4 tensor cores; "
            f"this device is sm_{major}{minor}. Use the blockwise FP8 path instead."
        )


def unavailable_reason() -> str | None:
    """``None`` if NVFP4 is usable; otherwise a human-readable failure reason."""
    if not torch.cuda.is_available():
        return f"ltx_kernels.nvfp4 requires a CUDA GPU (Blackwell, SM >= 10.0). Rebuild with {REBUILD_HINT}."
    try:
        _extension()
        _check_sm(torch.cuda.current_device())
    except RuntimeError as exc:
        return str(exc)
    return None


def is_available() -> bool:
    """True when the extension is built and the current device has FP4 tensor cores."""
    return unavailable_reason() is None


# --------------------------------------------------------------------------------------
# custom ops
# --------------------------------------------------------------------------------------


@torch.library.custom_op("ltx_kernels_nvfp4::amax_scale", mutates_args=(), device_types="cuda")
def _amax_scale_op(x: torch.Tensor, divisor: float) -> torch.Tensor:
    _check_sm(x.device.index or 0)
    return _extension().amax_scale(x, divisor)


@_amax_scale_op.register_fake
def _amax_scale_op_fake(x: torch.Tensor, divisor: float) -> torch.Tensor:
    del divisor
    return torch.empty((), device=x.device, dtype=torch.float32)


def per_tensor_decode_scale(x: torch.Tensor) -> torch.Tensor:
    """``amax / (E4M3_MAX * E2M1_MAX)`` as an fp32 scalar (the *decode* convention).
    One fused pass over ``x`` on CUDA. The obvious ATen spelling,
    ``x.float().abs().nan_to_num().amax() / GLOBAL_RANGE``, is three full-size passes plus
    several scalar launches; ``aminmax`` fixes the passes but loses the true amax whenever
    ``x`` contains a NaN (it propagates into both ends of the range). The kernel drops NaN
    during the reduction, so it is both faster and matches the original semantics.
    """
    if not x.is_cuda:
        return (x.detach().float().abs().nan_to_num().amax() / GLOBAL_RANGE).reshape(())
    return torch.ops.ltx_kernels_nvfp4.amax_scale(x.detach().contiguous(), GLOBAL_RANGE)


@torch.library.custom_op("ltx_kernels_nvfp4::quantize", mutates_args=(), device_types="cuda")
def _quantize_op(
    x: torch.Tensor, per_tensor_scale: torch.Tensor, hi_first: bool, variant: int
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_sm(x.device.index or 0)
    scale = per_tensor_scale.reshape(1).to(torch.float32)
    return _extension().quantize_nvfp4(x.contiguous(), scale, hi_first, variant)


@_quantize_op.register_fake
def _quantize_op_fake(
    x: torch.Tensor, per_tensor_scale: torch.Tensor, hi_first: bool, variant: int
) -> tuple[torch.Tensor, torch.Tensor]:
    del per_tensor_scale, hi_first, variant
    rows, k = x.shape
    packed = torch.empty((rows, k // 2), device=x.device, dtype=torch.uint8)
    scales = torch.empty(scale_shape(rows, k), device=x.device, dtype=torch.uint8)
    return packed, scales


@torch.library.custom_op("ltx_kernels_nvfp4::scaled_mm", mutates_args=(), device_types="cuda")
def _scaled_mm_op(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    has_bias: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    _check_sm(a.device.index or 0)
    return _extension().scaled_mm_nvfp4(
        a,
        b,
        block_scale_a,
        block_scale_b,
        alpha.reshape(1),
        bias if has_bias else None,
        out_dtype,
    )


@_scaled_mm_op.register_fake
def _scaled_mm_op_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    has_bias: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    del block_scale_a, block_scale_b, alpha, bias, has_bias
    return torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=out_dtype)


_EMPTY_BIAS: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_ONES: dict[torch.device, torch.Tensor] = {}


def _empty_bias(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Placeholder tensor for the bias-less case (``custom_op`` needs a real Tensor)."""
    key = (device, dtype)
    tensor = _EMPTY_BIAS.get(key)
    if tensor is None:
        tensor = torch.empty(0, device=device, dtype=dtype)
        _EMPTY_BIAS[key] = tensor
    return tensor


def fixed_scale_one(device: torch.device) -> torch.Tensor:
    """Cached fp32 ``1.0`` on ``device`` (the ``act_scale="fixed_1"`` per-tensor scale).
    Cached so the hot path neither allocates nor launches a fill kernel per call.
    """
    tensor = _ONES.get(device)
    if tensor is None:
        tensor = torch.ones((), device=device, dtype=torch.float32)
        _ONES[device] = tensor
    return tensor


# --------------------------------------------------------------------------------------
# public functional API
# --------------------------------------------------------------------------------------


QUANT_KERNEL_TILED = 2
QUANT_KERNEL_ELEMENTWISE = 1


def quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    *,
    pad_16x: bool = True,
    hi_first: bool = True,
    variant: int = QUANT_KERNEL_TILED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D tensor to NVFP4.
    Args:
        x: ``(rows, K)`` bf16/fp16/fp32 CUDA tensor. ``K % 16 == 0`` is required; ``K % 32 == 0``
            is additionally required by the GEMM.
        per_tensor_scale: fp32 *decode* scale (``amax / 2688``), read on device (no sync).
        pad_16x: zero-pad ``K`` up to a multiple of 16 when needed. Row counts need no
            padding; ragged ``M`` is accepted and the GEMM keeps it (no output slice).
        hi_first: nibble order; ``True`` (default) puts element ``2j`` in the high nibble.
        variant: quantize kernel to use -- 2 (default) stages the block scales in shared
            memory per 128x4 tile, 1 writes them elementwise. Only differs in speed.
    Returns:
        ``(packed uint8 (rows, K // 2), block scales uint8 (roundup(rows,128), roundup(K//16,4)))``.
        Block scales are E4M3 bit patterns stored as ``uint8`` (view them as
        ``torch.float8_e4m3fn`` if you need the dtype tag).
    """
    if x.dim() != 2:
        raise ValueError(f"quantize_nvfp4 expects a 2-D tensor, got shape {tuple(x.shape)}")
    k = x.shape[1]
    if k % BLOCK_SIZE != 0:
        if not pad_16x:
            raise ValueError(f"K must be a multiple of {BLOCK_SIZE} when pad_16x=False, got {k}")
        x = torch.nn.functional.pad(x, (0, _round_up(k, BLOCK_SIZE) - k))
    return torch.ops.ltx_kernels_nvfp4.quantize(x, per_tensor_scale, hi_first, variant)


def scaled_mm_nvfp4(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    block_a: torch.Tensor,
    block_b: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``alpha * (a @ b.T) + bias`` for NVFP4 operands, where ``alpha = scale_a * scale_b``.
    Args:
        a: packed activations ``(M, K // 2)`` uint8.
        b: packed weights ``(N, K // 2)`` uint8 (row-major, i.e. already "transposed").
        scale_a / scale_b: fp32 per-tensor decode scales.
        block_a / block_b: cuBLAS-tiled E4M3 block scales (uint8 or float8_e4m3fn).
        bias: optional ``(N,)`` tensor in ``out_dtype``, fused in the GEMM epilogue.
        alpha: precomputed ``scale_a * scale_b`` (skips one tiny kernel when the caller
            already has it, e.g. a fixed activation scale of 1.0).
    """
    if alpha is None:
        alpha = (scale_a.reshape(()) * scale_b.reshape(())).to(torch.float32)
    has_bias = bias is not None
    return torch.ops.ltx_kernels_nvfp4.scaled_mm(
        a,
        b.contiguous(),
        block_a.view(torch.uint8),
        block_b.view(torch.uint8),
        alpha,
        bias if has_bias else _empty_bias(a.device, out_dtype),
        has_bias,
        out_dtype,
    )


def dequantize_nvfp4(
    packed: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    hi_first: bool = True,
) -> torch.Tensor:
    """Reference dequantize (tests / debugging); not a fast path."""
    _check_sm(packed.device.index or 0)
    return _extension().dequantize_nvfp4(
        packed.contiguous(),
        per_tensor_scale.reshape(1).to(torch.float32),
        block_scales.view(torch.uint8),
        out_dtype,
        hi_first,
    )


def linear_nvfp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_block_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    act_scale: Optional[torch.Tensor] = None,
    hi_first: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """NVFP4 ``x @ weight.T + bias`` with on-the-fly activation quantization.
    ``act_scale=None`` uses the fixed per-tensor scale 1.0, which keeps ``alpha`` equal to
    the weight scale and avoids both an amax reduction over ``x`` and a scale multiply.
    """
    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1])
    if x_2d.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        x_2d = x_2d.to(torch.bfloat16)

    if act_scale is None:
        act_scale = fixed_scale_one(x_2d.device)
        alpha = weight_scale  # scale_a == 1, so alpha is the weight scale as-is
    else:
        alpha = None
    x_fp4, x_block = quantize_nvfp4(x_2d, act_scale, hi_first=hi_first)
    out = scaled_mm_nvfp4(
        x_fp4,
        weight,
        act_scale,
        weight_scale,
        x_block,
        weight_block_scale,
        bias=bias,
        out_dtype=out_dtype,
        alpha=alpha,
    )
    return out.view(*original_shape[:-1], weight.shape[0])
