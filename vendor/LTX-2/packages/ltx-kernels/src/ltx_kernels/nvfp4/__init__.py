"""In-house NVFP4 (FP4 E2M1 + FP8 E4M3 1x16 block scales) quantize + GEMM.
Blackwell only (SM >= 10.0). Default nibble order is ``hi_first=True`` (element ``2j``
in the high nibble). See ``docs/NVFP4.md``.
"""

from ltx_kernels.nvfp4.functional import (
    E2M1_MAX,
    E4M3_MAX,
    GLOBAL_RANGE,
    REBUILD_HINT,
    dequantize_nvfp4,
    fixed_scale_one,
    is_available,
    linear_nvfp4,
    per_tensor_decode_scale,
    quantize_nvfp4,
    scale_shape,
    scaled_mm_nvfp4,
    unavailable_reason,
)

__all__ = [
    "E2M1_MAX",
    "E4M3_MAX",
    "GLOBAL_RANGE",
    "REBUILD_HINT",
    "dequantize_nvfp4",
    "fixed_scale_one",
    "is_available",
    "linear_nvfp4",
    "per_tensor_decode_scale",
    "quantize_nvfp4",
    "scale_shape",
    "scaled_mm_nvfp4",
    "unavailable_reason",
]
