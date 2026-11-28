"""Blockwise FP8/FP6 quantization: activation quantize/dequantize, weight
quantizers, FP6 pack/unpack, the FP8 GEMM, and the quantized linear layers.
Importing this subpackage needs ``triton`` (pulled in by ``torch`` on Linux) but
not the compiled ``ops_cpp`` / ``blockwise_cpp`` extensions -- those are resolved
lazily on first kernel call. The ``ltx_core.quantization.blockwise`` gate
surfaces a clean error when the extensions are absent.
"""

from ltx_kernels.blockwise.linear import BlockwiseFP6Linear, BlockwiseFP8Linear
from ltx_kernels.blockwise.ops.fp6_pack import fp6_pack_tensor, fp6_unpack_tensor
from ltx_kernels.blockwise.ops.gemm import blockwise_fp8_gemm
from ltx_kernels.blockwise.ops.quantize import (
    blockwise_dequantize,
    blockwise_quantize,
    blockwise_quantize_weights,
    fp6_blockwise_quantize_weights_torch,
    fp8_blockwise_quantize_weights_torch,
)

__all__ = [
    "BlockwiseFP6Linear",
    "BlockwiseFP8Linear",
    "blockwise_dequantize",
    "blockwise_fp8_gemm",
    "blockwise_quantize",
    "blockwise_quantize_weights",
    "fp6_blockwise_quantize_weights_torch",
    "fp6_pack_tensor",
    "fp6_unpack_tensor",
    "fp8_blockwise_quantize_weights_torch",
]
