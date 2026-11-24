"""Blockwise FP8 GEMM custom op + architecture dispatch.
The FP8 GEMM kernels live in the ``blockwise_cpp`` extension. The kernel is
resolved on first call (architecture-specific) so importing this module needs
neither a CUDA device nor the compiled extension.
"""

from typing import List, Optional

import torch

from ltx_kernels.arch import get_device_arch

# Lazily-initialized, architecture-specific FP8 GEMM callable.
_fp8_gemm = None


def get_fp8_gemm_nt():
    """Runtime GPU architecture dispatch for FP8 GEMM."""
    arch = get_device_arch()
    if arch in ["ada", "blackwell"]:
        # Ada/Blackwell use SM89 kernels (GeForce path)
        from blockwise_cpp import fp8_gemm_nt_sm89

        def _func(a, b, d, bias=None, c=None, num_sms=132, use_fast_accum=True):
            return fp8_gemm_nt_sm89(a, b, d, bias=bias, use_fast_accum=use_fast_accum)

        return _func
    elif arch == "hopper":
        # Hopper uses SM90 kernels (H100 path)
        from blockwise_cpp import fp8_gemm_nt_sm90

        def _func(a, b, d, bias=None, c=None, num_sms=132, use_fast_accum=True):
            return fp8_gemm_nt_sm90(a, b, d, bias=bias, c=c, num_sms=num_sms)

        return _func
    else:
        raise RuntimeError(f"Unsupported GPU architecture: {arch}")


def _fp8_gemm_dispatch(a, b, d, bias=None, c=None, num_sms=132, use_fast_accum=True):
    global _fp8_gemm
    if _fp8_gemm is None:
        _fp8_gemm = get_fp8_gemm_nt()
    return _fp8_gemm(a, b, d, bias=bias, c=c, num_sms=num_sms, use_fast_accum=use_fast_accum)


@torch.library.custom_op("blockwise::fp8_gemm", mutates_args=())
def blockwise_fp8_gemm(
    a: List[torch.Tensor],
    b: List[torch.Tensor],
    bias: Optional[torch.Tensor],
    use_fast_accum: bool,
) -> torch.Tensor:
    d = torch.empty(a[0].shape[0], b[0].shape[0], dtype=torch.bfloat16, device=b[0].device)
    _fp8_gemm_dispatch(a, b, d, bias=bias, use_fast_accum=use_fast_accum)
    return d


@blockwise_fp8_gemm.register_fake
def blockwise_fp8_gemm_fake(
    a: List[torch.Tensor],
    b: List[torch.Tensor],
    bias: Optional[torch.Tensor],
    use_fast_accum: bool,
) -> torch.Tensor:
    return torch.empty(a[0].shape[0], b[0].shape[0], dtype=torch.bfloat16, device=b[0].device)
