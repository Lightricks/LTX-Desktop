"""Shared capability probe for the CuTe DSL VAE kernels.
Private: each kernel module wraps this in its own named predicate
(``block_fna_available`` / ``na_attn_available``) and enforces it in its launcher.
"""

from __future__ import annotations

import functools

import torch

# The kernels need two separate Blackwell features, and one implies neither the other
# nor the architecture family:
# * ``tcgen05`` MMA, with the operands coming from SMEM.
# * **Tensor Memory**, where every accumulator lives -- the per-head attention output
#   for the whole KV loop plus the two Q@K accumulators, all 512 columns.
# Consumer Blackwell (sm_120/sm_121, e.g. RTX 5090) has tcgen05 MMA but no TMEM, so a
# tcgen05 test alone would pass there and then fail inside the JIT. Hopper and Ada have
# neither: they are not a slower fallback, the instructions are not in the ISA.
# Jetson Thor (sm_110) has both tcgen05 MMA and TMEM, same as datacenter Blackwell.
_TCGEN05_CAPS = frozenset({(10, 0), (10, 1), (10, 3), (11, 0), (12, 0), (12, 1)})
_TMEM_CAPS = frozenset({(10, 0), (10, 1), (10, 3), (11, 0)})

UNSUPPORTED_MESSAGE = (
    "the CuTe DSL VAE kernels need a GPU with both tcgen05 MMA and Tensor Memory "
    "(sm_100/sm_101/sm_103, e.g. B200 or GB200, or sm_110 Jetson Thor) and "
    "nvidia-cutlass-dsl installed; consumer Blackwell has tcgen05 but no TMEM, "
    "and these kernels hold all 512 TMEM columns"
)


@functools.lru_cache(maxsize=8)
def gpu_supports_dsl_kernels(device_index: int = 0) -> bool:
    """Whether CUDA, ``nvidia-cutlass-dsl``, tcgen05 MMA and TMEM are all present.
    Cached: the import probe and the device query are both too slow to repeat per
    forward, and neither answer can change within a process.
    """
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(device_index)
    if cap not in _TCGEN05_CAPS or cap not in _TMEM_CAPS:
        return False
    try:
        import cutlass.cute  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True
