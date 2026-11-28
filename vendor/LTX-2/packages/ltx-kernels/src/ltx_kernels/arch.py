"""GPU architecture detection for blockwise kernel dispatch."""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=8)
def _arch_for_device(device_index: int) -> str:
    major, minor = torch.cuda.get_device_capability(device_index)
    if major == 8 and (minor >= 0 and minor < 9):
        return "ampere"
    if major == 8 and minor == 9:
        return "ada"
    if major == 9 and minor == 0:
        return "hopper"
    if major in {10, 12}:
        return "blackwell"
    raise NotImplementedError(f"Unsupported GPU compute capability sm_{major}{minor}.")


def get_device_arch() -> str:
    """Return a coarse architecture name for the current CUDA device.
    Used to pick the FP8 GEMM kernel variant (SM89 vs SM90) at runtime.
    Cached per device index: capability does not change within a process.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; blockwise kernels require a CUDA device.")
    return _arch_for_device(torch.cuda.current_device())
