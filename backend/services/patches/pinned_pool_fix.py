"""Monkey-patch: do not report pinned-host allocation failure as CUDA VRAM OOM.

``OffloadMode.CPU`` streaming pre-allocates one pinned host buffer per
transformer block via ``ltx_core.block_streaming.utils.alloc_buffer``. For the
22B distilled checkpoint that is ~23 GiB of page-locked RAM (more if PyTorch's
CachingHostAllocator rounds each block up to the next power of two).

``alloc_buffer`` first tries ``cudaHostRegister`` on a regular CPU tensor. On
Windows that call typically fails (pointer/size are not page-aligned for WDDM).
The failure is left as CUDA's sticky last error. The fallback
``torch.empty(..., pin_memory=True)`` then raises ``CUDA error: out of memory``
even with ~14 GiB VRAM free, and every later CUDA call (including
``cudaMemGetInfo``) fails until the process is restarted.

See https://github.com/Lightricks/LTX-Desktop/issues/141

This patch:
- On Windows, skips both ``cudaHostRegister`` and the CachingHostAllocator
  fallback and allocates pageable host memory. H2D copies stay correct; they
  just will not overlap compute.
- On other platforms, clears the sticky CUDA error after a failed register and
  falls back to pageable memory if ``pin_memory=True`` itself OOMs.

Remove once ltx-core ``alloc_buffer`` does not poison the CUDA context or
mis-report pinned-host failure as VRAM OOM.

Usage:
    import services.patches.pinned_pool_fix  # noqa: F401
"""

from __future__ import annotations

import logging
import sys

import torch
from ltx_core.block_streaming import utils as bs_utils

logger = logging.getLogger(__name__)


def _require_attr(name: str) -> None:
    # Explicit raise so this still fires under `python -O` (asserts are stripped).
    if not hasattr(bs_utils, name):
        raise RuntimeError(
            f"ltx_core.block_streaming.utils.{name} not found — patch needs updating."
        )


_require_attr("alloc_buffer")
_require_attr("_alloc_pinned_exact")

_orig_alloc_pinned_exact = bs_utils._alloc_pinned_exact
_windows_pageable_logged = False


def _clear_cuda_sticky_error() -> None:
    """Drop a leftover CUDA runtime error so later API calls are not poisoned."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.cudart().cudaGetLastError()
    except Exception:
        logger.debug("pinned_pool_fix: failed to clear CUDA last error", exc_info=True)


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cudaerrormemoryallocation" in msg


def _alloc_pinned_exact_cleared(nbytes: int) -> torch.Tensor | None:
    buf = _orig_alloc_pinned_exact(nbytes)
    if buf is None:
        # cudaHostRegister returns the error to Python but also sets the thread's
        # sticky last error. Clear it before any later CUDA call.
        _clear_cuda_sticky_error()
    return buf


def _patched_alloc_buffer(nbytes: int, device: torch.device | None, pin_memory: bool) -> torch.Tensor:
    """Like upstream ``alloc_buffer``, but never raises a misleading CUDA VRAM OOM."""
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False
    cpu_pin = pin_memory and (device is None or torch.device(device).type == "cpu")
    if cpu_pin and sys.platform == "win32":
        global _windows_pageable_logged
        if not _windows_pageable_logged:
            logger.warning(
                "Using pageable host memory for streaming weight buffers on Windows. "
                "ltx-core would pin via cudaHostRegister / CachingHostAllocator, which "
                "fails as a misleading CUDA OOM while VRAM is free (LTX-Desktop#141)."
            )
            _windows_pageable_logged = True
        return torch.empty(nbytes, dtype=torch.uint8, device=device, pin_memory=False)
    if cpu_pin:
        buf = _alloc_pinned_exact_cleared(nbytes)
        if buf is not None:
            return buf
        try:
            return torch.empty(nbytes, dtype=torch.uint8, device=device, pin_memory=True)
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            _clear_cuda_sticky_error()
            logger.warning(
                "Pinned host-memory allocation of %s bytes failed (%s). "
                "Falling back to pageable CPU memory. This is not a GPU VRAM shortage.",
                nbytes,
                exc,
            )
            pin_memory = False
    return torch.empty(nbytes, dtype=torch.uint8, device=device, pin_memory=pin_memory)


bs_utils.alloc_buffer = _patched_alloc_buffer
