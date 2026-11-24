"""Monkey-patch: page-locked host buffers must degrade to pageable memory, never crash.

``OffloadMode.CPU`` and ``OffloadMode.DISK`` streaming on CUDA ask ``ltx_core.loader.buffers.host_bytes``
for page-locked host buffers (``pin=True``). For the 22B distilled checkpoint that is ~23 GiB of
RAM-resident blocks, or the staging pool of disk streaming.

``host_bytes`` builds a pinned buffer of 2 MiB or more from an anonymous ``mmap`` created with
``mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS`` and registers it with ``cudaHostRegister``. Those mmap
flags do not exist on Windows, so every pinned request raises ``AttributeError`` there. Where the
flags exist, a failed registration raises ``RuntimeError`` and leaves CUDA's sticky last error set;
a later fallback that pins through torch's caching host allocator can then surface as
``CUDA error: out of memory`` with VRAM free, and every later CUDA call fails until restart
(https://github.com/Lightricks/LTX-Desktop/issues/141).

This patch wraps ``host_bytes``:
- On Windows, requests of ``_EXACT_PIN_BYTES`` or more skip pinning and return pageable host
  memory. H2D copies stay correct; they just do not overlap compute. Smaller requests take
  upstream's ``torch.empty(pin_memory=True)`` path, which works on Windows.
- Elsewhere it uses upstream, and when pinning fails it clears the sticky CUDA error and returns
  pageable memory. An ``OSError`` from the anonymous mapping (``vm.overcommit_memory=2``) counts
  as a failed pin too.

``registry`` and ``state_loader`` record placement from the result (``buffer.is_pinned()``), so
they stay correct with a pageable buffer. ``streaming_builder`` records the placement it
requested instead; on Windows that says pinned for a pageable buffer. Nothing in ltx-core
asserts on it.

Not covered: ``DeviceStager`` (``ltx_core/loader/raw.py``) pins two 64 MiB staging buckets per
device with ``torch.empty(pin_memory=True)``, which goes through torch's host allocator and
not ``host_bytes``. It normally works on Windows, but if it fails the error can surface as the
misleading "CUDA out of memory" of LTX-Desktop#141 and leave the sticky error set.

Remove once ltx-core ``host_bytes`` works on Windows and does not leave a sticky CUDA error.

Usage:
    import services.patches.pinned_pool_fix  # noqa: F401
"""

from __future__ import annotations

import logging
import sys

import torch
from ltx_core.loader import buffers as _buffers
from ltx_core.loader import registry as _registry
from ltx_core.loader import state_loader as _state_loader
from ltx_core.loader import streaming_builder as _streaming_builder

logger = logging.getLogger(__name__)

_REBOUND_MODULES = (_registry, _state_loader, _streaming_builder)

_orig_host_bytes = _buffers.host_bytes
_windows_pageable_logged = False


def _clear_cuda_sticky_error() -> None:
    """Drop a leftover CUDA runtime error so later API calls are not poisoned."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.cudart().cudaGetLastError()
    except Exception:
        logger.debug("pinned_pool_fix: failed to clear CUDA last error", exc_info=True)


def is_cuda_oom_error(exc: BaseException) -> bool:
    """Whether an exception is a CUDA out-of-memory, by message.

    Public: the queue runner classifies failures with the same definition.
    torch surfaces OOMs as ``OutOfMemoryError`` but also as plain
    ``RuntimeError`` with the message, so the type alone under-counts.
    Deliberately message-tight — telemetry must not over-label.
    """
    msg = str(exc).lower()
    return "out of memory" in msg or "cudaerrormemoryallocation" in msg


def _is_pin_failure(exc: BaseException) -> bool:
    if isinstance(exc, OSError):
        return True
    return isinstance(exc, RuntimeError) and ("page-lock" in str(exc) or is_cuda_oom_error(exc))


def _patched_host_bytes(nbytes: int, *, pin: bool = False) -> torch.Tensor:
    """Like upstream ``host_bytes``, but a pin request made through it never fails the load."""
    if not pin:
        return _orig_host_bytes(nbytes, pin=False)
    if sys.platform == "win32" and nbytes >= _buffers._EXACT_PIN_BYTES:  # pyright: ignore[reportPrivateUsage]
        global _windows_pageable_logged
        if not _windows_pageable_logged:
            logger.warning(
                "Using pageable host memory for streaming weight buffers on Windows. "
                "ltx-core would pin through an anonymous mmap that Windows does not support "
                "(LTX-Desktop#141)."
            )
            _windows_pageable_logged = True
        return torch.empty(nbytes, dtype=torch.uint8)
    try:
        return _orig_host_bytes(nbytes, pin=True)
    except (RuntimeError, OSError) as exc:
        if not _is_pin_failure(exc):
            raise
        _clear_cuda_sticky_error()
        logger.warning(
            "Pinned host-memory allocation of %s bytes failed (%s). "
            "Falling back to pageable CPU memory. This is not a GPU VRAM shortage.",
            nbytes,
            exc,
        )
        return torch.empty(nbytes, dtype=torch.uint8)


for _module in _REBOUND_MODULES:
    if getattr(_module, "host_bytes", None) is not _orig_host_bytes:
        raise RuntimeError(f"{_module.__name__}.host_bytes is not ltx_core.loader.buffers.host_bytes — patch needs updating.")
    _module.host_bytes = _patched_host_bytes  # type: ignore[attr-defined]
_buffers.host_bytes = _patched_host_bytes

_stale = sorted(
    name
    for name, module in list(sys.modules.items())
    if name.startswith("ltx_core.") and getattr(module, "host_bytes", None) is _orig_host_bytes
)
if _stale:
    raise RuntimeError(f"{', '.join(_stale)} still hold the unpatched host_bytes — patch needs updating.")
