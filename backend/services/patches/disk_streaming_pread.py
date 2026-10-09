"""Open disk-streaming checkpoints with safetensors' ``pread`` backend on Darwin.

``DiskTensorReader`` keeps a ``safe_open`` handle and ``get_tensor``s each block
during every denoise step. With no ``backend`` argument that handle is mmap.
On Apple Silicon mmap is about half the throughput of ``pread`` and varies run
to run once the checkpoint no longer fits in the page cache
(https://github.com/Lightricks/LTX-2/issues/304).

The same reader is the CUDA disk-streaming path (``OffloadMode.DISK``): Linux
hosts that cannot pin the transformer, and any other machine that streams
blocks from the checkpoint instead of keeping them resident. Pinned and
fully resident loads do not construct ``DiskTensorReader``. ``pread`` does
not shrink the allocation. It only changes the read. On Windows, safetensors
implements it with ``File::seek_read``.

That Windows read is slower than mmap. Measured on an RTX 5090 / Samsung
9100 PRO NVMe, 63.45 GiB RAM, streamed T2V, DISK forced, Sage on, cold
process (same bytes, same allocation peak):

- 720p/20s: mmap 232 s, pread 394.82 s (1.70x), 68.1 GiB read, alloc 14.15 GiB
- 1080p/10s: mmap 258 s, pread 420.11 s (1.63x), 74.6 GiB read, alloc 16.37 GiB

Windows DISK-vs-CPU evidence was measured with mmap (DISK ~0.79x CPU). Under
pread, DISK is ~1.4x slower than that CPU control. Linux DISK is unmeasured,
and the MPS result does not transfer. ``install`` therefore no-ops unless
``sys.platform == "darwin"``. The reader never sees the consuming device, so
the platform check is the scoping we can implement.

The issue's patch is ``SafetensorsStateDictLoader.load``. Desktop streaming does
not use that loader for the denoise loop, so this wraps ``DiskTensorReader``
instead.

Remove once ltx-core passes ``backend="pread"`` from ``DiskTensorReader`` on
Darwin only. An unconditional upstream ``pread`` would regress Windows.

Usage:
    import services.patches.disk_streaming_pread as disk_streaming_pread
    disk_streaming_pread.install()
"""

from __future__ import annotations

import inspect
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

_installed = False
_checked = False
_original_init: Any = None


def install() -> None:
    """Use ``backend="pread"`` for disk-streaming opens on Darwin. Idempotent.

    No-op on every other platform. Does not log. This runs at import time,
    before ``logging.basicConfig``, and an info record then is dropped. Call
    ``log_status`` once logging exists.
    """
    global _installed, _checked, _original_init
    if _installed:
        return
    if sys.platform != "darwin":
        _checked = True
        return

    import safetensors  # noqa: PLC0415

    from ltx_core.block_streaming.disk import DiskTensorReader  # noqa: PLC0415

    if "backend" not in inspect.signature(safetensors.safe_open).parameters:
        _checked = True
        return

    _original_init = DiskTensorReader.__init__

    def _init(self: DiskTensorReader, paths: list[str]) -> None:
        original_open = safetensors.safe_open

        def _open(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("backend", "pread")
            return original_open(*args, **kwargs)

        safetensors.safe_open = _open  # type: ignore[assignment]
        try:
            _original_init(self, paths)
        finally:
            safetensors.safe_open = original_open

    DiskTensorReader.__init__ = _init  # type: ignore[method-assign]
    _installed = True
    _checked = True


def log_status() -> None:
    """Report the disk-streaming read backend. Call after logging is configured."""
    if not _checked:
        return
    if _installed:
        logger.info("Disk streaming opens safetensors with backend=pread")
        return
    if sys.platform != "darwin":
        logger.info("Disk streaming keeps mmap; pread is Darwin-only")
        return
    logger.warning("safetensors.safe_open has no backend=; disk streaming stays on mmap")
