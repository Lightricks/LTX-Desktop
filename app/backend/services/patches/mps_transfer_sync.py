"""Monkey-patch: wait for queued GPU compute before each MPS weight transfer.

ltx-core 1.6.0 streams transformer weights through ``MpsTransport.transfer``
(``ltx_core/loader/streaming_transport.py``). A read-ahead worker stages the next units while the
consumer thread computes. Each transfer is ordered after earlier compute only through
``torch.mps.Event`` markers on MPS's single queue. ltx-core 1.5.0 had no read-ahead and no
host-blocking event wait on a worker thread.

On an M4 Pro (48 GB), a 1080p stage 2 (about 32k video tokens, about 7 s per block) then stalled in
the first blocks of step 1. The worker blocked in ``torch.mps.Event.synchronize()``, the event never
signalled, and Metal ended a command buffer with ``kIOGPUCommandBufferCallbackErrorTimeout``. The
backend then hung until it was killed. 720p and stage 1 were not affected. An M5 Max (128 GB), which
runs a block in about 2.6 s, did not hit it.

Measured on the M4 Pro (same case, same seed, ``mpsgraph_zc`` active, no swap growth):

- No sync: fails at stage 2 block 385 to 389 (two runs).
- ``torch.mps.synchronize()`` only after the copy: fails at block 387.
- ``torch.mps.synchronize()`` before the copy, with or without one after it: stage 2 passes, at
  about 320 s per step (1.5.0: 319 s).

This patch adds the sync before the copy. Why a missing wait before the copy produces an event that
never signals is not proven. The measurements above are what the patch rests on.

Only ``MpsTransport`` is patched. CUDA and CPU transports are untouched, so this is a no-op off Mac.

Remove once ltx-core waits for queued MPS compute before a transfer, and a 1080p stage 2 passes on a
48 GB Mac without this patch.

Usage:
    import services.patches.mps_transfer_sync  # noqa: F401
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from ltx_core.loader import streaming_transport as _transport

_orig_transfer = _transport.MpsTransport.transfer


@contextmanager
def _transfer_after_compute(self: Any, after: Any) -> Iterator[Any]:
    """Like upstream ``MpsTransport.transfer``, but only after all queued MPS work has finished."""
    torch.mps.synchronize()
    with _orig_transfer(self, after) as handle:
        yield handle


if not hasattr(_orig_transfer, "__wrapped__"):
    raise RuntimeError("MpsTransport.transfer is no longer a context manager function — patch needs updating.")
_transport.MpsTransport.transfer = _transfer_after_compute  # type: ignore[method-assign]
