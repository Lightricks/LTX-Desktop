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

The wait must not hold the GIL. ``torch.mps.synchronize()`` and ``torch.mps.Event.synchronize()``
both block in ``MPSStream::synchronize`` / ``waitUntilCompleted`` with the GIL held (torch 2.11.0).
Every other Python thread then stops for as long as the GPU needs to drain its queue: the asyncio
event loop (``/health`` answered after 9 to 16 s on an M4 Pro during Extend, so Electron logged
``Backend liveness probe failed``), the heartbeat thread, and ltx-core's read-ahead worker. The wait
here records an event on the same stream and polls ``Event.query()`` with a short sleep, then calls
``torch.mps.synchronize()`` as before (it returns at once). That keeps the same ordering (the event
signals after all earlier queued work), keeps the original call in the same place, and releases the
GIL for the long wait. A micro-benchmark with a second Python thread measured a 6 ms maximum gap with
polling, against a gap of the full wait (4.6 to 7.2 s) for both blocking calls, with no visible
increase in total time.

Only ``MpsTransport`` is patched. CUDA and CPU transports are untouched, so this is a no-op off Mac.

Remove once ltx-core waits for queued MPS compute before a transfer, and a 1080p stage 2 passes on a
48 GB Mac without this patch. Replace the polling wait with ``torch.mps.synchronize()`` only once
PyTorch releases the GIL in it (CUDA's ``synchronize`` already does).

Usage:
    import services.patches.mps_transfer_sync  # noqa: F401
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from ltx_core.loader import streaming_transport as _transport

_orig_transfer = _transport.MpsTransport.transfer

# A sleep of 1 ms takes about 1.2 ms on macOS. A wait that is already done never sleeps.
_POLL_INTERVAL_S = 0.001
# A block takes about 7 s at the slowest measured case (1080p stage 2, M4 Pro). A poll that
# outlasts this limit falls back to the blocking call, so a GPU error cannot spin forever.
_POLL_TIMEOUT_S = 60.0


def _wait_for_queued_compute() -> None:
    """Block until all MPS work queued so far has finished, without holding the GIL.

    The event poll does the long wait with the GIL released. ``torch.mps.synchronize()`` then runs
    exactly as before, so its commit and flush semantics stay. It returns at once, because the
    queue is already empty (measured 0.1 ms), so the GIL is held only for that moment.

    If the event is still not signalled after ``_POLL_TIMEOUT_S``, the poll stops and the same
    ``torch.mps.synchronize()`` call blocks (with the GIL), as it did before the poll existed.
    """
    event = torch.mps.Event()
    event.record()
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while not event.query() and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)
    torch.mps.synchronize()


@contextmanager
def _transfer_after_compute(self: Any, after: Any) -> Iterator[Any]:
    """Like upstream ``MpsTransport.transfer``, but only after all queued MPS work has finished."""
    _wait_for_queued_compute()
    with _orig_transfer(self, after) as handle:
        yield handle


if not hasattr(_orig_transfer, "__wrapped__"):
    raise RuntimeError("MpsTransport.transfer is no longer a context manager function — patch needs updating.")
_transport.MpsTransport.transfer = _transfer_after_compute  # type: ignore[method-assign]
