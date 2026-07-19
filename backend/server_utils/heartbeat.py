"""Generic progress heartbeat for long, callback-less blocking calls (generation).

A daemon thread logs elapsed time on an interval while the wrapped block runs, so a
long generation is distinguishable from a hang. It also logs GPU memory per device:

- CUDA: allocated / reserved / peak vs total device memory
- MPS: torch-tracked / driver / max memory + the active mps-sdpa attention backend
  counts (surfaces the pyobjc-fallback leak — driver climbs while torch stays flat)
- CPU-only: elapsed time only

Used by both plain video generation and IC-LoRA generation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Event, Thread

from mps_prebuilt_ext import mps_memory_sample, reset_mps_sdpa_stats
from server_utils.units import gib

logger = logging.getLogger(__name__)


def _cuda_memory_sample() -> str | None:
    """One-line CUDA memory snapshot, or None when CUDA isn't the active device."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        return None
    try:
        free, total = torch.cuda.mem_get_info()
        return (
            f"allocated={gib(torch.cuda.memory_allocated()):.2f}GiB "
            f"reserved={gib(torch.cuda.memory_reserved()):.2f}GiB "
            f"peak={gib(torch.cuda.max_memory_allocated()):.2f}GiB "
            f"free={gib(free):.2f}/{gib(total):.2f}GiB"
        )
    except Exception:
        logger.warning("[heartbeat] cuda sample failed", exc_info=True)
        return None


def _device_sample() -> str | None:
    """GPU memory snapshot for whichever accelerator is active (CUDA or MPS), else None."""
    return _cuda_memory_sample() or mps_memory_sample()


def _reset_stats() -> None:
    """Reset per-run counters so a heartbeat's peak/backend counts start fresh."""
    reset_mps_sdpa_stats()  # no-op off MPS
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


@contextmanager
def log_heartbeat(label: str, interval_s: float = 15.0) -> Iterator[None]:
    """Tick elapsed time (+ GPU memory when available) while the wrapped block runs."""
    _reset_stats()
    stop = Event()
    start = time.perf_counter()

    def _beat() -> None:
        while not stop.wait(interval_s):
            elapsed = time.perf_counter() - start
            sample = _device_sample()
            if sample is not None:
                logger.info("[heartbeat] %s %.0fs | %s", label, elapsed, sample)
            else:
                logger.info("[heartbeat] %s still running… %.0fs elapsed", label, elapsed)

    thread = Thread(target=_beat, name="generation-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        elapsed = time.perf_counter() - start
        sample = _device_sample()
        if sample is not None:
            logger.info("[heartbeat] %s done %.0fs | %s", label, elapsed, sample)
        else:
            logger.info("[heartbeat] %s done %.0fs", label, elapsed)
