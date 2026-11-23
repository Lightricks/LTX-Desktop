"""Whole-run CUDA reserved peak and lowest free VRAM, across per-phase peak resets.

Phase logs, the IC-LoRA pipeline, and the heartbeat each reset torch's peak
counter so their own lines read one phase. Device calibration needs the peak
of the whole run, so every reset goes through ``fold_and_reset_cuda_peak``,
which folds the phase peak into the run first. Process-wide: one GPU job runs
at a time.
"""

from __future__ import annotations

import logging
from threading import Lock

import torch

logger = logging.getLogger(__name__)


class _RunWatch:
    def __init__(self) -> None:
        self.lock = Lock()
        self.active = False
        self.peak_reserved_bytes: int | None = None
        self.min_free_bytes: int | None = None

    def fold_peak(self, peak: int | None) -> None:
        if not self.active or peak is None:
            return
        current = self.peak_reserved_bytes
        self.peak_reserved_bytes = peak if current is None else max(current, peak)

    def fold_free(self, free: int | None) -> None:
        if not self.active or free is None:
            return
        current = self.min_free_bytes
        self.min_free_bytes = free if current is None else min(current, free)


_watch = _RunWatch()


def _cuda_ready() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _max_reserved(device: torch.device | None) -> int | None:
    if not _cuda_ready():
        return None
    try:
        return int(torch.cuda.max_memory_reserved(device))
    except Exception:
        logger.debug("run peak read failed", exc_info=True)
        return None


def _reset_peak(device: torch.device | None) -> None:
    if not _cuda_ready():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        logger.debug("peak reset failed", exc_info=True)


def _free_now() -> int | None:
    if not _cuda_ready():
        return None
    try:
        return int(torch.cuda.mem_get_info()[0])
    except Exception:
        logger.debug("free VRAM read failed", exc_info=True)
        return None


def begin_run() -> None:
    with _watch.lock:
        _watch.active = True
        _watch.peak_reserved_bytes = None
        _watch.min_free_bytes = None
        _reset_peak(None)


def fold_and_reset_cuda_peak(device: torch.device | None = None) -> None:
    with _watch.lock:
        _watch.fold_peak(_max_reserved(device))
        _reset_peak(device)


def note_free_sample(free_bytes: int) -> None:
    with _watch.lock:
        _watch.fold_free(free_bytes)


def end_run() -> tuple[int | None, int | None]:
    """(peak reserved, lowest free) over the run; None for a probe that never read."""
    with _watch.lock:
        _watch.fold_peak(_max_reserved(None))
        _watch.fold_free(_free_now())
        result = (_watch.peak_reserved_bytes, _watch.min_free_bytes)
        _watch.active = False
        return result
