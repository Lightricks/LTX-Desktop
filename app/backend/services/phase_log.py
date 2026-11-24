"""Timing log for the slow steps of a 4K or 8K run (cutout bake, windowed video).

A run at 4K or 8K lasts minutes to hours. When it fails or stalls, the log must say which
step ran, how long it took, and how much memory the backend process held.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import psutil

from server_utils.heartbeat import log_heartbeat
from services.gpu_info.run_peak import fold_and_reset_cuda_peak

if TYPE_CHECKING:
    import torch

PREFIX = "[alpha-gen]"


def rss_gb() -> float:
    """Memory of the backend and its child processes (ffmpeg holds 10 GB or more at 8K)."""
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            continue
    return total / 1e9


@contextmanager
def logged_phase(
    logger: logging.Logger, label: str, *, heartbeat: bool = False, prefix: str = PREFIX
) -> Iterator[None]:
    """Log the start, the end and the time of one step. A failure logs a warning, no trace.

    ``heartbeat`` also ticks every 15 s while the step runs. The boundary owns the traceback.
    """
    started = time.monotonic()
    logger.info("%s %s: start (memory %.1f GB)", prefix, label, rss_gb())
    try:
        if heartbeat:
            with log_heartbeat(f"{prefix} {label}"):
                yield
        else:
            yield
    except BaseException as exc:
        logger.warning(
            "%s %s: %s after %.1fs (memory %.1f GB)",
            prefix, label, type(exc).__name__, time.monotonic() - started, rss_gb(),
        )
        raise
    logger.info("%s %s: done in %.1fs (memory %.1f GB)", prefix, label, time.monotonic() - started, rss_gb())


@contextmanager
def measured_gpu_phase(
    logger: logging.Logger,
    label: str,
    *,
    device: "torch.device",
    prefix: str,
    estimate_gib: float | None = None,
) -> Iterator[None]:
    """``logged_phase`` with a heartbeat, plus the peak VRAM of the step next to the budget estimate.

    The estimate is the job-budget curve. Real peaks from a 4K run refit it.
    """
    import torch

    cuda = device.type == "cuda" and torch.cuda.is_available()
    if cuda:
        fold_and_reset_cuda_peak(device)
    try:
        with logged_phase(logger, label, heartbeat=True, prefix=prefix):
            yield
    finally:
        if cuda:
            logger.info(
                "%s %s: peak VRAM allocated %.1f GiB, reserved %.1f GiB (budget estimate %s)",
                prefix,
                label,
                torch.cuda.max_memory_allocated(device) / 1024**3,
                torch.cuda.max_memory_reserved(device) / 1024**3,
                "none" if estimate_gib is None else f"{estimate_gib:.1f} GiB",
            )
