"""Timing / peak-VRAM log for a GPU step. No desktop heartbeat.

Start, done, and failure lines include this process and its children so a
stalled 4K or 8K run shows host memory, not only the CUDA peak.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import psutil

from inference.run_peak import fold_and_reset_cuda_peak

if TYPE_CHECKING:
    import torch


def rss_gb() -> float:
    """Memory of this process and its children. ffmpeg holds 10 GB or more at 8K."""
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            continue
    return total / 1e9


@contextmanager
def measured_gpu_phase(
    logger: logging.Logger,
    label: str,
    *,
    device: "torch.device",
    prefix: str,
    estimate_gib: float | None = None,
) -> Iterator[None]:
    """Log start/end of one GPU step and, on CUDA, the peak VRAM of that step."""
    import torch

    started = time.monotonic()
    logger.info("%s %s: start (memory %.1f GB)", prefix, label, rss_gb())
    cuda = device.type == "cuda" and torch.cuda.is_available()
    if cuda:
        fold_and_reset_cuda_peak(device)
    try:
        yield
    except BaseException as exc:
        logger.warning(
            "%s %s: %s after %.1fs (memory %.1f GB)",
            prefix,
            label,
            type(exc).__name__,
            time.monotonic() - started,
            rss_gb(),
        )
        raise
    else:
        logger.info(
            "%s %s: done in %.1fs (memory %.1f GB)",
            prefix,
            label,
            time.monotonic() - started,
            rss_gb(),
        )
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
