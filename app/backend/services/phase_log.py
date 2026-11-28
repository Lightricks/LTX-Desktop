"""Timing log for the slow steps of a 4K or 8K run (cutout bake, windowed video).

A run at 4K or 8K lasts minutes to hours. When it fails or stalls, the log must say which
step ran, how long it took, and how much memory the backend process held.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from inference.phase_log import rss_gb as rss_gb
from server_utils.heartbeat import log_heartbeat

PREFIX = "[alpha-gen]"


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
