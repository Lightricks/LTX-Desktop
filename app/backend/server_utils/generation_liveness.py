"""Proof-of-life file for local generation, read by Electron's liveness monitor.

A local generation can hold the Python GIL for long, uninterrupted stretches
(MPS model builds release it far less eagerly than CUDA inference), which
starves the asyncio event loop that serves ``/health`` — the backend isn't
hung, it's just busy, but the HTTP liveness probe can't tell the difference
and would otherwise SIGTERM it mid-generation. While any ``liveness_hold``
is open, a daemon thread refreshes ``generation-heartbeat.json`` under
``LTX_APP_DATA_DIR`` every interval; Electron skips the kill while the file
is fresh and carries the backend's pid. Refcounted: overlapping holds
(video + remote queue runners in one process) share one writer, and the
file is removed only when the last hold exits, so a hung backend with no
live generation still fails the probe and restarts.

Writes are atomic (temp file + rename): a torn read parses as missing and
would drop the grace mid-stall. Telemetry must never break generation, so
every filesystem failure is swallowed to a debug log. Without
``LTX_APP_DATA_DIR`` (unit tests, headless runs) the hold is a no-op.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock, Thread

logger = logging.getLogger(__name__)

GENERATION_HEARTBEAT_FILENAME = "generation-heartbeat.json"
HEARTBEAT_TOUCH_INTERVAL_S = 15.0

_lock = Lock()
_held = 0
_stop: Event | None = None
_thread: Thread | None = None


def _heartbeat_path() -> Path | None:
    app_data = os.environ.get("LTX_APP_DATA_DIR")
    if not app_data:
        return None
    return Path(app_data) / GENERATION_HEARTBEAT_FILENAME


def _write_heartbeat(path: Path) -> None:
    """Atomically refresh the heartbeat file. Never raises."""
    try:
        payload = json.dumps({"pid": os.getpid(), "updatedAtMs": int(time.time() * 1000)})
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        logger.debug("[liveness] heartbeat write failed", exc_info=True)


def _remove_heartbeat(path: Path) -> None:
    """Best-effort removal. Never raises."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("[liveness] heartbeat removal failed", exc_info=True)


def _touch_loop(path: Path, stop: Event, interval_s: float) -> None:
    while not stop.wait(interval_s):
        _write_heartbeat(path)


@contextmanager
def liveness_hold(*, interval_s: float = HEARTBEAT_TOUCH_INTERVAL_S) -> Iterator[None]:
    """Hold a proof-of-life file for the wrapped generation span.

    The first hold writes the file immediately (no interval wait before the
    grace exists) and starts the refresher; the last exit stops it and
    removes the file so a stale hold can never grace a hung backend.
    """
    path = _heartbeat_path()
    if path is None:
        yield
        return
    global _held, _stop, _thread
    with _lock:
        _held += 1
        if _held == 1:
            _write_heartbeat(path)
            stop = Event()
            _stop = stop
            thread = Thread(
                target=_touch_loop,
                args=(path, stop, interval_s),
                name="generation-liveness",
                daemon=True,
            )
            _thread = thread
            thread.start()
    try:
        yield
    finally:
        with _lock:
            _held -= 1
            last = _held == 0
            stop = _stop if last else None
            thread = _thread if last else None
            if last:
                _stop = None
                _thread = None
        if stop is not None and thread is not None:
            stop.set()
            thread.join(timeout=2.0)
            # Re-check under the lock: a new hold may have started (and
            # written its own file) while this exit was outside the lock.
            # Unlinking unconditionally would steal that hold's grace until
            # its writer recreates the file on the next interval tick.
            with _lock:
                if _held == 0:
                    _remove_heartbeat(path)
