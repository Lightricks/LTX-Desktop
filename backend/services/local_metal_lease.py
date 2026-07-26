"""Cross-product advisory lease for heavy local Apple Silicon inference."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import ContextManager, Literal, TypedDict, cast

logger = logging.getLogger(__name__)

LOCAL_METAL_LOCK_PATH = (
    Path.home() / "Library" / "Application Support" / "LTX Shared" / "local-metal.lock"
)
_POLL_SECONDS = 0.25


class LocalMetalLeaseSnapshot(TypedDict):
    status: Literal["idle", "waiting", "held"]
    reason: str | None
    waited_seconds: float
    owner: dict[str, object] | None


_snapshot_lock = threading.Lock()
_snapshot: LocalMetalLeaseSnapshot = {
    "status": "idle",
    "reason": None,
    "waited_seconds": 0.0,
    "owner": None,
}


def get_local_metal_lease_snapshot() -> LocalMetalLeaseSnapshot:
    with _snapshot_lock:
        return {
            "status": _snapshot["status"],
            "reason": _snapshot["reason"],
            "waited_seconds": _snapshot["waited_seconds"],
            "owner": dict(_snapshot["owner"]) if _snapshot["owner"] is not None else None,
        }


def _set_snapshot(
    status: Literal["idle", "waiting", "held"],
    *,
    reason: str | None = None,
    waited_seconds: float = 0.0,
    owner: dict[str, object] | None = None,
) -> None:
    with _snapshot_lock:
        _snapshot.update(
            status=status,
            reason=reason,
            waited_seconds=waited_seconds,
            owner=owner,
        )


def _read_owner(fd: int) -> dict[str, object] | None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        payload = os.read(fd, 16 * 1024).decode("utf-8")
        parsed = json.loads(payload)
        return cast(dict[str, object], parsed) if isinstance(parsed, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _emit(event: str, **fields: object) -> None:
    logger.info("%s", json.dumps({"event": event, **fields}, sort_keys=True))


@contextmanager
def local_metal_lease(
    *,
    job_id: str,
    workload: str,
    reason: str,
    is_cancelled: Callable[[], bool],
    on_wait: Callable[[float, dict[str, object] | None], None] | None = None,
) -> Iterator[None]:
    """Acquire the shared Metal lock, polling so cancellation remains responsive."""
    if sys.platform != "darwin":
        yield
        return

    import fcntl

    path = LOCAL_METAL_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    os.chmod(path, 0o600)
    started = time.monotonic()
    acquired = False
    metadata: dict[str, object] = {
        "schema": "ltx.local-metal-lock.v1",
        "product": "LTX Desktop",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "job_id": job_id,
        "workload": workload,
        "reason": reason,
        "acquired_at_utc": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
    }
    try:
        while True:
            if is_cancelled():
                raise RuntimeError("Generation cancelled while waiting for the local Metal accelerator")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                waited = time.monotonic() - started
                owner = _read_owner(fd)
                _set_snapshot(
                    "waiting",
                    reason=reason,
                    waited_seconds=waited,
                    owner=owner,
                )
                _emit(
                    "local_metal_lock_wait",
                    path=str(path),
                    product="LTX Desktop",
                    pid=os.getpid(),
                    job_id=job_id,
                    reason=reason,
                    waited_seconds=round(waited, 3),
                    owner=owner,
                )
                if on_wait is not None:
                    on_wait(waited, owner)
                time.sleep(_POLL_SECONDS)

        waited = time.monotonic() - started
        encoded = json.dumps(metadata, sort_keys=True).encode("utf-8")
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, encoded)
        os.fsync(fd)
        _set_snapshot("held", reason=reason, waited_seconds=waited, owner=metadata)
        _emit(
            "local_metal_lock_acquired",
            path=str(path),
            **metadata,
            waited_seconds=round(waited, 3),
        )
        yield
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
            _emit(
                "local_metal_lock_released",
                path=str(path),
                product="LTX Desktop",
                pid=os.getpid(),
                job_id=job_id,
                reason=reason,
            )
        os.close(fd)
        _set_snapshot("idle")


class LocalMetalLeaseHandle:
    """Explicit-lifetime wrapper for handlers whose existing try/finally owns cleanup."""

    def __init__(self, lease: ContextManager[None]) -> None:
        self._lease = lease
        self._closed = False
        lease.__enter__()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._lease.__exit__(None, None, None)
