"""Tests for the generation proof-of-life file read by Electron's liveness monitor."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import AbstractContextManager

from server_utils import generation_liveness, heartbeat
from server_utils.generation_liveness import (
    GENERATION_HEARTBEAT_FILENAME,
    liveness_hold,
)


def test_liveness_hold_writes_pid_and_removes_file_on_exit(tmp_path, monkeypatch) -> None:
    """A leaked file would grace a hung backend forever — exit must remove it."""
    monkeypatch.setenv("LTX_APP_DATA_DIR", str(tmp_path))
    with liveness_hold():
        payload = json.loads(
            (tmp_path / GENERATION_HEARTBEAT_FILENAME).read_text(encoding="utf-8")
        )
        assert payload["pid"] == os.getpid()
        assert payload["updatedAtMs"] <= int(time.time() * 1000)
    assert not (tmp_path / GENERATION_HEARTBEAT_FILENAME).exists()


def test_nested_liveness_hold_keeps_file_until_outer_exits(tmp_path, monkeypatch) -> None:
    """Without a refcount the inner exit deletes the file and the outer job loses its grace."""
    monkeypatch.setenv("LTX_APP_DATA_DIR", str(tmp_path))
    with liveness_hold():
        with liveness_hold():
            pass
        assert (tmp_path / GENERATION_HEARTBEAT_FILENAME).exists()
    assert not (tmp_path / GENERATION_HEARTBEAT_FILENAME).exists()


def test_liveness_hold_without_app_data_dir_is_noop(monkeypatch) -> None:
    """Headless runs without the env var must not fail generation on a missing dir."""
    monkeypatch.delenv("LTX_APP_DATA_DIR", raising=False)
    with liveness_hold():
        pass


def test_liveness_hold_refreshes_file_while_held(tmp_path, monkeypatch) -> None:
    """A write-once file goes stale mid-job and drops the grace — the hold must refresh it."""
    monkeypatch.setenv("LTX_APP_DATA_DIR", str(tmp_path))
    with liveness_hold(interval_s=0.02):
        path = tmp_path / GENERATION_HEARTBEAT_FILENAME
        first = json.loads(path.read_text(encoding="utf-8"))["updatedAtMs"]
        deadline = time.monotonic() + 5.0
        refreshed = first
        while refreshed == first and time.monotonic() < deadline:
            time.sleep(0.02)
            refreshed = json.loads(path.read_text(encoding="utf-8"))["updatedAtMs"]
        assert refreshed > first


def test_log_heartbeat_holds_liveness_file(tmp_path, monkeypatch) -> None:
    """The inference heartbeat must carry the liveness hold or executions run without grace."""
    monkeypatch.setenv("LTX_APP_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(heartbeat, "_device_sample", lambda: None)
    with heartbeat.log_heartbeat("test-label", interval_s=1000.0):
        assert (tmp_path / GENERATION_HEARTBEAT_FILENAME).exists()
    assert not (tmp_path / GENERATION_HEARTBEAT_FILENAME).exists()


def test_last_exit_during_new_hold_keeps_the_new_file(tmp_path, monkeypatch) -> None:
    """The last exit unlinks outside the lock: a hold starting mid-teardown loses its grace file."""
    monkeypatch.setenv("LTX_APP_DATA_DIR", str(tmp_path))
    path = tmp_path / GENERATION_HEARTBEAT_FILENAME
    entered: list[AbstractContextManager[None]] = []

    class _JoinStartsHold(threading.Thread):
        def join(self, timeout: float | None = None) -> None:
            if not entered:
                hold = liveness_hold()
                hold.__enter__()
                entered.append(hold)
            super().join(timeout)

    monkeypatch.setattr(generation_liveness, "Thread", _JoinStartsHold)
    with liveness_hold():
        pass
    assert len(entered) == 1
    assert path.exists()
    entered[0].__exit__(None, None, None)
    assert not path.exists()
