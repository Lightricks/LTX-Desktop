"""Tests for the generation heartbeat's CPU-only and device-sample branches."""

from __future__ import annotations

import logging

import pytest
import torch

from server_utils import heartbeat


def _heartbeat_records(caplog, *, contains: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "server_utils.heartbeat" and contains in r.getMessage()]


def test_cuda_memory_sample_none_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert heartbeat.cuda_memory_sample() is None


def test_device_sample_none_when_no_accelerator(monkeypatch) -> None:
    monkeypatch.setattr(heartbeat, "cuda_memory_sample", lambda: None)
    monkeypatch.setattr(heartbeat, "mps_memory_sample", lambda: None)
    assert heartbeat._device_sample() is None


def test_log_heartbeat_emits_done_line_on_cpu_only(caplog, monkeypatch) -> None:
    """Regression test: CPU-only runs used to never log a completion line."""
    monkeypatch.setattr(heartbeat, "_device_sample", lambda: None)
    caplog.set_level(logging.INFO, logger="server_utils.heartbeat")

    with heartbeat.log_heartbeat("test-label", interval_s=1000.0):
        pass

    done_records = _heartbeat_records(caplog, contains="test-label done")
    assert len(done_records) == 1


def test_log_heartbeat_emits_failed_line_when_block_raises(caplog, monkeypatch) -> None:
    monkeypatch.setattr(heartbeat, "_device_sample", lambda: None)
    caplog.set_level(logging.INFO, logger="server_utils.heartbeat")

    with pytest.raises(RuntimeError, match="boom"):
        with heartbeat.log_heartbeat("test-label", interval_s=1000.0):
            raise RuntimeError("boom")

    assert _heartbeat_records(caplog, contains="test-label failed")
    assert not _heartbeat_records(caplog, contains="test-label done")


def test_log_heartbeat_includes_device_sample_when_available(caplog, monkeypatch) -> None:
    monkeypatch.setattr(heartbeat, "_device_sample", lambda: "fake=1.00GiB")
    caplog.set_level(logging.INFO, logger="server_utils.heartbeat")

    with heartbeat.log_heartbeat("test-label", interval_s=1000.0):
        pass

    done_records = _heartbeat_records(caplog, contains="test-label done")
    assert len(done_records) == 1
    assert "fake=1.00GiB" in done_records[0].getMessage()


def test_nested_log_heartbeat_does_not_start_a_second_ticker(caplog, monkeypatch) -> None:
    monkeypatch.setattr(heartbeat, "_device_sample", lambda: None)
    caplog.set_level(logging.INFO, logger="server_utils.heartbeat")
    started: list[str] = []

    class _SpyThread(heartbeat.Thread):
        def start(self) -> None:  # type: ignore[override]
            started.append(self.name)
            super().start()

    monkeypatch.setattr(heartbeat, "Thread", _SpyThread)
    with heartbeat.log_heartbeat("outer", interval_s=1000.0):
        with heartbeat.log_heartbeat("inner", interval_s=1000.0):
            pass

    assert started == ["generation-heartbeat"]
    assert _heartbeat_records(caplog, contains="outer done")
    assert not _heartbeat_records(caplog, contains="inner")


def test_sequential_log_heartbeat_ticks_each_generation(caplog, monkeypatch) -> None:
    monkeypatch.setattr(heartbeat, "_device_sample", lambda: None)
    caplog.set_level(logging.INFO, logger="server_utils.heartbeat")
    with heartbeat.log_heartbeat("first", interval_s=1000.0):
        pass
    with heartbeat.log_heartbeat("second", interval_s=1000.0):
        pass
    assert len(_heartbeat_records(caplog, contains="first done")) == 1
    assert len(_heartbeat_records(caplog, contains="second done")) == 1


def test_overlapping_log_heartbeat_only_ticks_the_owner(caplog, monkeypatch) -> None:
    from threading import Event, Thread

    monkeypatch.setattr(heartbeat, "_device_sample", lambda: None)
    caplog.set_level(logging.INFO, logger="server_utils.heartbeat")
    started: list[str] = []

    class _SpyThread(heartbeat.Thread):
        def start(self) -> None:  # type: ignore[override]
            started.append(self.name)
            super().start()

    monkeypatch.setattr(heartbeat, "Thread", _SpyThread)
    owner_ready = Event()
    other_tried = Event()
    owner_release = Event()

    def _other() -> None:
        owner_ready.wait(timeout=2.0)
        with heartbeat.log_heartbeat("queued", interval_s=1000.0):
            other_tried.set()
            owner_release.wait(timeout=2.0)

    worker = Thread(target=_other)
    worker.start()
    with heartbeat.log_heartbeat("owner", interval_s=1000.0):
        owner_ready.set()
        assert other_tried.wait(timeout=2.0)
        owner_release.set()
    worker.join(timeout=2.0)
    assert started == ["generation-heartbeat"]
    assert _heartbeat_records(caplog, contains="owner done")
    assert not _heartbeat_records(caplog, contains="queued")
