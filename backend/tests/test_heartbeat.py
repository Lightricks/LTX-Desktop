"""Tests for the generation heartbeat's CPU-only and device-sample branches."""

from __future__ import annotations

import logging

import torch

from server_utils import heartbeat


def _heartbeat_records(caplog, *, contains: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "server_utils.heartbeat" and contains in r.getMessage()]


def test_cuda_memory_sample_none_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert heartbeat._cuda_memory_sample() is None


def test_device_sample_none_when_no_accelerator(monkeypatch) -> None:
    monkeypatch.setattr(heartbeat, "_cuda_memory_sample", lambda: None)
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


def test_log_heartbeat_includes_device_sample_when_available(caplog, monkeypatch) -> None:
    monkeypatch.setattr(heartbeat, "_device_sample", lambda: "fake=1.00GiB")
    caplog.set_level(logging.INFO, logger="server_utils.heartbeat")

    with heartbeat.log_heartbeat("test-label", interval_s=1000.0):
        pass

    done_records = _heartbeat_records(caplog, contains="test-label done")
    assert len(done_records) == 1
    assert "fake=1.00GiB" in done_records[0].getMessage()
