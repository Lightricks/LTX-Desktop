"""Disk streaming opens safetensors with the pread backend on Darwin only."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import safetensors
import torch
from ltx_core.block_streaming.disk import DiskTensorReader
from safetensors.torch import save_file

import services.patches.disk_streaming_pread as pread


@pytest.fixture(autouse=True)
def _restore_reader() -> None:
    original = DiskTensorReader.__init__
    yield
    DiskTensorReader.__init__ = original  # type: ignore[method-assign]
    pread._installed = False
    pread._checked = False
    pread._original_init = None


@pytest.fixture
def on_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")


def test_disk_reader_passes_pread(tmp_path: Path, on_darwin: None) -> None:
    path = tmp_path / "tiny.safetensors"
    save_file({"weight": torch.ones(2, 2)}, str(path))
    pread.install()
    seen: list[dict[str, object]] = []
    original_open = safetensors.safe_open

    def _spy(*args: object, **kwargs: object):
        seen.append(dict(kwargs))
        return original_open(*args, **kwargs)

    safetensors.safe_open = _spy  # type: ignore[assignment]
    try:
        reader = DiskTensorReader([str(path)])
        try:
            torch.testing.assert_close(reader.get_tensor("weight"), torch.ones(2, 2))
        finally:
            reader.close()
    finally:
        safetensors.safe_open = original_open

    assert seen == [{"framework": "pt", "device": "cpu", "backend": "pread"}]


def test_install_is_idempotent(on_darwin: None) -> None:
    pread.install()
    first = DiskTensorReader.__init__
    pread.install()
    assert DiskTensorReader.__init__ is first


def test_install_leaves_mmap_when_backend_is_unsupported(
    monkeypatch: pytest.MonkeyPatch, on_darwin: None
) -> None:
    monkeypatch.setattr(pread.inspect, "signature", lambda _fn: type("Sig", (), {"parameters": {}})())
    before = DiskTensorReader.__init__
    pread.install()
    assert DiskTensorReader.__init__ is before
    assert pread._installed is False
    assert pread._checked is True


def test_log_status_reports_pread_after_logging_is_configured(
    caplog: pytest.LogCaptureFixture, on_darwin: None
) -> None:
    pread.install()
    with caplog.at_level(logging.INFO, logger=pread.logger.name):
        pread.log_status()
    assert "Disk streaming opens safetensors with backend=pread" in caplog.text


def test_log_status_reports_mmap_when_backend_is_unsupported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, on_darwin: None
) -> None:
    monkeypatch.setattr(pread.inspect, "signature", lambda _fn: type("Sig", (), {"parameters": {}})())
    pread.install()
    with caplog.at_level(logging.WARNING, logger=pread.logger.name):
        pread.log_status()
    assert "disk streaming stays on mmap" in caplog.text
    assert "Darwin-only" not in caplog.text


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_install_is_noop_off_darwin(platform: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    before = DiskTensorReader.__init__
    pread.install()
    assert DiskTensorReader.__init__ is before
    assert pread._installed is False
    assert pread._checked is True


def test_log_status_reports_darwin_only_skip(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    pread.install()
    with caplog.at_level(logging.INFO, logger=pread.logger.name):
        pread.log_status()
    assert "pread is Darwin-only" in caplog.text
    assert "safetensors.safe_open has no backend" not in caplog.text
