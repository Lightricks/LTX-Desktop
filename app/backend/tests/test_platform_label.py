"""Tests for startup platform labeling (macOS version in logs)."""

from __future__ import annotations

from services.gpu_info.gpu_info_impl import platform_label


def test_platform_label_includes_macos_version(monkeypatch) -> None:
    monkeypatch.setattr("services.gpu_info.gpu_info_impl.platform.system", lambda: "Darwin")
    monkeypatch.setattr("services.gpu_info.gpu_info_impl.platform.machine", lambda: "arm64")
    monkeypatch.setattr("services.gpu_info.gpu_info_impl.platform.mac_ver", lambda: ("15.6.1", ("", "", ""), "arm64"))
    assert platform_label() == "Darwin 15.6.1 (arm64)"


def test_platform_label_omits_empty_macos_version(monkeypatch) -> None:
    monkeypatch.setattr("services.gpu_info.gpu_info_impl.platform.system", lambda: "Darwin")
    monkeypatch.setattr("services.gpu_info.gpu_info_impl.platform.machine", lambda: "arm64")
    monkeypatch.setattr("services.gpu_info.gpu_info_impl.platform.mac_ver", lambda: ("", ("", "", ""), "arm64"))
    assert platform_label() == "Darwin (arm64)"


def test_platform_label_non_darwin_has_no_mac_ver(monkeypatch) -> None:
    monkeypatch.setattr("services.gpu_info.gpu_info_impl.platform.system", lambda: "Linux")
    monkeypatch.setattr("services.gpu_info.gpu_info_impl.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("services.gpu_info.gpu_info_impl.platform.mac_ver", lambda: ("", ("", "", ""), ""))
    assert platform_label() == "Linux (x86_64)"
