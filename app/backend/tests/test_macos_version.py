"""Tests for macOS version detection."""

from __future__ import annotations

import pytest

from runtime_config import macos_version
from runtime_config.macos_version import detect_macos_major, parse_macos_major


@pytest.mark.parametrize(
    ("version", "expected"),
    [("26.6.2", 26), ("14.8.2", 14), ("15", 15), ("", None), ("unknown", None)],
)
def test_parse_macos_major(version: str, expected: int | None) -> None:
    assert parse_macos_major(version) == expected


def test_detect_macos_major_reads_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macos_version.platform, "mac_ver", lambda: ("14.8.2", ("", "", ""), "arm64"))
    assert detect_macos_major() == 14


def test_detect_macos_major_unknown_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macos_version.platform, "mac_ver", lambda: ("", ("", "", ""), ""))
    assert detect_macos_major() is None

