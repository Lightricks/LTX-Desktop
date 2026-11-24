"""Detect the running macOS major version."""

from __future__ import annotations

import platform


def parse_macos_major(version: str) -> int | None:
    """Return the major number of a macOS version string such as "26.6.2".

    Return None when the string is empty or not numeric (non-macOS hosts return "").
    """
    head = version.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def detect_macos_major() -> int | None:
    """Return the macOS major version of this host, or None when it is unknown."""
    return parse_macos_major(platform.mac_ver()[0])
