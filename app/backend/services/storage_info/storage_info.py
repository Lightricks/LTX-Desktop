"""Storage drive-type probe protocol definitions."""

from __future__ import annotations

from typing import Literal, Protocol

DriveType = Literal["nvme", "ssd", "hdd"]


class StorageInfo(Protocol):
    def get_drive_type(self) -> DriveType | None:
        """Drive class of the checkpoint volume. None when unknown."""
        ...
