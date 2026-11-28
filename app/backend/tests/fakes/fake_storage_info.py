"""Fake storage drive-type probe for tests."""

from __future__ import annotations

from services.storage_info.storage_info import DriveType


class FakeStorageInfo:
    def __init__(self, drive_type: DriveType | None = None) -> None:
        self.drive_type = drive_type

    def get_drive_type(self) -> DriveType | None:
        return self.drive_type
