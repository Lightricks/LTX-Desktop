"""Protocol for scanning model files in a directory."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from api_types import DetectedModel


class ModelScanner(Protocol):
    def scan_video_models(self, folder: Path) -> list[DetectedModel]:
        """Scan a folder for video model files and return structured results."""
        ...
