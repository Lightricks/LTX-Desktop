"""Model downloader service protocol definitions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class ModelDownloader(Protocol):
    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        token: str | None,
        on_progress: Callable[[int], None] | None = None,
        revision: str | None = None,
    ) -> Path: ...

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: str,
        token: str | None,
        on_progress: Callable[[int], None] | None = None,
        revision: str | None = None,
    ) -> Path: ...
