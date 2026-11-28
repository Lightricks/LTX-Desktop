"""Cache for preprocessed IC-LoRA conditioning control videos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

from api_types import ConditioningType

logger = logging.getLogger(__name__)


class ConditioningCacheKey(NamedTuple):
    video_path: str
    conditioning_type: ConditioningType


class ConditioningCacheEntry(NamedTuple):
    control_video_path: str
    frame_count: int
    fps: float


class ConditioningCache:
    """Caches preprocessed control videos keyed by (video_path, conditioning_type).

    Not thread-safe — caller is expected to hold the state lock.
    """

    def __init__(self) -> None:
        self._entries: dict[ConditioningCacheKey, ConditioningCacheEntry] = {}

    def get(self, key: ConditioningCacheKey) -> ConditioningCacheEntry | None:
        return self._entries.get(key)

    def put(self, key: ConditioningCacheKey, entry: ConditioningCacheEntry) -> None:
        self._entries[key] = entry

    def cleanup(self) -> None:
        """Delete all cached control video files and clear entries."""
        for entry in self._entries.values():
            try:
                Path(entry.control_video_path).unlink(missing_ok=True)
            except Exception:
                logger.warning("Could not remove cached control video: %s", entry.control_video_path, exc_info=True)
        self._entries.clear()

    def __del__(self) -> None:
        self.cleanup()
