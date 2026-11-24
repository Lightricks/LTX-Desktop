"""Dev feature flags persisted in feature_flags.json next to settings.json."""

from __future__ import annotations

import json
import logging
import os
from threading import RLock

from api_types import FeatureFlags, FeatureFlagsPatch
from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

FEATURE_FLAGS_FILENAME = "feature_flags.json"


class FeatureFlagsHandler:
    def __init__(self, lock: RLock, config: RuntimeConfig) -> None:
        self._lock = lock
        self._path = config.app_data_dir / FEATURE_FLAGS_FILENAME

    def get_flags(self) -> FeatureFlags:
        with self._lock:
            return self._read()

    def update_flags(self, patch: FeatureFlagsPatch) -> FeatureFlags:
        with self._lock:
            # None means "leave unchanged" (omitted or explicit null).
            merged = FeatureFlags.model_validate(
                {**self._read().model_dump(), **patch.model_dump(exclude_none=True)}
            )
            tmp_path = self._path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(merged.model_dump(), indent=2), encoding="utf-8")
            os.replace(tmp_path, self._path)
            return merged

    def _read(self) -> FeatureFlags:
        try:
            return FeatureFlags.model_validate_json(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return FeatureFlags()
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s, using defaults: %s", self._path, exc)
            return FeatureFlags()
