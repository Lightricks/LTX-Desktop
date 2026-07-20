"""Settings state mutations and persistence."""

from __future__ import annotations

import json
import logging
from threading import RLock
from typing import TYPE_CHECKING

from _routes._errors import HTTPError
from state.app_settings import AppSettings, UpdateSettingsRequest
from handlers._settings_utils import (
    collect_changed_paths,
    deep_merge_dicts,
    ensure_json_object,
    migrate_legacy_settings,
    strip_none_values,
)
from handlers.base import StateHandlerBase, with_state_lock
from state.app_state_types import AppState
from server_utils.secure_files import harden_file_permissions, secure_write_text

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class SettingsHandler(StateHandlerBase):
    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig) -> None:
        super().__init__(state, lock, config)

    @with_state_lock
    def load_settings(self, default_settings: AppSettings) -> AppSettings:
        settings_file = self.config.settings_file
        if settings_file.exists():
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                migrated = migrate_legacy_settings(ensure_json_object(payload))
                merged = deep_merge_dicts(
                    ensure_json_object(default_settings.model_dump(by_alias=False)),
                    migrated,
                )
                loaded = AppSettings.model_validate(merged)
                harden_file_permissions(settings_file)
                logger.info("Settings loaded from %s", settings_file)
                self.state.app_settings = loaded
                return loaded
            except Exception as exc:
                logger.warning("Could not load settings: %s", exc, exc_info=True)

        self.state.app_settings = default_settings.model_copy(deep=True)
        return self.state.app_settings

    def save_settings(self) -> None:
        try:
            payload = self.get_settings_snapshot().model_dump(by_alias=False)
            secure_write_text(self.config.settings_file, json.dumps(payload, indent=2))
        except Exception as exc:
            logger.warning("Could not save settings: %s", exc, exc_info=True)

    @with_state_lock
    def get_settings_snapshot(self) -> AppSettings:
        return self.state.app_settings.model_copy(deep=True)

    @with_state_lock
    def update_settings(self, patch: UpdateSettingsRequest) -> tuple[AppSettings, AppSettings, set[str]]:
        patch_payload = strip_none_values(ensure_json_object(patch.model_dump(by_alias=False, exclude_unset=True)))

        if "models_dir" in patch_payload and not self.config.models_dir_editable:
            raise HTTPError(409, "MODELS_DIR_MANAGED_BY_SERVER")

        for key_field in ("ltx_api_key", "gemini_api_key", "fal_api_key"):
            if key_field in patch_payload and patch_payload[key_field] == "":
                del patch_payload[key_field]

        before = self.state.app_settings.model_copy(deep=True)
        before_payload = ensure_json_object(before.model_dump(by_alias=False))

        if patch_payload:
            merged_payload = deep_merge_dicts(before_payload, patch_payload)
            self.state.app_settings = AppSettings.model_validate(merged_payload)

        after = self.state.app_settings.model_copy(deep=True)
        after_payload = ensure_json_object(after.model_dump(by_alias=False))

        if "prompt_cache_size" in patch_payload and self.state.text_encoder is not None:
            self._trim_prompt_cache()

        changed_paths = collect_changed_paths(before_payload, after_payload)
        self.save_settings()
        return before, after, changed_paths

    def _trim_prompt_cache(self) -> None:
        te = self.state.text_encoder
        if te is None:
            return

        max_size = self.state.app_settings.prompt_cache_size
        while len(te.prompt_cache) > max_size:
            oldest = next(iter(te.prompt_cache))
            del te.prompt_cache[oldest]
