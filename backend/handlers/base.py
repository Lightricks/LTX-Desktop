"""Shared base types for state handlers."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar

from _routes._errors import HTTPError
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
)
from state.app_settings import MAX_GENERATION_SEED
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")
_S = TypeVar("_S", bound="StateHandlerBase")

def resolve_models_dir(state: AppState | None, config: RuntimeConfig) -> Path:
    """Effective models dir: custom from settings, or the startup default.

    Shared by ``StateHandlerBase.models_dir`` and the queued-generation handler so
    the custom-vs-default rule lives in one place. ``state`` may be ``None`` for
    callers that only hold a config.
    """
    custom = state.app_settings.models_dir if state is not None else None
    return Path(custom) if custom else config.default_models_dir


class StateHandlerBase:
    """Base handler with shared state and lock references."""

    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig) -> None:
        self._state = state
        self._lock = lock
        self._config = config

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def lock(self) -> RLock:
        return self._lock

    @property
    def config(self) -> RuntimeConfig:
        return self._config

    @property
    def models_dir(self) -> Path:
        """Effective models dir: custom from settings, or startup default."""
        return resolve_models_dir(self._state, self._config)

    def _require_local_generation_possible(self, *, api_key: str | None = None) -> None:
        """422 when this machine cannot generate locally and no API route is left.

        Pass the relevant key for entry points that can fall back to the API; omit it
        for local-only entry points (the Explore queue, IC-LoRA), which can never
        succeed on unsupported hardware no matter what keys are configured.
        """
        if self.config.local_generations_mode != "unsupported":
            return
        if api_key is not None and api_key.strip():
            return
        raise HTTPError(
            422,
            LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
            code=LOCAL_GENERATION_UNSUPPORTED,
        )

    def _resolve_seed(self) -> int:
        """Resolve the generation seed from the lock or a fresh random draw."""
        settings = self.state.app_settings
        if settings.seed_locked:
            logger.info("Using locked seed: %s", settings.locked_seed)
            return settings.locked_seed
        return random.randint(0, MAX_GENERATION_SEED)


def with_state_lock(
    method: Callable[Concatenate[_S, _P], _R],
) -> Callable[Concatenate[_S, _P], _R]:
    @wraps(method)
    def wrapped(self: _S, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self.lock:
            return method(self, *args, **kwargs)

    return wrapped
