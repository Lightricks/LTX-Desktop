"""Prompt library and wildcard management handler."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import Literal

from handlers.base import StateHandlerBase, with_state_lock
from services.wildcard_parser import WildcardDef, expand_prompt, expand_random
from state.app_state_types import AppState
from state.prompt_store import PromptStore, SavedPrompt, WildcardEntry

logger = logging.getLogger(__name__)


class PromptHandler(StateHandlerBase):
    """Domain handler for saved prompts and wildcard definitions."""

    def __init__(self, state: AppState, lock: RLock, store_path: Path) -> None:
        super().__init__(state, lock)
        self._store = PromptStore(store_path)

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    @with_state_lock
    def list_prompts(
        self,
        search: str | None = None,
        tag: str | None = None,
        sort_by: str | None = None,
    ) -> list[SavedPrompt]:
        return self._store.list_prompts(search=search, tag=tag, sort_by=sort_by)

    @with_state_lock
    def save_prompt(self, text: str, tags: list[str], category: str) -> SavedPrompt:
        return self._store.save_prompt(text, tags, category)

    @with_state_lock
    def delete_prompt(self, prompt_id: str) -> bool:
        return self._store.delete_prompt(prompt_id)

    @with_state_lock
    def increment_usage(self, prompt_id: str) -> SavedPrompt | None:
        return self._store.increment_usage(prompt_id)

    # ------------------------------------------------------------------
    # Wildcards
    # ------------------------------------------------------------------

    @with_state_lock
    def list_wildcards(self) -> list[WildcardEntry]:
        return self._store.list_wildcards()

    @with_state_lock
    def create_wildcard(self, name: str, values: list[str]) -> WildcardEntry:
        return self._store.create_wildcard(name, values)

    @with_state_lock
    def update_wildcard(self, wildcard_id: str, values: list[str]) -> WildcardEntry | None:
        return self._store.update_wildcard(wildcard_id, values)

    @with_state_lock
    def delete_wildcard(self, wildcard_id: str) -> bool:
        return self._store.delete_wildcard(wildcard_id)

    # ------------------------------------------------------------------
    # Wildcard expansion
    # ------------------------------------------------------------------

    @with_state_lock
    def expand_wildcards(
        self,
        prompt: str,
        mode: Literal["all", "random"] = "random",
        count: int = 1,
    ) -> list[str]:
        """Expand wildcards in *prompt* using stored wildcard definitions."""
        entries = self._store.list_wildcards()
        defs = [WildcardDef(name=e.name, values=e.values) for e in entries]
        if mode == "all":
            return expand_prompt(prompt, defs)
        return expand_random(prompt, defs, count=count)
