"""JSON-file-backed persistence for saved prompts and wildcard definitions."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SavedPrompt:
    id: str
    text: str
    tags: list[str]
    category: str
    used_count: int
    created_at: str
    last_used_at: str | None


@dataclass
class WildcardEntry:
    id: str
    name: str
    values: list[str]
    created_at: str


def _empty_prompts() -> list[SavedPrompt]:
    return []


def _empty_wildcards() -> list[WildcardEntry]:
    return []


@dataclass
class PromptStoreData:
    prompts: list[SavedPrompt] = field(default_factory=_empty_prompts)
    wildcards: list[WildcardEntry] = field(default_factory=_empty_wildcards)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class PromptStore:
    """Simple JSON-file persistence for prompts and wildcards.

    Not thread-safe on its own; callers (the handler) are expected to
    hold the shared lock when calling mutating methods.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data = PromptStoreData()
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            prompts = [SavedPrompt(**p) for p in raw.get("prompts", [])]
            wildcards = [WildcardEntry(**w) for w in raw.get("wildcards", [])]
            self._data = PromptStoreData(prompts=prompts, wildcards=wildcards)
        except Exception as exc:
            logger.warning("Could not load prompt store from %s: %s", self._path, exc)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "prompts": [asdict(p) for p in self._data.prompts],
                "wildcards": [asdict(w) for w in self._data.wildcards],
            }
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not save prompt store to %s: %s", self._path, exc)

    # ------------------------------------------------------------------
    # Prompt CRUD
    # ------------------------------------------------------------------

    def list_prompts(
        self,
        search: str | None = None,
        tag: str | None = None,
        sort_by: str | None = None,
    ) -> list[SavedPrompt]:
        results = list(self._data.prompts)
        if search:
            lower = search.lower()
            results = [p for p in results if lower in p.text.lower()]
        if tag:
            results = [p for p in results if tag in p.tags]
        if sort_by == "used_count":
            results.sort(key=lambda p: p.used_count, reverse=True)
        elif sort_by == "created_at":
            results.sort(key=lambda p: p.created_at, reverse=True)
        elif sort_by == "last_used_at":
            results.sort(key=lambda p: p.last_used_at or "", reverse=True)
        return results

    def save_prompt(self, text: str, tags: list[str], category: str) -> SavedPrompt:
        prompt = SavedPrompt(
            id=_new_id(),
            text=text,
            tags=tags,
            category=category,
            used_count=0,
            created_at=_now_iso(),
            last_used_at=None,
        )
        self._data.prompts.append(prompt)
        self._save()
        return prompt

    def get_prompt(self, prompt_id: str) -> SavedPrompt | None:
        for p in self._data.prompts:
            if p.id == prompt_id:
                return p
        return None

    def delete_prompt(self, prompt_id: str) -> bool:
        before = len(self._data.prompts)
        self._data.prompts = [p for p in self._data.prompts if p.id != prompt_id]
        if len(self._data.prompts) < before:
            self._save()
            return True
        return False

    def increment_usage(self, prompt_id: str) -> SavedPrompt | None:
        for p in self._data.prompts:
            if p.id == prompt_id:
                p.used_count += 1
                p.last_used_at = _now_iso()
                self._save()
                return p
        return None

    # ------------------------------------------------------------------
    # Wildcard CRUD
    # ------------------------------------------------------------------

    def list_wildcards(self) -> list[WildcardEntry]:
        return list(self._data.wildcards)

    def create_wildcard(self, name: str, values: list[str]) -> WildcardEntry:
        entry = WildcardEntry(
            id=_new_id(),
            name=name,
            values=values,
            created_at=_now_iso(),
        )
        self._data.wildcards.append(entry)
        self._save()
        return entry

    def get_wildcard(self, wildcard_id: str) -> WildcardEntry | None:
        for w in self._data.wildcards:
            if w.id == wildcard_id:
                return w
        return None

    def update_wildcard(self, wildcard_id: str, values: list[str]) -> WildcardEntry | None:
        for w in self._data.wildcards:
            if w.id == wildcard_id:
                w.values = values
                self._save()
                return w
        return None

    def delete_wildcard(self, wildcard_id: str) -> bool:
        before = len(self._data.wildcards)
        self._data.wildcards = [w for w in self._data.wildcards if w.id != wildcard_id]
        if len(self._data.wildcards) < before:
            self._save()
            return True
        return False
