"""Persistent local library store for characters, styles, and references."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeVar


ReferenceCategory = Literal["people", "places", "props", "other"]

_T = TypeVar("_T")


@dataclass
class Character:
    id: str
    name: str
    role: str
    description: str
    reference_image_paths: list[str] = field(default_factory=lambda: list[str]())
    created_at: str = ""


@dataclass
class Style:
    id: str
    name: str
    description: str
    reference_image_path: str = ""
    created_at: str = ""


@dataclass
class Reference:
    id: str
    name: str
    category: ReferenceCategory
    image_path: str = ""
    created_at: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _load_json_list(path: Path, cls: type[_T]) -> list[_T]:
    """Load a list of dataclass instances from a JSON file."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [cls(**item) for item in raw]  # type: ignore[arg-type]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []


def _write_json(path: Path, data: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class LibraryStore:
    """Manages JSON file persistence for library entities."""

    def __init__(self, library_dir: Path) -> None:
        self._dir = library_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._characters_file = self._dir / "characters.json"
        self._styles_file = self._dir / "styles.json"
        self._references_file = self._dir / "references.json"

        self._characters: list[Character] = _load_json_list(self._characters_file, Character)
        self._styles: list[Style] = _load_json_list(self._styles_file, Style)
        self._references: list[Reference] = _load_json_list(self._references_file, Reference)

    # ------------------------------------------------------------------
    # Characters
    # ------------------------------------------------------------------

    def list_characters(self) -> list[Character]:
        return list(self._characters)

    def get_character(self, character_id: str) -> Character | None:
        for c in self._characters:
            if c.id == character_id:
                return c
        return None

    def create_character(
        self,
        *,
        name: str,
        role: str,
        description: str,
        reference_image_paths: list[str] | None = None,
    ) -> Character:
        character = Character(
            id=_new_id(),
            name=name,
            role=role,
            description=description,
            reference_image_paths=reference_image_paths or [],
            created_at=_now_iso(),
        )
        self._characters.append(character)
        self._save_characters()
        return character

    def update_character(
        self,
        character_id: str,
        *,
        name: str | None = None,
        role: str | None = None,
        description: str | None = None,
        reference_image_paths: list[str] | None = None,
    ) -> Character | None:
        character = self.get_character(character_id)
        if character is None:
            return None
        if name is not None:
            character.name = name
        if role is not None:
            character.role = role
        if description is not None:
            character.description = description
        if reference_image_paths is not None:
            character.reference_image_paths = reference_image_paths
        self._save_characters()
        return character

    def delete_character(self, character_id: str) -> bool:
        before = len(self._characters)
        self._characters = [c for c in self._characters if c.id != character_id]
        if len(self._characters) < before:
            self._save_characters()
            return True
        return False

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def list_styles(self) -> list[Style]:
        return list(self._styles)

    def get_style(self, style_id: str) -> Style | None:
        for s in self._styles:
            if s.id == style_id:
                return s
        return None

    def create_style(
        self,
        *,
        name: str,
        description: str,
        reference_image_path: str = "",
    ) -> Style:
        style = Style(
            id=_new_id(),
            name=name,
            description=description,
            reference_image_path=reference_image_path,
            created_at=_now_iso(),
        )
        self._styles.append(style)
        self._save_styles()
        return style

    def delete_style(self, style_id: str) -> bool:
        before = len(self._styles)
        self._styles = [s for s in self._styles if s.id != style_id]
        if len(self._styles) < before:
            self._save_styles()
            return True
        return False

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------

    def list_references(self, category: ReferenceCategory | None = None) -> list[Reference]:
        if category is None:
            return list(self._references)
        return [r for r in self._references if r.category == category]

    def get_reference(self, reference_id: str) -> Reference | None:
        for r in self._references:
            if r.id == reference_id:
                return r
        return None

    def create_reference(
        self,
        *,
        name: str,
        category: ReferenceCategory,
        image_path: str = "",
    ) -> Reference:
        ref = Reference(
            id=_new_id(),
            name=name,
            category=category,
            image_path=image_path,
            created_at=_now_iso(),
        )
        self._references.append(ref)
        self._save_references()
        return ref

    def delete_reference(self, reference_id: str) -> bool:
        before = len(self._references)
        self._references = [r for r in self._references if r.id != reference_id]
        if len(self._references) < before:
            self._save_references()
            return True
        return False

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_characters(self) -> None:
        _write_json(self._characters_file, [asdict(c) for c in self._characters])

    def _save_styles(self) -> None:
        _write_json(self._styles_file, [asdict(s) for s in self._styles])

    def _save_references(self) -> None:
        _write_json(self._references_file, [asdict(r) for r in self._references])
