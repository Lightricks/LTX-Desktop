"""Library handler for characters, styles, and references."""

from __future__ import annotations

from _routes._errors import HTTPError
from state.library_store import Character, LibraryStore, Reference, ReferenceCategory, Style


class LibraryHandler:
    """Business logic for the local library (characters, styles, references)."""

    def __init__(self, store: LibraryStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Characters
    # ------------------------------------------------------------------

    def list_characters(self) -> list[Character]:
        return self._store.list_characters()

    def create_character(
        self,
        *,
        name: str,
        role: str,
        description: str,
        reference_image_paths: list[str] | None = None,
    ) -> Character:
        if not name.strip():
            raise HTTPError(400, "Character name must not be empty")
        return self._store.create_character(
            name=name,
            role=role,
            description=description,
            reference_image_paths=reference_image_paths,
        )

    def update_character(
        self,
        character_id: str,
        *,
        name: str | None = None,
        role: str | None = None,
        description: str | None = None,
        reference_image_paths: list[str] | None = None,
    ) -> Character:
        if name is not None and not name.strip():
            raise HTTPError(400, "Character name must not be empty")
        character = self._store.update_character(
            character_id,
            name=name,
            role=role,
            description=description,
            reference_image_paths=reference_image_paths,
        )
        if character is None:
            raise HTTPError(404, f"Character {character_id} not found")
        return character

    def delete_character(self, character_id: str) -> None:
        deleted = self._store.delete_character(character_id)
        if not deleted:
            raise HTTPError(404, f"Character {character_id} not found")

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def list_styles(self) -> list[Style]:
        return self._store.list_styles()

    def create_style(
        self,
        *,
        name: str,
        description: str,
        reference_image_path: str = "",
    ) -> Style:
        if not name.strip():
            raise HTTPError(400, "Style name must not be empty")
        return self._store.create_style(
            name=name,
            description=description,
            reference_image_path=reference_image_path,
        )

    def delete_style(self, style_id: str) -> None:
        deleted = self._store.delete_style(style_id)
        if not deleted:
            raise HTTPError(404, f"Style {style_id} not found")

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------

    def list_references(self, category: ReferenceCategory | None = None) -> list[Reference]:
        return self._store.list_references(category)

    def create_reference(
        self,
        *,
        name: str,
        category: ReferenceCategory,
        image_path: str = "",
    ) -> Reference:
        if not name.strip():
            raise HTTPError(400, "Reference name must not be empty")
        return self._store.create_reference(
            name=name,
            category=category,
            image_path=image_path,
        )

    def delete_reference(self, reference_id: str) -> None:
        deleted = self._store.delete_reference(reference_id)
        if not deleted:
            raise HTTPError(404, f"Reference {reference_id} not found")
