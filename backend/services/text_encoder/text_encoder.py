"""Text encoder service protocol definitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from state.app_state_types import AppState, TextEncodingResult


class TextEncoder(Protocol):
    def install_patches(self, state_getter: Callable[[], AppState]) -> None:
        ...

    def encode_via_api(
        self,
        prompt: str,
        api_key: str,
        checkpoint_path: str,
        enhance_prompt: bool,
        api_model_id: str | None = None,
    ) -> TextEncodingResult | None:
        ...
