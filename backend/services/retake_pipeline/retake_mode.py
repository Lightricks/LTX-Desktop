"""Canonical retake mode → (regenerate_video, regenerate_audio). Raises on unknown."""

from __future__ import annotations

from api_types import RetakeMode


class InvalidRetakeModeError(ValueError):
    def __init__(self) -> None:
        super().__init__("INVALID_RETAKE_MODE")


def resolve_retake_mode(mode: RetakeMode) -> tuple[bool, bool]:
    if mode == "replace_audio_and_video":
        return True, True
    if mode == "replace_video":
        return True, False
    if mode == "replace_audio":
        return False, True
    raise InvalidRetakeModeError()
