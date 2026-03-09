"""Video API client protocol for cloud video generation."""

from __future__ import annotations

from typing import Protocol


class VideoAPIClient(Protocol):
    def generate_text_to_video(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        duration: int,
        resolution: str,
        aspect_ratio: str,
        generate_audio: bool,
        last_frame: str | None = None,
    ) -> bytes:
        ...
