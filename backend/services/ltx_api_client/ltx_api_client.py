"""LTX API client service protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from api_types import RetakeMode, VideoCameraMotion


@dataclass(frozen=True)
class LTXRetakeResult:
    video_bytes: bytes | None
    result_payload: dict[str, Any] | None


class LTXAPIClientError(RuntimeError):
    def __init__(self, status_code: int, detail: str, stage: str | None = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.stage = stage


class LTXAPIClient(Protocol):
    def upload_file(
        self,
        *,
        api_key: str,
        file_path: str,
    ) -> str:
        ...

    def generate_text_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        model: str,
        resolution: str,
        duration: float,
        fps: float,
        generate_audio: bool,
        camera_motion: VideoCameraMotion = "none",
    ) -> bytes:
        ...

    def generate_image_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        image_uri: str,
        model: str,
        resolution: str,
        duration: float,
        fps: float,
        generate_audio: bool,
        camera_motion: VideoCameraMotion = "none",
    ) -> bytes:
        ...

    def generate_audio_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        audio_uri: str,
        image_uri: str | None,
        model: str,
        resolution: str,
    ) -> bytes:
        ...

    def retake(
        self,
        *,
        api_key: str,
        video_path: str,
        start_time: float,
        duration: float,
        prompt: str,
        mode: RetakeMode,
    ) -> LTXRetakeResult:
        ...
