"""LTX API client service protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from api_types import ExtendMode, RetakeMode, VideoCameraMotion

INSUFFICIENT_FUNDS_ERROR_TYPE = "insufficient_funds_error"
PAYMENT_DECLINED_ERROR_TYPE = "payment_declined_error"
CONTENT_FILTERED_ERROR_TYPE = "content_filtered_error"


@dataclass(frozen=True)
class LTXRetakeResult:
    video_bytes: bytes | None
    result_payload: dict[str, Any] | None


class LTXAPIClientError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        detail: str,
        stage: Literal["upload_init", "upload_parse", "upload_put"] | None = None,
        *,
        provider_error_type: str | None = None,
        provider_message: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.stage = stage
        self.provider_error_type = provider_error_type
        self.provider_message = provider_message
        self.request_id = request_id


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
        duration: float | None,
        fps: float,
        generate_audio: bool,
        camera_motion: VideoCameraMotion = "none",
        enhance_prompt: bool = True,
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
        duration: float | None,
        fps: float,
        generate_audio: bool,
        camera_motion: VideoCameraMotion = "none",
        last_frame_uri: str | None = None,
        enhance_prompt: bool = True,
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
        last_frame_uri: str | None = None,
        enhance_prompt: bool = True,
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
        model: str,
    ) -> LTXRetakeResult:
        ...

    def extend(
        self,
        *,
        api_key: str,
        video_path: str,
        duration: float,
        prompt: str,
        mode: ExtendMode,
        model: str,
    ) -> LTXRetakeResult:
        ...
