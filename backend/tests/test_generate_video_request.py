from __future__ import annotations

import pytest
from pydantic import ValidationError

from api_types import GenerateVideoRequest


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": "test",
        "resolution": "540p",
        "model": "fast",
        "duration": 5,
        "fps": 24,
    }
    payload.update(overrides)
    return payload


def test_generate_video_request_strips_whitespace_from_nonblank_prompt() -> None:
    req = GenerateVideoRequest.model_validate(_payload(prompt="  fox  "))
    assert req.prompt == "fox"


def test_generate_video_request_rejects_whitespace_only_prompt_without_image() -> None:
    with pytest.raises(ValidationError):
        GenerateVideoRequest.model_validate(_payload(prompt="   "))


def test_generate_video_request_rejects_empty_prompt_without_image() -> None:
    with pytest.raises(ValidationError):
        GenerateVideoRequest.model_validate(_payload(prompt=""))


def test_generate_video_request_allows_blank_prompt_with_image() -> None:
    req = GenerateVideoRequest.model_validate(
        _payload(prompt="", imagePath="/tmp/start.png")
    )
    assert req.prompt == ""


def test_generate_video_request_normalizes_whitespace_prompt_with_image_to_empty() -> None:
    req = GenerateVideoRequest.model_validate(
        _payload(prompt="   ", imagePath="/tmp/start.png")
    )
    assert req.prompt == ""


def test_generate_video_request_strips_padded_prompt_with_image() -> None:
    req = GenerateVideoRequest.model_validate(
        _payload(prompt="  hello  ", imagePath="/tmp/start.png")
    )
    assert req.prompt == "hello"
