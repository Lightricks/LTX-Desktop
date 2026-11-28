from __future__ import annotations

from api_types import GenerateVideoRequest
from keyframe_request import validate_keyframe_inputs


def _request(**overrides: object) -> GenerateVideoRequest:
    payload: dict[str, object] = {
        "prompt": "test",
        "resolution": "540p",
        "model": "fast",
        "duration": 5,
        "fps": 24,
        "keyframes": [
            {"imagePath": "/opening.png", "frameIndex": 0},
            {"imagePath": "/closing.png", "frameIndex": 80},
        ],
    }
    payload.update(overrides)
    return GenerateVideoRequest.model_validate(payload)


def test_accepts_local_keyframes() -> None:
    assert validate_keyframe_inputs(
        _request(),
        use_api_specs=False,
        image_path=None,
        last_image_path=None,
        audio_path=None,
    ) is None


def test_rejects_api_keyframes() -> None:
    assert validate_keyframe_inputs(
        _request(),
        use_api_specs=True,
        image_path=None,
        last_image_path=None,
        audio_path=None,
    ) == "Multi-keyframe generation is only available for local generation"


def test_rejects_first_frame_mix() -> None:
    assert validate_keyframe_inputs(
        _request(),
        use_api_specs=False,
        image_path="/still.png",
        last_image_path=None,
        audio_path=None,
    ) == "Keyframes cannot be combined with a first or last frame"


def test_rejects_auto_duration() -> None:
    assert validate_keyframe_inputs(
        _request(duration=None),
        use_api_specs=False,
        image_path=None,
        last_image_path=None,
        audio_path=None,
    ) == "Keyframes cannot be combined with automatic duration"


def test_rejects_duplicate_frame_indices() -> None:
    assert validate_keyframe_inputs(
        _request(keyframes=[
            {"imagePath": "/a.png", "frameIndex": 10},
            {"imagePath": "/b.png", "frameIndex": 10},
        ]),
        use_api_specs=False,
        image_path=None,
        last_image_path=None,
        audio_path=None,
    ) == "Keyframe frame indices must be unique"


def test_rejects_frame_past_the_generated_clip() -> None:
    assert validate_keyframe_inputs(
        _request(keyframes=[{"imagePath": "/a.png", "frameIndex": 121}]),
        use_api_specs=False,
        image_path=None,
        last_image_path=None,
        audio_path=None,
    ) == "Keyframe frame index 121 is outside 0..120"


def test_rejects_frame_past_the_clip_at_25_fps() -> None:
    assert validate_keyframe_inputs(
        _request(fps=25, keyframes=[{"imagePath": "/a.png", "frameIndex": 124}]),
        use_api_specs=False,
        image_path=None,
        last_image_path=None,
        audio_path=None,
    ) == "Keyframe frame index 124 is outside 0..120"
