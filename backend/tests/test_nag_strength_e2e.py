"""Per-run NAG strength, end to end: create body -> stored row -> the NAG config the model gets.

The frontend setting "Per-run strength controls" sends `nagScale`, `nagAlpha` and `nagTau` in the create
body. These tests run the real queue and the real text-to-video executor against the fake
pipeline, and read the `nag` argument that the pipeline `generate` call received. Each test
names the bug it catches.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from pydantic import JsonValue
from starlette.testclient import TestClient

from app_factory import create_app
from services.fast_video_pipeline.nag_config import NagConfig
from services.sqlite_store import SqliteStore
from state import build_initial_state, set_state_service_for_tests
from state.app_state_types import HfAuthenticated
from tests.conftest import TEST_ADMIN_TOKEN
from tests.test_http import _a2v_params, _bundle, _wav  # pyright: ignore[reportPrivateUsage]

_BASE_PARAMS: dict[str, JsonValue] = {
    "prompt": "a fox runs",
    "negativePrompt": "clouds",
    "model": "ltx-2.5-fast",
    "aspectRatio": "16:9",
    "resolution": "540p",
    "duration": 5,
    "fps": 24,
}


def _run_text_to_video(
    test_state,
    fake_services,
    default_app_settings,
    create_fake_model_files,
    *,
    settings_scale: float,
    **strength: float,
) -> tuple[dict[str, JsonValue], NagConfig | None]:
    """Create one text-to-video run, wait for the pipeline call, return its stored params and the model's NAG."""
    create_fake_model_files()
    # The default test_state swaps in fake executors. Build one without them, so the real
    # text-to-video executor runs and calls the (fake) pipeline.
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(fake_services, SqliteStore(test_state.config.app_data_dir)),
    )
    handler.state.hf_auth_state = HfAuthenticated(access_token="fake-hf-token", expires_at=1e18)
    handler.state.app_settings.use_local_text_encoder = True
    handler.state.app_settings.nag_scale = settings_scale
    set_state_service_for_tests(handler)

    with TestClient(create_app(handler=handler, admin_token=TEST_ADMIN_TOKEN)) as client:
        created = client.post(
            "/api/generations/text-to-video", json={"params": {**_BASE_PARAMS, **strength}}
        )
        assert created.status_code == 200, created.text
        generation_id = created.json()["id"]
        # The fake pipeline writes no real video, so the run can fail after the call. The call
        # is what matters here: it holds the NAG config the model received.
        deadline = time.monotonic() + 10.0
        while not fake_services.fast_video_pipeline.generate_calls:
            assert time.monotonic() < deadline, "the pipeline was never called"
            time.sleep(0.01)
        stored = client.get(f"/api/generations/{generation_id}").json()["spec"]["params"]

    (call,) = fake_services.fast_video_pipeline.generate_calls
    return stored, call["nag"]


@pytest.mark.parametrize(
    ("strength", "expected"),
    [
        # Default preset sends nothing: Settings scale and the backend alpha and tau.
        ({}, NagConfig("clouds", scale=7.5, alpha=0.25, tau=2.5)),
        # Strong sends only alpha. Bug: it also replaced the Settings scale.
        ({"nagAlpha": 0.5}, NagConfig("clouds", scale=7.5, alpha=0.5, tau=2.5)),
        # Max.
        (
            {"nagScale": 20.0, "nagAlpha": 0.5, "nagTau": 5.0},
            NagConfig("clouds", scale=20.0, alpha=0.5, tau=5.0),
        ),
        # Custom, with alpha at its lowest.
        (
            {"nagScale": 3.5, "nagAlpha": 0.0, "nagTau": 1.0},
            NagConfig("clouds", scale=3.5, alpha=0.0, tau=1.0),
        ),
    ],
)
def test_text_to_video_run_gets_the_strength_the_form_sent(
    test_state, fake_services, default_app_settings, create_fake_model_files, strength, expected
) -> None:
    stored, nag = _run_text_to_video(
        test_state,
        fake_services,
        default_app_settings,
        create_fake_model_files,
        settings_scale=7.5,
        **strength,
    )

    assert nag == expected
    # The row holds what the model got, which is what the result line shows.
    assert (stored["nagScale"], stored["nagAlpha"], stored["nagTau"]) == (
        expected.scale,
        expected.alpha,
        expected.tau,
    )


def test_explicit_scale_turns_nag_on_even_when_settings_strength_is_zero(
    test_state, fake_services, default_app_settings, create_fake_model_files
) -> None:
    """Why the form must not send a strength while Settings has NAG off.

    The request scale wins over Settings, so a stale Max would force NAG on.
    """
    _, nag = _run_text_to_video(
        test_state,
        fake_services,
        default_app_settings,
        create_fake_model_files,
        settings_scale=0.0,
        nagScale=20.0,
        nagAlpha=0.5,
        nagTau=5.0,
    )

    assert nag == NagConfig("clouds", scale=20.0, alpha=0.5, tau=5.0)


def test_audio_to_video_run_gets_the_strength_the_form_sent(
    test_state, fake_services, default_app_settings, create_fake_model_files, tmp_path: Path
) -> None:
    """Audio to video decides NAG in its own handler path. Bug: it ignored the request knobs."""
    create_fake_model_files()
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(fake_services, SqliteStore(test_state.config.app_data_dir)),
    )
    handler.state.hf_auth_state = HfAuthenticated(access_token="fake-hf-token", expires_at=1e18)
    handler.state.app_settings.use_local_text_encoder = True
    handler.state.app_settings.nag_scale = 7.5
    set_state_service_for_tests(handler)

    with TestClient(create_app(handler=handler, admin_token=TEST_ADMIN_TOKEN)) as client:
        audio = client.post("/api/assets", json={"path": str(_wav(tmp_path / "line.wav"))}).json()
        created = client.post(
            "/api/generations/audio-to-video",
            json={
                "params": _a2v_params(
                    negativePrompt="clouds", nagScale=20.0, nagAlpha=0.5, nagTau=5.0
                ),
                "inputs": {"audio": {"assetId": audio["id"]}},
            },
        )
        assert created.status_code == 200, created.text
        deadline = time.monotonic() + 10.0
        while not fake_services.a2v_pipeline.generate_calls:
            assert time.monotonic() < deadline, "the pipeline was never called"
            time.sleep(0.01)

    (call,) = fake_services.a2v_pipeline.generate_calls
    assert call["nag"] == NagConfig("clouds", scale=20.0, alpha=0.5, tau=5.0)
