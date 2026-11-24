"""Queued text-to-image: provider resolution at create, execution, prompt rewrite, cancel."""

from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from PIL import Image

from services.zit_api_client.zit_api_client import ZitImageBlockedError
from services.zit_api_client.zit_api_client_impl import ZitAPIClientImpl
from tests.fakes import FakeResponse
from tests.fakes.services import FakeHTTPClient
from tests.http_error_assertions import assert_http_error

_CREATE = "/api/generations/text-to-image"


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _create(client, **params: object):
    return client.post(_CREATE, json={"params": {"prompt": "a red fox", **params}})


def _await_status(client, generation_id: str, *statuses: str) -> dict:
    deadline = time.monotonic() + 5
    while True:
        body = client.get(f"/api/generations/{generation_id}").json()
        if body["status"] in statuses:
            return body
        assert time.monotonic() < deadline, body["status"]
        time.sleep(0.02)


def _use_fal(test_state, fake_services) -> None:
    test_state.state.app_settings.fal_api_key = "fal-key"
    fake_services.zit_api_client.text_to_image_result = _png_bytes()


def _enable_enhancer(test_state, create_fake_model_files, fake_services) -> None:
    create_fake_model_files(include_prompt_enhancer=True)
    test_state.state.app_settings.explore_auto_enhance_prompts = True
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a richly detailed red fox"


def _fal_prompt(fake_services) -> str:
    return fake_services.zit_api_client.text_to_image_calls[-1]["prompt"]


def _fal_expansion(fake_services) -> bool:
    return fake_services.zit_api_client.text_to_image_calls[-1]["enable_prompt_expansion"]


def _use_gemini(test_state, reply: str | None) -> None:
    settings = test_state.state.app_settings
    settings.prompt_enhancer_provider_preference = "api"
    settings.gemini_api_key = "gemini-key"
    if reply is not None:
        test_state.http.queue(
            "post",
            FakeResponse(
                status_code=200,
                json_payload={"candidates": [{"content": {"parts": [{"text": reply}]}}]},
            ),
        )


def test_fal_without_a_key_is_refused_and_nothing_is_queued(client) -> None:
    response = _create(client, provider="fal")

    assert_http_error(
        response,
        status_code=422,
        code="FAL_API_KEY_NOT_CONFIGURED",
        message="Add a fal API key.",
    )
    queue = client.get("/api/generation-queue").json()
    assert queue["queued"] == [] and queue["active"] == []


def test_retry_keeps_the_stored_provider_after_the_setting_flips(
    client, test_state, fake_services
) -> None:
    _use_fal(test_state, fake_services)
    test_state.state.app_settings.user_prefers_fal_api_image_generations = True
    fake_services.zit_api_client.raise_on_text_to_image = RuntimeError("provider down")

    created = _create(client).json()
    assert created["spec"]["params"]["provider"] == "fal"
    _await_status(client, created["id"], "failed")

    test_state.state.app_settings.user_prefers_fal_api_image_generations = False
    fake_services.zit_api_client.raise_on_text_to_image = None
    assert client.post(f"/api/generations/{created['id']}/retry").status_code == 200
    done = _await_status(client, created["id"], "succeeded", "failed")

    assert done["status"] == "succeeded"
    assert len(fake_services.zit_api_client.text_to_image_calls) == 2
    assert fake_services.image_generation_pipeline.generate_calls == []


def test_local_is_refused_when_the_machine_forces_api_generations(
    client, test_state
) -> None:
    test_state.config.local_generations_mode = "unsupported"

    response = _create(client, provider="local")

    assert_http_error(
        response,
        status_code=422,
        code="LOCAL_GENERATION_UNSUPPORTED",
        message="This machine cannot run local generation.",
    )


def test_local_without_the_image_model_is_refused_as_not_downloaded(client) -> None:
    response = _create(client, provider="local")

    assert_http_error(
        response,
        status_code=409,
        code="IMAGE_MODEL_NOT_DOWNLOADED",
        message="Download the Z-Image Turbo model.",
    )


@pytest.mark.parametrize(
    "body",
    [
        {"params": {"prompt": "a fox", "imagePath": "/tmp/source.png"}},
        {"params": {"prompt": "a fox"}, "inputs": {"image": {"assetId": "a1"}}},
    ],
    ids=["imagePath-in-params", "image-in-inputs"],
)
def test_create_refuses_any_image_input_so_edit_is_never_reachable(client, body) -> None:
    assert client.post(_CREATE, json=body).status_code == 422


def test_fal_job_saves_one_png_as_an_image_asset(client, test_state, fake_services) -> None:
    _use_fal(test_state, fake_services)

    created = _create(client, provider="fal", width=768, height=512).json()
    done = _await_status(client, created["id"], "succeeded", "failed")

    assert done["status"] == "succeeded"
    assert len(done["outputs"]) == 1
    asset = done["outputs"][0]
    assert asset["media_kind"] == "image"
    assert (asset["metadata"]["metadata"]["width"], asset["metadata"]["metadata"]["height"]) == (16, 16)
    assert Path(asset["path"]).read_bytes() == _png_bytes()
    call = fake_services.zit_api_client.text_to_image_calls[0]
    assert (call["width"], call["height"], call["prompt"]) == (768, 512, "a red fox")


def test_a_blocked_fal_result_fails_with_a_clear_code_and_saves_no_asset(
    client, test_state, fake_services
) -> None:
    _use_fal(test_state, fake_services)
    fake_services.zit_api_client.raise_on_text_to_image = ZitImageBlockedError("blocked")

    created = _create(client, provider="fal").json()
    done = _await_status(client, created["id"], "failed")

    assert done["error_code"] == "IMAGE_BLOCKED_BY_SAFETY_CHECKER"
    assert done["outputs"] == []


def test_fal_client_still_returns_the_placeholder_when_a_caller_does_not_reject_blocked() -> None:
    # GenSpace's sync path never rejected a blocked result. Only the queued job does.
    http = FakeHTTPClient()
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            json_payload={"images": [{"url": "https://fal.test/black.png"}], "has_nsfw_concepts": [True]},
        ),
    )
    http.queue("get", FakeResponse(status_code=200, content=b"black-pixels"))

    image = ZitAPIClientImpl(http).generate_text_to_image(
        api_key="k", prompt="p", width=512, height=512, seed=1, num_inference_steps=4
    )

    assert image == b"black-pixels"


def test_auto_enhance_rewrites_the_prompt(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    _use_fal(test_state, fake_services)
    _enable_enhancer(test_state, create_fake_model_files, fake_services)

    created = _create(client, provider="fal", prompt="a fox").json()
    _await_status(client, created["id"], "succeeded")

    assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls[0]["prompt"] == "a fox"
    assert _fal_prompt(fake_services) == "a richly detailed red fox"
    assert _fal_expansion(fake_services) is False


@pytest.mark.parametrize(
    ("auto_enhance", "provenance"),
    [(False, "typed"), (True, "enhanced")],
    ids=["auto-enhance-off", "already-enhanced-by-hand"],
)
def test_prompt_is_sent_as_typed_when_no_rewrite_is_wanted(
    client, test_state, fake_services, create_fake_model_files, auto_enhance, provenance
) -> None:
    _use_fal(test_state, fake_services)
    _enable_enhancer(test_state, create_fake_model_files, fake_services)
    test_state.state.app_settings.explore_auto_enhance_prompts = auto_enhance

    created = _create(client, provider="fal", promptProvenance=provenance).json()
    _await_status(client, created["id"], "succeeded")

    assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
    assert _fal_prompt(fake_services) == "a red fox"
    assert _fal_expansion(fake_services) is False


def test_api_only_machine_never_rewrites_on_local_gemma(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    """status() reports local enhancement as unsupported there, so the job must not load Gemma."""
    test_state.config.local_generations_mode = "unsupported"
    _use_fal(test_state, fake_services)
    _enable_enhancer(test_state, create_fake_model_files, fake_services)

    created = _create(client, provider="fal").json()
    _await_status(client, created["id"], "succeeded")

    assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
    assert _fal_prompt(fake_services) == "a red fox"
    assert _fal_expansion(fake_services) is True


def test_a_failed_local_rewrite_falls_back_to_fal_expansion(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    # Bug caught: a crashed Gemma rewrite sends a bare prompt, so auto-enhance silently does nothing.
    _use_fal(test_state, fake_services)
    _enable_enhancer(test_state, create_fake_model_files, fake_services)
    fake_services.prompt_enhancer_pipeline.raise_on_enhance = RuntimeError("gemma crashed")

    created = _create(client, provider="fal").json()
    done = _await_status(client, created["id"], "succeeded", "failed")

    assert done["status"] == "succeeded"
    assert _fal_prompt(fake_services) == "a red fox"
    assert _fal_expansion(fake_services) is True


def test_gemma_rewrites_when_the_gpu_is_idle_and_fal_does_not_expand(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    # Bug caught: fal expansion is paid for although the GPU could rewrite for free.
    _use_fal(test_state, fake_services)
    _enable_enhancer(test_state, create_fake_model_files, fake_services)

    created = _create(client, provider="fal").json()
    _await_status(client, created["id"], "succeeded")

    assert len(fake_services.prompt_enhancer_pipeline.enhance_t2v_calls) == 1
    assert _fal_prompt(fake_services) == "a richly detailed red fox"
    assert _fal_expansion(fake_services) is False


def test_fal_expands_when_the_gpu_is_busy_and_gemma_stays_unloaded(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    # Bug caught: the fal job loads Gemma while a local job owns the GPU.
    _use_fal(test_state, fake_services)
    _enable_enhancer(test_state, create_fake_model_files, fake_services)

    with test_state.generation.queue_yields():
        created = _create(client, provider="fal").json()
        _await_status(client, created["id"], "succeeded")

    assert fake_services.prompt_enhancer_pipeline.enhance_t2v_calls == []
    assert _fal_prompt(fake_services) == "a red fox"
    assert _fal_expansion(fake_services) is True


def test_gemini_rewrites_when_gemma_is_missing_and_fal_does_not_expand(
    client, test_state, fake_services
) -> None:
    # Bug caught: a user without Gemma but with Gemini pays for fal expansion.
    _use_fal(test_state, fake_services)
    test_state.state.app_settings.explore_auto_enhance_prompts = True
    _use_gemini(test_state, "a fox, rewritten by gemini")

    created = _create(client, provider="fal").json()
    _await_status(client, created["id"], "succeeded")

    assert _fal_prompt(fake_services) == "a fox, rewritten by gemini"
    assert _fal_expansion(fake_services) is False


def test_a_failed_gemini_rewrite_falls_back_to_fal_expansion(client, test_state, fake_services) -> None:
    # Bug caught: a Gemini error fails the whole job instead of falling back.
    _use_fal(test_state, fake_services)
    test_state.state.app_settings.explore_auto_enhance_prompts = True
    _use_gemini(test_state, None)
    test_state.http.queue("post", FakeResponse(status_code=500, text="boom"))

    created = _create(client, provider="fal").json()
    done = _await_status(client, created["id"], "succeeded", "failed")

    assert done["status"] == "succeeded"
    assert _fal_prompt(fake_services) == "a red fox"
    assert _fal_expansion(fake_services) is True


def test_fal_expands_when_no_enhancer_is_available(client, test_state, fake_services) -> None:
    # Bug caught: auto-enhance on with no enhancer sends the bare prompt without expansion.
    _use_fal(test_state, fake_services)
    test_state.state.app_settings.explore_auto_enhance_prompts = True

    created = _create(client, provider="fal").json()
    _await_status(client, created["id"], "succeeded")

    assert _fal_prompt(fake_services) == "a red fox"
    assert _fal_expansion(fake_services) is True


def test_the_prompt_fal_expanded_replaces_the_typed_prompt_on_the_finished_row(
    client, test_state, fake_services
) -> None:
    # Bug caught: the user never sees the prompt fal actually used.
    _use_fal(test_state, fake_services)
    test_state.state.app_settings.explore_auto_enhance_prompts = True
    fake_services.zit_api_client.queued_provider_prompt = "a red fox, expanded by fal"

    created = _create(client, provider="fal").json()
    done = _await_status(client, created["id"], "succeeded")

    assert done["spec"]["params"]["prompt"] == "a red fox, expanded by fal"


def test_cancelling_a_running_local_image_job_ends_cancelled_with_no_asset(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files(include_zit=True)
    pipeline = fake_services.image_generation_pipeline
    pipeline.inference_steps = 200
    pipeline.step_delay_s = 0.02

    created = _create(client, provider="local").json()
    assert pipeline.entered_inference.wait(timeout=5)
    assert client.post(f"/api/generations/{created['id']}/cancel").status_code == 200
    done = _await_status(client, created["id"], "cancelled")

    assert done["outputs"] == []
    assert not list(test_state.config.outputs_dir.glob("*.png"))


def test_sides_are_stored_rounded_down_to_the_16_grid(client, test_state, fake_services) -> None:
    _use_fal(test_state, fake_services)

    created = _create(client, provider="fal", width=1000, height=777).json()
    _await_status(client, created["id"], "succeeded")

    assert (created["spec"]["params"]["width"], created["spec"]["params"]["height"]) == (992, 768)
    call = fake_services.zit_api_client.text_to_image_calls[0]
    assert (call["width"], call["height"]) == (992, 768)


@pytest.mark.parametrize(("width", "height"), [(255, 512), (512, 2049)])
def test_sides_outside_the_limits_are_refused(client, width, height) -> None:
    assert _create(client, provider="local", width=width, height=height).status_code == 422


@pytest.mark.parametrize(
    ("setup", "provider", "blocker"),
    [
        ("no-fal-key", "fal", "FAL_API_KEY_NOT_CONFIGURED"),
        ("unsupported-hardware", "local", "LOCAL_GENERATION_UNSUPPORTED"),
        ("model-missing", "local", "IMAGE_MODEL_NOT_DOWNLOADED"),
    ],
)
def test_status_reports_the_same_blocker_create_refuses_with(
    client, test_state, setup, provider, blocker
) -> None:
    if setup == "unsupported-hardware":
        test_state.config.local_generations_mode = "unsupported"

    status = client.get("/api/image-generation/status").json()
    created = _create(client, provider=provider)

    assert status["falBlocker" if provider == "fal" else "localBlocker"] == blocker
    assert created.json()["code"] == blocker


def test_status_defaults_to_fal_when_the_machine_cannot_run_locally(
    client, test_state
) -> None:
    test_state.config.local_generations_mode = "unsupported"
    test_state.state.app_settings.fal_api_key = "fal-key"

    status = client.get("/api/image-generation/status").json()

    assert status == {
        "defaultProvider": "fal",
        "localBlocker": "LOCAL_GENERATION_UNSUPPORTED",
        "falBlocker": None,
    }
