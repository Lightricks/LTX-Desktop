"""Test infrastructure for backend integration-style endpoint tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import torch

from app_factory import create_app
from app_handler import ServiceBundle
from runtime_config.model_download_specs import (
    DEPTH_PROCESSOR_CP_ID,
    IMG_GEN_MODEL_CP_ID,
    get_ic_loras_cp_ids,
    get_latest_ltx_model_id,
    get_ltx_model_spec,
    resolve_model_path,
)
from runtime_config.port_constant import PORT
from state import RuntimeConfig, build_initial_state, set_state_service_for_tests
from state.app_settings import AppSettings
from state.app_state_types import HfAuthenticated
from tests.fake_camera_motion_prompts import FAKE_CAMERA_MOTION_PROMPTS
from tests.fakes.services import FakeServices

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture"
)

DEFAULT_APP_SETTINGS = AppSettings()


@pytest.fixture
def fake_services() -> FakeServices:
    return FakeServices()


@pytest.fixture(autouse=True)
def test_state(tmp_path: Path, fake_services: FakeServices):
    """Provide a fresh AppHandler per test and register it in DI."""
    app_data = tmp_path / "app_data"
    default_models_dir = app_data / "models"
    outputs_dir = tmp_path / "outputs"

    for directory in (default_models_dir, outputs_dir, app_data):
        directory.mkdir(parents=True, exist_ok=True)

    config = RuntimeConfig(
        device=torch.device("cpu"),
        app_data_dir=app_data,
        default_models_dir=default_models_dir,
        outputs_dir=outputs_dir,
        settings_file=app_data / "settings.json",
        ltx_api_base_url="https://api.ltx.video",
        local_generations_mode="full_models_loading",
        use_sage_attention=False,
        camera_motion_prompts=FAKE_CAMERA_MOTION_PROMPTS,
        default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        dev_mode=False,
        hf_oauth_client_id="test-client-id",
        backend_port=PORT,
    )

    bundle = ServiceBundle(
        http=fake_services.http,
        gpu_cleaner=fake_services.gpu_cleaner,
        model_downloader=fake_services.model_downloader,
        lora_catalog_provider=fake_services.lora_catalog_provider,
        gpu_info=fake_services.gpu_info,
        video_processor=fake_services.video_processor,
        text_encoder=fake_services.text_encoder,
        task_runner=fake_services.task_runner,
        ltx_api_client=fake_services.ltx_api_client,
        zit_api_client=fake_services.zit_api_client,
        fast_video_pipeline_class=type(fake_services.fast_video_pipeline),
        mlx_fast_video_pipeline_class=type(fake_services.fast_video_pipeline),
        image_generation_pipeline_class=type(fake_services.image_generation_pipeline),
        ic_lora_pipeline_class=type(fake_services.ic_lora_pipeline),
        depth_processor_pipeline_class=type(fake_services.depth_processor_pipeline),
        pose_processor_pipeline_class=type(fake_services.pose_processor_pipeline),
        a2v_pipeline_class=type(fake_services.a2v_pipeline),
        retake_pipeline_class=type(fake_services.retake_pipeline),
        prompt_enhancer_pipeline_class=type(fake_services.prompt_enhancer_pipeline),
    )

    handler = build_initial_state(
        config,
        DEFAULT_APP_SETTINGS.model_copy(deep=True),
        service_bundle=bundle,
    )
    handler.state.hf_auth_state = HfAuthenticated(
        access_token="fake-hf-token",
        expires_at=1e18,
    )
    set_state_service_for_tests(handler)
    yield handler


TEST_ADMIN_TOKEN = "test-admin-token"


@pytest.fixture
def client(test_state):
    from starlette.testclient import TestClient

    app = create_app(handler=test_state, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def default_app_settings() -> AppSettings:
    return DEFAULT_APP_SETTINGS.model_copy(deep=True)


def _test_model_path(test_state, cp_id: str) -> Path:
    return resolve_model_path(test_state.config.default_models_dir, cp_id)


@pytest.fixture
def create_fake_model_files(test_state):
    def _create(include_zit: bool = False):
        ltx_spec = get_ltx_model_spec(get_latest_ltx_model_id())

        for cp_id in (ltx_spec.model_cp, ltx_spec.upscale_cp):
            path = _test_model_path(test_state, cp_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * 1024)

        te_dir = _test_model_path(test_state, ltx_spec.text_encoder_cp)
        te_dir.mkdir(parents=True, exist_ok=True)
        (te_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
        (te_dir / "tokenizer.model").write_bytes(b"\x00" * 1024)

        if include_zit:
            zit_dir = _test_model_path(test_state, IMG_GEN_MODEL_CP_ID)
            zit_dir.mkdir(parents=True, exist_ok=True)
            (zit_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

    return _create


@pytest.fixture
def create_fake_lora(test_state):
    """Write a fake LoRA file under the models dir and return its resolved path.

    Refs are now contained to models_dir + must exist (resolve_lora_ref), so tests
    that forward LoRAs need a real on-disk file rather than an arbitrary path.
    """
    def _create(name: str = "style.safetensors") -> str:
        loras_dir = test_state.config.default_models_dir / "loras"
        loras_dir.mkdir(parents=True, exist_ok=True)
        path = loras_dir / name
        path.write_bytes(b"\x00" * 1024)
        return str(path.resolve())

    return _create


@pytest.fixture
def create_fake_ic_lora_files(test_state):
    def _create(include_depth: bool = True):
        ltx_spec = get_ltx_model_spec(get_latest_ltx_model_id())
        for cp_id in get_ic_loras_cp_ids(ltx_spec.ic_loras_spec):
            path = _test_model_path(test_state, cp_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * 1024)

        if include_depth:
            depth_path = _test_model_path(test_state, DEPTH_PROCESSOR_CP_ID)
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            if depth_path.suffix:
                depth_path.write_bytes(b"\x00" * 1024)
            else:
                depth_path.mkdir(parents=True, exist_ok=True)
                (depth_path / "config.json").write_text("{}", encoding="utf-8")

    return _create


@pytest.fixture
def make_test_image():
    def _make(w: int = 64, h: int = 64, color: str = "red"):
        from PIL import Image

        img = Image.new("RGB", (w, h), color)
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    return _make
