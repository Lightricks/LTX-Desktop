"""Test infrastructure for backend integration-style endpoint tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from state.app_settings import AppSettings
from app_factory import create_app
from state import RuntimeConfig, build_initial_state, set_state_service_for_tests
from app_handler import ServiceBundle
from runtime_config.model_download_specs import DEFAULT_MODEL_DOWNLOAD_SPECS, DEFAULT_REQUIRED_MODEL_TYPES, resolve_model_path
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
        device="cpu",
        default_models_dir=default_models_dir,
        model_download_specs=DEFAULT_MODEL_DOWNLOAD_SPECS,
        required_model_types=DEFAULT_REQUIRED_MODEL_TYPES,
        outputs_dir=outputs_dir,
        settings_file=app_data / "settings.json",
        ltx_api_base_url="https://api.ltx.video",
        force_api_generations=False,
        use_sage_attention=False,
        camera_motion_prompts=FAKE_CAMERA_MOTION_PROMPTS,
        default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        dev_mode=False,
    )

    bundle = ServiceBundle(
        http=fake_services.http,
        gpu_cleaner=fake_services.gpu_cleaner,
        model_downloader=fake_services.model_downloader,
        gpu_info=fake_services.gpu_info,
        video_processor=fake_services.video_processor,
        text_encoder=fake_services.text_encoder,
        task_runner=fake_services.task_runner,
        ltx_api_client=fake_services.ltx_api_client,
        zit_api_client=fake_services.zit_api_client,
        fast_video_pipeline_class=type(fake_services.fast_video_pipeline),
        image_generation_pipeline_class=type(fake_services.image_generation_pipeline),
        ic_lora_pipeline_class=type(fake_services.ic_lora_pipeline),
        depth_processor_pipeline_class=type(fake_services.depth_processor_pipeline),
        pose_processor_pipeline_class=type(fake_services.pose_processor_pipeline),
        a2v_pipeline_class=type(fake_services.a2v_pipeline),
        retake_pipeline_class=type(fake_services.retake_pipeline),
    )

    handler = build_initial_state(
        config,
        DEFAULT_APP_SETTINGS.model_copy(deep=True),
        service_bundle=bundle,
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


def _test_model_path(test_state, model_type):
    return resolve_model_path(test_state.config.default_models_dir, test_state.config.model_download_specs, model_type)


@pytest.fixture
def create_fake_model_files(test_state):
    def _create(include_zit: bool = False):
        for path in (
            _test_model_path(test_state, "checkpoint"),
            _test_model_path(test_state, "upsampler"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * 1024)

        te_dir = _test_model_path(test_state, "text_encoder")
        te_dir.mkdir(parents=True, exist_ok=True)
        (te_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
        (te_dir / "tokenizer.model").write_bytes(b"\x00" * 1024)

        if include_zit:
            zit_dir = _test_model_path(test_state, "zit")
            zit_dir.mkdir(parents=True, exist_ok=True)
            (zit_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

    return _create


@pytest.fixture
def create_fake_ic_lora_files(test_state):
    def _create(names: list[str]):
        for name in names:
            path = _test_model_path(test_state, "ic_lora").parent / f"{name}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * 1024)

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
