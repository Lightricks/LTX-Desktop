"""Test infrastructure for backend integration-style endpoint tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from state.app_settings import AppSettings
from app_factory import create_app
from state import RuntimeConfig, build_initial_state, set_state_service_for_tests
from app_handler import ServiceBundle
from runtime_config.model_download_specs import DEFAULT_MODEL_DOWNLOAD_SPECS, DEFAULT_REQUIRED_MODEL_TYPES
from tests.fakes.services import FakeServices

CAMERA_MOTION_PROMPTS = {
    "none": "",
    "static": ", static camera, locked off shot, no camera movement",
    "focus_shift": ", focus shift, rack focus, changing focal point",
    "dolly_in": ", dolly in, camera pushing forward, smooth forward movement",
    "dolly_out": ", dolly out, camera pulling back, smooth backward movement",
    "dolly_left": ", dolly left, camera tracking left, lateral movement",
    "dolly_right": ", dolly right, camera tracking right, lateral movement",
    "jib_up": ", jib up, camera rising up, upward crane movement",
    "jib_down": ", jib down, camera lowering down, downward crane movement",
}

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
    models_dir = app_data / "models"
    outputs_dir = tmp_path / "outputs"
    ic_lora_dir = models_dir / "ic-loras"

    for directory in (models_dir, outputs_dir, ic_lora_dir, app_data):
        directory.mkdir(parents=True, exist_ok=True)

    config = RuntimeConfig(
        device="cpu",
        models_dir=models_dir,
        model_download_specs=DEFAULT_MODEL_DOWNLOAD_SPECS,
        required_model_types=DEFAULT_REQUIRED_MODEL_TYPES,
        outputs_dir=outputs_dir,
        ic_lora_dir=ic_lora_dir,
        settings_file=app_data / "settings.json",
        ltx_api_base_url="https://api.ltx.video",
        force_api_generations=False,
        use_sage_attention=False,
        camera_motion_prompts=CAMERA_MOTION_PROMPTS,
        default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
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
        image_api_client=fake_services.image_api_client,
        video_api_client=fake_services.video_api_client,
        palette_sync_client=fake_services.palette_sync_client,
        fast_video_pipeline_class=type(fake_services.fast_video_pipeline),
        gguf_video_pipeline_class=None,
        nf4_video_pipeline_class=None,
        image_generation_pipeline_class=type(fake_services.image_generation_pipeline),
        flux_klein_pipeline_class=None,
        ic_lora_pipeline_class=type(fake_services.ic_lora_pipeline),
        a2v_pipeline_class=type(fake_services.a2v_pipeline),
        retake_pipeline_class=type(fake_services.retake_pipeline),
        ic_lora_model_downloader=fake_services.ic_lora_model_downloader,
        model_scanner=fake_services.model_scanner,
    )

    handler = build_initial_state(
        config,
        DEFAULT_APP_SETTINGS.model_copy(deep=True),
        service_bundle=bundle,
    )
    set_state_service_for_tests(handler)
    yield handler


@pytest.fixture
def client(test_state):
    from starlette.testclient import TestClient

    app = create_app(handler=test_state)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def default_app_settings() -> AppSettings:
    return DEFAULT_APP_SETTINGS.model_copy(deep=True)


@pytest.fixture
def create_fake_model_files(test_state):
    def _create(include_zit: bool = False, include_flux_klein: bool = False):
        for path in (
            test_state.config.model_path("checkpoint"),
            test_state.config.model_path("upsampler"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * 1024)

        te_dir = test_state.config.model_path("text_encoder")
        te_dir.mkdir(parents=True, exist_ok=True)
        (te_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
        (te_dir / "tokenizer.model").write_bytes(b"\x00" * 1024)

        if include_zit:
            zit_dir = test_state.config.model_path("zit")
            zit_dir.mkdir(parents=True, exist_ok=True)
            (zit_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
            # Default is now flux-klein-9b; switch to ZIT when creating ZIT files
            test_state.state.app_settings.image_model = "z-image-turbo"

        if include_flux_klein:
            flux_dir = test_state.config.model_path("flux_klein")
            flux_dir.mkdir(parents=True, exist_ok=True)
            (flux_dir / "model_index.json").write_bytes(b"{}")
            (flux_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

    return _create


@pytest.fixture
def create_fake_ic_lora_files(test_state):
    def _create(names: list[str]):
        for name in names:
            path = test_state.config.ic_lora_dir / f"{name}.safetensors"
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
