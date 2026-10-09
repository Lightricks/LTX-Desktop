"""Day to Night Home create: stored params, missing weights, and the sized canvas."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import av
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from starlette.testclient import TestClient

from api_types import (
    DownloadSpec,
    DownloadVariant,
    IcLoraCatalogItem,
    IcLoraControl,
    IcLoraGenerateCancelledResponse,
    IcLoraRecipeStoredParams,
    IcLoraSettings,
    InputSpec,
    LoraEntry,
    PreprocessingStep,
    PromptTemplatePlaceholder,
    PromptTemplateSpec,
    VideoAssetMetadata,
    VideoMeta,
    _HOME_RECIPE_FIELDS,
    parse_lora_catalog,
)
from app_factory import create_app
from app_handler import ServiceBundle
from handlers.base import resolve_models_dir
from runtime_config.ltx_capabilities import budget_pixels, local_caps
from runtime_config.model_download_specs import resolve_ic_lora_path
from services.features.ic_lora_recipes import (
    IC_LORA_RECIPES,
    ic_lora_fps_choices,
    ic_lora_original_fps,
)
from services.features.video.ic_lora_recipe import ic_lora_video_filters
from services.features.video.ic_lora_recipe import IcLoraRecipeExecutor
from services.generation_interrupt import GenerationCancelledError
from services.generation_queue.types import OutputAllocation, OutputPlan
from services.media_probe import _stream_fps, probe_file
from services.records import AssetRecord, CapabilityFailedError, GenerationRecord
from services.sqlite_store import SqliteStore
from state import build_initial_state, set_state_service_for_tests
from tests.conftest import TEST_ADMIN_TOKEN
from tests.fakes.services import FakeCapture
from tests.http_error_assertions import assert_http_error
from tests.lora_catalog_helpers import add_ic_lora, download_spec

_LOCAL_2_3 = "ltx-2.3-22b-distilled-1.1"
_LOCAL_2_5 = "ltx-2.5-22b-distilled"
_WEIGHTS = "day-to-night.safetensors"


def _mp4(path: Path, *, fps: int = 8, seconds: float = 1) -> Path:
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=None)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    for _ in range(max(1, round(fps * seconds))):
        writer.append_data(frame)
    writer.close()
    return path


def _params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "prompt": "turn the street to night",
        "model": "ltx-2.5-fast",
        "resolution": "540p",
        "audioMode": "generated",
        "variantId": "default",
    }
    params.update(overrides)
    if params.get("variantId") is None:
        del params["variantId"]
    return params


def _two_variant_spec() -> DownloadSpec:
    return DownloadSpec(
        repo_id="org/day-23",
        variants=[
            DownloadVariant(
                id="ltx-2.5__day-to-night",
                label="LTX-2.5",
                filename="new.safetensors",
                size_bytes=10,
                base_model="LTX-2.5",
                repo_id="org/day-25",
            ),
            DownloadVariant(
                id="default",
                label="LTX-2.3",
                filename="old.safetensors",
                size_bytes=10,
                base_model="LTX-2.3",
            ),
        ],
    )


def _install_catalog(fake_services, download=None) -> None:
    add_ic_lora(
        fake_services,
        id="day-to-night",
        download=download if download is not None else download_spec(_WEIGHTS),
        input={"kind": "video"},
        default_settings=IcLoraSettings(
            skip_stage_2=True, resolution_factor=1.5, audio_mode="source"
        ),
        prompt_template=PromptTemplateSpec(
            template=(
                "A realistic nighttime scene. {lighting} Only the lighting changes from "
                "day to night; identical composition, framing, camera movement and motion."
            ),
            placeholders={"lighting": PromptTemplatePlaceholder()},
        ),
    )


def _write_weights(test_state, filename: str = _WEIGHTS, ic_lora_id: str = "day-to-night") -> None:
    models_dir = resolve_models_dir(test_state.state, test_state.config)
    path = resolve_ic_lora_path(models_dir, ic_lora_id, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weights")


def test_create_stores_day_to_night_params_and_rejects_a_pipeline_model_id(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_catalog(fake_services)
    _write_weights(test_state)
    video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()
    response = client.post(
        "/api/generations/ic-lora-recipes/day-to-night",
        json={
            "params": _params(),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert response.status_code == 200, response.text
    spec = response.json()["spec"]
    assert spec["inputs"]["video"]["assetId"] == video["id"]
    assert spec["params"]["model"] == "ltx-2.5-fast"
    assert spec["params"]["resolution"] == "540p"
    assert spec["params"]["audioMode"] == "generated"
    assert "scale" not in spec["params"]
    assert spec["params"]["loras"] == [
        {
            "ref": "",
            "scale": 1.0,
            "catalogId": "day-to-night",
            "displayName": "Test IC-LoRA",
            "variantId": "default",
        }
    ]

    rejected = client.post(
        "/api/generations/ic-lora-recipes/day-to-night",
        json={
            "params": _params(model="fast"),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert rejected.status_code == 422


def test_create_rejects_missing_day_to_night_weights_without_changing_settings(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    create_fake_model_files(model_id=_LOCAL_2_3)
    _install_catalog(fake_services)
    test_state.state.app_settings.active_ltx_model_id = _LOCAL_2_3
    video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()
    response = client.post(
        "/api/generations/ic-lora-recipes/day-to-night",
        json={
            "params": _params(model="ltx-2.5-fast"),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert_http_error(
        response,
        status_code=409,
        code="IC_LORA_NOT_DOWNLOADED",
        message="IC_LORA_NOT_DOWNLOADED",
    )
    assert test_state.state.app_settings.active_ltx_model_id == _LOCAL_2_3


def test_create_rejects_a_2_5_job_for_an_entry_that_supports_only_2_3(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    create_fake_model_files(model_id=_LOCAL_2_3)
    _install_catalog(fake_services)
    fake_services.lora_catalog_provider._catalog.ic_loras[-1].supported_models = ["LTX-2.3"]
    _write_weights(test_state)
    video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()
    response = client.post(
        "/api/generations/ic-lora-recipes/day-to-night",
        json={
            "params": _params(model="ltx-2.5-fast"),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert_http_error(
        response,
        status_code=422,
        code="IC_LORA_UNSUPPORTED_MODEL",
        message="'day-to-night' does not support the selected LTX model",
    )


def test_create_fills_omitted_audio_and_scale_from_catalog_defaults(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_catalog(fake_services)
    fake_services.lora_catalog_provider._catalog.ic_loras[-1].default_settings = IcLoraSettings(
        skip_stage_2=True,
        resolution_factor=1.5,
        audio_mode="generated",
        lora_strength=1.5,
    )
    _write_weights(test_state)
    video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()
    response = client.post(
        "/api/generations/ic-lora-recipes/day-to-night",
        json={
            "params": {
                "prompt": "turn the street to night",
                "model": "ltx-2.5-fast",
                "variantId": "default",
            },
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert response.status_code == 200, response.text
    spec = response.json()["spec"]
    assert spec["params"]["audioMode"] == "generated"
    assert spec["params"]["loras"][0]["scale"] == 1.5
    assert "scale" not in spec["params"]


def test_alpha_gen_create_accepts_a_blank_prompt_and_runs_on_2_5_only(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    """AlphaGen is the shipped catalog row: no prompt, LTX-2.5 weights only."""
    create_fake_model_files()
    create_fake_model_files(model_id=_LOCAL_2_3)
    catalog_path = Path(__file__).resolve().parents[1] / "runtime_config" / "lora_catalog.json"
    item = next(
        entry
        for entry in parse_lora_catalog(catalog_path.read_text()).ic_loras
        if entry.id == "alpha-gen"
    )
    fake_services.lora_catalog_provider._catalog.ic_loras.append(item)
    models_dir = resolve_models_dir(test_state.state, test_state.config)
    weights = resolve_ic_lora_path(models_dir, item.id, item.download.variants[0].filename)
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"weights")
    video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()

    def create(model: str) -> object:
        return client.post(
            "/api/generations/ic-lora-recipes/alpha-gen",
            json={
                "params": {"prompt": "", "model": model, "resolution": "540p"},
                "inputs": {"video": {"assetId": video["id"]}},
            },
        )

    accepted = create("ltx-2.5-fast")
    assert accepted.status_code == 200, accepted.text
    params = accepted.json()["spec"]["params"]
    assert params["prompt"] == ""
    assert params["audioMode"] == "source"
    assert params["loras"][0]["scale"] == 1.0
    assert params["loras"][0]["variantId"] == "ltx-2.5__alpha-gen"
    assert_http_error(
        create("ltx-2.3-fast"),
        status_code=422,
        code="IC_LORA_UNSUPPORTED_MODEL",
        message="'alpha-gen' does not support the selected LTX model",
    )


def test_create_rejects_a_clip_longer_than_10_seconds(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_catalog(fake_services)
    _write_weights(test_state)
    video = client.post(
        "/api/assets", json={"path": str(_mp4(tmp_path / "long.mp4", seconds=11))}
    ).json()
    response = client.post(
        "/api/generations/ic-lora-recipes/day-to-night",
        json={
            "params": _params(),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert_http_error(
        response,
        status_code=422,
        code="IC_LORA_INPUT_TOO_LONG",
        message="This clip is longer than 10s at 8fps. Trim it to continue.",
    )


def test_create_accepts_a_10_second_30fps_clip_at_24fps(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_catalog(fake_services)
    _write_weights(test_state)
    video = client.post(
        "/api/assets",
        json={"path": str(_mp4(tmp_path / "phone.mp4", fps=30, seconds=10))},
    ).json()
    response = client.post(
        "/api/generations/ic-lora-recipes/day-to-night",
        json={
            "params": _params(fps=24),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["spec"]["params"]["fps"] == 24
    assert video["metadata"]["metadata"]["fps"] == 30


def test_omitted_fps_stores_the_original_rate_not_24(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_catalog(fake_services)
    _write_weights(test_state)
    phone = client.post(
        "/api/assets",
        json={"path": str(_mp4(tmp_path / "phone.mp4", fps=30, seconds=1))},
    ).json()
    slow = client.post(
        "/api/assets",
        json={"path": str(_mp4(tmp_path / "slow.mp4", fps=15, seconds=1))},
    ).json()

    def created(asset_id: str) -> object:
        response = client.post(
            "/api/generations/ic-lora-recipes/day-to-night",
            json={"params": _params(), "inputs": {"video": {"assetId": asset_id}}},
        )
        assert response.status_code == 200, response.text
        return response.json()["spec"]["params"]["fps"]

    assert created(phone["id"]) == 25
    assert created(slow["id"]) == 15


def test_create_rejects_a_10_second_30fps_clip_when_the_rate_keeps_all_its_frames(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_catalog(fake_services)
    _write_weights(test_state)
    video = client.post(
        "/api/assets",
        json={"path": str(_mp4(tmp_path / "phone.mp4", fps=30, seconds=10))},
    ).json()
    response = client.post(
        "/api/generations/ic-lora-recipes/day-to-night",
        json={
            "params": _params(fps=30),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert_http_error(
        response,
        status_code=422,
        code="IC_LORA_INPUT_TOO_LONG",
        message="This clip is longer than 8s at 30fps. Trim it to continue.",
    )


def test_create_rejects_a_two_stage_catalog_entry(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_catalog(fake_services)
    fake_services.lora_catalog_provider._catalog.ic_loras[-1].default_settings = IcLoraSettings(
        skip_stage_2=False, resolution_factor=1.5, audio_mode="source"
    )
    _write_weights(test_state)
    video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()
    response = client.post(
        "/api/generations/ic-lora-recipes/day-to-night",
        json={
            "params": _params(),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert_http_error(
        response,
        status_code=422,
        code="IC_LORA_UNSUPPORTED_RECIPE",
        message="'day-to-night' needs its own form",
    )


def test_create_rejects_an_entry_the_home_form_cannot_run(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_catalog(fake_services)
    _write_weights(test_state)
    item = fake_services.lora_catalog_provider._catalog.ic_loras[-1]
    video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()

    def rejected() -> object:
        return client.post(
            "/api/generations/ic-lora-recipes/day-to-night",
            json={
                "params": _params(),
                "inputs": {"video": {"assetId": video["id"]}},
            },
        )

    item.controls = [IcLoraControl(id="pads", label="Pads", kind="position_canvas")]
    assert_http_error(
        rejected(),
        status_code=422,
        code="IC_LORA_UNSUPPORTED_RECIPE",
        message="'day-to-night' needs its own form",
    )
    item.controls = []
    item.preprocessing = [PreprocessingStep(utility="canny")]
    assert_http_error(
        rejected(),
        status_code=422,
        code="IC_LORA_UNSUPPORTED_RECIPE",
        message="'day-to-night' needs its own form",
    )
    item.preprocessing = []
    item.allows_reference_image = True
    assert_http_error(
        rejected(),
        status_code=422,
        code="IC_LORA_UNSUPPORTED_RECIPE",
        message="'day-to-night' needs its own form",
    )
    item.allows_reference_image = False
    item.input = InputSpec(kind="image")
    assert_http_error(
        rejected(),
        status_code=422,
        code="IC_LORA_UNSUPPORTED_RECIPE",
        message="'day-to-night' needs its own form",
    )


def test_every_ic_lora_recipe_stays_a_home_recipe_until_a_new_field_is_classified() -> None:
    catalog_path = Path(__file__).resolve().parents[1] / "runtime_config" / "lora_catalog.json"
    catalog = parse_lora_catalog(catalog_path.read_text())
    for recipe in IC_LORA_RECIPES.values():
        item = next(entry for entry in catalog.ic_loras if entry.id == recipe.catalog_id)
        assert item.is_home_recipe(), recipe.recipe_id
    assert frozenset(IcLoraCatalogItem.model_fields) == _HOME_RECIPE_FIELDS


class _Assets:
    def get_asset(self, asset_id: str) -> None:
        return None


class _Generator:
    def generate_local_reserved(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("unused")


def test_ic_lora_spec_without_one_catalog_lora_is_a_capability_error() -> None:
    executor = IcLoraRecipeExecutor(
        _Generator(),
        _Assets(),
        lambda: Path("."),
        catalog_id="day-to-night",
    )
    missing = {
        "params": {"prompt": "night", "model": "ltx-2.5-fast", "audioMode": "source"},
        "inputs": {"video": {"assetId": "clip"}},
    }
    blank = {
        "params": {
            "prompt": "night",
            "model": "ltx-2.5-fast",
            "audioMode": "source",
            "loras": [{"ref": "", "scale": 1.0, "catalogId": ""}],
        },
        "inputs": {"video": {"assetId": "clip"}},
    }
    for spec in (missing, blank):
        try:
            executor.validate_params(spec, contract_version=1)
        except CapabilityFailedError as exc:
            assert exc.code == "INVALID_GENERATION_SPEC"
        else:
            raise AssertionError("spec without one catalog LoRA was accepted")

def test_create_stores_the_requested_variant_and_the_run_loads_that_file(
    test_state, fake_services, create_fake_model_files, tmp_path: Path, default_app_settings
) -> None:
    create_fake_model_files()
    create_fake_model_files(model_id=_LOCAL_2_3)
    _install_catalog(fake_services, download=_two_variant_spec())
    _write_weights(test_state, "old.safetensors")
    app, _handler = _real_app(test_state, fake_services, default_app_settings)
    with TestClient(app) as client:
        video = client.post(
            "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
        ).json()
        fake_services.video_processor.register_video(
            video["path"], FakeCapture(frames=["f"] * 9, fps=24, width=16, height=16)
        )
        response = client.post(
            "/api/generations/ic-lora-recipes/day-to-night",
            json={
                "params": _params(scale=0.4, variantId="default"),
                "inputs": {"video": {"assetId": video["id"]}},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["spec"]["params"]["loras"] == [
            {
                "ref": "",
                "scale": 0.4,
                "catalogId": "day-to-night",
                "displayName": "Test IC-LoRA",
                "variantId": "default",
            }
        ]
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if fake_services.ic_lora_pipeline.last_lora_path:
                break
            time.sleep(0.01)
    assert fake_services.ic_lora_pipeline.last_lora_path is not None
    assert fake_services.ic_lora_pipeline.last_lora_path.endswith("old.safetensors")


def test_omitted_variant_id_does_not_select_an_installed_file(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_catalog(fake_services, download=_two_variant_spec())
    _write_weights(test_state, "old.safetensors")
    video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()
    response = client.post(
        "/api/generations/ic-lora-recipes/day-to-night",
        json={
            "params": _params(variantId=None),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert_http_error(
        response,
        status_code=404,
        code="UNKNOWN_DOWNLOAD_VARIANT",
        message="UNKNOWN_DOWNLOAD_VARIANT",
    )


def test_create_rejects_an_unknown_variant_id(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_catalog(fake_services, download=_two_variant_spec())
    _write_weights(test_state, "old.safetensors")
    video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()
    response = client.post(
        "/api/generations/ic-lora-recipes/day-to-night",
        json={
            "params": _params(variantId="nope"),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert_http_error(
        response,
        status_code=404,
        code="UNKNOWN_DOWNLOAD_VARIANT",
        message="UNKNOWN_DOWNLOAD_VARIANT",
    )


def _real_app(test_state, fake_services, default_app_settings):
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=ServiceBundle(
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
            image_generation_pipeline_class=type(fake_services.image_generation_pipeline),
            ic_lora_pipeline_class=type(fake_services.ic_lora_pipeline),
            depth_processor_pipeline_class=type(fake_services.depth_processor_pipeline),
            pose_processor_pipeline_class=type(fake_services.pose_processor_pipeline),
            a2v_pipeline_class=type(fake_services.a2v_pipeline),
            retake_pipeline_class=type(fake_services.retake_pipeline),
            prompt_enhancer_pipeline_class=type(fake_services.prompt_enhancer_pipeline),
            store=SqliteStore(test_state.config.app_data_dir),
        ),
    )
    handler.state.app_settings.use_local_text_encoder = True
    handler.state.app_settings.active_ltx_model_id = _LOCAL_2_3
    set_state_service_for_tests(handler)
    return create_app(handler=handler, admin_token=TEST_ADMIN_TOKEN), handler


def test_day_to_night_explicit_resolution_renders_that_size(
    test_state, fake_services, create_fake_model_files, tmp_path: Path, default_app_settings
) -> None:
    create_fake_model_files()
    create_fake_model_files(model_id=_LOCAL_2_3)
    _install_catalog(fake_services)
    _write_weights(test_state)
    app, handler = _real_app(test_state, fake_services, default_app_settings)
    with TestClient(app) as client:
        video = client.post(
            "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
        ).json()
        fake_services.video_processor.register_video(
            video["path"], FakeCapture(frames=["f"] * 9, fps=24, width=16, height=16)
        )
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = (
            '{"lighting": "Warm amber light spills from the windows."}'
        )
        response = client.post(
            "/api/generations/ic-lora-recipes/day-to-night",
            json={
                "params": _params(resolution="540p", audioMode="source"),
                "inputs": {"video": {"assetId": video["id"]}},
            },
        )
        assert response.status_code == 200, response.text
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if fake_services.ic_lora_pipeline.generate_calls:
                break
            time.sleep(0.01)
    assert len(fake_services.ic_lora_pipeline.generate_calls) == 1
    call = fake_services.ic_lora_pipeline.generate_calls[0]
    width, height = budget_pixels(local_caps(_LOCAL_2_5), "540p", "1:1", mode="video")
    assert (call["width"], call["height"], call["resolution_factor"]) == (
        width * 2,
        height * 2,
        1.0,
    )
    assert call["width"] != 768
    checkpoint = fake_services.ic_lora_pipeline.last_checkpoint_path or ""
    assert "ltx-2.5" in checkpoint
    gemma = fake_services.ic_lora_pipeline.last_gemma_root or ""
    assert "gemma4" in gemma
    assert handler.state.app_settings.active_ltx_model_id == _LOCAL_2_3
    assert fake_services.ic_lora_pipeline.last_lora_strength == 1.0
    assert call["prompt"] == (
        "A realistic nighttime scene. Warm amber light spills from the windows. "
        "Only the lighting changes from day to night; identical composition, framing, "
        "camera movement and motion."
    )


def _install_layout_catalog(fake_services) -> None:
    """The Layout to Render shape: stage 2 keeps the IC-LoRA, one required look still at -1."""
    add_ic_lora(
        fake_services,
        id="layout-to-render",
        download=download_spec(_WEIGHTS),
        input={"kind": "video"},
        allows_reference_image=True,
        reference_image_required=True,
        reference_image_frame=-1,
        default_settings=IcLoraSettings(
            skip_stage_2=False, stage_2_ic_lora=True, resolution_factor=1.0, audio_mode="source"
        ),
    )


def _png(path: Path) -> Path:
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(path, format="PNG")
    return path


def test_layout_create_without_the_required_reference_image_is_rejected(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_layout_catalog(fake_services)
    _write_weights(test_state, ic_lora_id="layout-to-render")
    video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()
    response = client.post(
        "/api/generations/ic-lora-recipes/layout-to-render",
        json={"params": _params(), "inputs": {"video": {"assetId": video["id"]}}},
    )
    assert_http_error(
        response,
        status_code=422,
        code="INVALID_GENERATION_SPEC",
        message="Reference image is required.",
    )


def test_create_rejects_a_reference_image_for_a_recipe_that_takes_none(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_catalog(fake_services)
    _write_weights(test_state)
    video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()
    image = client.post("/api/assets", json={"path": str(_png(tmp_path / "look.png"))}).json()
    response = client.post(
        "/api/generations/ic-lora-recipes/day-to-night",
        json={
            "params": _params(),
            "inputs": {"video": {"assetId": video["id"]}, "image": {"assetId": image["id"]}},
        },
    )
    assert_http_error(
        response,
        status_code=422,
        code="INVALID_GENERATION_SPEC",
        message="'day-to-night' takes no reference image.",
    )


def test_layout_run_gives_the_pipeline_the_reference_image_at_the_still_frame(
    test_state, fake_services, create_fake_model_files, tmp_path: Path, default_app_settings
) -> None:
    create_fake_model_files()
    create_fake_model_files(model_id=_LOCAL_2_3)
    _install_layout_catalog(fake_services)
    _write_weights(test_state, ic_lora_id="layout-to-render")
    app, _handler = _real_app(test_state, fake_services, default_app_settings)
    with TestClient(app) as client:
        video = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()
        image = client.post("/api/assets", json={"path": str(_png(tmp_path / "look.png"))}).json()
        fake_services.video_processor.register_video(
            video["path"], FakeCapture(frames=["f"] * 9, fps=24, width=16, height=16)
        )
        response = client.post(
            "/api/generations/ic-lora-recipes/layout-to-render",
            json={
                "params": _params(),
                "inputs": {"video": {"assetId": video["id"]}, "image": {"assetId": image["id"]}},
            },
        )
        assert response.status_code == 200, response.text
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if fake_services.ic_lora_pipeline.generate_calls:
                break
            time.sleep(0.01)
    assert len(fake_services.ic_lora_pipeline.generate_calls) == 1
    call = fake_services.ic_lora_pipeline.generate_calls[0]
    assert [(img.path, img.frame_idx) for img in call["images"]] == [(image["path"], -1)]
    assert fake_services.ic_lora_pipeline.last_stage_2_ic_lora is True
    assert call["skip_stage_2"] is False


def test_original_fps_is_the_closest_supported_rate_that_is_not_above_the_source() -> None:
    assert ic_lora_fps_choices(60) == (50, 48, 25, 24)
    assert ic_lora_fps_choices(30) == (25, 24)
    assert ic_lora_fps_choices(29.97) == (25, 24)
    assert ic_lora_fps_choices(25) == (25, 24)
    assert ic_lora_fps_choices(24) == (24,)
    assert ic_lora_fps_choices(23.976) == (24,)
    assert ic_lora_fps_choices(15) == (15,)
    # 24.5 is equally close to 24 and 25. The higher rate is above the source.
    assert ic_lora_original_fps(24.5) == 24
    # 49 is equally close to 48 and 50.
    assert ic_lora_original_fps(49) == 48


def test_a_missing_frame_rate_does_not_fail_the_probe(tmp_path: Path) -> None:
    assert _stream_fps(object()) is None
    kind, _mime, metadata = probe_file(_mp4(tmp_path / "clip.mp4", fps=30))
    assert kind == "video"
    assert metadata.mediaType == "video"
    assert metadata.metadata.fps == 30


def test_create_rejects_an_unreadable_clip_that_has_no_stored_fps(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_catalog(fake_services)
    _write_weights(test_state)
    video = client.post(
        "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
    ).json()
    db = test_state.config.app_data_dir / "store.sqlite3"
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT metadata FROM assets WHERE id = ?", (video["id"],)
    ).fetchone()
    assert row is not None
    meta = json.loads(str(row[0]))
    del meta["metadata"]["fps"]
    conn.execute(
        "UPDATE assets SET metadata = ? WHERE id = ?",
        (json.dumps(meta), video["id"]),
    )
    conn.commit()
    conn.close()
    Path(str(video["path"])).write_bytes(b"not a video")
    response = client.post(
        "/api/generations/ic-lora-recipes/day-to-night",
        json={"params": _params(), "inputs": {"video": {"assetId": video["id"]}}},
    )
    assert_http_error(
        response,
        status_code=422,
        code="INVALID_GENERATION_SPEC",
        message="INVALID_VIDEO_ASSET",
    )


class _RecordingGenerator:
    def __init__(self) -> None:
        self.fps: float | None = None
        self.frames: int | None = None

    def generate_local_reserved(self, request: object, **_kwargs: object) -> object:
        path = Path(str(getattr(request, "input_path")))
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate
            assert rate is not None
            self.fps = float(rate)
            self.frames = sum(1 for _ in container.decode(stream))
        return IcLoraGenerateCancelledResponse(status="cancelled")


def test_a_slower_rate_is_the_file_the_pipeline_receives(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    source = _mp4(tmp_path / "phone.mp4", fps=30, seconds=1)
    asset = AssetRecord(
        id="clip",
        media_kind="video",
        origin="uploaded",
        path=str(source),
        mime_type="video/mp4",
        name="phone.mp4",
        metadata=VideoAssetMetadata(
            mediaType="video",
            metadata=VideoMeta(
                width=16,
                height=16,
                durationMs=1000,
                sizeBytes=source.stat().st_size,
                audioStreamCount=0,
                fps=30,
            ),
        ),
        created_at=0,
    )
    params = IcLoraRecipeStoredParams(
        prompt="night",
        model="ltx-2.5-fast",
        resolution="540p",
        audioMode="off",
        fps=25,
        loras=[LoraEntry(ref="", scale=1.0, catalogId="day-to-night")],
    )
    generation = GenerationRecord(
        id="gen",
        feature="day-to-night",
        contract_version=1,
        status="running",
        spec={
            "params": params.model_dump(mode="json"),
            "inputs": {"video": {"assetId": asset.id}},
        },
        created_at=0,
        queued_at=0,
        attempt_count=1,
        started_at=0,
        outputs=(),
    )
    generator = _RecordingGenerator()

    class _Clip:
        def get_asset(self, asset_id: str) -> AssetRecord | None:
            return asset if asset_id == asset.id else None

    executor = IcLoraRecipeExecutor(
        generator,
        _Clip(),
        lambda: test_state.config.default_models_dir,
        catalog_id="day-to-night",
    )
    try:
        executor.execute(
            generation,
            (
                OutputAllocation(
                    plan=OutputPlan(
                        slot="output",
                        media_kind="video",
                        mime_type="video/mp4",
                        name="output.mp4",
                    ),
                    asset_id="out",
                    dest_path=str(tmp_path / "output.mp4"),
                ),
            ),
        )
    except GenerationCancelledError:
        pass
    else:
        raise AssertionError("the reserved call did not return")
    assert generator.fps == 25
    assert generator.frames == 25


def test_the_crop_filter_sets_fps_only_when_the_selected_rate_is_slower() -> None:
    assert ic_lora_video_filters(None, None) is None
    assert ic_lora_video_filters(None, 25) == "fps=25"
    assert ic_lora_video_filters((8, 8, 0, 0), 25) == "fps=25,crop=8:8:0:0"
    assert ic_lora_video_filters((8, 8, 0, 0), None) == "crop=8:8:0:0"


def test_ic_lora_recipe_table_matches_shared_json() -> None:
    contract_path = Path(__file__).resolve().parents[2] / "shared" / "ic-lora-recipes.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = {
        entry["id"]: entry["catalogId"]
        for entry in contract
    }
    actual = {
        recipe.recipe_id: recipe.catalog_id for recipe in IC_LORA_RECIPES.values()
    }
    assert actual == expected
