"""Motion Track Home recipe: create validation, manual Enhance, and Remote projection."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image

from api_motion_path import MotionPath
from api_types import CreateIcLoraTrackRequest, parse_lora_catalog
from handlers.base import resolve_models_dir
from remote.dto import project_remote_spec
from services.features.ic_lora_recipes import RecipeCreateError, resolve_synthesized_ic_lora_create
from runtime_config.model_download_specs import resolve_ic_lora_path
from tests.lora_catalog_helpers import download_spec

_WEIGHTS = "motion-track.safetensors"
_LOCAL_ENHANCER_MODEL_ID = "ltx-2.3-22b-distilled-1.1"
_PATH: dict[str, object] = {
    "points": [
        {"x": 0.2, "y": 0.3, "handleOut": {"x": 0.1, "y": 0.0}},
        {"x": 0.8, "y": 0.7},
    ],
    "easing": "linear",
    "startHoldSec": 0,
    "endHoldSec": 0,
}


def _png(path: Path, size: tuple[int, int] = (64, 36)) -> Path:
    Image.new("RGB", size, color=(200, 30, 30)).save(path, format="PNG")
    return path


def _install_motion_track(fake_services, test_state) -> None:
    """The shipped entry, so the test follows its real shape. Only the download is swapped."""
    catalog = Path(__file__).resolve().parents[1] / "runtime_config" / "lora_catalog.json"
    shipped = next(
        entry
        for entry in parse_lora_catalog(catalog.read_text()).ic_loras
        if entry.id == "motion-track"
    )
    fake_services.lora_catalog_provider._catalog.ic_loras.append(
        shipped.model_copy(update={"download": download_spec(_WEIGHTS)})
    )
    path = resolve_ic_lora_path(
        resolve_models_dir(test_state.state, test_state.config), "motion-track", _WEIGHTS
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weights")


def _create(
    client,
    image: str,
    path: dict[str, object] | None = None,
    *,
    paths: list[dict[str, object]] | None = None,
    **param_overrides: object,
):
    params: dict[str, object] = {
        "prompt": "the frog conducts",
        "model": "ltx-2.5-fast",
        "resolution": "540p",
        "duration": 4,
        "fps": 24,
        "audioMode": "generated",
        "variantId": "default",
    }
    params.update(param_overrides)
    return client.post(
        "/api/generations/ic-lora-recipes/motion-track",
        json={
            "params": params,
            "inputs": {
                "image": {"assetId": image},
                "paths": paths if paths is not None else [path or _PATH],
            },
        },
    )


def _register_image(client, path: Path) -> str:
    return client.post("/api/assets", json={"path": str(path)}).json()["id"]


def test_create_stores_the_path_and_the_aspect_ratio_closest_to_the_image(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_motion_track(fake_services, test_state)
    image = _register_image(client, _png(tmp_path / "tall.png", size=(36, 64)))
    response = _create(client, image, duration=6, fps=25)
    assert response.status_code == 200, response.text
    spec = response.json()["spec"]
    assert spec["inputs"] == {"image": {"assetId": image}, "paths": [_PATH]}
    assert (spec["params"]["aspectRatio"], spec["params"]["duration"], spec["params"]["fps"]) == (
        "9:16",
        6,
        25,
    )


@pytest.mark.parametrize("count", [1, 8])
def test_create_rejects_a_path_with_fewer_than_2_or_more_than_7_points(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path, count: int
) -> None:
    create_fake_model_files()
    _install_motion_track(fake_services, test_state)
    image = _register_image(client, _png(tmp_path / "a.png"))
    points = [{"x": 0.1 * (i + 1), "y": 0.5} for i in range(count)]
    assert _create(client, image, {**_PATH, "points": points}).status_code == 422


@pytest.mark.parametrize("bad_value", [-11.0, 11.0, 1e308, math.nan, math.inf, -math.inf])
def test_create_rejects_out_of_range_or_non_finite_path_coordinates_and_handles(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path, bad_value: float
) -> None:
    if not math.isfinite(bad_value):
        with pytest.raises(ValueError):
            MotionPath.model_validate(
                {**_PATH, "points": [{"x": bad_value, "y": 0.3}, {"x": 0.8, "y": 0.7}]}
            )
        with pytest.raises(ValueError):
            MotionPath.model_validate(
                {
                    **_PATH,
                    "points": [
                        {"x": 0.2, "y": 0.3, "handleOut": {"x": bad_value, "y": 0.0}},
                        {"x": 0.8, "y": 0.7},
                    ],
                }
            )
        return
    create_fake_model_files()
    _install_motion_track(fake_services, test_state)
    image = _register_image(client, _png(tmp_path / "a.png"))
    bad_anchor = _create(
        client, image, {**_PATH, "points": [{"x": bad_value, "y": 0.3}, {"x": 0.8, "y": 0.7}]}
    )
    bad_handle = _create(
        client,
        image,
        {
            **_PATH,
            "points": [
                {"x": 0.2, "y": 0.3, "handleOut": {"x": bad_value, "y": 0.0}},
                {"x": 0.8, "y": 0.7},
            ],
        },
    )
    assert bad_anchor.status_code == 422
    assert bad_handle.status_code == 422


def test_create_rejects_pauses_that_leave_no_time_to_move(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_motion_track(fake_services, test_state)
    image = _register_image(client, _png(tmp_path / "a.png"))
    fits = _create(client, image, {**_PATH, "startHoldSec": 1, "endHoldSec": 0.5}, duration=2)
    full = _create(client, image, {**_PATH, "startHoldSec": 1, "endHoldSec": 1}, duration=2)
    assert fits.status_code == 200, fits.text
    assert full.status_code == 422
    assert full.json()["code"] == "INVALID_GENERATION_SPEC"


def test_create_stores_every_path_and_checks_the_pauses_of_each(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_motion_track(fake_services, test_state)
    image = _register_image(client, _png(tmp_path / "a.png"))
    second = {**_PATH, "points": [{"x": 0.2, "y": 0.8}, {"x": 0.8, "y": 0.2}]}
    stored = _create(client, image, paths=[_PATH, second], duration=2)
    assert stored.status_code == 200, stored.text
    assert stored.json()["spec"]["inputs"]["paths"] == [_PATH, second]
    full = {**second, "startHoldSec": 1, "endHoldSec": 1}
    refused = _create(client, image, paths=[_PATH, full], duration=2)
    assert refused.status_code == 422
    assert refused.json()["code"] == "INVALID_GENERATION_SPEC"
    assert _create(client, image, paths=[_PATH] * 4).status_code == 422
    assert _create(client, image, paths=[]).status_code == 422


def test_create_rejects_a_sheet_body_on_motion_track_and_a_track_body_on_ingredients(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_motion_track(fake_services, test_state)
    image = _register_image(client, _png(tmp_path / "a.png"))
    sheet = client.post(
        "/api/generations/ic-lora-recipes/motion-track",
        json={
            "params": {
                "prompt": "p",
                "model": "ltx-2.5-fast",
                "resolution": "540p",
                "aspectRatio": "16:9",
                "duration": 4,
                "fps": 24,
            },
            "inputs": {"images": [{"assetId": image}]},
        },
    )
    track = client.post(
        "/api/generations/ic-lora-recipes/ingredients",
        json={
            "params": {
                "prompt": "p",
                "model": "ltx-2.5-fast",
                "resolution": "540p",
                "duration": 4,
                "fps": 24,
            },
            "inputs": {"image": {"assetId": image}, "paths": [_PATH]},
        },
    )
    assert sheet.status_code == 422
    assert track.status_code == 422


def test_create_refuses_a_track_job_over_the_budget_or_above_1080p(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_motion_track(fake_services, test_state)
    image = _register_image(client, _png(tmp_path / "a.png"))
    high_res = _create(client, image, resolution="2160p")
    assert high_res.status_code == 422
    assert high_res.json()["code"] == "IC_LORA_HIGH_RES_UNSUPPORTED"
    # The shared helper names the recipe, not Ingredients.
    assert high_res.json()["message"].startswith("Motion Track Control cannot render above 1080p")

    test_state.config.vram_gb = 8
    over_budget = _create(client, image, resolution="1080p", duration=20, fps=50)
    assert over_budget.status_code == 422
    assert over_budget.json()["code"] == "VIDEO_JOB_TOO_LARGE"


def test_manual_enhance_reads_the_start_frame_with_motion_track_instructions(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    _install_motion_track(fake_services, test_state)
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = "The frog waves his baton."
    frame = _png(tmp_path / "frog.png")
    image_id = _register_image(client, frame)

    response = client.post(
        "/api/enhance-prompt",
        json={
            "prompt": "the frog conducts",
            "icLoraId": "motion-track",
            "imageAssetId": image_id,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["enhancedPrompt"] == "The frog waves his baton."
    call = fake_services.prompt_enhancer_pipeline.enhance_i2v_calls[0]
    assert call["image_path"] == test_state.assets.get_asset(image_id).path
    assert "Motion Track Control" in (call["system_prompt"] or "")


def test_remote_projects_the_paths_and_drops_a_path_shaped_image_id() -> None:
    spec = project_remote_spec(
        {
            "params": {"prompt": "p"},
            "inputs": {"image": {"assetId": "/etc/passwd"}, "paths": [_PATH]},
        }
    )
    assert spec.inputs is not None
    assert spec.inputs.image is None
    assert spec.inputs.paths == [MotionPath.model_validate(_PATH)]


def test_remote_drops_every_path_when_one_stored_path_no_longer_validates() -> None:
    spec = project_remote_spec(
        {
            "params": {"prompt": "p"},
            "inputs": {"paths": [_PATH, {"points": [{"x": 0.1, "y": 0.1}]}]},
        }
    )
    assert spec.inputs is None


def test_create_refuses_a_start_frame_with_no_height_instead_of_failing_with_a_500(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files()
    _install_motion_track(fake_services, test_state)
    req = CreateIcLoraTrackRequest.model_validate(
        {
            "params": {
                "prompt": "p",
                "model": "ltx-2.5-fast",
                "resolution": "540p",
                "duration": 4,
                "fps": 24,
            },
            "inputs": {"image": {"assetId": "a"}, "paths": [_PATH]},
        }
    )
    with pytest.raises(RecipeCreateError) as raised:
        resolve_synthesized_ic_lora_create(
            "motion-track",
            req,
            catalog=fake_services.lora_catalog_provider,
            models_dir=resolve_models_dir(test_state.state, test_state.config),
            hf_authenticated=False,
            require_image=lambda _asset_id: (64, 0),
        )
    assert raised.value.code == "INVALID_GENERATION_SPEC"
