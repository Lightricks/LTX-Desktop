"""Ingredients Home recipe: the image list joins into one reference sheet and loops into a clip."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image

from api_types import (
    IcLoraGenerateCancelledResponse,
    IcLoraSynthesizedStoredParams,
    ImageAssetMetadata,
    ImageMeta,
    LoraEntry,
    parse_lora_catalog,
)
from frame_math import compute_num_frames
from handlers.base import resolve_models_dir
from remote.dto import project_remote_spec
from runtime_config.ltx_capabilities import budget_pixels, local_caps
from runtime_config.model_download_specs import resolve_ic_lora_path
from services.features.ic_lora_recipes import IC_LORA_RECIPES
from services.features.video.ic_lora_sheet_recipe import IcLoraSheetRecipeExecutor
from services.features.video.job_common import resolve_job_local_model_id
from services.features.video.reference_sheet import build_reference_sheet, layout_panels
from services.generation_interrupt import GenerationCancelledError
from services.generation_queue.types import OutputAllocation, OutputPlan
from tests.fakes.generation import FakeJobSession
from services.records import AssetRecord, GenerationRecord
from tests.fakes import FakeResponse
from tests.lora_catalog_helpers import add_ic_lora, download_spec

_WEIGHTS = "ingredients.safetensors"
# 2.3's gemma3 encoder doubles as the local enhancer, so its plain bundle is enough.
_LOCAL_ENHANCER_MODEL_ID = "ltx-2.3-22b-distilled-1.1"
_LOCAL_ENHANCER_OFFERING = "ltx-2.3-fast"
_FILLED = "Reference sheet: a panda; a lemonade stand. Generated video: the panda pours a cup."


def _png(path: Path, size: tuple[int, int] = (64, 48), color: tuple[int, int, int] = (200, 30, 30)) -> Path:
    Image.new("RGB", size, color=color).save(path, format="PNG")
    return path


def _mp4(path: Path) -> Path:
    writer = imageio.get_writer(str(path), fps=8, codec="libx264", macro_block_size=None)
    for _ in range(8):
        writer.append_data(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.close()
    return path


def _install_ingredients(fake_services, test_state) -> None:
    """The shipped entry, so the test follows its real shape. Only the download is swapped."""
    catalog = Path(__file__).resolve().parents[1] / "runtime_config" / "lora_catalog.json"
    shipped = next(
        entry
        for entry in parse_lora_catalog(catalog.read_text()).ic_loras
        if entry.id == "ingredients"
    )
    fake_services.lora_catalog_provider._catalog.ic_loras.append(
        shipped.model_copy(update={"download": download_spec(_WEIGHTS)})
    )
    path = resolve_ic_lora_path(
        resolve_models_dir(test_state.state, test_state.config), "ingredients", _WEIGHTS
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weights")


def _install_local_enhancer(test_state, create_fake_model_files) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    test_state.state.app_settings.active_ltx_model_id = _LOCAL_ENHANCER_MODEL_ID
    test_state.state.app_settings.use_local_text_encoder = True


def _params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "prompt": "the dog sits beside the man",
        "model": "ltx-2.5-fast",
        "resolution": "540p",
        "aspectRatio": "16:9",
        "duration": 6,
        "fps": 24,
        "audioMode": "generated",
        "variantId": "default",
    }
    params.update(overrides)
    return params


def _create(client, inputs: dict[str, object], **param_overrides: object):
    return client.post(
        "/api/generations/ic-lora-recipes/ingredients",
        json={"params": _params(**param_overrides), "inputs": inputs},
    )


def _register_image(client, path: Path) -> str:
    return client.post("/api/assets", json={"path": str(path)}).json()["id"]


# --- the sheet ---------------------------------------------------------------------------


def test_a_panel_never_leaves_the_canvas_or_overlaps_another_panel() -> None:
    aspects = [16 / 9, 9 / 16, 1.0, 4 / 3, 3 / 4, 21 / 9, 16 / 9, 9 / 16, 1.0, 3 / 2]
    for width, height in ((1280, 720), (720, 1280), (960, 960)):
        panels = layout_panels(aspects, width, height, 16)
        assert len(panels) == len(aspects)
        for _, x, y, w, h in panels:
            assert x >= -1e-6 and y >= -1e-6
            assert x + w <= width + 1 and y + h <= height + 1
        for index, (_, ax, ay, aw, ah) in enumerate(panels):
            for _, bx, by, bw, bh in panels[index + 1 :]:
                apart = ax + aw <= bx + 1e-6 or bx + bw <= ax + 1e-6 or ay + ah <= by + 1e-6 or by + bh <= ay + 1e-6
                assert apart


def test_one_image_keeps_its_aspect_on_a_black_canvas_instead_of_stretching(tmp_path: Path) -> None:
    square = _png(tmp_path / "square.png", size=(500, 500))
    sheet = tmp_path / "sheet.png"
    build_reference_sheet([str(square)], 1280, 720, sheet)
    with Image.open(sheet) as image:
        assert image.size == (1280, 720)
        assert image.getpixel((640, 360)) == (200, 30, 30)
        assert image.getpixel((100, 360)) == (0, 0, 0)
        assert image.getpixel((1200, 360)) == (0, 0, 0)
        # The panel is centred: the black margin is the same on both sides.
        assert image.getpixel((279, 360)) == (0, 0, 0)
        assert image.getpixel((280, 360)) == (200, 30, 30)
        assert image.getpixel((999, 360)) == (200, 30, 30)
        assert image.getpixel((1000, 360)) == (0, 0, 0)


# --- create ------------------------------------------------------------------------------


def test_create_stores_the_images_in_order_with_duration_aspect_and_loop_rate(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_ingredients(fake_services, test_state)
    first = _register_image(client, _png(tmp_path / "a.png"))
    second = _register_image(client, _png(tmp_path / "b.png", size=(48, 64)))
    response = _create(
        client,
        {"images": [{"assetId": second}, {"assetId": first}]},
        duration=8,
        aspectRatio="9:16",
        fps=25,
    )
    assert response.status_code == 200, response.text
    spec = response.json()["spec"]
    assert [ref["assetId"] for ref in spec["inputs"]["images"]] == [second, first]
    assert "video" not in spec["inputs"]
    assert (spec["params"]["duration"], spec["params"]["aspectRatio"], spec["params"]["fps"]) == (
        8,
        "9:16",
        25,
    )


@pytest.mark.parametrize("count", [0, 11])
def test_create_rejects_an_image_list_outside_1_to_10(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path, count: int
) -> None:
    create_fake_model_files()
    _install_ingredients(fake_services, test_state)
    image = _register_image(client, _png(tmp_path / "a.png"))
    response = _create(client, {"images": [{"assetId": image}] * count})
    assert response.status_code == 422


def test_create_rejects_a_video_asset_in_the_image_list(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_ingredients(fake_services, test_state)
    image = _register_image(client, _png(tmp_path / "a.png"))
    clip = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()["id"]
    response = _create(client, {"images": [{"assetId": image}, {"assetId": clip}]})
    assert response.status_code == 422
    assert response.json() == {
        "code": "INVALID_GENERATION_SPEC",
        "message": "image must reference an image asset",
    }


def test_create_refuses_a_sheet_over_the_job_budget(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_ingredients(fake_services, test_state)
    test_state.config.vram_gb = 8
    image = _register_image(client, _png(tmp_path / "a.png"))
    response = _create(
        client, {"images": [{"assetId": image}]}, resolution="1080p", duration=20, fps=50
    )
    assert response.status_code == 422
    assert response.json()["code"] == "VIDEO_JOB_TOO_LARGE"


def test_create_rejects_images_with_a_video_and_rejects_neither(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_ingredients(fake_services, test_state)
    image = _register_image(client, _png(tmp_path / "a.png"))
    clip = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}).json()["id"]
    both = _create(client, {"images": [{"assetId": image}], "video": {"assetId": clip}})
    neither = _create(client, {})
    assert both.status_code == 422
    assert neither.status_code == 422


def test_create_rejects_a_duration_or_loop_rate_outside_the_lists_and_a_missing_aspect(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_ingredients(fake_services, test_state)
    inputs = {"images": [{"assetId": _register_image(client, _png(tmp_path / "a.png"))}]}
    assert _create(client, inputs, duration=7).status_code == 422
    rate = _create(client, inputs, fps=30)
    assert rate.status_code == 422
    assert rate.json()["code"] == "INVALID_GENERATION_SPEC"
    missing = _create(client, inputs, aspectRatio=None)
    assert missing.status_code == 422


# --- run ---------------------------------------------------------------------------------


class _SheetRecorder:
    def __init__(self) -> None:
        self.request: object | None = None
        self.sheet_size: tuple[int, int] | None = None
        self.sheet_path: Path | None = None

    def generate_local_reserved(self, request: object, **_kwargs: object) -> object:
        self.request = request
        self.sheet_path = Path(str(getattr(request, "input_path")))
        with Image.open(self.sheet_path) as sheet:
            self.sheet_size = sheet.size
        return IcLoraGenerateCancelledResponse(status="cancelled")


def _image_asset(path: Path, asset_id: str) -> AssetRecord:
    return AssetRecord(
        id=asset_id,
        media_kind="image",
        origin="uploaded",
        path=str(path),
        mime_type="image/png",
        name=path.name,
        metadata=ImageAssetMetadata(
            mediaType="image",
            metadata=ImageMeta(width=64, height=48),
        ),
        created_at=0,
    )


class _Assets:
    def __init__(self, assets: list[AssetRecord]) -> None:
        self._assets = {asset.id: asset for asset in assets}

    def get_asset(self, asset_id: str) -> AssetRecord | None:
        return self._assets.get(asset_id)


def _sheet_job(
    tmp_path: Path,
    *,
    duration: int,
    fps: int,
    images: int = 1,
    newest_first: bool = False,
    model: str = "ltx-2.5-fast",
) -> tuple[GenerationRecord, _Assets, tuple[OutputAllocation, ...]]:
    assets = [
        _image_asset(_png(tmp_path / f"p{index}.png", color=(40 * index + 10, 30, 30)), f"img{index}")
        for index in range(images)
    ]
    ordered = list(reversed(assets)) if newest_first else assets
    params = IcLoraSynthesizedStoredParams(
        prompt="p",
        model=model,
        resolution="540p",
        audioMode="generated",
        aspectRatio="9:16",
        duration=duration,
        fps=fps,
        loras=[LoraEntry(ref="", scale=1.35, catalogId="ingredients")],
    )
    generation = GenerationRecord(
        id="gen",
        feature="ingredients",
        contract_version=1,
        status="running",
        spec={
            "params": params.model_dump(mode="json"),
            "inputs": {"images": [{"assetId": asset.id} for asset in ordered]},
        },
        created_at=0,
        queued_at=0,
        attempt_count=1,
        started_at=0,
        outputs=(),
    )
    in_flight = tmp_path / "app_data" / "assets" / ".in-flight"
    in_flight.mkdir(parents=True, exist_ok=True)
    outputs = (
        OutputAllocation(
            plan=OutputPlan(slot="output", media_kind="video", mime_type="video/mp4", name="o.mp4"),
            asset_id="out",
            dest_path=str(in_flight / "out.mp4"),
        ),
    )
    return generation, _Assets(assets), outputs


def test_run_builds_the_sheet_at_the_denoise_size_and_deletes_it(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    generation, assets, outputs = _sheet_job(tmp_path, duration=8, fps=25)
    recorder = _SheetRecorder()
    executor = IcLoraSheetRecipeExecutor(
        recorder, assets, lambda: test_state.config.default_models_dir, recipe=IC_LORA_RECIPES["ingredients"]
    )
    with pytest.raises(GenerationCancelledError):
        executor.execute(generation, outputs, FakeJobSession())
    assert recorder.sheet_size == budget_pixels(local_caps(resolve_job_local_model_id(test_state.config.default_models_dir, "ltx-2.5-fast")), "540p", "9:16", mode="video")
    assert recorder.sheet_path is not None and not recorder.sheet_path.exists()


def test_run_makes_the_chosen_duration_at_the_chosen_rate_not_the_catalog_default(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # 12 s is not in the catalog duration list. A run that drops loop_seconds makes the 5 s default.
    create_fake_model_files()
    _install_ingredients(fake_services, test_state)
    generation, assets, outputs = _sheet_job(tmp_path, duration=12, fps=25)
    executor = IcLoraSheetRecipeExecutor(
        test_state.ic_lora, assets, lambda: test_state.config.default_models_dir, recipe=IC_LORA_RECIPES["ingredients"]
    )
    executor.execute(generation, outputs, FakeJobSession())
    assert fake_services.ic_lora_pipeline.generate_calls[-1]["num_frames"] == compute_num_frames(12, 25)


# --- Enhance -----------------------------------------------------------------------------


def _enhance(client, **body: object):
    return client.post("/api/enhance-prompt", json={"icLoraId": "ingredients", **body})


def test_enhance_builds_the_reference_sheet_in_in_flight_for_the_rewriter(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # A text-only call or raw panel pass would skip constructing the reference sheet.
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    _install_ingredients(fake_services, test_state)
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _FILLED
    first = _png(tmp_path / "a.png")
    second = _png(tmp_path / "b.png", size=(48, 64))
    ids = [_register_image(client, second), _register_image(client, first)]

    response = _enhance(
        client,
        prompt="panda selling lemonade",
        imageAssetIds=ids,
        resolution="540p",
        aspectRatio="9:16",
    )

    assert response.status_code == 200, response.text
    assert response.json()["enhancedPrompt"] == _FILLED
    call = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[0]
    assert len(call["frame_paths"]) == 1
    sheet = Path(call["frame_paths"][0])
    assert sheet.parent == test_state.config.app_data_dir / "assets" / ".in-flight"
    assert sheet.name.startswith("sheet-") and sheet.is_file()
    with Image.open(sheet) as image:
        assert image.size == (576, 1024)


def test_enhance_evicts_old_sheets_without_deleting_other_in_flight_files(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # Sheet cleanup must never touch in-flight generation outputs in .in-flight.
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    _install_ingredients(fake_services, test_state)
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _FILLED
    in_flight = test_state.config.app_data_dir / "assets" / ".in-flight"
    in_flight.mkdir(parents=True, exist_ok=True)
    other_output = in_flight / "active-generation.mp4"
    other_output.write_bytes(b"video")
    other_image = in_flight / "other-temp.png"
    other_image.write_bytes(b"image")

    created_sheets: list[Path] = []
    for index in range(10):
        panel = _png(tmp_path / f"panel-{index}.png", color=(index * 20, 40, 60))
        asset_id = _register_image(client, panel)
        assert _enhance(client, prompt="p", imageAssetIds=[asset_id]).status_code == 200
        call = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[-1]
        created_sheets.append(Path(call["frame_paths"][0]))

    assert other_output.is_file()
    assert other_image.is_file()
    assert not created_sheets[0].exists()
    assert created_sheets[-1].is_file()


def test_enhance_writes_from_the_panel_images_when_the_prompt_is_blank(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    _install_ingredients(fake_services, test_state)
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _FILLED
    image = _register_image(client, _png(tmp_path / "a.png"))

    assert _enhance(client, prompt="", imageAssetIds=[image]).status_code == 200


def test_enhance_asks_for_the_two_labelled_long_prompt_with_the_card_examples(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # Without the examples and the rule the rewrite is one short action again.
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    _install_ingredients(fake_services, test_state)
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _FILLED
    image = _register_image(client, _png(tmp_path / "a.png"))

    assert _enhance(client, prompt="", imageAssetIds=[image]).status_code == 200

    call = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[0]
    assert call["role"] == "panels"
    assert "FreshMart" in call["system_prompt"]
    assert "Reference sheet:" in call["system_prompt"] and "Generated video:" in call["system_prompt"]


def test_enhance_rejects_a_blank_prompt_with_no_images(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    _install_ingredients(fake_services, test_state)

    assert _enhance(client, prompt="  ").status_code == 422
    assert _enhance(client, prompt="  ", imageAssetIds=[]).status_code == 422


def test_enhance_rejects_more_than_the_image_limit(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    _install_ingredients(fake_services, test_state)
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _FILLED
    image = _register_image(client, _png(tmp_path / "a.png"))

    assert _enhance(client, prompt="p", imageAssetIds=[image] * 10).status_code == 200
    assert _enhance(client, prompt="p", imageAssetIds=[image] * 11).status_code == 422


def test_enhance_rejects_a_list_of_images_for_an_entry_that_takes_a_video(
    client, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    add_ic_lora(fake_services, id="deblur", name="Deblur")
    image = _register_image(client, _png(tmp_path / "a.png"))

    response = client.post(
        "/api/enhance-prompt", json={"prompt": "p", "icLoraId": "deblur", "imageAssetIds": [image]}
    )

    assert response.status_code == 422


# --- Enhance at Generate -----------------------------------------------------------------

def _run_sheet_job(test_state, assets: _Assets, generation, outputs) -> None:
    IcLoraSheetRecipeExecutor(
        test_state.ic_lora, assets, lambda: test_state.config.default_models_dir, recipe=IC_LORA_RECIPES["ingredients"]
    ).execute(generation, outputs, FakeJobSession())


def test_generate_shows_the_reference_sheet_to_the_automatic_rewrite(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # Automatic enhancement at Generate must see the constructed reference sheet.
    _install_local_enhancer(test_state, create_fake_model_files)
    _install_ingredients(fake_services, test_state)
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _FILLED
    generation, assets, outputs = _sheet_job(
        tmp_path, duration=5, fps=24, images=2, newest_first=True, model=_LOCAL_ENHANCER_OFFERING
    )

    _run_sheet_job(test_state, assets, generation, outputs)

    assert fake_services.ic_lora_pipeline.generate_calls[-1]["prompt"] == _FILLED
    call = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[0]
    assert len(call["frame_paths"]) == 1
    sheet = Path(call["frame_paths"][0])
    assert sheet.parent.name == ".in-flight" and sheet.name.startswith("sheet-")
    assert not sheet.exists()


def test_generate_reuses_the_sheet_from_manual_enhance_and_unlinks_it(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # When manual Enhance already built the sheet in .in-flight, Generate reuses that file.
    create_fake_model_files()
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    _install_ingredients(fake_services, test_state)
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _FILLED
    generation, assets, outputs = _sheet_job(
        tmp_path, duration=5, fps=24, images=2, newest_first=True
    )
    ids = [
        _register_image(client, Path(assets.get_asset("img1").path)),
        _register_image(client, Path(assets.get_asset("img0").path)),
    ]
    for index, asset_id in enumerate(ids):
        key = "img1" if index == 0 else "img0"
        assets._assets[key] = _image_asset(Path(test_state.assets.get_asset(asset_id).path), key)

    response = _enhance(
        client,
        prompt="panda",
        imageAssetIds=ids,
        resolution="540p",
        aspectRatio="9:16",
    )
    assert response.status_code == 200
    sheet = Path(fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[0]["frame_paths"][0])
    assert sheet.is_file()
    # Mark the cached sheet with a distinctive pixel so we can tell if Generate overwrote it.
    marker = Image.new("RGB", (576, 1024), color=(12, 34, 56))
    marker.save(sheet, format="PNG")
    seen_pixel: tuple[int, int, int] | None = None

    class _PixelRecorder(_SheetRecorder):
        def generate_local_reserved(self, request: object, **kwargs: object) -> object:
            nonlocal seen_pixel
            path = Path(str(getattr(request, "input_path")))
            with Image.open(path) as img:
                seen_pixel = img.getpixel((0, 0))
            return super().generate_local_reserved(request, **kwargs)

    recorder = _PixelRecorder()
    executor = IcLoraSheetRecipeExecutor(
        recorder, assets, lambda: test_state.config.default_models_dir, recipe=IC_LORA_RECIPES["ingredients"]
    )
    with pytest.raises(GenerationCancelledError):
        executor.execute(generation, outputs, FakeJobSession())

    assert recorder.sheet_path == sheet
    assert seen_pixel == (12, 34, 56)
    assert not sheet.exists()


@pytest.mark.parametrize(
    ("preference", "key", "expected"),
    [
        ("local", "key", _FILLED),
        (None, "key", _FILLED),
        ("api", "", _FILLED),
        ("api", "key", "Reference sheet: a panda. Generated video: it waves."),
    ],
    ids=["local-choice", "no-choice", "api-choice-no-key", "api-choice-with-key"],
)
def test_only_an_explicit_api_choice_with_a_key_sends_the_panel_images_to_gemini(
    test_state, fake_services, create_fake_model_files, tmp_path: Path, preference, key, expected
) -> None:
    # Automatic enhancement would upload the user's images, so Gemini must be a choice.
    _install_local_enhancer(test_state, create_fake_model_files)
    test_state.state.app_settings.prompt_enhancer_provider_preference = preference
    test_state.state.app_settings.gemini_api_key = key
    _install_ingredients(fake_services, test_state)
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _FILLED
    gemini_reply = "Reference sheet: a panda. Generated video: it waves."
    test_state.http.queue(
        "post",
        FakeResponse(
            status_code=200,
            json_payload={"candidates": [{"content": {"parts": [{"text": gemini_reply}]}}]},
        ),
    )
    item = fake_services.lora_catalog_provider.get_ic_lora("ingredients")

    resolved = test_state.prompt_enhancement.resolve_for_generation(
        "panda selling lemonade",
        provenance="typed",
        generation_seed=1,
        explicit_generation_seed=True,
        explore_generation=True,
        ic_lora=item,
        panel_paths=[str(_png(tmp_path / "a.png"))],
    )

    assert resolved.prompt == expected
    if expected == _FILLED:
        assert test_state.http.calls == []


# --- Remote ------------------------------------------------------------------------------


def test_remote_projects_image_ids_in_order_and_drops_a_path_shaped_id() -> None:
    spec = project_remote_spec(
        {
            "params": {"prompt": "p"},
            "inputs": {"images": [{"assetId": "a"}, {"assetId": "/etc/passwd"}, {"assetId": "b"}]},
        }
    )
    assert spec.inputs is not None and spec.inputs.images is not None
    assert [ref.assetId for ref in spec.inputs.images] == ["a", "b"]
