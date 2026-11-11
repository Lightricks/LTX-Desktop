"""Memory advice for a Home IC-LoRA video recipe, before and after a clip is picked.

The advice must agree with what create would run, so ``recipe_run_frames`` (the count create and
execute use) is the oracle wherever a frame count matters.
"""

from __future__ import annotations

import pytest

from _routes._errors import HTTPError
from api_types import IcLoraRecipeStoredParams, IcLoraSettings, LoraEntry
from tests.http_error_assertions import assert_http_error
from tests.lora_catalog_helpers import add_ic_lora

_GIB = 1024**3
_SINGLE_STAGE = IcLoraSettings(skip_stage_2=True, resolution_factor=1.0, audio_mode="source")
# Its canvas follows the clip size, so a crop changes the ask.
_STAGE_2_LORA = IcLoraSettings(skip_stage_2=False, stage_2_ic_lora=True, resolution_factor=1.0)
_STAND_IN_540P = (960, 540)


def _configure_windows(test_state, vram_gib: float) -> None:
    test_state.config.system = "Windows"
    test_state.config.vram_gb = int(vram_gib)
    test_state.config.vram_bytes = int(vram_gib * _GIB)
    test_state.config.total_ram_bytes = 64 * _GIB
    test_state.config.local_generations_mode = "streaming_models_loading"


@pytest.fixture
def recipes(test_state, fake_services, create_fake_model_files):
    """Day to Night and the cutout recipe on the catalog, on a 32 GB Windows card."""

    def install(settings: IcLoraSettings = _SINGLE_STAGE) -> None:
        create_fake_model_files()
        _configure_windows(test_state, 32)
        test_state.state.app_settings.budget_mode = "cautious"
        add_ic_lora(fake_services, id="day-to-night", default_settings=settings)
        add_ic_lora(fake_services, id="alpha-gen", default_settings=settings)

    return install


def _memory(
    client,
    recipe: str = "day-to-night",
    *,
    resolution: str = "540p",
    cells: list[str] | None = None,
    source: tuple[int, int, int] | None = None,
) -> dict:
    body: dict[str, object] = {
        "model": "ltx-2.5-fast",
        "resolution": resolution,
        "cells": cells or [resolution],
        "audioMode": "generated",
        "scale": 1.0,
    }
    if source is not None:
        width, height, frames = source
        body["source"] = {"width": width, "height": height, "frames": frames}
    response = client.post(f"/api/generations/ic-lora-recipes/{recipe}/memory", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _run_frames(test_state, clip_frames: int, *, width: int, height: int) -> int | None:
    """Frames create runs for Day to Night at 540p, or None when it refuses the clip."""
    stored = IcLoraRecipeStoredParams(
        prompt="p",
        model="ltx-2.5-fast",
        resolution="540p",
        audioMode="generated",
        fps=24,
        loras=[LoraEntry(ref="", scale=1.0, catalogId="day-to-night")],
    )
    try:
        count, risk = test_state.ic_lora.recipe_run_frames(
            ic_lora_id="day-to-night",
            source_width=width,
            source_height=height,
            stored=stored,
            clip_frames=clip_frames,
        )
    except HTTPError:
        return None
    return count if risk is None else None


def test_a_full_length_clip_that_runs_by_dropping_its_last_frames_reads_as_full(
    client, test_state, recipes
) -> None:
    recipes()
    full_by_dropping = 0
    for quarter_gib in range(40, 129):
        _configure_windows(test_state, quarter_gib / 4)
        ran = _run_frames(test_state, 240, width=_STAND_IN_540P[0], height=_STAND_IN_540P[1])
        cell = _memory(client)["cells"][0]
        assert cell["full"] == (ran is not None), f"{quarter_gib / 4} GiB: create runs {ran}"
        full_by_dropping += ran is not None and ran < 240
    assert full_by_dropping, "no card size ran the capped clip by dropping frames"


def test_a_cell_never_reports_more_frames_than_the_input_cap(client, recipes) -> None:
    recipes()

    response = _memory(client)

    assert response["max_input_frames"] == 240
    assert response["cells"] == [{"resolution": "540p", "max_frames": 240, "full": True}]


@pytest.mark.parametrize(
    ("settings", "flag", "fits"),
    [
        pytest.param(_SINGLE_STAGE, False, False, id="flag-off"),
        pytest.param(_SINGLE_STAGE, True, True, id="flag-on"),
        pytest.param(_STAGE_2_LORA, True, False, id="stage-2-lora-recipe"),
    ],
)
def test_a_high_res_cell_fits_only_when_create_would_run_it(
    client, recipes, settings: IcLoraSettings, flag: bool, fits: bool
) -> None:
    recipes(settings)
    patched = client.request("PATCH", "/api/feature-flags", json={"highResVideoToVideo": flag})
    assert patched.status_code == 200, patched.text

    cells = _memory(client, resolution="1440p", cells=["1440p", "2160p"])["cells"]

    assert [cell["max_frames"] > 0 for cell in cells] == [fits, fits]


def test_the_clips_ask_is_priced_at_the_frame_count_create_runs(
    client, test_state, recipes
) -> None:
    recipes()
    _configure_windows(test_state, 16)
    width, height = 1920, 1080
    limit = next(
        frames
        for frames in range(9, 2000, 8)
        if _run_frames(test_state, frames, width=width, height=height) is None
    )

    def ask(frames: int) -> float:
        selected = _memory(client, source=(width, height, frames))["selected"]
        assert selected is not None
        return selected["ask_gib"]

    for clip in (limit - 11, limit - 3):
        ran = _run_frames(test_state, clip, width=width, height=height)
        assert ran is not None
        assert ask(clip) == ask(ran), f"a {clip}-frame clip runs {ran} frames"


def test_the_cutout_recipe_prices_the_whole_frame_not_the_crop(client, recipes) -> None:
    recipes(_STAGE_2_LORA)
    # 1920x1200 crops to 3:2 (1800x1200) for a recipe that crops. The cutout keeps every pixel.
    wide, cropped = (1920, 1200, 121), (1800, 1200, 121)

    def ask(recipe: str, source: tuple[int, int, int]) -> float:
        selected = _memory(client, recipe, source=source)["selected"]
        assert selected is not None
        return selected["ask_gib"]

    assert ask("day-to-night", wide) == ask("day-to-night", cropped)
    assert ask("alpha-gen", wide) > ask("day-to-night", wide)


def test_no_number_appears_before_a_clip_is_picked(client, recipes) -> None:
    recipes()

    response = _memory(client, cells=["540p", "720p"])

    assert response["selected"] is None
    assert [cell["resolution"] for cell in response["cells"]] == ["540p", "720p"]


@pytest.mark.parametrize("recipe", ["no-such-recipe", "ingredients"])
def test_an_unknown_or_non_video_recipe_is_not_found(
    client, recipes, fake_services, recipe: str
) -> None:
    recipes()
    add_ic_lora(fake_services, id="ingredients", default_settings=_SINGLE_STAGE)

    response = client.post(
        f"/api/generations/ic-lora-recipes/{recipe}/memory",
        json={"model": "ltx-2.5-fast", "resolution": "540p", "cells": ["540p"]},
    )

    assert_http_error(response, status_code=404, code="UNKNOWN_IC_LORA")


def test_ic_lora_recipe_memory_matches_admit_verdict(
    client, test_state, fake_services, recipes
) -> None:
    recipes()
    patched = client.request("PATCH", "/api/feature-flags", json={"highResVideoToVideo": True})
    assert patched.status_code == 200, patched.text

    # On a 32 GB RAM Windows machine, Day to Night hits ram_shortfall: Cautious
    # refuses it (0 frames), while Brave admits the cells whose GPU ask fits.
    test_state.config.total_ram_bytes = 32 * _GIB
    test_state.state.app_settings.budget_mode = "cautious"
    assert _memory(client, cells=["540p"])["cells"] == [
        {"resolution": "540p", "max_frames": 0, "full": False}
    ]
    test_state.state.app_settings.budget_mode = "brave"
    assert _memory(client, cells=["540p"])["cells"] == [
        {"resolution": "540p", "max_frames": 240, "full": True}
    ]

    # On Mac, 1440p IC-LoRA is an uncovered Extrapolated load: Cautious refuses
    # it at create and in recipe_memory, while Brave admits it.
    test_state.config.system = "Darwin"
    test_state.config.darwin_unified_memory = True
    test_state.config.vram_gb = 64
    test_state.config.vram_bytes = None
    test_state.config.available_ram_gb = 64
    test_state.config.total_ram_bytes = 64 * _GIB
    fake_services.gpu_info.mps_pool_bytes = 48 * _GIB
    test_state.state.app_settings.budget_mode = "cautious"
    assert _memory(client, resolution="1440p", cells=["1440p"])["cells"] == [
        {"resolution": "1440p", "max_frames": 0, "full": False}
    ]
    test_state.state.app_settings.budget_mode = "brave"
    assert _memory(client, resolution="1440p", cells=["1440p"])["cells"][0]["max_frames"] > 0

