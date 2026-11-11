"""Tile size of Refine Details: the preflight verdicts, the create refusal and the stored choice."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import imageio.v2 as imageio
import numpy as np
from starlette.testclient import TestClient

from api_types import IcLoraSettings, parse_lora_catalog
from handlers.base import resolve_models_dir
from runtime_config.ic_lora_tiling import (
    TILE_SIZES,
    IcLoraTileSize,
    IcLoraTiling,
    with_tiling_overrides,
)
from runtime_config.model_download_specs import resolve_ic_lora_path
from tests.fakes.services import FakeCapture
from tests.lora_catalog_helpers import add_ic_lora, download_spec

_REFINE = "refine-details"
_DAY_TO_NIGHT = "day-to-night"
_WEIGHTS = "day-to-night.safetensors"
_CATALOG = Path(__file__).resolve().parents[1] / "runtime_config" / "lora_catalog.json"
_GIB = 1024**3
_SIZES = ["standard", "large", "extra-large"]


def test_tile_size_names_match_the_controls_file() -> None:
    assert set(get_args(IcLoraTileSize)) == set(TILE_SIZES)


def test_tiling_override_error_is_the_validator_text() -> None:
    tiling = IcLoraTiling(long_side=1024, short_side=576, window_frames=73, carry_frames=25)
    try:
        with_tiling_overrides(tiling, overlap_fraction=None, carry_frames=41, max_window_frames=57)
    except ValueError as exc:
        assert str(exc) == "window_frames must be more than twice carry_frames"
    else:
        raise AssertionError("expected ValueError")


def _mp4(path: Path, *, width: int = 1920, height: int = 1080, frames: int = 49) -> Path:
    writer = imageio.get_writer(str(path), fps=24, codec="libx264", macro_block_size=None)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.append_data(frame)
    writer.close()
    return path


def _install(fake_services, test_state, ic_lora_id: str) -> None:
    item = next(e for e in parse_lora_catalog(_CATALOG.read_text()).ic_loras if e.id == ic_lora_id)
    fake_services.lora_catalog_provider._catalog.ic_loras.append(item)
    models_dir = resolve_models_dir(test_state.state, test_state.config)
    path = resolve_ic_lora_path(models_dir, item.id, item.download.variants[0].filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weights")


def _configure_windows(test_state, *, vram_gib: float, budget_mode: str = "brave") -> None:
    test_state.config.system = "Windows"
    test_state.config.vram_gb = int(vram_gib)
    test_state.config.vram_bytes = int(vram_gib * _GIB)
    test_state.config.total_ram_bytes = 64 * _GIB
    test_state.config.local_generations_mode = "streaming_models_loading"
    test_state.state.app_settings.budget_mode = budget_mode


def _body(video_id: str, resolution: str, **params: object) -> dict[str, object]:
    return {
        "params": {
            "prompt": "restore fine detail",
            "model": "ltx-2.5-fast",
            "resolution": resolution,
            **params,
        },
        "inputs": {"video": {"assetId": video_id}},
    }


def _preflight(client: TestClient, recipe_id: str, body: dict[str, object]):
    return client.post(
        f"/api/generations/ic-lora-recipes/{recipe_id}/preflight",
        json={**body, "tileSizes": _SIZES},
    )


def _ingest(client: TestClient, tmp_path: Path, **kwargs) -> str:
    return client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4", **kwargs))}).json()["id"]


def test_each_tile_size_is_refused_as_generate_refuses_it(
    client, fake_services, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    # A card where 2160p is refused for every size. Cautious refuses what Brave would run as a bet.
    _configure_windows(test_state, vram_gib=13.25, budget_mode="cautious")
    video_id = _ingest(client, tmp_path)
    body = _body(video_id, "2160p")

    cells = {cell["tileSize"]: cell for cell in _preflight(client, _REFINE, body).json()["cells"]}

    assert all(not cell["ok"] for cell in cells.values())
    for size, cell in cells.items():
        params = {} if size == "standard" else {"tilingSize": size}
        refused = client.post(
            f"/api/generations/ic-lora-recipes/{_REFINE}", json=_body(video_id, "2160p", **params)
        )
        assert refused.status_code == 422, size
        assert refused.json()["message"] == cell["message"], size


def test_a_size_that_fits_has_a_clear_verdict_and_is_stored_with_the_generation(
    client, fake_services, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=24)
    video_id = _ingest(client, tmp_path)
    body = _body(video_id, "1080p", tilingSize="large")

    cells = _preflight(client, _REFINE, body).json()["cells"]
    assert [(cell["ok"], cell["message"]) for cell in cells] == [(True, None)] * 3

    created = client.post(f"/api/generations/ic-lora-recipes/{_REFINE}", json=body)
    assert created.status_code == 200, created.text
    assert created.json()["spec"]["params"]["tilingSize"] == "large"


def test_a_recipe_without_tiling_controls_refuses_a_tile_size(
    client, fake_services, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    add_ic_lora(
        fake_services,
        id=_DAY_TO_NIGHT,
        download=download_spec(_WEIGHTS),
        input={"kind": "video"},
        default_settings=IcLoraSettings(skip_stage_2=True, resolution_factor=1.5, audio_mode="source"),
    )
    models_dir = resolve_models_dir(test_state.state, test_state.config)
    weights = resolve_ic_lora_path(models_dir, _DAY_TO_NIGHT, _WEIGHTS)
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"weights")
    video_id = _ingest(client, tmp_path)

    refused = client.post(
        f"/api/generations/ic-lora-recipes/{_DAY_TO_NIGHT}",
        json=_body(video_id, "540p", tilingSize="large", variantId="default"),
    )

    assert refused.status_code == 422, refused.text


def test_preflight_accepts_an_empty_prompt_and_create_still_requires_one(
    client, fake_services, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=24)
    video_id = _ingest(client, tmp_path)
    body = _body(video_id, "1080p", prompt="")

    preflight = _preflight(client, _REFINE, body)
    assert preflight.status_code == 200, preflight.text
    assert len(preflight.json()["cells"]) == 3

    created = client.post(f"/api/generations/ic-lora-recipes/{_REFINE}", json=body)
    assert created.status_code == 422
    assert created.json()["message"] == "Prompt is required."


def test_an_unknown_recipe_at_1440p_is_unknown_not_a_closed_high_cell(
    client, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _configure_windows(test_state, vram_gib=24)
    video_id = _ingest(client, tmp_path)

    refused = client.post(
        "/api/generations/ic-lora-recipes/not-a-recipe",
        json=_body(video_id, "1440p"),
    )

    assert refused.status_code == 422
    assert refused.json()["code"] == "IC_LORA_UNKNOWN"


def test_preflight_of_an_unknown_video_fails_like_the_create_not_with_empty_verdicts(
    client, fake_services, test_state, create_fake_model_files
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    body = _body("missing-asset", "1080p")

    preflight = _preflight(client, _REFINE, body)
    created = client.post(f"/api/generations/ic-lora-recipes/{_REFINE}", json=body)

    assert preflight.status_code == created.status_code
    assert preflight.status_code >= 400
    assert preflight.json() == created.json()


def _window_preflight(client: TestClient, body: dict[str, object], windows: list[int]):
    return client.post(
        f"/api/generations/ic-lora-recipes/{_REFINE}/preflight", json={**body, "maxWindows": windows}
    )


def test_a_window_under_the_shortest_one_is_a_422_not_a_server_error_that_drops_every_verdict(
    client, fake_services, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=24)
    video_id = _ingest(client, tmp_path)

    refused = _window_preflight(client, _body(video_id, "1080p"), [73, 33])

    assert refused.status_code == 422


def test_a_window_is_stored_and_a_window_the_pipeline_refuses_is_refused_at_create(
    client, fake_services, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=24)
    video_id = _ingest(client, tmp_path)
    url = f"/api/generations/ic-lora-recipes/{_REFINE}"

    created = client.post(url, json=_body(video_id, "1080p", maxWindowFrames=57))
    assert created.status_code == 200, created.text
    assert created.json()["spec"]["params"]["maxWindowFrames"] == 57

    # 105 is above the 97-frame maximum.
    refused = client.post(url, json=_body(video_id, "1080p", maxWindowFrames=105))
    assert refused.status_code == 422
    assert refused.json()["code"] == "INVALID_GENERATION_SPEC"
    # 41 is not more than twice a carry the user set to 25.
    refused = client.post(url, json=_body(video_id, "1080p", maxWindowFrames=41, carryFrames=25))
    assert refused.status_code == 422
    assert refused.json()["code"] == "INVALID_GENERATION_SPEC"


def test_each_window_verdict_is_the_refusal_generate_returns_for_that_window(
    client, fake_services, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=24)
    video_id = _ingest(client, tmp_path)
    windows = [73, 57, 41, 81]

    for params in ({}, {"carryFrames": 17}):
        body = _body(video_id, "1080p", **params)
        cells = _window_preflight(client, body, windows).json()["maxWindowCells"]
        assert [cell["maxWindowFrames"] for cell in cells] == windows
        for cell in cells:
            created = client.post(
                f"/api/generations/ic-lora-recipes/{_REFINE}",
                json=_body(video_id, "1080p", maxWindowFrames=cell["maxWindowFrames"], **params),
            )
            assert cell["ok"] == (created.status_code == 200), (params, cell)
            if not cell["ok"]:
                assert cell["message"] == created.json()["message"], (params, cell)


def test_tile_size_verdicts_are_judged_with_the_window_the_form_holds(
    client, fake_services, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=24)
    video_id = _ingest(client, tmp_path)

    # A window that cannot run makes every tile size fail: the tile preflight kept the window.
    cells = _preflight(client, _REFINE, _body(video_id, "1080p", maxWindowFrames=105)).json()["cells"]

    assert [cell["ok"] for cell in cells] == [False] * 3


def _carry_preflight(client: TestClient, body: dict[str, object], carries: list[int]):
    return client.post(
        f"/api/generations/ic-lora-recipes/{_REFINE}/preflight",
        json={**body, "tileSizes": [], "maxWindows": [], "carries": carries},
    )


def test_preflight_returns_carry_verdicts_and_the_defaults_the_run_uses(
    client, fake_services, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=24)
    video_id = _ingest(client, tmp_path)
    carries = [17, 25, 33]

    local = _carry_preflight(client, _body(video_id, "1080p", maxWindowFrames=57), carries)
    assert local.status_code == 200, local.text
    payload = local.json()
    # The test clip is short, and 1080p fits it in one pass, so Auto is a single pass.
    assert payload["defaults"] == {
        "carryFrames": 25,
        "overlapFraction": 0.25,
        "auto": {"kind": "single-pass"},
    }
    cells = {cell["carryFrames"]: cell for cell in payload["carryCells"]}
    assert [cells[carry]["ok"] for carry in carries] == [True, True, False]
    refused = client.post(
        f"/api/generations/ic-lora-recipes/{_REFINE}",
        json=_body(video_id, "1080p", maxWindowFrames=57, carryFrames=33),
    )
    assert cells[33]["message"] == refused.json()["message"]

    high = _carry_preflight(client, _body(video_id, "1440p"), carries)
    assert high.status_code == 200, high.text
    assert high.json()["defaults"]["carryFrames"] == 17


def test_auto_carry_gives_way_to_a_chunk_length_the_default_carry_cannot_run(
    client, fake_services, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=24)
    video_id = _ingest(client, tmp_path)

    # A 41-frame chunk is not more than twice the carry of 25, so Auto carry runs 17.
    auto = _carry_preflight(client, _body(video_id, "1080p", maxWindowFrames=41), [])
    assert auto.json()["defaults"]["carryFrames"] == 17
    # A carry the user chose stays, and the same chunk length is then refused.
    pinned = client.post(
        f"/api/generations/ic-lora-recipes/{_REFINE}",
        json=_body(video_id, "1080p", maxWindowFrames=41, carryFrames=25),
    )
    assert pinned.status_code == 422


def _ingest_sized(client: TestClient, test_state, tmp_path: Path, *, width: int, height: int, frames: int) -> str:
    """An asset whose probe reports this size and length. The backend probe is a fake in tests."""
    asset = client.post("/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4", frames=1))}).json()
    test_state.video_processor.register_video(
        asset["path"], FakeCapture(frames=["f"] * frames, fps=24, width=width, height=height)
    )
    return asset["id"]


def test_auto_reports_the_window_a_long_clip_runs_and_a_short_clip_runs_whole(
    client, fake_services, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=24)
    long_clip = _ingest_sized(client, test_state, tmp_path, width=1920, height=1080, frames=137)

    defaults = _carry_preflight(client, _body(long_clip, "1080p"), []).json()["defaults"]

    # 1080p stops at 97 frames, so a 137-frame clip is chunked.
    assert defaults["auto"]["kind"] == "windowed"
    assert defaults["auto"]["frames"] <= 97


def test_a_maximum_above_the_cap_is_a_disabled_cell_with_the_message_create_returns(
    client, fake_services, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=48)
    video_id = _ingest_sized(client, test_state, tmp_path, width=3840, height=2160, frames=137)
    windows = [97, 73, 57, 41]

    cells = {
        cell["maxWindowFrames"]: cell
        for cell in _window_preflight(client, _body(video_id, "2160p"), windows).json()["maxWindowCells"]
    }

    assert cells[97]["ok"] is False
    assert [cells[window]["ok"] for window in (73, 57, 41)] == [True] * 3
    refused = client.post(
        f"/api/generations/ic-lora-recipes/{_REFINE}",
        json=_body(video_id, "2160p", maxWindowFrames=97),
    )
    assert refused.status_code == 422
    assert refused.json()["message"] == cells[97]["message"]


def test_generate_runs_the_carry_auto_picked_for_the_chunk_length_the_run_chose(
    client, fake_services, test_state, create_fake_model_files, tmp_path
) -> None:
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    _install(fake_services, test_state, _REFINE)
    _configure_windows(test_state, vram_gib=24)
    video_id = _ingest_sized(client, test_state, tmp_path, width=1280, height=704, frames=241)
    path = client.get(f"/api/assets/{video_id}").json()["path"]

    response = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": _REFINE,
            "input_path": path,
            "prompt": "sharper",
            "conditioning_type": "custom",
            "max_window_frames": 41,
        },
    )

    # A 41-frame chunk is not more than twice the carry of 25, so Auto carry runs 17 on the pipeline.
    assert response.status_code == 200, response.text
    pipeline = fake_services.ic_lora_pipeline
    assert pipeline.last_tiling is not None
    assert pipeline.last_tiling.carry_frames == 17
    assert pipeline.generate_calls[-1]["chunk_pixel_frames"] == 41
