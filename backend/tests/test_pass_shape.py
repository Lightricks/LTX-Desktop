"""One pass or parts: the per-job shape, the "Never split into parts" switch, and the Brave alternative."""

from __future__ import annotations

import threading
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import JsonValue
from starlette.testclient import TestClient

from app_factory import create_app
from frame_math import compute_num_frames
from remote.app import create_remote_app
from runtime_config.fast_chunking import job_window_count
from runtime_config.video_job_budget import (
    MPS_SEQUENCE_OVER_MESSAGE,
    ONE_PASS_UNAVAILABLE,
    ONE_PASS_UNAVAILABLE_MESSAGE,
    VIDEO_JOB_TOO_LARGE,
)
from services.features.video import TextToVideoExecutor
from services.generation_queue.types import OutputAllocation
from services.sqlite_store import SqliteStore
from tests.conftest import TEST_ADMIN_TOKEN
from tests.fakes.generation import FakeJobSession
from tests.http_error_assertions import assert_http_error

_GIB = 1024**3
_RTX_5090_GIB = 31.84


def _configure_windows(test_state, create_fake_model_files, *, vram_gib: float, mode: str) -> None:
    create_fake_model_files()
    test_state.config.system = "Windows"
    test_state.config.vram_gb = int(vram_gib)
    test_state.config.vram_bytes = int(vram_gib * _GIB)
    test_state.config.total_ram_bytes = 64 * _GIB
    test_state.config.local_generations_mode = "streaming_models_loading"
    test_state.state.app_settings.use_local_text_encoder = True
    test_state.state.app_settings.budget_mode = mode


def _configure_mac(test_state, fake_services, create_fake_model_files, *, pool_gib: float, mode: str) -> None:
    create_fake_model_files()
    test_state.config.system = "Darwin"
    test_state.config.total_ram_bytes = 48 * _GIB
    test_state.config.local_generations_mode = "streaming_models_loading"
    fake_services.gpu_info.mps_pool_bytes = int(pool_gib * _GIB)
    test_state.state.app_settings.budget_mode = mode


def _t2v(
    resolution: str, duration: int | None, *, fps: int = 24, pass_shape: str | None = None
) -> dict[str, JsonValue]:
    params: dict[str, JsonValue] = {
        "prompt": "fox",
        "model": "ltx-2.5-fast",
        "aspectRatio": "16:9",
        "resolution": resolution,
        "duration": duration,
        "fps": fps,
    }
    if pass_shape is not None:
        params["passShape"] = pass_shape
    return {"params": params}


@contextmanager
def _home(test_state) -> Iterator[TestClient]:
    with TestClient(create_app(handler=test_state, admin_token=TEST_ADMIN_TOKEN)) as client:
        yield client


def _create(test_state, body: dict[str, JsonValue]):
    with _home(test_state) as client:
        return client.post("/api/generations/text-to-video", json=body)


def _a2v_create(
    test_state, tmp_path: Path, resolution: str, *, seconds: float, pass_shape: str
):
    wav_path = tmp_path / "line.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * int(seconds * 8000))
    with _home(test_state) as client:
        audio = client.post("/api/assets", json={"path": str(wav_path)}).json()
        return client.post(
            "/api/generations/audio-to-video",
            json={
                "params": {
                    "prompt": "the character speaks",
                    "resolution": resolution,
                    "model": "ltx-2.5-fast",
                    "fps": 24,
                    "passShape": pass_shape,
                },
                "inputs": {"audio": {"assetId": audio["id"]}},
            },
        )


def _stored_spec(test_state, generation_id: str) -> dict[str, object]:
    record = SqliteStore(test_state.config.app_data_dir).get_generation(generation_id)
    assert record is not None
    return dict(record.spec)


def _run(test_state, fake_services, tmp_path: Path, generation_id: str) -> int | None:
    """Execute the queued job and return the window the pipeline got (None is one pass)."""
    record = SqliteStore(test_state.config.app_data_dir).get_generation(generation_id)
    assert record is not None
    executor = TextToVideoExecutor(test_state.video_generation)
    plans = executor.plan_outputs(record)
    outputs = (
        OutputAllocation(plan=plans[0], asset_id=f"{generation_id}-out", dest_path=str(tmp_path / f"{generation_id}.mp4")),
    )
    with test_state.generation.wait_for_generation_slot(threading.Event()):
        executor.execute(record, outputs, FakeJobSession())
    window = fake_services.fast_video_pipeline.generate_calls[-1]["chunk_pixel_frames"]
    assert window is None or isinstance(window, int)
    return window


def _fast_spec(test_state) -> dict:
    with _home(test_state) as client:
        response = client.get("/api/generate/models-specs")
    assert response.status_code == 200, response.text
    return next(item["spec"] for item in response.json()["local_models"] if item["pipeline"] == "fast")


def _cell(cells: list[dict], resolution: str, duration: int, fps: int = 24) -> dict | None:
    return next(
        (
            c
            for c in cells
            if (c["resolution"], c["duration"], c["fps"], c["audio"]) == (resolution, duration, fps, False)
        ),
        None,
    )


def _kept(spec: dict, resolution: str, fps: int = 24) -> list[int]:
    cell = spec["supported_resolutions_durations"].get(resolution)
    return [] if cell is None else cell["fps_to_durations"].get(str(fps), [])


def test_a_forced_one_pass_over_the_card_runs_as_an_over_bet_in_brave(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # A 16 GB card runs 1080p 10 s in parts at 16 GB; one pass asks 22.7.
    _configure_windows(test_state, create_fake_model_files, vram_gib=16, mode="brave")
    response = _create(test_state, _t2v("1080p", 10, pass_shape="one_pass"))
    assert response.status_code == 200, response.text
    assert response.json()["risk"]["tier"] == "over"
    assert _run(test_state, fake_services, tmp_path, response.json()["id"]) is None


def test_a_forced_one_pass_over_the_card_is_refused_in_cautious(
    test_state, create_fake_model_files
) -> None:
    _configure_windows(test_state, create_fake_model_files, vram_gib=16, mode="cautious")
    assert _create(test_state, _t2v("1080p", 10)).status_code == 200
    response = _create(test_state, _t2v("1080p", 10, pass_shape="one_pass"))
    assert_http_error(
        response,
        status_code=422,
        code=VIDEO_JOB_TOO_LARGE,
        message=(
            "This job needs 22.7 GB of GPU memory but this card has 16.0 GB. "
            "Lower the resolution, duration, frame rate, or aspect width."
        ),
    )
    assert len(test_state.queued_generations.list_generations("text-to-video")) == 1


@pytest.mark.parametrize("mode", ["brave", "cautious"])
def test_a_forced_one_pass_past_the_mac_fence_is_refused_in_every_mode(
    test_state, fake_services, create_fake_model_files, mode: str
) -> None:
    # 1080p 20 s is past the MPS sequence fence in one pass; in parts it runs on this pool.
    _configure_mac(test_state, fake_services, create_fake_model_files, pool_gib=32, mode=mode)
    assert _create(test_state, _t2v("1080p", 20)).status_code == 200
    response = _create(test_state, _t2v("1080p", 20, pass_shape="one_pass"))
    assert_http_error(
        response, status_code=422, code=VIDEO_JOB_TOO_LARGE, message=MPS_SEQUENCE_OVER_MESSAGE
    )


def test_a_forced_one_pass_at_1440p_longer_than_one_part_is_refused(
    test_state, create_fake_model_files
) -> None:
    _configure_windows(test_state, create_fake_model_files, vram_gib=_RTX_5090_GIB, mode="brave")
    response = _create(test_state, _t2v("1440p", 10, pass_shape="one_pass"))
    assert_http_error(
        response, status_code=422, code=ONE_PASS_UNAVAILABLE, message=ONE_PASS_UNAVAILABLE_MESSAGE
    )
    assert _create(test_state, _t2v("1440p", 2, pass_shape="one_pass")).status_code == 200


def test_forced_parts_run_even_where_brave_would_run_one_pass(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    # 540p 20 s at 48 fps is past the fitted tokens in one pass: Brave's own pick is one pass.
    _configure_windows(test_state, create_fake_model_files, vram_gib=_RTX_5090_GIB, mode="brave")
    auto = _create(test_state, _t2v("540p", 20, fps=48))
    parts = _create(test_state, _t2v("540p", 20, fps=48, pass_shape="parts"))
    assert auto.status_code == 200 and parts.status_code == 200, (auto.text, parts.text)
    assert _run(test_state, fake_services, tmp_path, auto.json()["id"]) is None
    assert _run(test_state, fake_services, tmp_path, parts.json()["id"]) is not None


def test_the_stored_shape_and_not_the_live_setting_decides_the_run(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    _configure_windows(test_state, create_fake_model_files, vram_gib=16, mode="brave")
    test_state.state.app_settings.never_split_into_parts = True
    never_split = _create(test_state, _t2v("1080p", 10))
    test_state.state.app_settings.never_split_into_parts = False
    automatic = _create(test_state, _t2v("1080p", 10))
    assert never_split.status_code == 200 and automatic.status_code == 200
    test_state.state.app_settings.never_split_into_parts = True
    assert _run(test_state, fake_services, tmp_path, automatic.json()["id"]) is not None
    test_state.state.app_settings.never_split_into_parts = False
    assert _run(test_state, fake_services, tmp_path, never_split.json()["id"]) is None
    # An automatic length is sized for the stored shape too: in Cautious one pass
    # stops at 6 s, and the live parts range (20 s) would be refused in one pass.
    test_state.state.app_settings.budget_mode = "cautious"
    test_state.state.app_settings.never_split_into_parts = True
    auto_length = _create(test_state, _t2v("1080p", None))
    assert auto_length.status_code == 200, auto_length.text
    test_state.state.app_settings.never_split_into_parts = False
    assert _run(test_state, fake_services, tmp_path, auto_length.json()["id"]) is None
    assert fake_services.fast_video_pipeline.generate_calls[-1]["num_frames"].max_seconds == 6


def test_never_split_withholds_cells_that_only_fit_in_parts_in_cautious(
    test_state, create_fake_model_files
) -> None:
    _configure_windows(test_state, create_fake_model_files, vram_gib=16, mode="cautious")
    assert 10 in _kept(_fast_spec(test_state), "1080p")
    test_state.state.app_settings.never_split_into_parts = True
    spec = _fast_spec(test_state)
    assert 10 not in _kept(spec, "1080p")
    dropped = _cell(spec["dropped_cells"], "1080p", 10)
    assert dropped is not None and dropped["reason"] == "stream_over"


def test_never_split_offers_cells_that_only_fit_in_parts_as_over_in_brave(
    test_state, create_fake_model_files
) -> None:
    _configure_windows(test_state, create_fake_model_files, vram_gib=16, mode="brave")
    test_state.state.app_settings.never_split_into_parts = True
    spec = _fast_spec(test_state)
    assert 10 in _kept(spec, "1080p")
    warned = _cell(spec["warned_cells"], "1080p", 10)
    assert warned is not None and warned["tier"] == "over"
    ask = _cell(spec["cell_asks"], "1080p", 10)
    assert ask is not None and ask["windows"] == 1


@pytest.mark.parametrize("mode", ["brave", "cautious"])
def test_never_split_withholds_1440p_cells_longer_than_one_part(
    test_state, create_fake_model_files, mode: str
) -> None:
    _configure_windows(test_state, create_fake_model_files, vram_gib=_RTX_5090_GIB, mode=mode)
    test_state.state.app_settings.never_split_into_parts = True
    spec = _fast_spec(test_state)
    assert 2 in _kept(spec, "1440p") and 10 not in _kept(spec, "1440p")
    dropped = _cell(spec["dropped_cells"], "1440p", 10)
    assert dropped is not None
    assert (dropped["reason"], dropped["message"]) == ("one_pass_unavailable", ONE_PASS_UNAVAILABLE_MESSAGE)


def test_an_explicit_parts_request_wins_over_the_switch(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    _configure_windows(test_state, create_fake_model_files, vram_gib=16, mode="brave")
    test_state.state.app_settings.never_split_into_parts = True
    response = _create(test_state, _t2v("1080p", 10, pass_shape="parts"))
    assert response.status_code == 200, response.text
    assert response.json()["risk"] is None
    assert _run(test_state, fake_services, tmp_path, response.json()["id"]) is not None
    # An automatic length is not cut to the one-pass maximum.
    auto_length = _create(test_state, _t2v("1080p", None, pass_shape="parts"))
    assert auto_length.status_code == 200, auto_length.text
    assert _run(test_state, fake_services, tmp_path, auto_length.json()["id"]) is not None
    assert fake_services.fast_video_pipeline.generate_calls[-1]["num_frames"].max_seconds == 20
    # Every 4K audio-to-video length is longer than one part, so the switch alone
    # drops the whole row.
    _configure_windows(test_state, create_fake_model_files, vram_gib=_RTX_5090_GIB, mode="brave")
    a2v = _a2v_create(test_state, tmp_path, "2160p", seconds=5.0, pass_shape="parts")
    assert a2v.status_code == 200, a2v.text
    assert _stored_spec(test_state, a2v.json()["id"])["_passShape"] == "parts"


def test_the_alternative_ask_is_sent_only_in_brave_where_the_toggle_shows(
    test_state, fake_services, create_fake_model_files
) -> None:
    # On a 16 GB card 1080p 10 s runs in parts (one pass is over), and 2 s fits in one pass.
    _configure_windows(test_state, create_fake_model_files, vram_gib=16, mode="brave")
    asks = _fast_spec(test_state)["cell_asks"]
    parts = _cell(asks, "1080p", 10)
    one_pass = _cell(asks, "1080p", 2)
    assert parts is not None and parts["alternative"]["windows"] == 1
    assert one_pass is not None and "alternative" not in one_pass
    test_state.state.app_settings.budget_mode = "cautious"
    assert all("alternative" not in ask for ask in _fast_spec(test_state)["cell_asks"])
    # On a 5090, 540p 20 s at 48 fps runs one pass past the fitted tokens, so parts are
    # the alternative, until the switch makes one pass the only shape.
    _configure_windows(test_state, create_fake_model_files, vram_gib=_RTX_5090_GIB, mode="brave")
    asks = _fast_spec(test_state)["cell_asks"]
    past_fitted = _cell(asks, "540p", 20, fps=48)
    assert past_fitted is not None and past_fitted["alternative"]["windows"] > 1
    # 1440p 10 s runs in parts; one pass there is never offered.
    high_res = _cell(asks, "1440p", 10)
    assert high_res is not None and high_res["windows"] > 1 and "alternative" not in high_res
    test_state.state.app_settings.never_split_into_parts = True
    assert all("alternative" not in ask for ask in _fast_spec(test_state)["cell_asks"])
    test_state.state.app_settings.never_split_into_parts = False
    # 1080p 20 s on a Mac runs in parts; one pass is past the sequence fence.
    _configure_mac(test_state, fake_services, create_fake_model_files, pool_gib=32, mode="brave")
    fenced = _cell(_fast_spec(test_state)["cell_asks"], "1080p", 20)
    assert fenced is not None and fenced["windows"] > 1 and "alternative" not in fenced


@pytest.mark.parametrize(
    ("vram_gib", "resolution", "duration", "fps", "shape"),
    [
        (16, "1080p", 10, 24, "one_pass"),  # parts by default, one pass over the card
        (_RTX_5090_GIB, "540p", 20, 48, "parts"),  # one pass past the fitted tokens by default
    ],
)
def test_the_alternative_ask_is_what_create_runs_for_that_shape(
    test_state,
    fake_services,
    create_fake_model_files,
    tmp_path: Path,
    vram_gib: float,
    resolution: str,
    duration: int,
    fps: int,
    shape: str,
) -> None:
    _configure_windows(test_state, create_fake_model_files, vram_gib=vram_gib, mode="brave")
    ask = _cell(_fast_spec(test_state)["cell_asks"], resolution, duration, fps)
    assert ask is not None
    alternative = ask["alternative"]
    response = _create(test_state, _t2v(resolution, duration, fps=fps, pass_shape=shape))
    assert response.status_code == 200, response.text
    if shape == "one_pass":
        assert response.json()["risk"]["readout"]["need_gb"] == pytest.approx(alternative["ask_gib"], abs=0.1)
    else:
        # A clean parts job carries no readout; Cautious's own pick for this cell is
        # parts, and its ask is the one Cautious create runs.
        assert response.json()["risk"] is None
        test_state.state.app_settings.budget_mode = "cautious"
        cautious = _cell(_fast_spec(test_state)["cell_asks"], resolution, duration, fps)
        assert cautious is not None
        assert (cautious["windows"], cautious["ask_gib"]) == (alternative["windows"], alternative["ask_gib"])
    window = _run(test_state, fake_services, tmp_path, response.json()["id"])
    assert job_window_count(compute_num_frames(duration, fps), window) == alternative["windows"]


def test_remote_create_keeps_the_pass_shape(test_state, create_fake_model_files) -> None:
    _configure_windows(test_state, create_fake_model_files, vram_gib=16, mode="brave")
    app = create_remote_app(handler=test_state, remote_token="pair-token")
    with TestClient(app) as client:
        response = client.post(
            "/api/generations/text-to-video",
            headers={"Authorization": "Bearer pair-token"},
            json=_t2v("1080p", 10, pass_shape="one_pass"),
        )
    assert response.status_code == 200, response.text
    assert _stored_spec(test_state, response.json()["id"])["_passShape"] == "one_pass"
