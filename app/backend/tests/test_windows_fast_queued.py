"""Queued Home/Remote Fast jobs must apply Windows LTX 2.5 policy before insert."""

from __future__ import annotations

import dataclasses
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from PIL import Image
from pydantic import JsonValue
from starlette.testclient import TestClient

from app_factory import create_app
from handlers import queued_generation_handler as queued_handler_mod
from handlers import video_generation_handler as handler_mod
from remote.app import create_remote_app
from runtime_config.model_download_specs import resolve_lora_path
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_RAM_SHORTFALL,
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    LocalJobFilter,
    VIDEO_JOB_TOO_LARGE,
)
from services.sqlite_store import SqliteStore
from tests.conftest import TEST_ADMIN_TOKEN
from tests.http_error_assertions import assert_http_error

_GIB = 1024**3
_REMOTE_AUTH = {"Authorization": "Bearer pair-token"}


def _store(test_state) -> SqliteStore:
    return SqliteStore(test_state.config.app_data_dir)


def _configure_windows_fast(
    test_state,
    *,
    vram_gib: float,
    total_ram_gib: float = 48,
    vram_bytes: int | None = None,
) -> None:
    test_state.config.system = "Windows"
    test_state.config.vram_gb = int(vram_gib)
    test_state.config.vram_bytes = (
        int(vram_gib * _GIB) if vram_bytes is None else vram_bytes
    )
    test_state.config.total_ram_bytes = int(total_ram_gib * _GIB)
    test_state.config.local_generations_mode = "streaming_models_loading"


def _png(path: Path, *, size: tuple[int, int] = (16, 16)) -> Path:
    Image.new("RGB", size, color=(1, 2, 3)).save(path)
    return path


def _wav(path: Path, *, duration_seconds: float = 8.0) -> Path:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * int(duration_seconds * 8000))
    return path


def _t2v_body(
    *,
    resolution: str = "540p",
    duration: int | None = 5,
    aspect_ratio: str = "16:9",
    fps: int = 24,
) -> dict[str, JsonValue]:
    return {
        "params": {
            "prompt": "fox",
            "model": "ltx-2.5-fast",
            "aspectRatio": aspect_ratio,
            "resolution": resolution,
            "duration": duration,
            "fps": fps,
        }
    }


def _i2v_body(
    start_id: str,
    *,
    resolution: str = "540p",
    duration: int = 5,
    aspect_ratio: str = "16:9",
) -> dict[str, JsonValue]:
    return {
        "params": {
            "prompt": "fox",
            "model": "ltx-2.5-fast",
            "aspectRatio": aspect_ratio,
            "resolution": resolution,
            "duration": duration,
            "fps": 24,
        },
        "inputs": {"startFrame": {"assetId": start_id}},
    }


def _a2v_body(
    audio_id: str,
    *,
    resolution: str = "540p",
    aspect_ratio: str = "16:9",
    start_id: str | None = None,
) -> dict[str, JsonValue]:
    inputs: dict[str, JsonValue] = {"audio": {"assetId": audio_id}}
    if start_id is not None:
        inputs["startFrame"] = {"assetId": start_id}
    return {
        "params": {
            "prompt": "the character speaks",
            "model": "ltx-2.5-fast",
            "resolution": resolution,
            "aspectRatio": aspect_ratio,
            "fps": 24,
        },
        "inputs": inputs,
    }


def _install_recipe_weights(
    test_state, *, catalog_id: str, filename: str
) -> None:
    path = resolve_lora_path(
        test_state.config.default_models_dir, catalog_id, filename
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 64)


def _t2v_recipe_body(
    *, resolution: str = "720p", duration: int = 20, aspect_ratio: str = "21:9"
) -> dict[str, JsonValue]:
    return {
        "params": {
            "prompt": "a fox on a skateboard",
            "model": "ltx-2.5-fast",
            "catalogId": "cozy-felt-style",
            "scale": 1.0,
            "resolution": resolution,
            "duration": duration,
            "fps": 24,
            "aspectRatio": aspect_ratio,
        }
    }


def _i2v_recipe_body(
    start_id: str,
    *,
    resolution: str = "720p",
    duration: int = 20,
    aspect_ratio: str = "auto",
) -> dict[str, JsonValue]:
    return {
        "params": {
            "prompt": "push in on her face",
            "model": "ltx-2.5-fast",
            "catalogId": "dolly-in",
            "scale": 1.0,
            "resolution": resolution,
            "duration": duration,
            "fps": 24,
            "aspectRatio": aspect_ratio,
        },
        "inputs": {"startFrame": {"assetId": start_id}},
    }


@contextmanager
def _surface_client(test_state, surface: str) -> Iterator[tuple[TestClient, dict[str, str]]]:
    if surface == "home":
        app = create_app(handler=test_state, admin_token=TEST_ADMIN_TOKEN)
        headers: dict[str, str] = {}
    else:
        app = create_remote_app(handler=test_state, remote_token="pair-token")
        headers = _REMOTE_AUTH
    with TestClient(app) as client:
        yield client, headers


def _create_start_frame(
    client: TestClient,
    headers: dict[str, str],
    tmp_path: Path,
    *,
    size: tuple[int, int] = (16, 16),
) -> str:
    name = f"start-{size[0]}x{size[1]}.png"
    if headers:
        uploaded = client.post(
            "/api/assets/upload",
            headers=headers,
            files={"file": (name, _png(tmp_path / name, size=size).read_bytes(), "image/png")},
        )
        assert uploaded.status_code == 200, uploaded.text
        return uploaded.json()["id"]
    created = client.post("/api/assets", json={"path": str(_png(tmp_path / name, size=size))})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _create_audio(
    client: TestClient,
    headers: dict[str, str],
    tmp_path: Path,
    test_state,
    *,
    duration_seconds: float = 8.0,
) -> str:
    name = f"line-{duration_seconds:g}s.wav"
    if headers:
        audio = _store(test_state).ingest_upload(
            str(_wav(tmp_path / name, duration_seconds=duration_seconds))
        )
        return audio.id
    created = client.post(
        "/api/assets",
        json={"path": str(_wav(tmp_path / name, duration_seconds=duration_seconds))},
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _post_queued_job(
    client: TestClient,
    headers: dict[str, str],
    modality: str,
    tmp_path: Path,
    test_state,
    *,
    resolution: str = "540p",
    duration: int = 5,
):
    if modality == "t2v":
        return client.post(
            "/api/generations/text-to-video",
            headers=headers,
            json=_t2v_body(resolution=resolution, duration=duration),
        )
    if modality == "i2v":
        start_id = _create_start_frame(client, headers, tmp_path)
        return client.post(
            "/api/generations/image-to-video",
            headers=headers,
            json=_i2v_body(start_id, resolution=resolution, duration=duration),
        )
    audio_id = _create_audio(client, headers, tmp_path, test_state)
    return client.post(
        "/api/generations/audio-to-video",
        headers=headers,
        json=_a2v_body(audio_id),
    )


def _feature_for(modality: str) -> str:
    return {
        "t2v": "text-to-video",
        "i2v": "image-to-video",
        "a2v": "audio-to-video",
    }[modality]


def test_local_job_filter_is_named_immutable_value(test_state) -> None:
    test_state.config.system = "Linux"
    job_filter = test_state.video_generation._local_job_filter(
        model_id="ltx-2.5-22b-distilled"
    )

    assert isinstance(job_filter, LocalJobFilter)
    assert not isinstance(job_filter, tuple)
    assert dataclasses.is_dataclass(job_filter)
    assert dataclasses.fields(job_filter)[0].name != "0"
    assert job_filter.profile.model_id == "ltx-2.5-22b-distilled"
    assert job_filter.profile.process_mode == test_state.config.local_generations_mode
    assert job_filter.profile.darwin == test_state.config.darwin_unified_memory
    assert job_filter.profile.system == test_state.config.system
    assert job_filter.budget.total_vram_bytes == test_state.config.vram_bytes
    assert job_filter.budget.total_ram_bytes == test_state.config.total_ram_bytes
    with pytest.raises(dataclasses.FrozenInstanceError):
        job_filter.profile = job_filter.profile  # type: ignore[misc]


def test_model_specs_pass_named_local_job_filter(test_state, monkeypatch) -> None:
    seen: list[object] = []

    real = handler_mod.build_generate_video_model_specs_response

    def wrapped(*args, **kwargs):
        seen.append(kwargs.get("job_filter", args[0] if args else None))
        return real(*args, **kwargs)

    monkeypatch.setattr(handler_mod, "build_generate_video_model_specs_response", wrapped)
    test_state.video_generation.get_model_specs()

    assert seen
    assert isinstance(seen[0], LocalJobFilter)


@pytest.mark.parametrize("surface", ["home", "remote"])
@pytest.mark.parametrize("modality", ["t2v", "i2v", "a2v"])
def test_windows_low_ram_rejects_queued_fast_before_insert(
    test_state,
    create_fake_model_files,
    tmp_path: Path,
    surface: str,
    modality: str,
) -> None:
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16, total_ram_gib=32)

    with _surface_client(test_state, surface) as (client, headers):
        response = _post_queued_job(client, headers, modality, tmp_path, test_state)

    assert_http_error(
        response,
        status_code=422,
        code=LOCAL_GENERATION_RAM_SHORTFALL,
        message=(
            "This job needs 40 GB of system RAM but this machine has 32 GB. "
            "Close heavy programs; on desktops, more RAM is the real fix."
        ),
    )
    assert response.json()["code"] != "INVALID_VIDEO_GENERATION_SPEC"
    assert response.json()["code"] != VIDEO_JOB_TOO_LARGE
    assert response.json()["code"] != LOCAL_GENERATION_UNSUPPORTED
    assert _store(test_state).list_generations(_feature_for(modality)) == []


@pytest.mark.parametrize("surface", ["home", "remote"])
@pytest.mark.parametrize(
    ("modality", "resolution", "duration", "vram_gib"),
    [
        ("t2v", "720p", 20, 15.5),
        ("i2v", "1080p", 10, 18.0),
    ],
)
@pytest.mark.usefixtures("single_pass_only")
def test_windows_over_envelope_rejects_queued_fast_before_insert(
    test_state,
    create_fake_model_files,
    tmp_path: Path,
    surface: str,
    modality: str,
    resolution: str,
    duration: int,
    vram_gib: float,
) -> None:
    create_fake_model_files()
    _configure_windows_fast(
        test_state,
        vram_gib=vram_gib,
        total_ram_gib=48,
        vram_bytes=int(vram_gib * _GIB) - 1,
    )

    with _surface_client(test_state, surface) as (client, headers):
        response = _post_queued_job(
            client,
            headers,
            modality,
            tmp_path,
            test_state,
            resolution=resolution,
            duration=duration,
        )

    expected_message = {
        15.5: (
            "This job needs 20.0 GB of GPU memory but this card has 15.5 GB. "
            "Lower the resolution, duration, frame rate, or aspect width."
        ),
        18.0: (
            "This job needs 22.7 GB of GPU memory but this card has 18.0 GB. "
            "Lower the resolution, duration, frame rate, or aspect width."
        ),
    }[vram_gib]
    assert_http_error(
        response,
        status_code=422,
        code=VIDEO_JOB_TOO_LARGE,
        message=expected_message,
    )
    assert response.json()["code"] != "INVALID_VIDEO_GENERATION_SPEC"
    assert response.json()["code"] != LOCAL_GENERATION_UNSUPPORTED
    assert _store(test_state).list_generations(_feature_for(modality)) == []


def test_windows_cell_over_free_vram_enqueues_queued_fast_for_execute_recheck(
    test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=24, total_ram_gib=40)
    fake_services.gpu_info.free_vram_bytes = int(19.0 * _GIB)

    with _surface_client(test_state, "home") as (client, headers):
        response = _post_queued_job(
            client, headers, "t2v", tmp_path, test_state, resolution="720p", duration=20
        )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "queued"
    assert len(_store(test_state).list_generations("text-to-video")) == 1


@pytest.mark.parametrize("surface", ["home", "remote"])
def test_windows_a2v_ram_shortage_is_not_invalid_spec(
    test_state, create_fake_model_files, tmp_path: Path, surface: str
) -> None:
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16, total_ram_gib=32)

    with _surface_client(test_state, surface) as (client, headers):
        response = _post_queued_job(client, headers, "a2v", tmp_path, test_state)

    payload = assert_http_error(
        response,
        status_code=422,
        code=LOCAL_GENERATION_RAM_SHORTFALL,
        message=(
            "This job needs 40 GB of system RAM but this machine has 32 GB. "
            "Close heavy programs; on desktops, more RAM is the real fix."
        ),
    )
    assert payload["code"] != "INVALID_VIDEO_GENERATION_SPEC"
    assert payload["code"] != LOCAL_GENERATION_UNSUPPORTED
    assert "not available at the selected resolution" not in payload["message"]
    assert _store(test_state).list_generations("audio-to-video") == []


def test_linux_queued_t2v_still_enqueues_stream_over_cell(
    test_state, create_fake_model_files
) -> None:
    create_fake_model_files()
    test_state.config.system = "Linux"
    test_state.config.vram_gb = 16
    test_state.config.vram_bytes = 16 * _GIB
    test_state.config.total_ram_bytes = 32 * _GIB
    test_state.config.local_generations_mode = "streaming_models_loading"
    test_state.state.app_settings.budget_mode = "brave"

    with _surface_client(test_state, "home") as (client, headers):
        response = client.post(
            "/api/generations/text-to-video",
            headers=headers,
            json=_t2v_body(resolution="720p", duration=20),
        )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "queued"
    assert response.json()["risk"] is not None
    assert len(_store(test_state).list_generations("text-to-video")) == 1


@pytest.mark.parametrize("surface", ["home", "remote"])
@pytest.mark.usefixtures("single_pass_only")
def test_windows_queued_21_9_refuses_before_insert_where_16_9_enqueues(
    test_state, create_fake_model_files, surface: str
) -> None:
    # The create preflight mapped every shape to 16:9, so a 1080p/5s 21:9 job
    # (17.2 GB on a 16 GB card) enqueued and died at execute instead of 422ing.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16, total_ram_gib=48)

    with _surface_client(test_state, surface) as (client, headers):
        refused = client.post(
            "/api/generations/text-to-video",
            headers=headers,
            json=_t2v_body(resolution="1080p", duration=5, aspect_ratio="21:9"),
        )
        assert_http_error(
            refused,
            status_code=422,
            code=VIDEO_JOB_TOO_LARGE,
            message=(
                "This job needs 17.2 GB of GPU memory but this card has 16.0 GB. "
                "Lower the resolution, duration, frame rate, or aspect width."
            ),
        )
        admitted = client.post(
            "/api/generations/text-to-video",
            headers=headers,
            json=_t2v_body(resolution="1080p", duration=5, aspect_ratio="16:9"),
        )
        assert admitted.status_code == 200, admitted.text

    assert len(_store(test_state).list_generations("text-to-video")) == 1


@pytest.mark.parametrize("surface", ["home", "remote"])
def test_windows_queued_270p_budgets_denoised_pixels_not_parent_canvas(
    test_state, create_fake_model_files, surface: str
) -> None:
    # Queue preflight budgeted the 270p parent canvas (4x tokens), so an
    # offered 270p/20s cell 422d at queue-create on a 10 GB card.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=10, total_ram_gib=48)

    with _surface_client(test_state, surface) as (client, headers):
        response = client.post(
            "/api/generations/text-to-video",
            headers=headers,
            json=_t2v_body(resolution="270p", duration=20),
        )

    assert response.status_code == 200, response.text
    assert len(_store(test_state).list_generations("text-to-video")) == 1


@pytest.mark.parametrize("surface", ["home", "remote"])
def test_windows_queued_auto_fully_dropped_cell_names_figures_not_unsupported(
    test_state, create_fake_model_files, surface: str
) -> None:
    # Queued auto mapped a fully budget-dropped resolution/fps to flat
    # UNSUPPORTED ("cannot run local generation") instead of the enriched
    # stream_over figures the gate compared.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=8, total_ram_gib=48)

    with _surface_client(test_state, surface) as (client, headers):
        response = client.post(
            "/api/generations/text-to-video",
            headers=headers,
            json=_t2v_body(resolution="1080p", duration=None, fps=50),
        )

    assert_http_error(
        response,
        status_code=422,
        code=VIDEO_JOB_TOO_LARGE,
        message=(
            "This job needs 11.2 GB of GPU memory but this card has 8.0 GB. "
            "Lower the resolution, duration, frame rate, or aspect width."
        ),
    )
    assert _store(test_state).list_generations("text-to-video") == []


@pytest.mark.parametrize("surface", ["home", "remote"])
@pytest.mark.usefixtures("single_pass_only")
def test_windows_queued_i2v_auto_resolves_start_frame_before_insert(
    test_state, create_fake_model_files, tmp_path: Path, surface: str
) -> None:
    # Auto preflighted at the anchor though the start frame fixes the shape:
    # an ultrawide frame resolves to 21:9 and must refuse like an explicit one.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16, total_ram_gib=48)

    with _surface_client(test_state, surface) as (client, headers):
        wide_id = _create_start_frame(client, headers, tmp_path, size=(64, 16))
        refused = client.post(
            "/api/generations/image-to-video",
            headers=headers,
            json=_i2v_body(
                wide_id, resolution="1080p", duration=5, aspect_ratio="auto"
            ),
        )
        assert_http_error(
            refused,
            status_code=422,
            code=VIDEO_JOB_TOO_LARGE,
            message=(
                "This job needs 17.2 GB of GPU memory but this card has 16.0 GB. "
                "Lower the resolution, duration, frame rate, or aspect width."
            ),
        )
        square_id = _create_start_frame(client, headers, tmp_path, size=(16, 16))
        admitted = client.post(
            "/api/generations/image-to-video",
            headers=headers,
            json=_i2v_body(
                square_id, resolution="1080p", duration=5, aspect_ratio="auto"
            ),
        )
        assert admitted.status_code == 200, admitted.text

    assert len(_store(test_state).list_generations("image-to-video")) == 1


@pytest.mark.parametrize("surface", ["home", "remote"])
@pytest.mark.usefixtures("single_pass_only")
def test_windows_queued_recipe_21_9_refuses_before_insert_where_16_9_enqueues(
    test_state, create_fake_model_files, surface: str, monkeypatch
) -> None:
    # create_lora_recipe never called the Fast preflight, so a 1080p/5s 21:9 recipe
    # job (17.2 GB on a 16 GB card) enqueued and died at execute instead of 422ing.
    create_fake_model_files()
    monkeypatch.setattr(queued_handler_mod, "accelerator_backend", lambda: "cuda")
    _configure_windows_fast(test_state, vram_gib=16, total_ram_gib=48)
    _install_recipe_weights(
        test_state, catalog_id="cozy-felt-style", filename="CozyFelt.safetensors"
    )

    with _surface_client(test_state, surface) as (client, headers):
        refused = client.post(
            "/api/generations/recipes/cozy-felt",
            headers=headers,
            json=_t2v_recipe_body(resolution="1080p", duration=5, aspect_ratio="21:9"),
        )
        assert_http_error(
            refused,
            status_code=422,
            code=VIDEO_JOB_TOO_LARGE,
            message=(
                "This job needs 17.2 GB of GPU memory but this card has 16.0 GB. "
                "Lower the resolution, duration, frame rate, or aspect width."
            ),
        )
        admitted = client.post(
            "/api/generations/recipes/cozy-felt",
            headers=headers,
            json=_t2v_recipe_body(resolution="1080p", duration=5, aspect_ratio="16:9"),
        )
        assert admitted.status_code == 200, admitted.text

    assert len(_store(test_state).list_generations("cozy-felt")) == 1


@pytest.mark.parametrize("surface", ["home", "remote"])
@pytest.mark.usefixtures("single_pass_only")
def test_windows_queued_recipe_i2v_auto_resolves_start_frame_before_insert(
    test_state, create_fake_model_files, tmp_path: Path, surface: str, monkeypatch
) -> None:
    # Recipe auto preflighted at the anchor though the start frame fixes the shape:
    # an ultrawide frame resolves to 21:9 and must refuse like an explicit one.
    create_fake_model_files()
    monkeypatch.setattr(queued_handler_mod, "accelerator_backend", lambda: "cuda")
    _configure_windows_fast(test_state, vram_gib=16, total_ram_gib=48)
    _install_recipe_weights(
        test_state,
        catalog_id="dolly-in",
        filename="ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    )

    with _surface_client(test_state, surface) as (client, headers):
        wide_id = _create_start_frame(client, headers, tmp_path, size=(64, 16))
        refused = client.post(
            "/api/generations/recipes/dolly-in",
            headers=headers,
            json=_i2v_recipe_body(
                wide_id, resolution="1080p", duration=5, aspect_ratio="auto"
            ),
        )
        assert_http_error(
            refused,
            status_code=422,
            code=VIDEO_JOB_TOO_LARGE,
            message=(
                "This job needs 17.2 GB of GPU memory but this card has 16.0 GB. "
                "Lower the resolution, duration, frame rate, or aspect width."
            ),
        )
        square_id = _create_start_frame(client, headers, tmp_path, size=(16, 16))
        admitted = client.post(
            "/api/generations/recipes/dolly-in",
            headers=headers,
            json=_i2v_recipe_body(
                square_id, resolution="1080p", duration=5, aspect_ratio="auto"
            ),
        )
        assert admitted.status_code == 200, admitted.text

    assert len(_store(test_state).list_generations("dolly-in")) == 1


@pytest.mark.parametrize("surface", ["home", "remote"])
@pytest.mark.usefixtures("single_pass_only")
def test_windows_queued_a2v_21_9_refuses_before_insert_where_16_9_enqueues(
    test_state, create_fake_model_files, tmp_path: Path, surface: str
) -> None:
    # A2V created with no VRAM budget, so a 1080p/5s-audio 21:9 job (16.4 GB on
    # a 16 GB card) enqueued and died at execute instead of 422ing.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16, total_ram_gib=48)

    with _surface_client(test_state, surface) as (client, headers):
        refused_id = _create_audio(
            client, headers, tmp_path, test_state, duration_seconds=5.0
        )
        refused = client.post(
            "/api/generations/audio-to-video",
            headers=headers,
            json=_a2v_body(refused_id, resolution="1080p", aspect_ratio="21:9"),
        )
        assert_http_error(
            refused,
            status_code=422,
            code=VIDEO_JOB_TOO_LARGE,
            message=(
                "This job needs 16.4 GB of GPU memory but this card has 16.0 GB. "
                "Lower the resolution, duration, frame rate, or aspect width."
            ),
        )
        admitted_id = _create_audio(
            client, headers, tmp_path, test_state, duration_seconds=5.0
        )
        admitted = client.post(
            "/api/generations/audio-to-video",
            headers=headers,
            json=_a2v_body(admitted_id, resolution="1080p", aspect_ratio="16:9"),
        )
        assert admitted.status_code == 200, admitted.text

    assert len(_store(test_state).list_generations("audio-to-video")) == 1


@pytest.mark.parametrize("surface", ["home", "remote"])
@pytest.mark.usefixtures("single_pass_only")
def test_windows_queued_a2v_auto_resolves_start_frame_before_insert(
    test_state, create_fake_model_files, tmp_path: Path, surface: str
) -> None:
    # A2V auto budgeted at the anchor though the start frame fixes the shape:
    # an ultrawide frame resolves to 21:9 and must refuse like an explicit one,
    # while a square frame and a missing frame (16:9 default) admit.
    create_fake_model_files()
    _configure_windows_fast(test_state, vram_gib=16, total_ram_gib=48)

    with _surface_client(test_state, surface) as (client, headers):
        wide_audio = _create_audio(
            client, headers, tmp_path, test_state, duration_seconds=5.0
        )
        wide_id = _create_start_frame(client, headers, tmp_path, size=(64, 16))
        refused = client.post(
            "/api/generations/audio-to-video",
            headers=headers,
            json=_a2v_body(
                wide_audio,
                resolution="1080p",
                aspect_ratio="auto",
                start_id=wide_id,
            ),
        )
        assert_http_error(
            refused,
            status_code=422,
            code=VIDEO_JOB_TOO_LARGE,
            message=(
                "This job needs 16.4 GB of GPU memory but this card has 16.0 GB. "
                "Lower the resolution, duration, frame rate, or aspect width."
            ),
        )
        square_audio = _create_audio(
            client, headers, tmp_path, test_state, duration_seconds=5.0
        )
        square_id = _create_start_frame(client, headers, tmp_path, size=(16, 16))
        admitted = client.post(
            "/api/generations/audio-to-video",
            headers=headers,
            json=_a2v_body(
                square_audio,
                resolution="1080p",
                aspect_ratio="auto",
                start_id=square_id,
            ),
        )
        assert admitted.status_code == 200, admitted.text
        default_audio = _create_audio(
            client, headers, tmp_path, test_state, duration_seconds=5.0
        )
        defaulted = client.post(
            "/api/generations/audio-to-video",
            headers=headers,
            json=_a2v_body(
                default_audio, resolution="1080p", aspect_ratio="auto"
            ),
        )
        assert defaulted.status_code == 200, defaulted.text

    assert len(_store(test_state).list_generations("audio-to-video")) == 2
