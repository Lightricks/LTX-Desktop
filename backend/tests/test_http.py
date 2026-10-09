import logging
import sqlite3
import time
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from PIL import Image
from pydantic import JsonValue, ValidationError
from starlette.testclient import TestClient

from api_types import RetakeParams, RetakeRequestParams
from app_factory import create_app
from app_handler import ServiceBundle, open_store
from frame_math import compute_num_frames, num_frames_for_audio_duration
from handlers.queued_generation_handler import QueuedGenerationHandler
from services.store import Store
from services.media_probe import MAX_IMAGE_BYTES, MAX_VIDEO_BYTES
from services.sqlite_store import SqliteStore
from services.records import AssetRecord, OutputSpec, UnavailableError
from services.unavailable_store import UnavailableStore
from state import build_initial_state, set_state_service_for_tests
from state.app_state_types import HfAuthenticated
from tests.conftest import TEST_ADMIN_TOKEN
from tests.http_error_assertions import assert_http_error


class _RecordingQueueControl:
    def __init__(self, store: SqliteStore) -> None:
        self._store = store
        self.queued_visible_on_wake: list[bool] = []

    def wake(self) -> None:
        self.queued_visible_on_wake.append(self._store.has_queued())

    def interrupt(self, generation_id: str, attempt_count: int) -> None:
        del generation_id, attempt_count


def _png(path: Path, size: tuple[int, int] = (16, 16)) -> Path:
    Image.new("RGB", size, color=(1, 2, 3)).save(path)
    return path


def _wav(path: Path, *, duration_seconds: float = 8.0) -> Path:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * int(duration_seconds * 8000))
    return path


def _store(test_state) -> SqliteStore:
    return SqliteStore(test_state.config.app_data_dir)


def _params(prompt: str = "fox") -> dict[str, JsonValue]:
    return {"prompt": prompt, "model": "ltx-2.5-fast"}


def _spec(prompt: str | None = None) -> dict[str, JsonValue]:
    spec: dict[str, JsonValue] = {"params": {}, "inputs": {}}
    if prompt is not None:
        spec["params"] = _params(prompt)
    return spec


def _succeed_png(test_state, *, name: str = "out.png") -> AssetRecord:
    store = _store(test_state)
    gen = store.insert_generation("text-to-video", _spec("p"))
    claimed = store.claim_next_queued()
    assert claimed is not None
    asset_id, dest = store.allocate_output_path("image", "image/png")
    Image.new("RGB", (8, 8)).save(dest)
    done = store.mark_succeeded(
        gen.id,
        [
            OutputSpec(
                asset_id=asset_id,
                dest_path=dest,
                ordinal=0,
                mime_type="image/png",
                name=name,
            )
        ],
        attempt_count=claimed.attempt_count,
    )
    return done.outputs[0]


def _a2v_params(**overrides: JsonValue) -> dict[str, JsonValue]:
    """Public A2V create params; the server derives persisted frame count."""
    params: dict[str, JsonValue] = {
        "prompt": "the character speaks",
        "resolution": "540p",
        "model": "ltx-2.5-fast",
        "fps": 24,
    }
    params.update(overrides)
    return params


def _mp4(path: Path, *, duration_seconds: float = 4.0, fps: int = 8) -> Path:
    import numpy as np
    import imageio.v2 as imageio

    frames = max(1, int(duration_seconds * fps))
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", macro_block_size=None
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.append_data(frame)
    writer.close()
    return path


def _retake_params(**overrides: JsonValue) -> dict[str, JsonValue]:
    params: dict[str, JsonValue] = {
        "prompt": "the character turns",
        "model": "ltx-2.5-fast",
        "startTime": 0,
        "duration": 2,
        "mode": "replace_audio_and_video",
    }
    params.update(overrides)
    return params


def _extend_params(**overrides: JsonValue) -> dict[str, JsonValue]:
    params: dict[str, JsonValue] = {
        "prompt": "the scene continues",
        "model": "ltx-2.5-fast",
        "duration": 4,
        "mode": "end",
    }
    params.update(overrides)
    return params


def _create_body(
    *,
    prompt: str = "fox",
    contract_version: int | None = None,
) -> dict[str, JsonValue]:
    body: dict[str, JsonValue] = {
        "params": _params(prompt),
    }
    if contract_version is not None:
        body["contract_version"] = contract_version
    return body


def _bundle(fake_services, store: Store) -> ServiceBundle:
    return ServiceBundle(
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
        store=store,
    )


@contextmanager
def _reserved_generation_start(test_state) -> Iterator[None]:
    assert test_state.generation.try_reserve_generation_start() is True
    try:
        yield
    finally:
        test_state.generation.release_generation_start_reservation()


def wait_for_http_status(
    client: TestClient, generation_id: str, status: str, timeout: float = 3.0
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/generations/{generation_id}")
        if response.status_code == 200 and response.json()["status"] == status:
            return True
        time.sleep(0.01)
    response = client.get(f"/api/generations/{generation_id}")
    return response.status_code == 200 and response.json()["status"] == status


def test_test_client_starts_and_stops_generation_queue_runner(test_state) -> None:
    app = create_app(handler=test_state)
    assert test_state.generation_queue.is_running is False
    with TestClient(app):
        assert test_state.generation_queue.is_running is True
    assert test_state.generation_queue.is_running is False


def test_http_create_is_processed_by_queue_runner(client) -> None:
    response = client.post(
        "/api/generations/text-to-video",
        json=_create_body(),
    )
    assert response.status_code == 200
    generation_id = response.json()["id"]
    assert response.json()["status"] == "queued"
    assert wait_for_http_status(client, generation_id, "succeeded") is True
    completed = client.get(f"/api/generations/{generation_id}")
    assert completed.status_code == 200
    assert completed.json()["outputs"][0]["media_kind"] == "video"
    assert Path(completed.json()["outputs"][0]["path"]).is_file()


def test_create_image_to_video_persists_asset_ids_without_paths(
    client, test_state, tmp_path: Path
) -> None:
    with _reserved_generation_start(test_state):
        start = client.post(
            "/api/assets", json={"path": str(_png(tmp_path / "start.png"))}
        ).json()

        response = client.post(
            "/api/generations/image-to-video",
            json={
                "params": {**_params("the fox turns toward camera")},
                "inputs": {"startFrame": {"assetId": start["id"]}},
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["feature"] == "image-to-video"
        assert payload["status"] == "queued"
        assert payload["spec"]["params"]["prompt"] == "the fox turns toward camera"
        assert payload["spec"]["inputs"] == {
            "startFrame": {"assetId": start["id"]}
        }
        assert "imagePath" not in payload["spec"]["params"]
        assert start["path"] not in str(payload["spec"])


def test_create_image_to_video_allows_blank_prompt_with_start_frame(
    client, test_state, tmp_path: Path
) -> None:
    with _reserved_generation_start(test_state):
        start = client.post(
            "/api/assets", json={"path": str(_png(tmp_path / "start.png"))}
        ).json()

        response = client.post(
            "/api/generations/image-to-video",
            json={
                "params": {**_params("")},
                "inputs": {"startFrame": {"assetId": start["id"]}},
            },
        )

        assert response.status_code == 200
        assert response.json()["spec"]["params"]["prompt"] == ""


def test_create_image_to_video_omitted_aspect_ratio_persists_auto(
    client, test_state, tmp_path: Path
) -> None:
    with _reserved_generation_start(test_state):
        start = client.post(
            "/api/assets",
            json={"path": str(_png(tmp_path / "portrait.png", (16, 32)))},
        ).json()

        response = client.post(
            "/api/generations/image-to-video",
            json={
                "params": {**_params("fox")},
                "inputs": {"startFrame": {"assetId": start["id"]}},
            },
        )

        assert response.status_code == 200
        assert response.json()["spec"]["params"]["aspectRatio"] == "auto"


def test_create_image_to_video_maps_optional_end_frame(
    client, test_state, tmp_path: Path
) -> None:
    with _reserved_generation_start(test_state):
        start = client.post(
            "/api/assets", json={"path": str(_png(tmp_path / "start.png"))}
        ).json()
        end = client.post(
            "/api/assets", json={"path": str(_png(tmp_path / "end.png"))}
        ).json()

        response = client.post(
            "/api/generations/image-to-video",
            json={
                "params": {**_params("camera pushes in")},
                "inputs": {
                    "startFrame": {"assetId": start["id"]},
                    "endFrame": {"assetId": end["id"]},
                },
            },
        )

        assert response.status_code == 200
        assert response.json()["spec"]["inputs"] == {
            "startFrame": {"assetId": start["id"]},
            "endFrame": {"assetId": end["id"]},
        }


def test_create_image_to_video_rejects_unknown_asset(client, test_state) -> None:
    response = client.post(
        "/api/generations/image-to-video",
        json={
            "params": {**_params("fox")},
            "inputs": {"startFrame": {"assetId": "missing-asset"}},
        },
    )

    data = assert_http_error(
        response,
        status_code=422,
        code="INPUT_ASSET_NOT_FOUND",
        message="startFrame image is no longer available",
    )
    assert "missing-asset" not in data["message"]
    assert _store(test_state).list_generations("image-to-video") == []


def test_create_image_to_video_rejects_non_image_assets_before_queueing(
    client, test_state, tmp_path: Path
) -> None:
    image = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "start.png"))}
    ).json()
    audio = client.post(
        "/api/assets", json={"path": str(_wav(tmp_path / "input.wav"))}
    ).json()
    invalid_inputs = (
        {"startFrame": {"assetId": audio["id"]}},
        {
            "startFrame": {"assetId": image["id"]},
            "endFrame": {"assetId": audio["id"]},
        },
    )

    for inputs in invalid_inputs:
        response = client.post(
            "/api/generations/image-to-video",
            json={"params": {**_params("fox")}, "inputs": inputs},
        )
        assert response.status_code == 422
        assert response.json()["code"] == "INVALID_GENERATION_SPEC"

    assert _store(test_state).list_generations("image-to-video") == []


def test_create_image_to_video_rejects_missing_or_invalid_start_frame(
    client, test_state
) -> None:
    invalid_inputs = (
        {},
        {"startFrame": {"assetId": ""}},
        {"endFrame": {"assetId": "end-without-start"}},
    )

    for inputs in invalid_inputs:
        response = client.post(
            "/api/generations/image-to-video",
            json={"params": {**_params("")}, "inputs": inputs},
        )
        assert response.status_code == 422

    assert _store(test_state).list_generations("image-to-video") == []


def test_create_image_to_video_rejects_loras(client, test_state, tmp_path: Path) -> None:
    start = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "start.png"))}
    ).json()

    response = client.post(
        "/api/generations/image-to-video",
        json={
            "params": {
                **_params("fox"),
                "loras": [{"ref": "style.safetensors", "scale": 1.0}],
            },
            "inputs": {"startFrame": {"assetId": start["id"]}},
        },
    )

    assert response.status_code == 422
    assert _store(test_state).list_generations("image-to-video") == []


def test_create_audio_to_video_persists_asset_ids_without_paths(
    client, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    with _reserved_generation_start(test_state):
        audio = client.post(
            "/api/assets", json={"path": str(_wav(tmp_path / "line.wav"))}
        ).json()
        response = client.post(
            "/api/generations/audio-to-video",
            json={
                "params": _a2v_params(),
                "inputs": {"audio": {"assetId": audio["id"]}},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["feature"] == "audio-to-video"
        assert payload["status"] == "queued"
        assert payload["spec"]["inputs"] == {"audio": {"assetId": audio["id"]}}
        assert audio["path"] not in str(payload["spec"])
        assert "audioPath" not in payload["spec"]["params"]


def test_create_audio_to_video_allows_blank_prompt_with_start_frame(
    client, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    with _reserved_generation_start(test_state):
        audio = client.post(
            "/api/assets", json={"path": str(_wav(tmp_path / "line.wav"))}
        ).json()
        start = client.post(
            "/api/assets", json={"path": str(_png(tmp_path / "start.png"))}
        ).json()

        response = client.post(
            "/api/generations/audio-to-video",
            json={
                "params": _a2v_params(prompt=""),
                "inputs": {
                    "audio": {"assetId": audio["id"]},
                    "startFrame": {"assetId": start["id"]},
                },
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "queued"
        assert payload["spec"]["params"]["prompt"] == ""
        assert payload["spec"]["params"]["resolution"] == "540p"
        assert payload["spec"]["params"]["numFrames"] == 185
        assert payload["spec"]["params"]["fps"] == 24
        assert payload["spec"]["params"]["model"] == "ltx-2.5-fast"
        assert payload["spec"]["inputs"] == {
            "audio": {"assetId": audio["id"]},
            "startFrame": {"assetId": start["id"]},
        }
        assert audio["path"] not in str(payload["spec"])
        assert start["path"] not in str(payload["spec"])


def test_create_audio_to_video_rejects_blank_prompt_without_image(
    client, test_state, tmp_path: Path
) -> None:
    audio = client.post(
        "/api/assets", json={"path": str(_wav(tmp_path / "line.wav"))}
    ).json()

    response = client.post(
        "/api/generations/audio-to-video",
        json={
            "params": _a2v_params(prompt=""),
            "inputs": {"audio": {"assetId": audio["id"]}},
        },
    )

    assert_http_error(
        response,
        status_code=422,
        code="INVALID_GENERATION_SPEC",
        message="Connect a prompt or a start image before running this node.",
    )
    assert _store(test_state).list_generations("audio-to-video") == []


@pytest.mark.parametrize(
    "params",
    [
        _a2v_params(model="pro"),
        _a2v_params(model="fast"),
        _a2v_params(numFrames=185),
    ],
)
def test_create_audio_to_video_rejects_cells_outside_the_derived_one(
    client, test_state, tmp_path: Path, params: dict[str, JsonValue]
) -> None:
    audio = client.post(
        "/api/assets", json={"path": str(_wav(tmp_path / "line.wav"))}
    ).json()

    response = client.post(
        "/api/generations/audio-to-video",
        json={"params": params, "inputs": {"audio": {"assetId": audio["id"]}}},
    )

    assert response.status_code == 422
    assert _store(test_state).list_generations("audio-to-video") == []


def test_create_audio_to_video_rejects_legacy_duration(
    client, test_state, tmp_path: Path
) -> None:
    audio = client.post(
        "/api/assets",
        json={"path": str(_wav(tmp_path / "line.wav", duration_seconds=8.0))},
    ).json()

    response = client.post(
        "/api/generations/audio-to-video",
        json={
            "params": _a2v_params(duration=20),
            "inputs": {"audio": {"assetId": audio["id"]}},
        },
    )

    assert response.status_code == 422
    assert _store(test_state).list_generations("audio-to-video") == []


def test_create_audio_to_video_derives_audio_grid_frames_without_ceil_snapping(
    client, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    with _reserved_generation_start(test_state):
        audio = client.post(
            "/api/assets",
            json={"path": str(_wav(tmp_path / "line.wav", duration_seconds=10.563))},
        ).json()

        response = client.post(
            "/api/generations/audio-to-video",
            json={
                "params": _a2v_params(),
                "inputs": {"audio": {"assetId": audio["id"]}},
            },
        )

    assert response.status_code == 200
    assert response.json()["spec"]["params"]["numFrames"] == 249


def test_create_audio_to_video_derives_frames_at_48_fps(
    client, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    duration_seconds = 10.563
    with _reserved_generation_start(test_state):
        audio = client.post(
            "/api/assets",
            json={"path": str(_wav(tmp_path / "line.wav", duration_seconds=duration_seconds))},
        ).json()

        response = client.post(
            "/api/generations/audio-to-video",
            json={
                "params": _a2v_params(fps=48),
                "inputs": {"audio": {"assetId": audio["id"]}},
            },
        )

    assert response.status_code == 200
    expected = num_frames_for_audio_duration(
        duration_seconds,
        48,
        max_frames=compute_num_frames(20, 48),
    )
    assert response.json()["spec"]["params"]["numFrames"] == expected
    assert response.json()["spec"]["params"]["fps"] == 48


def test_create_audio_to_video_accepts_advertised_1080p(
    client, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    with _reserved_generation_start(test_state):
        audio = client.post(
            "/api/assets",
            json={"path": str(_wav(tmp_path / "line.wav", duration_seconds=8.0))},
        ).json()

        response = client.post(
            "/api/generations/audio-to-video",
            json={
                "params": _a2v_params(resolution="1080p"),
                "inputs": {"audio": {"assetId": audio["id"]}},
            },
        )

    assert response.status_code == 200
    assert response.json()["spec"]["params"]["resolution"] == "1080p"
    assert response.json()["spec"]["params"]["numFrames"] == 185


def test_create_audio_to_video_uses_1080p_prefix_when_audio_exceeds_cell(
    client, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    with _reserved_generation_start(test_state):
        audio = client.post(
            "/api/assets",
            json={"path": str(_wav(tmp_path / "line.wav", duration_seconds=20.0))},
        ).json()

        response = client.post(
            "/api/generations/audio-to-video",
            json={
                "params": _a2v_params(resolution="1080p"),
                "inputs": {"audio": {"assetId": audio["id"]}},
            },
        )

    assert response.status_code == 200
    assert response.json()["spec"]["params"]["resolution"] == "1080p"
    assert response.json()["spec"]["params"]["numFrames"] == 233


def test_create_audio_to_video_rejects_unadvertised_resolution(
    client, test_state, tmp_path: Path
) -> None:
    audio = client.post(
        "/api/assets", json={"path": str(_wav(tmp_path / "line.wav"))}
    ).json()

    response = client.post(
        "/api/generations/audio-to-video",
        json={
            "params": _a2v_params(resolution="1440p"),
            "inputs": {"audio": {"assetId": audio["id"]}},
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "HTTP_422"
    assert _store(test_state).list_generations("audio-to-video") == []


@pytest.mark.parametrize(
    ("duration_seconds", "message"),
    [
        (1.0, "Audio is too short for audio-to-video."),
        (20.1, "Audio is too long for audio-to-video."),
    ],
)
def test_create_audio_to_video_rejects_audio_outside_duration_envelope(
    client, test_state, create_fake_model_files, tmp_path: Path, duration_seconds: float, message: str
) -> None:
    create_fake_model_files()
    audio = client.post(
        "/api/assets",
        json={"path": str(_wav(tmp_path / "line.wav", duration_seconds=duration_seconds))},
    ).json()

    response = client.post(
        "/api/generations/audio-to-video",
        json={
            "params": _a2v_params(),
            "inputs": {"audio": {"assetId": audio["id"]}},
        },
    )

    assert_http_error(
        response,
        status_code=422,
        code="INVALID_VIDEO_GENERATION_SPEC",
        message=message,
    )
    assert _store(test_state).list_generations("audio-to-video") == []


def test_create_audio_to_video_rejects_unadvertised_local_cell(
    client, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    test_state.config.local_generations_mode = "unsupported"
    audio = client.post(
        "/api/assets",
        json={"path": str(_wav(tmp_path / "line.wav", duration_seconds=5.0))},
    ).json()

    response = client.post(
        "/api/generations/audio-to-video",
        json={
            "params": _a2v_params(),
            "inputs": {"audio": {"assetId": audio["id"]}},
        },
    )

    assert_http_error(
        response,
        status_code=422,
        code="INVALID_VIDEO_GENERATION_SPEC",
        message="Audio-to-video is not available at the selected resolution.",
    )
    assert _store(test_state).list_generations("audio-to-video") == []


def test_create_audio_to_video_rejects_missing_offering_checkpoint(
    client, test_state, tmp_path: Path
) -> None:
    audio = client.post(
        "/api/assets", json={"path": str(_wav(tmp_path / "line.wav"))}
    ).json()

    response = client.post(
        "/api/generations/audio-to-video",
        json={
            "params": _a2v_params(),
            "inputs": {"audio": {"assetId": audio["id"]}},
        },
    )

    assert_http_error(
        response,
        status_code=409,
        code="LTX_MODEL_NOT_INSTALLED",
        message="LTX_MODEL_NOT_INSTALLED",
    )
    assert _store(test_state).list_generations("audio-to-video") == []


def test_create_audio_to_video_uses_installed_2_3_checkpoint_not_catalog_preferred(
    client, test_state, create_fake_model_files, tmp_path: Path, monkeypatch
) -> None:
    create_fake_model_files(model_id="ltx-2.3-22b-distilled")
    seen: list[object] = []
    import handlers.video_generation_handler as handler_mod

    real = handler_mod.get_local_video_generation_model_specs

    def wrapped(model_id, **kwargs):
        seen.append(model_id)
        return real(model_id, **kwargs)

    monkeypatch.setattr(handler_mod, "get_local_video_generation_model_specs", wrapped)
    with _reserved_generation_start(test_state):
        audio = client.post(
            "/api/assets", json={"path": str(_wav(tmp_path / "line.wav"))}
        ).json()
        response = client.post(
            "/api/generations/audio-to-video",
            json={
                "params": _a2v_params(model="ltx-2.3-fast"),
                "inputs": {"audio": {"assetId": audio["id"]}},
            },
        )

    assert response.status_code == 200, response.text
    assert "ltx-2.3-22b-distilled" in seen
    assert "ltx-2.3-22b-distilled-1.1" not in seen


def test_create_audio_to_video_rejects_missing_audio(client, test_state) -> None:
    response = client.post(
        "/api/generations/audio-to-video",
        json={
            "params": _a2v_params(),
            "inputs": {},
        },
    )

    assert response.status_code == 422
    assert _store(test_state).list_generations("audio-to-video") == []


def test_create_audio_to_video_rejects_image_as_audio(
    client, test_state, tmp_path: Path
) -> None:
    image = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "start.png"))}
    ).json()

    response = client.post(
        "/api/generations/audio-to-video",
        json={
            "params": _a2v_params(),
            "inputs": {"audio": {"assetId": image["id"]}},
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_GENERATION_SPEC"
    assert _store(test_state).list_generations("audio-to-video") == []


def test_create_audio_to_video_rejects_unknown_audio_asset(
    client, test_state
) -> None:
    response = client.post(
        "/api/generations/audio-to-video",
        json={
            "params": _a2v_params(),
            "inputs": {"audio": {"assetId": "missing-asset"}},
        },
    )

    data = assert_http_error(
        response,
        status_code=422,
        code="INPUT_ASSET_NOT_FOUND",
        message="audio is no longer available",
    )
    assert "missing-asset" not in data["message"]
    assert _store(test_state).list_generations("audio-to-video") == []


def test_create_audio_to_video_runs_local_a2v_pipeline_not_ltx_api(
    test_state,
    fake_services,
    create_fake_model_files,
    tmp_path: Path,
    default_app_settings,
) -> None:
    create_fake_model_files()
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(
            fake_services, SqliteStore(test_state.config.app_data_dir)
        ),
    )
    handler.state.hf_auth_state = HfAuthenticated(
        access_token="fake-hf-token",
        expires_at=1e18,
    )
    handler.state.app_settings.use_local_text_encoder = True
    set_state_service_for_tests(handler)
    app = create_app(handler=handler, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        audio = client.post(
            "/api/assets", json={"path": str(_wav(tmp_path / "line.wav"))}
        ).json()
        response = client.post(
            "/api/generations/audio-to-video",
            json={
                "params": _a2v_params(),
                "inputs": {"audio": {"assetId": audio["id"]}},
            },
        )
        assert response.status_code == 200
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if fake_services.a2v_pipeline.generate_calls:
                break
            time.sleep(0.01)

    assert fake_services.ltx_api_client.audio_to_video_calls == []
    assert len(fake_services.a2v_pipeline.generate_calls) == 1
    assert fake_services.a2v_pipeline.generate_calls[0]["audio_path"] == audio["path"]


def test_create_audio_to_video_loads_2_5_weights_when_settings_is_2_3(
    test_state,
    fake_services,
    create_fake_model_files,
    tmp_path: Path,
    default_app_settings,
) -> None:
    create_fake_model_files()
    create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(
            fake_services, SqliteStore(test_state.config.app_data_dir)
        ),
    )
    handler.state.hf_auth_state = HfAuthenticated(
        access_token="fake-hf-token",
        expires_at=1e18,
    )
    handler.state.app_settings.use_local_text_encoder = True
    handler.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled-1.1"
    set_state_service_for_tests(handler)
    app = create_app(handler=handler, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        audio = client.post(
            "/api/assets", json={"path": str(_wav(tmp_path / "line.wav"))}
        ).json()
        response = client.post(
            "/api/generations/audio-to-video",
            json={
                "params": _a2v_params(),
                "inputs": {"audio": {"assetId": audio["id"]}},
            },
        )
        assert response.status_code == 200, response.text
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if fake_services.a2v_pipeline.generate_calls:
                break
            time.sleep(0.01)

    slot = handler.state.gpu_slot
    assert slot is not None
    assert slot.active_pipeline.ltx_model_id == "ltx-2.5-22b-distilled"
    assert handler.state.app_settings.active_ltx_model_id == "ltx-2.3-22b-distilled-1.1"
    assert len(fake_services.a2v_pipeline.generate_calls) == 1


def test_create_audio_to_video_loads_2_3_weights_when_settings_is_2_5(
    test_state,
    fake_services,
    create_fake_model_files,
    tmp_path: Path,
    default_app_settings,
) -> None:
    create_fake_model_files()
    create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(
            fake_services, SqliteStore(test_state.config.app_data_dir)
        ),
    )
    handler.state.hf_auth_state = HfAuthenticated(
        access_token="fake-hf-token",
        expires_at=1e18,
    )
    handler.state.app_settings.use_local_text_encoder = True
    handler.state.app_settings.active_ltx_model_id = "ltx-2.5-22b-distilled"
    set_state_service_for_tests(handler)
    app = create_app(handler=handler, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        audio = client.post(
            "/api/assets", json={"path": str(_wav(tmp_path / "line.wav"))}
        ).json()
        response = client.post(
            "/api/generations/audio-to-video",
            json={
                "params": _a2v_params(model="ltx-2.3-fast"),
                "inputs": {"audio": {"assetId": audio["id"]}},
            },
        )
        assert response.status_code == 200, response.text
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if fake_services.a2v_pipeline.generate_calls:
                break
            time.sleep(0.01)

    slot = handler.state.gpu_slot
    assert slot is not None
    assert slot.active_pipeline.ltx_model_id == "ltx-2.3-22b-distilled-1.1"
    assert handler.state.app_settings.active_ltx_model_id == "ltx-2.5-22b-distilled"
    assert len(fake_services.a2v_pipeline.generate_calls) == 1


def test_create_retake_persists_asset_ids_without_paths(
    client, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    with _reserved_generation_start(test_state):
        video = client.post(
            "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
        ).json()
        response = client.post(
            "/api/generations/retake",
            json={
                "params": _retake_params(),
                "inputs": {"video": {"assetId": video["id"]}},
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["feature"] == "retake"
        assert payload["status"] == "queued"
        assert payload["spec"]["inputs"] == {"video": {"assetId": video["id"]}}
        assert payload["spec"]["params"]["model"] == "ltx-2.5-fast"
        assert payload["spec"]["params"]["startTime"] == 0
        assert payload["spec"]["params"]["duration"] == 2
        assert payload["spec"]["params"]["mode"] == "replace_audio_and_video"
        assert video["path"] not in str(payload["spec"])
        assert "video_path" not in payload["spec"]["params"]


def test_create_retake_rejects_selection_over_cap(
    client, test_state, tmp_path: Path
) -> None:
    video = client.post(
        "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
    ).json()
    response = client.post(
        "/api/generations/retake",
        json={
            "params": {**_retake_params(), "duration": 10.1},
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert response.status_code == 422
    assert _store(test_state).list_generations("retake") == []


def test_retake_cap_accepts_exactly_ten_seconds() -> None:
    params = {**_retake_params(), "duration": 10.0}
    assert RetakeRequestParams.model_validate(params).duration == 10.0


def test_retake_queued_before_cap_still_parses() -> None:
    params = {**_retake_params(), "duration": 12.0}
    assert RetakeParams.model_validate(params).duration == 12.0
    with pytest.raises(ValidationError):
        RetakeRequestParams.model_validate(params)


def test_create_retake_rejects_missing_video(client, test_state) -> None:
    response = client.post(
        "/api/generations/retake",
        json={
            "params": _retake_params(),
            "inputs": {"video": {"assetId": "missing-asset"}},
        },
    )
    data = assert_http_error(
        response,
        status_code=422,
        code="INPUT_ASSET_NOT_FOUND",
        message="video is no longer available",
    )
    assert "missing-asset" not in data["message"]
    assert _store(test_state).list_generations("retake") == []


def test_create_retake_rejects_oversized_video(
    client, test_state, tmp_path: Path
) -> None:
    video = client.post(
        "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
    ).json()
    with Path(video["path"]).open("r+b") as handle:
        handle.truncate(MAX_VIDEO_BYTES + 1)
    response = client.post(
        "/api/generations/retake",
        json={
            "params": _retake_params(),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert_http_error(response, status_code=400, code="FILE_TOO_LARGE")
    assert _store(test_state).list_generations("retake") == []


def test_create_retake_rejects_image_as_video(
    client, test_state, tmp_path: Path
) -> None:
    image = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "still.png"))}
    ).json()
    response = client.post(
        "/api/generations/retake",
        json={
            "params": _retake_params(),
            "inputs": {"video": {"assetId": image["id"]}},
        },
    )
    assert_http_error(
        response,
        status_code=422,
        code="INVALID_GENERATION_SPEC",
        message="video must reference a video asset",
    )
    assert _store(test_state).list_generations("retake") == []


def test_create_retake_rejects_missing_offering_checkpoint(
    client, test_state, tmp_path: Path
) -> None:
    video = client.post(
        "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
    ).json()
    response = client.post(
        "/api/generations/retake",
        json={
            "params": _retake_params(),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert_http_error(
        response,
        status_code=409,
        code="LTX_MODEL_NOT_INSTALLED",
        message="LTX_MODEL_NOT_INSTALLED",
    )
    assert _store(test_state).list_generations("retake") == []


@pytest.mark.parametrize("pipeline_id", ["fast", "pro", "pro-2.5", "fast-2.5"])
def test_create_retake_rejects_pipeline_model_ids(
    client, test_state, tmp_path: Path, pipeline_id: str
) -> None:
    video = client.post(
        "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
    ).json()
    response = client.post(
        "/api/generations/retake",
        json={
            "params": _retake_params(model=pipeline_id),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert response.status_code == 422
    assert _store(test_state).list_generations("retake") == []


def test_create_retake_runs_local_pipeline_not_ltx_api(
    test_state,
    fake_services,
    create_fake_model_files,
    tmp_path: Path,
    default_app_settings,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="services.features.video.retake")
    caplog.set_level(logging.INFO, logger="handlers.generation_handler")
    create_fake_model_files()
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(
            fake_services, SqliteStore(test_state.config.app_data_dir)
        ),
    )
    handler.state.hf_auth_state = HfAuthenticated(
        access_token="fake-hf-token",
        expires_at=1e18,
    )
    handler.state.app_settings.use_local_text_encoder = True
    set_state_service_for_tests(handler)
    app = create_app(handler=handler, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        video = client.post(
            "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
        ).json()
        response = client.post(
            "/api/generations/retake",
            json={
                "params": _retake_params(),
                "inputs": {"video": {"assetId": video["id"]}},
            },
        )
        assert response.status_code == 200, response.text
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if fake_services.retake_pipeline.generate_calls:
                break
            time.sleep(0.01)

    assert fake_services.ltx_api_client.retake_calls == []
    assert len(fake_services.retake_pipeline.generate_calls) == 1
    call = fake_services.retake_pipeline.generate_calls[0]
    assert call["video_path"] == video["path"]
    assert call["start_time"] == 0
    assert call["encode_max_duration"] is not None
    assert call["target_frames"] is not None
    assert call["regenerate_video"] is True
    assert call["regenerate_audio"] is True
    assert any(
        "Retake " in record.getMessage() and " started (" in record.getMessage()
        for record in caplog.records
        if record.name == "services.features.video.retake"
    )
    assert any(
        "started (gpu)" in record.getMessage()
        for record in caplog.records
        if record.name == "handlers.generation_handler"
    )


def test_create_retake_loads_2_5_weights_when_settings_is_2_3(
    test_state,
    fake_services,
    create_fake_model_files,
    tmp_path: Path,
    default_app_settings,
) -> None:
    create_fake_model_files()
    create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(
            fake_services, SqliteStore(test_state.config.app_data_dir)
        ),
    )
    handler.state.hf_auth_state = HfAuthenticated(
        access_token="fake-hf-token",
        expires_at=1e18,
    )
    handler.state.app_settings.use_local_text_encoder = True
    handler.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled-1.1"
    set_state_service_for_tests(handler)
    app = create_app(handler=handler, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        video = client.post(
            "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
        ).json()
        response = client.post(
            "/api/generations/retake",
            json={
                "params": _retake_params(),
                "inputs": {"video": {"assetId": video["id"]}},
            },
        )
        assert response.status_code == 200, response.text
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if fake_services.retake_pipeline.generate_calls:
                break
            time.sleep(0.01)

    slot = handler.state.gpu_slot
    assert slot is not None
    assert slot.active_pipeline.ltx_model_id == "ltx-2.5-22b-distilled"
    assert handler.state.app_settings.active_ltx_model_id == "ltx-2.3-22b-distilled-1.1"
    assert len(fake_services.retake_pipeline.generate_calls) == 1


def test_create_extend_persists_asset_ids_without_paths(
    client, test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    with _reserved_generation_start(test_state):
        video = client.post(
            "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
        ).json()
        response = client.post(
            "/api/generations/extend",
            json={
                "params": _extend_params(),
                "inputs": {"video": {"assetId": video["id"]}},
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["feature"] == "extend"
        assert payload["status"] == "queued"
        assert payload["spec"]["inputs"] == {"video": {"assetId": video["id"]}}
        assert payload["spec"]["params"]["model"] == "ltx-2.5-fast"
        assert payload["spec"]["params"]["duration"] == 4
        assert payload["spec"]["params"]["mode"] == "end"
        assert video["path"] not in str(payload["spec"])
        assert "video_path" not in payload["spec"]["params"]


def test_create_extend_rejects_missing_video(client, test_state) -> None:
    response = client.post(
        "/api/generations/extend",
        json={
            "params": _extend_params(),
            "inputs": {"video": {"assetId": "missing-asset"}},
        },
    )
    data = assert_http_error(
        response,
        status_code=422,
        code="INPUT_ASSET_NOT_FOUND",
        message="video is no longer available",
    )
    assert "missing-asset" not in data["message"]
    assert _store(test_state).list_generations("extend") == []


def test_create_extend_rejects_image_as_video(
    client, test_state, tmp_path: Path
) -> None:
    image = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "still.png"))}
    ).json()
    response = client.post(
        "/api/generations/extend",
        json={
            "params": _extend_params(),
            "inputs": {"video": {"assetId": image["id"]}},
        },
    )
    assert_http_error(
        response,
        status_code=422,
        code="INVALID_GENERATION_SPEC",
        message="video must reference a video asset",
    )
    assert _store(test_state).list_generations("extend") == []


def test_create_extend_rejects_missing_offering_checkpoint(
    client, test_state, tmp_path: Path
) -> None:
    video = client.post(
        "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
    ).json()
    response = client.post(
        "/api/generations/extend",
        json={
            "params": _extend_params(),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert_http_error(
        response,
        status_code=409,
        code="LTX_MODEL_NOT_INSTALLED",
        message="LTX_MODEL_NOT_INSTALLED",
    )
    assert _store(test_state).list_generations("extend") == []


@pytest.mark.parametrize("pipeline_id", ["fast", "pro", "pro-2.5", "fast-2.5"])
def test_create_extend_rejects_pipeline_model_ids(
    client, test_state, tmp_path: Path, pipeline_id: str
) -> None:
    video = client.post(
        "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
    ).json()
    response = client.post(
        "/api/generations/extend",
        json={
            "params": _extend_params(model=pipeline_id),
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )
    assert response.status_code == 422
    assert _store(test_state).list_generations("extend") == []


@pytest.mark.parametrize("mode", ["start", "end"])
def test_create_extend_runs_local_pipeline_not_ltx_api(
    test_state,
    fake_services,
    create_fake_model_files,
    tmp_path: Path,
    default_app_settings,
    mode: str,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="services.features.video.extend")
    caplog.set_level(logging.INFO, logger="handlers.generation_handler")
    create_fake_model_files()
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(
            fake_services, SqliteStore(test_state.config.app_data_dir)
        ),
    )
    handler.state.hf_auth_state = HfAuthenticated(
        access_token="fake-hf-token",
        expires_at=1e18,
    )
    handler.state.app_settings.use_local_text_encoder = True
    set_state_service_for_tests(handler)
    app = create_app(handler=handler, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        video = client.post(
            "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
        ).json()
        response = client.post(
            "/api/generations/extend",
            json={
                "params": _extend_params(mode=mode),
                "inputs": {"video": {"assetId": video["id"]}},
            },
        )
        assert response.status_code == 200, response.text
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if fake_services.retake_pipeline.extend_calls:
                break
            time.sleep(0.01)

    assert fake_services.ltx_api_client.extend_calls == []
    assert len(fake_services.retake_pipeline.extend_calls) == 1
    call = fake_services.retake_pipeline.extend_calls[0]
    assert call["video_path"] == video["path"]
    assert call["mode"] == mode
    assert call["encode_max_duration"] is not None
    assert call["extend_frames"] % 8 == 0
    assert any(
        "Extend " in record.getMessage() and " started (" in record.getMessage()
        for record in caplog.records
        if record.name == "services.features.video.extend"
    )
    assert any(
        "started (gpu)" in record.getMessage()
        for record in caplog.records
        if record.name == "handlers.generation_handler"
    )


def test_create_extend_loads_2_5_weights_when_settings_is_2_3(
    test_state,
    fake_services,
    create_fake_model_files,
    tmp_path: Path,
    default_app_settings,
) -> None:
    create_fake_model_files()
    create_fake_model_files(model_id="ltx-2.3-22b-distilled-1.1")
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(
            fake_services, SqliteStore(test_state.config.app_data_dir)
        ),
    )
    handler.state.hf_auth_state = HfAuthenticated(
        access_token="fake-hf-token",
        expires_at=1e18,
    )
    handler.state.app_settings.use_local_text_encoder = True
    handler.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled-1.1"
    set_state_service_for_tests(handler)
    app = create_app(handler=handler, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        video = client.post(
            "/api/assets", json={"path": str(_mp4(tmp_path / "clip.mp4"))}
        ).json()
        response = client.post(
            "/api/generations/extend",
            json={
                "params": _extend_params(),
                "inputs": {"video": {"assetId": video["id"]}},
            },
        )
        assert response.status_code == 200, response.text
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if fake_services.retake_pipeline.extend_calls:
                break
            time.sleep(0.01)

    slot = handler.state.gpu_slot
    assert slot is not None
    assert slot.active_pipeline.ltx_model_id == "ltx-2.5-22b-distilled"
    assert handler.state.app_settings.active_ltx_model_id == "ltx-2.3-22b-distilled-1.1"
    assert len(fake_services.retake_pipeline.extend_calls) == 1


@pytest.mark.parametrize("pipeline_id", ["fast", "pro", "pro-2.5", "fast-2.5"])
def test_create_text_to_video_rejects_pipeline_model_ids(
    client, test_state, pipeline_id: str
) -> None:
    response = client.post(
        "/api/generations/text-to-video",
        json={"params": {"prompt": "fox", "model": pipeline_id}},
    )

    assert response.status_code == 422
    assert _store(test_state).list_generations("text-to-video") == []


@pytest.mark.parametrize("pipeline_id", ["fast", "pro", "pro-2.5", "fast-2.5"])
def test_create_image_to_video_rejects_pipeline_model_ids(
    client, test_state, tmp_path: Path, pipeline_id: str
) -> None:
    start = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "start.png"))}
    ).json()
    response = client.post(
        "/api/generations/image-to-video",
        json={
            "params": {"prompt": "fox", "model": pipeline_id},
            "inputs": {"startFrame": {"assetId": start["id"]}},
        },
    )

    assert response.status_code == 422
    assert _store(test_state).list_generations("image-to-video") == []


def test_create_text_to_video_requires_model(client, test_state) -> None:
    response = client.post(
        "/api/generations/text-to-video",
        json={"params": {"prompt": "fox"}},
    )

    assert response.status_code == 422
    assert _store(test_state).list_generations("text-to-video") == []


def test_create_text_to_video_still_rejects_blank_prompt(client, test_state) -> None:
    response = client.post(
        "/api/generations/text-to-video",
        json={"params": {**_params("")}},
    )

    assert response.status_code == 422
    assert _store(test_state).list_generations("text-to-video") == []


def test_http_create_rejects_unknown_recipe_id(client, test_state) -> None:
    # The recipe create route matches any id under /generations/recipes/; an
    # unmapped id is a typed LORA_UNKNOWN miss, and nothing is enqueued.
    response = client.post(
        "/api/generations/recipes/not-a-real-recipe",
        json={
            "params": {
                "prompt": "fox",
                "model": "ltx-2.5-fast",
                "catalogId": "cozy-felt-style",
                "scale": 1.0,
            }
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "LORA_UNKNOWN"
    assert _store(test_state).list_generations("not-a-real-recipe") == []


def test_http_create_rejects_unsupported_contract_version_before_insert(
    client, test_state
) -> None:
    response = client.post(
        "/api/generations/text-to-video",
        json=_create_body(contract_version=2),
    )

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_GENERATION_SPEC"
    assert _store(test_state).list_generations("text-to-video") == []


def test_http_create_rejects_structurally_invalid_text_to_video_params_before_insert(
    client, test_state
) -> None:
    response = client.post(
        "/api/generations/text-to-video",
        json={
            "params": {**_params("fox"), "imagePath": "/tmp/start.png"},
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "HTTP_422"
    assert _store(test_state).list_generations("text-to-video") == []


def test_http_cancel_queued_generation(client, test_state) -> None:
    assert test_state.generation.try_reserve_generation_start() is True
    try:
        response = client.post(
            "/api/generations/text-to-video",
            json=_create_body(prompt="stop"),
        )
        generation_id = response.json()["id"]
        cancelled = client.post(f"/api/generations/{generation_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
    finally:
        test_state.generation.release_generation_start_reservation()


def test_queue_snapshot_lists_running_then_ranked_queued_work(
    client, test_state, tmp_path: Path
) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        input_asset = store.ingest_upload(str(_png(tmp_path / "start.png")))
        running = store.insert_generation(
            "image-to-video",
            {
                "params": {"prompt": "running"},
                "inputs": {"startFrame": {"assetId": input_asset.id}},
            },
        )
        second = store.insert_generation("text-to-video", _spec("second"))
        third = store.insert_generation("text-to-video", _spec("third"))
        claimed = store.claim_next_queued()
        assert claimed is not None
        assert claimed.id == running.id
        store.reorder_queued_generation(third.id, second.id)

        response = client.get("/api/generation-queue")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["active"]["generation"]["id"] == running.id
    assert body["active"]["generation"]["status"] == "running"
    assert body["active"]["input_assets"][0]["id"] == input_asset.id
    assert body["active"]["progress"] is None
    assert [entry["generation"]["id"] for entry in body["queued"]] == [
        third.id,
        second.id,
    ]
    assert body["done"] == []
    assert body["failed"] == []
    assert body["unseen_ids"] == []


def test_queue_snapshot_includes_progress_only_for_matching_active_id(
    client, test_state
) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        active = store.insert_generation("text-to-video", _spec("active"))
        queued = store.insert_generation("text-to-video", _spec("queued"))
        claimed = store.claim_next_queued()
        assert claimed is not None
        assert claimed.id == active.id

        test_state.generation.start_api_generation("unrelated-generation")
        unmatched = client.get("/api/generation-queue")
        test_state.generation.complete_generation(result="done")
        test_state.generation.start_api_generation(active.id)
        matched = client.get("/api/generation-queue")

    assert unmatched.status_code == 200, unmatched.text
    assert unmatched.json()["active"]["progress"] is None
    assert matched.status_code == 200, matched.text
    progress = matched.json()["active"]["progress"]
    assert progress == {
        "phase": "",
        "progress": 0,
        "currentStep": None,
        "totalSteps": None,
    }
    assert matched.json()["queued"][0]["generation"]["id"] == queued.id


def test_queue_snapshot_keeps_cancelling_active_and_queued_without_progress(
    client, test_state
) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        active = store.insert_generation("text-to-video", _spec("active"))
        queued = store.insert_generation("text-to-video", _spec("queued"))
        claimed = store.claim_next_queued()
        assert claimed is not None
        assert claimed.id == active.id
        test_state.generation.start_api_generation(active.id)

        cancelled = client.post(f"/api/generations/{active.id}/cancel")
        snapshot = client.get("/api/generation-queue")

    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelling"
    assert snapshot.status_code == 200, snapshot.text
    body = snapshot.json()
    assert body["active"]["generation"]["id"] == active.id
    assert body["active"]["generation"]["status"] == "cancelling"
    assert body["queued"][0]["generation"]["id"] == queued.id
    assert body["queued"][0]["progress"] is None


def test_http_reorder_moves_queued_generation_before_target(client, test_state) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        first = store.insert_generation("text-to-video", _spec("first"))
        second = store.insert_generation("text-to-video", _spec("second"))
        third = store.insert_generation("text-to-video", _spec("third"))

        reordered = client.post(
            "/api/generation-queue/reorder",
            json={"generation_id": third.id, "before_generation_id": second.id},
        )
        persisted = client.get("/api/generation-queue")

    assert reordered.status_code == 200, reordered.text
    assert reordered.json()["active"] is None
    assert [entry["generation"]["id"] for entry in reordered.json()["queued"]] == [
        first.id,
        third.id,
        second.id,
    ]
    assert persisted.json() == reordered.json()


def test_http_reorder_returns_post_mutation_queue_snapshot(
    client, test_state, tmp_path: Path
) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        input_asset = store.ingest_upload(str(_png(tmp_path / "start.png")))
        running = store.insert_generation(
            "image-to-video",
            {
                "params": {"prompt": "running"},
                "inputs": {"startFrame": {"assetId": input_asset.id}},
            },
        )
        first = store.insert_generation("text-to-video", _spec("first"))
        second = store.insert_generation(
            "image-to-video",
            {
                "params": {"prompt": "second"},
                "inputs": {"startFrame": {"assetId": input_asset.id}},
            },
        )
        claimed = store.claim_next_queued()
        assert claimed is not None
        assert claimed.id == running.id
        test_state.generation.start_api_generation(running.id)

        reordered = client.post(
            "/api/generation-queue/reorder",
            json={"generation_id": second.id, "before_generation_id": first.id},
        )

    assert reordered.status_code == 200, reordered.text
    body = reordered.json()
    assert "id" not in body
    assert body["active"]["generation"]["id"] == running.id
    assert body["active"]["generation"]["status"] == "running"
    assert body["active"]["input_assets"][0]["id"] == input_asset.id
    assert body["active"]["progress"] == {
        "phase": "",
        "progress": 0,
        "currentStep": None,
        "totalSteps": None,
    }
    assert [entry["generation"]["id"] for entry in body["queued"]] == [
        second.id,
        first.id,
    ]
    assert body["queued"][0]["generation"]["status"] == "queued"
    assert body["queued"][0]["input_assets"][0]["id"] == input_asset.id
    assert body["queued"][0]["progress"] is None
    assert body["queued"][1]["generation"]["id"] == first.id
    assert body["queued"][1]["input_assets"] == []
    assert body["queued"][1]["progress"] is None


def test_http_reorder_rejects_unknown_fields_preserves_relative_contract(
    client, test_state
) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        first = store.insert_generation("text-to-video", _spec("first"))
        second = store.insert_generation("text-to-video", _spec("second"))
        third = store.insert_generation("text-to-video", _spec("third"))

        revision_rejected = client.post(
            "/api/generation-queue/reorder",
            json={
                "generation_id": third.id,
                "before_generation_id": second.id,
                "revision": 1,
            },
        )
        array_rejected = client.post(
            "/api/generation-queue/reorder",
            json={
                "generation_id": third.id,
                "queue": [third.id, first.id, second.id],
            },
        )
        reordered = client.post(
            "/api/generation-queue/reorder",
            json={"generation_id": third.id, "before_generation_id": second.id},
        )
        persisted = client.get("/api/generation-queue")

    assert revision_rejected.status_code == 422
    assert array_rejected.status_code == 422
    assert reordered.status_code == 200, reordered.text
    assert [entry["generation"]["id"] for entry in reordered.json()["queued"]] == [
        first.id,
        third.id,
        second.id,
    ]
    assert persisted.json() == reordered.json()


def test_http_reorder_rejects_task_claimed_before_command(client, test_state) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        claimed = store.insert_generation("text-to-video", _spec("claimed"))
        target = store.insert_generation("text-to-video", _spec("target"))
        assert store.claim_next_queued() is not None

        response = client.post(
            "/api/generation-queue/reorder",
            json={
                "generation_id": claimed.id,
                "before_generation_id": target.id,
            },
        )

    assert_http_error(response, status_code=409, code="INVALID_GENERATION_STATUS")


def test_http_retry_failed_generation_returns_same_id(client, test_state) -> None:
    assert test_state.generation.try_reserve_generation_start() is True
    try:
        store = _store(test_state)
        generation = store.insert_generation("text-to-video", _spec("retry"))
        claimed = store.claim_next_queued()
        assert claimed is not None
        store.mark_failed(
            generation.id,
            "OUTPUT_UNREADABLE",
            attempt_count=claimed.attempt_count,
        )
        retried = client.post(f"/api/generations/{generation.id}/retry")
        assert retried.status_code == 200
        assert retried.json()["id"] == generation.id
        assert retried.json()["status"] == "queued"
    finally:
        test_state.generation.release_generation_start_reservation()


def test_ingest_asset_via_http(client, tmp_path: Path) -> None:
    source = _png(tmp_path / "up.png")
    r = client.post("/api/assets", json={"path": str(source)})
    assert r.status_code == 200
    data = r.json()
    assert data["origin"] == "uploaded"
    assert data["media_kind"] == "image"
    assert Path(data["path"]).is_file()


def test_get_asset_via_http(client, tmp_path: Path) -> None:
    ingested = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "lookup.png"))}
    )
    assert ingested.status_code == 200
    asset_id = ingested.json()["id"]

    r = client.get(f"/api/assets/{asset_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == asset_id
    assert data["media_kind"] == "image"
    assert data["origin"] == "uploaded"
    assert Path(data["path"]).is_file()


def test_list_assets_via_http(client, test_state, tmp_path: Path) -> None:
    ingested = client.post("/api/assets", json={"path": str(_png(tmp_path / "a.png"))})
    assert ingested.status_code == 200
    with _reserved_generation_start(test_state):
        output = _succeed_png(test_state)
    r = client.get("/api/assets")
    assert r.status_code == 200
    body = r.json()
    ids = [item["id"] for item in body["items"]]
    assert ingested.json()["id"] not in ids
    assert ids == [output.id]
    assert body["items"][0]["origin"] == "generated"
    assert "path" in body["items"][0]
    assert "in_use" in body["items"][0]


def test_missing_asset_file_is_tombstoned_via_http(client, test_state) -> None:
    with _reserved_generation_start(test_state):
        asset = _succeed_png(test_state, name="gone.png")
    Path(asset.path).unlink()
    listed = client.get("/api/assets")
    assert listed.status_code == 200
    assert asset.id not in [item["id"] for item in listed.json()["items"]]
    assert_http_error(
        client.get(f"/api/assets/{asset.id}"),
        status_code=404,
        code="ASSET_NOT_FOUND",
    )


def test_missing_asset_bytes_do_not_tombstone(client, tmp_path: Path) -> None:
    ingested = client.post("/api/assets", json={"path": str(_png(tmp_path / "gone.png"))})
    asset = ingested.json()
    payload = Path(asset["path"]).read_bytes()
    Path(asset["path"]).unlink()
    assert_http_error(
        client.get(f"/api/assets/{asset['id']}/bytes"),
        status_code=404,
        code="ASSET_NOT_FOUND",
    )
    Path(asset["path"]).write_bytes(payload)
    restored = client.get(f"/api/assets/{asset['id']}")
    assert restored.status_code == 200
    assert restored.json()["id"] == asset["id"]


def test_delete_asset_via_http(client, tmp_path: Path) -> None:
    ingested = client.post("/api/assets", json={"path": str(_png(tmp_path / "a.png"))})
    asset_id = ingested.json()["id"]
    path = Path(ingested.json()["path"])
    deleted = client.delete(f"/api/assets/{asset_id}")
    assert deleted.status_code == 200
    assert not path.exists()
    assert client.get(f"/api/assets/{asset_id}").status_code == 404


def test_delete_in_use_asset_409(client, tmp_path: Path) -> None:
    start = client.post("/api/assets", json={"path": str(_png(tmp_path / "s.png"))}).json()
    created = client.post(
        "/api/generations/image-to-video",
        json={
            "params": {**_params("x")},
            "inputs": {"startFrame": {"assetId": start["id"]}},
        },
    )
    assert created.status_code == 200
    r = client.delete(f"/api/assets/{start['id']}")
    assert_http_error(r, status_code=409, code="ASSET_IN_USE")


def test_desktop_serves_asset_bytes_for_the_trim_preview(
    client, tmp_path: Path
) -> None:
    source = _png(tmp_path / "preview.png")
    ingested = client.post("/api/assets", json={"path": str(source)})
    assert ingested.status_code == 200
    asset_id = ingested.json()["id"]

    # The renderer streams these bytes for waveform decoding instead of
    # marshalling a base64 copy of the whole file through IPC.
    served = client.get(f"/api/assets/{asset_id}/bytes")
    assert served.status_code == 200
    assert served.content == source.read_bytes()
    assert served.headers["content-disposition"].startswith("inline")
    assert client.head(f"/api/assets/{asset_id}/bytes").status_code == 200


def test_desktop_asset_bytes_404_for_unknown_asset(client) -> None:
    assert_http_error(
        client.get("/api/assets/missing-asset/bytes"),
        status_code=404,
        code="ASSET_NOT_FOUND",
    )


def test_phone_asset_routes_stay_off_desktop(client, test_state, tmp_path: Path) -> None:
    png_bytes = _png(tmp_path / "phone.png").read_bytes()
    uploaded = client.post(
        "/api/assets/upload",
        files={"file": ("phone.png", png_bytes, "image/png")},
    )
    ingested = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "lookup.png"))}
    )
    assert ingested.status_code == 200
    asset_id = ingested.json()["id"]

    # POST /api/assets/upload collides with GET /api/assets/{asset_id}, so FastAPI
    # returns 405 rather than a missing-route 404. Bytes are registered on Desktop.
    assert uploaded.status_code == 405
    assert client.get(f"/api/assets/{asset_id}/bytes").status_code == 200
    assert client.head(f"/api/assets/{asset_id}/bytes").status_code == 200
    assert client.get(f"/api/assets/{asset_id}/thumbnail/bytes").status_code == 200
    assert client.get(f"/api/assets/{asset_id}").status_code == 200

    schema = create_app(handler=test_state).openapi()
    asset_paths = {
        path: sorted(
            method
            for method in methods
            if method in {"get", "post", "put", "delete", "head", "patch"}
        )
        for path, methods in schema["paths"].items()
        if path.startswith("/api/assets")
    }
    assert asset_paths == {
        "/api/assets": ["get", "post"],
        "/api/assets/{asset_id}": ["delete", "get"],
        "/api/assets/{asset_id}/bytes": ["get", "head"],
        "/api/assets/{asset_id}/extract-audio": ["post"],
        "/api/assets/{asset_id}/thumbnail/bytes": ["get", "head"],
        "/api/assets/{asset_id}/trim-audio": ["post"],
        "/api/assets/{asset_id}/trim-video": ["post"],
    }
    operation_ids = [
        operation["operationId"]
        for path_item in schema["paths"].values()
        for method, operation in path_item.items()
        if method in {"get", "post", "put", "delete", "head", "patch"}
        and "operationId" in operation
    ]
    assert len(operation_ids) == len(set(operation_ids))


def test_get_missing_asset_404(client) -> None:
    r = client.get("/api/assets/missing-asset")
    assert_http_error(r, status_code=404, code="ASSET_NOT_FOUND")


def test_get_deleted_asset_404(client, test_state, tmp_path: Path) -> None:
    ingested = client.post(
        "/api/assets", json={"path": str(_png(tmp_path / "gone.png"))}
    )
    assert ingested.status_code == 200
    asset_id = ingested.json()["id"]
    db_path = test_state.config.app_data_dir / "store.sqlite3"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE assets SET deleted_at = 1 WHERE id = ?",
            (asset_id,),
        )
        conn.commit()
    finally:
        conn.close()

    r = client.get(f"/api/assets/{asset_id}")
    assert_http_error(r, status_code=404, code="ASSET_NOT_FOUND")


def test_get_asset_store_unavailable_503(
    test_state, fake_services, default_app_settings
) -> None:
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(fake_services, UnavailableStore()),
    )
    set_state_service_for_tests(handler)
    app = create_app(handler=handler, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        r = client.get("/api/assets/any-asset")
        assert_http_error(r, status_code=503, code="STORE_UNAVAILABLE")
        listed = client.get("/api/assets")
        assert_http_error(listed, status_code=503, code="STORE_UNAVAILABLE")
        deleted = client.delete("/api/assets/any-asset")
        assert_http_error(deleted, status_code=503, code="STORE_UNAVAILABLE")


def test_ingest_missing_file_400(client, tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    r = client.post("/api/assets", json={"path": str(missing)})
    assert_http_error(
        r,
        status_code=400,
        code="FILE_NOT_FOUND",
        message=f"file not found: {missing}",
    )


def test_ingest_unsupported_type_400(client, tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("not media")
    r = client.post("/api/assets", json={"path": str(notes)})
    assert r.status_code == 400
    assert r.json()["code"] == "UNSUPPORTED_MEDIA"


def test_ingest_oversized_image_400(client, tmp_path: Path) -> None:
    source = tmp_path / "huge.png"
    with source.open("wb") as handle:
        handle.truncate(MAX_IMAGE_BYTES + 1)
    r = client.post("/api/assets", json={"path": str(source)})
    assert r.status_code == 400
    assert r.json()["code"] == "FILE_TOO_LARGE"


def test_ingest_decompression_bomb_400(client, tmp_path: Path) -> None:
    original = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 1
    try:
        source = _png(tmp_path / "bomb.png", (16, 16))
        r = client.post("/api/assets", json={"path": str(source)})
        assert r.status_code == 400
        assert r.json()["code"] == "UNREADABLE_MEDIA"
    finally:
        Image.MAX_IMAGE_PIXELS = original


def test_list_generations_filters_feature(client, test_state, tmp_path: Path) -> None:
    assert test_state.generation.try_reserve_generation_start() is True
    try:
        store = _store(test_state)
        png = _png(tmp_path / "a.png")
        asset = client.post("/api/assets", json={"path": str(png)}).json()
        store.insert_generation(
            "text-to-video",
            {
                "params": {"prompt": "a"},
                "inputs": {"startFrame": {"assetId": asset["id"]}},
            },
        )
        store.insert_generation("cozy-felt", {"params": {"prompt": "b"}})
        listed = client.get("/api/generations", params={"feature": "text-to-video"})
        assert listed.status_code == 200
        body = listed.json()
        assert len(body) == 1
        assert body[0]["feature"] == "text-to-video"
        assert body[0]["status"] == "queued"
        assert body[0]["queued_at"] > 0
        assert body[0]["attempt_count"] == 0
        assert body[0]["error_code"] is None
        assert "error" not in body[0]
    finally:
        test_state.generation.release_generation_start_reservation()


def test_get_missing_generation_404(client) -> None:
    r = client.get("/api/generations/not-a-real-id")
    assert r.status_code == 404


def test_http_cancel_missing_generation_404(client) -> None:
    response = client.post("/api/generations/not-a-real-id/cancel")

    assert_http_error(
        response,
        status_code=404,
        code="GENERATION_NOT_FOUND",
    )


def test_http_cancel_terminal_generation_returns_conflict(client, test_state) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        generation = store.insert_generation("text-to-video", _spec("terminal"))
        claimed = store.claim_next_queued()
        assert claimed is not None
        store.mark_failed(
            generation.id,
            "OUTPUT_UNREADABLE",
            attempt_count=claimed.attempt_count,
        )

        response = client.post(f"/api/generations/{generation.id}/cancel")

        assert_http_error(
            response,
            status_code=409,
            code="INVALID_GENERATION_STATUS",
        )


def test_http_cancel_unowned_running_generation_marks_cancelling(
    client, test_state
) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        generation = store.insert_generation("text-to-video", _spec("owned"))
        claimed = store.claim_next_queued()
        assert claimed is not None
        assert claimed.id == generation.id

        response = client.post(f"/api/generations/{generation.id}/cancel")

        assert response.status_code == 200
        assert response.json()["status"] == "cancelling"


def test_delete_generation_keeps_file(client, test_state, tmp_path: Path) -> None:
    assert test_state.generation.try_reserve_generation_start() is True
    try:
        store = _store(test_state)
        png = _png(tmp_path / "z.png")
        asset = client.post("/api/assets", json={"path": str(png)}).json()
        gen = store.insert_generation(
            "text-to-video",
            {"inputs": {"startFrame": {"assetId": asset["id"]}}},
        )
        deleted = client.delete(f"/api/generations/{gen.id}")
        assert deleted.status_code == 200
        assert deleted.json() == {"status": "ok"}
        assert Path(asset["path"]).is_file()
        assert client.get(f"/api/generations/{gen.id}").status_code == 404
    finally:
        test_state.generation.release_generation_start_reservation()


def test_delete_running_generation_returns_conflict(client, test_state) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        generation = store.insert_generation("text-to-video", {})
        claimed = store.claim_next_queued()
        assert claimed is not None
        assert claimed.id == generation.id

        response = client.delete(f"/api/generations/{generation.id}")

        assert response.status_code == 409


def test_missing_output_file_omitted_from_get(client, test_state) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        created = store.insert_generation("text-to-video", {"params": {"prompt": "p"}})
        claimed = store.claim_next_queued()
        assert claimed is not None
        assert claimed.id == created.id
        asset_id, dest = store.allocate_output_path("image", "image/png")
        Image.new("RGB", (8, 8)).save(dest)
        done = store.mark_succeeded(
            created.id,
            [
                OutputSpec(
                    asset_id=asset_id,
                    dest_path=dest,
                    ordinal=0,
                    mime_type="image/png",
                    name="out.png",
                )
            ],
            attempt_count=claimed.attempt_count,
        )
        Path(done.outputs[0].path).unlink()
        r = client.get(f"/api/generations/{created.id}")
        assert r.status_code == 200
        assert r.json()["outputs"] == []


def test_missing_input_file_stays_on_get(client, test_state, tmp_path: Path) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        png = _png(tmp_path / "start.png")
        asset = client.post("/api/assets", json={"path": str(png)}).json()
        gen = store.insert_generation(
            "image-to-video",
            {"inputs": {"startFrame": {"assetId": asset["id"]}}},
        )
        Path(asset["path"]).unlink()
        r = client.get(f"/api/generations/{gen.id}")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "queued"
        assert body["spec"]["inputs"] == {"startFrame": {"assetId": asset["id"]}}
        assert body["outputs"] == []


def test_create_generation_returns_queued(client, test_state) -> None:
    with _reserved_generation_start(test_state):
        response = client.post(
            "/api/generations/text-to-video",
            json=_create_body(),
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "queued"
        assert payload["feature"] == "text-to-video"
        assert payload["spec"]["inputs"] == {}
        assert payload["attempt_count"] == 0
        assert payload["started_at"] is None


def test_create_text_to_video_rejects_unexpected_inputs(client, test_state) -> None:
    response = client.post(
        "/api/generations/text-to-video",
        json={
            "params": {**_params("fox")},
            "inputs": {"startFrame": {"assetId": "missing-asset"}},
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == "HTTP_422"
    assert _store(test_state).list_generations("text-to-video") == []


def test_create_text_to_video_rejects_audio_flag(client, test_state) -> None:
    response = client.post(
        "/api/generations/text-to-video",
        json={"params": {**_params("fox"), "audio": True}},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "HTTP_422"
    assert _store(test_state).list_generations("text-to-video") == []


def test_create_and_retry_wake_after_queued_row_is_visible(client, test_state) -> None:
    assert test_state.generation.try_reserve_generation_start() is True
    try:
        store = _store(test_state)
        recorder = _RecordingQueueControl(store)
        test_state.queued_generations._queue_control = recorder

        failed = client.post(
            "/api/generations/text-to-video",
            json=_create_body(contract_version=2),
        )
        assert failed.status_code == 422
        assert failed.json()["code"] == "INVALID_GENERATION_SPEC"
        assert recorder.queued_visible_on_wake == []
        assert store.has_queued() is False

        created = client.post(
            "/api/generations/text-to-video",
            json=_create_body(),
        )
        assert created.status_code == 200
        generation_id = created.json()["id"]
        assert recorder.queued_visible_on_wake == [True]
        loaded = store.get_generation(generation_id)
        assert loaded is not None
        assert loaded.status == "queued"

        claimed = store.claim_next_queued()
        assert claimed is not None
        store.mark_failed(
            generation_id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count
        )
        assert store.has_queued() is False
        recorder.queued_visible_on_wake.clear()

        retried = client.post(f"/api/generations/{generation_id}/retry")
        assert retried.status_code == 200
        assert recorder.queued_visible_on_wake == [True]
        reloaded = store.get_generation(generation_id)
        assert reloaded is not None
        assert reloaded.status == "queued"
    finally:
        test_state.generation.release_generation_start_reservation()


def test_retry_keeps_same_generation_id(client, test_state) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        generation = store.insert_generation("text-to-video", _spec("retry"))
        claimed = store.claim_next_queued()
        assert claimed is not None
        store.mark_failed(
            generation.id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count
        )

        response = client.post(f"/api/generations/{generation.id}/retry")

        assert response.status_code == 200
        assert response.json()["id"] == generation.id
        assert response.json()["status"] == "queued"


def test_retry_missing_generation_404(client) -> None:
    response = client.post("/api/generations/not-a-real-id/retry")
    assert response.status_code == 404


def test_retry_invalid_status_409(client, test_state) -> None:
    with _reserved_generation_start(test_state):
        store = _store(test_state)
        generation = store.insert_generation("text-to-video", _spec("retry"))
        response = client.post(f"/api/generations/{generation.id}/retry")
        assert response.status_code == 409


def test_lifecycle_routes_are_not_public(client) -> None:
    for path in (
        "/api/assets/allocate",
        "/api/generations/x/succeeded",
        "/api/generations/x/failed",
        "/api/generations/x/cancelled",
    ):
        assert client.post(path, json={}).status_code in {404, 405}


def test_handler_construction_does_not_fail_running_generations(test_state) -> None:
    store = _store(test_state)
    generation = store.insert_generation("text-to-video", {})
    claimed = store.claim_next_queued()
    assert claimed is not None
    QueuedGenerationHandler(
        store=store,
        executor_registry=test_state.executor_registry,
        derive_local_a2v=test_state.video_generation.derive_local_a2v_params,
    )
    loaded = store.get_generation(generation.id)
    assert loaded is not None
    assert loaded.status == "running"


def test_load_persistent_state_fails_running_generations(
    test_state, fake_services, default_app_settings
) -> None:
    store = _store(test_state)
    generation = store.insert_generation("text-to-video", {})
    claimed = store.claim_next_queued()
    assert claimed is not None
    build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(fake_services, store),
    )
    loaded = store.get_generation(generation.id)
    assert loaded is not None
    assert loaded.status == "failed"
    assert loaded.error_code == "INTERRUPTED"


def test_store_unavailable_returns_503_and_other_routes_work(
    test_state, fake_services, default_app_settings
) -> None:
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(fake_services, UnavailableStore()),
    )
    set_state_service_for_tests(handler)
    app = create_app(handler=handler, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        assert handler.generation_queue.is_running is False
        r = client.get("/api/generations", params={"feature": "text-to-video"})
        assert_http_error(r, status_code=503, code="STORE_UNAVAILABLE")
        settings = client.get("/api/settings")
        assert settings.status_code == 200


class _RecoverFailsStore(SqliteStore):
    def fail_running_on_boot(self) -> int:
        raise UnavailableError()


def test_i2v_executor_uses_store_rebound_by_app_handler(
    test_state, fake_services, default_app_settings, tmp_path: Path
) -> None:
    old_store = SqliteStore(tmp_path / "old")
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(fake_services, old_store),
    )
    new_store = SqliteStore(tmp_path / "new")
    start = new_store.ingest_upload(str(_png(tmp_path / "start.png")))
    generation = new_store.insert_generation(
        "image-to-video",
        {
            "params": {**_params("fox"), "aspectRatio": "16:9"},
            "inputs": {"startFrame": {"assetId": start.id}},
        },
    )

    handler._bind_store(new_store)
    executor = handler.executor_registry.get("image-to-video")

    with pytest.raises(
        ValueError, match="text-to-video requires a single 'output' allocation"
    ):
        executor.execute(generation, ())


def test_boot_recovery_fallback_rebinds_i2v_asset_resolver(
    test_state, fake_services, default_app_settings, tmp_path: Path
) -> None:
    store = _RecoverFailsStore(tmp_path / "failing")
    start = store.ingest_upload(str(_png(tmp_path / "start.png")))
    generation = store.insert_generation(
        "image-to-video",
        {
            "params": {**_params("fox"), "aspectRatio": "16:9"},
            "inputs": {"startFrame": {"assetId": start.id}},
        },
    )

    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(fake_services, store),
    )
    executor = handler.executor_registry.get("image-to-video")

    with pytest.raises(UnavailableError):
        executor.execute(generation, ())


def test_boot_recovery_failure_replaces_handler_and_queue_store(
    test_state, fake_services, default_app_settings
) -> None:
    store = _RecoverFailsStore(test_state.config.app_data_dir)
    generation = store.insert_generation("text-to-video", _spec())
    claimed = store.claim_next_queued()
    assert claimed is not None
    handler = build_initial_state(
        test_state.config,
        default_app_settings,
        service_bundle=_bundle(fake_services, store),
    )
    set_state_service_for_tests(handler)
    app = create_app(handler=handler, admin_token=TEST_ADMIN_TOKEN)
    with TestClient(app) as client:
        assert handler.generation_queue.is_running is False
        r = client.get("/api/generations", params={"feature": "text-to-video"})
        assert_http_error(r, status_code=503, code="STORE_UNAVAILABLE")
    loaded = store.get_generation(generation.id)
    assert loaded is not None
    assert loaded.status == "running"


def test_open_store_degrades_when_assets_path_is_a_file(tmp_path: Path) -> None:
    app_data = tmp_path / "blocked_app_data"
    app_data.mkdir()
    (app_data / "assets").write_text("not a directory")
    db = open_store(app_data)
    assert isinstance(db, UnavailableStore)
