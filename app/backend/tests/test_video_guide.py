"""Video Guide: guide-clip preprocessing, create rules, and the chained processor download."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from pydantic import ValidationError

from api_types import (
    IcLoraCatalogItem,
    IcLoraControl,
    IcLoraSettings,
    InputSpec,
    ModelCheckpointID,
    PreprocessingStep,
)
from handlers.base import resolve_models_dir
from ic_lora_preprocessing import MediaArtifact, PreprocessingContext, run_preprocessing
from remote.dto import project_remote_spec
from runtime_config.model_download_specs import (
    get_model_cp_spec,
    is_cp_downloaded,
    resolve_catalog_staging_dir,
    resolve_catalog_staging_root,
    resolve_ic_lora_path,
    resolve_model_path,
)
from tests.fakes.services import FakeCapture, FakeVideoProcessor
from tests.http_error_assertions import assert_http_error
from tests.lora_catalog_helpers import add_ic_lora, download_spec

_WEIGHTS = "video-guide.safetensors"
_UNION_CONTROL_CP: ModelCheckpointID = "ltx-2.3-22b-ic-lora-union-control-ref0.5"
_PROCESSOR_CPS: list[ModelCheckpointID] = [
    "dpt-hybrid-midas",
    "yolox-l-torchscript",
    "dw-ll-ucoco-384-bs5",
]


def _install_video_guide(fake_services, **overrides: object) -> None:
    fields: dict[str, object] = dict(
        id="video-guide",
        download=download_spec(_WEIGHTS),
        allows_reference_image=True,
        reference_image_frame=0,
        controls=[
            IcLoraControl(
                id="control_type",
                label="Control mode",
                kind="select",
                default="pose",
                options=["pose", "depth", "canny"],
            )
        ],
        preprocessing=[PreprocessingStep(utility="video_control", params={})],
        requires_checkpoints=_PROCESSOR_CPS,
        default_settings=IcLoraSettings(skip_stage_2=True, audio_mode="source"),
    )
    fields.update(overrides)
    add_ic_lora(fake_services, **fields)


def _write_weights(test_state, ic_lora_id: str = "video-guide", filename: str = _WEIGHTS) -> None:
    path = resolve_ic_lora_path(resolve_models_dir(test_state.state, test_state.config), ic_lora_id, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weights")


def _write_processor_models(test_state, cps: list[ModelCheckpointID] = _PROCESSOR_CPS) -> None:
    models_dir = resolve_models_dir(test_state.state, test_state.config)
    for cp in cps:
        path = resolve_model_path(models_dir, cp)
        if get_model_cp_spec(cp).is_folder:
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.safetensors").write_bytes(b"model")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model")


def _create(client, tmp_path: Path, recipe_id: str, **params: object):
    writer = imageio.get_writer(str(tmp_path / "clip.mp4"), fps=8, codec="libx264", macro_block_size=None)
    for _ in range(8):
        writer.append_data(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.close()
    video = client.post("/api/assets", json={"path": str(tmp_path / "clip.mp4")}).json()
    return client.post(
        f"/api/generations/ic-lora-recipes/{recipe_id}",
        json={
            "params": {"prompt": "a dancer", "model": "ltx-2.5-fast", "variantId": "default", **params},
            "inputs": {"video": {"assetId": video["id"]}},
        },
    )


def test_video_control_writes_one_guide_frame_per_source_frame_at_the_source_rate(tmp_path: Path) -> None:
    source = str(tmp_path / "source.mp4")
    video_processor = FakeVideoProcessor()
    video_processor.register_video(source, FakeCapture(["a", "b", "c"], fps=12))
    control = run_preprocessing(
        [PreprocessingStep(utility="video_control", params={})],
        MediaArtifact(path=source, kind="video"),
        PreprocessingContext(
            num_frames=3,
            outputs_dir=tmp_path,
            video_processor=video_processor,
            control_frame=lambda frame: f"guide:{frame}",
        ),
    )
    assert video_processor.writers[0].frames == ["guide:a", "guide:b", "guide:c"]
    assert control.fps == 12
    # No frame_count: the handler then keeps the whole source clip length.
    assert control.frame_count is None


def test_create_stores_the_catalog_default_control_type_when_the_body_omits_it(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_video_guide(fake_services)
    _write_weights(test_state)
    _write_processor_models(test_state)
    response = _create(client, tmp_path, "video-guide")
    assert response.status_code == 200, response.text
    assert response.json()["spec"]["params"]["controlType"] == "pose"


def test_create_refuses_a_control_type_for_a_recipe_without_the_control(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    add_ic_lora(
        fake_services,
        id="day-to-night",
        download=download_spec("day-to-night.safetensors"),
        default_settings=IcLoraSettings(skip_stage_2=True, audio_mode="source"),
    )
    _write_weights(test_state, "day-to-night", "day-to-night.safetensors")
    response = _create(client, tmp_path, "day-to-night", controlType="depth")
    assert_http_error(
        response,
        status_code=422,
        code="INVALID_GENERATION_SPEC",
        message="'day-to-night' takes no control type.",
    )


def test_create_refuses_while_a_processor_model_is_missing_though_the_lora_is_installed(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_video_guide(fake_services)
    _write_weights(test_state)
    _write_processor_models(test_state, ["dpt-hybrid-midas", "yolox-l-torchscript"])
    response = _create(client, tmp_path, "video-guide")
    assert_http_error(
        response,
        status_code=409,
        code="IC_LORA_NOT_DOWNLOADED",
        message="IC_LORA_NOT_DOWNLOADED",
    )


def test_list_reports_not_downloaded_until_every_processor_model_is_on_disk(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files()
    _install_video_guide(fake_services)
    _write_weights(test_state)

    def entry() -> dict[str, object]:
        items = client.get("/api/ic-loras").json()["ic_loras"]
        return next(item for item in items if item["ic_lora"]["id"] == "video-guide")

    assert entry()["downloaded"] is False
    assert entry()["downloaded_variant_ids"] == []
    _write_processor_models(test_state)
    assert entry()["downloaded"] is True
    assert entry()["downloaded_variant_ids"] == ["default"]


def test_download_installs_the_lora_and_every_missing_processor_model(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files()
    _install_video_guide(fake_services)
    models_dir = resolve_models_dir(test_state.state, test_state.config)
    # One processor model is already installed. The download must not fetch it again.
    _write_processor_models(test_state, ["yolox-l-torchscript"])
    start = client.post("/api/ic-loras/download", json={"ic_lora_id": "video-guide"})
    assert start.status_code == 200, start.text
    progress = client.get("/api/ic-loras/download/progress", params={"sessionId": start.json()["sessionId"]})
    assert progress.json()["status"] == "complete"
    assert resolve_ic_lora_path(models_dir, "video-guide", _WEIGHTS).exists()
    assert all(is_cp_downloaded(models_dir, cp) for cp in _PROCESSOR_CPS)
    # The fake downloader writes zero bytes. The installed model keeps its own bytes.
    assert resolve_model_path(models_dir, "yolox-l-torchscript").read_bytes() == b"model"


def test_catalog_rejects_a_video_control_step_without_the_three_control_modes() -> None:
    with pytest.raises(ValidationError):
        IcLoraCatalogItem(
            id="video-guide",
            name="Video Guide",
            description="d",
            download=download_spec(_WEIGHTS),
            requires_hf_login=False,
            input=InputSpec(kind="video"),
            preprocessing=[PreprocessingStep(utility="video_control", params={})],
            controls=[
                IcLoraControl(
                    id="control_type",
                    label="Control mode",
                    kind="select",
                    default="pose",
                    options=["pose", "depth"],
                )
            ],
        )


def _guide_clips(test_state) -> list[Path]:
    return sorted(test_state.config.outputs_dir.glob("_video_control_*"))


def _generate(client, test_state, source: Path, ic_lora_id: str = "video-guide", **body: object):
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"clip")
    test_state.video_processor.register_video(str(source), FakeCapture(frames=["a", "b"]))
    return client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": ic_lora_id,
            "input_path": str(source),
            "control_values": {"control_type": "canny"},
            "prompt": "a dancer",
            "conditioning_type": "custom",
            **body,
        },
    )


@pytest.mark.parametrize(
    ("exit_kind", "status_code"), [("success", 200), ("cancelled", 200), ("inference_error", 500)]
)
def test_guide_clip_is_removed_after_success_cancel_and_inference_error(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path, exit_kind: str, status_code: int
) -> None:
    create_fake_model_files()
    _install_video_guide(fake_services)
    _write_weights(test_state)
    _write_processor_models(test_state)
    pipeline = fake_services.ic_lora_pipeline
    real_generate = pipeline.generate
    clip_existed: list[bool] = []

    def generate(**kwargs):
        clip_existed.append(Path(kwargs["video_conditioning"][0][0]).exists())
        if exit_kind == "inference_error":
            raise RuntimeError("out of memory")
        real_generate(**kwargs)
        if exit_kind == "cancelled":
            test_state.generation.cancel_generation()

    pipeline.generate = generate  # type: ignore[method-assign]
    response = _generate(client, test_state, tmp_path / "in" / "source.mp4")
    assert response.status_code == status_code, response.text
    assert clip_existed == [True]
    assert _guide_clips(test_state) == []


def test_a_run_without_preprocessing_keeps_the_source_clip_inside_the_outputs_folder(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files()
    add_ic_lora(
        fake_services,
        id="plain-video",
        download=download_spec("plain-video.safetensors"),
        default_settings=IcLoraSettings(skip_stage_2=True, audio_mode="source"),
    )
    _write_weights(test_state, "plain-video", "plain-video.safetensors")
    source = test_state.config.outputs_dir / "source.mp4"
    response = _generate(client, test_state, source, "plain-video", control_values={})
    assert response.status_code == 200, response.text
    assert source.exists()


def test_video_control_deletes_its_partial_clip_when_the_run_stops_midway(tmp_path: Path) -> None:
    source = str(tmp_path / "source.mp4")
    video_processor = FakeVideoProcessor()
    video_processor.register_video(source, FakeCapture(["a", "b", "c"]))
    seen: list[int] = []

    def stop_on_second_frame() -> None:
        seen.append(1)
        if len(seen) == 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError):
        run_preprocessing(
            [PreprocessingStep(utility="video_control", params={})],
            MediaArtifact(path=source, kind="video"),
            PreprocessingContext(
                num_frames=3,
                outputs_dir=tmp_path,
                video_processor=video_processor,
                control_frame=lambda frame: frame,
                on_frame=stop_on_second_frame,
            ),
        )
    assert list(tmp_path.glob("_video_control_*")) == []


def test_video_control_keeps_the_finished_clip(tmp_path: Path) -> None:
    source = str(tmp_path / "source.mp4")
    video_processor = FakeVideoProcessor()
    video_processor.register_video(source, FakeCapture(["a", "b"]))
    control = run_preprocessing(
        [PreprocessingStep(utility="video_control", params={})],
        MediaArtifact(path=source, kind="video"),
        PreprocessingContext(
            num_frames=2,
            outputs_dir=tmp_path,
            video_processor=video_processor,
            control_frame=lambda frame: frame,
        ),
    )
    assert Path(control.path).exists()


def test_generate_refuses_while_a_processor_model_is_missing_though_the_lora_is_installed(
    client, test_state, fake_services, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    _install_video_guide(fake_services)
    _write_weights(test_state)
    _write_processor_models(test_state, ["dpt-hybrid-midas", "yolox-l-torchscript"])
    response = _generate(client, test_state, tmp_path / "source.mp4")
    assert_http_error(
        response,
        status_code=409,
        code="IC_LORA_NOT_DOWNLOADED",
        message="IC_LORA_NOT_DOWNLOADED",
    )


def test_model_download_cleanup_leaves_the_staging_folder_of_a_catalog_download(test_state) -> None:
    models_dir = resolve_models_dir(test_state.state, test_state.config)
    staged = resolve_catalog_staging_dir(models_dir, "session-a") / "partial.bin"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"partial")
    test_state.downloads.cleanup_downloading_dir()
    assert staged.exists()


def test_failed_download_removes_its_staging_folder(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files()
    _install_video_guide(fake_services)
    models_dir = resolve_models_dir(test_state.state, test_state.config)
    fake_services.model_downloader.fail_next = RuntimeError("network error")
    start = client.post("/api/ic-loras/download", json={"ic_lora_id": "video-guide"})
    progress = client.get("/api/ic-loras/download/progress", params={"sessionId": start.json()["sessionId"]})
    assert progress.json()["status"] == "error"
    assert list(resolve_catalog_staging_root(models_dir).glob("*")) == []


def test_successful_download_removes_its_staging_folder(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files()
    _install_video_guide(fake_services)
    models_dir = resolve_models_dir(test_state.state, test_state.config)
    start = client.post("/api/ic-loras/download", json={"ic_lora_id": "video-guide"})
    progress = client.get("/api/ic-loras/download/progress", params={"sessionId": start.json()["sessionId"]})
    assert progress.json()["status"] == "complete"
    assert list(resolve_catalog_staging_root(models_dir).glob("*")) == []


def _write_union_control_checkpoint(test_state) -> Path:
    path = resolve_model_path(resolve_models_dir(test_state.state, test_state.config), _UNION_CONTROL_CP)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"union")
    return path


def _video_guide_entry(client) -> dict[str, object]:
    items = client.get("/api/ic-loras").json()["ic_loras"]
    return next(item for item in items if item["ic_lora"]["id"] == "video-guide")


def test_download_fetches_its_own_weights_though_the_genspace_union_checkpoint_is_on_disk(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files()
    _install_video_guide(fake_services)
    checkpoint = _write_union_control_checkpoint(test_state)
    _write_processor_models(test_state)
    assert _video_guide_entry(client)["downloaded"] is False
    start = client.post("/api/ic-loras/download", json={"ic_lora_id": "video-guide"})
    progress = client.get("/api/ic-loras/download/progress", params={"sessionId": start.json()["sessionId"]})
    assert progress.json()["status"] == "complete"
    models_dir = resolve_models_dir(test_state.state, test_state.config)
    assert resolve_ic_lora_path(models_dir, "video-guide", _WEIGHTS).exists()
    assert checkpoint.read_bytes() == b"union"


def test_delete_removes_only_the_video_guide_file_and_keeps_the_genspace_union_checkpoint(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files()
    _install_video_guide(fake_services)
    _write_weights(test_state)
    checkpoint = _write_union_control_checkpoint(test_state)
    response = client.request("DELETE", "/api/ic-loras/installation", json={"ic_lora_id": "video-guide"})
    assert response.status_code == 200
    models_dir = resolve_models_dir(test_state.state, test_state.config)
    assert not resolve_ic_lora_path(models_dir, "video-guide", _WEIGHTS).exists()
    assert checkpoint.read_bytes() == b"union"


def test_deleting_the_genspace_union_checkpoint_leaves_video_guide_installed(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files()
    _install_video_guide(fake_services)
    _write_weights(test_state)
    _write_processor_models(test_state)
    _write_union_control_checkpoint(test_state).unlink()
    assert _video_guide_entry(client)["downloaded"] is True


def test_delete_then_download_restores_the_weights_and_the_entry_is_installed_again(
    client, test_state, fake_services, create_fake_model_files
) -> None:
    create_fake_model_files()
    _install_video_guide(fake_services)
    _write_weights(test_state)
    _write_processor_models(test_state)
    client.request("DELETE", "/api/ic-loras/installation", json={"ic_lora_id": "video-guide"})
    start = client.post("/api/ic-loras/download", json={"ic_lora_id": "video-guide"})
    progress = client.get("/api/ic-loras/download/progress", params={"sessionId": start.json()["sessionId"]})
    assert progress.json()["status"] == "complete"
    models_dir = resolve_models_dir(test_state.state, test_state.config)
    assert resolve_ic_lora_path(models_dir, "video-guide", _WEIGHTS).exists()
    assert _video_guide_entry(client)["downloaded"] is True


def test_remote_projection_keeps_a_valid_control_type_and_drops_an_unknown_one() -> None:
    assert project_remote_spec({"params": {"controlType": "depth"}, "inputs": {}}).params.controlType == "depth"
    assert project_remote_spec({"params": {"controlType": "sketch"}, "inputs": {}}).params.controlType is None
