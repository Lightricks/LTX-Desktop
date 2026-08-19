"""Integration-style tests for IC-LoRA endpoints."""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

from tests.http_error_assertions import assert_http_error
from tests.fakes import FakeCapture
from tests.conftest import _IC_LORA_MODEL_ID


def _write_ic_lora_file(path: Path) -> None:
    """Header-only safetensors carrying the IC-LoRA reference marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps({"__metadata__": {"reference_downscale_factor": "1"}}).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)


def _install_ic_lora_capable_model(create_fake_model_files, create_fake_ic_lora_files, *, include_depth: bool = True) -> None:
    # Built-in control IC-LoRA is 2.3-only; 2.5 (latest) returns 409 for canny/depth.
    create_fake_model_files(model_id=_IC_LORA_MODEL_ID)
    create_fake_ic_lora_files(include_depth=include_depth)


class TestIcLoraExtractConditioning:
    def test_canny_extraction(self, client, test_state):
        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a"]))

        response = client.post(
            "/api/ic-lora/extract-conditioning",
            json={"video_path": str(video_path), "conditioning_type": "canny", "frame_time": 0},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["conditioning_type"] == "canny"
        assert payload["conditioning"].startswith("data:image/jpeg;base64,")

    def test_depth_extraction(self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files):
        _install_ic_lora_capable_model(create_fake_model_files, create_fake_ic_lora_files)
        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a"]))

        response = client.post(
            "/api/ic-lora/extract-conditioning",
            json={"video_path": str(video_path), "conditioning_type": "depth", "frame_time": 0},
        )
        assert response.status_code == 200
        assert response.json()["conditioning_type"] == "depth"
        assert fake_services.depth_processor_pipeline.apply_calls == ["frame-a"]

    def test_depth_extraction_requires_downloaded_ltx_model(self, client, test_state):
        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a"]))

        response = client.post(
            "/api/ic-lora/extract-conditioning",
            json={"video_path": str(video_path), "conditioning_type": "depth", "frame_time": 0},
        )
        assert_http_error(response, status_code=409, code="NO_DOWNLOADED_LTX_MODEL")


class TestIcLoraGenerate:
    def test_happy_path(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        _install_ic_lora_capable_model(create_fake_model_files, create_fake_ic_lora_files)
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [],
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert Path(response.json()["video_path"]).exists()

    def test_stop_after_generate_unlinks_and_returns_cancelled(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files, caplog
    ):
        # Denoiser Stop after the last step still runs VAE/ffmpeg; video/retake/extend then
        # drop the file. IC-LoRA must do the same instead of importing it as complete.
        _install_ic_lora_capable_model(create_fake_model_files, create_fake_ic_lora_files)
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        pipeline = fake_services.ic_lora_pipeline
        real_generate = pipeline.generate

        def generate_then_cancel(**kwargs):
            real_generate(**kwargs)
            test_state.generation.cancel_generation()

        pipeline.generate = generate_then_cancel  # type: ignore[method-assign]

        caplog.set_level(logging.INFO, logger="handlers.ic_lora_handler")
        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [],
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        written = Path(pipeline.generate_calls[-1]["output_path"])
        assert not written.exists()
        assert any(record.getMessage() == "Generation cancelled by user" for record in caplog.records)

    def test_builtin_control_rejected_on_2_5(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [],
            },
        )
        assert_http_error(
            response,
            status_code=409,
            code="UNSUPPORTED_IC_LORA",
            message=(
                "Built-in control IC-LoRA is not available for the active LTX model. "
                "Switch to an LTX 2.3 local model to use depth/canny control."
            ),
        )

    def test_canny_does_not_require_depth_cp(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        # canny preprocessing uses apply_canny, not the depth processor, so generation
        # must succeed even when the depth cp isn't installed (previously 500'd).
        _install_ic_lora_capable_model(create_fake_model_files, create_fake_ic_lora_files, include_depth=False)
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [],
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"

    def test_local_ic_lora_recoverable_via_progress(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        # IC-LoRA is local-only and drives the generation state machine, so a page that
        # unmounted mid-generation can recover the output via /generation/progress.
        _install_ic_lora_capable_model(create_fake_model_files, create_fake_ic_lora_files)
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video_recover.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [],
            },
        )
        assert response.status_code == 200
        result_path = response.json()["video_path"]

        progress = client.get("/api/generation/progress").json()
        assert progress["status"] == "complete"
        assert progress["result"] == result_path

    def test_custom_uses_supplied_control_video(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        # Custom IC-LoRA: user's own weights + a pre-rendered control video. No
        # preprocessing — the control video is fed to the pipeline directly.
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        lora_ref = test_state.config.default_models_dir / "loras" / "my-custom-ic-lora.safetensors"
        _write_ic_lora_file(lora_ref)
        control_video = test_state.config.outputs_dir / "control.mp4"
        control_video.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(control_video), FakeCapture(frames=["c0", "c1", "c2"], fps=24)
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": "",
                "conditioning_type": "custom",
                "custom_lora_ref": str(lora_ref),
                "control_video_path": str(control_video),
                "prompt": "make it night",
                "images": [],
            },
        )
        assert response.status_code == 200
        result_path = response.json()["video_path"]

        # The pipeline got the supplied control video verbatim — not a derived _control_*.mp4.
        call = fake_services.ic_lora_pipeline.generate_calls[-1]
        assert call["video_conditioning"] == [(str(control_video), 1.0)]
        assert call["num_frames"] == 3

        # Recoverable via the progress endpoint, like every other local generation.
        progress = client.get("/api/generation/progress").json()
        assert progress["status"] == "complete"
        assert progress["result"] == result_path

    def test_custom_requires_lora_and_control_video(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": "",
                "conditioning_type": "custom",
                "prompt": "make it night",
                "images": [],
            },
        )
        assert response.status_code == 400
