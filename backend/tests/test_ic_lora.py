"""Integration-style tests for IC-LoRA endpoints."""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

from tests.http_error_assertions import assert_http_error
from tests.fakes import FakeCapture


def _write_ic_lora_file(path: Path) -> None:
    """Header-only safetensors carrying the IC-LoRA reference marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps({"__metadata__": {"reference_downscale_factor": "1"}}).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)


def _install_ic_lora_capable_model(create_fake_model_files, create_fake_ic_lora_files, *, include_depth: bool = True) -> None:
    # Union Control lives next to the active LTX bundle (2.3 and 2.5 share the 2.3 adapter).
    create_fake_model_files()
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
        assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count == 2

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

    def test_stage_2_on_sizes_to_source_not_768_bucket(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        # Built-in canny defaults to two-stage. 540p must not land in the skip-stage-2 768 bucket.
        _install_ic_lora_capable_model(create_fake_model_files, create_fake_ic_lora_files)
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["frame-a", "frame-b"], width=960, height=540)
        )

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
        call = fake_services.ic_lora_pipeline.generate_calls[-1]
        assert (call["width"], call["height"]) == (896, 512)
        assert call["skip_stage_2"] is False

    def test_skip_stage_2_source_mode_matches_catalog(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        # resolution_factor 0 is the same sentinel on canny/depth as on catalog IC-LoRAs.
        _install_ic_lora_capable_model(create_fake_model_files, create_fake_ic_lora_files)
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["frame-a", "frame-b"], width=960, height=540)
        )

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [],
                "skip_stage_2": True,
                "resolution_factor": 0,
            },
        )
        assert response.status_code == 200
        call = fake_services.ic_lora_pipeline.generate_calls[-1]
        assert (call["width"], call["height"]) == (1920, 1080)
        assert call["resolution_factor"] == 1.0
        assert call["skip_stage_2"] is True

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

    def test_builtin_control_runs_on_2_5(
        self, client, test_state, create_fake_model_files, create_fake_ic_lora_files
    ):
        # 2.5 reuses the 2.3 Union Control adapter; canny must not 409.
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
        # A 3-frame clip snaps up to the 9-frame minimum on the 8k+1 grid.
        assert call["num_frames"] == 9

        # Recoverable via the progress endpoint, like every other local generation.
        progress = client.get("/api/generation/progress").json()
        assert progress["status"] == "complete"
        assert progress["result"] == result_path

    def test_custom_control_video_off_the_frame_grid_is_snapped_down_to_121(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        # 123 frames are off the (n - 1) % 8 grid. The pipeline gets 121, not 123 or 129.
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True
        lora_ref = test_state.config.default_models_dir / "loras" / "my-custom-ic-lora.safetensors"
        _write_ic_lora_file(lora_ref)
        control_video = test_state.config.outputs_dir / "control.mp4"
        control_video.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(control_video), FakeCapture(frames=["c"] * 123, fps=24)
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

        assert response.status_code == 200, response.text
        assert fake_services.ic_lora_pipeline.generate_calls[-1]["num_frames"] == 121

    def test_canny_source_video_off_the_frame_grid_is_snapped_down_to_121(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        _install_ic_lora_capable_model(create_fake_model_files, create_fake_ic_lora_files)
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["f"] * 123))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [],
            },
        )

        assert response.status_code == 200, response.text
        assert fake_services.ic_lora_pipeline.generate_calls[-1]["num_frames"] == 121

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


def test_resolve_settings_keeps_a_tiled_entry_single_stage() -> None:
    from api_types import IcLoraGenerateRequest, IcLoraSettings
    from handlers.ic_lora_handler import _resolve_settings
    from runtime_config.ic_lora_tiling import IcLoraTiling

    tiled = IcLoraSettings(skip_stage_2=True, tiling=IcLoraTiling(long_side=960, short_side=544))
    for override in (None, True, False):
        req = IcLoraGenerateRequest(conditioning_type="custom", prompt="p", skip_stage_2=override)
        resolved = _resolve_settings(req, tiled)
        assert resolved.skip_stage_2 is True
        assert resolved.tiling == tiled.tiling


def test_resolve_settings_still_lets_an_untiled_entry_override_stage_2() -> None:
    from api_types import IcLoraGenerateRequest, IcLoraSettings
    from handlers.ic_lora_handler import _resolve_settings

    untiled = IcLoraSettings(skip_stage_2=True)
    assert _resolve_settings(IcLoraGenerateRequest(conditioning_type="custom", prompt="p", skip_stage_2=False), untiled).skip_stage_2 is False
    assert _resolve_settings(IcLoraGenerateRequest(conditioning_type="custom", prompt="p"), untiled).skip_stage_2 is True
