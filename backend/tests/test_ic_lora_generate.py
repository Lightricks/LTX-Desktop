import logging
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from frame_math import compute_num_frames
from runtime_config.ic_lora_local_envelope import IC_LORA_SOURCE_TOO_LARGE, IC_LORA_V1_ENVELOPE_MESSAGE
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    VIDEO_JOB_TOO_LARGE,
    VIDEO_JOB_TOO_LARGE_MESSAGE,
)
from tests.fakes import FakeCapture
from tests.http_error_assertions import assert_http_error


def _write_png(path: Path, size: tuple[int, int] = (1, 1)) -> Path:
    Image.new("RGB", size, color=(1, 2, 3)).save(path, format="PNG")
    return path


def test_ic_lora_generate_runs_inference(client: TestClient, tmp_path: Path, create_fake_model_files, fake_services):
    create_fake_model_files()  # base LTX model + upscale + text encoder for load_ic_lora
    # Download the fake IC-LoRA weights so the handler finds them.
    start = client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    assert start.status_code == 200
    img = tmp_path / "in.png"
    _write_png(img)
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(img),
            "control_values": {"duration": 5},
            "prompt": "a slow orbit",
            "conditioning_type": "custom",  # ignored in IC-LoRA mode
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "complete"
    # 5s at the IC-LoRA's 24fps maps to 121 frames ((5*24)//8*8 + 1).
    assert fake_services.ic_lora_pipeline.generate_calls[-1]["num_frames"] == 121


def test_ic_lora_generate_rejects_when_required_ltx_family_missing(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services
):
    create_fake_model_files(model_id="ltx-2.5-22b-distilled")
    catalog = fake_services.lora_catalog_provider._catalog
    fake_services.lora_catalog_provider._catalog = catalog.model_copy(
        update={
            "ic_loras": [
                catalog.ic_loras[0].model_copy(update={"supported_models": ["LTX-2.3"]}),
                *catalog.ic_loras[1:],
            ]
        }
    )
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    img = tmp_path / "in.png"
    _write_png(img)
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(img),
            "prompt": "a slow orbit",
            "conditioning_type": "custom",
        },
    )
    assert_http_error(resp, status_code=422, code="IC_LORA_UNSUPPORTED_MODEL")


def test_catalog_stop_after_generate_unlinks_and_returns_cancelled(
    client: TestClient, tmp_path: Path, test_state, create_fake_model_files, fake_services, caplog
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    img = tmp_path / "in.png"
    _write_png(img)

    pipeline = fake_services.ic_lora_pipeline
    real_generate = pipeline.generate

    def generate_then_cancel(**kwargs):
        real_generate(**kwargs)
        test_state.generation.cancel_generation()

    pipeline.generate = generate_then_cancel  # type: ignore[method-assign]

    caplog.set_level(logging.INFO, logger="handlers.ic_lora_handler")
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(img),
            "control_values": {"duration": 5},
            "prompt": "a slow orbit",
            "conditioning_type": "custom",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    written = Path(pipeline.generate_calls[-1]["output_path"])
    assert not written.exists()
    assert any(record.getMessage() == "Generation cancelled by user" for record in caplog.records)


def test_ic_lora_generate_uses_ic_lora_default_when_no_override(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    img = tmp_path / "in.png"
    _write_png(img)
    resp = client.post(
        "/api/ic-lora/generate",
        json={"ic_lora_id": "ingredients-v1", "input_path": str(img), "prompt": "p", "conditioning_type": "custom"},
    )
    assert resp.status_code == 200
    # Fake IC-LoRA's default_settings has skip_stage_2=True; no override sent -> respected.
    assert fake_services.ic_lora_pipeline.generate_calls[-1]["skip_stage_2"] is True


def test_ic_lora_generate_respects_skip_stage_2_override(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    img = tmp_path / "in.png"
    _write_png(img)
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(img),
            "prompt": "p",
            "conditioning_type": "custom",
            "skip_stage_2": False,  # override beats the IC-LoRA default of True
        },
    )
    assert resp.status_code == 200
    assert fake_services.ic_lora_pipeline.generate_calls[-1]["skip_stage_2"] is False


def test_video_ic_lora_matches_source_length_not_duration(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services, test_state
):
    # A video-input transform IC-LoRA must output the source clip's length, not the
    # duration control's default — otherwise an 8s clip rendered to 5s freezes on the
    # last frame while audio runs on.
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(str(src), FakeCapture(frames=["f"] * 50, fps=24))
    resp = client.post(
        "/api/ic-lora/generate",
        json={"ic_lora_id": "colorize-v1", "input_path": str(src), "prompt": "p", "conditioning_type": "custom"},
    )
    assert resp.status_code == 200
    # 50 source frames snap to the (n-1)%8==0 grid -> 49, not 121 (the 5s default).
    assert fake_services.ic_lora_pipeline.generate_calls[-1]["num_frames"] == 49


def test_ic_lora_resolution_factor_zero_matches_source(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services, test_state
):
    # resolution_factor 0 = "source dimensions": the handler sizes the output to the source
    # (2*source, rendered at factor 1.0 since skip_stage_2 halves) instead of the 768 bucket.
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * 50, fps=24, width=960, height=540)
    )
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "skip_stage_2": True,
            "resolution_factor": 0,
        },
    )
    assert resp.status_code == 200
    call = fake_services.ic_lora_pipeline.generate_calls[-1]
    assert (call["width"], call["height"]) == (1920, 1080)  # 2 * source
    assert call["resolution_factor"] == 1.0  # multiplier ignored in source mode


def test_ic_lora_stage_2_on_sizes_to_source_not_768_bucket(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services, test_state
):
    # Two-stage output is the passed canvas (stage 1 runs at half). A 540p source must
    # not fall through to the skip-stage-2 768-wide bucket (768x384).
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * 50, fps=24, width=960, height=540)
    )
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "skip_stage_2": False,
        },
    )
    assert resp.status_code == 200
    call = fake_services.ic_lora_pipeline.generate_calls[-1]
    assert (call["width"], call["height"]) == (896, 512)  # source snapped down to /128
    assert call["skip_stage_2"] is False


def test_ic_lora_stage_2_on_caps_explicit_resolution_to_source(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services, test_state
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * 50, fps=24, width=960, height=540)
    )
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "skip_stage_2": False,
            "resolution": {"width": 512, "height": 256},
        },
    )
    assert resp.status_code == 200
    call = fake_services.ic_lora_pipeline.generate_calls[-1]
    assert (call["width"], call["height"]) == (512, 256)


def test_ic_lora_rejects_image_for_video_input(
    client: TestClient, tmp_path: Path, create_fake_model_files
):
    # colorize-v1 expects a video input; feeding an image must fail with a clean 400
    # (not a deep pipeline 500). Fires before the download check; still needs an LTX
    # family on disk so the catalog visibility gate doesn't 422 first.
    create_fake_model_files()
    img = tmp_path / "in.png"
    _write_png(img)
    resp = client.post(
        "/api/ic-lora/generate",
        json={"ic_lora_id": "colorize-v1", "input_path": str(img), "prompt": "p", "conditioning_type": "custom"},
    )
    assert resp.status_code == 400


def test_ic_lora_rejects_video_for_image_input(
    client: TestClient, tmp_path: Path, create_fake_model_files
):
    # ingredients-v1 expects an image input; feeding a video must 400.
    create_fake_model_files()
    clip = tmp_path / "in.mp4"
    clip.write_bytes(b"\x00" * 10)
    resp = client.post(
        "/api/ic-lora/generate",
        json={"ic_lora_id": "ingredients-v1", "input_path": str(clip), "prompt": "p", "conditioning_type": "custom"},
    )
    assert resp.status_code == 400


def test_ic_lora_generate_unknown_ic_lora_404(client: TestClient, tmp_path: Path):
    img = tmp_path / "in.png"
    img.write_bytes(b"x")
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "does-not-exist",
            "input_path": str(img),
            "control_values": {"duration": 5},
            "prompt": "p",
            "conditioning_type": "custom",
        },
    )
    assert resp.status_code == 404


def test_outpaint_ic_lora_threads_mask_to_pipeline(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "outpaint-v1"})
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00")
    # Real numpy frames so outpaint_canvas can composite onto the larger canvas.
    fake_services.video_processor.videos[str(src)] = FakeCapture(
        frames=[np.full((96, 96, 3), 0x40, dtype=np.uint8) for _ in range(9)],
        width=96,
        height=96,
        fps=24,
    )
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "outpaint-v1",
            "input_path": str(src),
            "prompt": "extend the scene",
            "conditioning_type": "custom",
            "outpaint_pads": {"left": 48, "right": 0, "top": 0, "bottom": 0},
        },
    )
    assert resp.status_code == 200, resp.text
    assert fake_services.ic_lora_pipeline.generate_calls[-1]["conditioning_mask_path"] is not None


def test_outpaint_ic_lora_rejects_negative_pad(
    client: TestClient, tmp_path: Path, create_fake_model_files
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "outpaint-v1"})
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00")
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "outpaint-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "outpaint_pads": {"left": -1, "right": 0, "top": 0, "bottom": 0},
        },
    )
    assert resp.status_code == 422  # DTO ge=0 rejects negative pads


def test_outpaint_runs_with_empty_prompt(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services
):
    # outpaint-v1 sets allows_empty_prompt → a blank prompt is accepted.
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "outpaint-v1"})
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00")
    fake_services.video_processor.videos[str(src)] = FakeCapture(
        frames=[np.full((96, 96, 3), 0x40, dtype=np.uint8) for _ in range(9)],
        width=96,
        height=96,
        fps=24,
    )
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "outpaint-v1",
            "input_path": str(src),
            "prompt": "",
            "conditioning_type": "custom",
            "outpaint_pads": {"left": 48, "right": 0, "top": 0, "bottom": 0},
        },
    )
    assert resp.status_code == 200, resp.text


def test_ic_lora_empty_prompt_rejected_without_flag(
    client: TestClient, tmp_path: Path, create_fake_model_files
):
    # colorize-v1 has no allows_empty_prompt → a blank prompt is rejected.
    create_fake_model_files()
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00")
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "   ",
            "conditioning_type": "custom",
        },
    )
    assert resp.status_code == 400


def test_control_value_out_of_options_rejected(
    client: TestClient, tmp_path: Path, create_fake_model_files
):
    # Generic control validation: an int control value outside the entry's options is rejected.
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    src = tmp_path / "in.png"
    src.write_bytes(b"\x00")
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "control_values": {"duration": 999},
        },
    )
    assert resp.status_code == 400


def test_reference_image_forwarded_when_allowed(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services
):
    # refimg-v1 sets allows_reference_image → the request's images reach the pipeline.
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "refimg-v1"})
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00")
    ref = tmp_path / "ref.png"
    _write_png(ref)
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "refimg-v1",
            "input_path": str(src),
            "prompt": "3DREAL a harbor",
            "conditioning_type": "custom",
            "images": [{"path": str(ref), "frame": 0, "strength": 1.0}],
        },
    )
    assert resp.status_code == 200, resp.text
    images = fake_services.ic_lora_pipeline.generate_calls[-1]["images"]
    assert len(images) == 1
    assert images[0].path == str(ref)
    assert images[0].frame_idx == 0


def test_reference_image_missing_path_rejected(
    client: TestClient, tmp_path: Path, create_fake_model_files
):
    # An allows_reference_image entry rejects a non-existent reference path with a clean 400.
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "refimg-v1"})
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00")
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "refimg-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "images": [{"path": str(tmp_path / "nope.png"), "frame": 0, "strength": 1.0}],
        },
    )
    assert resp.status_code == 400


def test_reference_image_ignored_when_not_allowed(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services
):
    # colorize-v1 does not opt in → images are dropped even if sent.
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00")
    ref = tmp_path / "ref.png"
    _write_png(ref)
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "images": [{"path": str(ref), "frame": 0, "strength": 1.0}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert fake_services.ic_lora_pipeline.generate_calls[-1]["images"] == []


def test_request_cannot_pin_a_reference_image_at_the_negative_still_frame(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services
):
    # Only a catalog entry pins the look still at -1. A request frame of -1 is a 422.
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "refimg-v1"})
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00")
    ref = tmp_path / "ref.png"
    _write_png(ref)
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "refimg-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "images": [{"path": str(ref), "frame": -1, "strength": 1.0}],
        },
    )
    assert resp.status_code == 422
    assert fake_services.ic_lora_pipeline.generate_calls == []


def test_canny_empty_prompt_rejected(client: TestClient, tmp_path: Path):
    # The non-catalog canny path has no entry to opt into promptless runs.
    src = tmp_path / "v.mp4"
    src.write_bytes(b"\x00")
    resp = client.post(
        "/api/ic-lora/generate",
        json={"video_path": str(src), "conditioning_type": "canny", "prompt": ""},
    )
    assert resp.status_code == 400


def test_generate_with_variant_id_uses_that_checkpoint(
    client: TestClient, tmp_path: Path, create_fake_model_files
):
    create_fake_model_files()
    assert client.post(
        "/api/ic-loras/download", json={"ic_lora_id": "multi-v1", "variant_id": "light"}
    ).status_code == 200
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00")
    ok = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "multi-v1",
            "variant_id": "light",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
        },
    )
    assert ok.status_code == 200, ok.text
    missing = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "multi-v1",
            "variant_id": "strong",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
        },
    )
    assert missing.status_code == 409
    unknown = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "multi-v1",
            "variant_id": "nope",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
        },
    )
    assert unknown.status_code == 404


def test_generate_omits_variant_id_uses_any_installed_checkpoint(
    client: TestClient, tmp_path: Path, create_fake_model_files
):
    # Only the non-default (light) file is on disk; omit variant_id → still finds it.
    create_fake_model_files()
    assert client.post(
        "/api/ic-loras/download", json={"ic_lora_id": "multi-v1", "variant_id": "light"}
    ).status_code == 200
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00")
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "multi-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
        },
    )
    assert resp.status_code == 200, resp.text


def test_download_unknown_variant_id_404(client: TestClient):
    resp = client.post(
        "/api/ic-loras/download",
        json={"ic_lora_id": "multi-v1", "variant_id": "does-not-exist"},
    )
    assert resp.status_code == 404


def test_list_reports_downloaded_variant_ids_after_partial_download(
    client: TestClient, create_fake_model_files
):
    create_fake_model_files()
    before = client.get("/api/ic-loras").json()["ic_loras"]
    multi = next(i for i in before if i["ic_lora"]["id"] == "multi-v1")
    assert multi["downloaded"] is False
    assert multi["downloaded_variant_ids"] == []

    assert client.post(
        "/api/ic-loras/download", json={"ic_lora_id": "multi-v1", "variant_id": "light"}
    ).status_code == 200

    after = client.get("/api/ic-loras").json()["ic_loras"]
    multi = next(i for i in after if i["ic_lora"]["id"] == "multi-v1")
    assert multi["downloaded"] is True
    assert multi["downloaded_variant_ids"] == ["light"]
    # Default (strong) still missing.
    assert "strong" not in multi["downloaded_variant_ids"]


def test_ic_lora_generate_uses_process_loading_mode(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services, test_state
):
    create_fake_model_files()
    assert test_state.config.local_generations_mode == "full_models_loading"
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    img = tmp_path / "in.png"
    _write_png(img)
    resp = client.post(
        "/api/ic-lora/generate",
        json={"ic_lora_id": "ingredients-v1", "input_path": str(img), "prompt": "p", "conditioning_type": "custom"},
    )
    assert resp.status_code == 200
    assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count is None
    assert test_state.config.local_generations_mode == "full_models_loading"


def test_ic_lora_10s_540p_streams(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services, test_state
):
    create_fake_model_files()
    assert test_state.config.local_generations_mode == "full_models_loading"
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * compute_num_frames(10, 24), fps=24, width=960, height=540)
    )
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "skip_stage_2": True,
            "resolution_factor": 0,
        },
    )
    assert resp.status_code == 200
    assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count == 2
    assert test_state.config.local_generations_mode == "full_models_loading"


def test_ic_lora_rejects_2160p_source_dimensions(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * 50, fps=24, width=3840, height=2160)
    )
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "skip_stage_2": True,
            "resolution_factor": 0,
        },
    )
    assert_http_error(
        resp,
        status_code=422,
        code=IC_LORA_SOURCE_TOO_LARGE,
        message=IC_LORA_V1_ENVELOPE_MESSAGE,
    )


def test_ic_lora_streams_longer_than_20s_540p(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services, test_state
):
    create_fake_model_files()
    assert test_state.config.local_generations_mode == "full_models_loading"
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * 521, fps=24, width=960, height=540)
    )
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "skip_stage_2": True,
            "resolution_factor": 0,
        },
    )
    assert resp.status_code == 200
    assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count == 2
    assert test_state.config.local_generations_mode == "full_models_loading"


def test_ic_lora_accepts_snapped_20s_540p(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services, test_state
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    frames = compute_num_frames(20, 24)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * frames, fps=24, width=960, height=540)
    )
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "skip_stage_2": True,
            "resolution_factor": 0,
        },
    )
    assert resp.status_code == 200
    assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count == 2
    assert test_state.config.local_generations_mode == "full_models_loading"


def test_ic_lora_rebuilds_pipeline_when_job_switches_to_stream(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services, test_state
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    short = tmp_path / "short.mp4"
    long = tmp_path / "long.mp4"
    short.write_bytes(b"\x00" * 100)
    long.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(
        str(short), FakeCapture(frames=["f"] * compute_num_frames(5, 24), fps=24, width=960, height=540)
    )
    test_state.video_processor.register_video(
        str(long), FakeCapture(frames=["f"] * compute_num_frames(20, 24), fps=24, width=960, height=540)
    )
    payload = {
        "ic_lora_id": "colorize-v1",
        "prompt": "p",
        "conditioning_type": "custom",
        "skip_stage_2": True,
        "resolution_factor": 0,
    }
    assert client.post("/api/ic-lora/generate", json={**payload, "input_path": str(short)}).status_code == 200
    assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count is None
    first_creates = fake_services.ic_lora_pipeline.create_count
    assert client.post("/api/ic-lora/generate", json={**payload, "input_path": str(long)}).status_code == 200
    assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count == 2
    assert fake_services.ic_lora_pipeline.create_count == first_creates + 1


def test_ic_lora_list_filters_duration_options_on_small_vram(
    client: TestClient, test_state, create_fake_model_files
):
    create_fake_model_files()
    listed = client.get("/api/ic-loras").json()["ic_loras"]
    ingredients = next(i for i in listed if i["ic_lora"]["id"] == "ingredients-v1")
    duration = next(c for c in ingredients["ic_lora"]["controls"] if c["id"] == "duration")
    assert duration["options"] == [5, 8, 20]

    test_state.config.vram_gb = 10
    listed = client.get("/api/ic-loras").json()["ic_loras"]
    ingredients = next(i for i in listed if i["ic_lora"]["id"] == "ingredients-v1")
    duration = next(c for c in ingredients["ic_lora"]["controls"] if c["id"] == "duration")
    assert 20 not in duration["options"]
    assert 5 in duration["options"]
    assert duration["default"] in duration["options"]


def test_ic_lora_list_survives_junk_catalog_fps(
    client: TestClient, fake_services, create_fake_model_files
):
    create_fake_model_files()
    from api_types import PreprocessingStep

    catalog = fake_services.lora_catalog_provider._catalog
    updated = []
    for item in catalog.ic_loras:
        if item.id == "ingredients-v1":
            updated.append(
                item.model_copy(
                    update={
                        "preprocessing": [
                            PreprocessingStep(utility="image_to_frames", params={"fps": None})
                        ]
                    }
                )
            )
        else:
            updated.append(item)
    fake_services.lora_catalog_provider._catalog = catalog.model_copy(update={"ic_loras": updated})
    resp = client.get("/api/ic-loras")
    assert resp.status_code == 200
    assert any(row["ic_lora"]["id"] == "ingredients-v1" for row in resp.json()["ic_loras"])


def test_ic_lora_stream_over_540p_returns_job_too_large(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state
):
    create_fake_model_files()
    test_state.config.vram_gb = 8
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * compute_num_frames(20, 24), fps=24, width=960, height=540)
    )
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "skip_stage_2": True,
            "resolution_factor": 0,
        },
    )
    assert_http_error(
        resp,
        status_code=422,
        code=VIDEO_JOB_TOO_LARGE,
        message=VIDEO_JOB_TOO_LARGE_MESSAGE,
    )


def test_ic_lora_unsupported_returns_local_generation_unsupported(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    test_state.config.local_generations_mode = "unsupported"
    img = tmp_path / "in.png"
    _write_png(img, size=(1024, 576))
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(img),
            "prompt": "p",
            "conditioning_type": "custom",
        },
    )
    assert_http_error(
        resp,
        status_code=422,
        code=LOCAL_GENERATION_UNSUPPORTED,
        message=LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    )
    body = resp.json()["message"].lower()
    assert "api" not in body
    assert "resolution" not in body


def test_ic_lora_unsupported_without_weights_is_unsupported_not_not_downloaded(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state
):
    """The real fresh-install case: downloads are disabled, so weights are never present."""
    create_fake_model_files()
    test_state.config.local_generations_mode = "unsupported"
    img = tmp_path / "in.png"
    _write_png(img, size=(1024, 576))
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(img),
            "prompt": "p",
            "conditioning_type": "custom",
        },
    )
    assert_http_error(
        resp,
        status_code=422,
        code=LOCAL_GENERATION_UNSUPPORTED,
        message=LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    )


def test_ic_lora_unsupported_oversized_source_is_unsupported_not_spatial(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    test_state.config.local_generations_mode = "unsupported"
    img = tmp_path / "in.png"
    _write_png(img, size=(1280, 720))
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "ingredients-v1",
            "input_path": str(img),
            "prompt": "p",
            "conditioning_type": "custom",
        },
    )
    assert_http_error(
        resp,
        status_code=422,
        code=LOCAL_GENERATION_UNSUPPORTED,
        message=LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    )
    body = resp.json()["message"].lower()
    assert "api" not in body
    assert "resolution" not in body


def test_ic_lora_darwin_does_not_422_20s_540p(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services, test_state
):
    create_fake_model_files()
    test_state.config.darwin_unified_memory = True
    test_state.config.available_ram_gb = 20
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * compute_num_frames(20, 24), fps=24, width=960, height=540)
    )
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
            "skip_stage_2": True,
            "resolution_factor": 0,
        },
    )
    assert resp.status_code == 200
    assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count == 2


def test_ic_lora_fps_override_budgets_resampled_frames(
    client: TestClient,
    tmp_path: Path,
    create_fake_model_files,
    create_fake_ic_lora_files,
    fake_services,
    test_state,
):
    create_fake_model_files()
    create_fake_ic_lora_files()
    test_state.config.vram_gb = 20
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    frames = compute_num_frames(20, 24)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * frames, fps=24, width=960, height=540)
    )
    payload = {
        "video_path": str(src),
        "conditioning_type": "canny",
        "prompt": "p",
        "skip_stage_2": True,
        "resolution_factor": 0,
    }
    over = client.post("/api/ic-lora/generate", json=payload)
    assert_http_error(
        over,
        status_code=422,
        code=VIDEO_JOB_TOO_LARGE,
        message=VIDEO_JOB_TOO_LARGE_MESSAGE,
    )
    ok = client.post("/api/ic-lora/generate", json={**payload, "fps_override": 12})
    assert ok.status_code == 200
    assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count == 2


def test_ic_lora_rejects_unreadable_image(client: TestClient, tmp_path: Path, create_fake_model_files):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    img = tmp_path / "in.png"
    img.write_bytes(b"not-a-png")
    resp = client.post(
        "/api/ic-lora/generate",
        json={"ic_lora_id": "ingredients-v1", "input_path": str(img), "prompt": "p", "conditioning_type": "custom"},
    )
    assert resp.status_code == 400
    assert "Could not read the input image" in resp.json()["message"]


def test_ic_lora_rejects_unreadable_video(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(str(src), FakeCapture(opened=False))
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
        },
    )
    assert resp.status_code == 400
    assert "Could not read the input video" in resp.json()["message"]


def test_ic_lora_rejects_zero_size_video(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "colorize-v1"})
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(
        str(src), FakeCapture(frames=["f"] * 9, width=0, height=0, fps=24)
    )
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "colorize-v1",
            "input_path": str(src),
            "prompt": "p",
            "conditioning_type": "custom",
        },
    )
    assert resp.status_code == 400
    assert "Could not read the input video" in resp.json()["message"]


def test_outpaint_over_spatial_cap_unlinks_mask(
    client: TestClient, tmp_path: Path, create_fake_model_files, fake_services
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "outpaint-v1"})
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00")
    fake_services.video_processor.videos[str(src)] = FakeCapture(
        frames=[np.full((720, 1280, 3), 0x40, dtype=np.uint8) for _ in range(9)],
        width=1280,
        height=720,
        fps=24,
    )
    resp = client.post(
        "/api/ic-lora/generate",
        json={
            "ic_lora_id": "outpaint-v1",
            "input_path": str(src),
            "prompt": "extend the scene",
            "conditioning_type": "custom",
            "skip_stage_2": True,
            "resolution_factor": 0,
            "outpaint_pads": {"left": 1280, "right": 0, "top": 0, "bottom": 0},
        },
    )
    assert_http_error(
        resp,
        status_code=422,
        code=IC_LORA_SOURCE_TOO_LARGE,
        message=IC_LORA_V1_ENVELOPE_MESSAGE,
    )
    written = [writer.path for writer in fake_services.video_processor.writers]
    assert len(written) >= 2
    for path in written:
        assert not path.exists(), path


def test_catalog_preprocess_runs_after_generation_started(
    client: TestClient, tmp_path: Path, create_fake_model_files, test_state, monkeypatch
):
    from services.generation_interrupt import GenerationCancelledError

    import handlers.ic_lora_handler as handler_mod

    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    img = tmp_path / "in.png"
    _write_png(img)
    seen: dict[str, object] = {}

    def spy(steps, artifact, ctx):
        running = test_state.generation._running_generation()
        assert running is not None
        _, gen = running
        assert gen.progress.phase == "loading_model"
        assert ctx.on_frame is not None
        seen["ok"] = True
        raise GenerationCancelledError()

    monkeypatch.setattr(handler_mod, "run_preprocessing", spy)
    resp = client.post(
        "/api/ic-lora/generate",
        json={"ic_lora_id": "ingredients-v1", "input_path": str(img), "prompt": "p", "conditioning_type": "custom"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    assert seen["ok"] is True
