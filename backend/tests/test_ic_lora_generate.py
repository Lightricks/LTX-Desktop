import logging
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from tests.fakes import FakeCapture


def test_ic_lora_generate_runs_inference(client: TestClient, tmp_path: Path, create_fake_model_files, fake_services):
    create_fake_model_files()  # base LTX model + upscale + text encoder for load_ic_lora
    # Download the fake IC-LoRA weights so the handler finds them.
    start = client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    assert start.status_code == 200
    img = tmp_path / "in.png"
    img.write_bytes(b"\x89PNG\r\n")
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


def test_catalog_stop_after_generate_unlinks_and_returns_cancelled(
    client: TestClient, tmp_path: Path, test_state, create_fake_model_files, fake_services, caplog
):
    create_fake_model_files()
    client.post("/api/ic-loras/download", json={"ic_lora_id": "ingredients-v1"})
    img = tmp_path / "in.png"
    img.write_bytes(b"\x89PNG\r\n")

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
    img.write_bytes(b"\x89PNG\r\n")
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
    img.write_bytes(b"\x89PNG\r\n")
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


def test_ic_lora_rejects_image_for_video_input(client: TestClient, tmp_path: Path):
    # colorize-v1 expects a video input; feeding an image must fail with a clean 400
    # (not a deep pipeline 500). Fires before the download check, so no setup needed.
    img = tmp_path / "in.png"
    img.write_bytes(b"\x89PNG\r\n")
    resp = client.post(
        "/api/ic-lora/generate",
        json={"ic_lora_id": "colorize-v1", "input_path": str(img), "prompt": "p", "conditioning_type": "custom"},
    )
    assert resp.status_code == 400


def test_ic_lora_rejects_video_for_image_input(client: TestClient, tmp_path: Path):
    # ingredients-v1 expects an image input; feeding a video must 400.
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


def test_ic_lora_empty_prompt_rejected_without_flag(client: TestClient, tmp_path: Path):
    # colorize-v1 has no allows_empty_prompt → a blank prompt is rejected.
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
    ref.write_bytes(b"\x89PNG\r\n")
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
    ref.write_bytes(b"\x89PNG\r\n")
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


def test_list_reports_downloaded_variant_ids_after_partial_download(client: TestClient):
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
