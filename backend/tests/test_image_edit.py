"""Integration-style tests for image editing (image-to-image) endpoints."""

from __future__ import annotations

import io
from pathlib import Path

from services.services_utils import compute_edit_dimensions
from tests.http_error_assertions import assert_http_error


def _write_source_png(tmp_path: Path, *, w: int = 64, h: int = 64, color: str = "red") -> str:
    """Write a real PNG to disk and return its path (edit source must pass validation)."""
    from PIL import Image

    path = tmp_path / "edit_source.png"
    Image.new("RGB", (w, h), color).save(path, format="PNG")
    return str(path)


# ============================================================
# Local edit path
# ============================================================


class TestEditImage:
    def test_happy_path(self, client, create_fake_model_files, tmp_path):
        create_fake_model_files(include_zit=True)
        src = _write_source_png(tmp_path)

        r = client.post(
            "/api/generate-image",
            json={"prompt": "make it photorealistic", "imagePath": src, "strength": 0.6},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert len(data["image_paths"]) == 1
        assert Path(data["image_paths"][0]).exists()

    def test_routes_to_edit_not_generate(self, client, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files(include_zit=True)
        src = _write_source_png(tmp_path)

        r = client.post(
            "/api/generate-image",
            json={"prompt": "edit me", "imagePath": src, "strength": 0.7},
        )
        assert r.status_code == 200

        assert len(fake_services.image_generation_pipeline.edit_calls) == 1
        assert len(fake_services.image_generation_pipeline.generate_calls) == 0

    def test_edit_call_payload(self, client, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files(include_zit=True)
        src = _write_source_png(tmp_path)

        r = client.post(
            "/api/generate-image",
            json={
                "prompt": "photoreal harbor",
                "imagePath": src,
                "strength": 0.8,
                "numSteps": 6,
            },
        )
        assert r.status_code == 200

        call = fake_services.image_generation_pipeline.edit_calls[0]
        assert call["prompt"] == "photoreal harbor"
        assert call["image"].size == (64, 64)  # decoded + resized once by the handler
        assert call["strength"] == 0.8
        assert call["num_inference_steps"] == 6
        assert isinstance(call["seed"], int)

    def test_multi_image_seed_increments(self, client, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files(include_zit=True)
        src = _write_source_png(tmp_path)

        r = client.post(
            "/api/generate-image",
            json={"prompt": "edit", "imagePath": src, "strength": 0.6, "numImages": 3},
        )
        assert r.status_code == 200
        assert len(r.json()["image_paths"]) == 3

        calls = fake_services.image_generation_pipeline.edit_calls
        assert len(calls) == 3
        # Seeds increment by 1 per image regardless of the (possibly random) base seed.
        assert calls[1]["seed"] - calls[0]["seed"] == 1
        assert calls[2]["seed"] - calls[1]["seed"] == 1

    def test_strength_above_range_rejected(self, client, create_fake_model_files, tmp_path):
        create_fake_model_files(include_zit=True)
        src = _write_source_png(tmp_path)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": 1.5},
        )
        assert r.status_code == 422  # pydantic Field(le=1.0)

    def test_strength_below_range_rejected(self, client, create_fake_model_files, tmp_path):
        create_fake_model_files(include_zit=True)
        src = _write_source_png(tmp_path)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": -0.1},
        )
        assert r.status_code == 422  # pydantic Field(ge=0.0)

    def test_strength_too_low_for_steps(self, client, create_fake_model_files, tmp_path):
        create_fake_model_files(include_zit=True)
        src = _write_source_png(tmp_path)
        # strength == 0.0 is the only value that drives the pipeline's real
        # get_timesteps formula to 0 effective steps -> guarded before it runs.
        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": 0.0, "numSteps": 4},
        )
        assert_http_error(r, status_code=422, code="EDIT_STRENGTH_TOO_LOW_FOR_STEPS")

    def test_low_strength_low_steps_still_valid(self, client, fake_services, create_fake_model_files, tmp_path):
        # Any positive strength, however small, yields >=1 effective step per the real
        # formula -- this combo used to be a false-positive reject under the old
        # round()-based guard.
        create_fake_model_files(include_zit=True)
        src = _write_source_png(tmp_path)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": 0.01, "numSteps": 1},
        )
        assert r.status_code == 200
        assert len(fake_services.image_generation_pipeline.edit_calls) == 1

    def test_source_missing(self, client, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": "/does/not/exist.png", "strength": 0.6},
        )
        assert_http_error(r, status_code=400, code="HTTP_400", message="Image file not found: /does/not/exist.png")

    def test_source_unreadable(self, client, create_fake_model_files, tmp_path):
        create_fake_model_files(include_zit=True)
        bad = tmp_path / "not_really.png"
        bad.write_bytes(b"this is not a png")
        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": str(bad), "strength": 0.6},
        )
        assert_http_error(r, status_code=400, code="HTTP_400", message=f"Invalid image file: {bad}")

    def test_source_too_large_rejected(self, client, create_fake_model_files, tmp_path):
        create_fake_model_files(include_zit=True)
        src = _write_source_png(tmp_path, w=8000, h=8000)  # 64M px > validate_image_file's 50M cap
        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": 0.6},
        )
        assert_http_error(r, status_code=400, code="HTTP_400", message=f"Image dimensions too large: {src}")

    def test_pipeline_error(self, client, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files(include_zit=True)
        src = _write_source_png(tmp_path)
        fake_services.image_generation_pipeline.raise_on_edit = RuntimeError("GPU OOM")

        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": 0.6},
        )
        assert r.status_code == 500

    def test_cancelled(self, client, fake_services, create_fake_model_files, tmp_path):
        create_fake_model_files(include_zit=True)
        src = _write_source_png(tmp_path)
        fake_services.image_generation_pipeline.raise_on_edit = RuntimeError("cancelled")

        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": 0.6},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"

    def test_partial_outputs_cleaned_up_on_mid_batch_error(
        self, client, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files(include_zit=True)
        src = _write_source_png(tmp_path)
        fake_services.image_generation_pipeline.fail_edit_after = 1
        fake_services.image_generation_pipeline.raise_on_edit = RuntimeError("GPU OOM")

        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": 0.6, "numImages": 3},
        )
        assert r.status_code == 500
        assert list((tmp_path / "outputs").glob("zit_edit_*.png")) == []

    def test_no_image_path_still_generates(self, client, fake_services, create_fake_model_files):
        """Backward compatibility: omitting imagePath keeps the plain T2I path."""
        create_fake_model_files(include_zit=True)
        r = client.post("/api/generate-image", json={"prompt": "a cat"})
        assert r.status_code == 200
        assert len(fake_services.image_generation_pipeline.generate_calls) == 1
        assert len(fake_services.image_generation_pipeline.edit_calls) == 0


# ============================================================
# Forced-API (FAL) edit path
# ============================================================


class TestForcedApiEditImage:
    def _force_api(self, test_state, *, key: str = "fal-key") -> None:
        test_state.config.local_generations_mode = "unsupported"
        test_state.state.app_settings.fal_api_key = key

    def test_routes_to_image_to_image(self, client, test_state, fake_services, tmp_path):
        self._force_api(test_state)
        src = _write_source_png(tmp_path)

        r = client.post(
            "/api/generate-image",
            json={"prompt": "photoreal", "imagePath": src, "strength": 0.6, "numImages": 2},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert len(data["image_paths"]) == 2
        assert len(fake_services.zit_api_client.image_to_image_calls) == 2
        assert len(fake_services.zit_api_client.text_to_image_calls) == 0
        assert len(fake_services.image_generation_pipeline.edit_calls) == 0

    def test_payload(self, client, test_state, fake_services, tmp_path):
        self._force_api(test_state)
        src = _write_source_png(tmp_path)

        r = client.post(
            "/api/generate-image",
            json={"prompt": "photoreal", "imagePath": src, "strength": 0.75, "numSteps": 5},
        )
        assert r.status_code == 200

        call = fake_services.zit_api_client.image_to_image_calls[0]
        assert call["prompt"] == "photoreal"
        assert call["strength"] == 0.75
        assert call["num_inference_steps"] == 5
        assert call["api_key"] == "fal-key"
        assert call["image_bytes_len"] > 0

    def test_jpeg_source_sent_as_real_png(self, client, test_state, fake_services, tmp_path):
        # The client always labels the payload image/png; a non-PNG source must actually
        # be re-encoded to PNG, not sent as-is under a mismatched mimetype.
        from PIL import Image

        self._force_api(test_state)
        src = tmp_path / "edit_source.jpg"
        Image.new("RGB", (64, 64), "blue").save(src, format="JPEG")

        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": str(src), "strength": 0.6},
        )
        assert r.status_code == 200

        sent_bytes = fake_services.zit_api_client.image_to_image_calls[0]["image_bytes"]
        assert Image.open(io.BytesIO(sent_bytes)).format == "PNG"

    def test_source_downscaled_to_edit_target_before_upload(self, client, test_state, fake_services, tmp_path):
        # The API path must downscale to the same /16 target the local path uses,
        # instead of uploading the full-resolution source.
        from PIL import Image

        self._force_api(test_state)
        src = _write_source_png(tmp_path, w=1000, h=800)

        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": 0.6},
        )
        assert r.status_code == 200

        sent_bytes = fake_services.zit_api_client.image_to_image_calls[0]["image_bytes"]
        assert Image.open(io.BytesIO(sent_bytes)).size == compute_edit_dimensions(1000, 800)

    def test_strength_zero_rejected_before_fal_call(self, client, test_state, fake_services, tmp_path):
        # Same guard as the local path: strength == 0.0 must 422 before ever reaching FAL,
        # not get forwarded and come back as an opaque upstream error.
        self._force_api(test_state)
        src = _write_source_png(tmp_path)

        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": 0.0, "numSteps": 4},
        )
        assert_http_error(r, status_code=422, code="EDIT_STRENGTH_TOO_LOW_FOR_STEPS")
        assert len(fake_services.zit_api_client.image_to_image_calls) == 0

    def test_missing_fal_key(self, client, test_state, tmp_path):
        self._force_api(test_state, key="")
        src = _write_source_png(tmp_path)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": 0.6},
        )
        assert_http_error(r, status_code=500, code="FAL_API_KEY_NOT_CONFIGURED")

    def test_cancelled(self, client, test_state, fake_services, tmp_path):
        self._force_api(test_state)
        src = _write_source_png(tmp_path)
        fake_services.zit_api_client.raise_on_image_to_image = RuntimeError("cancelled")
        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": 0.6},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"

    def test_api_failure(self, client, test_state, fake_services, tmp_path):
        self._force_api(test_state)
        src = _write_source_png(tmp_path)
        fake_services.zit_api_client.raise_on_image_to_image = RuntimeError("boom")
        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": 0.6},
        )
        assert r.status_code == 500

    def test_partial_outputs_cleaned_up_on_mid_batch_error(self, client, test_state, fake_services, tmp_path):
        self._force_api(test_state)
        src = _write_source_png(tmp_path)
        fake_services.zit_api_client.fail_image_to_image_after = 1
        fake_services.zit_api_client.raise_on_image_to_image = RuntimeError("boom")

        r = client.post(
            "/api/generate-image",
            json={"prompt": "x", "imagePath": src, "strength": 0.6, "numImages": 3},
        )
        assert r.status_code == 500
        assert list((tmp_path / "outputs").glob("zit_api_edit_*.png")) == []


# ============================================================
# Pure unit: dimension flooring
# ============================================================


class TestComputeEditDimensions:
    def test_floors_to_multiple_of_16(self):
        assert compute_edit_dimensions(1000, 800) == (992, 800)
        assert compute_edit_dimensions(1023, 1023) == (1008, 1008)

    def test_already_aligned(self):
        assert compute_edit_dimensions(1024, 1024) == (1024, 1024)

    def test_never_below_16(self):
        assert compute_edit_dimensions(10, 10) == (16, 16)


# ============================================================
# User preference: FAL API even when local generation is available
# ============================================================


class TestUserPrefersFalApiImageGenerations:
    def test_generate_prefers_api_routes_to_fal(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_fal_api_image_generations = True
        test_state.state.app_settings.fal_api_key = "fal-key"

        r = client.post("/api/generate-image", json={"prompt": "a cat"})

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.zit_api_client.text_to_image_calls) == 1
        assert len(fake_services.image_generation_pipeline.generate_calls) == 0

    def test_generate_prefers_api_without_key_falls_back_to_local(
        self, client, test_state, fake_services, create_fake_model_files
    ):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_fal_api_image_generations = True
        test_state.state.app_settings.fal_api_key = ""
        create_fake_model_files(include_zit=True)

        r = client.post("/api/generate-image", json={"prompt": "a cat"})

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.image_generation_pipeline.generate_calls) == 1
        assert len(fake_services.zit_api_client.text_to_image_calls) == 0

    def test_edit_prefers_api_routes_to_fal(self, client, test_state, fake_services, tmp_path):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_fal_api_image_generations = True
        test_state.state.app_settings.fal_api_key = "fal-key"
        src = _write_source_png(tmp_path)

        r = client.post(
            "/api/generate-image",
            json={"prompt": "edit", "imagePath": src, "strength": 0.6},
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.zit_api_client.image_to_image_calls) == 1
        assert len(fake_services.image_generation_pipeline.edit_calls) == 0

    def test_edit_prefers_api_without_key_falls_back_to_local(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_fal_api_image_generations = True
        test_state.state.app_settings.fal_api_key = ""
        create_fake_model_files(include_zit=True)
        src = _write_source_png(tmp_path)

        r = client.post(
            "/api/generate-image",
            json={"prompt": "edit", "imagePath": src, "strength": 0.6},
        )

        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(fake_services.image_generation_pipeline.edit_calls) == 1
        assert len(fake_services.zit_api_client.image_to_image_calls) == 0
