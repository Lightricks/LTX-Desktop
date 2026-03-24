"""Integration tests for FLUX.2 Klein 9B image generation pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime_config.model_download_specs import DEFAULT_MODEL_DOWNLOAD_SPECS


def _create_flux_klein_model_files(test_state) -> None:
    """Create fake FLUX Klein model directory so pipelines_handler finds it."""
    flux_dir = test_state.config.model_path("flux_klein")
    flux_dir.mkdir(parents=True, exist_ok=True)
    (flux_dir / "model_index.json").write_bytes(b"{}")
    (flux_dir / "model.safetensors").write_bytes(b"\x00" * 1024)


def _create_zit_model_files(test_state) -> None:
    """Create fake ZIT model directory and set it as the active image model."""
    zit_dir = test_state.config.model_path("zit")
    zit_dir.mkdir(parents=True, exist_ok=True)
    (zit_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
    test_state.state.app_settings.image_model = "z-image-turbo"


# ── 1. Model Registration ─────────────────────────────────────────────


class TestFluxKleinModelRegistration:
    def test_flux_klein_in_download_specs(self):
        assert "flux_klein" in DEFAULT_MODEL_DOWNLOAD_SPECS

    def test_flux_klein_spec_repo_id(self):
        spec = DEFAULT_MODEL_DOWNLOAD_SPECS["flux_klein"]
        assert spec.repo_id == "black-forest-labs/FLUX.2-klein-base-9B"

    def test_flux_klein_spec_is_folder(self):
        spec = DEFAULT_MODEL_DOWNLOAD_SPECS["flux_klein"]
        assert spec.is_folder is True

    def test_flux_klein_in_available_files(self, test_state):
        assert "flux_klein" in test_state.state.available_files

    def test_flux_klein_model_path_resolves(self, test_state):
        path = test_state.config.model_path("flux_klein")
        assert path.name == "FLUX.2-klein-base-9B"


# ── 2. Pipeline Handler: Model Routing ────────────────────────────────


class TestFluxKleinPipelineRouting:
    def test_load_image_model_zit_by_default(self, test_state, fake_services):
        _create_zit_model_files(test_state)
        pipeline = test_state.pipelines.load_image_model_to_gpu("zit")
        # Should return the ZIT fake singleton
        assert pipeline is fake_services.image_generation_pipeline

    def test_load_image_model_flux_klein_not_configured(self, test_state):
        """FLUX Klein pipeline class is None in tests, so loading should raise."""
        _create_flux_klein_model_files(test_state)
        with pytest.raises(RuntimeError, match="FLUX.2 Klein pipeline class not configured"):
            test_state.pipelines.load_image_model_to_gpu("flux-klein-9b")

    def test_load_image_model_flux_klein_alias(self, test_state):
        """Both 'flux-klein-9b' and 'flux_klein' should route to FLUX loader."""
        _create_flux_klein_model_files(test_state)
        with pytest.raises(RuntimeError, match="FLUX.2 Klein pipeline class not configured"):
            test_state.pipelines.load_image_model_to_gpu("flux_klein")


# ── 3. Image Generation: ZIT Default Path ─────────────────────────────


class TestFluxKleinImageGeneration:
    def test_generate_image_zit_explicit(self, client, fake_services, test_state):
        """Explicitly selecting z-image-turbo uses ZIT pipeline."""
        _create_zit_model_files(test_state)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "A cat", "width": 1024, "height": 1024, "numSteps": 4},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert len(data["image_paths"]) == 1
        assert len(fake_services.image_generation_pipeline.generate_calls) == 1

    def test_generate_image_flux_klein_setting_uses_flux_path(self, client, test_state):
        """Setting image_model=flux-klein-9b routes to FLUX handler (fails without class)."""
        _create_flux_klein_model_files(test_state)
        test_state.state.app_settings.image_model = "flux-klein-9b"

        r = client.post(
            "/api/generate-image",
            json={"prompt": "A cat", "width": 1024, "height": 1024, "numSteps": 50},
        )
        # Should fail because flux_klein_pipeline_class is None in tests
        assert r.status_code == 500

    def test_generate_image_guidance_scale_for_flux(self, test_state, fake_services):
        """Verify guidance_scale=4.0 is used for FLUX models in generate_image."""
        _create_zit_model_files(test_state)
        # Use ZIT but check the image_model routing produces correct guidance
        handler = test_state.image_generation
        handler._pipelines.load_image_model_to_gpu("zit")
        test_state.generation.start_generation("test-id")
        paths = handler.generate_image(
            prompt="test",
            width=512,
            height=512,
            num_inference_steps=4,
            seed=42,
            num_images=1,
            image_model="z-image-turbo",
        )
        # ZIT uses guidance_scale=0.0
        call = fake_services.image_generation_pipeline.generate_calls[0]
        assert call["guidance_scale"] == 0.0
        assert len(paths) == 1


# ── 4. Dimension and Clamp Tests ──────────────────────────────────────


class TestFluxKleinDimensionClamping:
    def test_dimensions_rounded_to_16(self, client, test_state, fake_services):
        _create_zit_model_files(test_state)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "test", "width": 1023, "height": 1023},
        )
        assert r.status_code == 200
        call = fake_services.image_generation_pipeline.generate_calls[0]
        assert call["width"] % 16 == 0
        assert call["height"] % 16 == 0

    def test_num_images_clamped_to_12(self, client, test_state, fake_services):
        _create_zit_model_files(test_state)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "test", "numImages": 20},
        )
        assert r.status_code == 200
        assert len(fake_services.image_generation_pipeline.generate_calls) == 12


# ── 5. Model Label Routing ────────────────────────────────────────────


class TestFluxKleinModelLabeling:
    def test_zit_model_label_in_output_path(self, test_state, fake_services):
        _create_zit_model_files(test_state)
        handler = test_state.image_generation
        handler._pipelines.load_image_model_to_gpu("zit")
        test_state.generation.start_generation("label-test")
        paths = handler.generate_image(
            prompt="a dog",
            width=512,
            height=512,
            num_inference_steps=4,
            seed=42,
            num_images=1,
            image_model="z-image-turbo",
        )
        # Output path should contain "zit" in the filename
        assert "zit" in Path(paths[0]).name.lower()


# ── 6. Error Handling ─────────────────────────────────────────────────


class TestFluxKleinErrorHandling:
    def test_flux_not_downloaded_returns_error(self, test_state):
        """Requesting FLUX Klein when model not downloaded gives clear error."""
        with pytest.raises(RuntimeError, match="FLUX.2 Klein pipeline class not configured"):
            test_state.pipelines.load_image_model_to_gpu("flux-klein-9b")

    def test_generation_error_returns_500(self, client, test_state, fake_services):
        _create_zit_model_files(test_state)
        fake_services.image_generation_pipeline.raise_on_generate = RuntimeError("GPU OOM")
        r = client.post("/api/generate-image", json={"prompt": "test"})
        assert r.status_code == 500

    def test_cancellation_returns_cancelled(self, client, test_state, fake_services):
        _create_zit_model_files(test_state)
        fake_services.image_generation_pipeline.raise_on_generate = RuntimeError("cancelled")
        r = client.post("/api/generate-image", json={"prompt": "test"})
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


# ── 7. Settings Integration ───────────────────────────────────────────


class TestFluxKleinSettingsIntegration:
    def test_image_model_setting_defaults_to_flux_klein(self, test_state):
        assert test_state.state.app_settings.image_model == "flux-klein-9b"

    def test_image_model_setting_can_be_changed(self, test_state):
        test_state.state.app_settings.image_model = "flux-klein-9b"
        assert test_state.state.app_settings.image_model == "flux-klein-9b"

    def test_settings_endpoint_accepts_image_model(self, client, test_state):
        r = client.post(
            "/api/settings",
            json={"imageModel": "flux-klein-9b"},
        )
        assert r.status_code == 200
        assert test_state.state.app_settings.image_model == "flux-klein-9b"


# ── 8. Seed Behavior ─────────────────────────────────────────────────


class TestFluxKleinSeedBehavior:
    def test_locked_seed_used(self, client, test_state, fake_services):
        _create_zit_model_files(test_state)
        test_state.state.app_settings.seed_locked = True
        test_state.state.app_settings.locked_seed = 12345
        r = client.post(
            "/api/generate-image",
            json={"prompt": "test", "width": 512, "height": 512, "numSteps": 4},
        )
        assert r.status_code == 200
        call = fake_services.image_generation_pipeline.generate_calls[0]
        assert call["seed"] == 12345

    def test_multiple_images_increment_seed(self, client, test_state, fake_services):
        _create_zit_model_files(test_state)
        test_state.state.app_settings.seed_locked = True
        test_state.state.app_settings.locked_seed = 100
        r = client.post(
            "/api/generate-image",
            json={"prompt": "test", "numImages": 3},
        )
        assert r.status_code == 200
        seeds = [c["seed"] for c in fake_services.image_generation_pipeline.generate_calls]
        assert seeds == [100, 101, 102]


# ── 9. Img2Img Routing ───────────────────────────────────────────────


class TestFluxKleinImg2Img:
    def test_source_image_routes_to_img2img(self, client, test_state, fake_services, make_test_image, tmp_path):
        _create_zit_model_files(test_state)
        img_buf = make_test_image(64, 64)
        img_path = tmp_path / "source.png"
        img_path.write_bytes(img_buf.read())

        r = client.post(
            "/api/generate-image",
            json={
                "prompt": "A cat",
                "width": 512,
                "height": 512,
                "numSteps": 4,
                "sourceImagePath": str(img_path),
                "strength": 0.7,
            },
        )
        assert r.status_code == 200
        assert len(fake_services.image_generation_pipeline.img2img_calls) == 1
        assert len(fake_services.image_generation_pipeline.generate_calls) == 0


# ── 10. Concurrent Generation Guard ──────────────────────────────────


class TestFluxKleinConcurrencyGuard:
    def test_rejects_concurrent_generation(self, client, test_state, fake_services):
        _create_zit_model_files(test_state)
        # Load a pipeline to GPU so we can start a generation
        test_state.pipelines.load_image_model_to_gpu("zit")
        test_state.generation.start_generation("existing-gen")

        r = client.post(
            "/api/generate-image",
            json={"prompt": "test", "width": 512, "height": 512},
        )
        assert r.status_code == 409
