"""Integration-style tests for model-related endpoints."""

import inspect

from huggingface_hub import file_download

from state.app_state_types import FileDownloadCompleted, FileDownloadRunning


class TestModelsList:
    def test_defaults(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert data[0]["id"] == "fast"
        assert "8 steps" in data[0]["description"]
        assert data[1]["id"] == "pro"
        assert "20 steps" in data[1]["description"]

    def test_custom_pro_steps(self, client, test_state):
        test_state.state.app_settings.pro_model.steps = 30
        r = client.get("/api/models")
        assert "30 steps" in r.json()[1]["description"]


class TestModelsStatus:
    def test_nothing_downloaded(self, client):
        r = client.get("/api/models/status")
        assert r.status_code == 200
        assert r.json()["all_downloaded"] is False

    def test_all_downloaded(self, client, create_fake_model_files):
        create_fake_model_files(include_flux_klein=True)
        r = client.get("/api/models/status")
        assert r.json()["all_downloaded"] is True

    def test_with_api_key(self, client, create_fake_model_files, test_state):
        create_fake_model_files(include_zit=True)
        test_state.state.app_settings.ltx_api_key = "test-key"

        r = client.get("/api/models/status")
        te_model = next(m for m in r.json()["models"] if m["name"] == "gemma-3-12b-it-qat-q4_0-unquantized")
        assert te_model["required"] is False

    def test_forced_mode_requires_no_local_models(self, client, test_state):
        test_state.config.force_api_generations = True
        test_state.config.required_model_types = frozenset()

        r = client.get("/api/models/status")
        data = r.json()
        assert data["all_downloaded"] is True

        required_names = {m["name"] for m in data["models"] if m["required"]}
        assert required_names == set()


class TestDownloadProgress:
    def test_idle(self, client):
        r = client.get("/api/models/download/progress")
        assert r.status_code == 200
        assert r.json()["status"] == "idle"

    def test_active(self, client, test_state):
        test_state.state.downloading_session = {
            "checkpoint": FileDownloadRunning(
                target_path="checkpoint",
                progress=0.5,
                downloaded_bytes=5_000_000_000,
                total_bytes=10_000_000_000,
                speed_bytes_per_sec=50_000_000.0,
            )
        }
        r = client.get("/api/models/download/progress")
        data = r.json()
        assert data["status"] == "downloading"
        assert data["currentFile"] == "checkpoint"


class TestModelDownload:
    def test_start_success(self, client, test_state):
        r = client.post("/api/models/download", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "started"
        assert data["skippingTextEncoder"] is False

        snapshot_calls = [c for c in test_state.model_downloader.calls if c["kind"] == "snapshot"]
        assert snapshot_calls

    def test_already_in_progress(self, client, test_state):
        test_state.downloads.start_download({"checkpoint": ("checkpoint", 100)})
        r = client.post("/api/models/download", json={})
        assert r.status_code == 409

    def test_skip_text_encoder(self, client, test_state):
        r = client.post("/api/models/download", json={"skipTextEncoder": True})
        assert r.status_code == 200

        te_spec = test_state.config.spec_for("text_encoder")
        te_calls = [c for c in test_state.model_downloader.calls if c.get("repo_id") == te_spec.repo_id]
        assert not te_calls, "text encoder download should have been skipped"

    def test_api_key_auto_skip(self, client, test_state):
        test_state.state.app_settings.ltx_api_key = "test-key"

        r = client.post("/api/models/download", json={})
        assert r.status_code == 200
        assert r.json()["skippingTextEncoder"] is True

    def test_forced_mode_downloads_no_local_models(self, client, test_state):
        test_state.config.force_api_generations = True
        test_state.config.required_model_types = frozenset()

        r = client.post("/api/models/download", json={})
        assert r.status_code == 200

        calls = test_state.model_downloader.calls
        assert len(calls) == 0
        assert r.json()["skippingTextEncoder"] is False


class TestTextEncoderDownload:
    def test_start_download(self, client):
        r = client.post("/api/text-encoder/download")
        assert r.status_code == 200
        assert r.json()["status"] == "started"

    def test_already_downloaded(self, client, test_state):
        te_dir = test_state.config.model_path("text_encoder")
        te_dir.mkdir(parents=True, exist_ok=True)
        (te_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

        r = client.post("/api/text-encoder/download")
        assert r.status_code == 200
        assert r.json()["status"] == "already_downloaded"

    def test_already_in_progress(self, client, test_state):
        test_state.downloads.start_download({"checkpoint": ("checkpoint", 100)})
        r = client.post("/api/text-encoder/download")
        assert r.status_code == 409


class TestDownloadProgressCallbacks:
    def test_download_passes_progress_callback(self, client, test_state):
        r = client.post("/api/models/download", json={})
        assert r.status_code == 200

        calls = test_state.model_downloader.calls
        assert len(calls) > 0
        for call in calls:
            assert call["on_progress"] is not None, f"on_progress missing for {call['kind']} call"

    def test_text_encoder_download_passes_progress_callback(self, client, test_state):
        r = client.post("/api/text-encoder/download")
        assert r.status_code == 200

        calls = test_state.model_downloader.calls
        assert len(calls) > 0
        for call in calls:
            assert call["on_progress"] is not None

    def test_progress_callback_updates_state(self, client, test_state):
        r = client.post("/api/models/download", json={})
        assert r.status_code == 200

        # The fake downloader invokes on_progress(512, 1024) then on_progress(1024, 1024).
        # After download completes, each file is marked completed.
        # Verify that the download session was populated (files are now completed).
        r = client.get("/api/models/download/progress")
        data = r.json()
        assert data["status"] == "complete"
        assert data["filesCompleted"] > 0

    def test_progress_callback_updates_running_state(self, test_state):
        """Directly invoke callback to verify it updates FileDownloadRunning state."""
        test_state.downloads.start_download({"checkpoint": ("checkpoint", 10_000)})
        cb = test_state.downloads._make_progress_callback("checkpoint")
        cb(5_000, 10_000)

        r = test_state.downloads.get_download_progress()
        assert r.currentFileProgress == 50
        assert r.downloadedBytes == 5_000


class TestAtomicDownloads:
    """Verify downloads use .downloading/ staging dir and atomic moves."""

    def test_partial_file_in_downloading_dir_not_detected(self, test_state):
        """Files in .downloading/ must NOT be reported as downloaded."""
        downloading = test_state.config.downloading_dir
        downloading.mkdir(parents=True, exist_ok=True)
        (downloading / "ltx-2-19b-distilled-fp8.safetensors").write_bytes(b"\x00" * 1024)

        test_state.models.refresh_available_files()
        assert test_state.state.available_files["checkpoint"] is None

    def test_cleanup_downloading_dir_on_startup(self, test_state):
        """cleanup_downloading_dir() removes stale .downloading/ dir."""
        downloading = test_state.config.downloading_dir
        downloading.mkdir(parents=True, exist_ok=True)
        (downloading / "partial-file.safetensors").write_bytes(b"\x00" * 1024)

        test_state.downloads.cleanup_downloading_dir()
        assert not downloading.exists()

    def test_cleanup_downloading_dir_noop_when_absent(self, test_state):
        """cleanup_downloading_dir() is safe when dir doesn't exist."""
        test_state.downloads.cleanup_downloading_dir()
        assert not test_state.config.downloading_dir.exists()

    def test_download_moves_files_to_final_location(self, client, test_state):
        """After download, files exist at final location, not in .downloading/."""
        r = client.post("/api/models/download", json={})
        assert r.status_code == 200

        # Files should be at their final locations
        assert test_state.config.model_path("checkpoint").exists()
        assert test_state.config.model_path("upsampler").exists()

        # .downloading/ should be gone (or empty)
        downloading = test_state.config.downloading_dir
        assert not downloading.exists() or not any(downloading.iterdir())

    def test_text_encoder_download_moves_to_final(self, client, test_state):
        """Text encoder download uses .downloading/ and moves to final."""
        r = client.post("/api/text-encoder/download")
        assert r.status_code == 200

        te_path = test_state.config.model_path("text_encoder")
        assert te_path.exists()

        downloading = test_state.config.downloading_dir
        assert not downloading.exists() or not any(downloading.iterdir())

    def test_failed_download_cleans_up_downloading_dir(self, test_state):
        """On download failure, .downloading/ is cleaned up."""
        test_state.model_downloader.fail_next = RuntimeError("network error")

        test_state.downloads.start_model_download()

        # The error handler should have been called
        assert len(test_state.task_runner.errors) == 1

        downloading = test_state.config.downloading_dir
        assert not downloading.exists()


class TestHuggingFaceInternals:
    """Guard tests for huggingface_hub internals we rely on.

    We monkey-patch ``file_download.http_get`` and ``file_download.xet_get``
    to inject a custom tqdm bar for progress tracking during
    ``hf_hub_download`` (which has no public ``tqdm_class`` parameter,
    unlike ``snapshot_download``).

    If these tests break after a huggingface_hub upgrade, the internal API
    has changed.  Find an alternative approach and raise to a developer.
    """

    def test_http_get_exists_and_is_callable(self):
        assert hasattr(file_download, "http_get"), (
            "file_download.http_get no longer exists — progress patch for hf_hub_download is broken"
        )
        assert callable(file_download.http_get)

    def test_http_get_accepts_tqdm_bar(self):
        sig = inspect.signature(file_download.http_get)
        assert "_tqdm_bar" in sig.parameters, (
            "file_download.http_get no longer accepts _tqdm_bar — progress patch for hf_hub_download is broken"
        )

    def test_xet_get_accepts_tqdm_bar(self):
        xet_get = getattr(file_download, "xet_get", None)
        if xet_get is None:
            return  # xet_get not present in this version; patch skips it gracefully
        sig = inspect.signature(xet_get)
        assert "_tqdm_bar" in sig.parameters, (
            "file_download.xet_get no longer accepts _tqdm_bar — progress patch for xet downloads is broken"
        )
