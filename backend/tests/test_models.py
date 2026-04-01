"""Integration-style tests for model-related endpoints."""

import inspect
from pathlib import Path

from huggingface_hub import file_download

from runtime_config.model_download_specs import resolve_downloading_dir, resolve_model_path
from state.app_state_types import DownloadingSession, FileDownloadRunning


def _model_path(test_state, model_type: str) -> Path:
    return resolve_model_path(
        test_state.config.default_models_dir,
        test_state.config.model_download_specs,
        model_type,
    )


def _downloading_dir(test_state) -> Path:
    return resolve_downloading_dir(test_state.config.default_models_dir)

DEFAULT_REQUIRED_MODEL_TYPES = ["checkpoint", "upsampler", "text_encoder", "zit"]
DEFAULT_REQUIRED_MODEL_TYPES_WITHOUT_TEXT_ENCODER = ["checkpoint", "upsampler", "zit"]


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
        assert all("id" in model for model in r.json()["models"])

    def test_ic_lora_is_optional_and_reported(self, client):
        r = client.get("/api/models/status")
        assert r.status_code == 200
        ic_lora = next(m for m in r.json()["models"] if m["id"] == "ic_lora")
        assert ic_lora["required"] is False

    def test_depth_person_detector_and_pose_are_optional_and_reported(self, client):
        r = client.get("/api/models/status")
        assert r.status_code == 200
        depth = next(m for m in r.json()["models"] if m["id"] == "depth_processor")
        person_detector = next(m for m in r.json()["models"] if m["id"] == "person_detector")
        pose = next(m for m in r.json()["models"] if m["id"] == "pose_processor")
        assert depth["required"] is False
        assert person_detector["required"] is False
        assert pose["required"] is False

    def test_all_downloaded(self, client, create_fake_model_files):
        create_fake_model_files(include_zit=True)
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
    def test_unknown_session_returns_404(self, client):
        r = client.get("/api/models/download/progress", params={"sessionId": "nonexistent"})
        assert r.status_code == 404

    def test_missing_session_id_returns_422(self, client):
        r = client.get("/api/models/download/progress")
        assert r.status_code == 422

    def test_active(self, client, test_state):
        test_state.state.downloading_session = DownloadingSession(
            id="test-session",
            current_running_file=FileDownloadRunning(
                file_type="checkpoint",
                target_path="checkpoint",
                downloaded_bytes=5_000_000_000,
                speed_bytes_per_sec=50_000_000.0,
            ),
            files_to_download={"checkpoint"},
            completed_files=set(),
            completed_bytes=0,
        )
        r = client.get("/api/models/download/progress", params={"sessionId": "test-session"})
        data = r.json()
        assert data["status"] == "downloading"
        assert data["current_downloading_file"] == "checkpoint"

    def test_completed_session(self, client, test_state):
        test_state.state.completed_download_sessions["done-session"] = "complete"
        r = client.get("/api/models/download/progress", params={"sessionId": "done-session"})
        data = r.json()
        assert data["status"] == "complete"

    def test_error_session(self, client, test_state):
        test_state.state.completed_download_sessions["err-session"] = "network error"
        r = client.get("/api/models/download/progress", params={"sessionId": "err-session"})
        data = r.json()
        assert data["status"] == "error"
        assert data["error"] == "network error"


class TestRequiredModels:
    def test_default_required_models(self, client):
        r = client.get("/api/models/required-models")
        assert r.status_code == 200
        assert r.json()["modelTypes"] == DEFAULT_REQUIRED_MODEL_TYPES

    def test_skip_text_encoder_flag(self, client):
        r = client.get("/api/models/required-models", params={"skipTextEncoder": "true"})
        assert r.status_code == 200
        assert r.json()["modelTypes"] == DEFAULT_REQUIRED_MODEL_TYPES_WITHOUT_TEXT_ENCODER

    def test_api_key_auto_excludes_text_encoder(self, client, test_state):
        test_state.state.app_settings.ltx_api_key = "test-key"
        r = client.get("/api/models/required-models")
        assert r.status_code == 200
        assert r.json()["modelTypes"] == DEFAULT_REQUIRED_MODEL_TYPES_WITHOUT_TEXT_ENCODER

    def test_forced_mode_returns_empty_set(self, client, test_state):
        test_state.config.force_api_generations = True
        test_state.config.required_model_types = frozenset()
        r = client.get("/api/models/required-models")
        assert r.status_code == 200
        assert r.json()["modelTypes"] == []


class TestModelDownload:
    def test_start_success(self, client, test_state):
        r = client.post("/api/models/download", json={"modelTypes": DEFAULT_REQUIRED_MODEL_TYPES})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "started"
        assert data["sessionId"] is not None

        snapshot_calls = [c for c in test_state.model_downloader.calls if c["kind"] == "snapshot"]
        assert snapshot_calls

    def test_already_in_progress(self, client, test_state):
        test_state.downloads.start_download({"checkpoint"})
        r = client.post("/api/models/download", json={"modelTypes": DEFAULT_REQUIRED_MODEL_TYPES})
        assert r.status_code == 409

    def test_download_without_text_encoder(self, client, test_state):
        r = client.post("/api/models/download", json={"modelTypes": DEFAULT_REQUIRED_MODEL_TYPES_WITHOUT_TEXT_ENCODER})
        assert r.status_code == 200

        te_spec = test_state.config.spec_for("text_encoder")
        te_calls = [c for c in test_state.model_downloader.calls if c.get("repo_id") == te_spec.repo_id]
        assert not te_calls, "text encoder download should have been skipped"

    def test_download_ic_lora_depth_person_detector_pose_bundle(self, client, test_state):
        bundle = ["ic_lora", "depth_processor", "person_detector", "pose_processor"]
        r = client.post("/api/models/download", json={"modelTypes": bundle})
        assert r.status_code == 200

        file_calls = [c for c in test_state.model_downloader.calls if c["kind"] == "file"]
        snapshot_calls = [c for c in test_state.model_downloader.calls if c["kind"] == "snapshot"]
        downloaded_filenames = {c["filename"] for c in file_calls}
        downloaded_repos = {c["repo_id"] for c in snapshot_calls}
        for model_type in bundle:
            spec = test_state.config.spec_for(model_type)
            if spec.is_folder:
                assert spec.repo_id in downloaded_repos
            else:
                assert spec.name in downloaded_filenames

    def test_empty_model_types_is_valid_noop(self, client, test_state):
        r = client.post("/api/models/download", json={"modelTypes": []})
        assert r.status_code == 200
        assert r.json()["status"] == "started"
        assert test_state.model_downloader.calls == []

    def test_forced_mode_downloads_no_local_models(self, client, test_state):
        test_state.config.force_api_generations = True
        test_state.config.required_model_types = frozenset()

        r = client.post("/api/models/download", json={"modelTypes": []})
        assert r.status_code == 200

        calls = test_state.model_downloader.calls
        assert len(calls) == 0


class TestTextEncoderDownload:
    def test_start_download(self, client):
        r = client.post("/api/text-encoder/download")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "started"
        assert data["sessionId"] is not None

    def test_already_downloaded(self, client, test_state):
        te_dir = _model_path(test_state,"text_encoder")
        te_dir.mkdir(parents=True, exist_ok=True)
        (te_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

        r = client.post("/api/text-encoder/download")
        assert r.status_code == 200
        assert r.json()["status"] == "already_downloaded"

    def test_already_in_progress(self, client, test_state):
        test_state.downloads.start_download({"checkpoint"})
        r = client.post("/api/text-encoder/download")
        assert r.status_code == 409


class TestDownloadProgressCallbacks:
    def test_download_passes_progress_callback(self, client, test_state):
        r = client.post("/api/models/download", json={"modelTypes": DEFAULT_REQUIRED_MODEL_TYPES})
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
        r = client.post("/api/models/download", json={"modelTypes": DEFAULT_REQUIRED_MODEL_TYPES})
        assert r.status_code == 200
        session_id = r.json()["sessionId"]

        # The fake downloader invokes on_progress(512, 1024) then on_progress(1024, 1024).
        # After download completes, each file is marked completed.
        # Verify that the download session was populated (files are now completed).
        r = client.get("/api/models/download/progress", params={"sessionId": session_id})
        data = r.json()
        assert data["status"] == "complete"

    def test_progress_callback_updates_running_state(self, test_state):
        """Directly invoke callback to verify it updates FileDownloadRunning state."""
        session_id = test_state.downloads.start_download({"checkpoint"})
        test_state.downloads.start_file("checkpoint", "checkpoint")
        cb = test_state.downloads._make_progress_callback("checkpoint")
        cb(5_000)

        r = test_state.downloads.get_download_progress(session_id)
        assert r.total_downloaded_bytes == 5_000
        assert r.current_downloading_file == "checkpoint"


class TestAtomicDownloads:
    """Verify downloads use .downloading/ staging dir and atomic moves."""

    def test_partial_file_in_downloading_dir_not_detected(self, test_state):
        """Files in .downloading/ must NOT be reported as downloaded."""
        downloading = _downloading_dir(test_state)
        downloading.mkdir(parents=True, exist_ok=True)
        (downloading / "ltx-2-19b-distilled-fp8.safetensors").write_bytes(b"\x00" * 1024)

        test_state.models.refresh_available_files()
        assert test_state.state.available_files["checkpoint"] is None

    def test_cleanup_downloading_dir_on_startup(self, test_state):
        """cleanup_downloading_dir() removes stale .downloading/ dir."""
        downloading = _downloading_dir(test_state)
        downloading.mkdir(parents=True, exist_ok=True)
        (downloading / "partial-file.safetensors").write_bytes(b"\x00" * 1024)

        test_state.downloads.cleanup_downloading_dir()
        assert not downloading.exists()

    def test_cleanup_downloading_dir_noop_when_absent(self, test_state):
        """cleanup_downloading_dir() is safe when dir doesn't exist."""
        test_state.downloads.cleanup_downloading_dir()
        assert not _downloading_dir(test_state).exists()

    def test_download_moves_files_to_final_location(self, client, test_state):
        """After download, files exist at final location, not in .downloading/."""
        r = client.post("/api/models/download", json={"modelTypes": DEFAULT_REQUIRED_MODEL_TYPES})
        assert r.status_code == 200

        # Files should be at their final locations
        assert _model_path(test_state,"checkpoint").exists()
        assert _model_path(test_state,"upsampler").exists()

        # .downloading/ should be gone (or empty)
        downloading = _downloading_dir(test_state)
        assert not downloading.exists() or not any(downloading.iterdir())

    def test_text_encoder_download_moves_to_final(self, client, test_state):
        """Text encoder download uses .downloading/ and moves to final."""
        r = client.post("/api/text-encoder/download")
        assert r.status_code == 200

        te_path = _model_path(test_state,"text_encoder")
        assert te_path.exists()

        downloading = _downloading_dir(test_state)
        assert not downloading.exists() or not any(downloading.iterdir())

    def test_failed_download_cleans_up_downloading_dir(self, test_state):
        """On download failure, .downloading/ is cleaned up."""
        test_state.model_downloader.fail_next = RuntimeError("network error")

        test_state.downloads.start_model_download({"checkpoint"})

        # The error handler should have been called
        assert len(test_state.task_runner.errors) == 1

        downloading = _downloading_dir(test_state)
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
