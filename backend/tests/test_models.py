"""Integration-style tests for checkpoint recommendation and download endpoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from _routes._errors import HTTPError
import handlers.models_handler as models_handler_module
from runtime_config.model_download_specs import (
    ALL_LTX_LOCAL_MODEL_IDS,
    DEPTH_PROCESSOR_CP_ID,
    IMG_GEN_MODEL_CP_ID,
    LTXLocalModelDeprecated,
    get_ic_loras_cp_ids,
    get_latest_ltx_model_id,
    get_ltx_model_spec,
    get_model_cp_spec,
    resolve_active_ltx_model_id,
    resolve_downloading_dir,
    resolve_model_path,
)
from state.app_state_types import DownloadSessionComplete, DownloadSessionError, DownloadingSession, FileDownloadRunning
from tests.http_error_assertions import assert_http_error


def _current_ltx_spec():
    return get_ltx_model_spec(get_latest_ltx_model_id())


def _cp_path(test_state, cp_id: str) -> Path:
    return resolve_model_path(test_state.config.default_models_dir, cp_id)


class TestRecommendations:
    def test_ltx_recommendation_requires_primary_local_bundle(self, client):
        spec = _current_ltx_spec()
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "status": "download",
            "cps_to_download": [
                spec.model_cp,
                spec.upscale_cp,
                spec.text_encoder_cp,
            ],
        }

    def test_ltx_recommendation_skips_text_encoder_when_api_key_exists(self, client, test_state):
        test_state.state.app_settings.ltx_api_key = "test-key"
        spec = _current_ltx_spec()
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "status": "download",
            "cps_to_download": [
                spec.model_cp,
                spec.upscale_cp,
            ],
        }

    def test_ltx_recommendation_ok_when_required_bundle_is_downloaded(self, client, create_fake_model_files):
        create_fake_model_files()
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_recommendation_surfaces_missing_shared_companion_for_current_base(self, client, test_state):
        # Existing-user scenario: an older base transformer is on disk but a required shared
        # companion (the hotfixed 2x upscaler) is missing. This must surface as a 'download'
        # for the upscaler — not be buried behind a base 'upgrade' — so the missing-models gate
        # prompts it. (API key set so the text encoder isn't also required, isolating the upscaler.)
        test_state.state.app_settings.ltx_api_key = "test-key"
        older_spec = get_ltx_model_spec("ltx-2.3-22b-distilled")
        base_path = _cp_path(test_state, older_spec.model_cp)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_bytes(b"\x00" * 1024)

        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "status": "download",
            "cps_to_download": [older_spec.upscale_cp],
        }

    def test_ltx_recommendation_reports_missing_text_encoder_for_current_model(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        text_encoder_path = _cp_path(test_state, _current_ltx_spec().text_encoder_cp)
        for child in text_encoder_path.iterdir():
            child.unlink()
        text_encoder_path.rmdir()

        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "status": "download",
            "cps_to_download": [_current_ltx_spec().text_encoder_cp],
        }

    def test_img_gen_recommendation(self, client, create_fake_model_files):
        response = client.get("/api/models/img-gen-recommendation")
        assert response.status_code == 200
        assert response.json()["cp_to_download"] == IMG_GEN_MODEL_CP_ID

        create_fake_model_files(include_zit=True)
        response = client.get("/api/models/img-gen-recommendation")
        assert response.status_code == 200
        assert response.json()["cp_to_download"] is None

    def test_text_encoder_recommendation(self, client, create_fake_model_files, test_state):
        create_fake_model_files()
        text_encoder_path = _cp_path(test_state, _current_ltx_spec().text_encoder_cp)
        for child in text_encoder_path.iterdir():
            child.unlink()
        text_encoder_path.rmdir()

        response = client.get("/api/models/text-encoder-recommendation")
        assert response.status_code == 200
        assert response.json()["cp_to_download"] == _current_ltx_spec().text_encoder_cp
        assert response.json()["expected_size_bytes"] > 0

    def test_describe_checkpoints(self, client, create_fake_model_files):
        spec = _current_ltx_spec()
        create_fake_model_files()  # base bundle on disk; upscaler not part of fake bundle by default
        response = client.post(
            "/api/models/describe",
            json={"cp_ids": [IMG_GEN_MODEL_CP_ID, spec.text_encoder_cp, spec.model_cp]},
        )
        assert response.status_code == 200
        checkpoints = response.json()["checkpoints"]
        by_id = {c["cp_id"]: c for c in checkpoints}

        assert by_id[spec.model_cp]["role"] == "base"
        assert by_id[spec.text_encoder_cp]["role"] == "text_encoder"
        assert by_id[IMG_GEN_MODEL_CP_ID]["role"] == "image"
        for cp in checkpoints:
            assert cp["size_bytes"] > 0
            # info copy moved to the frontend (keyed off role); backend only ships name + role.
            assert cp["name"] and cp["role"]
        assert by_id[spec.model_cp]["downloaded"] is True
        assert by_id[IMG_GEN_MODEL_CP_ID]["downloaded"] is False

    def test_describe_classifies_any_base_version_as_base(self, client):
        # An OLDER base transformer must be classified as "base", not "support" — _cp_role
        # has to match every version's model_cp, not only the latest.
        older = get_ltx_model_spec("ltx-2.3-22b-distilled")
        assert older.model_cp != _current_ltx_spec().model_cp
        response = client.post("/api/models/describe", json={"cp_ids": [older.model_cp]})
        assert response.status_code == 200
        cp = response.json()["checkpoints"][0]
        assert cp["cp_id"] == older.model_cp
        assert cp["role"] == "base"

    def test_ic_lora_recommendation(self, client, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        response = client.get("/api/models/ltx-ic-lora-recommendation")
        assert response.status_code == 200
        assert response.json()["cps_to_download"] == [
            *get_ic_loras_cp_ids(_current_ltx_spec().ic_loras_spec),
            DEPTH_PROCESSOR_CP_ID,
        ]

        create_fake_ic_lora_files()
        response = client.get("/api/models/ltx-ic-lora-recommendation")
        assert response.status_code == 200
        assert response.json()["cps_to_download"] == []


class TestDownloadProgress:
    def test_unknown_session_returns_404(self, client):
        response = client.get("/api/models/download/progress", params={"sessionId": "nonexistent"})
        assert_http_error(response, status_code=404, code="UNKNOWN_DOWNLOAD_SESSION")

    def test_active_progress(self, client, test_state):
        test_state.state.downloading_session = DownloadingSession(
            id="test-session",
            current_running_file=FileDownloadRunning(
                file_type="ltx-2.3-22b-distilled",
                target_path="ltx-2.3-22b-distilled.safetensors",
                downloaded_bytes=5_000_000_000,
                speed_bytes_per_sec=50_000_000.0,
            ),
            files_to_download={"ltx-2.3-22b-distilled"},
            completed_files=set(),
            completed_bytes=0,
        )
        response = client.get("/api/models/download/progress", params={"sessionId": "test-session"})
        assert response.status_code == 200
        assert response.json()["status"] == "downloading"
        assert response.json()["current_downloading_file"] == "ltx-2.3-22b-distilled"

    def test_active_download_reports_running_session(self, client, test_state):
        test_state.state.downloading_session = DownloadingSession(
            id="dl-1",
            current_running_file=None,
            files_to_download={"ltx-2.3-22b-distilled-1.1"},
            completed_files=set(),
            completed_bytes=0,
        )
        response = client.get("/api/models/download/active")
        assert response.status_code == 200
        assert response.json()["session_id"] == "dl-1"
        assert "ltx-2.3-22b-distilled-1.1" in response.json()["cp_ids"]

    def test_active_download_null_when_idle(self, client):
        response = client.get("/api/models/download/active")
        assert response.status_code == 200
        assert response.json()["session_id"] is None
        assert response.json()["cp_ids"] == []

    def test_completed_and_error_sessions(self, client, test_state):
        test_state.state.completed_download_sessions["done-session"] = DownloadSessionComplete()
        test_state.state.completed_download_sessions["err-session"] = DownloadSessionError(error_message="network error")

        complete = client.get("/api/models/download/progress", params={"sessionId": "done-session"})
        assert complete.status_code == 200
        assert complete.json()["status"] == "complete"

        failed = client.get("/api/models/download/progress", params={"sessionId": "err-session"})
        assert failed.status_code == 200
        assert failed.json()["status"] == "error"
        assert failed.json()["error"] == "network error"


class TestModelDownloads:
    def test_download_start_success(self, client, test_state):
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "started"
        assert _cp_path(test_state, IMG_GEN_MODEL_CP_ID).exists()

    def test_download_conflicts_when_another_session_is_running(self, client, test_state):
        test_state.downloads.start_download({"ltx-2.3-22b-distilled"})
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert_http_error(response, status_code=409, code="DOWNLOAD_ALREADY_RUNNING")

    def test_upgrade_without_downloaded_model_is_rejected(self, client):
        response = client.post(
            "/api/models/download",
            json={"type": "upgrade", "cp_ids": [_current_ltx_spec().model_cp]},
        )
        assert_http_error(response, status_code=409, code="NO_DOWNLOADED_LTX_MODEL")

    def test_upgrade_raises_500_for_internal_ltx_mapping_inconsistency(self, test_state, monkeypatch):
        monkeypatch.setattr(test_state.models, "_current_downloaded_ltx_model_id", lambda: "ltx-legacy")
        monkeypatch.setattr(models_handler_module, "get_ltx_model_id_for_cp", lambda cp_id: None)

        with pytest.raises(HTTPError) as exc_info:
            test_state.models.resolve_upgrade_download({_current_ltx_spec().model_cp})

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "INVALID_LTX_MODEL_CONFIG"

    def test_upgrade_raises_500_when_latest_ltx_model_is_not_relevant(self, test_state, monkeypatch):
        monkeypatch.setattr(test_state.models, "_current_downloaded_ltx_model_id", lambda: "ltx-legacy")
        monkeypatch.setattr(models_handler_module, "get_latest_ltx_model_id", lambda: "ltx-2.3-22b-distilled")
        monkeypatch.setattr(models_handler_module, "get_ltx_model_id_for_cp", lambda cp_id: "ltx-2.3-22b-distilled")

        original_get_ltx_model_spec = models_handler_module.get_ltx_model_spec

        def _get_ltx_model_spec(model_id):
            spec = original_get_ltx_model_spec(model_id)
            if model_id == "ltx-2.3-22b-distilled":
                return replace(spec, relevance=LTXLocalModelDeprecated())
            return spec

        monkeypatch.setattr(models_handler_module, "get_ltx_model_spec", _get_ltx_model_spec)

        with pytest.raises(HTTPError) as exc_info:
            test_state.models.resolve_upgrade_download({_current_ltx_spec().model_cp})

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "INVALID_LTX_MODEL_CONFIG"

    def test_download_error_is_reported(self, client, test_state):
        test_state.model_downloader.fail_next = RuntimeError("Connection refused")

        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert response.status_code == 200
        session_id = response.json()["sessionId"]

        progress = client.get("/api/models/download/progress", params={"sessionId": session_id})
        assert progress.status_code == 200
        assert progress.json()["status"] == "error"

    def test_download_uses_progress_callback(self, client, test_state):
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert response.status_code == 200
        assert test_state.model_downloader.calls
        assert all(call["on_progress"] is not None for call in test_state.model_downloader.calls)

    def test_failed_download_cleans_staging_dir(self, test_state):
        test_state.model_downloader.fail_next = RuntimeError("network error")
        test_state.downloads.start_model_download(download_type="download", cp_ids={IMG_GEN_MODEL_CP_ID})
        assert len(test_state.task_runner.errors) == 1
        assert not resolve_downloading_dir(test_state.config.default_models_dir).exists()


class TestCheckpointDeletion:
    def test_delete_missing_checkpoint_is_noop(self, client):
        response = client.request(
            "DELETE",
            "/api/models/delete",
            json={"cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_delete_rejects_current_ltx_bundle(self, client, create_fake_model_files):
        create_fake_model_files()
        response = client.request(
            "DELETE",
            "/api/models/delete",
            json={"cp_ids": [_current_ltx_spec().model_cp]},
        )
        assert_http_error(response, status_code=409, code="DELETE_PROTECTED_CHECKPOINT")

    def test_delete_removes_non_protected_checkpoint(self, client, test_state):
        img_gen_path = _cp_path(test_state, IMG_GEN_MODEL_CP_ID)
        img_gen_path.mkdir(parents=True, exist_ok=True)
        (img_gen_path / "model.safetensors").write_bytes(b"\x00" * 1024)

        response = client.request(
            "DELETE",
            "/api/models/delete",
            json={"cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert not img_gen_path.exists()


class TestLtxVersions:
    def test_two_base_versions_registered_newest_first(self):
        assert ALL_LTX_LOCAL_MODEL_IDS[0] == "ltx-2.3-22b-distilled-1.1"
        assert "ltx-2.3-22b-distilled" in ALL_LTX_LOCAL_MODEL_IDS
        assert get_latest_ltx_model_id() == "ltx-2.3-22b-distilled-1.1"

    def test_versions_share_companions(self):
        v11 = get_ltx_model_spec("ltx-2.3-22b-distilled-1.1")
        v10 = get_ltx_model_spec("ltx-2.3-22b-distilled")
        assert v11.upscale_cp == v10.upscale_cp
        assert v11.text_encoder_cp == v10.text_encoder_cp
        assert v11.ic_loras_spec == v10.ic_loras_spec
        assert v11.model_cp != v10.model_cp

    def test_version_labels(self):
        assert get_ltx_model_spec("ltx-2.3-22b-distilled-1.1").version_label == "1.1"
        assert get_ltx_model_spec("ltx-2.3-22b-distilled").version_label == "1.0"

    def test_1_1_checkpoint_spec(self):
        spec = get_model_cp_spec("ltx-2.3-22b-distilled-1.1")
        assert spec.repo_id == "Lightricks/LTX-2.3"
        assert str(spec.relative_path) == "ltx-2.3-22b-distilled-1.1.safetensors"
        assert spec.expected_size_bytes == 46_149_345_334

    def test_distilled_1_1_has_upgrade_notes(self):
        # The 1.1 spec carries authored "what's new" notes for the 1.0 -> 1.1 upgrade prompt.
        from runtime_config.model_download_specs import LTXLocalModelRelevant
        relevance = get_ltx_model_spec("ltx-2.3-22b-distilled-1.1").relevance
        assert isinstance(relevance, LTXLocalModelRelevant)
        assert relevance.upgrade_messages.get("ltx-2.3-22b-distilled")


class TestActiveModelResolution:
    def _write_transformer(self, test_state, model_id: str) -> None:
        # resolve_active_ltx_model_id requires the full generation bundle (transformer +
        # upscaler), so write both — otherwise the version isn't considered runnable.
        spec = get_ltx_model_spec(model_id)
        for cp in (spec.model_cp, spec.upscale_cp):
            path = resolve_model_path(test_state.config.default_models_dir, cp)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * 1024)

    def test_prefers_explicit_when_installed(self, test_state):
        models_dir = test_state.config.default_models_dir
        self._write_transformer(test_state, "ltx-2.3-22b-distilled")
        self._write_transformer(test_state, "ltx-2.3-22b-distilled-1.1")
        assert resolve_active_ltx_model_id(models_dir, "ltx-2.3-22b-distilled") == "ltx-2.3-22b-distilled"

    def test_falls_back_to_newest_installed_when_preferred_missing(self, test_state):
        models_dir = test_state.config.default_models_dir
        self._write_transformer(test_state, "ltx-2.3-22b-distilled")
        # preferred 1.1 is NOT on disk -> fall back to the only installed (1.0)
        assert resolve_active_ltx_model_id(models_dir, "ltx-2.3-22b-distilled-1.1") == "ltx-2.3-22b-distilled"

    def test_none_when_nothing_installed(self, test_state):
        assert resolve_active_ltx_model_id(test_state.config.default_models_dir, None) is None

    def test_generation_uses_active_setting(self, client, test_state):
        models_dir = test_state.config.default_models_dir
        self._write_transformer(test_state, "ltx-2.3-22b-distilled")
        self._write_transformer(test_state, "ltx-2.3-22b-distilled-1.1")
        test_state.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled"
        resolved = resolve_active_ltx_model_id(models_dir, test_state.state.app_settings.active_ltx_model_id)
        assert resolved == "ltx-2.3-22b-distilled"


class TestLtxVersionEndpoints:
    def _write_transformer(self, test_state, model_id: str) -> None:
        cp = get_ltx_model_spec(model_id).model_cp
        path = resolve_model_path(test_state.config.default_models_dir, cp)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 1024)

    def test_versions_list_newest_first_with_flags(self, client, test_state, create_fake_model_files):
        # create_fake_model_files installs the latest (1.1) bundle
        create_fake_model_files()
        test_state.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled-1.1"
        response = client.get("/api/models/ltx-versions")
        assert response.status_code == 200
        versions = response.json()["versions"]
        assert [v["model_id"] for v in versions] == [
            "ltx-2.3-22b-distilled-1.1",
            "ltx-2.3-22b-distilled",
        ]
        newest, older = versions[0], versions[1]
        assert newest["label"] == "1.1"
        assert newest["installed"] is True
        assert newest["active"] is True
        assert older["installed"] is False
        assert older["is_newest"] is False
        assert newest["is_newest"] is True
        # an uninstalled version reports the cp(s) it still needs
        assert "ltx-2.3-22b-distilled" in older["cps_to_download"]

    def test_set_active_requires_installed(self, client, test_state):
        # 1.0 is not on disk -> cannot activate
        response = client.post(
            "/api/models/active-ltx-model", json={"model_id": "ltx-2.3-22b-distilled"}
        )
        assert response.status_code == 409
        assert response.json()["code"] == "LTX_MODEL_NOT_INSTALLED"

    def test_set_active_rejects_when_companions_missing(self, client, test_state):
        # Transformer present but the required upscaler/text-encoder are missing -> the
        # version isn't runnable, so activating it must be rejected (not a broken active state).
        self._write_transformer(test_state, "ltx-2.3-22b-distilled")
        response = client.post(
            "/api/models/active-ltx-model", json={"model_id": "ltx-2.3-22b-distilled"}
        )
        assert response.status_code == 409
        assert response.json()["code"] == "LTX_MODEL_NOT_INSTALLED"

    def test_set_active_succeeds_and_persists(self, client, test_state):
        import json
        # Activation requires the full required bundle. With an API key the text encoder
        # isn't required, so transformer + upscaler is a complete, runnable version.
        test_state.state.app_settings.ltx_api_key = "test-key"
        spec = get_ltx_model_spec("ltx-2.3-22b-distilled")
        for cp in (spec.model_cp, spec.upscale_cp):
            path = resolve_model_path(test_state.config.default_models_dir, cp)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * 1024)
        response = client.post(
            "/api/models/active-ltx-model", json={"model_id": "ltx-2.3-22b-distilled"}
        )
        assert response.status_code == 200
        # in-memory state updated
        assert test_state.state.app_settings.active_ltx_model_id == "ltx-2.3-22b-distilled"
        # AND persisted to disk (the hardening: must go through SettingsHandler.save_settings)
        saved = json.loads(test_state.config.settings_file.read_text())
        assert saved["active_ltx_model_id"] == "ltx-2.3-22b-distilled"


class TestDeleteGuard:
    def _write_transformer(self, test_state, model_id: str) -> None:
        cp = get_ltx_model_spec(model_id).model_cp
        path = resolve_model_path(test_state.config.default_models_dir, cp)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 1024)

    def test_cannot_delete_active_version_transformer(self, client, test_state, create_fake_model_files):
        create_fake_model_files()  # installs 1.1 bundle
        self._write_transformer(test_state, "ltx-2.3-22b-distilled")  # 1.0 also present
        test_state.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled-1.1"
        response = client.request(
            "DELETE", "/api/models/delete", json={"cp_ids": ["ltx-2.3-22b-distilled-1.1"]}
        )
        assert response.status_code == 409
        assert response.json()["code"] == "DELETE_PROTECTED_CHECKPOINT"

    def test_can_delete_non_active_version_transformer(self, client, test_state, create_fake_model_files):
        create_fake_model_files()  # 1.1 bundle, active by default resolution
        self._write_transformer(test_state, "ltx-2.3-22b-distilled")  # 1.0 present, not active
        test_state.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled-1.1"
        response = client.request(
            "DELETE", "/api/models/delete", json={"cp_ids": ["ltx-2.3-22b-distilled"]}
        )
        assert response.status_code == 200

    def test_active_older_version_protected_newer_deletable(self, client, test_state, create_fake_model_files):
        # Both versions installed; active is the OLDER 1.0.
        # Old impl (protected = newest installed = 1.1) would ALLOW deleting 1.0 and BLOCK deleting 1.1.
        # New impl (protected = active = 1.0) must BLOCK deleting 1.0 and ALLOW deleting 1.1.
        create_fake_model_files()  # installs 1.1 bundle
        self._write_transformer(test_state, "ltx-2.3-22b-distilled")  # 1.0 also present
        test_state.state.app_settings.active_ltx_model_id = "ltx-2.3-22b-distilled"  # active = 1.0

        # Active (1.0) must be protected
        response_protected = client.request(
            "DELETE", "/api/models/delete", json={"cp_ids": ["ltx-2.3-22b-distilled"]}
        )
        assert response_protected.status_code == 409
        assert response_protected.json()["code"] == "DELETE_PROTECTED_CHECKPOINT"

        # Non-active newer (1.1) must be deletable
        response_allowed = client.request(
            "DELETE", "/api/models/delete", json={"cp_ids": ["ltx-2.3-22b-distilled-1.1"]}
        )
        assert response_allowed.status_code == 200
