"""Integration-style tests for /api/suggest-gap-prompt, /api/retake, /api/extend."""

from __future__ import annotations

import logging
import uuid

from services.interfaces import HttpTransportError
from services.ltx_api_client.ltx_api_client import LTXAPIClientError, LTXRetakeResult
from tests.http_error_assertions import assert_http_error
from tests.fakes import FakeResponse

_LOCAL_2_3 = "ltx-2.3-22b-distilled-1.1"


def _install_local_2_3(test_state, create_fake_model_files, **kwargs) -> None:
    create_fake_model_files(model_id=_LOCAL_2_3, **kwargs)
    test_state.state.app_settings.active_ltx_model_id = _LOCAL_2_3


def _gemini_ok(text: str = "Enhanced prompt text") -> FakeResponse:
    return FakeResponse(
        status_code=200,
        json_payload={"candidates": [{"content": {"parts": [{"text": text}]}}]},
    )


def _gemini_error(status: int = 429, body: str = "rate limited") -> FakeResponse:
    return FakeResponse(status_code=status, text=body)


def _gemini_empty_candidates() -> FakeResponse:
    return FakeResponse(status_code=200, json_payload={"candidates": []})


class TestSuggestGapPrompt:
    def test_happy_path_with_prompts(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("post", _gemini_ok("A smooth transition scene"))

        r = client.post(
            "/api/suggest-gap-prompt",
            json={"beforePrompt": "sunset on a beach", "afterPrompt": "sunrise over mountains", "gapDuration": 3},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["suggested_prompt"] == "A smooth transition scene"
        assert "models/gemini-2.5-flash-lite:generateContent" in test_state.http.calls[-1].url

    def test_uses_configured_gemini_model(self, client, test_state, caplog):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.state.app_settings.gemini_model = "gemini-2.0-flash"
        test_state.http.queue("post", _gemini_ok("A smooth transition scene"))
        caplog.set_level(logging.INFO, logger="handlers.suggest_gap_prompt_handler")

        r = client.post(
            "/api/suggest-gap-prompt",
            json={"beforePrompt": "sunset on a beach", "afterPrompt": "sunrise over mountains", "gapDuration": 3},
        )
        assert r.status_code == 200
        url = test_state.http.calls[-1].url
        assert "models/gemini-2.0-flash:generateContent" in url
        assert "gemini-2.5-flash:" not in url
        assert any(
            record.getMessage() == "Suggesting gap prompt via Gemini API (gemini-2.0-flash)"
            for record in caplog.records
        )
        assert "thinkingConfig" not in test_state.http.calls[-1].json_payload["generationConfig"]

    def test_stored_non_text_gemini_model_uses_default(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.state.app_settings.gemini_model = "nano-banana-pro"
        test_state.http.queue("post", _gemini_ok("A smooth transition scene"))

        r = client.post(
            "/api/suggest-gap-prompt",
            json={"beforePrompt": "sunset on a beach", "afterPrompt": "sunrise over mountains", "gapDuration": 3},
        )
        assert r.status_code == 200
        assert "models/gemini-2.5-flash-lite:generateContent" in test_state.http.calls[-1].url
        assert test_state.http.calls[-1].json_payload["generationConfig"]["thinkingConfig"] == {
            "thinkingBudget": 0
        }

    def test_happy_path_with_frames(self, client, test_state, make_test_image, tmp_path):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("post", _gemini_ok("Transition clip"))

        before_path = tmp_path / "before.png"
        after_path = tmp_path / "after.png"
        before_path.write_bytes(make_test_image().getvalue())
        after_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/suggest-gap-prompt",
            json={"beforeFrame": str(before_path), "afterFrame": str(after_path)},
        )
        assert r.status_code == 200

        user_parts = test_state.http.calls[-1].json_payload["contents"][0]["parts"]
        inline_parts = [part for part in user_parts if "inlineData" in part]
        assert len(inline_parts) == 2

    def test_no_context_400(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        r = client.post("/api/suggest-gap-prompt", json={})
        assert r.status_code == 400

    def test_missing_gemini_key_400(self, client):
        r = client.post("/api/suggest-gap-prompt", json={"beforePrompt": "test"})
        assert_http_error(r, status_code=400, code="GEMINI_API_KEY_MISSING")

    def test_timeout_504(self, client, test_state):
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("post", HttpTransportError("timeout"))

        r = client.post("/api/suggest-gap-prompt", json={"beforePrompt": "test"})
        assert r.status_code == 504


class TestRetake:
    def _make_video(self, test_state) -> str:
        video_file = test_state.config.outputs_dir / f"retake_input_{uuid.uuid4().hex[:6]}.mp4"
        video_file.write_bytes(b"\x00" * 2048)
        return str(video_file)

    # ~3s by default (73 frames @ 24fps) so the retake selection in _base_payload fits
    # inside the (frame-count-corrected) clip; the selection is now clamped to it.
    def _make_valid_video(self, test_state, *, frames: int = 73, width: int = 64, height: int = 64, fps: int = 24) -> str:
        import numpy as np
        import imageio.v2 as imageio

        video_file = test_state.config.outputs_dir / f"retake_valid_{uuid.uuid4().hex[:6]}.mp4"
        writer = imageio.get_writer(str(video_file), fps=fps, codec="libx264", macro_block_size=None)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(frames):
            writer.append_data(frame)
        writer.close()
        return str(video_file)

    def _force_api(self, test_state) -> None:
        test_state.config.local_generations_mode = "unsupported"

    def _base_payload(self, video_path: str) -> dict[str, object]:
        return {
            "video_path": video_path,
            "start_time": 1.0,
            "duration": 3.0,
            "prompt": "make it dramatic",
        }

    def test_happy_path_binary_response(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.retake_result = LTXRetakeResult(
            video_bytes=b"\x00\x00\x00\x1cftypisom" + b"\x00" * 500,
            result_payload=None,
        )

        r = client.post("/api/retake", json=self._base_payload(video_path))
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert data["video_path"]

    def test_happy_path_json_video_url(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.retake_result = LTXRetakeResult(
            video_bytes=b"\x00\x00\x00\x1cftypisom" + b"\x00" * 500,
            result_payload=None,
        )

        r = client.post("/api/retake", json=self._base_payload(video_path))
        assert r.status_code == 200
        assert r.json()["status"] == "complete"

    def test_retake_defaults_to_ltx_2_3_pro(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.retake_result = LTXRetakeResult(
            video_bytes=b"\x00\x00\x00\x1cftypisom" + b"\x00" * 500,
            result_payload=None,
        )

        r = client.post("/api/retake", json=self._base_payload(video_path))
        assert r.status_code == 200
        assert len(test_state.ltx_api_client.retake_calls) == 1
        assert test_state.ltx_api_client.retake_calls[0]["model"] == "ltx-2-3-pro"

    def test_retake_rejects_2_5_model(self, client, test_state):
        # ltxv-api retake only accepts ltx-2-pro / ltx-2-3-pro.
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)

        r = client.post("/api/retake", json={**self._base_payload(video_path), "model": "pro-2.5"})
        assert r.status_code == 422

    def test_api_retake_recoverable_via_progress(self, client, test_state):
        # After an API retake, /generation/progress must report complete + the result
        # path, so a page that unmounted mid-generation can recover the output.
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.retake_result = LTXRetakeResult(
            video_bytes=b"\x00\x00\x00\x1cftypisom" + b"\x00" * 500,
            result_payload=None,
        )

        r = client.post("/api/retake", json=self._base_payload(video_path))
        assert r.status_code == 200
        result_path = r.json()["video_path"]

        progress = client.get("/api/generation/progress").json()
        assert progress["status"] == "complete"
        assert progress["result"] == result_path

    def test_api_retake_remote_payload_completes_without_recovery(self, client, test_state):
        # A remote payload is a success with no local file: the response is "complete"
        # and /progress reports complete with no result (nothing to recover) — not error.
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.retake_result = LTXRetakeResult(
            video_bytes=None,
            result_payload={"remote": "url"},
        )

        r = client.post("/api/retake", json=self._base_payload(video_path))
        assert r.status_code == 200
        assert r.json()["status"] == "complete"

        progress = client.get("/api/generation/progress").json()
        assert progress["status"] == "complete"
        assert progress["result"] is None

    def test_duration_too_short(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)

        r = client.post("/api/retake", json={"video_path": video_path, "start_time": 0, "duration": 1})
        assert r.status_code == 400

    def test_video_not_found(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        r = client.post("/api/retake", json={"video_path": "/nonexistent/video.mp4", "start_time": 0, "duration": 3})
        assert r.status_code == 400

    def test_no_api_key(self, client, test_state):
        self._force_api(test_state)
        video_path = self._make_video(test_state)
        r = client.post("/api/retake", json=self._base_payload(video_path))
        assert r.status_code == 400

    def test_rejects_fast_tier_model(self, client, test_state):
        # "fast" is not a RetakeExtendModel member, so pydantic rejects it (422)
        # before the request handler runs.
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        r = client.post("/api/retake", json={**self._base_payload(video_path), "model": "fast"})
        assert r.status_code == 422

    def test_upload_url_failure(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.raise_on_retake = LTXAPIClientError(401, "Failed to get upload URL: Unauthorized")

        r = client.post("/api/retake", json=self._base_payload(video_path))
        assert r.status_code == 401

    def test_video_upload_failure(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.raise_on_retake = LTXAPIClientError(500, "Video upload failed: Storage error")

        r = client.post("/api/retake", json=self._base_payload(video_path))
        assert r.status_code == 500

    def test_retake_api_422_safety_filter(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.raise_on_retake = LTXAPIClientError(422, "Content rejected by safety filters")

        r = client.post("/api/retake", json=self._base_payload(video_path))
        assert r.status_code == 422

    def test_prompt_and_mode_forwarded(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.retake_result = LTXRetakeResult(
            video_bytes=b"\x00\x00\x00\x1cftypisom" + b"\x00" * 500,
            result_payload=None,
        )

        client.post(
            "/api/retake",
            json={
                "video_path": video_path,
                "start_time": 2.0,
                "duration": 4.0,
                "prompt": "epic explosion",
                "mode": "replace_video",
            },
        )

        retake_call = test_state.ltx_api_client.retake_calls[-1]
        assert retake_call["prompt"] == "epic explosion"
        assert retake_call["mode"] == "replace_video"

    def test_local_retake_happy_path(self, client, test_state, create_fake_model_files):
        _install_local_2_3(test_state, create_fake_model_files, include_zit=False)
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.config.local_generations_mode = "full_models_loading"

        video_path = self._make_valid_video(test_state)
        r = client.post("/api/retake", json=self._base_payload(video_path))
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert data["video_path"]

    def test_local_retake_rejected_on_2_5(self, client, test_state, create_fake_model_files):
        create_fake_model_files(include_zit=False)
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.config.local_generations_mode = "full_models_loading"

        video_path = self._make_valid_video(test_state)
        r = client.post("/api/retake", json=self._base_payload(video_path))
        assert_http_error(
            r,
            status_code=409,
            code="UNSUPPORTED_RETAKE",
            message="Retake is not supported for the active LTX model.",
        )

    def test_local_retake_mode_mapping(self, client, test_state, create_fake_model_files, fake_services):
        _install_local_2_3(test_state, create_fake_model_files, include_zit=False)
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.config.local_generations_mode = "full_models_loading"

        video_path = self._make_valid_video(test_state)
        client.post(
            "/api/retake",
            json={
                "video_path": video_path,
                "start_time": 2.0,
                "duration": 4.0,
                "prompt": "epic explosion",
                "mode": "replace_video",
            },
        )
        retake_call = fake_services.retake_pipeline.generate_calls[-1]
        assert retake_call["regenerate_video"] is True
        assert retake_call["regenerate_audio"] is False

    def test_local_retake_forwards_selected_resolution(self, client, test_state, create_fake_model_files, fake_services):
        _install_local_2_3(test_state, create_fake_model_files, include_zit=False)
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.config.local_generations_mode = "full_models_loading"

        video_path = self._make_valid_video(test_state, width=256, height=256)
        client.post(
            "/api/retake",
            json={"video_path": video_path, "start_time": 0, "duration": 3, "resolution": {"width": 128, "height": 96}},
        )
        call = fake_services.retake_pipeline.generate_calls[-1]
        assert call["target_width"] == 128
        assert call["target_height"] == 96

    def test_prefers_api_video_routes_retake_to_api(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.retake_result = LTXRetakeResult(
            video_bytes=b"\x00\x00\x00\x1cftypisom" + b"\x00" * 500,
            result_payload=None,
        )

        r = client.post("/api/retake", json=self._base_payload(video_path))
        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(test_state.ltx_api_client.retake_calls) == 1
        assert len(fake_services.retake_pipeline.generate_calls) == 0

    def test_prefers_api_video_without_key_falls_back_to_local_retake(
        self,
        client,
        test_state,
        create_fake_model_files,
        fake_services,
    ):
        _install_local_2_3(test_state, create_fake_model_files, include_zit=False)
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True
        test_state.state.app_settings.ltx_api_key = ""
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = self._make_valid_video(test_state)
        r = client.post("/api/retake", json=self._base_payload(video_path))
        assert r.status_code == 200
        assert r.json()["status"] == "complete"
        assert len(test_state.ltx_api_client.retake_calls) == 0
        assert len(fake_services.retake_pipeline.generate_calls) == 1


class TestExtend:
    def _make_video(self, test_state) -> str:
        video_file = test_state.config.outputs_dir / f"extend_input_{uuid.uuid4().hex[:6]}.mp4"
        video_file.write_bytes(b"\x00" * 2048)
        return str(video_file)

    def _make_valid_video(self, test_state, *, frames: int = 9, width: int = 64, height: int = 64, fps: int = 24) -> str:
        import numpy as np
        import imageio.v2 as imageio

        video_file = test_state.config.outputs_dir / f"extend_valid_{uuid.uuid4().hex[:6]}.mp4"
        writer = imageio.get_writer(str(video_file), fps=fps, codec="libx264", macro_block_size=None)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(frames):
            writer.append_data(frame)
        writer.close()
        return str(video_file)

    def _force_api(self, test_state) -> None:
        test_state.config.local_generations_mode = "unsupported"

    def _base_payload(self, video_path: str) -> dict[str, object]:
        return {"video_path": video_path, "duration": 4.0, "prompt": "continue the motion"}

    def test_happy_path_binary_response(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.extend_result = LTXRetakeResult(
            video_bytes=b"\x00\x00\x00\x1cftypisom" + b"\x00" * 500,
            result_payload=None,
        )

        r = client.post("/api/extend", json=self._base_payload(video_path))
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert data["video_path"]

    def test_extend_defaults_to_ltx_2_3_pro(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.extend_result = LTXRetakeResult(
            video_bytes=b"\x00\x00\x00\x1cftypisom" + b"\x00" * 500,
            result_payload=None,
        )

        r = client.post("/api/extend", json=self._base_payload(video_path))
        assert r.status_code == 200
        assert len(test_state.ltx_api_client.extend_calls) == 1
        assert test_state.ltx_api_client.extend_calls[0]["model"] == "ltx-2-3-pro"

    def test_extend_rejects_2_5_model(self, client, test_state):
        # ltxv-api extend only accepts ltx-2-pro / ltx-2-3-pro.
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)

        r = client.post("/api/extend", json={**self._base_payload(video_path), "model": "pro-2.5"})
        assert r.status_code == 422

    def test_api_extend_recoverable_via_progress(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.extend_result = LTXRetakeResult(
            video_bytes=b"\x00\x00\x00\x1cftypisom" + b"\x00" * 500,
            result_payload=None,
        )

        r = client.post("/api/extend", json=self._base_payload(video_path))
        assert r.status_code == 200
        result_path = r.json()["video_path"]

        progress = client.get("/api/generation/progress").json()
        assert progress["status"] == "complete"
        assert progress["result"] == result_path

    def test_duration_too_short(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        r = client.post("/api/extend", json={"video_path": video_path, "duration": 1})
        assert r.status_code == 400

    def test_duration_too_long(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        r = client.post("/api/extend", json={"video_path": video_path, "duration": 21})
        assert r.status_code == 400

    def test_video_not_found(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        r = client.post("/api/extend", json={"video_path": "/nonexistent/video.mp4", "duration": 4})
        assert r.status_code == 400

    def test_negative_duration_rejected_as_422(self, client, test_state):
        # gt=0 rejects negative/zero duration at validation (clean 422) rather than letting
        # it slip into the frame-count math as a 500.
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        r = client.post("/api/extend", json={"video_path": video_path, "duration": -4})
        assert r.status_code == 422

    def test_nonpositive_resolution_rejected_as_422(self, client, test_state):
        # gt=0 on width/height rejects 0/negative instead of silently clamping to 32.
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        r = client.post(
            "/api/extend",
            json={"video_path": video_path, "duration": 4, "resolution": {"width": 0, "height": 720}},
        )
        assert r.status_code == 422

    def test_no_api_key(self, client, test_state):
        self._force_api(test_state)
        video_path = self._make_video(test_state)
        r = client.post("/api/extend", json=self._base_payload(video_path))
        assert r.status_code == 400

    def test_extend_api_422_safety_filter(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.raise_on_extend = LTXAPIClientError(422, "Content rejected by safety filters")

        r = client.post("/api/extend", json=self._base_payload(video_path))
        assert r.status_code == 422

    def test_prompt_and_mode_forwarded(self, client, test_state):
        self._force_api(test_state)
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.extend_result = LTXRetakeResult(
            video_bytes=b"\x00\x00\x00\x1cftypisom" + b"\x00" * 500,
            result_payload=None,
        )

        client.post(
            "/api/extend",
            json={"video_path": video_path, "duration": 6.0, "prompt": "intro shot", "mode": "start"},
        )

        extend_call = test_state.ltx_api_client.extend_calls[-1]
        assert extend_call["prompt"] == "intro shot"
        assert extend_call["mode"] == "start"
        assert extend_call["duration"] == 6.0

    def test_local_extend_happy_path(self, client, test_state, create_fake_model_files):
        _install_local_2_3(test_state, create_fake_model_files, include_zit=False)
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.config.local_generations_mode = "full_models_loading"

        video_path = self._make_valid_video(test_state)
        r = client.post("/api/extend", json=self._base_payload(video_path))
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert data["video_path"]

    def test_local_extend_rejected_on_2_5(self, client, test_state, create_fake_model_files):
        create_fake_model_files(include_zit=False)
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.config.local_generations_mode = "full_models_loading"

        video_path = self._make_valid_video(test_state)
        r = client.post("/api/extend", json=self._base_payload(video_path))
        assert_http_error(
            r,
            status_code=409,
            code="UNSUPPORTED_EXTEND",
            message="Extend is not supported for the active LTX model.",
        )

    def test_local_extend_snaps_frames_and_forwards_mode(self, client, test_state, create_fake_model_files, fake_services):
        _install_local_2_3(test_state, create_fake_model_files, include_zit=False)
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.config.local_generations_mode = "full_models_loading"

        video_path = self._make_valid_video(test_state, fps=24)
        client.post(
            "/api/extend",
            json={"video_path": video_path, "duration": 4.0, "prompt": "more", "mode": "start"},
        )
        extend_call = fake_services.retake_pipeline.extend_calls[-1]
        assert extend_call["mode"] == "start"
        # 4s * 24fps = 96 frames, already a multiple of 8.
        assert extend_call["extend_frames"] == 96
        assert extend_call["extend_frames"] % 8 == 0

    def test_local_extend_corrects_source_resolution_to_div32(self, client, test_state, create_fake_model_files, fake_services):
        _install_local_2_3(test_state, create_fake_model_files, include_zit=False)
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.config.local_generations_mode = "full_models_loading"

        # 160x90 source, no resolution selected → "Original", corrected to ÷32 (90 -> 64).
        video_path = self._make_valid_video(test_state, width=160, height=90)
        client.post("/api/extend", json={"video_path": video_path, "duration": 4.0, "prompt": "go", "mode": "end"})
        call = fake_services.retake_pipeline.extend_calls[-1]
        assert call["target_width"] == 160
        assert call["target_height"] == 64

    def test_local_extend_corrects_frame_count(self, client, test_state, create_fake_model_files, fake_services):
        _install_local_2_3(test_state, create_fake_model_files, include_zit=False)
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.config.local_generations_mode = "full_models_loading"

        # 12 frames is not 8k+1; it is trimmed down to 9 instead of being rejected.
        video_path = self._make_valid_video(test_state, frames=12)
        r = client.post("/api/extend", json={"video_path": video_path, "duration": 4.0, "mode": "end"})
        assert r.status_code == 200
        call = fake_services.retake_pipeline.extend_calls[-1]
        assert call["target_frames"] == 9

    def test_local_extend_forwards_selected_resolution(self, client, test_state, create_fake_model_files, fake_services):
        _install_local_2_3(test_state, create_fake_model_files, include_zit=False)
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.config.local_generations_mode = "full_models_loading"

        video_path = self._make_valid_video(test_state, width=256, height=256)
        client.post(
            "/api/extend",
            json={"video_path": video_path, "duration": 4.0, "mode": "end", "resolution": {"width": 128, "height": 128}},
        )
        call = fake_services.retake_pipeline.extend_calls[-1]
        assert call["target_width"] == 128
        assert call["target_height"] == 128

    def test_prefers_api_video_routes_extend_to_api(self, client, test_state, fake_services):
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True
        test_state.state.app_settings.ltx_api_key = "test-key"
        video_path = self._make_video(test_state)
        test_state.ltx_api_client.extend_result = LTXRetakeResult(
            video_bytes=b"\x00\x00\x00\x1cftypisom" + b"\x00" * 500,
            result_payload=None,
        )

        r = client.post("/api/extend", json=self._base_payload(video_path))
        assert r.status_code == 200
        assert len(test_state.ltx_api_client.extend_calls) == 1
        assert len(fake_services.retake_pipeline.extend_calls) == 0

    def test_prefers_api_video_without_key_falls_back_to_local_extend(
        self,
        client,
        test_state,
        create_fake_model_files,
        fake_services,
    ):
        _install_local_2_3(test_state, create_fake_model_files, include_zit=False)
        test_state.config.local_generations_mode = "full_models_loading"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True
        test_state.state.app_settings.ltx_api_key = ""
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = self._make_valid_video(test_state)
        r = client.post("/api/extend", json=self._base_payload(video_path))
        assert r.status_code == 200
        assert len(test_state.ltx_api_client.extend_calls) == 0
        assert len(fake_services.retake_pipeline.extend_calls) == 1
