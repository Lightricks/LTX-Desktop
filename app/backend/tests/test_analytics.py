from __future__ import annotations

import json
from pathlib import Path

import pytest

from api_model_specs import video_generation_feature
from handlers.resolved_generation_prompt import ResolvedGenerationPrompt, prompt_provenance_label
from services.analytics import (
    ENDED_DETAIL_KEYS,
    AnalyticsService,
    QueuedEnhancement,
    queued_generation_details,
)
from services.records import GenerationRecord
from services.sqlite_store import SqliteStore
from tests.fakes import FakeResponse
from tests.fakes.services import FakeHTTPClient, FakeTaskRunner


def _service(http: FakeHTTPClient, *, sink_url: str = "http://127.0.0.1:9/analytics") -> AnalyticsService:
    return AnalyticsService(
        http=http,
        task_runner=FakeTaskRunner(),
        sink_url=sink_url,
        token="sink-token",
    )


def test_missing_sink_does_not_emit() -> None:
    http = FakeHTTPClient()
    service = _service(http, sink_url="")

    with service.generation({"generation_id": "generation-1"}):
        pass

    assert http.calls == []


def test_home_audio_to_video_derives_duration_sec_from_num_frames() -> None:
    record = _home_record(
        feature="audio-to-video",
        params={"model": "ltx-2.5", "numFrames": 121, "fps": 24},
    )

    details = queued_generation_details(record)
    assert details is not None
    assert details["duration_sec"] == 5.0
    assert details["fps"] == 24


def test_forwarded_body_is_only_the_event_name_and_details() -> None:
    http = FakeHTTPClient()
    http.queue("post", FakeResponse())
    service = _service(http)

    service.send_generation_started(
        {"generation_id": "generation-1", "prompt": "a fox"}
    )

    assert len(http.calls) == 1
    call = http.calls[0]
    assert call.url == "http://127.0.0.1:9/analytics"
    assert call.headers is not None
    assert call.headers["Authorization"] == "Bearer sink-token"
    assert call.json_payload == {
        "eventName": "generate_started",
        "extraDetails": {"generation_id": "generation-1", "prompt": "a fox"},
    }
    assert call.json_payload is not None
    assert "events" not in call.json_payload
    assert "ltx-desktop.lightricks.com" not in call.url


def test_successful_generation_emits_one_matching_start_and_end() -> None:
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(), FakeResponse())
    service = _service(http)

    with service.generation(
        {
            "generation_id": "generation-1",
            "surface": "genspace",
            "feature": "text-to-video",
        }
    ):
        pass

    payloads = [call.json_payload for call in http.calls]
    assert [payload["eventName"] for payload in payloads if payload is not None] == [
        "generate_started",
        "generate_ended",
    ]
    start = payloads[0]
    end = payloads[1]
    assert start is not None and end is not None
    start_details = start["extraDetails"]
    end_details = end["extraDetails"]
    assert isinstance(start_details, dict)
    assert isinstance(end_details, dict)
    assert start_details["surface"] == "genspace"
    assert start_details["feature"] == "text-to-video"
    assert set(end_details) == {"generation_id", "outcome", "runtime_ms"}
    assert end_details["generation_id"] == start_details["generation_id"]
    assert end_details["outcome"] == "succeeded"
    assert isinstance(end_details["runtime_ms"], int)
    assert end_details["runtime_ms"] >= 0


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_terminal_generation_outcome_is_not_reported_as_succeeded(outcome: str) -> None:
    http = FakeHTTPClient()
    http.queue("post", FakeResponse(), FakeResponse())
    service = _service(http)

    if outcome == "failed":
        with pytest.raises(RuntimeError):
            with service.generation({"generation_id": "generation-1"}):
                raise RuntimeError("generation failed")
    else:
        with service.generation({"generation_id": "generation-1"}) as tracker:
            tracker.finish("cancelled")

    end = http.calls[1].json_payload
    assert end is not None
    details = end["extraDetails"]
    assert isinstance(details, dict)
    assert set(details) == {"generation_id", "outcome", "runtime_ms"}
    assert details["outcome"] == outcome


def test_ended_event_keeps_the_join_key_and_drops_start_fields() -> None:
    http = FakeHTTPClient()
    http.queue("post", FakeResponse())
    service = _service(http)

    service.send_generation_ended(
        {
            "generation_id": "generation-1",
            "attempt": 2,
            "surface": "home",
            "feature": "cozy-felt",
            "prompt_provenance": "auto-enhanced",
        },
        outcome="failed",
        runtime_ms=10,
        error_code="EXECUTOR_FAILED",
    )

    payload = http.calls[0].json_payload
    assert payload is not None
    assert payload["extraDetails"] == {
        "generation_id": "generation-1",
        "attempt": 2,
        "outcome": "failed",
        "runtime_ms": 10,
        "error_code": "EXECUTOR_FAILED",
    }


def test_prompt_provenance_label_names_the_three_modes() -> None:
    manual = ResolvedGenerationPrompt("rewritten", False, False)
    via_api = ResolvedGenerationPrompt("a fox", True, False)
    via_local = ResolvedGenerationPrompt("a fox", False, True)
    untouched = ResolvedGenerationPrompt("a fox", False, False)

    assert prompt_provenance_label("enhanced", manual) == "manually-enhanced"
    assert prompt_provenance_label("typed", via_api) == "auto-enhanced"
    assert prompt_provenance_label("typed", via_local) == "auto-enhanced"
    assert prompt_provenance_label("typed", untouched) == "raw"


_LOCAL_ENHANCE = QueuedEnhancement(
    explore_auto_enhance_prompts=True,
    prompt_enhancer_enabled=True,
    use_local_encoding=True,
)
_API_ENHANCE = QueuedEnhancement(
    explore_auto_enhance_prompts=True,
    prompt_enhancer_enabled=True,
    use_local_encoding=False,
)


def test_ended_detail_keys_match_the_shared_contract() -> None:
    contract_path = Path(__file__).resolve().parents[2] / "shared" / "analytics-ended-details.json"
    assert json.loads(contract_path.read_text()) == list(ENDED_DETAIL_KEYS)


def test_started_detail_keys_match_the_shared_contract() -> None:
    """Every started key the backend emits must survive Electron's allowlist.

    A writer key missing from shared/generation-detail-keys.json is silently
    dropped from generate_started; a contract key nothing emits is dead.
    outcome/runtime_ms/error_code/output_count/prompt_rewrite are ended-only by design.
    """
    record = _home_record(
        feature="text-to-video",
        params={
            "prompt": "a fox",
            "promptProvenance": "typed",
            "model": "ltx-2.5-fast",
            "resolution": "1080p",
            "duration": 5,
            "fps": 24,
            "loras": [{"catalogId": "cozy-felt"}, {"path": "/tmp/custom.safetensors"}],
        },
    )
    details = queued_generation_details(
        record,
        enhancement=_LOCAL_ENHANCE,
        hardware={
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "total_vram_bytes": 34089730048,
            "free_vram_bytes": 30000000000,
            "drive_type": "nvme",
        },
        modes={"load_mode": "streaming_models_loading", "offload_mode": "DISK"},
    )
    assert details is not None
    contract_path = Path(__file__).resolve().parents[2] / "shared" / "generation-detail-keys.json"
    contract = set(json.loads(contract_path.read_text()))
    assert set(details) == contract - {
        "outcome",
        "runtime_ms",
        "error_code",
        "output_count",
        "prompt_rewrite",
    }


def test_video_generation_feature_prefers_keyframes_then_audio() -> None:
    assert video_generation_feature(
        audio_path="clip.wav",
        image_path="still.png",
        keyframes=[{"imagePath": "opening.png"}],
    ) == "multi-keyframe"
    assert video_generation_feature(
        audio_path="clip.wav",
        image_path="still.png",
    ) == "audio-to-video"
    assert video_generation_feature(
        audio_path=None,
        image_path=None,
        last_image_path="end.png",
    ) == "image-to-video"
    assert video_generation_feature(audio_path=None, image_path=None) == "text-to-video"


def test_home_queue_records_stored_prompt_provenance() -> None:
    enhanced = _home_record(
        feature="text-to-video",
        params={"prompt": "a long caption", "promptProvenance": "enhanced", "model": "ltx-2.5-fast"},
    )
    typed = _home_record(
        feature="text-to-video",
        params={"prompt": "a fox", "promptProvenance": "typed", "model": "ltx-2.5-fast"},
    )
    retake = _home_record(
        feature="retake",
        params={"prompt": "a fox", "promptProvenance": "typed"},
    )
    cozy = _home_record(
        feature="cozy-felt",
        params={"prompt": "a fox", "promptProvenance": "typed", "model": "ltx-2.5-fast"},
    )
    cozy_manual = _home_record(
        feature="cozy-felt",
        params={"prompt": "a fox", "promptProvenance": "enhanced", "model": "ltx-2.5-fast"},
    )

    assert "prompt_provenance" not in (queued_generation_details(typed) or {})
    assert queued_generation_details(enhanced, enhancement=_LOCAL_ENHANCE)["prompt_provenance"] == "manually-enhanced"
    assert queued_generation_details(typed, enhancement=_LOCAL_ENHANCE)["prompt_provenance"] == "auto-enhanced"
    # Retake and extend rewrite from stills of their source video, so a typed prompt is auto-enhanced.
    assert queued_generation_details(retake, enhancement=_LOCAL_ENHANCE)["prompt_provenance"] == "auto-enhanced"
    assert queued_generation_details(
        typed,
        enhancement=QueuedEnhancement(
            explore_auto_enhance_prompts=False,
            prompt_enhancer_enabled=True,
            use_local_encoding=True,
        ),
    )["prompt_provenance"] == "raw"
    # API text encoding plus a recipe wrap sends the typed scene when the local
    # enhancer checkpoint is not installed.
    assert queued_generation_details(cozy, enhancement=_API_ENHANCE)["prompt_provenance"] == "raw"
    assert queued_generation_details(
        cozy,
        enhancement=QueuedEnhancement(
            explore_auto_enhance_prompts=True,
            prompt_enhancer_enabled=True,
            use_local_encoding=False,
            local_enhancer_available=True,
        ),
    )["prompt_provenance"] == "auto-enhanced"
    assert queued_generation_details(cozy, enhancement=_LOCAL_ENHANCE)["prompt_provenance"] == "auto-enhanced"
    assert queued_generation_details(cozy_manual, enhancement=_API_ENHANCE)["prompt_provenance"] == "manually-enhanced"
    assert queued_generation_details(typed, enhancement=_API_ENHANCE)["prompt_provenance"] == "auto-enhanced"


def _home_record(
    *, feature: str, params: dict[str, object], inputs: dict[str, object] | None = None
) -> GenerationRecord:
    return GenerationRecord(
        id="generation-1",
        feature=feature,
        contract_version=1,
        status="running",
        spec={
            "_analytics": {"client": "desktop"},
            "params": params,
            **({"inputs": inputs} if inputs is not None else {}),
        },
        created_at=1,
        queued_at=1,
        attempt_count=1,
        started_at=10,
        outputs=(),
    )


def test_genspace_preview_records_manual_auto_and_raw(test_state) -> None:
    enhancement = test_state.prompt_enhancement
    assert enhancement.preview_prompt_provenance(
        "a fox",
        provenance="enhanced",
        explore_generation=False,
    ) == "manually-enhanced"

    test_state.state.app_settings.prompt_enhancer_enabled = False
    assert enhancement.preview_prompt_provenance(
        "a fox",
        provenance="typed",
        explore_generation=False,
        force_api=True,
    ) == "raw"

    test_state.state.app_settings.prompt_enhancer_enabled = True
    assert enhancement.preview_prompt_provenance(
        "a fox",
        provenance="typed",
        explore_generation=False,
        force_api=True,
    ) == "auto-enhanced"


def test_home_queue_details_stay_local_video_and_omit_prompt() -> None:
    record = GenerationRecord(
        id="generation-1",
        feature="text-to-video",
        contract_version=1,
        status="running",
        spec={
            "_analytics": {"client": "desktop"},
            "params": {
                "prompt": "a fox in /tmp/secret.mp4",
                "model": "ltx-2.5-fast",
                "resolution": "1080p",
                "duration": 5,
                "fps": 24,
                "video_path": "/tmp/in.mp4",
            },
        },
        created_at=1,
        queued_at=1,
        attempt_count=1,
        started_at=10,
        outputs=(),
    )

    details = queued_generation_details(record)

    assert details is not None
    assert details["surface"] == "home"
    assert details["kind"] == "video"
    assert details["execution"] == "local"
    assert details["model"] == "ltx-2.5-fast"
    assert details["resolution"] == "1080p"
    assert "prompt" not in details
    assert "video_path" not in details


def test_home_queue_pixel_resolution_is_recorded_as_width_times_height() -> None:
    record = GenerationRecord(
        id="generation-1",
        feature="retake",
        contract_version=1,
        status="running",
        spec={
            "_analytics": {"client": "desktop"},
            "params": {
                "prompt": "a fox",
                "resolution": {"width": 1920, "height": 1080, "path": "/tmp/in.mp4"},
            },
        },
        created_at=1,
        queued_at=1,
        attempt_count=1,
        started_at=10,
        outputs=(),
    )

    details = queued_generation_details(record)

    assert details is not None
    assert details["resolution"] == "1920x1080"
    assert "path" not in details
    assert "prompt" not in details


def test_desktop_create_response_omits_internal_analytics(client, test_state) -> None:
    response = client.post(
        "/api/generations/text-to-video",
        json={
            "params": {
                "prompt": "a fox in /tmp/secret.mp4",
                "model": "ltx-2.5-fast",
            }
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "_analytics" not in body["spec"]
    assert "a fox" not in str(body["spec"].get("_analytics", ""))
    stored = SqliteStore(test_state.config.app_data_dir).get_generation(body["id"])
    assert stored is not None
    assert stored.spec["_analytics"] == {"client": "desktop"}
    details = queued_generation_details(stored)
    assert details is not None
    assert "prompt" not in details


def test_started_details_carry_backend_hardware_snapshot() -> None:
    """Hardware passed to the builder must land on the started event, not vanish."""
    record = _home_record(feature="text-to-video", params={"model": "ltx-2.5-fast"})

    details = queued_generation_details(
        record,
        hardware={
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "total_vram_bytes": 34089730048,
            "free_vram_bytes": 30000000000,
            "drive_type": "nvme",
        },
    )

    assert details is not None
    assert details["gpu_model"] == "NVIDIA GeForce RTX 5090"
    assert details["total_vram_bytes"] == 34089730048
    assert details["free_vram_bytes"] == 30000000000
    assert details["drive_type"] == "nvme"


def test_started_details_omit_unknown_hardware_fields() -> None:
    """Absent snapshot values must stay absent, not flow as explicit nulls."""
    record = _home_record(feature="text-to-video", params={"model": "ltx-2.5-fast"})

    details = queued_generation_details(
        record,
        hardware={"gpu_model": None, "total_vram_bytes": None},
    )

    assert details is not None
    assert "gpu_model" not in details
    assert "total_vram_bytes" not in details


def test_ended_event_keeps_load_and_offload_modes() -> None:
    """Modes must survive the ended allowlist filter to the sink payload."""
    http = FakeHTTPClient()
    http.queue("post", FakeResponse())
    service = _service(http)
    record = _home_record(feature="text-to-video", params={"model": "ltx-2.5-fast"})
    details = queued_generation_details(
        record,
        modes={"load_mode": "streaming_models_loading", "offload_mode": "DISK"},
    )
    assert details is not None

    service.send_generation_ended(details, outcome="succeeded", runtime_ms=1000)

    assert len(http.calls) == 1
    payload = http.calls[0].json_payload
    assert payload is not None
    ended = payload["extraDetails"]
    assert isinstance(ended, dict)
    assert ended["load_mode"] == "streaming_models_loading"
    assert ended["offload_mode"] == "DISK"


def test_a_fal_image_job_with_no_enhancer_is_auto_enhanced_by_fal_expansion() -> None:
    # Bug caught: a fal job that fal expands is reported as raw when no local or Gemini rewrite exists.
    no_enhancer = QueuedEnhancement(
        explore_auto_enhance_prompts=True, prompt_enhancer_enabled=False, use_local_encoding=False
    )
    auto_off = QueuedEnhancement(
        explore_auto_enhance_prompts=False, prompt_enhancer_enabled=False, use_local_encoding=False
    )

    def image(provider: str, provenance: str = "typed") -> GenerationRecord:
        return _home_record(
            feature="text-to-image",
            params={"prompt": "a fox", "promptProvenance": provenance, "provider": provider},
        )

    def label(record: GenerationRecord, enhancement: QueuedEnhancement) -> object:
        return (queued_generation_details(record, enhancement=enhancement) or {})["prompt_provenance"]

    assert label(image("fal"), no_enhancer) == "auto-enhanced"
    assert label(image("local"), no_enhancer) == "raw"
    assert label(image("fal"), auto_off) == "raw"
    assert label(image("fal", "enhanced"), no_enhancer) == "manually-enhanced"


def test_ended_event_keeps_the_prompt_rewrite_source() -> None:
    # Bug caught: the rewrite source is dropped by the ended allowlist, so fal expansion cost
    # cannot be told apart from a free local or Gemini rewrite.
    http = FakeHTTPClient()
    http.queue("post", FakeResponse())
    service = _service(http)
    record = _home_record(feature="text-to-image", params={"prompt": "a fox"})
    details = queued_generation_details(record, modes={"prompt_rewrite": "fal_expansion"})
    assert details is not None

    service.send_generation_ended(details, outcome="succeeded", runtime_ms=1000)

    payload = http.calls[0].json_payload
    assert payload is not None
    ended = payload["extraDetails"]
    assert isinstance(ended, dict)
    assert ended["prompt_rewrite"] == "fal_expansion"


def test_extend_with_gemini_video_enhance_is_auto_enhanced_even_with_api_text_encoding() -> None:
    extend = _home_record(
        feature="extend",
        params={"prompt": "a fox", "promptProvenance": "typed", "model": "ltx-2.5-fast"},
        inputs={"video": {"assetId": "asset-1"}},
    )
    gemini = QueuedEnhancement(
        explore_auto_enhance_prompts=True,
        prompt_enhancer_enabled=True,
        use_local_encoding=False,
        gemini_video_enhance=True,
    )

    assert queued_generation_details(extend, enhancement=gemini)["prompt_provenance"] == "auto-enhanced"
    assert queued_generation_details(extend, enhancement=_API_ENHANCE)["prompt_provenance"] == "raw"


def test_sheet_recipe_with_gemini_media_enhance_is_auto_enhanced_not_raw() -> None:
    ingredients = _home_record(
        feature="ingredients",
        params={"prompt": "a fox runs", "promptProvenance": "typed", "model": "ltx-2.3-fast"},
        inputs={"images": [{"assetId": "asset-1"}, {"assetId": "asset-2"}]},
    )

    def enhancement(*, gemini_video_enhance: bool) -> QueuedEnhancement:
        return QueuedEnhancement(
            explore_auto_enhance_prompts=True,
            prompt_enhancer_enabled=False,
            use_local_encoding=False,
            gemini_video_enhance=gemini_video_enhance,
        )

    details = queued_generation_details(ingredients, enhancement=enhancement(gemini_video_enhance=True))
    assert details["prompt_provenance"] == "auto-enhanced"
    details = queued_generation_details(ingredients, enhancement=enhancement(gemini_video_enhance=False))
    assert details["prompt_provenance"] == "raw"


def test_motion_track_is_raw_even_when_auto_enhance_would_rewrite_a_typed_prompt() -> None:
    # The job wraps the scene with the camera lock and never rewrites it.
    params = {"prompt": "a frog", "promptProvenance": "typed", "model": "ltx-2.5-fast"}
    typed = _home_record(feature="motion-track", params=params)
    manual = _home_record(feature="motion-track", params={**params, "promptProvenance": "enhanced"})

    assert queued_generation_details(typed, enhancement=_LOCAL_ENHANCE)["prompt_provenance"] == "raw"
    assert queued_generation_details(manual, enhancement=_LOCAL_ENHANCE)["prompt_provenance"] == "manually-enhanced"
