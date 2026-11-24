"""Prompt enhancement that sees stills of a source video (IC-LoRA, Extend, Retake).

Manual Enhance goes through POST /api/enhance-prompt. Automatic enhancement goes through
``resolve_for_generation``, the single policy every queued Home job asks.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from api_types import (
    IcLoraGenerateRequest,
    IcLoraImageInput,
    InputSpec,
    PromptTemplatePlaceholder,
    PromptTemplateSpec,
    ReplacedWindow,
    parse_lora_catalog,
)
from services.generation_interrupt import GenerationCancelledError
from runtime_config.model_download_specs import resolve_ic_lora_path
from services.prompt_enhancement import (
    build_video_user_prompt_text,
    resolve_video_frames,
    with_video_role_rules,
)
from services.prompt_enhancement.i2v_frames import SceneFrame, reference_still
from services.prompt_enhancement.source_video import SourceVideo
from services.prompt_enhancement.video_frames import scene_frame_count
from tests.fakes import FakeResponse
from tests.lora_catalog_helpers import add_ic_lora
from tests.video_fixtures import write_mp4

# 2.3's gemma3 encoder doubles as the local enhancer, so its plain bundle is enough.
_LOCAL_ENHANCER_MODEL_ID = "ltx-2.3-22b-distilled-1.1"
_TYPED = "a slow dolly in"


def _gemini_ok(text: str) -> FakeResponse:
    return FakeResponse(
        status_code=200,
        json_payload={"candidates": [{"content": {"parts": [{"text": text}]}}]},
    )


def _video(tmp_path: Path) -> str:
    return str(write_mp4(tmp_path / "source.mp4", seconds=6, fps=12))


def _install_local_enhancer(test_state, create_fake_model_files) -> None:
    create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
    test_state.state.app_settings.active_ltx_model_id = _LOCAL_ENHANCER_MODEL_ID
    test_state.state.app_settings.use_local_text_encoder = True


def _resolve_home(test_state, video_path: str):
    return test_state.prompt_enhancement.resolve_for_generation(
        _TYPED,
        provenance="typed",
        generation_seed=1,
        explicit_generation_seed=True,
        explore_generation=True,
        source_video=SourceVideo(path=video_path),
    )


def test_driving_and_scene_roles_get_different_rules_on_top_of_the_base_prompt() -> None:
    driving = with_video_role_rules("BASE", "driving")
    scene = with_video_role_rules("BASE", "scene")

    assert driving.startswith("BASE")
    assert scene.startswith("BASE")
    assert driving != scene


@pytest.mark.parametrize("role", ["driving", "scene"])
def test_blank_prompt_leaves_no_empty_instruction_label_for_the_model(role: str) -> None:
    blank = build_video_user_prompt_text("   ", role=role)  # type: ignore[arg-type]

    assert blank == build_video_user_prompt_text("", role=role)  # type: ignore[arg-type]
    assert "User instruction" not in blank
    assert "primary, follow it" not in blank


class TestManualEnhanceRequest:
    def test_empty_prompt_with_a_video_is_enhanced_instead_of_rejected(
        self, client, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "a caption from the stills"

        r = client.post("/api/enhance-prompt", json={"prompt": "", "videoPath": _video(tmp_path)})

        assert r.status_code == 200
        assert r.json()["enhancedPrompt"] == "a caption from the stills"

    @pytest.mark.parametrize(
        "conflict",
        [
            {"imagePath": "still.png"},
            {"loraCatalogIds": ["some-lora"]},
            {"conditioningType": "depth"},
            {"videoAssetId": "asset-1"},
            {"mediaType": "image"},
        ],
        ids=["image", "plain-lora", "built-in-conditioning", "second-video", "image-media-type"],
    )
    def test_a_video_mixed_with_another_input_is_rejected_before_it_reaches_a_provider(
        self, client, create_fake_model_files, tmp_path, conflict
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)

        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "x", "videoPath": _video(tmp_path), **conflict},
        )

        assert r.status_code == 422

    @pytest.mark.parametrize(
        "extra",
        [
            {"icLoraId": "day-to-night"},
            {"videoPath": None},
        ],
        ids=["with-an-ic-lora", "without-a-video"],
    )
    def test_a_replaced_window_is_rejected_unless_it_belongs_to_a_plain_video_edit(
        self, client, fake_services, create_fake_model_files, tmp_path, extra
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
        add_ic_lora(fake_services, id="day-to-night", name="Day to Night")
        body = {
            "prompt": "x",
            "videoPath": _video(tmp_path),
            "replacedWindow": {"start": 1, "end": 2},
            **extra,
        }
        if body["videoPath"] is None:
            del body["videoPath"]

        r = client.post("/api/enhance-prompt", json=body)

        assert r.status_code == 422

    @pytest.mark.parametrize(
        "window",
        [{"start": 2, "end": 2}, {"start": 3, "end": 2}, {"start": -1, "end": 2}],
        ids=["empty", "backwards", "negative-start"],
    )
    def test_a_replaced_window_that_is_empty_or_backwards_is_rejected(
        self, client, create_fake_model_files, tmp_path, window
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)

        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "x", "videoPath": _video(tmp_path), "replacedWindow": window},
        )

        assert r.status_code == 422

    def test_a_file_that_is_not_a_video_is_a_400_not_a_500(
        self, client, create_fake_model_files, tmp_path
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
        not_video = tmp_path / "notes.mp4"
        not_video.write_text("not a video")

        r = client.post("/api/enhance-prompt", json={"prompt": "x", "videoPath": str(not_video)})

        assert r.status_code == 400


class TestManualEnhanceWithVideo:
    @pytest.mark.parametrize("provider", ["local", "api"])
    def test_every_sampled_still_reaches_the_chosen_provider(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path, provider
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
        test_state.state.app_settings.gemini_api_key = "key"
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "rewritten on gemma"
        test_state.http.queue("post", _gemini_ok("rewritten on gemini"))

        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": _TYPED, "videoPath": _video(tmp_path), "provider": provider},
        )

        assert r.status_code == 200
        if provider == "local":
            call = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[0]
            assert call["frames_existed"]
            assert call["role"] == "scene"
            stills_seen = len(call["frame_times"])
            assert r.json()["enhancedPrompt"] == "rewritten on gemma"
        else:
            parts = test_state.http.calls[-1].json_payload["contents"][0]["parts"]
            stills_seen = len([part for part in parts if "inlineData" in part])
            assert r.json()["enhancedPrompt"] == "rewritten on gemini"
        assert stills_seen == scene_frame_count(6)

    def test_manual_enhance_with_a_replaced_window_shows_the_model_only_stills_from_around_it(
        self, client, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)

        r = client.post(
            "/api/enhance-prompt",
            json={
                "prompt": _TYPED,
                "videoPath": _video(tmp_path),
                "replacedWindow": {"start": 2, "end": 4},
                "provider": "local",
            },
        )

        assert r.status_code == 200
        times = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[0]["frame_times"]
        assert len(times) == scene_frame_count(6)
        assert not any(2 < time < 4 for time in times)
        assert any(time < 2 for time in times)
        assert any(time > 4 for time in times)

    def test_a_template_fill_ic_lora_still_stitches_its_template_when_given_stills(
        self, client, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
        add_ic_lora(
            fake_services,
            id="day-to-night",
            name="Day to Night",
            prompt_template=PromptTemplateSpec(
                template="A night scene. {lighting} Only the lighting changes.",
                placeholders={"lighting": PromptTemplatePlaceholder()},
            ),
        )
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = '{"lighting": "Moonlit street."}'

        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "", "icLoraId": "day-to-night", "videoPath": _video(tmp_path)},
        )

        assert r.status_code == 200
        assert r.json()["enhancedPrompt"] == "A night scene. Moonlit street. Only the lighting changes."
        pipeline = fake_services.prompt_enhancer_pipeline
        assert pipeline.enhance_video_frames_calls[0]["role"] == "driving"
        assert pipeline.enhance_t2v_calls == []


class TestAutomaticEnhanceVideoPrivacyGate:
    """Automatic enhancement would upload stills of the user's video. Gemini must be a choice."""

    @pytest.mark.parametrize(
        ("preference", "key", "expected"),
        [
            ("local", "key", "rewritten on gemma"),
            (None, "key", "rewritten on gemma"),
            ("api", "", "rewritten on gemma"),
            ("api", "key", "rewritten on gemini"),
        ],
        ids=["local-choice", "no-choice", "api-choice-no-key", "api-choice-with-key"],
    )
    def test_only_an_explicit_api_choice_with_a_key_sends_the_video_to_gemini(
        self, test_state, fake_services, create_fake_model_files, tmp_path, preference, key, expected
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        settings = test_state.state.app_settings
        settings.prompt_enhancer_provider_preference = preference
        settings.gemini_api_key = key
        fake_services.prompt_enhancer_pipeline.enhanced_prompt = "rewritten on gemma"
        test_state.http.queue("post", _gemini_ok("rewritten on gemini"))

        resolved = _resolve_home(test_state, _video(tmp_path))

        assert resolved.prompt == expected
        if expected == "rewritten on gemma":
            assert test_state.http.calls == []

    def test_gemini_is_used_even_when_no_local_enhancer_is_installed(
        self, test_state, tmp_path
    ):
        settings = test_state.state.app_settings
        settings.prompt_enhancer_provider_preference = "api"
        settings.gemini_api_key = "key"
        test_state.http.queue("post", _gemini_ok("rewritten on gemini"))

        assert _resolve_home(test_state, _video(tmp_path)).prompt == "rewritten on gemini"


class TestAutomaticEnhanceVideoFailure:
    def test_a_failing_local_rewrite_submits_the_typed_prompt(
        self, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.raise_on_enhance = RuntimeError("boom")

        resolved = _resolve_home(test_state, _video(tmp_path))

        assert resolved.prompt == _TYPED
        assert resolved.enhanced_locally is False

    def test_a_failing_gemini_rewrite_submits_the_typed_prompt(self, test_state, tmp_path):
        settings = test_state.state.app_settings
        settings.prompt_enhancer_provider_preference = "api"
        settings.gemini_api_key = "key"
        test_state.http.queue("post", FakeResponse(status_code=500, text="down"))

        assert _resolve_home(test_state, _video(tmp_path)).prompt == _TYPED

    def test_an_unreadable_source_video_submits_the_typed_prompt(
        self, test_state, create_fake_model_files, tmp_path
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        broken = tmp_path / "broken.mp4"
        broken.write_text("not a video")

        assert _resolve_home(test_state, str(broken)).prompt == _TYPED

    def test_cancellation_is_raised_instead_of_falling_back_to_the_typed_prompt(
        self, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        fake_services.prompt_enhancer_pipeline.raise_on_enhance = GenerationCancelledError()

        with pytest.raises(GenerationCancelledError):
            _resolve_home(test_state, _video(tmp_path))


def _look_image(tmp_path: Path) -> str:
    path = tmp_path / "look.png"
    Image.new("RGB", (64, 64), "red").save(path)
    return str(path)


def _add_look_ic_lora(fake_services, *, pinned_frame: int | None, allows: bool = True, **extra: object):
    return add_ic_lora(
        fake_services,
        id="layout-to-render",
        name="Layout to Render",
        description="Turn a plain 3D layout video into a finished shot.",
        input=InputSpec(kind="video"),
        allows_empty_prompt=True,
        allows_reference_image=allows,
        reference_image_frame=pinned_frame if allows else None,
        **extra,
    )


class TestLookReferenceLabels:
    def test_a_look_image_before_the_clip_is_labeled_as_not_a_frame_and_comes_first(self) -> None:
        stills = [SceneFrame("b.jpg", 3.0), SceneFrame("a.jpg", 1.0)]

        labeled = resolve_video_frames([*stills, reference_still("look.png", is_clip_frame=False)])

        assert labeled[0] == ("look.png", "Look reference image (not a frame of the clip).")
        assert [label for _path, label in labeled[1:]] == [
            "Frame 1 of 2 (1.00s).",
            "Frame 2 of 2 (3.00s).",
        ]

    def test_a_look_image_that_is_the_first_frame_is_labeled_as_the_finished_first_frame(self) -> None:
        labeled = resolve_video_frames(
            [SceneFrame("a.jpg", 1.0), reference_still("look.png", is_clip_frame=True)]
        )

        assert labeled[0][1] == "Look reference image (the finished first frame of the clip)."


class TestLookReferenceUserTurn:
    @pytest.mark.parametrize("prompt", ["", "make it night"], ids=["blank", "typed"])
    def test_an_ic_lora_with_a_look_image_asks_for_the_look_in_the_user_turn(self, prompt) -> None:
        text = build_video_user_prompt_text(prompt, role="driving", has_look_reference=True)

        assert "look reference image" in text
        assert "Never mention the reference image" in text

    def test_no_look_is_asked_for_without_a_look_image_or_for_a_scene_role(self) -> None:
        assert "look reference" not in build_video_user_prompt_text("x", role="driving")
        assert "look reference" not in build_video_user_prompt_text(
            "x", role="scene", has_look_reference=True
        )

    def test_gemini_gets_the_look_request_in_its_last_user_part(
        self, client, test_state, fake_services, tmp_path
    ):
        _add_look_ic_lora(fake_services, pinned_frame=-1)
        test_state.state.app_settings.gemini_api_key = "key"
        test_state.http.queue("post", _gemini_ok("a rewritten shot"))

        r = client.post(
            "/api/enhance-prompt",
            json={
                "prompt": "",
                "icLoraId": "layout-to-render",
                "videoPath": _video(tmp_path),
                "imagePath": _look_image(tmp_path),
                "provider": "api",
            },
        )

        parts = test_state.http.calls[-1].json_payload["contents"][0]["parts"]
        assert r.status_code == 200
        assert "look reference image" in parts[-1]["text"]


class TestManualEnhanceWithALookImage:
    def test_the_model_sees_the_look_image_first_and_all_the_stills(
        self, client, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
        _add_look_ic_lora(fake_services, pinned_frame=-1, reference_image_required=True)

        r = client.post(
            "/api/enhance-prompt",
            json={
                "prompt": "",
                "icLoraId": "layout-to-render",
                "videoPath": _video(tmp_path),
                "imagePath": _look_image(tmp_path),
            },
        )

        call = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[0]
        assert r.status_code == 200
        assert call["role"] == "driving"
        assert call["look_image_first"] is True
        assert call["look_image_notes"] == ["not a frame of the clip"]
        assert len(call["frame_times"]) == scene_frame_count(6) + 1

    def test_an_optional_look_image_may_be_left_out(
        self, client, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
        _add_look_ic_lora(fake_services, pinned_frame=0)

        r = client.post(
            "/api/enhance-prompt",
            json={"prompt": "", "icLoraId": "layout-to-render", "videoPath": _video(tmp_path)},
        )

        call = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[0]
        assert r.status_code == 200
        assert call["look_images"] == 0

    def test_a_pinned_first_frame_look_image_is_told_to_the_model_as_the_first_frame(
        self, client, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
        _add_look_ic_lora(fake_services, pinned_frame=0)

        client.post(
            "/api/enhance-prompt",
            json={
                "prompt": "",
                "icLoraId": "layout-to-render",
                "videoPath": _video(tmp_path),
                "imagePath": _look_image(tmp_path),
            },
        )

        call = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[0]
        assert call["look_image_notes"] == ["the finished first frame of the clip"]

    def test_an_image_for_an_ic_lora_that_takes_none_is_a_400_and_never_reaches_a_provider(
        self, client, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files(model_id=_LOCAL_ENHANCER_MODEL_ID)
        _add_look_ic_lora(fake_services, pinned_frame=None, allows=False)

        r = client.post(
            "/api/enhance-prompt",
            json={
                "prompt": "",
                "icLoraId": "layout-to-render",
                "videoPath": _video(tmp_path),
                "imagePath": _look_image(tmp_path),
            },
        )

        assert r.status_code == 400
        assert fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls == []

    def test_a_look_image_that_is_not_an_image_is_a_400_before_the_provider_call(
        self, client, test_state, fake_services, tmp_path
    ):
        _add_look_ic_lora(fake_services, pinned_frame=-1)
        test_state.state.app_settings.gemini_api_key = "key"
        not_image = tmp_path / "notes.png"
        not_image.write_text("not an image")

        r = client.post(
            "/api/enhance-prompt",
            json={
                "prompt": "",
                "icLoraId": "layout-to-render",
                "videoPath": _video(tmp_path),
                "imagePath": str(not_image),
                "provider": "api",
            },
        )

        assert r.status_code == 400
        assert test_state.http.calls == []


class TestAutomaticEnhanceWithALookImage:
    def test_a_home_ic_lora_job_gives_the_model_its_look_image(
        self, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        _install_local_enhancer(test_state, create_fake_model_files)
        _add_look_ic_lora(fake_services, pinned_frame=-1)
        lora_file = resolve_ic_lora_path(
            test_state.ic_lora.models_dir, "layout-to-render", "x.safetensors"
        )
        lora_file.parent.mkdir(parents=True, exist_ok=True)
        lora_file.write_bytes(b"\x00")

        test_state.ic_lora.generate_local_reserved(
            IcLoraGenerateRequest(
                ic_lora_id="layout-to-render",
                input_path=_video(tmp_path),
                prompt=_TYPED,
                conditioning_type="custom",
                images=[IcLoraImageInput(path=_look_image(tmp_path))],
            ),
            generation_id="g-1",
            output_path=tmp_path / "out.mp4",
            local_model_id=_LOCAL_ENHANCER_MODEL_ID,
            seed=1,
            explicit_generation_seed=True,
        )

        calls = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls
        assert calls and calls[0]["look_image_first"] is True


class TestReplacedWindowValues:
    @pytest.mark.parametrize(
        ("start", "end"),
        [(0.0, float("inf")), (0.0, float("nan")), (float("nan"), 5.0)],
        ids=["infinite-end", "nan-end", "nan-start"],
    )
    def test_a_non_finite_window_is_rejected(self, start: float, end: float) -> None:
        with pytest.raises(ValidationError):
            ReplacedWindow(start=start, end=end)


class TestGeminiSlotForAutomaticRewrites:
    def test_an_automatic_gemini_rewrite_waits_for_a_manual_one_instead_of_running_beside_it(
        self, test_state, tmp_path
    ):
        settings = test_state.state.app_settings
        settings.prompt_enhancer_provider_preference = "api"
        settings.gemini_api_key = "key"
        test_state.http.queue("post", _gemini_ok("rewritten on gemini"))
        slot = test_state.prompt_enhancement._gemini_enhance_slot
        results: list[str] = []
        slot.acquire()
        worker = threading.Thread(
            target=lambda: results.append(_resolve_home(test_state, _video(tmp_path)).prompt)
        )
        try:
            worker.start()
            worker.join(timeout=0.5)
            still_waiting = worker.is_alive() and results == []
        finally:
            slot.release()
        worker.join(timeout=10)

        assert still_waiting
        assert results == ["rewritten on gemini"]


class TestAlphaGenKeepsItsEmptyPrompt:
    """AlphaGen takes no prompt. No rewrite may run for it, on any provider."""

    def _run_home_alpha_gen(
        self, test_state, fake_services, create_fake_model_files, tmp_path: Path, prompt: str
    ) -> None:
        model_id = "ltx-2.5-22b-distilled"
        create_fake_model_files(model_id=model_id)
        settings = test_state.state.app_settings
        settings.active_ltx_model_id = model_id
        settings.use_local_text_encoder = True
        settings.explore_auto_enhance_prompts = True
        settings.prompt_enhancer_provider_preference = "api"
        settings.gemini_api_key = "key"
        catalog = Path(__file__).resolve().parents[1] / "runtime_config" / "lora_catalog.json"
        item = next(
            entry
            for entry in parse_lora_catalog(catalog.read_text()).ic_loras
            if entry.id == "alpha-gen"
        )
        fake_services.lora_catalog_provider._catalog.ic_loras.append(item)
        weights = resolve_ic_lora_path(
            test_state.ic_lora.models_dir, item.id, item.download.variants[0].filename
        )
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"\x00")

        test_state.ic_lora.generate_local_reserved(
            IcLoraGenerateRequest(
                ic_lora_id="alpha-gen",
                input_path=_video(tmp_path),
                prompt=prompt,
                conditioning_type="custom",
            ),
            generation_id="g-1",
            output_path=tmp_path / "out.mp4",
            local_model_id=model_id,
            seed=1,
            explicit_generation_seed=True,
        )

    def test_an_empty_prompt_is_not_enhanced_on_gemma_or_gemini(
        self, test_state, fake_services, create_fake_model_files, tmp_path
    ) -> None:
        self._run_home_alpha_gen(test_state, fake_services, create_fake_model_files, tmp_path, "")

        enhancer = fake_services.prompt_enhancer_pipeline
        assert enhancer.enhance_video_frames_calls == []
        assert enhancer.enhance_t2v_calls == []
        assert enhancer.enhance_i2v_calls == []
        assert test_state.http.calls == []
