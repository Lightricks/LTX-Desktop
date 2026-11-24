"""AMD engine selection through the generation handler and queued request contract."""
from pathlib import Path

from PIL import Image
import pytest

from api_types import GenerateVideoRequest, ImageToVideoRecipeParams, TextToVideoParams
from runtime_config.amd_video_engine import MODEL_ID
from services.amd_comfy_client import AmdComfyClient, ComfyCancelled, ComfyError
from services.features.video.text_to_video import generate_request_from_offering_params, offering_payload_from_params


def request(**changes):
    return {"prompt": "A bird in flight", "model": "fast", "resolution": "540p",
            "duration": 5, "fps": 24, "seed": 9, **changes}


@pytest.fixture
def external_worker(monkeypatch):
    monkeypatch.setenv("LTX_REQUIRE_ROCM", "1")
    from services import amd_comfy_workflow
    calls = {}

    class Worker:
        gpu_name = "Test Radeon"

        def get_object_info(self):
            return {"nodes": True}

        def upload_image(self, path):
            with Image.open(path) as image:
                calls["uploaded_size"] = image.size
            return "input/frame.png"

        def run(self, graph, dest, *, output_node_id, on_progress, is_cancelled):
            assert graph == {"test": {}} and output_node_id == amd_comfy_workflow.SAVE_VIDEO_ID
            assert not is_cancelled()
            on_progress(2, 11)
            if "failure" in calls:
                raise calls["failure"]
            dest.write_bytes(b"test-result")

    def build(**kwargs):
        calls["graph"] = kwargs
        return {"test": {}}

    monkeypatch.setattr(AmdComfyClient, "discover", lambda **_: Worker())
    monkeypatch.setattr(amd_comfy_workflow, "build_workflow", build)
    return calls


def test_default_comfy_without_native_model_uses_real_http_route(client, test_state, external_worker, monkeypatch):
    test_state.state.app_settings.explore_auto_enhance_prompts = True
    test_state.state.app_settings.prompt_enhancer_enabled = True
    def unexpected_enhancement(*args, **kwargs):
        pytest.fail("The external engine requested a separate native or cloud prompt rewrite.")
    monkeypatch.setattr(test_state.video_generation._prompt_enhancement, "resolve_for_generation",
                        unexpected_enhancement)
    response = client.post("/api/generate", json=request())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "complete" and body["load_mode"] is None
    assert Path(body["video_path"]).read_bytes() == b"test-result"
    graph = external_worker["graph"]
    assert graph["frames"] == 121 and graph["seed"] == 9 and graph["image_name"] is None
    assert graph["object_info"] == {"nodes": True}
    assert graph["prompt"] == "A bird in flight"


def test_external_i2v_writes_requested_queue_output(test_state, external_worker, tmp_path):
    source = tmp_path / "frame.png"
    Image.new("RGB", (60, 90)).save(source)
    dest = tmp_path / "queue-output.mp4"
    req = GenerateVideoRequest.model_validate(request(resolution="270p", imagePath=str(source),
                                                    aspectRatio="9:16", amdFastEngine="comfy"))
    with test_state.generation.reserved_generation_start():
        result = test_state.video_generation.generate_local_reserved(
            req, generation_id="queued", output_path=dest, local_model_id=MODEL_ID)
    assert result.status == "complete" and Path(result.video_path) == dest
    graph = external_worker["graph"]
    assert graph["single_stage"] and graph["image_name"] == "input/frame.png"
    assert external_worker["uploaded_size"] == (graph["width"], graph["height"])
    assert graph["height"] > graph["width"]


def test_explicit_native_never_invokes_comfy(client, external_worker):
    response = client.post("/api/generate", json=request(amdFastEngine="native"))
    assert response.status_code == 409 and "NO_DOWNLOADED_LTX_MODEL" in response.text
    assert "graph" not in external_worker


@pytest.mark.parametrize("engine,changes", [
    ("comfy", {"lastImagePath": "end.png"}),
    ("comfy", {"duration": None}),
    ("comfy", {"negativePrompt": "blur"}),
    ("kandinsky", {"aspectRatio": "1:1"}),
    ("kandinsky", {"audioPath": "conditioning.wav"}),
    ("kandinsky", {"negativePrompt": "blur"}),
])
def test_explicit_external_refuses_unsupported_inputs(client, external_worker, engine, changes):
    response = client.post("/api/generate", json=request(amdFastEngine=engine, **changes))
    assert response.status_code == 422
    assert "graph" not in external_worker


def test_comfy_failure_and_cancel_never_submit_to_native(client, external_worker):
    external_worker["failure"] = ComfyError("ComfyUI model missing")
    response = client.post("/api/generate", json=request(amdFastEngine="comfy"))
    assert response.status_code == 502 and "ComfyUI model missing" in response.text
    external_worker["failure"] = ComfyCancelled("stopped")
    response = client.post("/api/generate", json=request(amdFastEngine="comfy"))
    assert response.status_code == 200 and response.json()["status"] == "cancelled"


def test_queue_model_resolution_without_native_weights(test_state):
    for engine in ("comfy", "kandinsky"):
        params = TextToVideoParams.model_validate(request(model="ltx-2.5-fast", amdFastEngine=engine))
        req, model_id = generate_request_from_offering_params(params, test_state.video_generation)
        assert req.amdFastEngine == engine and model_id == MODEL_ID


def test_queue_create_records_engine_without_native_checkpoints(client, external_worker):
    response = client.post("/api/generations/text-to-video",
                           json={"params": request(model="ltx-2.5-fast")})
    assert response.status_code in (200, 201), response.text
    assert response.json()["spec"]["params"]["amdFastEngine"] == "comfy"


def test_kandinsky_actual_sd_dims_and_audio_request(client, external_worker, monkeypatch):
    from services import amd_kandinsky_workflow

    def build(**kwargs):
        external_worker["kandinsky"] = kwargs
        return {"test": {}}, "90"

    monkeypatch.setattr(amd_kandinsky_workflow, "build_kandinsky_workflow", build)
    response = client.post("/api/generate", json=request(amdFastEngine="kandinsky", aspectRatio="9:16", audio=True))
    assert response.status_code == 200, response.text
    graph = external_worker["kandinsky"]
    assert (graph["width"], graph["height"], graph["frames"], graph["fps"]) == (480, 864, 121, 24)
    assert "graph" not in external_worker


def test_legacy_queued_job_and_recipe_keep_native_on_retry(test_state, external_worker, create_fake_model_files):
    create_fake_model_files()
    test_state.state.app_settings.amd_fast_engine = "comfy"
    params = TextToVideoParams.model_validate(request(model="ltx-2.5-fast"))
    queued, model = generate_request_from_offering_params(params, test_state.video_generation)
    assert queued.amdFastEngine == "native"
    assert test_state.video_generation._external_fast_engine(queued, model) is None
    recipe = ImageToVideoRecipeParams.model_validate(request(model="ltx-2.5-fast"))
    payload, _ = offering_payload_from_params(recipe)
    assert payload["amdFastEngine"] == "native"
