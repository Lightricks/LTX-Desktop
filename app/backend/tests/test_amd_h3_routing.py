"""Local H3 from the public HTTP contract through the real queue, up to the Comfy worker."""
from pathlib import Path

from PIL import Image
import pytest

from services.amd_comfy_client import AmdComfyClient, ComfyError, _sampling_steps
from services.amd_h3_models import acknowledge_h3_license
from services.features.video.image_to_video import ImageToVideoExecutor
from services.features.video.text_to_video import TextToVideoExecutor
from services.sqlite_store import SqliteStore
from tests.fakes.generation import _write_valid_mp4
from tests.test_amd_h3_workflow import info
from tests.test_http import wait_for_http_status


def params(engine, **overrides):
    return {"prompt": "Rain on the roofs", "amdFastEngine": engine, "resolution": "540p",
            "model": "ltx-2.5-fast", "duration": 5, "fps": 24, "seed": 143, **overrides}


@pytest.fixture
def worker(test_state, monkeypatch, h3_gpu):
    acknowledge_h3_license(test_state.config.app_data_dir)
    video = test_state.video_generation
    executors = test_state.executor_registry._executors
    monkeypatch.setitem(executors, "image-to-video", ImageToVideoExecutor(video, SqliteStore(test_state.config.app_data_dir)))
    monkeypatch.setitem(executors, "text-to-video", TextToVideoExecutor(video))
    calls = []

    class Worker:
        gpu_name = "Test GPU"
        backend = h3_gpu

        def get_object_info(self):
            return info()

        def upload_image(self, path):
            with Image.open(path) as image:
                return f"inputs/{image.getpixel((0, 0))[0]}.png"

        def run(self, graph, dest, *, output_node_id, on_progress, is_cancelled):
            assert graph[output_node_id]["class_type"] == "SaveVideo"
            assert not is_cancelled()
            steps = _sampling_steps(graph)["sample"]
            on_progress(steps, steps)
            _write_valid_mp4(dest)
            calls.append(graph)

    monkeypatch.setattr(AmdComfyClient, "discover", lambda **_: Worker())
    return calls


def image_ref(client, tmp_path, red, *, portrait=False):
    path = tmp_path / f"{red}.png"
    Image.new("RGB", (32, 64) if portrait else (32, 32), (red, 10, 80)).save(path)
    response = client.post("/api/assets", json={"path": str(path)})
    assert response.status_code == 200, response.text
    return {"assetId": response.json()["id"]}


def completed(client, created):
    assert created.status_code == 200, created.text
    job = created.json()
    assert wait_for_http_status(client, job["id"], "succeeded", timeout=5), client.get(f"/api/generations/{job['id']}").text
    return job


def test_actual_h3_available_shapes_independent_of_native_weights_and_license(client, test_state, h3_gpu):
    specs = client.get("/api/generate/models-specs").json()
    for key in ("h3_fl2va_local_model", "h3_fl2va_distill_local_model", "h3_ref2va_local_model"):
        row = specs[key]["spec"]
        assert row["capabilities"]["t2v"] == (key != "h3_ref2va_local_model")
        assert row["capabilities"]["i2v"]
        assert set(row["supported_resolutions_durations"]) == {"540p", "720p"}
        assert row["supported_resolutions_durations"]["720p"] == {
            "fps_to_durations": {"24": [5, 10, 15]}, "aspect_ratios": ["16:9", "9:16", "1:1"],
        }
    assert specs["downloaded_local_models"] == []
    test_state.config.local_generations_mode = "unsupported"
    assert client.get("/api/generate/models-specs").json()["h3_ref2va_local_model"] is None


@pytest.mark.parametrize("engine", ["h3-fl2va", "h3-fl2va-distill", "h3-ref2va"])
def test_unacknowledged_weights_never_render_or_enqueue(client, engine, tmp_path, h3_gpu):
    image = engine == "h3-ref2va"
    req = params(engine, model="fast", **({"imagePath": "unused.png"} if image else {}))
    response = client.post("/api/generate", json=req)
    assert response.status_code == 403 and response.json()["code"] == "H3_LICENSE_ACK_REQUIRED"
    feature = "image-to-video" if image else "text-to-video"
    body = {"params": params(engine)}
    if image:
        body["inputs"] = {"startFrame": image_ref(client, tmp_path, 7)}
    created = client.post(f"/api/generations/{feature}", json=body)
    assert created.status_code == 403 and created.json()["code"] == "H3_LICENSE_ACK_REQUIRED"


def test_distilled_queue_keeps_end_only_and_portrait_canvas(client, tmp_path, worker):
    end = image_ref(client, tmp_path, 33, portrait=True)
    job = completed(client, client.post("/api/generations/image-to-video", json={
        "params": params("h3-fl2va-distill", resolution="720p", duration=15, prompt=""),
        "inputs": {"endFrame": end},
    }))
    assert job["spec"]["inputs"] == {"endFrame": end}
    assert job["spec"]["params"]["amdFastEngine"] == "h3-fl2va-distill"
    graph, = worker
    condition = graph["condition"]["inputs"]
    assert "first_frame" not in condition and condition["last_frame"] == ["last", 0]
    assert graph["last"]["inputs"]["image"] == "inputs/33.png"
    assert (condition["width"], condition["height"], condition["length"]) == (768, 1344, 362)
    assert graph["noise"]["inputs"]["noise_seed"] == 143
    assert _sampling_steps(graph) == {"sample": 8}


def test_reference_queue_preserves_order_and_uses_real_ref_graph(client, tmp_path, worker):
    refs = [image_ref(client, tmp_path, r) for r in (4, 82, 5)]
    inputs = {"startFrame": refs[0], "referenceImages": refs[1:]}
    job = completed(client, client.post("/api/generations/image-to-video", json={
        "params": params("h3-ref2va", duration=10), "inputs": inputs,
    }))
    assert job["spec"]["inputs"] == inputs
    graph, = worker
    assert graph["condition"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    condition = graph["condition"]["inputs"]
    assert (condition["width"], condition["height"], condition["length"]) == (480, 480, 243)
    assert [graph[condition[f"ref_images.ref_image_{i}"][0]]["inputs"]["image"] for i in range(3)] == [
        "inputs/4.png", "inputs/82.png", "inputs/5.png",
    ]
    assert _sampling_steps(graph) == {"sample": 20}


def test_fl2va_queued_text_and_direct_render_stay_local_even_with_api_preference(client, worker, test_state):
    settings = test_state.state.app_settings
    settings.ltx_api_key = "test"
    settings.user_prefers_ltx_api_video_generations = True
    req = params("h3-fl2va")
    response = client.post("/api/generate", json={**req, "model": "fast"})
    assert response.status_code == 200, response.text
    assert Path(response.json()["video_path"]).is_file()
    completed(client, client.post("/api/generations/text-to-video", json={"params": req}))
    assert len(worker) == 2
    for graph in worker:
        assert graph["condition"]["inputs"]["length"] == 124
        assert graph["movie"]["inputs"]["audio"] == ["audio", 0]


@pytest.mark.parametrize("engine,changes", [
    ("h3-ref2va", {}), ("h3-ref2va", {"imagePath": "a.png", "lastImagePath": "b.png"}),
    ("h3-ref2va", {"imagePath": "a.png", "referenceImagePaths": ["b.png"] * 9}),
    ("h3-fl2va", {"referenceImagePaths": ["r.png"]}), ("h3-fl2va", {"negativePrompt": "blur"}),
    ("h3-fl2va", {"duration": None}), ("h3-fl2va", {"fps": 25}), ("h3-fl2va", {"resolution": "1080p"}),
    ("h3-fl2va", {"aspectRatio": "4:3"}), ("h3-fl2va", {"seed": -1}),
    ("h3-fl2va-distill", {"audioPath": "external.wav"}), ("native", {"referenceImagePaths": ["r.png"]}),
])
def test_invalid_h3_requests_never_reach_the_worker(client, worker, engine, changes):
    response = client.post("/api/generate", json=params(engine, **{"model": "fast", **changes}))
    assert response.status_code == 422, response.text
    assert worker == []


def test_missing_h3_worker_surfaces_error_without_hosted_or_native_fallback(client, worker, monkeypatch):
    def missing(**kwargs):
        raise ComfyError("H3 worker is offline")
    monkeypatch.setattr(AmdComfyClient, "discover", missing)
    response = client.post("/api/generate", json=params("h3-fl2va", model="fast"))
    assert response.status_code == 502 and response.json()["code"] == "AMD_COMFY_GENERATION_FAILED"
    assert "H3 worker is offline" in response.text
    assert worker == []


def test_worker_rechecks_license_when_a_queued_job_starts(client, test_state, worker, caplog):
    with test_state.generation.reserved_generation_start():
        created = client.post("/api/generations/text-to-video", json={"params": params("h3-fl2va")})
        assert created.status_code == 200
        (test_state.config.app_data_dir / "h3-license-v1.json").unlink()
    job_id = created.json()["id"]
    assert wait_for_http_status(client, job_id, "failed", timeout=5)
    assert client.get(f"/api/generations/{job_id}").json()["error_code"] == "CAPABILITY_FAILED"
    assert "Review and accept the MiniMax H3 model license" in caplog.text
    assert worker == []


def test_direct_reference_list_only_and_two_anchors_keep_the_original_roles(client, tmp_path, worker):
    first, last = tmp_path / "first.png", tmp_path / "last.png"
    for file, red in ((first, 25), (last, 70)):
        Image.new("RGB", (20, 30), (red, 0, 0)).save(file)
    response = client.post("/api/generate", json=params("h3-ref2va", model="fast", prompt="",
                                                         referenceImagePaths=[str(last), str(first)]))
    assert response.status_code == 200, response.text
    response = client.post("/api/generate", json=params("h3-fl2va", model="fast",
                                                         imagePath=str(first), lastImagePath=str(last)))
    assert response.status_code == 200, response.text
    reference, anchors = worker
    assert reference["reference_0"]["inputs"]["image"] == anchors["last"]["inputs"]["image"] == "inputs/70.png"
    assert reference["reference_1"]["inputs"]["image"] == anchors["first"]["inputs"]["image"] == "inputs/25.png"


def test_paired_queue_projection_returns_engine_and_ordered_asset_ids_without_paths():
    from remote.dto import project_remote_spec
    spec = project_remote_spec({
        "params": {"amdFastEngine": "h3-ref2va", "referenceImagePaths": ["/private/input.png"]},
        "inputs": {"startFrame": {"assetId": "ref1"},
                   "referenceImages": [{"assetId": "ref2", "path": "/private/second.png"}, {"assetId": "ref3"}]},
    }).model_dump(mode="json", exclude_none=True)
    assert spec == {
        "params": {"amdFastEngine": "h3-ref2va"},
        "inputs": {"startFrame": {"assetId": "ref1"}, "referenceImages": [{"assetId": "ref2"}, {"assetId": "ref3"}]},
    }
    assert project_remote_spec({"params": {"amdFastEngine": "/path/to/private/model"}}).params.amdFastEngine is None


def test_paired_create_and_get_round_trip_end_only_and_ordered_references(client, test_state, worker, tmp_path):
    from starlette.testclient import TestClient
    from remote.app import create_remote_app

    refs = [image_ref(client, tmp_path, red) for red in (51, 52, 53)]
    remote = create_remote_app(handler=test_state, remote_token="pair-token")
    auth = {"Authorization": "Bearer pair-token"}
    with TestClient(remote) as phone:
        for engine, inputs in (
            ("h3-fl2va-distill", {"endFrame": refs[0]}),
            ("h3-ref2va", {"startFrame": refs[1], "referenceImages": [refs[2], refs[0]]}),
        ):
            result = phone.post("/api/generations/image-to-video", headers=auth, json={
                "params": params(engine), "inputs": inputs,
            })
            assert result.status_code == 200, result.text
            generation = result.json()
            fetched = phone.get(f"/api/generations/{generation['id']}", headers=auth).json()
            for row in (generation, fetched):
                assert row["spec"]["params"]["amdFastEngine"] == engine
                assert row["spec"]["inputs"] == inputs
            assert wait_for_http_status(client, generation["id"], "succeeded", timeout=5)
    assert len(worker) == 2
