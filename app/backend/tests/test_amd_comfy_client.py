"""Exercise the transport across real loopback HTTP and WebSocket connections."""

import io
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import av
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from services import amd_comfy_client
from services.amd_comfy_client import AmdComfyClient, ComfyCancelled, ComfyError

K6_GRAPH = {
    "k_sampler": {"class_type": "Kandinsky6Sampler", "inputs": {"steps": 10}},
    "90": {"class_type": "SaveVideo"},
}
LTX_GRAPH = {
    "a": {"class_type": "ManualSigmas", "inputs": {"sigmas": "1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0"}},
    "b": {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.85,0.7,0.4,0"}},
    "base": {"class_type": "SamplerCustomAdvanced", "inputs": {"sigmas": ["a", 0]}},
    "refine": {"class_type": "SamplerCustomAdvanced", "inputs": {"sigmas": ["b", 0]}},
    "90": {"class_type": "SaveVideo"},
}


@pytest.fixture(autouse=True)
def selected_radeon(monkeypatch):
    monkeypatch.setattr(amd_comfy_client, "_selected_radeon_name", lambda: "AMD Radeon 8060S Graphics")


def _mp4():
    result = io.BytesIO()
    with av.open(result, "w", format="mp4") as out:
        stream = out.add_stream("mpeg4", rate=24)
        stream.width, stream.height = 32, 32
        for packet in stream.encode(av.VideoFrame(32, 32, "yuv420p")):
            out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return result.getvalue()


@dataclass
class Worker:
    device: str = "cuda"
    version: str = "2.10.0+rocm7.13.0"
    gpu_name: str = "cuda:0 AMD Radeon 8060S Graphics : native"
    result: bytes = field(default_factory=_mp4)
    own: str = ""
    queue: set[str] = field(default_factory=lambda: {"second-user"})
    uploads: dict[str, bytes] = field(default_factory=dict)
    execution_error: dict | None = None
    pending: bool = False
    start: float = 0
    steps_seen: threading.Event = field(default_factory=threading.Event)
    websocket: WebSocket | None = None

    async def serve_events(self, websocket: WebSocket):
        self.websocket = websocket
        await websocket.accept()
        while (await websocket.receive())["type"] != "websocket.disconnect":
            pass

    async def handle(self, req: Request):
        if req.url.path == "/system_stats":
            return JSONResponse(
                {
                    "system": {"comfyui_version": "0.34.3", "pytorch_version": self.version},
                    "devices": [{"type": self.device, "name": self.gpu_name}],
                }
            )
        if req.url.path == "/upload/image":
            async with req.form() as fields:
                image = fields["image"]
                self.uploads[image.filename] = await image.read()
                return JSONResponse(
                    {"name": image.filename, "subfolder": fields["subfolder"], "type": fields["type"]}
                )
        if req.url.path == "/prompt":
            post = await req.json()
            self.own = post["prompt_id"]
            self.queue.add(self.own)
            self.start = time.monotonic()
            if self.websocket is not None:
                node = "k_sampler" if "k_sampler" in post["prompt"] else "refine"
                for pid, value in (("second-user", 10), (self.own, 1)):
                    await self.websocket.send_json(
                        {
                            "type": "progress",
                            "data": {
                                "node": node,
                                "max": 10 if node == "k_sampler" else 3,
                                "value": value,
                                "prompt_id": pid,
                            },
                        }
                    )
                if node == "refine":
                    await self.websocket.send_json(
                        {"type": "execution_cached", "data": {"nodes": ["base"], "prompt_id": self.own}}
                    )
            return JSONResponse({"number": 1, "prompt_id": self.own, "node_errors": {}})
        if req.url.path == "/queue":
            if req.method == "POST":
                for job in (await req.json())["delete"]:
                    if job != self.own or self.pending:
                        self.queue.discard(job)
                return Response()
            return JSONResponse({"queue_running": [[1, pid] for pid in self.queue], "queue_pending": []})
        if req.url.path == "/interrupt":
            payload = await req.json()
            if "prompt_id" not in payload:
                self.queue.clear()
            elif not self.pending and payload["prompt_id"] == self.own:
                self.queue.discard(self.own)
            return Response()
        if req.url.path == f"/history/{self.own}":
            if (
                self.websocket is not None
                and not self.steps_seen.is_set()
                and time.monotonic() - self.start < 3
            ):
                return JSONResponse({})
            if self.execution_error is not None:
                return JSONResponse(
                    {
                        self.own: {
                            "status": {
                                "status_str": "error",
                                "completed": False,
                                "messages": [["execution_error", self.execution_error]],
                            }
                        }
                    }
                )
            ref = {"filename": "drummer.mp4", "type": "output", "subfolder": "ltx"}
            return JSONResponse(
                {
                    self.own: {
                        "status": {"status_str": "success", "completed": True},
                        "outputs": {"90": {"images": [ref]}},
                    }
                }
            )
        if req.url.path == "/view" and req.query_params["filename"] == "drummer.mp4":
            return Response(self.result, media_type="video/mp4")
        return Response(status_code=404)


@contextmanager
def _serve(worker: Worker) -> Iterator[int]:
    app = Starlette(routes=[
        WebSocketRoute("/ws", worker.serve_events),
        Route("/{path:path}", worker.handle, methods=["GET", "POST"]),
    ])
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", lifespan="off", log_config=None, access_log=False,
        ws="websockets-sansio",
    ))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                if not thread.is_alive() or time.monotonic() >= deadline:
                    raise RuntimeError("Local ComfyUI test fixture failed to start.")
                time.sleep(0.01)
            yield port
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive(), "Local ComfyUI fixture failed to stop."


@pytest.mark.parametrize("graph,expected", [(K6_GRAPH, [(1, 10)]), (LTX_GRAPH, [(1, 11), (9, 11)])])
def test_downloads_only_submitted_prompts_mp4_and_steps(tmp_path, graph, expected):
    worker = Worker()
    seen = []

    def progress(current, total):
        seen.append((current, total))
        if len(seen) == len(expected):
            worker.steps_seen.set()

    with _serve(worker) as port:
        client = AmdComfyClient.discover((port,), backend="rocm")
        client.run(graph, tmp_path / "result.mp4", output_node_id="90", on_progress=progress)
    assert seen == expected
    assert (tmp_path / "result.mp4").read_bytes() == worker.result


@pytest.mark.parametrize("pending", [True, False])
def test_cancel_leaves_other_users_prompt_alone(tmp_path, pending):
    worker = Worker(pending=pending)
    with _serve(worker) as port:
        with pytest.raises(ComfyCancelled):
            AmdComfyClient.discover((port,), backend="rocm").run(
                K6_GRAPH, tmp_path / "out.mp4", output_node_id="90", is_cancelled=lambda: bool(worker.own)
            )
    assert worker.queue == {"second-user"}
    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.parametrize(
    ("device", "name", "version"),
    [
        ("cpu", "AMD Ryzen 395", "2.10.0+rocm7.13"),
        ("cuda", "cuda:0 NVIDIA", "2.10.0+cu130"),
        ("cuda", "cuda:0 AMD Radeon", "2.10.0+cu130"),
    ],
)
def test_rejects_servers_whose_primary_device_is_not_rocm(device, name, version):
    worker = Worker(device=device, gpu_name=name, version=version)
    with _serve(worker) as port, pytest.raises(ComfyError, match="must run on the Radeon"):
        AmdComfyClient.discover((port,), backend="rocm")


def test_interrupted_result_download_never_replaces_existing_file(tmp_path):
    worker = Worker(result=b"not a video")
    destination = tmp_path / "downloads"
    destination.mkdir()
    output = destination / "result.mp4"
    output.write_bytes(b"original")
    with _serve(worker) as port, pytest.raises(ComfyError, match="MP4"):
        AmdComfyClient.discover((port,), backend="rocm").run(K6_GRAPH, output, output_node_id="90")
    assert output.read_bytes() == b"original"
    assert list(destination.iterdir()) == [output]


def test_two_different_reference_images_with_same_basename_have_separate_names(tmp_path: Path):
    worker = Worker()
    names = []
    with _serve(worker) as port:
        client = AmdComfyClient.discover((port,), backend="rocm")
        for contents in (b"first image", b"second image", b"first image"):
            path = tmp_path / "source.png"
            path.write_bytes(contents)
            names.append(client.upload_image(path))
    assert names[0] == names[2]
    assert names[0] != names[1]
    assert sorted(worker.uploads.values()) == [b"first image", b"second image"]


def test_audio_workflow_rejects_silent_mp4(tmp_path):
    graph = dict(K6_GRAPH)
    graph["75"] = {"class_type": "CreateVideo", "inputs": {"audio": ["71", 0]}}
    graph["90"] = {"class_type": "SaveVideo", "inputs": {"video": ["75", 0]}}
    with _serve(Worker()) as port, pytest.raises(ComfyError, match="missing the audio"):
        AmdComfyClient.discover((port,), backend="rocm").run(graph, tmp_path / "silent.mp4", output_node_id="90")
    assert not (tmp_path / "silent.mp4").exists()


@pytest.mark.parametrize(
    ("node", "message", "trace", "show_hint"),
    [
        ("LTXVLatentUpsampler", "exception: access violation reading 0xE0", [], True),
        ("LTXVLatentUpsampler", "OSError", ["lib.vbar_allocate(self._devctx, size, device)"], True),
        ("LTXVLatentUpsampler", "model file missing", [], False),
        ("Kandinsky6Sampler", "access violation", [], False),
    ],
)
def test_only_upscaler_allocation_crash_suggests_comfy_launch_workaround(
    tmp_path, node, message, trace, show_hint
):
    error = {"node_type": node, "exception_message": message, "traceback": trace}
    with _serve(Worker(execution_error=error)) as port, pytest.raises(ComfyError) as failure:
        AmdComfyClient.discover((port,), backend="rocm").run(LTX_GRAPH, tmp_path / "video.mp4", output_node_id="90")
    text = str(failure.value)
    assert message in text
    assert ("--disable-dynamic-vram" in text) is show_hint


def test_nvidia_discovery_and_render_recheck_match_the_selected_uuid(tmp_path, monkeypatch):
    selected = "GPU-00112233-4455-6677-8899-aabbccddeeff"
    other = "GPU-ffeeddcc-bbaa-9988-7766-554433221100"
    model = "NVIDIA GeForce RTX 4090"
    devices = [(other, "NVIDIA GeForce RTX 3080"), (selected, model)]
    monkeypatch.setattr(amd_comfy_client, "_nvidia_inventory", lambda: devices)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", selected)
    worker = Worker(version="2.10.0+cu128", gpu_name=f"cuda:1 {model} : cudaMallocAsync")
    with _serve(worker) as port:
        client = AmdComfyClient.discover((port,), backend="cuda")
        assert client.gpu_name == worker.gpu_name
        client.run(K6_GRAPH, tmp_path / "ok.mp4", output_node_id="90")
        assert (tmp_path / "ok.mp4").read_bytes() == worker.result
        worker.own = ""
        worker.gpu_name = "cuda:0 NVIDIA GeForce RTX 3080 : native"
        with pytest.raises(ComfyError, match="selected NVIDIA adapter"):
            client.run(K6_GRAPH, tmp_path / "wrong.mp4", output_node_id="90")
        assert not worker.own
        # External /system_stats has no physical UUID and both cards now have the same name.
        devices[:] = [(other, model), (selected, model)]
        worker.gpu_name = f"cuda:0 {model} : native"
        with pytest.raises(ComfyError, match="selected NVIDIA adapter"):
            AmdComfyClient.discover((port,), backend="cuda")


@pytest.mark.parametrize(("device", "name", "version"), [
    ("cuda", "AMD Radeon 8060S Graphics", "2.10.0+rocm"),
    ("cpu", "NVIDIA", "2.10.0+cu128"),
    ("cuda", "NVIDIA", "2.10.0+cpu"),
])
def test_cuda_discovery_rejects_incompatible_workers(device, name, version):
    worker = Worker(device=device, gpu_name=name, version=version)
    with _serve(worker) as port, pytest.raises(ComfyError, match="NVIDIA GPU with CUDA"):
        AmdComfyClient.discover((port,), backend="cuda")


@pytest.mark.parametrize("reported", ["AMD Radeon RX 7900 XTX", "AMD Radeon(TM) Graphics"])
def test_discovery_skips_a_different_or_unidentifiable_radeon(reported, monkeypatch):
    # Generic identical strings in two processes may refer to different Radeon chips.
    monkeypatch.setattr(amd_comfy_client, "_selected_radeon_name", lambda: "AMD Radeon(TM) Graphics")
    worker = Worker(gpu_name=f"cuda:0 {reported} : native")
    with _serve(worker) as port, pytest.raises(ComfyError, match="does not identify a supported Radeon"):
        AmdComfyClient.discover((port,), backend="rocm")


@pytest.mark.parametrize("model", ["8040S", "8050S", "8060S"])
def test_discovery_checks_the_specific_radeon_sibling(model, monkeypatch):
    actual = f"AMD Radeon(TM) {model} Graphics"
    monkeypatch.setattr(amd_comfy_client, "_selected_radeon_name", lambda: actual)
    worker = Worker(gpu_name=f"cuda:2 {actual} : native")
    with _serve(worker) as port:
        assert AmdComfyClient.discover((port,), backend="rocm").gpu_name == worker.gpu_name
        worker.gpu_name = "cuda:1 AMD Radeon(TM) " + ("8050S" if model == "8040S" else "8040S") + " Graphics : native"
        with pytest.raises(ComfyError, match="differs from the selected Radeon"):
            AmdComfyClient.discover((port,), backend="rocm")


def test_only_owned_apple_metal_worker_is_accepted():
    stats = {"system": {"comfyui_version": "0.38.0", "os": "darwin", "pytorch_version": "2.13.1"},
             "devices": [{"name": "mps", "type": "mps"}]}
    client = AmdComfyClient(8123, backend="mps")
    with pytest.raises(ComfyError, match="app-managed Metal"):
        client.check_primary_gpu(stats)
    client.private = True
    client.check_primary_gpu(stats)
    assert client.gpu_name == "mps"
    for wrong in ({"name": "cpu", "type": "cpu"}, {"name": "NVIDIA GPU", "type": "cuda"}):
        with pytest.raises(ComfyError, match="Apple GPU"):
            client.check_primary_gpu({**stats, "devices": [wrong]})


def test_external_linux_cuda_requires_the_visible_devices_uuid(monkeypatch):
    from types import SimpleNamespace
    import torch

    selected = "00112233-4455-6677-8899-aabbccddeeff"
    name = "NVIDIA GeForce RTX 5090"
    monkeypatch.setattr(amd_comfy_client.sys, "platform", "linux")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda ordinal: SimpleNamespace(uuid=selected))
    monkeypatch.setattr(amd_comfy_client, "_nvidia_inventory", lambda: [("GPU-" + selected, name)])
    amd_comfy_client._check_external_gpu("cuda:0 " + name + " : native", "cuda")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda ordinal: SimpleNamespace(uuid="different"))
    with pytest.raises(ComfyError, match="selected NVIDIA"):
        amd_comfy_client._check_external_gpu("cuda:0 " + name + " : native", "cuda")
