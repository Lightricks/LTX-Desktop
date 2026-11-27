r"""Smoke-test the installed private CUDA or ROCm engine with a short, weight-free video.

From the checkout, run in PowerShell:
    & "$env:LOCALAPPDATA\Programs\LTXDesktopAMD\application\resources\python\python.exe" .\checks\check_managed_comfy.py

The temporary engine, nodes and output MP4 are removed on exit.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "setupwork/src"))
from gpu_selection import child_environment, read_selection


def check(resources: Path, data: Path) -> None:
    selection = read_selection(resources)
    backend = selection["backend"]
    if backend not in {"cuda", "rocm"}:
        raise SystemExit("The private worker requires a local CUDA or ROCm runtime. This installation uses API mode.")
    # Apply the same adapter selection as Electron before importing torch.
    env = child_environment(selection)
    for key in os.environ.keys() - env.keys():
        del os.environ[key]
    os.environ.update(env, LTX_APP_DATA_DIR=str(data))
    sys.path.insert(0, str(resources / "backend"))
    from server_utils.win_dll_search import remove_cwd_from_dll_search_path
    remove_cwd_from_dll_search_path()
    import av
    import torch
    from runtime_config.accelerator import accelerator_backend, verify_rocm_execution
    from services.amd_comfy_engine import managed_comfy
    from services.amd_comfy_workflow import mp4_save_inputs
    from services.amd_kandinsky_node import KandinskyNodeSetup, kandinsky_nodes_loaded

    active = accelerator_backend()
    if active != backend:
        raise RuntimeError(f"The installation selected {backend}, but PyTorch reports {active}.")
    if backend == "rocm":
        verify_rocm_execution(require_gfx1151=True)
    elif torch.tensor([2.0, 3.0], device="cuda").square().sum().item() != 13.0:
        raise RuntimeError("The CUDA tensor calculation returned an unexpected result.")
    version = torch.version.hip if backend == "rocm" else torch.version.cuda
    print(f"{backend.upper()} {version} · {torch.cuda.get_device_name(0)}", flush=True)

    engine = managed_comfy()
    setup = KandinskyNodeSetup()
    try:
        engine.action("prepare")
        worker = engine.connect(start_prepared=True)
        if not worker.private or not engine.active_at(worker.base_url):
            raise RuntimeError("The private worker did not start.")
        installed = setup.run("install", managed_only=True)
        if installed.status == "restart_required":
            restarting = setup.run("restart", managed_only=True)
            if restarting.status != "restarting":
                raise RuntimeError(restarting.message)
            worker = engine.connect(start_prepared=True)
        elif installed.status != "ready":
            raise RuntimeError(installed.message)
        if not worker.private or not kandinsky_nodes_loaded(worker):
            raise RuntimeError("The private worker did not load the Kandinsky video and audio nodes.")

        graph = {
            "image": {"class_type": "EmptyImage", "inputs": {"height": 64, "width": 64, "batch_size": 4, "color": 0x3366AA}},
            "audio": {"class_type": "EmptyAudio", "inputs": {"duration": 4 / 24, "sample_rate": 44100, "channels": 2}},
            "movie": {"class_type": "CreateVideo", "inputs": {"images": ["image", 0], "audio": ["audio", 0], "fps": 24}},
            "file": {"class_type": "SaveVideo", "inputs": {
                "video": ["movie", 0], "filename_prefix": "smoke/clip", **mp4_save_inputs(worker.get_object_info()),
            }},
        }
        result = data / "streamed.mp4"
        worker.run(graph, result, output_node_id="file")
        with av.open(result) as media:
            video, audio = media.streams.video[0], media.streams.audio[0]
            if (video.width, video.height, sum(1 for _ in media.decode(video))) != (64, 64, 4):
                raise RuntimeError("The returned MP4 did not contain the expected four 64 × 64 frames.")
            print(f"{worker.gpu_name} · {video.codec_context.name} video · {audio.codec_context.name} audio", flush=True)
        with av.open(result) as media:
            if not any(frame.samples for frame in media.decode(audio=0)):
                raise RuntimeError("The returned MP4 did not contain decodable audio samples.")
        print("PASS · private setup, Kandinsky nodes, Comfy queue, MP4 streaming and decoding", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    installed = Path(sys.executable).resolve().parent.parent
    if not (installed / "comfy-engine" / "comfyui-v0.38.0.zip").is_file():
        raise SystemExit("Run this check with the installed LTX Desktop - AMD 395 MAX resources\\python\\python.exe.")
    with tempfile.TemporaryDirectory(prefix="ltx-comfy-check-") as temp:
        check(installed, Path(temp))
