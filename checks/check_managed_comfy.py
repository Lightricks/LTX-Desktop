r"""Smoke-test the installed private MPS, CUDA or ROCm engine with a short, weight-free video.

From the checkout, run in PowerShell:
    & "$env:LOCALAPPDATA\Programs\LTXDesktopAMD\application\resources\python\python.exe" .\checks\check_managed_comfy.py

The temporary engine, nodes and output MP4 are removed on exit.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "setupwork/src"))
from gpu_selection import child_environment, read_selection


def check(resources: Path, data: Path) -> None:
    os.environ["LTX_APP_DATA_DIR"] = str(data)
    sys.path.insert(0, str(resources / "backend"))
    if sys.platform == "win32":
        selection = read_selection(resources)
        backend = selection["backend"]
        if backend == "api":
            raise SystemExit("The Windows installer selected API-only mode.")
        # Use the adapter selected by the installer before loading PyTorch.
        env = child_environment(selection)
        for key in os.environ.keys() - env.keys():
            del os.environ[key]
        os.environ.update(env)
        from server_utils.win_dll_search import remove_cwd_from_dll_search_path
        remove_cwd_from_dll_search_path()
    elif sys.platform == "darwin":
        backend = "mps"
        # Imports during the check must preserve the signed application.
        sys.dont_write_bytecode = True
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    elif sys.platform == "linux":
        backend = "cuda"
    else:
        raise SystemExit("Run this check on a platform with a community installer.")

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
    else:
        device = "mps" if backend == "mps" else "cuda"
        values = torch.tensor([2.0, 3.0], device=device)
        if values.device.type != device or values.square().sum().item() != 13.0:
            raise RuntimeError(f"The {backend} GPU calculation failed.")
    print(f"Testing {backend.upper()} with PyTorch {torch.__version__}", flush=True)
    engine = managed_comfy()
    try:
        engine.action("prepare")
        worker = engine.connect(start_prepared=True)
        if not worker.private or not engine.active_at(worker.base_url):
            raise RuntimeError("The private worker did not start.")
        if sys.platform == "win32":
            setup = KandinskyNodeSetup()
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

        info = worker.get_object_info()
        required = {"MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo", "UNETLoader",
                    "CLIPLoader", "VAELoader", "LoraLoaderModelOnly", "BasicGuider", "KSamplerSelect",
                    "BasicScheduler", "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio", "CreateVideo",
                    "EmptyImage", "EmptyAudio", "SaveVideo", "LoadImage", "RandomNoise"}
        missing = required.difference(info)
        if missing:
            raise RuntimeError(f"Missing H3 worker nodes: {', '.join(sorted(missing))}")

        graph = {
            "image": {"class_type": "EmptyImage", "inputs": {"height": 64, "width": 64, "batch_size": 4, "color": 0x3366AA}},
            "audio": {"class_type": "EmptyAudio", "inputs": {"duration": 4 / 24, "sample_rate": 44100, "channels": 2}},
            "movie": {"class_type": "CreateVideo", "inputs": {"images": ["image", 0], "audio": ["audio", 0], "fps": 24}},
            "file": {"class_type": "SaveVideo", "inputs": {
                "video": ["movie", 0], "filename_prefix": "smoke/clip", **mp4_save_inputs(info),
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
        print("PASS · private worker, H3 node registration, Comfy queue, MP4 video and audio decoding", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # Windows ships python/python.exe; macOS ships python/bin/python3.
    default = Path(sys.executable).parent.parent
    if sys.platform == "darwin":
        default = default.parent
    parser.add_argument("--resources", type=Path, default=default,
                        help="Installed Electron resources directory (required for Linux)")
    installed = parser.parse_args().resources.resolve()
    if not (installed / "comfy-engine" / "comfyui-v0.38.0.zip").is_file():
        raise SystemExit("Run with the installed desktop interpreter and pass --resources for this installation.")
    with tempfile.TemporaryDirectory(prefix="ltx-comfy-check-") as temp:
        check(installed, Path(temp))
