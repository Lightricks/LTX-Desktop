"""Run a Desktop-owned prompt on an local CUDA or ROCm ComfyUI server."""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import re
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import httpx
from websockets.exceptions import WebSocketException
from websockets.sync.client import ClientConnection, connect

from runtime_config.community_runtime import LocalVideoBackend, local_video_backend

logger = logging.getLogger(__name__)

JSON = dict[str, Any]
_POLL_SECONDS = 2.0
_LOST_SECONDS = 30.0


class ComfyError(RuntimeError):
    """A local ComfyUI worker could not complete the requested operation."""


class ComfyCancelled(ComfyError):
    """The owned prompt was cancelled."""


def _obj(value: object) -> JSON:
    return cast(JSON, value) if isinstance(value, dict) else {}


def _items(value: object) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else []


def _response(response: httpx.Response) -> httpx.Response:
    if not response.is_success:
        detail = response.text.strip()[:1500]
        raise ComfyError(
            f"ComfyUI {response.request.url.path} returned HTTP {response.status_code}. {detail}"
        )
    return response


def comfy_json(response: httpx.Response) -> JSON:
    try:
        value = _response(response).json()
    except ValueError as exc:
        raise ComfyError("The local server did not return ComfyUI JSON.") from exc
    if not isinstance(value, dict):
        raise ComfyError("The local server returned an unexpected ComfyUI response.")
    return cast(JSON, value)


def _nvml_text(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _nvidia_inventory() -> list[tuple[str, str]]:
    """NVML reports physical adapters even when this process is pinned to one CUDA UUID."""
    try:
        import pynvml  # type: ignore[reportMissingModuleSource]
    except ImportError as exc:
        raise ComfyError("Could not verify the external NVIDIA adapter identity.") from exc
    try:
        pynvml.nvmlInit()
        try:
            adapters: list[tuple[str, str]] = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                identity = _nvml_text(pynvml.nvmlDeviceGetUUID(handle))
                model = _nvml_text(pynvml.nvmlDeviceGetName(handle))
                adapters.append((identity, model))
            return adapters
        finally:
            pynvml.nvmlShutdown()
    except (OSError, pynvml.NVMLError) as exc:
        raise ComfyError("Could not verify the external NVIDIA adapter identity.") from exc


def _selected_radeon_name() -> str:
    import torch

    try:
        return torch.cuda.get_device_name(0)
    except (RuntimeError, AssertionError) as exc:
        raise ComfyError("Could not verify the selected Radeon adapter.") from exc


def _check_external_gpu(name: str, backend: LocalVideoBackend) -> None:
    # ComfyUI's indices are process-local and /system_stats omits physical UUIDs.
    # A unique model name in the driver's inventory lets us verify the selected GPU.
    # Otherwise the managed worker is available with its UUID pinned at launch.
    primary = re.sub(r"^cuda:\d+\s+", "", name, flags=re.I).partition(" :")[0].strip().casefold()
    if backend == "rocm":
        # Two different Radeon architectures can both be reported as "AMD Radeon Graphics".
        # The private worker inherits the verified HIP ordinal even on those machines.
        if not re.search(r"\b(?:8040s|8050s|8060s)\b", primary):
            raise ComfyError(f"External ComfyUI's primary GPU ({name}) does not identify a supported Radeon model.")
        if primary != _selected_radeon_name().casefold():
            raise ComfyError(f"External ComfyUI's primary GPU ({name}) differs from the selected Radeon.")
        return
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "").casefold()
    matching = [identity.casefold() for identity, model in _nvidia_inventory()
                if model.casefold() == primary]
    if not selected or matching != [selected]:
        raise ComfyError(f"Could not match external ComfyUI's primary GPU ({name}) to the selected NVIDIA adapter.")


def gpu_name_from_stats(stats: JSON, backend: LocalVideoBackend) -> str:
    system = _obj(stats.get("system"))
    devices = _items(stats.get("devices"))
    device = _obj(devices[0]) if devices else {}
    name = str(device.get("name", ""))
    version = str(system.get("pytorch_version", "")).lower()
    is_amd = bool(re.search(r"amd|radeon|gfx1151", name, re.I))
    if backend == "rocm":
        compatible = (device.get("type") in ("cuda", "hip") and is_amd
                      and ("rocm" in version or "hip" in version))
        required = "the Radeon with ROCm PyTorch"
    else:
        compatible = (device.get("type") == "cuda" and bool(name) and not is_amd
                      and bool(re.search(r"\+cu\d+", version)))
        required = "the selected NVIDIA GPU with CUDA PyTorch"
    if not compatible:
        raise ComfyError(
            f"ComfyUI must run on {required}. Reported {name or 'no GPU'}, torch {version or 'unknown'}."
        )
    # Older ComfyUI servers ignore prompt_id on /interrupt and stop another user's active job.
    match = re.match(r"(\d+)\.(\d+)", str(system.get("comfyui_version", "")))
    if not match or tuple(map(int, match.groups())) < (0, 4):
        raise ComfyError(
            "Update ComfyUI Desktop to a version with prompt-specific interruption (ComfyUI core 0.4 or later)."
        )
    return name


def _video_ref(record: JSON, output_node_id: str, backend: LocalVideoBackend) -> dict[str, str]:
    status = _obj(record.get("status"))
    if status.get("status_str") != "success" or not status.get("completed"):
        for raw_event in reversed(_items(status.get("messages"))):
            event = _items(raw_event)
            if len(event) == 2:
                data = _obj(event[1])
                if event[0] == "execution_interrupted":
                    raise ComfyCancelled("The ComfyUI job was interrupted.")
                if event[0] == "execution_error":
                    node = data.get("node_type", "execution")
                    message = str(data.get("exception_message", "unknown error")).strip()
                    trace = " ".join(str(line) for line in _items(data.get("traceback")))
                    detail = f"{message} {trace}".casefold()
                    if backend == "rocm" and node == "LTXVLatentUpsampler" and (
                        "vbar_allocate" in detail or "access violation" in detail
                    ):
                        message += (
                            " For this crash on Windows Radeon, add --disable-dynamic-vram to "
                            "ComfyUI Desktop's launch arguments, restart ComfyUI Desktop, and retry."
                        )
                    raise ComfyError(f"ComfyUI {node} failed: {message}")
        raise ComfyError("ComfyUI ended the prompt without a successful execution.")
    output = _obj(_obj(record.get("outputs")).get(output_node_id))
    for entries in output.values():
        for item in _items(entries):
            entry = _obj(item)
            filename = entry.get("filename")
            subfolder = entry.get("subfolder", "")
            if (
                isinstance(filename, str)
                and filename.lower().endswith(".mp4")
                and entry.get("type") == "output"
                and isinstance(subfolder, str)
            ):
                return {"filename": filename, "subfolder": subfolder, "type": "output"}
    raise ComfyError("The requested SaveVideo node did not publish an MP4 in ComfyUI history.")


def _check_cancelled(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise ComfyCancelled("Cancelled the local ComfyUI job.")


def _sampling_steps(workflow: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node_id, node in workflow.items():
        inputs = _obj(node.get("inputs"))
        if node.get("class_type") == "SamplerCustomAdvanced":
            link = _items(inputs.get("sigmas"))
            if link:
                schedule = _obj(workflow.get(str(link[0])))
                sigmas = _obj(schedule.get("inputs")).get("sigmas")
                if schedule.get("class_type") == "ManualSigmas" and isinstance(sigmas, str):
                    counts[node_id] = max(0, len(sigmas.split(",")) - 1)
                elif schedule.get("class_type") == "BasicScheduler":
                    steps = _obj(schedule.get("inputs")).get("steps")
                    if isinstance(steps, int) and steps > 0:
                        counts[node_id] = steps
        elif node.get("class_type") == "Kandinsky6Sampler":
            count = inputs.get("steps")
            if isinstance(count, int) and count > 0:
                counts[node_id] = count
    return counts


class AmdComfyClient:
    def __init__(self, port: int, *, backend: LocalVideoBackend):
        if not 0 < port < 65536:
            raise ValueError("Invalid loopback port.")
        self.base_url = f"http://127.0.0.1:{port}"
        self.backend = backend
        self.gpu_name = ""
        self.private = False

    def check_primary_gpu(self, stats: JSON) -> None:
        name = gpu_name_from_stats(stats, self.backend)
        if not self.private:
            _check_external_gpu(name, self.backend)
        self.gpu_name = name

    def _http(self) -> httpx.Client:
        # No proxy, DNS or redirects; all image, prompt and result traffic remains on loopback.
        return httpx.Client(base_url=self.base_url, trust_env=False, follow_redirects=False, timeout=30.0)

    @classmethod
    def discover(cls, ports: tuple[int, ...] | None = None, *, backend: LocalVideoBackend | None = None,
                 start_prepared: bool = False,
                 cancelled: Callable[[], bool] | None = None) -> AmdComfyClient:
        if ports is None:
            from services.amd_comfy_engine import managed_comfy
            return managed_comfy().connect(start_prepared=start_prepared, cancelled=cancelled)
        backend = backend or local_video_backend()
        if backend is None:
            raise ComfyError("Local ComfyUI requires a supported CUDA or ROCm installation on Windows.")
        errors: list[str] = []
        for port in ports:
            client = cls(port, backend=backend)
            try:
                with client._http() as http:
                    client.check_primary_gpu(comfy_json(http.get("/system_stats", timeout=2.5)))
                return client
            except (httpx.HTTPError, ComfyError) as exc:
                errors.append(f"{port}: {exc}")
        raise ComfyError("Open ComfyUI Desktop on the selected GPU. " + " | ".join(errors))

    def get_object_info(self) -> JSON:
        with self._http() as http:
            return comfy_json(http.get("/object_info", timeout=90))

    def get_folder_paths(self) -> dict[str, list[str]]:
        with self._http() as http:
            paths = comfy_json(http.get("/internal/folder_paths"))
        return {k: [p for p in _items(v) if isinstance(p, str)] for k, v in paths.items()}

    def upload_image(self, path: Path) -> str:
        with path.open("rb") as image, self._http() as http:
            # Reusing a reference never clobbers a Comfy user file or uploads endless numbered copies.
            filename = hashlib.file_digest(image, "sha256").hexdigest() + path.suffix.lower()
            image.seek(0)
            folder = "ltx-desktop-amd"
            res = comfy_json(
                http.post(
                    "/upload/image",
                    files={
                        "image": (
                            filename,
                            image,
                            mimetypes.guess_type(filename)[0] or "application/octet-stream",
                        )
                    },
                    data={"type": "input", "subfolder": folder, "overwrite": "true"},
                )
            )
        if (res.get("type"), res.get("subfolder"), res.get("name")) != ("input", folder, filename):
            raise ComfyError("ComfyUI did not accept the reference image into its expected input folder.")
        return f"{folder}/{filename}"

    def run(
        self,
        workflow: Mapping[str, Any],
        output_path: Path,
        *,
        output_node_id: str,
        on_progress: Callable[[int, int], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        from services.amd_comfy_engine import managed_comfy
        with managed_comfy().rendering(self):
            self._run(workflow, output_path, output_node_id=output_node_id, on_progress=on_progress, is_cancelled=is_cancelled)

    def _run(self, workflow: Mapping[str, Any], output_path: Path, *, output_node_id: str,
             on_progress: Callable[[int, int], None] | None, is_cancelled: Callable[[], bool] | None) -> None:
        if output_node_id not in workflow:
            raise ComfyError("The requested output node is missing from the workflow.")
        prompt_id = str(uuid.uuid4())
        client_id = uuid.uuid4().hex
        samplers = _sampling_steps(workflow)
        events: ClientConnection | None = None
        with self._http() as http:
            # Re-read the primary device just before publishing, in case the worker was restarted.
            self.check_primary_gpu(comfy_json(http.get("/system_stats")))
            _check_cancelled(is_cancelled)
            if on_progress is not None:
                try:
                    events = connect(
                        self.base_url.replace("http:", "ws:", 1) + f"/ws?clientId={client_id}",
                        proxy=None,
                        open_timeout=3,
                        close_timeout=1,
                    )
                except (OSError, WebSocketException) as exc:
                    logger.info("ComfyUI live steps unavailable, polling job history: %s", exc)
            finished = False
            try:
                reply = comfy_json(
                    http.post(
                        "/prompt", json={"prompt": workflow, "prompt_id": prompt_id, "client_id": client_id}
                    )
                )
                if (
                    reply.get("prompt_id") != prompt_id
                    or not isinstance(reply.get("number"), (int, float))
                    or reply.get("node_errors")
                ):
                    raise ComfyError(
                        f"ComfyUI did not confirm the submitted prompt {prompt_id}: {str(reply)[:750]}"
                    )
                logger.info("ComfyUI accepted local prompt %s on %s.", prompt_id, self.gpu_name)
                record = self._wait(http, prompt_id, events, samplers, on_progress, is_cancelled)
                finished = True
                source_link = _items(_obj(_obj(workflow[output_node_id]).get("inputs")).get("video"))
                source = _obj(workflow.get(str(source_link[0]))) if source_link else {}
                expects_audio = source.get("class_type") == "CreateVideo" and bool(
                    _obj(source.get("inputs")).get("audio")
                )
                self._download(
                    http, _video_ref(record, output_node_id, self.backend), output_path, is_cancelled, expects_audio
                )
            except BaseException as failure:
                if not finished:
                    try:
                        # Delete only this waiting job, then interrupt only this ID if it raced into execution.
                        _response(http.post("/queue", json={"delete": [prompt_id]}, timeout=4))
                        _response(http.post("/interrupt", json={"prompt_id": prompt_id}, timeout=4))
                    except (httpx.HTTPError, ComfyError) as cleanup_error:
                        raise ComfyError(
                            f"Could not confirm cancellation of local ComfyUI prompt {prompt_id}. Check its queue. {cleanup_error}"
                        ) from failure
                raise
            finally:
                if events is not None:
                    events.close()

    @staticmethod
    def _wait(
        http: httpx.Client,
        prompt_id: str,
        events: ClientConnection | None,
        samplers: dict[str, int],
        progress: Callable[[int, int], None] | None,
        cancel: Callable[[], bool] | None,
    ) -> JSON:
        total = sum(samplers.values())
        completed = dict.fromkeys(samplers, 0)
        last_step = 0
        next_poll = 0.0
        last_seen = time.monotonic()
        while True:
            _check_cancelled(cancel)
            now = time.monotonic()
            if now >= next_poll:
                try:
                    history = comfy_json(http.get(f"/history/{prompt_id}", timeout=5))
                    if prompt_id in history:
                        return _obj(history[prompt_id])
                    queue = comfy_json(http.get("/queue", timeout=5))
                    if any(
                        len(row) > 1 and row[1] == prompt_id
                        for key in ("queue_running", "queue_pending")
                        for row in map(_items, _items(queue.get(key)))
                    ):
                        last_seen = now
                except httpx.TransportError as exc:
                    if now - last_seen >= _LOST_SECONDS:
                        raise ComfyError(f"Lost contact with ComfyUI while waiting for {prompt_id}.") from exc
                if now - last_seen >= _LOST_SECONDS:
                    raise ComfyError(f"ComfyUI lost prompt {prompt_id} from both queue and history.")
                next_poll = now + _POLL_SECONDS
            if events is None:
                time.sleep(min(0.5, max(0, next_poll - time.monotonic())))
                continue
            try:
                raw = events.recv(timeout=0.5)
                if not isinstance(raw, str):
                    continue
                message = _obj(json.loads(raw))
                data = _obj(message.get("data"))
                if data.get("prompt_id") != prompt_id or progress is None:
                    continue
                node_id = str(data.get("node"))
                if message.get("type") == "progress" and node_id in samplers:
                    value, maximum = data.get("value"), data.get("max")
                    if (
                        isinstance(value, int)
                        and isinstance(maximum, int)
                        and 0 < maximum
                        and 0 <= value <= maximum
                    ):
                        completed[node_id] = max(completed[node_id], value * samplers[node_id] // maximum)
                elif message.get("type") == "execution_cached":
                    for cached in data.get("nodes", []):
                        if str(cached) in samplers:
                            completed[str(cached)] = samplers[str(cached)]
                step = sum(completed.values())
                if step > last_step:
                    last_step = step
                    progress(step, total)
            except TimeoutError:
                continue
            except (OSError, WebSocketException, ValueError):
                events = None  # REST history remains authoritative after the progress channel closes.

    @staticmethod
    def _download(
        http: httpx.Client,
        reference: dict[str, str],
        output: Path,
        cancel: Callable[[], bool] | None,
        expects_audio: bool,
    ) -> None:
        import av
        from av.error import FFmpegError

        _check_cancelled(cancel)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp: Path | None = None
        try:
            with http.stream("GET", "/view", params=reference, timeout=60) as response:
                if not response.is_success:
                    raise ComfyError(
                        f"ComfyUI could not serve the finished MP4 (HTTP {response.status_code})."
                    )
                with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as file:
                    temp = Path(file.name)
                    for chunk in response.iter_bytes(1024 * 1024):
                        _check_cancelled(cancel)
                        file.write(chunk)
            with av.open(str(temp)) as container:
                if (
                    "mp4" not in container.format.name.split(",")
                    or not container.streams.video
                    or next(container.decode(video=0), None) is None
                ):
                    raise ComfyError("The returned result was not a playable MP4 video.")
                if expects_audio and not container.streams.audio:
                    raise ComfyError("The returned MP4 is missing the audio track requested by the workflow.")
            _check_cancelled(cancel)
            os.replace(temp, output)
        except FFmpegError as exc:
            raise ComfyError("ComfyUI returned an unreadable MP4 video.") from exc
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
