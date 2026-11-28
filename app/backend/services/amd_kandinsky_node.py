"""Install and verify the official Kandinsky pack on the selected local Comfy worker."""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import httpx
from pydantic import BaseModel

from services.amd_comfy_client import AmdComfyClient, ComfyError
from services.amd_kandinsky_pack import ARCHIVE_NAME, KandinskyPackError, install_pack, pack_installed

Action = Literal["status", "install", "restart"]
NodeState = Literal["ready", "available", "installing", "restart_required", "restarting", "busy", "manual", "offline"]
JSON = dict[str, Any]
_PACK = "kandinsky6"
_VERSION = "1.0.2"
_MANAGER = "/v2/manager"
_INSTALLED = "/v2/customnode/installed"
_GUIDE = (
    "In ComfyUI Desktop, open Manager → Custom Nodes, search kandinsky6 by kandinskylab, "
    "install version 1.0.2 and restart ComfyUI when its queues are empty."
)
_NODES = (
    "Kandinsky6Sampler", "Kandinsky6TextEncode", "Kandinsky6EmptyLatent",
    "Kandinsky6AudioVAELoader", "Kandinsky6AudioVAEDecode",
    "Kandinsky6ImageToVideoAudio", "Kandinsky6RemoveReferenceLatent",
)


class KandinskyNodeSetupStatus(BaseModel):
    status: NodeState
    message: str
    canInstall: bool
    canRestart: bool


def _state(status: NodeState, message: str) -> KandinskyNodeSetupStatus:
    return KandinskyNodeSetupStatus(status=status, message=message,
                                   canInstall=status == "available", canRestart=status == "restart_required")


def _get(http: httpx.Client, path: str, **params: str) -> JSON:
    response = http.get(path, params=params)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected ComfyUI response")
    return cast(JSON, data)


def _pack(records: JSON) -> JSON:
    for value in records.values():
        if isinstance(value, dict):
            record = cast(JSON, value)
            if record.get("cnr_id") == _PACK or record.get("aux_id") == "kandinskylab/kandinsky-6":
                return record
    return {}


def _queue(http: httpx.Client, **params: str) -> tuple[int, int, bool]:
    data = _get(http, f"{_MANAGER}/queue/status", **params)
    pending, active, processing = data.get("pending_count"), data.get("in_progress_count"), data.get("is_processing")
    if type(pending) is not int or type(active) is not int or type(processing) is not bool or min(pending, active) < 0:
        raise ValueError("Unknown ComfyUI Manager queue state")
    return pending, active, processing


def _worker_http(worker: AmdComfyClient) -> httpx.Client:
    return httpx.Client(base_url=worker.base_url, timeout=8, trust_env=False, follow_redirects=False)


def _connected() -> httpx.Client:
    return _worker_http(AmdComfyClient.discover())


def _core_version(http: httpx.Client) -> tuple[str, bool]:
    system = _get(http, "/system_stats").get("system")
    version = str(cast(JSON, system).get("comfyui_version", "")) if isinstance(system, dict) else ""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    return version, bool(match and tuple(map(int, match.groups())) >= (0, 38, 0))


def _nodes_loaded(http: httpx.Client) -> bool:
    return all(node in _get(http, f"/object_info/{node}") for node in _NODES)


def kandinsky_nodes_loaded(worker: AmdComfyClient) -> bool:
    """Verify core compatibility and all text/image-to-video audio nodes on this worker."""
    try:
        with _worker_http(worker) as http:
            return _core_version(http)[1] and _nodes_loaded(http)
    except (httpx.HTTPError, ValueError) as exc:
        raise ComfyError("Could not inspect the Kandinsky nodes in ComfyUI.") from exc


class _EngineResult(Protocol):
    @property
    def status(self) -> str: ...

    @property
    def message(self) -> str: ...


class _PrivateEngine(Protocol):
    @property
    def custom_nodes_dir(self) -> Path: ...

    def restart(self) -> _EngineResult: ...


def _owned_engine(origin: str) -> _PrivateEngine | None:
    from services.amd_comfy_engine import managed_comfy

    engine = managed_comfy()
    return engine if engine.active_at(origin.rstrip("/")) else None


def _generation_busy(http: httpx.Client) -> bool:
    jobs = _get(http, "/queue")
    if not all(isinstance(jobs.get(key), list) for key in ("queue_running", "queue_pending")):
        raise ValueError("Could not verify the ComfyUI generation queue")
    return bool(jobs["queue_running"] or jobs["queue_pending"])


class KandinskyNodeSetup:
    def __init__(
        self, connect: Callable[[], httpx.Client] = _connected,
        owned_engine: Callable[[str], _PrivateEngine | None] = _owned_engine,
        archive: Path | None = None,
    ) -> None:
        self._connect = connect
        self._owned_engine = owned_engine
        self._archive = archive or Path(__file__).resolve().parents[2] / "comfy-engine" / ARCHIVE_NAME
        self._private_restart = False
        self._change_lock = threading.Lock()
        self._client_id = f"LTXDesktopAMD-{uuid.uuid4()}"
        self._origin: str | None = None
        self._task_id: str | None = None
        self._restart_at: float | None = None

    def run(self, action: Action = "status", *, managed_only: bool = False) -> KandinskyNodeSetupStatus:
        # Serialize explicit changes and make overlapping status polls cheap.
        if not self._change_lock.acquire(blocking=False):
            return _state("busy", "LTX Desktop - AMD 395 MAX is checking ComfyUI. Refresh in a moment.")
        try:
            try:
                http = self._connect()
            except ComfyError as exc:
                if self._restart_pending():
                    return _state("restarting", "Waiting for the ComfyUI server to reopen.")
                return _state("offline", str(exc))
            with http:
                origin = str(http.base_url)
                engine = self._owned_engine(origin)
                if managed_only and engine is None:
                    return _state("busy", "The private worker is unavailable. No change was made to the external ComfyUI server.")
                if self._origin and self._origin != origin:
                    private_reconnected = self._private_restart and engine is not None
                    if self._restart_pending() and not private_reconnected:
                        return _state("restarting", f"Waiting for ComfyUI at {self._origin} to reopen.")
                    if not private_reconnected:
                        self._task_id = self._restart_at = None
                        self._private_restart = False
                        if action != "status":
                            return _state("busy", "The active ComfyUI server changed. Refresh before making changes.")
                self._origin = origin
                if engine is not None:
                    try:
                        return self._private_action(http, engine, action)
                    except (ComfyError, KandinskyPackError, OSError, ValueError, httpx.HTTPError) as exc:
                        return _state("manual", f"Private Kandinsky setup could not finish. {exc}")
                try:
                    status = self._status(http)
                    if action == "install" and status.canInstall:
                        return self._install(http)
                    if action == "restart" and status.canRestart:
                        return self._restart(http)
                    return status
                except httpx.HTTPStatusError as exc:
                    code = exc.response.status_code
                    if action == "restart":
                        return _state("manual", f"ComfyUI Manager returned HTTP {code}. When the queues are empty, quit and reopen ComfyUI Desktop, then refresh.")
                    return _state("manual", f"The local ComfyUI Manager returned HTTP {code}. {_GUIDE}")
                except (httpx.HTTPError, ValueError):
                    return _state("manual", f"LTX Desktop - AMD 395 MAX could not verify the local ComfyUI Manager response. {_GUIDE}")
        finally:
            self._change_lock.release()

    def _restart_pending(self) -> bool:
        return self._restart_at is not None and time.monotonic() - self._restart_at < 90

    def _private_action(
        self, http: httpx.Client, engine: _PrivateEngine, action: Action
    ) -> KandinskyNodeSetupStatus:
        version, compatible = _core_version(http)
        if not compatible:
            return _state("manual", f"The private server reports ComfyUI {version or 'unknown'}. Reinstall the current LTX Desktop - AMD 395 MAX release.")
        if _nodes_loaded(http):
            self._private_restart = False
            self._restart_at = None
            return _state("ready", "The private engine has loaded the Kandinsky video and audio nodes.")
        if self._private_restart:
            return _state("manual", "The private engine restarted but the Kandinsky nodes did not load. Open the private engine log for the startup error.")
        if _generation_busy(http):
            return _state("busy", "The private engine has active or queued generations. Setup can continue when they finish.")
        if not pack_installed(engine.custom_nodes_dir):
            if action != "install":
                return _state("available", f"Install the bundled Kandinsky {_VERSION} nodes in the private engine.")
            install_pack(engine.custom_nodes_dir, self._archive)
        if action != "restart":
            return _state("restart_required", "The Kandinsky nodes are installed. Restart the idle private engine to load them.")
        result = engine.restart()
        if result.status not in ("starting", "installing"):
            # The worker stays ready if its last queue/lease check refuses the restart.
            return _state("busy" if result.status == "ready" else "manual", result.message)
        self._private_restart = True
        self._restart_at = time.monotonic()
        return _state("restarting", "Restarting the private engine to load Kandinsky.")

    def _status(self, http: httpx.Client) -> KandinskyNodeSetupStatus:
        version, compatible = _core_version(http)
        if not compatible:
            return _state("manual", f"Kandinsky needs ComfyUI core 0.38.0 or newer. This server reports {version or 'an unknown version'}. Update ComfyUI Desktop and refresh.")
        if _nodes_loaded(http):
            self._task_id = self._restart_at = None
            return _state("ready", "ComfyUI has loaded the Kandinsky video and audio nodes.")
        if self._restart_pending():
            return _state("restarting", "Waiting for ComfyUI to load the Kandinsky nodes.")
        installed = _pack(_get(http, _INSTALLED))
        if installed and installed.get("enabled") is not True:
            return _state("manual", "Enable the Kandinsky pack in ComfyUI Manager, then restart ComfyUI when idle.")
        manager_queue = _queue(http)
        if manager_queue != (0, 0, False):
            if self._task_id:
                pending, active, processing = _queue(http, client_id=self._client_id)
                if active or (pending and manager_queue[2]):
                    return _state("installing", "ComfyUI Manager is installing Kandinsky. Check Manager for download or dependency details.")
                if pending and not processing:
                    return _state("manual", "Kandinsky is queued. Review the task queue in ComfyUI Manager and start it there.")
            return _state("busy", "ComfyUI Manager has other work queued. Refresh after it finishes.")
        if _generation_busy(http):
            return _state("busy", "ComfyUI has active or queued generations. Refresh when they finish.")
        if installed:
            previous = _pack(_get(http, _INSTALLED, mode="imported"))
            if previous == installed:
                return _state("manual", "The Kandinsky pack is installed, but its video and audio nodes did not load. Check ComfyUI's startup log and use Manager to repair its dependencies.")
            return _state("restart_required", "Kandinsky is installed. Restart the idle ComfyUI server to load its nodes.")
        if self._task_id:
            history = _get(http, f"{_MANAGER}/queue/history", ui_id=self._task_id).get("history")
            result = str(cast(JSON, history).get("result", ""))[:240] if isinstance(history, dict) else ""
            return _state("available", f"Manager finished without installing Kandinsky{f' ({result})' if result else ''}. Check its task history before retrying. {_GUIDE}")
        return _state("available", f"Install Kandinsky {_VERSION} by kandinskylab through ComfyUI Manager.")

    def _install(self, http: httpx.Client) -> KandinskyNodeSetupStatus:
        task_id = str(uuid.uuid4())
        http.post(f"{_MANAGER}/queue/task", json={
            "kind": "install", "ui_id": task_id, "client_id": self._client_id,
            "params": {"id": _PACK, "version": _VERSION, "selected_version": _VERSION,
                       "channel": "default", "mode": "cache"},
        }).raise_for_status()
        self._task_id = task_id
        # Starting Manager's worker releases its entire queue. Verify that it contains
        # just the task we submitted, even if another client enqueued in the meantime.
        if _queue(http) == (1, 0, False) and _queue(http, client_id=self._client_id) == (1, 0, False):
            http.post(f"{_MANAGER}/queue/start", json={}).raise_for_status()
            return _state("installing", "ComfyUI Manager accepted the Kandinsky installation. Refresh to check its progress.")
        return self._status(http)

    def _restart(self, http: httpx.Client) -> KandinskyNodeSetupStatus:
        try:
            http.post(f"{_MANAGER}/reboot", json={}).raise_for_status()
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout):
            # Manager replaces its process inside this handler and may close the socket.
            pass
        self._restart_at = time.monotonic()
        return _state("restarting", "Restart requested. Refresh to check that ComfyUI loaded the nodes. If it stays offline, reopen ComfyUI Desktop.")
