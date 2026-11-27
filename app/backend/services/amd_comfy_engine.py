"""Process ownership and on-demand startup for the private ComfyUI worker."""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Generator
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import httpx
from pydantic import BaseModel

from runtime_config.community_runtime import local_video_backend
from services.amd_comfy_client import (
    AmdComfyClient,
    ComfyCancelled,
    ComfyError,
    comfy_json,
)

if TYPE_CHECKING:
    from services.amd_comfy_provision import ManagedComfyLayout

EngineState = Literal[
    "unsupported",
    "not_installed",
    "installing",
    "stopped",
    "starting",
    "ready",
    "external",
    "error",
]
EngineAction = Literal["prepare", "start", "stop"]
_START_TIMEOUT = 240


class ComfyEngineStatus(BaseModel):
    status: EngineState
    message: str
    progress: float | None = None
    canPrepare: bool = False
    canStart: bool = False
    canStop: bool = False


def _allocate_port(preferred: int | None) -> int:
    # Reuse our previous port across idle restarts when available. All external
    # Comfy ports and any other local listeners are left alone.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if os.name == "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        elif preferred:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if preferred:
            try:
                sock.bind(("127.0.0.1", preferred))
            except OSError:
                preferred = None
        if not preferred:
            sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _PortConflict(ComfyError):
    pass


def _owned_client(layout: ManagedComfyLayout, port: int, pid: int) -> AmdComfyClient:
    client = AmdComfyClient(port, backend=layout.backend)
    with httpx.Client(
        base_url=client.base_url, trust_env=False, follow_redirects=False, timeout=2
    ) as http:
        stats = comfy_json(http.get("/system_stats"))
    # A different local application might win the small gap between reserving
    # and binding the port. Only attach if the spawned process owns the listener.
    import psutil

    try:
        owns_listener = any(
            connection.status == psutil.CONN_LISTEN and connection.laddr.port == port
            for connection in psutil.Process(pid).net_connections(kind="tcp4")
        )
    except psutil.Error as exc:
        raise ComfyError(
            "Could not verify the private engine process and its listening port."
        ) from exc
    if not owns_listener:
        raise _PortConflict("Another process owns the private engine port.")
    # Check the claimed worker's private base as well as its primary GPU. A
    # process that won the ephemeral bind race must never receive a render.
    try:
        argv = stats["system"]["argv"]
        base = argv[argv.index("--base-directory") + 1]
        served_port = argv[argv.index("--port") + 1]
        if Path(base) != layout.base_dir or str(port) != served_port:
            raise ValueError
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise _PortConflict(
            "Another process answered on the private engine port."
        ) from exc
    client.private = True
    client.check_primary_gpu(stats)
    return client


def _idle(client: AmdComfyClient, timeout: float = 3) -> bool:
    with httpx.Client(
        base_url=client.base_url,
        trust_env=False,
        follow_redirects=False,
        timeout=timeout,
    ) as http:
        queue = comfy_json(http.get("/queue"))
    return queue.get("queue_running") == [] and queue.get("queue_pending") == []


class ManagedComfyEngine:
    def __init__(self, layout: ManagedComfyLayout | None) -> None:
        self.layout = layout
        self._lock = threading.Condition(threading.RLock())
        self._phase: EngineState = "stopped"
        self._message = ""
        self._progress: float | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._release_job: Callable[[], None] | None = None
        self._client: AmdComfyClient | None = None
        self._port: int | None = None
        self._generation = 0
        self._leases = 0
        self._may_have_models = False
        self._external_cache: tuple[float, AmdComfyClient | None] | None = None

    @property
    def custom_nodes_dir(self) -> Path:
        if self.layout is None:
            raise ComfyError(
                "The private video engine is unavailable on this platform."
            )
        return self.layout.custom_nodes_dir

    @property
    def data_dir(self) -> Path:
        return self.custom_nodes_dir.parent

    def _installed(self) -> bool:
        if self.layout is None:
            return False
        from services.amd_comfy_provision import needs_install

        return not needs_install(self.layout)

    def _external(self) -> AmdComfyClient | None:
        now = time.monotonic()
        if self._external_cache and now - self._external_cache[0] < 2:
            return self._external_cache[1]
        try:
            assert self.layout is not None
            client = AmdComfyClient.discover((8000, 8188), backend=self.layout.backend)
        except ComfyError:
            client = None
        self._external_cache = now, client
        return client

    def _log_path(self) -> Path:
        assert self.layout is not None
        return self.layout.root / "worker.log"

    def _failed_worker(self, code: int) -> str:
        log = self._log_path()
        detail = ""
        try:
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 8192))
                for line in reversed(
                    stream.read().decode("utf-8", errors="replace").splitlines()
                ):
                    if "Error:" in line and "://" not in line:
                        detail = " " + line.strip()[:360]
                        break
        except OSError:
            pass
        return f"The private video engine exited with code {code}.{detail} See {log}"

    def _check_exit(self) -> None:
        if (
            self._phase == "ready"
            and self._process is not None
            and (code := self._process.poll()) is not None
        ):
            self._discard_process()
            self._set("error", self._failed_worker(code))

    def _set(
        self, phase: EngineState, message: str, progress: float | None = None
    ) -> None:
        self._phase, self._message, self._progress = phase, message, progress
        self._lock.notify_all()

    def status(self) -> ComfyEngineStatus:
        with self._lock:
            self._check_exit()
            if self.layout is None:
                return ComfyEngineStatus(
                    status="unsupported",
                    message="The private video engine requires a supported CUDA or ROCm installation on Windows.",
                )
            if self._phase in ("ready", "installing", "starting"):
                return ComfyEngineStatus(
                    status=self._phase,
                    message=self._message,
                    progress=self._progress,
                    canStop=self._phase == "ready" and self._leases == 0,
                )
            if external := self._external():
                return ComfyEngineStatus(
                    status="external",
                    message=f"Connected to ComfyUI Desktop on {external.gpu_name}.",
                    canPrepare=not self._installed(),
                    canStart=self._installed(),
                )
            if self._phase == "error":
                return ComfyEngineStatus(
                    status="error",
                    message=self._message,
                    canPrepare=True,
                    canStart=self._installed(),
                )
            if self._installed():
                return ComfyEngineStatus(
                    status="stopped",
                    message="The private video engine is installed and stopped.",
                    canStart=True,
                )
            return ComfyEngineStatus(
                status="not_installed",
                message="Set up the bundled video engine to generate inside LTX Desktop - AMD 395 MAX.",
                canPrepare=True,
            )

    def active_at(self, base_url: str) -> bool:
        with self._lock:
            self._check_exit()
            return self._client is not None and self._client.base_url == base_url

    @contextlib.contextmanager
    def rendering(self, client: AmdComfyClient) -> Generator[None, None, None]:
        # A local stop cannot race prompt submission or kill the worker while
        # the generated movie is still being copied out of the Comfy output.
        with self._lock:
            self._check_exit()
            if client.private and self._client is not client:
                raise ComfyError(
                    "The private engine was restarted. Retry this generation."
                )
            owned = self._client is client
            if owned:
                self._leases += 1
                self._may_have_models = True
        try:
            yield
        finally:
            if owned:
                with self._lock:
                    self._leases -= 1

    def release_models_for_native(self) -> None:
        """Ask our idle worker to free cached inference weights before a native render."""
        # A native job must not wait for an in-progress installation or external
        # discovery. /free is a hint: Comfy handles these flags on its own worker.
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._check_exit()
            client = self._client
            if client is None or self._leases or not self._may_have_models:
                return
            if _idle(client, timeout=0.35):
                with httpx.Client(
                    base_url=client.base_url,
                    timeout=0.35,
                    trust_env=False,
                    follow_redirects=False,
                ) as http:
                    http.post(
                        "/free", json={"unload_models": True, "free_memory": True}
                    ).raise_for_status()
                self._may_have_models = False
        except (OSError, httpx.HTTPError, ComfyError):
            pass
        finally:
            self._lock.release()

    def _begin(self, prepare: bool) -> None:
        if self.layout is None or self._phase in ("installing", "starting", "ready"):
            return
        installed = self._installed()
        if not prepare and not installed:
            return
        self._generation += 1
        ticket = self._generation
        if installed:
            self._set("starting", "Starting the private video engine.")
        else:
            self._set("installing", "Preparing the bundled ComfyUI files.")
        threading.Thread(
            target=self._launch, args=(ticket,), name="ltx-private-comfy", daemon=True
        ).start()

    def action(self, action: EngineAction) -> ComfyEngineStatus:
        with self._lock:
            self._check_exit()
            if action == "stop":
                return self._stop_idle()
            self._begin(prepare=action == "prepare")
            return self.status()

    def _launch(self, ticket: int) -> None:
        from services.amd_comfy_provision import (
            private_comfy_command,
            provision_private_comfy,
        )
        from services.amd_worker_job import bind_worker_to_parent

        assert self.layout is not None
        layout = self.layout

        def progress(message: str) -> None:
            with self._lock:
                if ticket != self._generation:
                    raise ComfyError("Engine setup was stopped.")
                self._set("installing", message)

        try:
            provision_private_comfy(layout, progress)
            for attempt in range(2):
                with self._lock:
                    if ticket != self._generation:
                        return
                    self._set(
                        "starting",
                        "Loading the private engine and checking its selected GPU.",
                    )
                    port = _allocate_port(self._port if attempt == 0 else None)
                    self._port = port
                    args, env, cwd = private_comfy_command(layout, port)
                    log = self._log_path()
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("wb") as output:
                        process = subprocess.Popen(
                            args,
                            cwd=cwd,
                            env=env,
                            stdin=subprocess.DEVNULL,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            close_fds=True,
                        )
                    self._process = process
                    self._release_job = bind_worker_to_parent(process)
                deadline = time.monotonic() + _START_TIMEOUT
                while time.monotonic() < deadline:
                    with self._lock:
                        if ticket != self._generation:
                            return
                        code = process.poll()
                        if code is not None:
                            # Windows returns EADDRINUSE for a listener claiming our port between reserve and launch.
                            with log.open("rb") as output:
                                output.seek(max(0, log.stat().st_size - 8192))
                                last = output.read().lower()
                            if attempt == 0 and (
                                b"already in use" in last or b"10048" in last
                            ):
                                self._discard_process()
                                break
                            raise ComfyError(self._failed_worker(code))
                    try:
                        client = _owned_client(layout, port, process.pid)
                    except _PortConflict:
                        if attempt:
                            raise
                        with self._lock:
                            if ticket != self._generation:
                                return
                            self._discard_process()
                        break
                    except httpx.TransportError:
                        time.sleep(0.35)
                        continue
                    with self._lock:
                        if ticket != self._generation:
                            return
                        if process.poll() is not None:
                            raise ComfyError(self._failed_worker(process.returncode))
                        self._client = client
                        self._set(
                            "ready",
                            f"The private video engine is ready on {client.gpu_name}.",
                            100,
                        )
                    return
                else:
                    raise ComfyError(
                        f"The private engine did not report its selected GPU within {_START_TIMEOUT} seconds. See {log}"
                    )
        except Exception as exc:
            with self._lock:
                if ticket == self._generation:
                    self._discard_process()
                    self._set("error", f"{exc}")

    def _discard_process(self) -> None:
        self._may_have_models = False
        self._client = None
        process, self._process = self._process, None
        close, self._release_job = self._release_job, None
        if close:
            close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=4)

    def _stop_idle(self) -> ComfyEngineStatus:
        if self._client is None:
            return self.status()
        try:
            if self._leases or not _idle(self._client):
                return self.status().model_copy(
                    update={
                        "canStop": False,
                        "message": "The private engine is processing a generation. Wait for its queue to finish.",
                    }
                )
        except (httpx.HTTPError, ComfyError) as exc:
            return self.status().model_copy(
                update={
                    "canStop": False,
                    "message": f"Could not confirm that the engine is idle. {exc}",
                }
            )
        self.shutdown()
        return self.status()

    def restart(self) -> ComfyEngineStatus:
        with self._lock:
            previous = self._client
            if previous is None:
                return self.status()
            result = self._stop_idle()
            if self._client is previous:
                return result
            self._begin(prepare=False)
            return self.status()

    def shutdown(self) -> None:
        with self._lock:
            self._generation += 1
            self._discard_process()
            self._set("stopped", "The private video engine has stopped.")

    def connect(
        self,
        *,
        start_prepared: bool = False,
        cancelled: Callable[[], bool] | None = None,
    ) -> AmdComfyClient:
        with self._lock:
            if self.layout is None:
                raise ComfyError("The private video engine requires a supported CUDA or ROCm installation on Windows.")
            self._check_exit()
            if self._client:
                return self._client
            if self._phase not in ("installing", "starting"):
                if external := self._external():
                    return external
                if start_prepared:
                    self._begin(prepare=False)
            if start_prepared:
                while self._phase in ("installing", "starting"):
                    if cancelled and cancelled():
                        raise ComfyCancelled(
                            "Local generation was cancelled while the engine was starting."
                        )
                    self._lock.wait(timeout=0.4)
            if self._client:
                return self._client
            if self._phase in ("error", "installing", "starting"):
                raise ComfyError(self._message)
            raise ComfyError(
                "In LTX Desktop - AMD 395 MAX, open Settings → Models to set up the private video engine, or open ComfyUI Desktop on the selected GPU."
            )


@cache
def managed_comfy() -> ManagedComfyEngine:
    data = os.environ.get("LTX_APP_DATA_DIR", "")
    backend = local_video_backend()
    if backend is None or not data or not Path(data).is_absolute():
        return ManagedComfyEngine(None)
    from services.amd_comfy_provision import ManagedComfyLayout

    resources = Path(__file__).resolve().parents[2]
    return ManagedComfyEngine(
        ManagedComfyLayout(Path(data) / "managed-comfy", resources, backend)
    )
