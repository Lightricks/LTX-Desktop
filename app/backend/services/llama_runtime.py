"""Pinned, app-owned llama.cpp installation and short-lived local inference."""

import asyncio
import hashlib
import logging
import os
import platform
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock

import httpx

logger = logging.getLogger(__name__)


class AssistantModelError(Exception):
    pass


VERSION = "b11136"


@dataclass(frozen=True)
class RuntimeAsset:
    filename: str
    size: int
    sha256: str


ASSETS = {
    ("Darwin", "arm64"): RuntimeAsset(
        f"llama-{VERSION}-bin-macos-arm64.tar.gz", 11_200_431,
        "5b0f317e437ea00c726c816903ff3dab9e29bb0476bff42a752831c9a7e32376",
    ),
    ("Windows", "AMD64"): RuntimeAsset(
        f"llama-{VERSION}-bin-win-vulkan-x64.zip", 32_126_967,
        "1b69e01b6d3041f881a94a0091a95dcf393a23b9225084e76b5dfe885529fbb2",
    ),
    ("Linux", "x86_64"): RuntimeAsset(
        f"llama-{VERSION}-bin-ubuntu-vulkan-x64.tar.gz", 30_600_042,
        "d091350ad67149df49113dd9b89d12dd8e07059f1a287c9791712c0cd0f58040",
    ),
}


@dataclass(frozen=True)
class LocalEndpoint:
    base_url: str
    api_key: str


def extract_runtime(archive: Path, destination: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as file:
            for member in file.infolist():
                if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                    raise AssistantModelError("Invalid runtime archive path.")
            file.extractall(destination)
    else:
        with tarfile.open(archive) as file:
            file.extractall(destination, filter="data")


class LlamaRuntime:
    def __init__(self, app_data_dir: Path, *, system: str | None = None, machine: str | None = None) -> None:
        self._platform = (system or platform.system(), machine or platform.machine())
        self.directory = app_data_dir / "runtimes" / f"llama-{VERSION}-{self._platform[0]}-{self._platform[1]}"
        # The backend_ prefix puts this file under Electron's log cleanup.
        self.log_path = app_data_dir / "logs" / f"backend_llama_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
        self._install_lock = Lock()
        self._inference_lock = Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def check_platform(self) -> None:
        if self._platform not in ASSETS:
            raise AssistantModelError("Local assistant supports Apple Silicon Macs, Windows x64 and Linux x64.")

    @property
    def download_size(self) -> int:
        self.check_platform()
        return ASSETS[self._platform].size

    def _executable(self, root: Path) -> Path | None:
        name = "llama-server.exe" if self._platform[0] == "Windows" else "llama-server"
        return next(root.rglob(name), None) if root.is_dir() else None

    def installed(self) -> bool:
        try:
            return (self.directory / ".complete").read_text() == ASSETS[self._platform].sha256 and self._executable(self.directory) is not None
        except (OSError, KeyError):
            return False

    def install(self, on_progress: Callable[[int], None]) -> None:
        self.check_platform()
        with self._install_lock:
            if self.installed():
                return
            asset = ASSETS[self._platform]
            self.directory.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self.directory.parent) as temporary:
                root = Path(temporary)
                archive = root / asset.filename
                digest = hashlib.sha256()
                downloaded = 0
                url = f"https://github.com/ggml-org/llama.cpp/releases/download/{VERSION}/{asset.filename}"
                with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
                    response.raise_for_status()
                    with archive.open("wb") as file:
                        for chunk in response.iter_bytes():
                            file.write(chunk)
                            digest.update(chunk)
                            downloaded += len(chunk)
                            if downloaded > asset.size:
                                raise AssistantModelError("Runtime download verification failed.")
                            on_progress(downloaded)
                if downloaded != asset.size or digest.hexdigest() != asset.sha256:
                    raise AssistantModelError("Runtime download verification failed.")
                extracted = root / "extracted"
                extracted.mkdir()
                extract_runtime(archive, extracted)
                executable = self._executable(extracted)
                if executable is None:
                    raise AssistantModelError("Runtime archive is missing its executable.")
                if self._platform[0] != "Windows":
                    executable.chmod(executable.stat().st_mode | 0o111)
                (extracted / ".complete").write_text(asset.sha256)
                if self.directory.exists():
                    shutil.rmtree(self.directory)
                extracted.replace(self.directory)

    @asynccontextmanager
    async def serve(self, model: str, path: Path, *, projector: Path | None = None) -> AsyncIterator[LocalEndpoint]:
        if not self._inference_lock.acquire(blocking=False):
            raise AssistantModelError("The local assistant is busy. Retry when the current request finishes.")
        process: asyncio.subprocess.Process | None = None
        try:
            executable = self._executable(self.directory)
            if not self.installed() or executable is None or not path.is_file() or (projector is not None and not projector.is_file()):
                raise AssistantModelError("Download the local assistant model in Settings > Models.")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            endpoint = LocalEndpoint(f"http://127.0.0.1:{port}/v1", secrets.token_urlsafe(32))
            # Ignore inherited llama settings and cloud credentials; inference is loopback-only.
            environment = {key: value for key, value in os.environ.items() if not key.startswith(("LLAMA_", "OPENAI_", "GEMINI_", "GOOGLE_API_"))}
            environment["LLAMA_API_KEY"] = endpoint.api_key
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("ab") as log:
                log.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} {model}\n".encode())
                log.flush()
                process = await asyncio.create_subprocess_exec(
                    sys.executable, str(Path(__file__).with_name("llama_worker.py")),
                    str(executable), "--model", str(path), "--alias", model,
                    "--host", "127.0.0.1", "--port", str(port), "--ctx-size", "16384",
                    "--parallel", "1", "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}',
                    "--n-gpu-layers", "auto", "--no-webui",
                    *(["--mmproj", str(projector)] if projector is not None else []),
                    stdin=asyncio.subprocess.PIPE, stdout=log, stderr=log, env=environment,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    # Its own group, so cleanup reaches llama-server even if the worker was killed.
                    start_new_session=sys.platform != "win32",
                )
                self._process = process
                self._loop = asyncio.get_running_loop()
                started = asyncio.get_running_loop().time()
                logger.info("Loading local assistant %s", model)
                health = httpx.Request(
                    "GET", f"http://127.0.0.1:{port}/health",
                    headers={"Authorization": f"Bearer {endpoint.api_key}"}, extensions={"timeout": httpx.Timeout(2).as_dict()},
                )
                # The transport, not a client: a client logs every poll, and llama-server answers 503 until the model loads.
                async with httpx.AsyncHTTPTransport() as transport:
                    for _ in range(480):
                        if process.returncode is not None:
                            raise AssistantModelError("Local assistant could not start. Check available memory and graphics drivers, then retry.")
                        try:
                            response = await transport.handle_async_request(health)
                            await response.aclose()
                            if response.status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(0.25)
                    else:
                        raise AssistantModelError("Local assistant took too long to load. Try the smaller model.")
                logger.info("Local assistant %s ready in %.1fs", model, asyncio.get_running_loop().time() - started)
                yield endpoint
        finally:
            try:
                if process is not None and process.returncode is None:
                    try:
                        if process.stdin is not None:
                            process.stdin.close()
                        await asyncio.wait_for(process.wait(), timeout=7)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
                    except ProcessLookupError:
                        pass
            finally:
                if process is not None and sys.platform != "win32":
                    # A worker that died abnormally leaves llama-server running in its group.
                    with suppress(ProcessLookupError, PermissionError):
                        os.killpg(process.pid, signal.SIGKILL)
                # A cancel during the waits above must not leave the next turn "busy".
                self._process = None
                self._loop = None
                self._inference_lock.release()

    def stop(self) -> None:
        process = self._process
        loop = self._loop
        if process is not None and process.returncode is None and process.stdin is not None and loop is not None:
            try:
                loop.call_soon_threadsafe(process.stdin.close)
            except RuntimeError:
                pass
