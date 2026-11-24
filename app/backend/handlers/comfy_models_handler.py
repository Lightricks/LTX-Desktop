"""Set up the LTX 2.5 graph in the running Comfy installation's own model folders."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Literal

import httpx
from filelock import FileLock
from pydantic import BaseModel, Field

from services.amd_comfy_models import (
    KANDINSKY_MODELS,
    LITE_MODEL,
    REQUIRED_MODELS,
    ComfyWeight,
)
from services.model_downloader import ModelDownloader
from services.task_runner import TaskRunner


class ComfyInstalledFile(BaseModel):
    filename: str
    category: str
    path: str
    installed: bool
    sizeBytes: int


DownloadStatus = Literal["offline", "missing", "ready", "downloading", "error"]


class ComfyBundleStatus(BaseModel):
    status: DownloadStatus
    message: str
    models: list[ComfyInstalledFile] = Field(default_factory=list[ComfyInstalledFile])
    currentFile: str | None = None
    completedFiles: int = 0
    totalFiles: int = len(REQUIRED_MODELS)
    completedBytes: int = 0
    totalBytes: int = sum(weight.size for weight in REQUIRED_MODELS)


class ComfyModelsStatus(ComfyBundleStatus):
    kandinskyStatus: DownloadStatus = "offline"
    kandinskyNodesInstalled: bool = False
    kandinskyWeightsReady: bool = False
    kandinskyReady: bool = False
    kandinskyMessage: str = (
        "Use the model setup panel to check the local video engine and Kandinsky files."
    )
    kandinskyModels: list[ComfyInstalledFile] = Field(
        default_factory=list[ComfyInstalledFile]
    )
    kandinskyCurrentFile: str | None = None
    kandinskyCompletedFiles: int = 0
    kandinskyTotalFiles: int = len(KANDINSKY_MODELS)
    kandinskyCompletedBytes: int = 0
    kandinskyTotalBytes: int = sum(weight.size for weight in KANDINSKY_MODELS)


_LICENSE_HELP = (
    "Open https://huggingface.co/Lightricks/LTX-2.5 and accept the model license. "
    "In LTX Desktop - AMD 395 MAX, open Settings → Models and sign in to the same Hugging Face account, then retry."
)


@dataclass(frozen=True)
class ComfyFolderRegistry:
    folders: dict[str, list[str]]
    server: str
    check_nodes: Callable[[], bool]


def running_comfy_folders() -> ComfyFolderRegistry:
    from services.amd_comfy_client import AmdComfyClient
    from services.amd_kandinsky_node import kandinsky_nodes_loaded

    worker = AmdComfyClient.discover()
    return ComfyFolderRegistry(
        worker.get_folder_paths(),
        worker.base_url,
        lambda: kandinsky_nodes_loaded(worker),
    )


_AUDIO_FOLDER = "kandinsky6_audio_vae"
_NODE_HELP = "In LTX Desktop - AMD 395 MAX, open Settings → Models → Kandinsky setup to install the nodes and restart the local engine."


def _audio_roots(folders: dict[str, list[str]]) -> list[str]:
    # The official Kandinsky extension registers Comfy's base models/audio_vae
    # when it loads. Before installation, derive that location from a built-in
    # category. Comfy puts the original VAE root after is_default overrides.
    registered = folders.get(_AUDIO_FOLDER)
    if registered:
        return registered
    # Comfy core registers both legacy alias pairs under the same base root.
    # Match this combination first so an unrelated large extra-models VAE root
    # does not redirect the extension's required hard-coded audio directory.
    diffusion = [Path(p) for p in folders.get("diffusion_models", [])]
    encoders = [Path(p) for p in folders.get("text_encoders", [])]
    for path in diffusion:
        base = path.parent
        if (
            path.is_absolute()
            and path.name.casefold() == "unet"
            and base / "diffusion_models" in diffusion
            and base / "clip" in encoders
            and base / "text_encoders" in encoders
        ):
            return [str(base / "audio_vae")]
    for category in ("configs", "vae", "checkpoints"):
        for directory in reversed(folders.get(category, [])):
            path = Path(directory)
            if path.is_absolute() and path.name.casefold() == category:
                return [str(path.parent / "audio_vae")]
    raise ValueError(
        "ComfyUI did not report its base model directory for Kandinsky audio. Restart the local video engine and retry."
    )


def _writable_parent(directory: Path) -> bool:
    existing = directory
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return existing.is_dir() and os.access(existing, os.W_OK)


def _existing_model(directories: Sequence[Path], filename: str) -> Path | None:
    # Comfy selects an exact, flat model name first. A truncated flat file must
    # not be silently hidden behind a valid nested copy that the graph won't load.
    selected = next(
        (path for directory in directories if (path := directory / filename).is_file()),
        None,
    )
    if selected is None:
        choices: dict[str, Path] = {}
        for directory in directories:
            for candidate in directory.glob(f"**/{filename}"):
                if (
                    candidate.is_file()
                    and ".ltx-comfy-downloads"
                    not in candidate.relative_to(directory).parts
                ):
                    choices.setdefault(
                        candidate.relative_to(directory).as_posix(), candidate
                    )
        if choices:
            key = min(choices, key=lambda name: (name.count("/"), name))
            selected = choices[key]
    return selected


def locate_comfy_weight(
    weight: ComfyWeight, folders: dict[str, list[str]]
) -> ComfyInstalledFile:
    roots = (
        _audio_roots(folders)
        if weight.category == "audio_vae"
        else folders.get(weight.category, [])
    )
    directories = [Path(p) for p in roots if Path(p).is_absolute()]
    if not directories:
        raise ValueError(
            f"ComfyUI did not report a model folder for {weight.category}."
        )
    if weight.category == "audio_vae":
        # The node only accepts top-level TOD weights. Its vocoder opens an exact
        # directory under the root the extension registered, never a recursive match.
        if weight.filename.startswith("bigvgan_vocoder/"):
            directories = directories[-1:]
        selected = next(
            (
                d / weight.filename
                for d in directories
                if (d / weight.filename).is_file()
            ),
            None,
        )
    else:
        selected = _existing_model(directories, weight.filename)
    if selected is None:
        # Stock Comfy registers the historical unet alias before diffusion_models.
        # Prefer its canonical sibling on the same root when creating a new file.
        if (
            weight.category == "diffusion_models"
            and directories[0].name.casefold() == "unet"
        ):
            canonical = directories[0].parent / "diffusion_models"
            if canonical in directories:
                directories = [canonical, *directories]
        folder = next((p for p in directories if _writable_parent(p)), directories[0])
        selected = folder / weight.filename
    return ComfyInstalledFile(
        filename=weight.filename,
        category=weight.category,
        path=str(selected),
        installed=_reusable(selected, weight),
        sizeBytes=weight.size,
    )


# Hashes of the sorted (tensor name, shape) pairs in each official safetensors
# header. These allow equivalent dtype/metadata serializations while rejecting
# Pro, partial exports, and unrelated files sharing a name. No tensors are read.
_KANDINSKY_LAYOUTS = {
    LITE_MODEL: "7d919c0252618fa12f410b3367d626e7629779a3b2de92a0f147bb60e6f7b4cc",
    "qwen_2.5_vl_7b.safetensors": "2d9dc935fb5ef274e4b1d66c40f9fac99fecfb570986fc113e7b889fb83bccb3",
    "clip_l.safetensors": "c33b3486637029cd671641815c8a53347aa06e02054f95018dd447fa434503a5",
    "hunyuan_video_vae_bf16.safetensors": "cb5aa1a8515442f6a9371face2e729682f5d28273d9f1334d700e35d928f4b31",
}
_BIGVGAN_CONFIG = "bba8eba26ab9c2c77b6749ed7b6aa4195937f371984cc7c63b79fcdfa7c9a09d"


def _json_digest(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


@lru_cache(maxsize=64)
def _alternate_file(path: Path, filename: str, size: int, modified: int) -> bool:
    # Size and mtime are both in the cache key to handle manual file replacement.
    if filename == "bigvgan_vocoder/config.json" and 0 < size <= 1024 * 1024:
        try:
            return _json_digest(json.loads(path.read_bytes())) == _BIGVGAN_CONFIG
        except (ValueError, UnicodeError):
            return False
    expected_layout = _KANDINSKY_LAYOUTS.get(filename)
    if expected_layout is None:
        return False
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(str(path), framework="np") as tensors:
            layout = [
                (key, tensors.get_slice(key).get_shape())
                for key in sorted(tensors.keys())
            ]
            return _json_digest(layout) == expected_layout
    except (OSError, SafetensorError, ValueError):
        return False


@lru_cache(maxsize=64)
def _sample_digest(path: Path, size: int, modified: int) -> str:
    # Fast identification for manually installed checkpoints. Some have identical
    # lengths and headers yet different tensors. Recheck if the file changes.
    block = min(256 * 1024, size // 2)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for offset in (0, size // 2, size - block):
            stream.seek(offset)
            digest.update(stream.read(block))
    return digest.hexdigest()


def _reusable(path: Path, weight: ComfyWeight) -> bool:
    if not path.is_file():
        return False
    stat = path.stat()
    if weight.sample_sha256 is not None:
        return (
            stat.st_size == weight.size
            and _sample_digest(path, stat.st_size, stat.st_mtime_ns)
            == weight.sample_sha256
        )
    if stat.st_size == weight.size and (
        weight.filename != "bigvgan_vocoder/config.json" or _verified(path, weight)
    ):
        return True
    return _alternate_file(path, weight.filename, stat.st_size, stat.st_mtime_ns)


def _verified(path: Path, weight: ComfyWeight) -> bool:
    if not path.is_file() or path.stat().st_size != weight.size:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == weight.sha256


def _publish(source: Path, target: Path, weight: ComfyWeight) -> None:
    # Both paths are on the destination volume. Publish with no replacement, including
    # when another process finishes downloading after this job's initial scan.
    try:
        if os.name == "nt":
            source.rename(target)
        else:
            os.link(source, target)
            source.unlink()
    except FileExistsError:
        if not _verified(target, weight):
            raise ValueError(
                f"A different file already exists at {target}. Move it aside and retry."
            ) from None
        source.unlink(missing_ok=True)


class ComfyBundleInstaller:
    """One independently observable, retryable download job per recipe."""

    def __init__(
        self,
        label: str,
        weights: Sequence[ComfyWeight],
        token: Callable[[], str | None],
        native_models: Callable[[], Path] | None,
        downloader: ModelDownloader,
        tasks: TaskRunner,
        invalidate: Callable[[], None],
        *,
        shared_files: bool = False,
    ) -> None:
        self.label = label
        self.weights = weights
        self._token = token
        self._native_models = native_models
        self._downloader = downloader
        self._tasks = tasks
        self._invalidate = invalidate
        self._shared_files = shared_files
        self._lock = Lock()
        self._progress: ComfyBundleStatus | None = None
        self._failure: str | None = None

    def inspect(
        self,
        folders: dict[str, list[str]],
        *,
        locate: Callable[[ComfyWeight], ComfyInstalledFile] | None = None,
    ) -> ComfyBundleStatus:
        try:
            files = [
                locate(w) if locate is not None else locate_comfy_weight(w, folders)
                for w in self.weights
            ]
        except (ValueError, OSError) as exc:
            return self.offline(str(exc), status="error")
        have = [f for f in files if f.installed]
        ready = len(have) == len(files)
        return ComfyBundleStatus(
            status="ready" if ready else "missing",
            message=f"{self.label} weights are ready."
            if ready
            else f"Download the missing {self.label} weights.",
            models=files,
            completedFiles=len(have),
            totalFiles=len(files),
            completedBytes=sum(f.sizeBytes for f in have),
            totalBytes=sum(f.sizeBytes for f in files),
        )

    def offline(
        self, message: str, *, status: DownloadStatus = "offline"
    ) -> ComfyBundleStatus:
        return ComfyBundleStatus(
            status=status,
            message=message,
            totalFiles=len(self.weights),
            totalBytes=sum(w.size for w in self.weights),
        )

    def status(self, observed: ComfyBundleStatus) -> ComfyBundleStatus:
        with self._lock:
            if self._progress is not None:
                return self._progress.model_copy(deep=True)
            if self._failure and observed.status == "missing":
                return observed.model_copy(
                    update={"status": "error", "message": self._failure}
                )
        return observed

    def start(self, observed: ComfyBundleStatus) -> None:
        if observed.status != "missing":
            return
        state = observed.model_copy(deep=True)
        with self._lock:
            if self._progress is not None:
                return
            self._failure = None
            state.status = "downloading"
            state.message = f"Preparing {self.label} files."
            self._progress = state
        try:
            self._tasks.run_background(
                lambda: self._install(state),
                task_name=f"{self.label} model setup",
                on_error=self._failed,
            )
        except Exception as exc:
            self._failed(exc)

    def _failed(self, exc: Exception) -> None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        message = str(exc)
        if status in (401, 403):
            message = (
                _LICENSE_HELP
                if self._native_models is not None
                else "Hugging Face denied access. Check your connection or sign in again under Settings → Models in LTX Desktop - AMD 395 MAX, then retry."
            )
        # A signed CDN URL or an HTTP header can appear inside third-party error messages.
        if "://" in message and message != _LICENSE_HELP:
            message = f"Hugging Face download failed ({type(exc).__name__}). Check the network and retry."
        with self._lock:
            self._failure = message or "ComfyUI model setup failed. Retry to continue."
            self._progress = None
        self._invalidate()

    def _update(self, name: str, done_bytes: int, message: str) -> None:
        with self._lock:
            if self._progress:
                self._progress.currentFile = name
                self._progress.completedBytes = done_bytes
                self._progress.message = message

    def _install(self, state: ComfyBundleStatus) -> None:
        done = state.completedBytes
        for weight, entry in zip(self.weights, state.models, strict=True):
            if entry.installed:
                continue
            target = Path(entry.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Recipes can share large files and their Hugging Face resume cache.
            # Serialize publication of each shared file across threads and app instances.
            if self._shared_files:
                locks = target.parent / ".ltx-comfy-downloads" / "locks"
                locks.mkdir(parents=True, exist_ok=True)
                self._update(weight.filename, done, f"Preparing {weight.filename}")
                guard = FileLock(str(locks / (weight.sha256 + ".lock")))
            else:
                guard = nullcontext()
            with guard:
                if target.exists():
                    if not _verified(target, weight):
                        raise ValueError(
                            f"A different or incomplete file exists at {target}. Move it aside and retry."
                        )
                else:
                    self._install_one(weight, target, done)
            done += weight.size
            with self._lock:
                entry.installed = True
                state.completedFiles += 1
                state.completedBytes = done
        with self._lock:
            self._progress = None
            self._failure = None
        self._invalidate()

    def _install_one(self, weight: ComfyWeight, target: Path, done: int) -> None:
        name = weight.filename
        self._update(name, done, f"Preparing {name}")
        category_dir = target.parents[len(Path(name).parts) - 1]
        stage_root = category_dir.parent
        if (
            not os.access(stage_root, os.W_OK)
            or stage_root.stat().st_dev != category_dir.stat().st_dev
        ):
            stage_root = category_dir
        stage = stage_root / ".ltx-comfy-downloads" / weight.category / weight.revision
        stage.mkdir(parents=True, exist_ok=True)
        # Desktop's BF16 VAEs and upsampler are identical to Comfy's files. Reuse them
        # even when the user has never authenticated in this installation.
        native: Path | None = None
        if self._native_models is not None:
            native_root = self._native_models()
            native = next(
                (
                    candidate
                    for candidate in (
                        native_root / "ltx-2.5" / name,
                        native_root / name,
                    )
                    if _verified(candidate, weight)
                ),
                None,
            )
        if native is not None:
            try:
                os.link(native, target)
                return
            except FileExistsError:
                if _verified(target, weight):
                    return
                raise
            except OSError:
                # Different volume or a filesystem without hard links.
                staged_copy = stage / (name + ".copying")
                shutil.copyfile(native, staged_copy)
                _publish(staged_copy, target, weight)
                return
        token = self._token()
        if not token and self._native_models is not None:
            raise ValueError(_LICENSE_HELP)
        self._update(name, done, f"Downloading {name}")
        source = self._downloader.download_file(
            repo_id=weight.repo,
            filename=weight.remote_path or f"{weight.category}/{name}",
            local_dir=str(stage),
            token=token,
            revision=weight.revision,
            on_progress=lambda count: self._update(
                name, done + min(count, weight.size), f"Downloading {name}"
            ),
        )
        self._update(name, done + weight.size, f"Verifying {name}")
        if not _verified(source, weight):
            source.unlink(missing_ok=True)
            raise ValueError(
                f"The SHA-256 check failed for {name}. Retry to download it again."
            )
        _publish(source, target, weight)


class ComfyModelsHandler:
    def __init__(
        self,
        folders: Callable[[], ComfyFolderRegistry],
        token: Callable[[], str | None],
        native_models: Callable[[], Path],
        downloader: ModelDownloader,
        task_runner: TaskRunner,
        weights: Sequence[ComfyWeight] = REQUIRED_MODELS,
        kandinsky_weights: Sequence[ComfyWeight] = KANDINSKY_MODELS,
    ) -> None:
        self._folders = folders
        self._scan_lock = Lock()
        self._node_cache: tuple[str, float, bool] | None = None
        self._cached: (
            tuple[float, ComfyBundleStatus, ComfyBundleStatus, bool] | None
        ) = None
        self._ltx = ComfyBundleInstaller(
            "LTX 2.5 Comfy",
            weights,
            token,
            native_models,
            downloader,
            task_runner,
            self._invalidate,
        )
        self._kandinsky = ComfyBundleInstaller(
            "Kandinsky Lite",
            kandinsky_weights,
            token,
            None,
            downloader,
            task_runner,
            self._invalidate,
        )

    def _invalidate(self) -> None:
        with self._scan_lock:
            self._cached = None

    def _nodes_loaded(self, registry: ComfyFolderRegistry, fresh: bool) -> bool:
        if not registry.folders.get(_AUDIO_FOLDER):
            self._node_cache = None
            return False
        now = time.monotonic()
        cached = self._node_cache
        if (
            not fresh
            and cached
            and cached[0] == registry.server
            and now - cached[1] < 12
        ):
            return cached[2]
        try:
            loaded = registry.check_nodes()
        except (httpx.HTTPError, RuntimeError, OSError, ValueError):
            # Weight acquisition remains available when a node import failed.
            loaded = False
        self._node_cache = registry.server, now, loaded
        return loaded

    def _scan(self, fresh: bool) -> tuple[ComfyBundleStatus, ComfyBundleStatus, bool]:
        with self._scan_lock:
            if not fresh and self._cached and time.monotonic() - self._cached[0] < 2:
                return self._cached[1], self._cached[2], self._cached[3]
            try:
                registry = self._folders()
                folders = registry.folders
                ltx, kandinsky = (
                    self._ltx.inspect(folders),
                    self._kandinsky.inspect(folders),
                )
                nodes = self._nodes_loaded(registry, fresh)
            except (httpx.HTTPError, RuntimeError, OSError, ValueError) as exc:
                self._node_cache = None
                message = f"The local video engine is unavailable. {exc}"
                ltx, kandinsky, nodes = (
                    self._ltx.offline(message),
                    self._kandinsky.offline(message),
                    False,
                )
            self._cached = time.monotonic(), ltx, kandinsky, nodes
            return ltx, kandinsky, nodes

    def status(self, fresh: bool = False) -> ComfyModelsStatus:
        ltx_files, kandinsky_files, nodes = self._scan(fresh)
        ltx = self._ltx.status(ltx_files)
        k = self._kandinsky.status(kandinsky_files)
        weights_ready = k.status == "ready"
        ready = nodes and weights_ready
        status, message = k.status, k.message
        if not nodes and status in ("missing", "ready"):
            status, message = "missing", f"{message} {_NODE_HELP}"
        return ComfyModelsStatus(
            **ltx.model_dump(),
            kandinskyStatus=status,
            kandinskyNodesInstalled=nodes,
            kandinskyReady=ready,
            kandinskyWeightsReady=weights_ready,
            kandinskyMessage=message,
            kandinskyModels=k.models,
            kandinskyCurrentFile=k.currentFile,
            kandinskyCompletedFiles=k.completedFiles,
            kandinskyTotalFiles=k.totalFiles,
            kandinskyCompletedBytes=k.completedBytes,
            kandinskyTotalBytes=k.totalBytes,
        )

    def start(self, *, kandinsky: bool = False) -> ComfyModelsStatus:
        ltx_files, kandinsky_files, _ = self._scan(fresh=True)
        if kandinsky:
            self._kandinsky.start(kandinsky_files)
        else:
            self._ltx.start(ltx_files)
        return self.status()
