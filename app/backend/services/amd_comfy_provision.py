"""Provision the pinned ComfyUI core for the private GPU worker."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from filelock import FileLock

from runtime_config.community_runtime import LocalVideoBackend
from services.amd_h3_mps import prepare_h3_mps_core, register_h3_metal_node

CORE_VERSION = "0.38.0"
CORE_REVISION = "6b747c0428c343e1417219641db93a4fb7cb69ae"
ARCHIVE_NAME = "comfyui-v0.38.0.zip"
ARCHIVE_SHA256 = "4a144a553c77397c81f6067f3f8c8ab9dffd42889af48fb35fec2c64a71c75b2"
ARCHIVE_BYTES = 13_285_248
_INSTALL_MARKER = ".ltx-core-sha256"


@dataclass(frozen=True)
class ManagedComfyLayout:
    root: Path
    resources_dir: Path
    backend: LocalVideoBackend
    platform: str = "win32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())
        object.__setattr__(self, "resources_dir", self.resources_dir.resolve())

    @property
    def asset_dir(self) -> Path:
        return self.resources_dir / "comfy-engine"

    @property
    def extra_packages(self) -> Path:
        if self.platform in ("darwin", "linux"):
            return self.core_dir / ".ltx-site"
        return self.asset_dir / "site-packages"

    @property
    def python_exe(self) -> Path:
        if self.platform == "darwin":
            # The installer signs this interpreter for loading private native
            # wheels. Its home and libpython are shared with the main runtime.
            return self.resources_dir / "python" / "bin" / "ltx-comfy-python"
        if self.platform == "linux":
            return Path(sys.executable)
        return self.resources_dir / "python" / "python.exe"

    @property
    def core_dir(self) -> Path:
        return self.root / "core" / CORE_VERSION

    @property
    def main_py(self) -> Path:
        return self.core_dir / "main.py"

    @property
    def base_dir(self) -> Path:
        return self.root / "base"

    @property
    def custom_nodes_dir(self) -> Path:
        return self.base_dir / "custom_nodes"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def user_dir(self) -> Path:
        return self.base_dir / "user"

    @property
    def input_dir(self) -> Path:
        return self.base_dir / "input"

    @property
    def output_dir(self) -> Path:
        return self.base_dir / "output"

    @property
    def temp_dir(self) -> Path:
        return self.base_dir / "temp"


def needs_install(layout: ManagedComfyLayout) -> bool:
    """Cheap readiness probe. The marker is committed only with a complete extraction."""
    try:
        marker = (layout.core_dir / _INSTALL_MARKER).read_text(encoding="ascii")
        if layout.platform in ("darwin", "linux"):
            from services.comfy_posix_wheels import manifest_file
            receipt = json.loads((layout.extra_packages / "ltx-comfy-deps.json").read_bytes())
            digest = hashlib.sha256(manifest_file(layout.platform).read_bytes()).hexdigest()
            if receipt.get("manifest_sha256") != digest:
                return True
    except (FileNotFoundError, ValueError, AttributeError):
        return True
    return marker != ARCHIVE_SHA256 or not all(
        (layout.core_dir / entry).is_file() for entry in ("main.py", "server.py", "comfy/ops.py")
    )


def _extract_verified(archive: Path, stage: Path) -> None:
    with archive.open("rb") as source:
        if os.fstat(source.fileno()).st_size != ARCHIVE_BYTES or hashlib.file_digest(source, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("The bundled ComfyUI archive failed verification. Reinstall the SpaceSpeak edition of LTX Desktop.")
        source.seek(0)
        prefix = f"ComfyUI-{CORE_REVISION}"
        with zipfile.ZipFile(source) as bundle:
            for entry in bundle.infolist():
                parts = PurePosixPath(entry.filename).parts
                if (not parts or parts[0] != prefix or ".." in parts or "\\" in entry.filename
                        or ":" in entry.filename or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ValueError("Unexpected path in the bundled ComfyUI archive.")
                target = stage.joinpath(*parts[1:])
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(entry) as data, target.open("xb") as out:
                        shutil.copyfileobj(data, out)
    (stage / _INSTALL_MARKER).write_text(ARCHIVE_SHA256, encoding="ascii")


def provision_private_comfy(
    layout: ManagedComfyLayout, progress: Callable[[str], None] | None = None,
) -> ManagedComfyLayout:
    """Install from the local release asset, preserving the separate data/model tree."""
    for directory in (layout.models_dir, layout.custom_nodes_dir, layout.input_dir, layout.output_dir,
                      layout.user_dir, layout.temp_dir, layout.root / "cache" / "tmp"):
        directory.mkdir(parents=True, exist_ok=True)
    metal = layout.platform == "darwin" and layout.backend == "mps"
    if not metal and not needs_install(layout):
        return layout
    with FileLock(str(layout.root / "provision.lock"), timeout=120):
        # The preceding release has the same core and wheel receipt. Add our
        # Mac registration atomically before the next start, reusing that installation.
        if not needs_install(layout) and (not metal or register_h3_metal_node(layout.core_dir)):
            return layout
        archive = layout.asset_dir / ARCHIVE_NAME
        if not archive.is_file():
            raise FileNotFoundError("The bundled ComfyUI archive is missing. Reinstall the SpaceSpeak edition of LTX Desktop.")
        if progress:
            progress(f"Verifying and unpacking ComfyUI {CORE_VERSION}")
        layout.core_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".setup-", dir=layout.core_dir.parent) as temporary:
            stage = Path(temporary) / "new"
            stage.mkdir()
            _extract_verified(archive, stage)
            if layout.platform in ("darwin", "linux"):
                from services.comfy_posix_wheels import manifest_file, provision_wheels
                if metal:
                    prepare_h3_mps_core(stage)
                    if not register_h3_metal_node(stage):
                        raise ValueError("Unexpected built-in node loader in the bundled ComfyUI archive.")
                provision_wheels(manifest_file(layout.platform), layout.root / "cache" / "wheels",
                                 stage / ".ltx-site", progress)
            backup = Path(temporary) / "old"
            if layout.core_dir.exists():
                layout.core_dir.rename(backup)
            try:
                stage.rename(layout.core_dir)
            except OSError:
                if backup.exists():
                    backup.rename(layout.core_dir)
                raise
    return layout


def _worker_env(layout: ManagedComfyLayout) -> dict[str, str]:
    # Keep the OS, driver and device selection while isolating the child's caches
    # and keeping the parent application's tokens out of the inference process.
    allowed = {
        "ALLUSERSPROFILE", "APPDATA", "COMSPEC", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "OS",
        "PATHEXT", "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS", "PROGRAMDATA", "PROGRAMFILES",
        "PROGRAMFILES(X86)", "PROGRAMW6432", "SYSTEMDRIVE", "SYSTEMROOT", "USERPROFILE", "WINDIR",
        "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
        "HOME", "LANG", "LC_ALL",
    }
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    if os.name == "nt":
        system = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        env["PATH"] = os.pathsep.join(map(str, (layout.python_exe.parent, system / "System32", system)))
    else:
        env["PATH"] = os.pathsep.join((str(layout.python_exe.parent), os.defpath))
    if layout.platform in ("darwin", "linux"):
        env["LTX_COMFY_GPU_PROBE"] = layout.backend
        env["LTX_COMFY_PARENT_PID"] = str(os.getpid())
        # Use the same policy as the native LTX process on Metal. Keep caches
        # and bytecode out of the app bundle during imports.
        if layout.backend == "mps":
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            if layout.platform == "darwin":
                env["LTX_COMFY_H3_METAL_NODE"] = str(Path(__file__).resolve().with_name("h3_metal"))
                if os.environ.get("LTX_H3_M5_INT8") == "0":
                    env["LTX_H3_M5_INT8"] = "0"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    cache = layout.root / "cache"
    env.update({
        "TEMP": str(cache / "tmp"), "TMP": str(cache / "tmp"), "TMPDIR": str(cache / "tmp"),
        "HF_HOME": str(cache / "huggingface"), "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TORCH_HOME": str(cache / "torch"), "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "NUMBA_CACHE_DIR": str(cache / "numba"), "MPLCONFIGDIR": str(cache / "matplotlib"),
        "XDG_CACHE_HOME": str(cache), "DO_NOT_TRACK": "1",
    })
    return env


def private_comfy_command(layout: ManagedComfyLayout, port: int) -> tuple[list[str], dict[str, str], Path]:
    """Return argv, environment and cwd for an owned, loopback-only worker process."""
    if not 0 < port < 65536:
        raise ValueError("Invalid private ComfyUI port.")
    if needs_install(layout):
        raise RuntimeError("Set up the private ComfyUI engine before starting it.")
    if not layout.python_exe.is_file() or not (layout.extra_packages / "ltx-comfy-deps.json").is_file():
        raise FileNotFoundError("The private GPU Python runtime or ComfyUI dependencies are missing. Reinstall the SpaceSpeak edition of LTX Desktop.")
    wrapper = Path(__file__).resolve().with_name("amd_comfy_bootstrap.py")
    # --base-directory directs ComfyUI's models, custom_nodes, temp, input,
    # output, user database and preferences to the private data tree.
    argv = [
        str(layout.python_exe), "-I", "-B", "-u", str(wrapper), str(layout.core_dir), str(layout.extra_packages),
        "--listen", "127.0.0.1", "--port", str(port), "--base-directory", str(layout.base_dir),
        "--user-directory", str(layout.user_dir), "--disable-auto-launch", "--disable-api-nodes",
        "--disable-all-custom-nodes", "--whitelist-custom-nodes", "kandinsky6",
    ]
    if layout.backend == "rocm":
        # Avoid the Windows gfx1151 VBAR allocation crash during latent upsampling.
        argv.append("--disable-dynamic-vram")
    elif layout.platform == "win32" and not re.fullmatch(r"GPU-[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}",
                          os.environ.get("CUDA_VISIBLE_DEVICES", ""), flags=re.I):
        raise RuntimeError("The selected NVIDIA device UUID is missing. Rerun the Windows installer.")
    return argv, _worker_env(layout), layout.core_dir
