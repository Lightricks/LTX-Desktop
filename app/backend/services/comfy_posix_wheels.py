"""Install the pinned worker-only wheels next to the verified ComfyUI core.

The parent desktop supplies PyTorch and its backend (Metal or CUDA). These
additional wheels stay inside the writable private engine, outside the signed
Mac app and the system-owned Linux package.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import ssl
import stat
import sys
import tempfile
from collections.abc import Callable
from typing import TypedDict
import zipfile

import urllib.request

import certifi
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


class Wheel(TypedDict):
    filename: str
    url: str
    size: int
    sha256: str


def manifest_file(platform: str) -> Path:
    target = {"darwin": "macos-arm64", "linux": "linux-x86_64"}[platform]
    return Path(__file__).with_name(f"comfy-wheels-{target}.json")


def _verified(path: Path, record: Wheel) -> bool:
    if not path.is_file() or path.stat().st_size != record["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]


def _download(tls: ssl.SSLContext, cache: Path, record: Wheel) -> Path:
    filename = record["filename"]
    path = cache / filename
    if _verified(path, record):
        return path
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        partial = Path(directory) / "download"
        with urllib.request.urlopen(record["url"], context=tls, timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, 1024 * 1024)
        if not _verified(partial, record):
            raise ValueError(f"ComfyUI dependency failed its SHA-256 check: {filename}")
        partial.replace(path)
    return path


def _install_wheel(wheel: Path, site: Path) -> None:
    # Only purelib and platlib belong on the child's import path. The worker
    # runs with the desktop interpreter, not console scripts from these wheels.
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            mode = member.external_attr >> 16
            if not parts or parts[0] == "/" or ".." in parts or "\\" in member.filename or stat.S_ISLNK(mode):
                raise ValueError(f"Invalid path in {wheel.name}")
            if member.is_dir():
                continue
            if parts[0].endswith(".data"):
                if len(parts) < 3 or parts[1] not in {"purelib", "platlib"}:
                    continue
                parts = parts[2:]
            target = site.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("xb") as out:
                shutil.copyfileobj(src, out, 1024 * 1024)
            if mode & 0o111:
                target.chmod(0o755)


def provision_wheels(manifest: Path, cache: Path, site: Path, progress: Callable[[str], None] | None) -> None:
    lock = manifest.read_bytes()
    data = json.loads(lock)
    if data["python"] != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise RuntimeError(f"This engine needs the bundled Python {data['python']} desktop runtime.")
    tags = set(sys_tags())
    records: list[Wheel] = data["wheels"]
    cache.mkdir(parents=True, exist_ok=True)
    site.mkdir()
    tls = ssl.create_default_context()
    tls.load_verify_locations(cafile=certifi.where())
    for index, record in enumerate(records, 1):
        filename = record["filename"]
        name, _, _, wheel_tags = parse_wheel_filename(filename)
        if (Path(filename).name != filename or not wheel_tags & tags
                or name in {"torch", "torchvision", "torchaudio", "triton"}
                or name.startswith(("nvidia-", "rocm-"))
                or not record["url"].startswith("https://files.pythonhosted.org/")):
            raise ValueError(f"Incompatible dependency for this desktop runtime: {filename}")
        if progress:
            progress(f"Preparing private engine dependencies ({index}/{len(records)}): {name}")
        _install_wheel(_download(tls, cache, record), site)
    receipt = {"manifest_sha256": hashlib.sha256(lock).hexdigest()}
    (site / "ltx-comfy-deps.json").write_text(json.dumps(receipt), encoding="ascii")
