"""Materialize the pinned Electron shell and private Comfy engine in a setup staging directory."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import zipfile


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verified(record, cache, fetch) -> Path:
    source = Path(fetch(record, cache))
    if source.stat().st_size != record["size"] or _sha256(source) != record["sha256"]:
        raise ValueError(f"Downloaded file failed verification: {record['filename']}")
    return source


def _long_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    absolute = str(path.absolute())
    prefix = "\\\\?\\"
    if absolute.startswith(prefix):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path(prefix + "UNC\\" + absolute[2:])
    return Path(prefix + absolute)


def _parts(member: str) -> list[str]:
    parts = member.split("/")
    for part in parts:
        if (not part or part in (".", "..") or part[-1] in (" ", ".")
                or any(ord(ch) < 32 or ch in '<>:"\\|?*' for ch in part)
                or PureWindowsPath(part).is_reserved()):
            raise ValueError(f"Invalid Windows archive path {member!r}")
    return parts


def _wheel_member(name: str) -> list[str]:
    parts = _parts(name)
    if not parts[0].endswith(".data"):
        return parts
    if len(parts) < 3:
        raise ValueError(f"Invalid wheel data member: {name}")
    prefix = {"purelib": [], "platlib": [], "scripts": ["Scripts"], "data": [], "headers": ["Include"]}
    return prefix[parts[1]] + parts[2:]


def _copy(stream, dest: Path, *, length=None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as output:
        shutil.copyfileobj(stream, output, 1024 * 1024)
    if length is not None and dest.stat().st_size != length:
        raise ValueError(f"Incomplete archive member: {dest.name}")


def _apply_launcher(launcher: Path, resources: Path, spec: dict) -> None:
    patch_file = resources / "vendor-launcher.zip"
    if _sha256(patch_file) != spec["patch_sha256"]:
        raise ValueError("The bundled launcher patch failed verification.")
    with zipfile.ZipFile(patch_file) as patch, launcher.open("r+b") as stream:
        for entry in patch.infolist():
            offset = int(entry.filename)
            if offset < 0 or entry.file_size > 65536 or offset + entry.file_size > spec["size"]:
                raise ValueError("Invalid launcher patch range.")
            stream.seek(offset)
            stream.write(patch.read(entry))
        stream.truncate(spec["size"])
    if _sha256(launcher) != spec["sha256"]:
        raise ValueError("Branded Electron launcher did not match the released version.")


def hydrate(stage: Path, resources: Path, cache: Path, fetch) -> None:
    """Populate a private installation stage. fetch(record, cache) supplies pinned archives."""
    stage, resources = _long_path(stage), _long_path(resources)
    lock = json.loads((resources / "vendor-lock.json").read_bytes())
    cache.mkdir(parents=True, exist_ok=True)
    electron = _verified(lock["electron"], cache, fetch)
    with zipfile.ZipFile(electron) as z:
        for entry in z.infolist():
            if entry.is_dir() or entry.filename == "resources/default_app.asar":
                continue
            name = lock["launcher"]["filename"] if entry.filename == "electron.exe" else entry.filename
            parts = _parts(name)
            if len(parts) > 1 and parts[0] != "locales":
                raise ValueError(f"Unexpected Electron member: {name}")
            target = stage.joinpath(*parts)
            with z.open(entry) as src:
                _copy(src, target, length=entry.file_size)
            if entry.filename == "electron.exe":
                _apply_launcher(target, resources, lock["launcher"])

    root = stage / "resources" / "comfy-engine"
    root.mkdir(parents=True, exist_ok=True)
    for record in lock["sources"]:
        source = _verified(record, cache, fetch)
        shutil.copyfile(source, root.joinpath(*_parts(record["filename"])))

    # The caller owns this private stage. Preserve any other staged app resources.
    site = root / "site-packages"
    seen = set()
    for record in lock["wheels"]:
        file = _verified(record, cache, fetch)
        with zipfile.ZipFile(file) as z:
            for entry in z.infolist():
                if entry.is_dir():
                    continue
                parts = _wheel_member(entry.filename)
                normalized = "/".join(parts).casefold()
                if normalized in seen:
                    raise ValueError(f"Overlapping wheel payload at {entry.filename}")
                seen.add(normalized)
                with z.open(entry) as src:
                    _copy(src, site.joinpath(*parts), length=entry.file_size)
    (site / "ltx-comfy-deps.json").write_bytes(lock["receipt"].encode("utf-8"))
