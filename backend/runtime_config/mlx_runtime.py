"""Portable discovery and exact identity validation for the external MLX runtime."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from runtime_config.runtime_policy import MLX_RUNTIME_REVISION, MLX_RUNTIME_VERSION

@dataclass(frozen=True)
class MLXRuntimeDiscovery:
    command_prefix: tuple[str, ...] | None
    source: str
    version: str
    revision: str | None
    dirty: bool | None
    core_version: str | None
    mlx_version: str | None
    compatible: bool
    reason: str


def _identity_python_for_entrypoint(executable: str) -> str | None:
    """Resolve the interpreter from an ordinary venv console-script shebang."""
    try:
        first_line = Path(executable).read_bytes().splitlines()[0].decode("utf-8")
    except (OSError, UnicodeDecodeError, IndexError):
        return None
    if not first_line.startswith("#!"):
        return None
    parts = shlex.split(first_line[2:])
    if not parts:
        return None
    if Path(parts[0]).name == "env" and len(parts) > 1:
        return shutil.which(parts[1])
    return parts[0]


def _candidate_runtimes() -> list[tuple[tuple[str, ...], str | None, str]]:
    candidates: list[tuple[tuple[str, ...], str | None, str]] = []
    configured_python = os.environ.get("LTX_MLX_PYTHON")
    if configured_python:
        python = str(Path(configured_python).expanduser())
        candidates.append(((python, "-m", "ltx_pipelines_mlx.cli"), python, "LTX_MLX_PYTHON"))

    default_python = Path("~/video-models/ltx-2-mlx/.venv/bin/python").expanduser()
    candidates.append(
        (
            (str(default_python), "-m", "ltx_pipelines_mlx.cli"),
            str(default_python),
            str(default_python),
        )
    )

    configured_executable = os.environ.get("LTX_MLX_EXECUTABLE")
    executable = shutil.which(configured_executable or "ltx-2-mlx")
    if executable is not None:
        candidates.append(((executable,), _identity_python_for_entrypoint(executable), executable))

    deduplicated: list[tuple[tuple[str, ...], str | None, str]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        if candidate[0] not in seen:
            seen.add(candidate[0])
            deduplicated.append(candidate)
    return deduplicated


def _probe_candidate(
    command_prefix: tuple[str, ...],
    identity_python: str | None,
    source: str,
) -> MLXRuntimeDiscovery | None:
    if identity_python is None or not Path(identity_python).expanduser().is_file():
        return None
    try:
        result = subprocess.run(
            [identity_python, "-m", "ltx_pipelines_mlx.utils.runtime_info"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        lines = result.stdout.strip().splitlines()
        if result.returncode != 0 or not lines:
            return None
        identity = json.loads(lines[-1])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None

    version = os.environ.get("LTX_MLX_RUNTIME_VERSION") or identity.get("runtime_version")
    revision = os.environ.get("LTX_MLX_RUNTIME_REVISION") or identity.get("runtime_commit")
    dirty = identity.get("runtime_dirty")
    exact_version = version == MLX_RUNTIME_VERSION
    exact_revision = revision == MLX_RUNTIME_REVISION
    clean = dirty is not True
    compatible = exact_version and exact_revision and clean
    mismatches: list[str] = []
    if not exact_version:
        mismatches.append(f"version {version or 'unknown'} != {MLX_RUNTIME_VERSION}")
    if not exact_revision:
        mismatches.append(f"revision {revision or 'unknown'} != {MLX_RUNTIME_REVISION}")
    if not clean:
        mismatches.append("runtime checkout is dirty")
    reason = "Exact pinned MLX runtime identity verified." if compatible else "; ".join(mismatches)
    return MLXRuntimeDiscovery(
        command_prefix=command_prefix if compatible else None,
        source=source,
        version=version or "unknown",
        revision=revision,
        dirty=dirty,
        core_version=identity.get("core_version"),
        mlx_version=identity.get("mlx_version"),
        compatible=compatible,
        reason=reason,
    )


@lru_cache(maxsize=1)
def discover_mlx_runtime() -> MLXRuntimeDiscovery:
    """Prefer an explicit Python, then the sibling runtime, then a PATH entrypoint."""
    first_detected: MLXRuntimeDiscovery | None = None
    for command_prefix, identity_python, source in _candidate_runtimes():
        discovered = _probe_candidate(command_prefix, identity_python, source)
        if discovered is None:
            continue
        if discovered.compatible:
            return discovered
        if first_detected is None:
            first_detected = discovered
    if first_detected is not None:
        return first_detected
    return MLXRuntimeDiscovery(
        command_prefix=None,
        source="not found",
        version="not installed",
        revision=None,
        dirty=None,
        core_version=None,
        mlx_version=None,
        compatible=False,
        reason="No discoverable MLX Python runtime or ltx-2-mlx entrypoint was found.",
    )
