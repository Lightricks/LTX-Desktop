"""Portable MLX runtime discovery and exact-pin validation."""

from __future__ import annotations

import json
from pathlib import Path

from runtime_config import mlx_runtime
from runtime_config.runtime_policy import MLX_RUNTIME_REVISION, MLX_RUNTIME_VERSION


class _Result:
    def __init__(self, identity: dict[str, object]) -> None:
        self.returncode = 0
        self.stdout = json.dumps(identity)
        self.stderr = ""


def _identity(*, version: str = MLX_RUNTIME_VERSION, revision: str = MLX_RUNTIME_REVISION, dirty: bool = False):
    return {
        "runtime_version": version,
        "runtime_commit": revision,
        "runtime_dirty": dirty,
        "core_version": "core-test",
        "mlx_version": "mlx-test",
    }


def test_explicit_python_precedes_default_and_requires_exact_clean_pin(monkeypatch, tmp_path: Path) -> None:
    explicit = tmp_path / "explicit-python"
    sibling = tmp_path / "sibling-python"
    explicit.touch()
    sibling.touch()
    monkeypatch.setenv("LTX_MLX_PYTHON", str(explicit))
    monkeypatch.setattr(mlx_runtime.Path, "expanduser", lambda self: sibling if "video-models" in str(self) else self)
    probed: list[str] = []

    def _run(command, **_kwargs):
        probed.append(command[0])
        return _Result(_identity())

    monkeypatch.setattr(mlx_runtime.subprocess, "run", _run)
    mlx_runtime.discover_mlx_runtime.cache_clear()
    discovered = mlx_runtime.discover_mlx_runtime()

    assert discovered.compatible is True
    assert discovered.command_prefix == (str(explicit), "-m", "ltx_pipelines_mlx.cli")
    assert probed == [str(explicit)]


def test_incompatible_explicit_runtime_falls_through_to_exact_sibling(monkeypatch, tmp_path: Path) -> None:
    explicit = tmp_path / "explicit-python"
    sibling = tmp_path / "sibling-python"
    explicit.touch()
    sibling.touch()
    monkeypatch.setenv("LTX_MLX_PYTHON", str(explicit))
    monkeypatch.setattr(mlx_runtime.Path, "expanduser", lambda self: sibling if "video-models" in str(self) else self)

    def _run(command, **_kwargs):
        identity = _identity(version="wrong", revision="wrong") if command[0] == str(explicit) else _identity()
        return _Result(identity)

    monkeypatch.setattr(mlx_runtime.subprocess, "run", _run)
    mlx_runtime.discover_mlx_runtime.cache_clear()
    discovered = mlx_runtime.discover_mlx_runtime()

    assert discovered.compatible is True
    assert discovered.command_prefix == (str(sibling), "-m", "ltx_pipelines_mlx.cli")


def test_dirty_exact_runtime_is_not_eligible(monkeypatch, tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setenv("LTX_MLX_PYTHON", str(python))
    monkeypatch.setattr(mlx_runtime.Path, "expanduser", lambda self: tmp_path / "missing" if "video-models" in str(self) else self)
    monkeypatch.setattr(mlx_runtime.subprocess, "run", lambda *_args, **_kwargs: _Result(_identity(dirty=True)))
    monkeypatch.setattr(mlx_runtime.shutil, "which", lambda _name: None)
    mlx_runtime.discover_mlx_runtime.cache_clear()
    discovered = mlx_runtime.discover_mlx_runtime()

    assert discovered.compatible is False
    assert discovered.command_prefix is None
    assert "dirty" in discovered.reason
