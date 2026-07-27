"""Isolation and cancellation invariants for the MLX Fast sidecar."""

from __future__ import annotations

import threading

import pytest

from runtime_config.mlx_runtime import MLXRuntimeDiscovery
from services.fast_video_pipeline import mlx_fast_video_pipeline as mlx_sidecar


class _BlockingProcess:
    pid = 43210

    def __init__(self, released: threading.Event) -> None:
        self._released = released
        self._return_code: int | None = None

    def poll(self) -> int | None:
        return self._return_code

    def wait(self) -> int:
        assert self._released.wait(timeout=2)
        self._return_code = -9
        return self._return_code


def test_generation_uses_fresh_process_group_and_cancellation_kills_it(monkeypatch) -> None:
    released = threading.Event()
    started = threading.Event()
    process = _BlockingProcess(released)
    popen_calls: list[tuple[list[str], dict[str, object]]] = []
    killed: list[tuple[int, int]] = []

    runtime = MLXRuntimeDiscovery(
        command_prefix=("/opt/ltx-2-mlx",),
        source="test",
        version="test",
        revision="test",
        dirty=False,
        core_version="test",
        mlx_version="test",
        compatible=True,
        reason="test",
    )
    monkeypatch.setattr(mlx_sidecar, "discover_mlx_runtime", lambda: runtime)

    def _popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        started.set()
        return process

    def _killpg(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        released.set()

    monkeypatch.setattr(mlx_sidecar.subprocess, "Popen", _popen)
    monkeypatch.setattr(mlx_sidecar.os, "killpg", _killpg)
    pipeline = mlx_sidecar.MLXFastVideoPipeline(
        model_source="test/model",
        low_ram=True,
        loras=[],
    )
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            pipeline.generate(
                prompt="test",
                seed=1,
                height=256,
                width=384,
                num_frames=9,
                frame_rate=8,
                images=[],
                output_path="/tmp/test.mp4",
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_run)
    thread.start()
    assert started.wait(timeout=2)
    assert mlx_sidecar.get_active_mlx_sidecar_pid() == process.pid
    assert mlx_sidecar.cancel_active_mlx_sidecar() is True
    thread.join(timeout=2)

    assert not thread.is_alive()
    command, kwargs = popen_calls[0]
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["LTX2_VAE_DECODE_BUDGET_GB"] == "2"
    assert "--auto-tiling" in command
    assert command[command.index("--model-precision") + 1] == "bf16"
    assert command[command.index("--profile-json") + 1].endswith(".jsonl")
    assert killed == [(process.pid, mlx_sidecar.signal.SIGKILL)]
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "cancelled" in str(errors[0]).lower()
    assert mlx_sidecar.get_active_mlx_sidecar_pid() is None


def test_distilled_sidecar_rejects_non_grid_dimensions_before_launch(monkeypatch) -> None:
    runtime = MLXRuntimeDiscovery(
        command_prefix=("/opt/ltx-2-mlx",),
        source="test",
        version="test",
        revision="test",
        dirty=False,
        core_version="test",
        mlx_version="test",
        compatible=True,
        reason="test",
    )
    monkeypatch.setattr(mlx_sidecar, "discover_mlx_runtime", lambda: runtime)
    pipeline = mlx_sidecar.MLXFastVideoPipeline(
        model_source="test/model",
        low_ram=False,
        loras=[],
    )

    with pytest.raises(ValueError, match="divisible by 64"):
        pipeline.generate(
            prompt="test",
            seed=1,
            height=544,
            width=960,
            num_frames=9,
            frame_rate=8,
            images=[],
            output_path="/tmp/must-not-launch.mp4",
        )


def test_cancel_without_active_sidecar_is_noop() -> None:
    assert mlx_sidecar.cancel_active_mlx_sidecar() is False
