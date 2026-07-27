"""Apple Silicon MLX adapter for parity-proven Fast T2V/I2V requests."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
from typing import Final

from api_types import ImageConditioningInput
from runtime_config.mlx_runtime import discover_mlx_runtime
from runtime_config.runtime_policy import MLX_BF16_MODEL_SOURCE
from services.fast_video_pipeline.mlx_profile import (
    allocate_mlx_profile_path,
    begin_mlx_profile,
    finish_mlx_profile,
)

logger = logging.getLogger(__name__)
_MLX_DISTILLED_SPATIAL_GRID = 64
_sidecar_lock = threading.Lock()
_active_sidecar_process: subprocess.Popen[bytes] | None = None
_cancelled_sidecar_pids: set[int] = set()


def get_active_mlx_sidecar_pid() -> int | None:
    with _sidecar_lock:
        process = _active_sidecar_process
        return process.pid if process is not None and process.poll() is None else None


def _register_active_sidecar(process: subprocess.Popen[bytes]) -> None:
    global _active_sidecar_process
    with _sidecar_lock:
        if _active_sidecar_process is not None and _active_sidecar_process.poll() is None:
            raise RuntimeError("An MLX sidecar is already active")
        _active_sidecar_process = process


def _release_active_sidecar(process: subprocess.Popen[bytes]) -> bool:
    global _active_sidecar_process
    with _sidecar_lock:
        was_cancelled = process.pid in _cancelled_sidecar_pids
        _cancelled_sidecar_pids.discard(process.pid)
        if _active_sidecar_process is process:
            _active_sidecar_process = None
        return was_cancelled


def cancel_active_mlx_sidecar() -> bool:
    """Kill the entire per-job MLX process group so cancellation returns memory."""
    with _sidecar_lock:
        process = _active_sidecar_process
        if process is None or process.poll() is not None:
            return False
        _cancelled_sidecar_pids.add(process.pid)
        pid = process.pid

    try:
        # start_new_session=True makes the sidecar PID its process-group ID. A
        # hard group kill is intentional: no MLX worker/grandchild may retain
        # unified memory after the user cancels the job.
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    logger.info("Killed MLX sidecar process group pid=%d after cancellation", pid)
    return True


class MLXFastVideoPipeline:
    """Wrap ``ltx-pipelines-mlx`` without importing MLX on non-Darwin hosts.

    This adapter intentionally covers only the distilled Fast T2V/I2V surface.
    A2V, Retake, Extend, IC-LoRA, prepared/API embeddings, and the rest of the
    product continue to use the official Torch pipeline.
    """

    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: object,
        streaming_prefetch_count: int | None,
        loras: list[tuple[str, float]] | None = None,
    ) -> "MLXFastVideoPipeline":
        del checkpoint_path, gemma_root, upsampler_path, device
        return MLXFastVideoPipeline(
            model_source=os.environ.get("LTX_MLX_MODEL_ID", MLX_BF16_MODEL_SOURCE),
            low_ram=streaming_prefetch_count is not None,
            loras=loras or [],
        )

    def __init__(
        self,
        *,
        model_source: str,
        low_ram: bool,
        loras: list[tuple[str, float]],
    ) -> None:
        self._model_source = model_source
        self._low_ram = low_ram
        self._loras = loras
        model_precision = os.environ.get("LTX_MLX_MODEL_VARIANT", "bf16").strip().lower()
        self._model_precision = model_precision if model_precision in {"bf16", "q8", "q4"} else "unknown"
        runtime = discover_mlx_runtime()
        command_prefix = runtime.command_prefix
        if command_prefix is None:
            raise RuntimeError(
                "Compatible MLX sidecar unavailable: "
                f"{runtime.reason}. Set LTX_MLX_PYTHON to the pinned runtime or select Torch."
            )
        self._command_prefix: tuple[str, ...] = command_prefix
        # The MLX VAE decoder estimates tile size from this budget. Keep the
        # setting child-local so the lean FastAPI process never acquires MLX
        # allocator state or job-specific runtime configuration.
        self._vae_decode_budget_gb = "2" if low_ram else "8"
        logger.info(
            "Created MLX Fast sidecar pipeline command=%s model=%s mode=%s loras=%d",
            " ".join(self._command_prefix),
            model_source,
            "low_ram" if low_ram else "eager",
            len(loras),
        )

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        if height % _MLX_DISTILLED_SPATIAL_GRID or width % _MLX_DISTILLED_SPATIAL_GRID:
            raise ValueError(
                "MLX distilled output dimensions must both be divisible by "
                f"{_MLX_DISTILLED_SPATIAL_GRID}; got {width}x{height}"
            )
        profile_path = allocate_mlx_profile_path(output_path)
        command: list[str] = [
            *self._command_prefix,
            "generate",
            "--distilled",
            "--prompt",
            prompt,
            "--output",
            output_path,
            "--model",
            self._model_source,
            "--model-precision",
            self._model_precision,
            "--height",
            str(height),
            "--width",
            str(width),
            "--frames",
            str(num_frames),
            "--frame-rate",
            str(frame_rate),
            "--seed",
            str(seed),
            "--profile-json",
            str(profile_path),
        ]
        if self._low_ram:
            command.extend(["--low-ram", "--auto-tiling"])
        for image in images:
            command.extend(
                [
                    "--image",
                    image.path,
                    str(image.frame_idx),
                    str(image.strength),
                ]
            )
        for path, strength in self._loras:
            command.extend(["--lora", path, str(strength)])
        child_env = os.environ.copy()
        child_env["LTX2_VAE_DECODE_BUDGET_GB"] = self._vae_decode_budget_gb
        # One fresh process group per generation is a memory invariant. A
        # persistent MLX worker retained ~50.8 GiB and made request 2 slower;
        # process exit is the allocator teardown boundary we can prove.
        process = subprocess.Popen(command, env=child_env, start_new_session=True)
        _register_active_sidecar(process)
        begin_mlx_profile(profile_path)
        return_code: int | None = None
        was_cancelled = False
        try:
            return_code = process.wait()
        finally:
            was_cancelled = _release_active_sidecar(process)
            profile_status = (
                "cancelled"
                if was_cancelled
                else "success"
                if return_code == 0
                else "error"
            )
            finish_mlx_profile(profile_path, profile_status)
        if was_cancelled:
            raise RuntimeError("Generation was cancelled")
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code or -1, command)

    def warmup(self, output_path: str) -> None:
        self.generate(
            prompt="test warmup",
            seed=42,
            height=256,
            width=384,
            num_frames=9,
            frame_rate=8,
            images=[],
            output_path=output_path,
        )
        try:
            os.unlink(output_path)
        except FileNotFoundError:
            pass

    def compile_transformer(self) -> None:
        # MLX compiles/evaluates its own lazy graphs; torch.compile is not
        # applicable. Keeping the protocol method makes selection transparent.
        logger.info("Skipping torch.compile for the MLX Fast pipeline")
