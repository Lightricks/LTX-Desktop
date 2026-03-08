from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, TypeVar

from huggingface_hub import hf_hub_download, snapshot_download

T = TypeVar("T")


def _is_winerror32(exc: BaseException) -> bool:
    if isinstance(exc, OSError):
        if getattr(exc, "winerror", None) == 32:
            return True
        return "WinError 32" in str(exc)
    return False


def _run_with_lock_retry(name: str, func: Callable[[], T], retries: int = 8) -> T:
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:
            if not _is_winerror32(exc) or attempt == retries:
                raise
            wait_s = min(45, 2 * attempt)
            print(
                f"[{name}] hit transient file lock (WinError 32), retry {attempt}/{retries} in {wait_s}s..."
            )
            time.sleep(wait_s)

    # Unreachable, but keeps type-checkers satisfied.
    raise RuntimeError(f"{name} failed after retries")


def main() -> None:
    models_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "LTXDesktop" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target models directory: {models_dir}")

    # 1) Main checkpoint
    print("Downloading checkpoint from Lightricks/LTX-2.3...")
    ckpt_path = _run_with_lock_retry(
        "checkpoint",
        lambda: hf_hub_download(
            repo_id="Lightricks/LTX-2.3",
            filename="ltx-2.3-22b-distilled.safetensors",
            local_dir=str(models_dir),
            resume_download=True,
        ),
    )
    print(f"Checkpoint ready: {ckpt_path}")

    # 2) Upscaler
    print("Downloading upscaler from Lightricks/LTX-2.3...")
    upscaler_path = _run_with_lock_retry(
        "upsampler",
        lambda: hf_hub_download(
            repo_id="Lightricks/LTX-2.3",
            filename="ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            local_dir=str(models_dir),
            resume_download=True,
        ),
    )
    print(f"Upscaler ready: {upscaler_path}")

    # 3) Gemma text encoder snapshot
    print("Downloading text encoder snapshot from Lightricks/gemma-3-12b-it-qat-q4_0-unquantized...")
    text_encoder_dir = _run_with_lock_retry(
        "text_encoder",
        lambda: snapshot_download(
            repo_id="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
            local_dir=str(models_dir / "gemma-3-12b-it-qat-q4_0-unquantized"),
            resume_download=True,
        ),
    )
    print(f"Text encoder ready: {text_encoder_dir}")

    # 4) Z-Image-Turbo snapshot
    print("Downloading Z-Image-Turbo snapshot from Tongyi-MAI/Z-Image-Turbo...")
    zit_dir = _run_with_lock_retry(
        "zit",
        lambda: snapshot_download(
            repo_id="Tongyi-MAI/Z-Image-Turbo",
            local_dir=str(models_dir / "Z-Image-Turbo"),
            resume_download=True,
        ),
    )
    print(f"Z-Image-Turbo ready: {zit_dir}")

    print("All requested model downloads completed.")


if __name__ == "__main__":
    main()
