"""Stage a model checkpoint download, then commit it to its canonical path.

The model download session and the catalog download of an IC-LoRA with processor models
share these two steps. Each caller passes its own staging folder.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from api_types import ModelCheckpointID
from runtime_config.model_download_specs import (
    get_model_cp_spec,
    is_cp_downloaded,
    resolve_model_path,
    staged_download_dir,
    staged_target_path,
)
from services.interfaces import ModelDownloader


def stage_checkpoint(
    downloader: ModelDownloader,
    staging_dir: Path,
    cp_id: ModelCheckpointID,
    token: str | None,
    on_progress: Callable[[int], None],
) -> None:
    spec = get_model_cp_spec(cp_id)
    staging_dir.mkdir(parents=True, exist_ok=True)

    if spec.is_folder:
        downloader.download_snapshot(
            repo_id=spec.repo_id,
            local_dir=str(staged_download_dir(staging_dir, cp_id)),
            on_progress=on_progress,
            token=token,
            revision=spec.revision,
        )
        return

    downloader.download_file(
        repo_id=spec.repo_id,
        filename=spec.download_filename,
        local_dir=str(staging_dir),
        on_progress=on_progress,
        token=token,
        revision=spec.revision,
    )
    # Nested HF paths land under staging_dir/<repo_filename> (e.g. vae/foo.safetensors
    # for LTX 2.5); move to the spec's local relative_path before commit (e.g.
    # ltx-2.5/foo.safetensors).
    downloaded = staging_dir / Path(spec.download_filename)
    target = staged_target_path(staging_dir, cp_id)
    if downloaded != target and downloaded.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        downloaded.replace(target)
        parent = downloaded.parent
        while parent != staging_dir and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent


def commit_staged_checkpoint(models_dir: Path, staging_dir: Path, cp_id: ModelCheckpointID) -> bool:
    """Move the staged checkpoint to its canonical path. False when it was already installed."""
    src = staged_target_path(staging_dir, cp_id)
    dst = resolve_model_path(models_dir, cp_id)
    spec = get_model_cp_spec(cp_id)

    if is_cp_downloaded(models_dir, cp_id):
        if src.exists():
            if spec.is_folder:
                shutil.rmtree(src)
            else:
                src.unlink(missing_ok=True)
        return False

    if not is_cp_downloaded(staging_dir, cp_id):
        raise ValueError(f"Incomplete download for {cp_id}. Retry the model download.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if spec.is_folder:
        if dst.exists():
            shutil.rmtree(dst)
        src.rename(dst)
    else:
        if dst.exists():
            dst.unlink()
        src.rename(dst)
    return True
