"""Hugging Face model download service wrapper."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore[reportUnknownVariableType]
from huggingface_hub.utils.tqdm import tqdm as HubTqdm


def _make_progress_tqdm_class(callback: Callable[[int], None]) -> type:
    """Give each download its own reconstructed-byte counter, including resumed bytes."""
    lock = Lock()

    class ProgressTqdm(HubTqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._report_bytes = kwargs.get("unit") == "B" and not str(kwargs.get("name")).endswith(".transfer")
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)  # type: ignore[reportUnknownMemberType]
            if self._report_bytes and self.n:
                callback(int(self.n))

        def update(self, n: float | int | None = 1) -> bool | None:
            if self._report_bytes and n:
                # tqdm suppresses updates on a hidden bar. Hub can update the
                # snapshot bar from several file workers at the same time.
                with lock:
                    self.n += n
                    callback(int(self.n))
            return None

    return ProgressTqdm


class HuggingFaceDownloader:
    """Wraps huggingface_hub download functions."""

    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        token: str | None,
        on_progress: Callable[[int], None] | None = None,
        revision: str | None = None,
    ) -> Path:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            token=token,
            revision=revision,
            tqdm_class=_make_progress_tqdm_class(on_progress) if on_progress is not None else None,
        )
        return Path(path)

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: str,
        token: str | None,
        on_progress: Callable[[int], None] | None = None,
        revision: str | None = None,
    ) -> Path:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=token,
            revision=revision,
            tqdm_class=_make_progress_tqdm_class(on_progress) if on_progress is not None else None,
        )
        return Path(path)
