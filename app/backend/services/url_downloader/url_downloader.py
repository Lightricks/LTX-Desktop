"""URL downloader service protocol definitions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from runtime_config.custom_lora_urls import HuggingFaceRepo


class UrlDownloadError(Exception):
    """A download that cannot finish. The message is safe to show to the user."""


class UrlDownloader(Protocol):
    def list_hugging_face_files(self, repo: HuggingFaceRepo, *, hf_token: str | None) -> list[str]:
        """The file names in a Hugging Face model repo, for the revision the link names.

        Raises ``UrlDownloadError`` when the repo is missing, needs access, or the server fails.
        """
        ...

    def fetch_hugging_face_card(self, repo: HuggingFaceRepo, *, hf_token: str | None) -> str:
        """The README text (the model card) of a Hugging Face model repo, capped in size.

        Returns an empty string when the repo has no README. Raises ``UrlDownloadError`` when the
        repo is missing, needs access, or the server fails.
        """
        ...

    def download(
        self,
        url: str,
        dest: Path,
        *,
        hf_token: str | None,
        max_bytes: int,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Stream ``url`` to ``dest``.

        ``on_progress`` gets ``(bytes_written, expected_bytes)``; ``expected_bytes`` is 0 when
        the server sends no length. Raises ``UrlDownloadError`` on a refused link, a server
        error, a sign-in wall, or a file larger than ``max_bytes``.
        """
        ...
