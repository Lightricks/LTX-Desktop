"""Streaming URL downloader built on ``requests``."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast
from urllib.parse import quote, urljoin

import requests

from runtime_config.custom_lora_urls import (
    CIVITAI_HOSTS,
    CustomLoraUrl,
    CustomLoraUrlError,
    HuggingFaceRepo,
    check_fetch_url,
)
from services.url_downloader.public_address_adapter import PublicAddressAdapter
from services.url_downloader.url_downloader import UrlDownloadError

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 1024 * 1024
_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
# (connect, read) seconds. The read timeout applies to each chunk, not to the whole file.
_TIMEOUT = (10, 60)
# A model card is a few kilobytes. The cap stops a repo that serves a huge README.
_MAX_CARD_BYTES = 256 * 1024


def _check(url: str) -> CustomLoraUrl:
    try:
        return check_fetch_url(url)
    except CustomLoraUrlError as exc:
        raise UrlDownloadError(str(exc)) from exc


def _auth_error(host: str) -> UrlDownloadError:
    if host in CIVITAI_HOSTS:
        return UrlDownloadError(
            "Civitai needs a sign-in for this file. Download it in your browser, "
            "then add it as a file."
        )
    return UrlDownloadError(
        "The server needs a sign-in for this file. Download it in your browser, "
        "then add it as a file."
    )


def public_only_session() -> requests.Session:
    """A session that opens connections to public addresses only (see ``PublicAddressAdapter``)."""
    session = requests.Session()
    # TODO(security): CWE-918. With a system proxy, the proxy opens the connection and the adapter
    # cannot see its address. Ignore HTTPS_PROXY, ALL_PROXY and .netrc so every connection is ours.
    # A user behind a mandatory proxy adds the file instead of a link.
    session.trust_env = False
    session.mount("https://", PublicAddressAdapter())
    return session


class UrlDownloaderImpl:
    def __init__(self, session_factory: Callable[[], requests.Session] = public_only_session) -> None:
        # Tests pass a session with a fake transport. The default checks every address it connects to.
        self._session_factory = session_factory

    def list_hugging_face_files(self, repo: HuggingFaceRepo, *, hf_token: str | None) -> list[str]:
        revision = quote(repo.revision, safe="")
        # The host is fixed, so this request never follows a user-supplied address.
        url = f"https://huggingface.co/api/models/{repo.repo_id}/revision/{revision}"
        headers = {"User-Agent": "LTX-Desktop"}
        if hf_token is not None:
            headers["Authorization"] = f"Bearer {hf_token}"
        try:
            with self._session_factory() as session:
                response = session.get(url, headers=headers, allow_redirects=False, timeout=_TIMEOUT)
        except requests.exceptions.RequestException as exc:
            logger.warning("Could not reach Hugging Face to list %s", repo.repo_id)
            raise UrlDownloadError("Could not reach Hugging Face.") from exc
        with response:
            if response.status_code in (401, 403):
                raise UrlDownloadError(
                    "This Hugging Face repository needs access. Sign in to Hugging Face in "
                    "Settings > Models and accept its license on the model page."
                )
            if response.status_code == 404:
                raise UrlDownloadError("The Hugging Face repository was not found.")
            if response.status_code in _REDIRECT_STATUSES:
                # A renamed or moved repository answers with a redirect. The link is not followed.
                raise UrlDownloadError("The repository moved. Copy the new link from the model page.")
            if response.status_code != 200:
                raise UrlDownloadError(f"Hugging Face answered with status {response.status_code}.")
            try:
                payload: object = response.json()
            except ValueError as exc:
                raise UrlDownloadError("Hugging Face sent an answer that could not be read.") from exc
        siblings = cast("dict[str, object]", payload).get("siblings") if isinstance(payload, dict) else None
        if not isinstance(siblings, list):
            raise UrlDownloadError("Hugging Face sent an answer that could not be read.")
        names: list[str] = []
        for sibling in cast("list[object]", siblings):
            name = cast("dict[str, object]", sibling).get("rfilename") if isinstance(sibling, dict) else None
            if isinstance(name, str):
                names.append(name)
        return names

    def fetch_hugging_face_card(self, repo: HuggingFaceRepo, *, hf_token: str | None) -> str:
        revision = quote(repo.revision, safe="")
        # The host is fixed, so this request never follows a user-supplied address.
        url = f"https://huggingface.co/{repo.repo_id}/raw/{revision}/README.md"
        headers = {"User-Agent": "LTX-Desktop"}
        if hf_token is not None:
            headers["Authorization"] = f"Bearer {hf_token}"
        try:
            with self._session_factory() as session:
                response = session.get(
                    url, headers=headers, allow_redirects=False, timeout=_TIMEOUT, stream=True
                )
                with response:
                    if response.status_code == 404:
                        return ""
                    if response.status_code in (401, 403):
                        raise UrlDownloadError("This Hugging Face repository needs access.")
                    if response.status_code != 200:
                        raise UrlDownloadError(f"Hugging Face answered with status {response.status_code}.")
                    data = bytearray()
                    for chunk in response.iter_content(chunk_size=16 * 1024):
                        data.extend(chunk)
                        if len(data) >= _MAX_CARD_BYTES:
                            break
        except requests.exceptions.RequestException as exc:
            logger.warning("Could not reach Hugging Face to read the card of %s", repo.repo_id)
            raise UrlDownloadError("Could not reach Hugging Face.") from exc
        return bytes(data[:_MAX_CARD_BYTES]).decode("utf-8", errors="replace")

    def download(
        self,
        url: str,
        dest: Path,
        *,
        hf_token: str | None,
        max_bytes: int,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        current = _check(url)
        # One session for the whole redirect chain. A streamed response needs it open until saved.
        with self._session_factory() as session:
            for _ in range(_MAX_REDIRECTS + 1):
                # Built again for each hop, so the token never reaches a redirect target such as a
                # signed CDN URL.
                headers = {"User-Agent": "LTX-Desktop"}
                if hf_token is not None and current.is_hugging_face:
                    headers["Authorization"] = f"Bearer {hf_token}"
                try:
                    response = session.get(
                        current.url,
                        headers=headers,
                        stream=True,
                        allow_redirects=False,
                        timeout=_TIMEOUT,
                    )
                except requests.exceptions.RequestException as exc:
                    logger.warning("Custom LoRA download could not reach %s", current.host)
                    raise UrlDownloadError("Could not reach the server.") from exc
                with response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("Location")
                        if not location:
                            raise UrlDownloadError("The server sent a bad redirect.")
                        current = _check(urljoin(current.url, location))
                        continue
                    self._save(response, current, dest, max_bytes, on_progress)
                    return
        raise UrlDownloadError("The link redirects too many times.")

    @staticmethod
    def _save(
        response: requests.Response,
        source: CustomLoraUrl,
        dest: Path,
        max_bytes: int,
        on_progress: Callable[[int, int], None] | None,
    ) -> None:
        status = response.status_code
        if status in (401, 403):
            raise _auth_error(source.host)
        if status == 404:
            raise UrlDownloadError("The file was not found at this link.")
        if status != 200:
            raise UrlDownloadError(f"The server answered with status {status}.")
        if "text/html" in response.headers.get("Content-Type", "").lower():
            # A web page, not a file: most often a sign-in or error page.
            raise UrlDownloadError("The link opens a web page, not a file.")
        length_header = response.headers.get("Content-Length", "")
        expected = int(length_header) if length_header.isdigit() else 0
        if expected > max_bytes:
            raise UrlDownloadError("The file is too large for a LoRA.")
        written = 0
        try:
            with open(dest, "wb") as out:
                for data in response.iter_content(chunk_size=_CHUNK_BYTES):
                    if not data:
                        continue
                    written += len(data)
                    if written > max_bytes:
                        raise UrlDownloadError("The file is too large for a LoRA.")
                    out.write(data)
                    if on_progress is not None:
                        on_progress(written, expected)
        except requests.exceptions.RequestException as exc:
            raise UrlDownloadError("The download was interrupted.") from exc
        # A compressed transfer is decoded on the fly, so its byte count never matches the header.
        compressed = bool(response.headers.get("Content-Encoding"))
        if expected and not compressed and written != expected:
            raise UrlDownloadError("The download was interrupted.")
