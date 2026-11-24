"""HTTP client wrapper service."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import requests

from services.http_client.http_client import (
    HttpBodyTooLargeError,
    HttpRedirectRefusedError,
    HttpTransportError,
)
from services.services_utils import JSONValue, RequestData

logger = logging.getLogger(__name__)

# Transport-level failures (timeout, connection reset, broken chunked stream). All are
# treated the same: the request didn't complete, so callers can retry. Connection resets
# mid-response (ChunkedEncodingError) are common on long-running sync generations.
_TRANSPORT_ERRORS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)


_STREAM_CHUNK_BYTES = 64 * 1024
_MAX_REDIRECTS = 3
# A stalled read ends after this long at most, so the download overruns its deadline by no more.
_MAX_READ_STALL_SECONDS = 10


@dataclass(frozen=True)
class _BufferedResponse:
    """A response whose body was read under a size cap."""

    status_code: int
    headers: Mapping[str, str]
    content: bytes

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> object:
        return json.loads(self.content)


def _redirect_target(url: str, location: str) -> str:
    """The next URL of a redirect. A redirect may keep the scheme or upgrade to https, never downgrade."""
    target = urljoin(url, location)
    scheme = urlsplit(target).scheme
    if scheme != "https" and scheme != urlsplit(url).scheme:
        raise HttpRedirectRefusedError("Redirect to a different scheme")
    return target


def _get_capped(
    url: str, headers: dict[str, str] | None, timeout: int, max_bytes: int
) -> _BufferedResponse:
    """Read the body in chunks and stop at max_bytes, so a huge body never fills memory.

    ``timeout`` is the total time for every hop and read together (CWE-770). Each hop gets the
    time left, and a read that stalls ends after ``_MAX_READ_STALL_SECONDS`` at most, so the
    call overruns the deadline by that much and no more. Redirects are followed by hand, so each
    hop is checked (CWE-918).
    """
    deadline = time.monotonic() + timeout
    first_host = urlsplit(url).netloc
    for _ in range(_MAX_REDIRECTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HttpTransportError(f"Download took longer than {timeout} seconds")
        # A credential header goes only to the host the caller named.
        hop_headers = headers if urlsplit(url).netloc == first_host else None
        with requests.get(
            url,
            headers=hop_headers,
            timeout=(remaining, min(remaining, _MAX_READ_STALL_SECONDS)),
            stream=True,
            allow_redirects=False,
        ) as response:
            location = response.headers.get("Location")
            if response.is_redirect and location:
                url = _redirect_target(url, location)
                continue
            declared = response.headers.get("Content-Length", "")
            if declared.isdigit() and int(declared) > max_bytes:
                raise HttpBodyTooLargeError(f"Response is larger than {max_bytes} bytes")
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=_STREAM_CHUNK_BYTES):
                if time.monotonic() > deadline:
                    raise HttpTransportError(f"Download took longer than {timeout} seconds")
                total += len(chunk)
                if total > max_bytes:
                    raise HttpBodyTooLargeError(f"Response is larger than {max_bytes} bytes")
                chunks.append(chunk)
            return _BufferedResponse(response.status_code, response.headers, b"".join(chunks))
    raise HttpRedirectRefusedError(f"More than {_MAX_REDIRECTS} redirects")


class HTTPClientImpl:
    """Wraps requests.* for external API calls."""

    def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        json_payload: Mapping[str, JSONValue] | None = None,
        data: RequestData = None,
        timeout: int = 30,
    ) -> requests.Response:
        try:
            return requests.post(url, headers=headers, json=json_payload, data=data, timeout=timeout)
        except _TRANSPORT_ERRORS as exc:
            logger.error("HTTP POST failed (transport): %s", url, exc_info=True)
            raise HttpTransportError(str(exc)) from exc

    def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
        max_bytes: int | None = None,
    ) -> requests.Response | _BufferedResponse:
        try:
            if max_bytes is not None:
                return _get_capped(url, headers, timeout, max_bytes)
            return requests.get(url, headers=headers, timeout=timeout)
        except _TRANSPORT_ERRORS as exc:
            logger.error("HTTP GET failed (transport): %s", url, exc_info=True)
            raise HttpTransportError(str(exc)) from exc

    def put(
        self,
        url: str,
        data: RequestData = None,
        headers: dict[str, str] | None = None,
        timeout: int = 300,
    ) -> requests.Response:
        try:
            return requests.put(url, data=data, headers=headers, timeout=timeout)
        except _TRANSPORT_ERRORS as exc:
            logger.error("HTTP PUT failed (transport): %s", url, exc_info=True)
            raise HttpTransportError(str(exc)) from exc
