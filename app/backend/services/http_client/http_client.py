"""HTTP client service protocol definitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from services.services_utils import JSONValue, RequestData


class HttpTransportError(Exception):
    """Raised by HTTP service implementations on a transport-level failure (timeout,
    connection reset, or broken chunked stream) — the request didn't complete."""


class HttpBodyTooLargeError(Exception):
    """Raised by a capped GET when the response body is larger than the allowed size."""


class HttpRedirectRefusedError(Exception):
    """Raised by a capped GET when a redirect changes the scheme to an unsafe one or loops (CWE-918)."""


class HttpResponseLike(Protocol):
    @property
    def status_code(self) -> int:
        ...

    @property
    def text(self) -> str:
        ...

    @property
    def headers(self) -> Mapping[str, str]:
        ...

    @property
    def content(self) -> bytes:
        ...

    def json(self) -> object:
        ...


class HTTPClient(Protocol):
    def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        json_payload: Mapping[str, JSONValue] | None = None,
        data: RequestData = None,
        timeout: int = 30,
    ) -> HttpResponseLike:
        ...

    def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
        max_bytes: int | None = None,
    ) -> HttpResponseLike:
        """With max_bytes, a body larger than that raises HttpBodyTooLargeError (CWE-770).

        A capped GET follows at most 3 redirects, never from https to another scheme, sends
        ``headers`` only to the first host, and gives up when ``timeout`` seconds pass in total.
        """
        ...

    def put(
        self,
        url: str,
        data: RequestData = None,
        headers: dict[str, str] | None = None,
        timeout: int = 300,
    ) -> HttpResponseLike:
        ...
