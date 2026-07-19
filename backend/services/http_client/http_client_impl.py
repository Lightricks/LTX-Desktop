"""HTTP client wrapper service."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import requests

from services.http_client.http_client import HttpTransportError
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
    ) -> requests.Response:
        try:
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
