"""Map LTX API auth failures onto one error the generation dialog can act on."""

from __future__ import annotations

from _routes._errors import HTTPError
from services.ltx_api_client.ltx_api_client import INSUFFICIENT_FUNDS_ERROR_TYPE, LTXAPIClientError

LTX_INVALID_API_KEY = "LTX_INVALID_API_KEY"
LTX_API_KEY_MISSING = "LTX_API_KEY_MISSING"
LTX_API_PROMPT_EMBEDDING_FAILED = "LTX_API_PROMPT_EMBEDDING_FAILED"

LTX_INVALID_API_KEY_MESSAGE = "This LTX API key isn’t valid."
LTX_API_KEY_MISSING_MESSAGE = "Add an LTX API key to generate."
LTX_API_PROMPT_EMBEDDING_FAILED_MESSAGE = "LTX API text encoding failed."
_LTX_INSUFFICIENT_FUNDS_MESSAGE = (
    "Your LTX API credits are insufficient for this generation. Buy more credits and try again."
)

_AUTH_REJECTED = frozenset({401, 403})


def missing_ltx_api_key_error() -> HTTPError:
    return HTTPError(400, LTX_API_KEY_MISSING_MESSAGE, code=LTX_API_KEY_MISSING)


def map_ltx_api_client_error(exc: LTXAPIClientError) -> HTTPError:
    """Turn a provider failure into the error the app shows.

    402 insufficient funds keeps its own code (Buy Credits). 401 and 403 share
    one code so the dialog can open Settings on the LTX key. Everything else
    stays the provider's status and detail.
    """
    if exc.status_code == 402 and exc.provider_error_type == INSUFFICIENT_FUNDS_ERROR_TYPE:
        return HTTPError(402, _LTX_INSUFFICIENT_FUNDS_MESSAGE, code="LTX_INSUFFICIENT_FUNDS")
    if exc.status_code in _AUTH_REJECTED:
        return HTTPError(exc.status_code, LTX_INVALID_API_KEY_MESSAGE, code=LTX_INVALID_API_KEY)
    return HTTPError(exc.status_code, exc.detail)
