"""Security response headers shared by the local app and the remote app."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from starlette.responses import Response as StarletteResponse

SECURITY_HEADERS: dict[str, str] = {
    # Stop other pages from putting our pages in a frame (clickjacking).
    "Content-Security-Policy": "frame-ancestors 'none'",
    # Stop the browser from guessing a content type that differs from the declared one.
    "X-Content-Type-Options": "nosniff",
}


def install_security_headers(app: FastAPI) -> None:
    """Add the security headers to every response.

    Call this after all other middleware. The last middleware added is the outermost,
    so the headers also reach the 401 and 403 responses of the auth middleware.
    """

    @app.middleware("http")
    async def _security_headers_middleware(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        call_next: Callable[[Request], Awaitable[StarletteResponse]],
    ) -> StarletteResponse:
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        return response
