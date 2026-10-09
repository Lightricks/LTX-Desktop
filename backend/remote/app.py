"""Dark remote FastAPI app. Not mounted on create_app; served on its own port."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response as StarletteResponse

from _routes._errors import HTTPError, build_http_error_response
from _routes.assets import phone_assets_router
from _routes.feature_flags import router as feature_flags_router
from _routes.generation_seed import phone_generation_seed_router
from _routes.stats import phone_stats_router
from _routes.generation import specs_router as generate_specs_router
from api_types import HTTPErrorResponse
from app_handler import AppHandler
from http_exception_handlers import install_http_exception_handlers
from remote.generations import router as generations_router
from remote.hf_auth import router as hf_auth_router
from remote.loras import router as loras_router
from remote.net import lan_ip, remote_port
from remote.pairing import ExchangeRequest, PairingError, RemotePairing
from remote.prompt_enhancement import router as prompt_enhancement_router
from remote.redact import RedactingJSONResponse
from state import get_state_service

_BYTES_PATH = re.compile(r"^/api/assets/([^/]+)/(?:thumbnail/)?bytes$")
_EXCHANGE_PATH = "/api/pairing/exchange"

FALLBACK_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>LTX Remote</title></head>
<body>
  <h1>Remote client is not built</h1>
  <p>The API is up. From the repo root run <code>pnpm build:remote</code>, then toggle Remote on again.</p>
</body>
</html>
"""

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    "4XX": {"model": HTTPErrorResponse, "description": "Client Error"},
    "5XX": {"model": HTTPErrorResponse, "description": "Server Error"},
}


class _RemoteHealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"


def _is_public_path(path: str) -> bool:
    if path == "/health" or path == _EXCHANGE_PATH:
        return True
    return not path.startswith("/api/")


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content=build_http_error_response(401, "Unauthorized").model_dump(),
    )


def _forbidden() -> JSONResponse:
    return JSONResponse(
        status_code=403,
        content=build_http_error_response(403, "Forbidden").model_dump(),
    )


def _unavailable() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=build_http_error_response(503, "STORE_UNAVAILABLE").model_dump(),
    )


def _peer_ip(request: Request) -> str | None:
    if request.client is None:
        return None
    return request.client.host


def _peer_host(request: Request) -> str:
    return _peer_ip(request) or "unknown"


def _bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


class SpaStaticFiles(StaticFiles):
    """Serve index.html for unknown client routes. Never mask missing /api paths."""

    async def get_response(self, path: str, scope: Any) -> StarletteResponse:
        posix_path = path.replace("\\", "/")
        if posix_path == "api" or posix_path.startswith("api/"):
            raise StarletteHTTPException(status_code=404)
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and self.html:
                return await super().get_response("index.html", scope)
            raise


def create_remote_app(
    *,
    handler: AppHandler,
    pairing: RemotePairing | None = None,
    remote_token: str | None = None,
    client_dir: Path | None = None,
) -> FastAPI:
    """Remote surface: Bearer sessions on /api, signed query on media bytes."""
    if pairing is None:
        if remote_token is None or remote_token == "":
            raise ValueError("pairing or remote_token is required")
        pairing = RemotePairing.for_known_session(remote_token)
    app = FastAPI(
        title="LTX Remote",
        responses=_ERROR_RESPONSES,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        default_response_class=RedactingJSONResponse,
    )
    app.state.remote_pairing = pairing
    app.dependency_overrides[get_state_service] = lambda: handler
    port = remote_port()
    lan = lan_ip()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
            f"http://{lan}:{port}",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "testserver",
            "localhost",
            "127.0.0.1",
            lan,
        ],
    )

    @app.middleware("http")
    async def _remote_auth_middleware(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        call_next: Callable[[Request], Awaitable[StarletteResponse]],
    ) -> StarletteResponse:
        if request.method == "OPTIONS":
            return await call_next(request)
        if _is_public_path(request.url.path):
            return await call_next(request)
        bytes_match = _BYTES_PATH.match(request.url.path)
        if bytes_match is not None:
            if request.query_params.get("t") is not None:
                return _unauthorized()
            try:
                device = pairing.authenticate_media(
                    request.url.path,
                    urlsplit(str(request.url)).query,
                    asset_id=bytes_match.group(1),
                )
            except PairingError as exc:
                if exc.status_code == 503:
                    return _unavailable()
                raise
            if device is None:
                if request.query_params.get("sig") is None:
                    return _unauthorized()
                return _forbidden()
            request.state.paired_device = device
            return await call_next(request)
        provided = _bearer_token(request)
        if provided is None:
            return _unauthorized()
        try:
            device = pairing.authenticate_bearer(
                provided,
                ip=_peer_ip(request),
                user_agent=request.headers.get("user-agent"),
            )
        except PairingError as exc:
            if exc.status_code == 503:
                return _unavailable()
            raise
        if device is None:
            return _unauthorized()
        request.state.paired_device = device
        return await call_next(request)

    install_http_exception_handlers(app)

    @app.get("/health", response_model=_RemoteHealthResponse)
    def route_health() -> _RemoteHealthResponse:  # pyright: ignore[reportUnusedFunction]
        return _RemoteHealthResponse()

    @app.get("/api/session", response_model=_RemoteHealthResponse)
    def route_session() -> _RemoteHealthResponse:  # pyright: ignore[reportUnusedFunction]
        return _RemoteHealthResponse()

    @app.post("/api/pairing/exchange")
    def route_pairing_exchange(  # pyright: ignore[reportUnusedFunction]
        body: ExchangeRequest, request: Request
    ) -> Any:
        client_host = _peer_host(request)
        try:
            return pairing.exchange(
                body.code,
                ip=client_host,
                user_agent=request.headers.get("user-agent"),
                existing_token=_bearer_token(request),
            )
        except PairingError as exc:
            raise HTTPError(exc.status_code, exc.code) from exc

    app.include_router(generations_router)
    app.include_router(phone_stats_router)
    app.include_router(phone_generation_seed_router)
    app.include_router(generate_specs_router)
    app.include_router(phone_assets_router)
    app.include_router(loras_router)
    app.include_router(hf_auth_router)
    app.include_router(feature_flags_router)
    app.include_router(prompt_enhancement_router)

    if client_dir is not None and client_dir.is_dir() and (client_dir / "index.html").is_file():
        app.mount("/", SpaStaticFiles(directory=str(client_dir), html=True), name="remote-client")
    else:

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def _missing_client() -> HTMLResponse:  # pyright: ignore[reportUnusedFunction]
            return HTMLResponse(FALLBACK_HTML)

    return app
