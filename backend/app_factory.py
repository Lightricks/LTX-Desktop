"""FastAPI app factory decoupled from runtime bootstrap side effects."""

from __future__ import annotations

import base64
import hmac
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import Response as StarletteResponse

from _routes._errors import build_http_error_response
from _routes.generation import router as generation_router
from _routes.generation import specs_router as generate_specs_router
from _routes.hf_auth import router as hf_auth_router
from _routes.health import router as health_router
from _routes.ic_lora import router as ic_lora_router
from _routes.lora_catalog import router as lora_catalog_router
from _routes.image_gen import router as image_gen_router
from _routes.prompt_enhancement import router as prompt_enhancement_router
from _routes.models import router as models_router
from _routes.suggest_gap_prompt import router as suggest_gap_prompt_router
from _routes.retake import router as retake_router
from _routes.extend import router as extend_router
from _routes.assets import router as assets_router
from _routes.generations import router as generations_router
from _routes.stats import router as stats_router
from _routes.generation_seed import router as generation_seed_router
from _routes.runtime_policy import router as runtime_policy_router
from _routes.feature_flags import router as feature_flags_router
from _routes.settings import router as settings_router
from _routes.remote import router as remote_status_router
from api_types import HTTPErrorResponse
from http_exception_handlers import install_http_exception_handlers
from remote.controller import RemoteExposureController, remote_controller_from_app
from state import get_state_service, init_state_service

if TYPE_CHECKING:
    from app_handler import AppHandler

DEFAULT_ALLOWED_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

DEFAULT_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    "4XX": {
        "model": HTTPErrorResponse,
        "description": "Client Error",
    },
    "5XX": {
        "model": HTTPErrorResponse,
        "description": "Server Error",
    },
}


def create_app(
    *,
    handler: "AppHandler",
    allowed_origins: list[str] | None = None,
    title: str = "LTX-2 Video Generation Server",
    auth_token: str = "",
    admin_token: str = "",
) -> FastAPI:
    """Create a configured FastAPI app bound to the provided handler."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        handler.generation_queue.start()
        controller = remote_controller_from_app(app)
        if controller is None:
            raise RuntimeError("Desktop app is missing RemoteExposureController")
        controller.sync()
        try:
            yield
        finally:
            controller.stop()
            handler.generation_queue.stop()

    app = FastAPI(title=title, responses=DEFAULT_ERROR_RESPONSES, lifespan=lifespan)
    app.state.admin_token = admin_token  # type: ignore[attr-defined]
    app.state.remote_controller = RemoteExposureController(handler)  # type: ignore[attr-defined]
    app.dependency_overrides[get_state_service] = lambda: handler
    init_state_service(handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or DEFAULT_ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _auth_middleware(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        call_next: Callable[[Request], Awaitable[StarletteResponse]],
    ) -> StarletteResponse:
        if not auth_token:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path == "/api/auth/huggingface/callback":
            return await call_next(request)
        def _token_matches(candidate: str) -> bool:
            return hmac.compare_digest(candidate, auth_token)

        # WebSocket: check query param
        if request.headers.get("upgrade", "").lower() == "websocket":
            if _token_matches(request.query_params.get("token", "")):
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content=build_http_error_response(401, "Unauthorized").model_dump(),
            )
        # HTTP: Bearer or Basic auth
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer ") and _token_matches(auth_header[7:]):
            return await call_next(request)
        if auth_header.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth_header[6:]).decode()
                _, _, password = decoded.partition(":")
                if _token_matches(password):
                    return await call_next(request)
            except Exception:
                pass
        return JSONResponse(
            status_code=401,
            content=build_http_error_response(401, "Unauthorized").model_dump(),
        )

    install_http_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(generation_router)
    app.include_router(generate_specs_router)
    app.include_router(models_router)
    app.include_router(settings_router)
    app.include_router(image_gen_router)
    app.include_router(suggest_gap_prompt_router)
    app.include_router(retake_router)
    app.include_router(extend_router)
    app.include_router(assets_router)
    app.include_router(generations_router)
    app.include_router(stats_router)
    app.include_router(generation_seed_router)
    app.include_router(ic_lora_router)
    app.include_router(lora_catalog_router)
    app.include_router(prompt_enhancement_router)
    app.include_router(runtime_policy_router)
    app.include_router(feature_flags_router)
    app.include_router(hf_auth_router)
    app.include_router(remote_status_router)

    return app
