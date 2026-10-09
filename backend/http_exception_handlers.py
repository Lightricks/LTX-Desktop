"""Shared HTTP exception handlers for desktop and remote FastAPI apps."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from _routes._errors import HTTPError, build_http_error_response
from logging_policy import log_http_error, log_unhandled_exception


async def _http_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPError):
        log_http_error(request, exc)
        return JSONResponse(status_code=exc.status_code, content=exc.response.model_dump())
    return JSONResponse(
        status_code=500,
        content=build_http_error_response(500, str(exc)).model_dump(),
    )


async def _starlette_http_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=build_http_error_response(exc.status_code, exc.detail).model_dump(),
        )
    return JSONResponse(
        status_code=500,
        content=build_http_error_response(500, str(exc)).model_dump(),
    )


async def _validation_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=build_http_error_response(422, str(exc)).model_dump(),
    )


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log_unhandled_exception(request, exc)
    return JSONResponse(
        status_code=500,
        content=build_http_error_response(500, str(exc)).model_dump(),
    )


def install_http_exception_handlers(app: FastAPI) -> None:
    """Register HTTPError, Starlette HTTP, validation, and unhandled exception handlers."""
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(HTTPError, _http_error_handler)
    app.add_exception_handler(StarletteHTTPException, _starlette_http_error_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
