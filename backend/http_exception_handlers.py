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
    if isinstance(exc, RequestValidationError):
        errors = exc.errors()
        message = f"{len(errors)} validation error{'s' if len(errors) != 1 else ''}:\n"
        for err in errors:
            message += f"  {err}\n"
        message = message.rstrip()
    else:
        message = str(exc)
    return JSONResponse(
        status_code=422,
        content=build_http_error_response(422, message).model_dump(),
    )


# Exception text can carry Desktop file paths, which must never reach a remote device.
UNHANDLED_REMOTE_MESSAGE = "Something went wrong on the Desktop. Try again."


def install_http_exception_handlers(app: FastAPI, *, expose_unhandled_details: bool = True) -> None:
    """Register HTTPError, Starlette HTTP, validation, and unhandled exception handlers."""

    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log_unhandled_exception(request, exc)
        message = str(exc) if expose_unhandled_details else UNHANDLED_REMOTE_MESSAGE
        return JSONResponse(status_code=500, content=build_http_error_response(500, message).model_dump())

    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(HTTPError, _http_error_handler)
    app.add_exception_handler(StarletteHTTPException, _starlette_http_error_handler)
    app.add_exception_handler(Exception, unhandled)
