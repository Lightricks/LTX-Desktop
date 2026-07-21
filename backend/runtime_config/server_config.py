"""Environment-backed HTTP server configuration.

This module intentionally has no torch or application imports so startup
configuration can be validated in lightweight tests.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from typing import Literal, Mapping
from urllib.parse import urlparse

from runtime_config.port_constant import PORT

DeploymentMode = Literal["managed_local", "standalone"]

DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    "null",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _parse_deployment_mode(value: str) -> DeploymentMode:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"", "managed_local"}:
        return "managed_local"
    if normalized == "standalone":
        return "standalone"
    raise RuntimeError("LTX_DEPLOYMENT_MODE must be 'managed_local' or 'standalone'")


def _parse_port(value: str) -> int:
    if not value:
        return PORT
    try:
        port = int(value)
    except ValueError:
        raise RuntimeError("LTX_PORT must be an integer") from None
    if not 1 <= port <= 65535:
        raise RuntimeError("LTX_PORT must be between 1 and 65535")
    return port


def _validate_public_base_url(value: str) -> str:
    candidate = value.rstrip("/")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path not in {"", "/"}:
        raise RuntimeError("LTX_PUBLIC_BASE_URL must be an HTTP(S) origin without a path")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RuntimeError("LTX_PUBLIC_BASE_URL must not contain credentials, a query, or a fragment")
    return candidate


@dataclass(frozen=True, slots=True)
class ServerConfig:
    deployment_mode: DeploymentMode
    bind_host: str
    port: int
    public_base_url: str
    allowed_origins: tuple[str, ...]
    auth_token: str = field(repr=False)
    admin_token: str = field(repr=False)

    @property
    def allow_legacy_path_inputs(self) -> bool:
        return self.deployment_mode == "managed_local"

    @property
    def models_dir_editable(self) -> bool:
        return self.deployment_mode == "managed_local"


def load_server_config(environ: Mapping[str, str] | None = None) -> ServerConfig:
    env = os.environ if environ is None else environ
    mode = _parse_deployment_mode(env.get("LTX_DEPLOYMENT_MODE", ""))
    port = _parse_port(env.get("LTX_PORT", ""))
    bind_host = env.get("LTX_BIND_HOST", "127.0.0.1").strip() or "127.0.0.1"
    auth_token = env.get("LTX_AUTH_TOKEN", "")
    admin_token = env.get("LTX_ADMIN_TOKEN", "")

    if mode == "standalone" and len(auth_token) < 32:
        raise RuntimeError("Standalone mode requires LTX_AUTH_TOKEN with at least 32 characters")
    if not _is_loopback_host(bind_host) and not auth_token:
        raise RuntimeError("A non-loopback LTX_BIND_HOST requires LTX_AUTH_TOKEN")

    public_base_url = _validate_public_base_url(
        env.get("LTX_PUBLIC_BASE_URL", f"http://127.0.0.1:{port}")
    )
    origins_value = env.get("LTX_ALLOWED_ORIGINS", "")
    allowed_origins = (
        tuple(origin.strip() for origin in origins_value.split(",") if origin.strip())
        if origins_value
        else DEFAULT_ALLOWED_ORIGINS
    )

    return ServerConfig(
        deployment_mode=mode,
        bind_host=bind_host,
        port=port,
        public_base_url=public_base_url,
        allowed_origins=allowed_origins,
        auth_token=auth_token,
        admin_token=admin_token,
    )
