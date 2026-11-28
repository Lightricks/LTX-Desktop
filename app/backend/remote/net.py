"""Env helpers and pairing URLs for the isolated remote server."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from runtime_config.port_constant import REMOTE_PORT

ENV_PORT = "LTX_REMOTE_PORT"
ENV_CLIENT_DIR = "LTX_REMOTE_CLIENT_DIR"
ENV_AUTH = "LTX_AUTH_TOKEN"
PAIRING_PATH = "/pairing"


def remote_port() -> int:
    """Configured listen port.

    ``LTX_REMOTE_PORT=0`` is a test hook so the kernel assigns an ephemeral port.
    It is not a supported app setting. Unset or invalid values use ``REMOTE_PORT``.
    """
    raw = os.environ.get(ENV_PORT, "").strip()
    if not raw:
        return REMOTE_PORT
    try:
        parsed = int(raw)
    except ValueError:
        return REMOTE_PORT
    if parsed < 0 or parsed > 65535:
        return REMOTE_PORT
    return parsed


def bound_server_port(server: object, fallback: int) -> int:
    """Port uvicorn actually bound, including kernel-assigned ``port=0``."""
    for asyncio_server in getattr(server, "servers", ()) or ():
        for sock in getattr(asyncio_server, "sockets", None) or ():
            try:
                host_port = cast(tuple[object, ...], sock.getsockname())
            except OSError:
                continue
            if len(host_port) < 2:
                continue
            port_raw = host_port[1]
            if isinstance(port_raw, int) and port_raw > 0:
                return port_raw
    return fallback


def resolve_client_dir() -> Path | None:
    env_dir = os.environ.get(ENV_CLIENT_DIR, "").strip()
    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path(__file__).resolve().parents[2] / "dist-remote")
    for path in candidates:
        if path.is_dir() and (path / "index.html").is_file():
            return path
    return None


def lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("1.1.1.1", 80))
        ip = sock.getsockname()[0]
        if isinstance(ip, str) and ip:
            return ip
    except OSError:
        pass
    finally:
        sock.close()
    return "127.0.0.1"


def origin_url(url: str) -> str:
    """Strip query/fragment so logs never include the pairing grant."""
    parts = urlsplit(url)
    path = parts.path if parts.path else "/"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def attach_token(base_url: str, token: str) -> str:
    """Put the one-time grant in `?t=` so QR codes and Open keep it.

    Fragments (`#t=`) are dropped by many phone scanners and by macOS
    `open` / Electron `openExternal`. The remote server disables access
    logs; `origin_url` still redacts query from our own logs.
    """
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["t"] = token
    path = parts.path if parts.path else "/"
    return urlunsplit((parts.scheme, parts.netloc, path, urlencode(query), parts.fragment))


def client_url(host: str, port: int, token: str, *, scheme: str = "http") -> str:
    if scheme == "https" and port == 443:
        return attach_token(f"https://{host}{PAIRING_PATH}", token)
    if scheme == "http" and port == 80:
        return attach_token(f"http://{host}{PAIRING_PATH}", token)
    return attach_token(f"{scheme}://{host}:{port}{PAIRING_PATH}", token)


def advertised_lan_urls(port: int, token: str) -> tuple[str, str]:
    """LAN and loopback pairing URLs, both carrying the generated token."""
    lan = client_url(lan_ip(), port, token)
    local = client_url("127.0.0.1", port, token)
    return lan, local
