"""Host-minted HMAC media URLs. Method-agnostic so GET/HEAD/Range share one sig."""

from __future__ import annotations

import base64
import hmac
import secrets
import time
from hashlib import sha256
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

MEDIA_TTL_S = 300
_SECRET_NAME = "remote_media_hmac.key"


def load_or_create_media_secret(app_data_dir: Path) -> bytes:
    path = app_data_dir / _SECRET_NAME
    if path.is_file():
        secret = path.read_bytes()
        if len(secret) == 32:
            return secret
    secret = secrets.token_bytes(32)
    path.write_bytes(secret)
    return secret


def bucket_exp(now_s: int | None = None) -> int:
    now = int(time.time() if now_s is None else now_s)
    return ((now // MEDIA_TTL_S) + 2) * MEDIA_TTL_S


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes | None:
    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, TypeError):
        return None


class MediaUrlSigner:
    def __init__(self, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("media HMAC secret must be at least 32 bytes")
        self._secret = secret

    def sign(self, path: str, *, asset_id: str, device_id: str, exp: int | None = None) -> str:
        expiry = bucket_exp() if exp is None else exp
        digest = hmac.new(
            self._secret,
            _payload(path, expiry, device_id, asset_id),
            sha256,
        ).digest()
        query = urlencode({"exp": expiry, "did": device_id, "sig": _b64(digest)})
        return f"{path}?{query}"

    def bytes_url(self, asset_id: str, device_id: str) -> str:
        return self.sign(
            f"/api/assets/{asset_id}/bytes",
            asset_id=asset_id,
            device_id=device_id,
        )

    def thumbnail_url(self, asset_id: str, device_id: str) -> str:
        return self.sign(
            f"/api/assets/{asset_id}/thumbnail/bytes",
            asset_id=asset_id,
            device_id=device_id,
        )

    def verify(
        self,
        path: str,
        query: str,
        *,
        asset_id: str,
        now_s: int | None = None,
    ) -> str | None:
        params = parse_qs(query, keep_blank_values=True)
        exp_raw = _one(params, "exp")
        device_id = _one(params, "did")
        sig_raw = _one(params, "sig")
        if exp_raw is None or device_id is None or sig_raw is None:
            return None
        try:
            expiry = int(exp_raw)
        except ValueError:
            return None
        now = int(time.time() if now_s is None else now_s)
        if expiry <= now:
            return None
        provided = _b64decode(sig_raw)
        if provided is None:
            return None
        expected = hmac.new(
            self._secret,
            _payload(path, expiry, device_id, asset_id),
            sha256,
        ).digest()
        if not hmac.compare_digest(provided, expected):
            return None
        return device_id


def path_without_query(url: str) -> str:
    return urlsplit(url).path


def _payload(path: str, exp: int, device_id: str, asset_id: str) -> bytes:
    return f"{path}\n{exp}\n{device_id}\n{asset_id}".encode("utf-8")


def _one(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    if values is None or len(values) != 1 or values[0] == "":
        return None
    return values[0]
