"""One-time QR grant exchange and hashed device sessions."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from remote.media_urls import MediaUrlSigner, load_or_create_media_secret
from services.records import PairedDeviceRecord, UnavailableError
from services.sqlite_store import SqliteStore

GRANT_TTL_S = 300.0
_MAX_EXCHANGE_FAILURES = 8
_FAILURE_WINDOW_S = 60.0
_TOUCH_MIN_INTERVAL_MS = 60_000
_MAX_USER_AGENT_LEN = 512
_DEVICE_NAME_RE = re.compile(r"[\x00-\x1f]+")


class PairingError(Exception):
    def __init__(self, status_code: int, code: str) -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(code)


class PairedDevicePublic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    created_at: int
    last_seen_at: int
    ip: str | None = None
    revoked_at: int | None = None


class ExchangeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: str
    token: str
    name: str


class ExchangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=128)


class PairedDeviceRepository(Protocol):
    def insert_paired_device(self, record: PairedDeviceRecord) -> PairedDeviceRecord: ...
    def get_paired_device_by_token_hash(
        self, token_hash: str
    ) -> PairedDeviceRecord | None: ...
    def get_paired_device(self, device_id: str) -> PairedDeviceRecord | None: ...
    def list_paired_devices(self) -> list[PairedDeviceRecord]: ...
    def revoke_paired_device(self, device_id: str, *, now_ms: int) -> bool: ...
    def touch_paired_device(
        self,
        device_id: str,
        *,
        now_ms: int | None = None,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> None: ...


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def device_name_from_user_agent(user_agent: str | None) -> str:
    if user_agent is None:
        return "Paired device"
    cleaned = _DEVICE_NAME_RE.sub(" ", user_agent).strip()
    if cleaned == "":
        return "Paired device"
    if not (cleaned.startswith("Mozilla/") or "AppleWebKit" in cleaned):
        return cleaned[:80]
    browser = _browser_from_user_agent(cleaned)
    platform = _platform_from_user_agent(cleaned)
    if browser and platform:
        return f"{browser} on {platform}"
    if browser:
        return browser
    if platform:
        return platform
    return "Paired device"


def _platform_from_user_agent(ua: str) -> str | None:
    if re.search(r"\biPhone\b", ua, re.I):
        return "iPhone"
    if re.search(r"\biPad\b", ua, re.I):
        return "iPad"
    if re.search(r"\bAndroid\b", ua, re.I):
        return "Android"
    if re.search(r"\bMacintosh\b|\bMac OS X\b", ua, re.I):
        return "Mac"
    if re.search(r"\bWindows\b", ua, re.I):
        return "Windows"
    if re.search(r"\bCrOS\b", ua, re.I):
        return "Chromebook"
    if re.search(r"\bLinux\b", ua, re.I):
        return "Linux"
    return None


def _browser_from_user_agent(ua: str) -> str | None:
    if re.search(r"\bEdg(?:e|A|iOS)?/", ua, re.I):
        return "Edge"
    if re.search(r"\bOPR/|\bOpera/", ua, re.I):
        return "Opera"
    if re.search(r"\bCriOS/", ua, re.I) or (
        re.search(r"\bChrome/", ua, re.I) and not re.search(r"\bEdg", ua, re.I)
    ):
        return "Chrome"
    if re.search(r"\bFxiOS/", ua, re.I) or re.search(r"\bFirefox/", ua, re.I):
        return "Firefox"
    if (
        re.search(r"\bSafari/", ua, re.I)
        and not re.search(r"\bChrome/", ua, re.I)
        and not re.search(r"\bCriOS/", ua, re.I)
    ):
        return "Safari"
    if re.search(r"\biPhone\b|\biPad\b", ua, re.I) and re.search(r"\bAppleWebKit\b", ua, re.I):
        return "Safari"
    return None


class MemoryPairedDeviceStore:
    def __init__(self) -> None:
        self._by_id: dict[str, PairedDeviceRecord] = {}
        self._lock = threading.Lock()

    def insert_paired_device(self, record: PairedDeviceRecord) -> PairedDeviceRecord:
        with self._lock:
            self._by_id[record.id] = record
            return record

    def get_paired_device_by_token_hash(
        self, token_hash: str
    ) -> PairedDeviceRecord | None:
        with self._lock:
            for record in self._by_id.values():
                if record.token_hash == token_hash:
                    return record
            return None

    def get_paired_device(self, device_id: str) -> PairedDeviceRecord | None:
        with self._lock:
            return self._by_id.get(device_id)

    def list_paired_devices(self) -> list[PairedDeviceRecord]:
        with self._lock:
            return sorted(
                self._by_id.values(),
                key=lambda record: (-record.created_at, record.id),
            )

    def revoke_paired_device(self, device_id: str, *, now_ms: int) -> bool:
        with self._lock:
            record = self._by_id.get(device_id)
            if record is None or record.revoked_at is not None:
                return False
            self._by_id[device_id] = record.model_copy(update={"revoked_at": now_ms})
            return True

    def touch_paired_device(
        self,
        device_id: str,
        *,
        now_ms: int | None = None,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        with self._lock:
            record = self._by_id.get(device_id)
            if record is None or record.revoked_at is not None:
                return
            updates: dict[str, object] = {}
            if now_ms is not None:
                updates["last_seen_at"] = now_ms
            cleaned_ip = _public_ip(ip)
            if cleaned_ip is not None and _public_ip(record.first_seen_ip) is None:
                updates["first_seen_ip"] = cleaned_ip
            cleaned_ua = _public_user_agent(user_agent)
            if cleaned_ua is not None and _public_user_agent(record.first_seen_user_agent) is None:
                updates["first_seen_user_agent"] = cleaned_ua
            if updates:
                self._by_id[device_id] = record.model_copy(update=updates)


class PairingGrant:
    def __init__(self, *, ttl_s: float = GRANT_TTL_S) -> None:
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._code: str | None = None
        self._expires_at = 0.0

    def issue(self) -> str:
        with self._lock:
            return self._issue_locked()

    def current(self) -> str:
        with self._lock:
            if self._code is None or time.monotonic() >= self._expires_at:
                return self._issue_locked()
            return self._code

    def consume(self, code: str) -> bool:
        with self._lock:
            current = self._code
            if (
                current is None
                or time.monotonic() >= self._expires_at
                or not hmac.compare_digest(current, code)
            ):
                return False
            self._code = None
            self._expires_at = 0.0
            return True

    def _issue_locked(self) -> str:
        self._code = secrets.token_urlsafe(32)
        self._expires_at = time.monotonic() + self._ttl_s
        return self._code


class ExchangeLimiter:
    def __init__(
        self,
        *,
        max_failures: int = _MAX_EXCHANGE_FAILURES,
        window_s: float = _FAILURE_WINDOW_S,
    ) -> None:
        self._max_failures = max_failures
        self._window_s = window_s
        self._lock = threading.Lock()
        self._failures: dict[str, list[float]] = {}

    def check(self, ip: str) -> None:
        with self._lock:
            now = time.monotonic()
            recent = [stamp for stamp in self._failures.get(ip, []) if now - stamp < self._window_s]
            self._failures[ip] = recent
            if len(recent) >= self._max_failures:
                raise PairingError(429, "PAIRING_RATE_LIMITED")

    def record_failure(self, ip: str) -> None:
        with self._lock:
            now = time.monotonic()
            recent = [stamp for stamp in self._failures.get(ip, []) if now - stamp < self._window_s]
            recent.append(now)
            self._failures[ip] = recent

    def record_success(self, ip: str) -> None:
        with self._lock:
            self._failures.pop(ip, None)


@dataclass(frozen=True)
class AuthenticatedDevice:
    id: str
    name: str


class RemotePairing:
    def __init__(
        self,
        store: PairedDeviceRepository,
        signer: MediaUrlSigner,
        *,
        grants: PairingGrant | None = None,
        limiter: ExchangeLimiter | None = None,
    ) -> None:
        self._store = store
        self.signer = signer
        self._grants = grants if grants is not None else PairingGrant()
        self._limiter = limiter if limiter is not None else ExchangeLimiter()

    @classmethod
    def from_app_data(cls, app_data_dir: Path) -> RemotePairing:
        return cls(
            SqliteStore(app_data_dir),
            MediaUrlSigner(load_or_create_media_secret(app_data_dir)),
        )

    @classmethod
    def for_known_session(cls, token: str, *, secret: bytes | None = None) -> RemotePairing:
        store = MemoryPairedDeviceStore()
        pairing = cls(
            store,
            MediaUrlSigner(secret if secret is not None else b"t" * 32),
        )
        pairing._insert_session(token, name="Test device", ip=None, user_agent=None)
        pairing.current_grant()
        return pairing

    def current_grant(self) -> str:
        return self._grants.current()

    def rotate_grant(self) -> str:
        return self._grants.issue()

    def exchange(
        self,
        code: str,
        *,
        ip: str,
        user_agent: str | None,
        existing_token: str | None = None,
    ) -> ExchangeResponse:
        self._limiter.check(ip)
        try:
            existing = (
                self._lookup_active_session(existing_token) if existing_token else None
            )
        except UnavailableError as exc:
            raise PairingError(503, "STORE_UNAVAILABLE") from exc
        if not self._grants.consume(code):
            self._limiter.record_failure(ip)
            raise PairingError(401, "PAIRING_INVALID")
        self._limiter.record_success(ip)
        if existing is not None:
            assert existing_token is not None
            self._touch(existing.id, ip=ip, user_agent=user_agent)
            self._grants.issue()
            return ExchangeResponse(
                device_id=existing.id, token=existing_token, name=existing.name
            )
        token = secrets.token_urlsafe(32)
        try:
            cleaned_ua = _public_user_agent(user_agent)
            record = self._insert_session(
                token,
                name=device_name_from_user_agent(cleaned_ua),
                ip=ip,
                user_agent=cleaned_ua,
            )
        except UnavailableError as exc:
            raise PairingError(503, "STORE_UNAVAILABLE") from exc
        self._grants.issue()
        return ExchangeResponse(device_id=record.id, token=token, name=record.name)

    def authenticate_bearer(
        self,
        token: str,
        *,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> AuthenticatedDevice | None:
        try:
            record = self._lookup_active_session(token)
        except UnavailableError as exc:
            raise PairingError(503, "STORE_UNAVAILABLE") from exc
        if record is None:
            return None
        self._touch(
            record.id,
            last_seen_at=record.last_seen_at,
            ip=None if _public_ip(record.first_seen_ip) else ip,
            user_agent=(
                None if _public_user_agent(record.first_seen_user_agent) else user_agent
            ),
        )
        return AuthenticatedDevice(id=record.id, name=record.name)

    def authenticate_media(
        self, path: str, query: str, *, asset_id: str
    ) -> AuthenticatedDevice | None:
        device_id = self.signer.verify(path, query, asset_id=asset_id)
        if device_id is None:
            return None
        try:
            record = self._store.get_paired_device(device_id)
        except UnavailableError as exc:
            raise PairingError(503, "STORE_UNAVAILABLE") from exc
        if record is None or not _is_active(record):
            return None
        return AuthenticatedDevice(id=record.id, name=record.name)

    def list_public_devices(self) -> list[PairedDevicePublic]:
        records = self._store.list_paired_devices()
        return [_to_public(record) for record in records]

    def revoke(self, device_id: str) -> bool:
        return self._store.revoke_paired_device(device_id, now_ms=_now_ms())

    def _insert_session(
        self,
        token: str,
        *,
        name: str,
        ip: str | None,
        user_agent: str | None,
    ) -> PairedDeviceRecord:
        now = _now_ms()
        record = PairedDeviceRecord(
            id=str(uuid.uuid4()),
            name=name,
            token_hash=hash_session_token(token),
            created_at=now,
            last_seen_at=now,
            first_seen_ip=_public_ip(ip),
            first_seen_user_agent=_public_user_agent(user_agent),
        )
        return self._store.insert_paired_device(record)

    def _lookup_active_session(self, token: str) -> PairedDeviceRecord | None:
        record = self._store.get_paired_device_by_token_hash(hash_session_token(token))
        if record is None or not _is_active(record):
            return None
        return record

    def _touch(
        self,
        device_id: str,
        *,
        last_seen_at: int | None = None,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        now = _now_ms()
        skip_last_seen = (
            last_seen_at is not None and now - last_seen_at < _TOUCH_MIN_INTERVAL_MS
        )
        cleaned_ip = _public_ip(ip)
        cleaned_ua = _public_user_agent(user_agent)
        if skip_last_seen and cleaned_ip is None and cleaned_ua is None:
            return
        try:
            self._store.touch_paired_device(
                device_id,
                now_ms=None if skip_last_seen else now,
                ip=cleaned_ip,
                user_agent=cleaned_ua,
            )
        except UnavailableError:
            return


def _is_active(record: PairedDeviceRecord) -> bool:
    return record.revoked_at is None


def _to_public(record: PairedDeviceRecord) -> PairedDevicePublic:
    source = record.first_seen_user_agent or record.name
    return PairedDevicePublic(
        id=record.id,
        name=device_name_from_user_agent(source),
        created_at=record.created_at,
        last_seen_at=record.last_seen_at,
        ip=_public_ip(record.first_seen_ip),
        revoked_at=record.revoked_at,
    )


def _public_ip(ip: str | None) -> str | None:
    if ip is None:
        return None
    cleaned = ip.strip()
    if cleaned == "" or cleaned.lower() == "unknown":
        return None
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    try:
        return ipaddress.ip_address(cleaned).compressed
    except ValueError:
        return None


def _public_user_agent(user_agent: str | None) -> str | None:
    if user_agent is None:
        return None
    cleaned = _DEVICE_NAME_RE.sub(" ", user_agent).strip()
    if cleaned == "":
        return None
    return cleaned[:_MAX_USER_AGENT_LEN]


def _now_ms() -> int:
    return int(time.time() * 1000)
