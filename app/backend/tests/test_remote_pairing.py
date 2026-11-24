from __future__ import annotations

import pytest

from remote.pairing import (
    PairingError,
    RemotePairing,
    _MAX_USER_AGENT_LEN,
    _public_ip,
    _public_user_agent,
    device_name_from_user_agent,
    hash_session_token,
)
from services.records import PairedDeviceRecord, UnavailableError


def test_device_name_from_user_agent_summarizes_browser_and_device() -> None:
    assert device_name_from_user_agent(None) == "Paired device"
    assert device_name_from_user_agent("Phone UA") == "Phone UA"
    chrome_mac = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    safari_iphone = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    )
    assert device_name_from_user_agent(chrome_mac) == "Chrome on Mac"
    assert device_name_from_user_agent(safari_iphone) == "Safari on iPhone"


def test_list_public_devices_includes_ip() -> None:
    pairing = RemotePairing.for_known_session("seed")
    pairing.exchange(pairing.current_grant(), ip="192.168.1.4", user_agent="Phone")
    listed = [item for item in pairing.list_public_devices() if item.ip == "192.168.1.4"]
    assert len(listed) == 1


def test_list_public_devices_uses_full_user_agent_for_display_name() -> None:
    pairing = RemotePairing.for_known_session("seed")
    chrome_mac = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    pairing.exchange(pairing.current_grant(), ip="10.0.0.9", user_agent=chrome_mac)
    match = next(item for item in pairing.list_public_devices() if item.ip == "10.0.0.9")
    assert match.name == "Chrome on Mac"


def test_authenticate_backfills_ip_and_browser_label() -> None:
    pairing = RemotePairing.for_known_session("seed")
    chrome_mac = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    pairing.authenticate_bearer("seed", ip="10.0.0.4", user_agent=chrome_mac)
    listed = next(item for item in pairing.list_public_devices() if item.name == "Chrome on Mac")
    assert listed.ip == "10.0.0.4"


def test_public_ip_accepts_addresses_and_rejects_junk() -> None:
    assert _public_ip(" 10.0.0.4 ") == "10.0.0.4"
    assert _public_ip("::1") == "::1"
    assert _public_ip("[::1]") == "::1"
    assert _public_ip(None) is None
    assert _public_ip("") is None
    assert _public_ip("unknown") is None
    assert _public_ip("UNKNOWN") is None
    assert _public_ip("testclient") is None
    assert _public_ip("not an ip") is None


def test_public_user_agent_strips_and_caps() -> None:
    assert _public_user_agent(None) is None
    assert _public_user_agent("  \n  ") is None
    assert _public_user_agent("Phone UA") == "Phone UA"
    assert _public_user_agent("x" * 800) == "x" * _MAX_USER_AGENT_LEN


def test_unknown_first_seen_ip_is_replaced_on_backfill() -> None:
    pairing = RemotePairing.for_known_session("seed")
    pairing._store.insert_paired_device(
        PairedDeviceRecord(
            id="legacy",
            name="Old phone",
            token_hash=hash_session_token("legacy"),
            created_at=1,
            last_seen_at=1,
            first_seen_ip="unknown",
        )
    )
    pairing.authenticate_bearer("legacy", ip="10.0.0.4", user_agent="Phone")
    listed = next(item for item in pairing.list_public_devices() if item.id == "legacy")
    assert listed.ip == "10.0.0.4"


def test_exchange_reuses_session_and_backfills_missing_ip() -> None:
    pairing = RemotePairing.for_known_session("seed")
    chrome_mac = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    first = pairing.exchange(pairing.current_grant(), ip="unknown", user_agent=None)
    assert next(item for item in pairing.list_public_devices() if item.id == first.device_id).ip is None
    second = pairing.exchange(
        pairing.current_grant(),
        ip="10.0.0.4",
        user_agent=chrome_mac,
        existing_token=first.token,
    )
    assert second.device_id == first.device_id
    listed = next(item for item in pairing.list_public_devices() if item.id == first.device_id)
    assert listed.ip == "10.0.0.4"
    assert listed.name == "Chrome on Mac"


def test_exchange_caps_stored_user_agent() -> None:
    pairing = RemotePairing.for_known_session("seed")
    pairing.exchange(pairing.current_grant(), ip="10.0.0.9", user_agent="x" * 800)
    stored = next(
        item for item in pairing._store.list_paired_devices() if item.first_seen_user_agent
    )
    assert stored.first_seen_user_agent == "x" * _MAX_USER_AGENT_LEN


def test_exchange_consumes_grant_and_authenticates_bearer() -> None:
    pairing = RemotePairing.for_known_session("seed")
    grant = pairing.current_grant()
    exchanged = pairing.exchange(grant, ip="192.168.1.4", user_agent="Phone UA")
    assert exchanged.token != "seed"
    assert exchanged.name == "Phone UA"
    with pytest.raises(PairingError) as raised:
        pairing.exchange(grant, ip="192.168.1.4", user_agent="Phone UA")
    assert raised.value.status_code == 401
    assert pairing.authenticate_bearer(exchanged.token) is not None
    assert pairing.authenticate_bearer(exchanged.token).id == exchanged.device_id
    assert pairing.authenticate_bearer("seed") is not None


def test_exchange_rotates_grant_for_a_second_device() -> None:
    pairing = RemotePairing.for_known_session("seed")
    first = pairing.exchange(pairing.current_grant(), ip="10.0.0.2", user_agent="A")
    second = pairing.exchange(pairing.current_grant(), ip="10.0.0.3", user_agent="B")
    assert first.device_id != second.device_id
    assert pairing.authenticate_bearer(first.token) is not None
    assert pairing.authenticate_bearer(second.token) is not None


def test_exchange_with_existing_session_reuses_same_device() -> None:
    pairing = RemotePairing.for_known_session("seed")
    first = pairing.exchange(pairing.current_grant(), ip="10.0.0.2", user_agent="Phone")
    second = pairing.exchange(
        pairing.current_grant(),
        ip="10.0.0.4",
        user_agent="Phone v2",
        existing_token=first.token,
    )
    assert second.device_id == first.device_id
    assert second.token == first.token
    assert second.name == "Phone"
    authed = pairing.authenticate_bearer(first.token)
    assert authed is not None
    assert authed.id == first.device_id
    active_ids = [item.id for item in pairing.list_public_devices() if item.revoked_at is None]
    assert active_ids.count(first.device_id) == 1
    assert len(active_ids) == 2


def test_exchange_does_not_insert_when_session_lookup_fails() -> None:
    pairing = RemotePairing.for_known_session("seed")
    first = pairing.exchange(pairing.current_grant(), ip="10.0.0.2", user_agent="Phone")
    before = [item.id for item in pairing.list_public_devices()]
    store = pairing._store
    original = store.get_paired_device_by_token_hash

    def boom(_token_hash: str):
        raise UnavailableError()

    store.get_paired_device_by_token_hash = boom  # type: ignore[method-assign]
    grant = pairing.current_grant()
    try:
        with pytest.raises(PairingError) as raised:
            pairing.exchange(
                grant,
                ip="10.0.0.2",
                user_agent="Phone",
                existing_token=first.token,
            )
        assert raised.value.status_code == 503
        assert raised.value.code == "STORE_UNAVAILABLE"
    finally:
        store.get_paired_device_by_token_hash = original  # type: ignore[method-assign]
    assert [item.id for item in pairing.list_public_devices()] == before


def test_exchange_ignores_revoked_session_and_inserts() -> None:
    pairing = RemotePairing.for_known_session("seed")
    first = pairing.exchange(pairing.current_grant(), ip="10.0.0.2", user_agent="Phone")
    assert pairing.revoke(first.device_id) is True
    second = pairing.exchange(
        pairing.current_grant(),
        ip="10.0.0.2",
        user_agent="Phone",
        existing_token=first.token,
    )
    assert second.device_id != first.device_id
    assert pairing.authenticate_bearer(first.token) is None
    assert pairing.authenticate_bearer(second.token) is not None


def test_revoke_rejects_bearer_and_media() -> None:
    pairing = RemotePairing.for_known_session("seed")
    device = pairing.authenticate_bearer("seed")
    assert device is not None
    url = pairing.signer.bytes_url("asset-1", device.id)
    path, _, query = url.partition("?")
    assert pairing.authenticate_media(path, query, asset_id="asset-1") is not None
    assert pairing.revoke(device.id) is True
    assert pairing.authenticate_bearer("seed") is None
    assert pairing.authenticate_media(path, query, asset_id="asset-1") is None


def test_rate_limit_lockout_on_failed_exchange() -> None:
    pairing = RemotePairing.for_known_session("seed")
    for _ in range(8):
        with pytest.raises(PairingError) as raised:
            pairing.exchange("nope", ip="192.168.0.9", user_agent=None)
        assert raised.value.status_code == 401
    with pytest.raises(PairingError) as raised:
        pairing.exchange("nope", ip="192.168.0.9", user_agent=None)
    assert raised.value.status_code == 429
    pairing.exchange(pairing.current_grant(), ip="10.0.0.8", user_agent=None)


def test_bearer_auth_store_outage_is_unavailable() -> None:
    pairing = RemotePairing.for_known_session("seed")

    def boom(_token: str):
        raise UnavailableError()

    pairing._lookup_active_session = boom  # type: ignore[method-assign]
    with pytest.raises(PairingError) as raised:
        pairing.authenticate_bearer("seed")
    assert raised.value.status_code == 503
    assert raised.value.code == "STORE_UNAVAILABLE"
    assert pairing.list_public_devices()[0].name == "Test device"


def test_list_and_revoke_store_outage_raises() -> None:
    pairing = RemotePairing.for_known_session("seed")

    def boom_list():
        raise UnavailableError()

    def boom_revoke(_device_id: str, *, now_ms: int):
        raise UnavailableError()

    pairing._store.list_paired_devices = boom_list  # type: ignore[method-assign]
    with pytest.raises(UnavailableError):
        pairing.list_public_devices()
    pairing._store.revoke_paired_device = boom_revoke  # type: ignore[method-assign]
    with pytest.raises(UnavailableError):
        pairing.revoke("any")


def test_media_auth_store_outage_is_unavailable() -> None:
    pairing = RemotePairing.for_known_session("seed")
    device = pairing.authenticate_bearer("seed")
    assert device is not None
    url = pairing.signer.bytes_url("asset-1", device.id)
    path, _, query = url.partition("?")

    def boom(_device_id: str):
        raise UnavailableError()

    pairing._store.get_paired_device = boom  # type: ignore[method-assign]
    with pytest.raises(PairingError) as raised:
        pairing.authenticate_media(path, query, asset_id="asset-1")
    assert raised.value.status_code == 503
    assert raised.value.code == "STORE_UNAVAILABLE"


def test_bearer_auth_throttles_last_seen_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"ms": 1_000_000}
    monkeypatch.setattr("remote.pairing._now_ms", lambda: clock["ms"])
    pairing = RemotePairing.for_known_session("seed")
    assert pairing.authenticate_bearer("seed") is not None
    assert pairing.list_public_devices()[0].last_seen_at == 1_000_000
    clock["ms"] = 1_000_000 + 59_999
    pairing.authenticate_bearer("seed")
    assert pairing.list_public_devices()[0].last_seen_at == 1_000_000
    clock["ms"] = 1_000_000 + 60_000
    pairing.authenticate_bearer("seed")
    assert pairing.list_public_devices()[0].last_seen_at == 1_060_000


def test_session_hash_is_lookup_key() -> None:
    pairing = RemotePairing.for_known_session("seed")
    listed = pairing.list_public_devices()
    assert listed[0].name == "Test device"
    assert hash_session_token("seed") != "seed"
    assert all(
        "token" not in item.model_dump() and "mac_key" not in item.model_dump()
        for item in listed
    )
