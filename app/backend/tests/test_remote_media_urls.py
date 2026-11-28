from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from remote.media_urls import MediaUrlSigner, bucket_exp, load_or_create_media_secret


def test_sign_and_verify_round_trip() -> None:
    signer = MediaUrlSigner(b"s" * 32)
    url = signer.bytes_url("asset-1", "dev-1")
    parts = urlsplit(url)
    assert parts.path == "/api/assets/asset-1/bytes"
    device_id = signer.verify(parts.path, parts.query, asset_id="asset-1")
    assert device_id == "dev-1"


def test_head_and_get_share_the_same_signature() -> None:
    signer = MediaUrlSigner(b"s" * 32)
    url = signer.bytes_url("asset-1", "dev-1")
    parts = urlsplit(url)
    assert signer.verify(parts.path, parts.query, asset_id="asset-1") == "dev-1"


def test_verify_rejects_expired_tampered_and_wrong_asset() -> None:
    signer = MediaUrlSigner(b"s" * 32)
    exp = bucket_exp(1_700_000_000)
    url = signer.sign(
        "/api/assets/asset-1/bytes",
        asset_id="asset-1",
        device_id="dev-1",
        exp=exp,
    )
    parts = urlsplit(url)
    assert (
        signer.verify(
            parts.path,
            parts.query,
            asset_id="asset-1",
            now_s=exp - 1,
        )
        == "dev-1"
    )
    assert (
        signer.verify(parts.path, parts.query, asset_id="asset-1", now_s=exp)
        is None
    )
    assert signer.verify(parts.path, parts.query, asset_id="asset-2") is None
    tampered = parts.query.replace("did=dev-1", "did=dev-2")
    assert signer.verify(parts.path, tampered, asset_id="asset-1") is None
    assert signer.verify(parts.path, "", asset_id="asset-1") is None


def test_same_bucket_is_stable() -> None:
    signer = MediaUrlSigner(b"s" * 32)
    exp = bucket_exp(1_700_000_100)
    first = signer.sign(
        "/api/assets/a/bytes", asset_id="a", device_id="d", exp=exp
    )
    second = signer.sign(
        "/api/assets/a/bytes", asset_id="a", device_id="d", exp=exp
    )
    assert first == second
    params = parse_qs(urlsplit(first).query)
    assert params["exp"] == [str(exp)]


def test_bucket_exp_stays_stable_and_covers_a_full_ttl() -> None:
    start = (1_700_000_000 // 300) * 300
    first = bucket_exp(start)
    assert bucket_exp(start + 299) == first
    assert bucket_exp(start + 300) != first
    assert first - start >= 300
    assert bucket_exp(start + 299) - (start + 299) >= 300


def test_load_or_create_media_secret_persists(tmp_path: Path) -> None:
    first = load_or_create_media_secret(tmp_path)
    second = load_or_create_media_secret(tmp_path)
    assert first == second
    assert len(first) == 32
