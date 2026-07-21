"""Standalone HTTP runtime configuration tests."""

from __future__ import annotations

import pytest

from runtime_config.server_config import load_server_config


def test_managed_local_defaults() -> None:
    config = load_server_config({})
    assert config.deployment_mode == "managed_local"
    assert config.bind_host == "127.0.0.1"
    assert config.allow_legacy_path_inputs is True
    assert "null" in config.allowed_origins


def test_standalone_requires_long_auth_token() -> None:
    with pytest.raises(RuntimeError, match="requires LTX_AUTH_TOKEN"):
        load_server_config({"LTX_DEPLOYMENT_MODE": "standalone", "LTX_AUTH_TOKEN": "short"})


def test_standalone_configuration() -> None:
    config = load_server_config(
        {
            "LTX_DEPLOYMENT_MODE": "standalone",
            "LTX_AUTH_TOKEN": "x" * 32,
            "LTX_PORT": "8123",
            "LTX_PUBLIC_BASE_URL": "http://127.0.0.1:8123",
            "LTX_ALLOWED_ORIGINS": "null,http://localhost:5173",
        }
    )
    assert config.deployment_mode == "standalone"
    assert config.port == 8123
    assert config.public_base_url == "http://127.0.0.1:8123"
    assert config.allow_legacy_path_inputs is False
    assert config.models_dir_editable is False


def test_non_loopback_bind_requires_authentication() -> None:
    with pytest.raises(RuntimeError, match="requires LTX_AUTH_TOKEN"):
        load_server_config({"LTX_BIND_HOST": "0.0.0.0"})


def test_authenticated_standalone_allows_plain_http_on_a_trusted_lan() -> None:
    config = load_server_config(
        {
            "LTX_DEPLOYMENT_MODE": "standalone",
            "LTX_AUTH_TOKEN": "x" * 32,
            "LTX_BIND_HOST": "0.0.0.0",
            "LTX_PUBLIC_BASE_URL": "http://192.168.1.50:8000",
        }
    )

    assert config.bind_host == "0.0.0.0"
    assert config.public_base_url == "http://192.168.1.50:8000"
