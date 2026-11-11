"""The real URL downloader, run against a fake HTTP transport.

The fake transport replaces the network only. Redirect handling, header building, the size cap
and the truncation check are the real code in ``UrlDownloaderImpl``.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import requests
from requests.adapters import BaseAdapter

from runtime_config.custom_lora_urls import HuggingFaceRepo
from services.url_downloader import UrlDownloadError
from services.url_downloader.public_address_adapter import public_addresses
from services.url_downloader.url_downloader_impl import _MAX_CARD_BYTES, UrlDownloaderImpl, public_only_session

_HF_FILE = "https://huggingface.co/acme/dolly/resolve/main/dolly.safetensors"
_CDN_FILE = "https://cdn.example.com/signed/dolly.safetensors"
_MAX_BYTES = 1024


class _FakeTransport(BaseAdapter):
    """Answers each URL with a canned response and keeps every request it was sent."""

    def __init__(self, routes: Mapping[str, tuple[int, Mapping[str, str], bytes]]) -> None:
        super().__init__()
        self._routes = routes
        self.sent: list[requests.PreparedRequest] = []

    def send(self, request: requests.PreparedRequest, *args: Any, **kwargs: Any) -> requests.Response:
        del args, kwargs
        self.sent.append(request)
        status, headers, body = self._routes[str(request.url)]
        response = requests.Response()
        response.status_code = status
        response.headers.update(headers)
        response.url = str(request.url)
        response.request = request
        response.raw = io.BytesIO(body)
        return response

    def close(self) -> None:
        return None


def _downloader(routes: Mapping[str, tuple[int, Mapping[str, str], bytes]]) -> tuple[UrlDownloaderImpl, _FakeTransport]:
    transport = _FakeTransport(routes)

    def session() -> requests.Session:
        made = requests.Session()
        made.mount("https://", transport)
        return made

    return UrlDownloaderImpl(session), transport


def _download(downloader: UrlDownloaderImpl, url: str, dest: Path, *, hf_token: str | None = "hf-secret") -> None:
    downloader.download(url, dest, hf_token=hf_token, max_bytes=_MAX_BYTES)


def test_hugging_face_token_is_not_sent_to_the_redirect_target(tmp_path: Path) -> None:
    downloader, transport = _downloader(
        {
            _HF_FILE: (302, {"Location": _CDN_FILE}, b""),
            _CDN_FILE: (200, {"Content-Length": "4"}, b"data"),
        }
    )

    _download(downloader, _HF_FILE, tmp_path / "out.bin")

    first, second = transport.sent
    assert first.headers["Authorization"] == "Bearer hf-secret"
    assert "Authorization" not in second.headers
    assert (tmp_path / "out.bin").read_bytes() == b"data"


def test_a_redirect_to_a_private_host_is_refused_before_a_request_is_sent(tmp_path: Path) -> None:
    downloader, transport = _downloader({_HF_FILE: (302, {"Location": "https://10.0.0.5/x.safetensors"}, b"")})

    with pytest.raises(UrlDownloadError, match="private network"):
        _download(downloader, _HF_FILE, tmp_path / "out.bin")

    assert [str(request.url) for request in transport.sent] == [_HF_FILE]


def test_a_download_over_the_size_cap_is_refused_with_and_without_a_length_header(tmp_path: Path) -> None:
    announced, _ = _downloader({_CDN_FILE: (200, {"Content-Length": str(_MAX_BYTES + 1)}, b"x")})
    unannounced, _ = _downloader({_CDN_FILE: (200, {}, b"x" * (_MAX_BYTES + 1))})

    for downloader in (announced, unannounced):
        with pytest.raises(UrlDownloadError, match="too large"):
            _download(downloader, _CDN_FILE, tmp_path / "out.bin")


def test_a_download_shorter_than_its_length_header_fails_as_interrupted(tmp_path: Path) -> None:
    downloader, _ = _downloader({_CDN_FILE: (200, {"Content-Length": "100"}, b"x" * 10)})

    with pytest.raises(UrlDownloadError, match="interrupted"):
        _download(downloader, _CDN_FILE, tmp_path / "out.bin")


def test_a_moved_hugging_face_repo_tells_the_user_to_copy_the_new_link() -> None:
    api_url = "https://huggingface.co/api/models/acme/old-name/revision/main"
    downloader, _ = _downloader({api_url: (301, {"Location": "https://huggingface.co/api/models/acme/new-name"}, b"")})

    with pytest.raises(UrlDownloadError, match="Copy the new link"):
        downloader.list_hugging_face_files(HuggingFaceRepo("acme/old-name"), hf_token=None)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_the_connect_time_check_refuses_a_loopback_address(host: str) -> None:
    with pytest.raises(UrlDownloadError, match="private network"):
        public_addresses(host, 443)


def test_the_default_session_ignores_a_system_proxy_that_would_hide_the_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:3128")
    url = "https://cdn.example.com/lora.safetensors"

    assert requests.Session().merge_environment_settings(url, {}, None, None, None)["proxies"]
    assert public_only_session().merge_environment_settings(url, {}, None, None, None)["proxies"] == {}


def test_the_default_session_refuses_a_private_address_before_it_connects() -> None:
    # Without the address check this request would try to connect to this computer instead.
    with public_only_session() as session, pytest.raises(UrlDownloadError, match="private network"):
        session.get("https://localhost/lora.safetensors", timeout=2)


_CARD_URL = "https://huggingface.co/acme/dolly/raw/main/README.md"


def test_a_missing_readme_gives_an_empty_card_and_an_oversize_one_is_cut_at_the_cap() -> None:
    repo = HuggingFaceRepo(repo_id="acme/dolly")
    missing, _ = _downloader({_CARD_URL: (404, {}, b"")})
    huge, _ = _downloader({_CARD_URL: (200, {}, b"a" * (_MAX_CARD_BYTES * 2))})

    assert missing.fetch_hugging_face_card(repo, hf_token=None) == ""
    assert len(huge.fetch_hugging_face_card(repo, hf_token=None)) == _MAX_CARD_BYTES
