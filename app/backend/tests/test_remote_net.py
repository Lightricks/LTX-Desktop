from remote.net import (
    advertised_lan_urls,
    attach_token,
    bound_server_port,
    client_url,
    origin_url,
    remote_port,
)
from runtime_config.port_constant import REMOTE_PORT


def test_client_url_puts_grant_in_the_pairing_query() -> None:
    assert client_url("192.168.1.9", 41955, "abc") == "http://192.168.1.9:41955/pairing?t=abc"


def test_https_default_port_omits_the_port() -> None:
    assert (
        client_url("studio.example.ts.net", 443, "abc", scheme="https")
        == "https://studio.example.ts.net/pairing?t=abc"
    )


def test_origin_url_strips_the_pairing_token() -> None:
    assert origin_url("http://192.168.1.9:41955/pairing?t=secret") == "http://192.168.1.9:41955/pairing"
    assert origin_url("http://127.0.0.1:41955/#t=secret") == "http://127.0.0.1:41955/"


def test_attach_token_keeps_an_existing_path() -> None:
    assert (
        attach_token("http://192.168.1.9:41955/assets", "secret")
        == "http://192.168.1.9:41955/assets?t=secret"
    )


def test_attach_token_preserves_existing_query() -> None:
    assert (
        attach_token("http://192.168.1.9:41955/assets?foo=1", "secret")
        == "http://192.168.1.9:41955/assets?foo=1&t=secret"
    )


def test_attach_token_overwrites_an_existing_t() -> None:
    assert (
        attach_token("http://192.168.1.9:41955/pairing?t=old", "new")
        == "http://192.168.1.9:41955/pairing?t=new"
    )


def test_remote_port_falls_back_to_constant(monkeypatch) -> None:
    monkeypatch.delenv("LTX_REMOTE_PORT", raising=False)
    assert remote_port() == REMOTE_PORT
    monkeypatch.setenv("LTX_REMOTE_PORT", "not-a-port")
    assert remote_port() == REMOTE_PORT
    monkeypatch.setenv("LTX_REMOTE_PORT", "47001")
    assert remote_port() == 47001
    monkeypatch.setenv("LTX_REMOTE_PORT", "0")
    assert remote_port() == 0


class _Sock:
    def __init__(self, port: int) -> None:
        self._port = port

    def getsockname(self) -> tuple[str, int]:
        return ("0.0.0.0", self._port)


class _AsyncioServer:
    def __init__(self, port: int) -> None:
        self.sockets = [_Sock(port)]


class _UvicornServer:
    def __init__(self, port: int) -> None:
        self.servers = [_AsyncioServer(port)]


def test_bound_server_port_reads_uvicorn_socket() -> None:
    assert bound_server_port(_UvicornServer(49259), 0) == 49259
    assert bound_server_port(object(), 41955) == 41955


def test_advertised_lan_urls_include_query_grant(monkeypatch) -> None:
    monkeypatch.setattr("remote.net.lan_ip", lambda: "192.168.1.9")
    lan, local = advertised_lan_urls(41955, "secret")
    assert lan == "http://192.168.1.9:41955/pairing?t=secret"
    assert local == "http://127.0.0.1:41955/pairing?t=secret"
