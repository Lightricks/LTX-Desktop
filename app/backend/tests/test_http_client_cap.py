"""HTTPClientImpl.get with max_bytes, against a real local server.

A fake cannot show that the cap stops the read, so these tests use a real socket.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import pytest

from services.http_client import (
    HttpBodyTooLargeError,
    HttpRedirectRefusedError,
    HttpTransportError,
)
from services.http_client.http_client_impl import HTTPClientImpl, _redirect_target

_BODY = b"x" * 1000


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        port = self.server.server_address[1]
        redirects = {
            "/redir": "/ok",
            "/loop": "/loop",
            "/slow-hop-1": "/slow-hop-2",
            "/slow-hop-2": "/slow-hop-3",
            "/slow-hop-3": "/ok",
            "/other-host": f"http://localhost:{port}/echo-auth",
        }
        if self.path.startswith("/slow-hop"):
            # Each hop answers inside a timeout of 1 second. Together they do not.
            time.sleep(0.6)
        if self.path in redirects:
            self.send_response(302)
            self.send_header("Location", redirects[self.path])
            self.end_headers()
            return
        if self.path == "/echo-auth":
            echoed = self.headers.get("Authorization", "none").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(echoed)))
            self.end_headers()
            self.wfile.write(echoed)
            return
        if self.path == "/slow":
            # No Content-Length and a body that arrives in pieces over more than the deadline.
            self.send_response(200)
            self.end_headers()
            for _ in range(6):
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(0.4)
            return
        self.send_response(404 if self.path == "/missing" else 200)
        # /chunked sends no Content-Length, so only the chunk counter can stop it.
        if self.path != "/chunked":
            self.send_header("Content-Length", str(len(_BODY)))
        self.end_headers()
        self.wfile.write(_BODY)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        del format, args


@contextmanager
def _server() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_a_body_under_the_cap_is_returned_whole() -> None:
    # Bug caught: the capped read drops or truncates a normal image.
    with _server() as base:
        response = HTTPClientImpl().get(f"{base}/ok", max_bytes=2000)

    assert response.status_code == 200
    assert response.content == _BODY


def test_a_declared_length_over_the_cap_is_refused() -> None:
    # Bug caught: the cap checks only after the whole body is read.
    with _server() as base, pytest.raises(HttpBodyTooLargeError):
        HTTPClientImpl().get(f"{base}/ok", max_bytes=999)


def test_a_body_with_no_declared_length_is_cut_at_the_cap() -> None:
    # Bug caught: a server that hides Content-Length avoids the cap.
    with _server() as base, pytest.raises(HttpBodyTooLargeError):
        HTTPClientImpl().get(f"{base}/chunked", max_bytes=999)


def test_an_error_status_is_returned_with_its_status() -> None:
    # Bug caught: the buffered response loses the status, so a 404 reads as an image.
    with _server() as base:
        response = HTTPClientImpl().get(f"{base}/missing", max_bytes=2000)

    assert response.status_code == 404


def test_a_redirect_is_followed_to_the_body() -> None:
    # Bug caught: the hand-made redirect loop returns the 302 instead of the image.
    with _server() as base:
        response = HTTPClientImpl().get(f"{base}/redir", max_bytes=2000)

    assert response.status_code == 200
    assert response.content == _BODY


def test_a_redirect_loop_is_refused() -> None:
    # Bug caught: a server that redirects forever holds the download thread (CWE-770).
    with _server() as base, pytest.raises(HttpRedirectRefusedError):
        HTTPClientImpl().get(f"{base}/loop", max_bytes=2000)


def test_the_credential_header_stays_on_the_first_host() -> None:
    # Bug caught: a redirect to another host receives the Authorization header (CWE-200).
    headers = {"Authorization": "Key <API_KEY>"}
    with _server() as base:
        same_host = HTTPClientImpl().get(f"{base}/echo-auth", headers=headers, max_bytes=2000)
        other_host = HTTPClientImpl().get(f"{base}/other-host", headers=headers, max_bytes=2000)

    assert same_host.content == b"Key <API_KEY>"
    assert other_host.content == b"none"


def test_a_redirect_from_https_to_http_is_refused() -> None:
    # Bug caught: a fal URL on https sends the download to a plain-http host (CWE-918).
    with pytest.raises(HttpRedirectRefusedError):
        _redirect_target("https://fal.example/a.png", "http://evil.example/a.png")
    assert _redirect_target("https://fal.example/a.png", "/b.png") == "https://fal.example/b.png"
    assert _redirect_target("http://fal.example/a.png", "https://fal.example/a.png")


def test_a_body_that_arrives_slowly_hits_the_total_deadline() -> None:
    # Bug caught: each read is fast, so the per-read timeout never fires and the call never ends.
    with _server() as base, pytest.raises(HttpTransportError):
        HTTPClientImpl().get(f"{base}/slow", timeout=1, max_bytes=2000)


def test_redirect_hops_share_one_deadline() -> None:
    # Bug caught: every hop gets the full timeout, so a chain of slow hops runs for a multiple of it.
    started = time.monotonic()
    with _server() as base, pytest.raises(HttpTransportError):
        HTTPClientImpl().get(f"{base}/slow-hop-1", timeout=1, max_bytes=2000)

    assert time.monotonic() - started < 1.5
