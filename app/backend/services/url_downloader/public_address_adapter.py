"""A ``requests`` adapter that connects to public internet addresses only.

``check_fetch_url`` reads the text of a link. It cannot see where a name points: a public name
can resolve to 127.0.0.1 or to a LAN address. This adapter looks up the name when it opens the
connection, refuses it if any address is not global, and connects to the checked address. The
lookup and the connection use the same address, so a name that answers differently a second
time (DNS rebinding) cannot send the request to a private host. The TLS name check still uses
the host name, so the certificate must match it.

A proxy would open the connection instead and hide the address from this adapter.
``public_only_session`` therefore turns off ``trust_env``, so the system proxy settings are not used.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable
from typing import Any

from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPSConnection
from urllib3.connectionpool import HTTPSConnectionPool
from urllib3.exceptions import NewConnectionError
from urllib3.util.connection import create_connection

from services.url_downloader.url_downloader import UrlDownloadError

PRIVATE_ADDRESS_MESSAGE = "The link points to this computer or a private network."


def public_addresses(host: str, port: int) -> list[str]:
    """The addresses of ``host``, or ``UrlDownloadError`` if any of them is not a global address.

    One private address is enough to refuse: a client that tries the addresses in turn would
    reach it when the public ones fail.
    """
    try:
        infos = socket.getaddrinfo(host.rstrip("."), port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UrlDownloadError("Could not find the server.") from exc
    addresses = list(dict.fromkeys(str(info[4][0]).split("%", 1)[0] for info in infos))
    if not addresses or not _all_global(addresses):
        raise UrlDownloadError(PRIVATE_ADDRESS_MESSAGE)
    return addresses


def _all_global(addresses: Iterable[str]) -> bool:
    try:
        return all(ipaddress.ip_address(address).is_global for address in addresses)
    except ValueError:
        return False


class _PublicOnlyHTTPSConnection(HTTPSConnection):
    def _new_conn(self) -> socket.socket:
        last_error: OSError | None = None
        for address in public_addresses(self.host, self.port):
            try:
                return create_connection(
                    (address, self.port),
                    self.timeout,
                    source_address=self.source_address,
                    socket_options=self.socket_options,
                )
            except OSError as exc:
                last_error = exc
        raise NewConnectionError(self, f"Failed to establish a new connection: {last_error}")


class _PublicOnlyHTTPSConnectionPool(HTTPSConnectionPool):
    # urllib3 types this slot with a protocol that its own HTTPSConnection does not satisfy.
    ConnectionCls = _PublicOnlyHTTPSConnection  # pyright: ignore[reportAssignmentType]


class PublicAddressAdapter(HTTPAdapter):
    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            **self.poolmanager.pool_classes_by_scheme,
            "https": _PublicOnlyHTTPSConnectionPool,
        }
