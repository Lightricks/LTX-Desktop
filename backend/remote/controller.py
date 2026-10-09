"""Start/stop the isolated remote HTTP server when Settings remote is on."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from logging_policy import log_background_exception
from remote.app import create_remote_app
from remote.net import ENV_AUTH, advertised_lan_urls, bound_server_port, origin_url, remote_port, resolve_client_dir
from remote.pairing import PairedDevicePublic, RemotePairing
from remote.status import RemoteStatusResponse
from state.app_settings import RemoteExposure, remote_is_on

if TYPE_CHECKING:
    from app_handler import AppHandler
    import uvicorn

logger = logging.getLogger(__name__)

_BIND_WAIT_S = 8.0
_BIND_POLL_S = 0.05
_JOIN_TIMEOUT_S = 5.0
_LAN_BIND_HOST = "0.0.0.0"


class AdvertisedEndpoints(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str
    local_url: str
    lan_url: str


@dataclass(frozen=True)
class _Running:
    server: uvicorn.Server
    thread: threading.Thread
    port: int


@dataclass(frozen=True)
class _Stopping:
    server: uvicorn.Server
    thread: threading.Thread


class RemoteExposureController:
    def __init__(self, handler: AppHandler) -> None:
        self._handler = handler
        self._lock = threading.Lock()
        self._ownership: _Running | _Stopping | None = None
        self._exposure: RemoteExposure = "off"
        self._last_reason: str = "not started"
        self._pairing = RemotePairing.from_app_data(handler.config.app_data_dir)

    def status(self) -> RemoteStatusResponse:
        with self._lock:
            self._reap_dead_locked()
            serving = False
            advertised: AdvertisedEndpoints | None = None
            match self._ownership:
                case _Running(server=server, port=port) if server.started:
                    serving = True
                    grant = self._pairing.current_grant()
                    lan, local = advertised_lan_urls(bound_server_port(server, port), grant)
                    advertised = AdvertisedEndpoints(url=lan, local_url=local, lan_url=lan)
                case _:
                    pass
            exposure = self._exposure
            reason = None if serving else self._last_reason
        permitted, gate_reason = self._permit()
        if not permitted:
            return RemoteStatusResponse(
                permitted=False,
                serving=False,
                reason=gate_reason,
                mode=exposure,
            )
        if not serving or advertised is None:
            return RemoteStatusResponse(
                permitted=True,
                serving=False,
                reason=reason,
                mode=exposure,
            )
        return RemoteStatusResponse(
            permitted=True,
            serving=True,
            url=advertised.url,
            localUrl=advertised.local_url,
            lanUrl=advertised.lan_url,
            mode=exposure,
        )

    def sync(self) -> None:
        permitted, reason = self._permit()
        exposure = self._handler.settings.get_settings_snapshot().remote_exposure
        should_run = permitted and remote_is_on(exposure)
        stop_target: _Stopping | None = None
        with self._lock:
            self._reap_dead_locked()
            if isinstance(self._ownership, _Stopping):
                if self._ownership.thread.is_alive():
                    return
                self._finish_stop_locked(self._ownership, reason=self._last_reason)
            running = isinstance(self._ownership, _Running)
            mode_changed = running and self._exposure != exposure
            if should_run and (not running or mode_changed):
                if running:
                    stop_target = self._begin_stop_locked()
                else:
                    self._start_locked()
            elif not should_run and running:
                stop_target = self._begin_stop_locked()
            elif not should_run:
                self._exposure = exposure
                if not permitted:
                    self._last_reason = reason
                else:
                    self._last_reason = "disabled in Settings"
        if stop_target is None:
            return
        self._join_stop(stop_target)
        with self._lock:
            self._finish_stop_locked(stop_target)
            self._restart_if_desired_locked()

    def stop(self) -> None:
        with self._lock:
            stop_target = self._begin_stop_locked()
        if stop_target is None:
            return
        self._join_stop(stop_target)
        with self._lock:
            self._finish_stop_locked(stop_target)

    def list_devices(self) -> list[PairedDevicePublic]:
        return self._pairing.list_public_devices()

    def revoke_device(self, device_id: str) -> bool:
        return self._pairing.revoke(device_id)

    def _reap_dead_locked(self) -> None:
        match self._ownership:
            case _Running(thread=thread) if not thread.is_alive():
                self._last_reason = "remote server stopped unexpectedly"
                logger.error("Remote server thread exited")
                self._clear_locked()
            case _:
                return

    def _permit(self) -> tuple[bool, str]:
        if not os.environ.get(ENV_AUTH, "").strip():
            return False, "LTX_AUTH_TOKEN is empty"
        return True, ""

    def _start_locked(self) -> None:
        import uvicorn

        if self._ownership is not None:
            return
        exposure = self._handler.settings.get_settings_snapshot().remote_exposure
        if not remote_is_on(exposure):
            self._exposure = exposure
            self._last_reason = "disabled in Settings"
            return
        port = remote_port()
        app = create_remote_app(
            handler=self._handler,
            pairing=self._pairing,
            client_dir=resolve_client_dir(),
        )
        log_config: dict[str, object] = {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
                "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
                "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
            },
        }
        config = uvicorn.Config(
            app,
            host=_LAN_BIND_HOST,
            port=port,
            log_level="info",
            access_log=False,
            log_config=log_config,
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=self._run_server,
            args=(server,),
            name="ltx-remote",
            daemon=True,
        )
        self._ownership = _Running(server=server, thread=thread, port=port)
        self._exposure = exposure
        self._last_reason = "starting"
        thread.start()
        threading.Thread(
            target=self._await_bind,
            args=(server, thread, port),
            name="ltx-remote-bind",
            daemon=True,
        ).start()

    def _await_bind(self, server: uvicorn.Server, thread: threading.Thread, port: int) -> None:
        deadline = time.monotonic() + _BIND_WAIT_S
        while time.monotonic() < deadline:
            if server.started:
                with self._lock:
                    owned = self._ownership
                    if not isinstance(owned, _Running) or owned.server is not server:
                        return
                    bound = bound_server_port(server, owned.port)
                    if bound != owned.port:
                        self._ownership = _Running(
                            server=owned.server, thread=owned.thread, port=bound
                        )
                    grant = self._pairing.current_grant()
                    lan, local = advertised_lan_urls(bound, grant)
                logger.info("Remote client listening on %s", origin_url(lan))
                logger.info("Remote client (this machine): %s", origin_url(local))
                return
            if not thread.is_alive():
                self._fail_start(
                    server,
                    reason="remote server failed to start",
                    log_msg="Remote server thread exited before bind",
                    signal_exit=False,
                )
                return
            time.sleep(_BIND_POLL_S)
        self._fail_start(
            server,
            reason="remote server failed to bind",
            log_msg=f"Remote server did not bind on port {port}",
            signal_exit=True,
        )

    def _fail_start(
        self,
        server: uvicorn.Server,
        *,
        reason: str,
        log_msg: str,
        signal_exit: bool,
    ) -> None:
        with self._lock:
            owned = self._ownership
            if not isinstance(owned, _Running) or owned.server is not server:
                return
            if signal_exit:
                server.should_exit = True
            self._last_reason = reason
            logger.error(log_msg)
            stopping = _Stopping(server=owned.server, thread=owned.thread)
            self._ownership = stopping
        self._join_stop(stopping)
        with self._lock:
            if stopping.thread.is_alive():
                return
            self._finish_stop_locked(stopping, reason=reason)

    def _run_server(self, server: uvicorn.Server) -> None:
        try:
            server.run()
        except SystemExit:
            # uvicorn.Server.startup calls sys.exit(1) on bind failure (EADDRINUSE).
            logger.error("Remote server exited during startup")
        except Exception as exc:
            log_background_exception("ltx-remote", exc)

    def _begin_stop_locked(self) -> _Stopping | None:
        match self._ownership:
            case _Stopping() as stopping:
                return stopping
            case None:
                self._clear_locked()
                self._last_reason = "stopped"
                logger.info("Remote client stopped")
                return None
            case _Running(server=server, thread=thread):
                server.should_exit = True
                stopping = _Stopping(server=server, thread=thread)
                self._ownership = stopping
                self._last_reason = "stopped"
                return stopping

    def _join_stop(self, target: _Stopping) -> None:
        if target.thread.is_alive():
            target.thread.join(timeout=_JOIN_TIMEOUT_S)

    def _finish_stop_locked(self, target: _Stopping, *, reason: str = "stopped") -> None:
        match self._ownership:
            case _Stopping(server=server, thread=thread) if (
                server is target.server and thread is target.thread
            ):
                self._clear_locked()
                self._last_reason = reason
                logger.info("Remote client stopped")
            case _:
                return

    def _restart_if_desired_locked(self) -> None:
        if self._ownership is not None:
            return
        permitted, _reason = self._permit()
        exposure = self._handler.settings.get_settings_snapshot().remote_exposure
        if permitted and remote_is_on(exposure):
            self._start_locked()

    def _clear_locked(self) -> None:
        self._ownership = None
        self._exposure = "off"


def remote_controller_from_app(app: object) -> RemoteExposureController | None:
    controller: object = getattr(getattr(app, "state", None), "remote_controller", None)
    if isinstance(controller, RemoteExposureController):
        return controller
    return None
